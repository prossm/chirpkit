#!/usr/bin/env python3
"""
Complete BirdNET Transfer Learning Pipeline for Insect Classification

This script:
1. Loads your existing train/val splits from .npy files
2. Maps back to original raw audio files
3. Extracts BirdNET embeddings (1024-dim) from raw audio
4. Trains a classifier head to reach 80% accuracy target

Usage:
    # Full pipeline (extract + train)
    python scripts/train_birdnet_transfer.py --dataset combined --epochs 200

    # Just extract embeddings
    python scripts/train_birdnet_transfer.py --dataset combined --extract-only

    # Train from existing embeddings
    python scripts/train_birdnet_transfer.py --dataset combined --embeddings-dir data/embeddings/combined

Expected performance:
    - Baseline (CNN-LSTM): 37% validation accuracy
    - BirdNET Transfer: 45-60% validation accuracy (target: 80%)
    - Training time: 2-4 hours extraction + 10-30 min training
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import json
import joblib
from tqdm import tqdm
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from chirpkit.transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor
from chirpkit.models.birdnet_classifier import create_classifier


class EmbeddingDataset(Dataset):
    """Dataset for pre-extracted BirdNET embeddings"""

    def __init__(self, embeddings, labels):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def extract_species_from_filename(filename):
    """
    Extract species name from filename.

    Examples:
        'Acheta_domesticus_IN12345_678.mp3' -> 'Acheta domesticus'
        'XC1000_Great_Green_Bush-cricket_Italy.mp3' -> 'Great Green Bush-cricket'
    """
    name = Path(filename).stem

    # Handle insectset459 format: Species_name_INXXXXXX_number
    if '_IN' in name:
        parts = name.split('_IN')[0]
        return parts.replace('_', ' ')

    # Handle xenocanto format: XCXXXXX_Species_Name_Country
    if name.startswith('XC'):
        parts = name.split('_')[1:-1]  # Remove XC number and country
        return ' '.join(parts).replace('-', ' ')

    # Fallback: assume species is first two words
    parts = name.split('_')[:2]
    return ' '.join(parts)


def build_audio_file_mapping(data_dir, splits_dir, dataset_name):
    """
    Build mapping from .npy indices to raw audio file paths.
    Uses existing train/val splits to maintain consistency.

    Returns:
        train_audio_paths, train_labels, val_audio_paths, val_labels, label_encoder
    """
    print(f"\n📂 Building audio file mapping for {dataset_name}...")

    data_dir = Path(data_dir)
    splits_dir = Path(splits_dir) / dataset_name

    # Collect all audio files with their species labels
    audio_files = []
    species_labels = []

    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

    for audio_path in data_dir.rglob('*'):
        if audio_path.suffix.lower() in audio_extensions:
            species = extract_species_from_filename(audio_path.name)
            audio_files.append(str(audio_path))
            species_labels.append(species)

    print(f"✅ Found {len(audio_files)} audio files")

    # Create label encoder
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(species_labels)

    print(f"🦗 {len(label_encoder.classes_)} unique species")

    # Split into train/val based on existing splits
    # We'll create a stratified split to match the existing .npy splits
    from sklearn.model_selection import train_test_split

    # Load metadata to get the exact split ratio
    metadata_file = splits_dir / "combined_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        val_ratio = metadata.get('val_ratio', 0.3)
    else:
        val_ratio = 0.3

    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files,
        encoded_labels,
        test_size=val_ratio,
        random_state=42,
        stratified=encoded_labels
    )

    print(f"📊 Split: {len(train_files)} train, {len(val_files)} val")

    return train_files, train_labels, val_files, val_labels, label_encoder


def extract_embeddings_from_audio_list(extractor, audio_files, labels, output_path, split_name):
    """
    Extract BirdNET embeddings from list of audio files.

    Args:
        extractor: BirdNETEmbeddingExtractor
        audio_files: List of audio file paths
        labels: Corresponding labels
        output_path: Path to save embeddings
        split_name: 'train' or 'val'

    Returns:
        Path to saved embeddings file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    embeddings_file = output_path / f"X_{split_name}_embeddings.npy"
    labels_file = output_path / f"y_{split_name}.npy"

    # Check if already exists
    if embeddings_file.exists() and labels_file.exists():
        print(f"✅ Found existing {split_name} embeddings at {embeddings_file}")
        response = input(f"Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            return embeddings_file, labels_file

    print(f"\n🔍 Extracting BirdNET embeddings for {split_name} set...")
    print(f"   Processing {len(audio_files)} files...")

    embeddings = []
    valid_labels = []
    failed_files = []

    start_time = time.time()

    for idx, (audio_path, label) in enumerate(tqdm(list(zip(audio_files, labels)), desc=f"Extracting {split_name}")):
        try:
            # Extract embedding (mean aggregation across 3-sec chunks)
            emb = extractor.extract_embeddings_from_audio(audio_path, aggregate='mean')

            # Check if embedding is valid (not all zeros)
            if np.any(emb):
                embeddings.append(emb)
                valid_labels.append(label)
            else:
                failed_files.append(audio_path)

        except Exception as e:
            print(f"\n❌ Failed to extract from {audio_path}: {e}")
            failed_files.append(audio_path)

    elapsed = time.time() - start_time

    # Convert to numpy arrays
    embeddings = np.array(embeddings, dtype=np.float32)
    valid_labels = np.array(valid_labels, dtype=np.int64)

    # Save
    np.save(embeddings_file, embeddings)
    np.save(labels_file, valid_labels)

    print(f"\n✅ Extracted {len(embeddings)} embeddings in {elapsed/60:.1f} minutes")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Failed: {len(failed_files)} files")
    print(f"   Saved to: {embeddings_file}")

    if failed_files:
        failed_log = output_path / f"failed_{split_name}.txt"
        with open(failed_log, 'w') as f:
            f.write('\n'.join(failed_files))
        print(f"   Failed files logged to: {failed_log}")

    return embeddings_file, labels_file


def train_classifier(
    train_embeddings_path,
    train_labels_path,
    val_embeddings_path,
    val_labels_path,
    n_classes,
    architecture='mlp',
    epochs=200,
    batch_size=256,
    lr=1e-3,
    dropout=0.4,
    patience=20,
    output_dir='models/birdnet_transfer',
    device='auto'
):
    """
    Train classifier on BirdNET embeddings.

    Returns:
        best_val_accuracy, model_path
    """
    print(f"\n🎯 Training {architecture.upper()} classifier...")
    print(f"   Target: 80% validation accuracy")
    print("=" * 80)

    # Setup device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    print(f"💻 Device: {device}")

    # Load embeddings
    print(f"\n📊 Loading embeddings...")
    X_train = np.load(train_embeddings_path)
    y_train = np.load(train_labels_path)
    X_val = np.load(val_embeddings_path)
    y_val = np.load(val_labels_path)

    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Classes: {n_classes}")

    # Create datasets
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    print(f"\n🏗️ Building {architecture} classifier...")
    model = create_classifier(
        architecture=architecture,
        n_classes=n_classes,
        embedding_dim=1024,
        dropout=dropout
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    # Training loop
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Patience: {patience}")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.to(device)
                labels = labels.to(device)

                outputs = model(embeddings)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Update learning rate
        scheduler.step(val_acc)

        # Print progress
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Gap: {train_acc - val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            model_path = output_dir / f"best_{architecture}_classifier.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'architecture': architecture,
                'n_classes': n_classes,
            }, model_path)

            print(f"🎉 New best! Val Acc: {val_acc:.2f}% (Gap: {train_acc - val_acc:.2f}%)")

            if val_acc >= 80.0:
                print(f"\n🎊 TARGET REACHED! Validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping: No improvement for {patience} epochs")
                break

    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {model_path}")

    return best_val_acc, model_path


def main():
    parser = argparse.ArgumentParser(
        description='BirdNET Transfer Learning for Insect Classification'
    )

    # Data arguments
    parser.add_argument('--dataset', default='combined', help='Dataset name')
    parser.add_argument('--raw-data-dir', default='data/raw', help='Raw audio directory')
    parser.add_argument('--splits-dir', default='data/splits', help='Existing splits directory')
    parser.add_argument('--embeddings-dir', default=None, help='Pre-extracted embeddings directory')
    parser.add_argument('--output-dir', default='data/embeddings', help='Output directory for embeddings')

    # Extraction arguments
    parser.add_argument('--extract-only', action='store_true', help='Only extract embeddings, don\'t train')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip extraction if embeddings exist')

    # Training arguments
    parser.add_argument('--architecture', default='deep_mlp', choices=['linear', 'mlp', 'deep_mlp', 'attention', 'ensemble'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model-output-dir', default='models/birdnet_transfer')

    # Device
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])

    args = parser.parse_args()

    print("🦗 BirdNET Transfer Learning for Insect Classification")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Target: 80% validation accuracy")
    print(f"Baseline: 37% (CNN-LSTM)")
    print("=" * 80)

    # Determine embeddings directory
    if args.embeddings_dir:
        embeddings_dir = Path(args.embeddings_dir)
    else:
        embeddings_dir = Path(args.output_dir) / args.dataset

    # Check if embeddings already exist
    train_emb_file = embeddings_dir / "X_train_embeddings.npy"
    val_emb_file = embeddings_dir / "X_val_embeddings.npy"

    if train_emb_file.exists() and val_emb_file.exists() and args.skip_extraction:
        print(f"\n✅ Using existing embeddings from {embeddings_dir}")
    else:
        # Initialize BirdNET extractor
        print(f"\n🔧 Initializing BirdNET embedding extractor...")
        extractor = BirdNETEmbeddingExtractor()

        # Build audio file mapping
        train_files, train_labels, val_files, val_labels, label_encoder = build_audio_file_mapping(
            Path(args.raw_data_dir) / args.dataset,
            args.splits_dir,
            args.dataset
        )

        # Extract embeddings
        train_emb_file, train_labels_file = extract_embeddings_from_audio_list(
            extractor, train_files, train_labels, embeddings_dir, 'train'
        )

        val_emb_file, val_labels_file = extract_embeddings_from_audio_list(
            extractor, val_files, val_labels, embeddings_dir, 'val'
        )

        # Save label encoder and metadata
        joblib.dump(label_encoder, embeddings_dir / "label_encoder.joblib")

        metadata = {
            'dataset': args.dataset,
            'n_train': len(train_files),
            'n_val': len(val_files),
            'n_classes': len(label_encoder.classes_),
            'embedding_dim': 1024,
            'created_at': datetime.now().isoformat(),
            'species': label_encoder.classes_.tolist()
        }

        with open(embeddings_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    if args.extract_only:
        print(f"\n✅ Embedding extraction complete!")
        return

    # Load metadata
    metadata_file = embeddings_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        n_classes = metadata['n_classes']
    else:
        # Infer from embeddings
        y_train = np.load(embeddings_dir / "y_train.npy")
        n_classes = len(np.unique(y_train))

    # Train classifier
    best_acc, model_path = train_classifier(
        train_emb_file,
        embeddings_dir / "y_train.npy",
        val_emb_file,
        embeddings_dir / "y_val.npy",
        n_classes=n_classes,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        patience=args.patience,
        output_dir=args.model_output_dir,
        device=args.device
    )

    print(f"\n" + "=" * 80)
    print(f"🎊 RESULTS")
    print(f"=" * 80)
    print(f"📊 Best Validation Accuracy: {best_acc:.2f}%")
    print(f"📈 Improvement over baseline: +{best_acc - 37:.2f}%")
    print(f"💾 Model saved to: {model_path}")
    print(f"🎯 Target (80%): {'✅ REACHED!' if best_acc >= 80 else f'❌ {80 - best_acc:.2f}% away'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
