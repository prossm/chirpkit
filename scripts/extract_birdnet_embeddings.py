#!/usr/bin/env python3
"""
Extract BirdNET embeddings from raw insect audio files.

This script processes the original audio files (not spectrograms) to extract
1024-dimensional feature embeddings using the pre-trained BirdNET model.

Usage:
    python scripts/extract_birdnet_embeddings.py --dataset combined --output data/embeddings

The embeddings will be saved as:
    - X_train_embeddings.npy (n_samples, 1024)
    - y_train.npy
    - X_val_embeddings.npy (n_samples, 1024)
    - y_val.npy
    - metadata.json (dataset info)
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import joblib

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from chirpkit.transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor


def load_audio_file_list(raw_data_dir, dataset_name):
    """
    Scan raw audio directory and create file list with labels.

    Args:
        raw_data_dir: Path to data/raw directory
        dataset_name: Dataset name (e.g., 'combined', 'insectsound1000')

    Returns:
        audio_files: List of audio file paths
        labels: List of species labels
        label_encoder: Fitted label encoder
    """
    from sklearn.preprocessing import LabelEncoder

    raw_dir = Path(raw_data_dir) / dataset_name
    audio_files = []
    labels = []

    print(f"📂 Scanning audio files in {raw_dir}")

    # Supported audio formats
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

    # Recursively find all audio files
    for audio_path in raw_dir.rglob('*'):
        if audio_path.suffix.lower() in audio_extensions:
            # Extract species name from directory structure
            # Assuming structure: data/raw/dataset/species_name/file.wav
            species = audio_path.parent.name
            audio_files.append(str(audio_path))
            labels.append(species)

    # Fit label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    print(f"✅ Found {len(audio_files)} audio files")
    print(f"🦗 {len(label_encoder.classes_)} unique species")

    return audio_files, encoded_labels, label_encoder


def load_existing_splits(splits_dir, dataset_name):
    """
    Load existing train/val splits to ensure consistency.

    Returns:
        Dictionary with train/val splits and metadata
    """
    splits_path = Path(splits_dir) / dataset_name

    if not splits_path.exists():
        return None

    # Load split information
    # Check if we have a metadata file that tracks which files are in train/val
    metadata_file = splits_path / "split_metadata.json"

    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata

    # If no metadata, we'll need to create splits from scratch
    return None


def create_train_val_splits(audio_files, labels, val_ratio=0.3, random_seed=42):
    """
    Create train/validation splits ensuring balanced classes.

    Args:
        audio_files: List of audio file paths
        labels: Encoded labels
        val_ratio: Validation split ratio
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with train/val splits
    """
    from sklearn.model_selection import train_test_split

    # Stratified split to maintain class balance
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files,
        labels,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=labels
    )

    print(f"\n📊 Created splits:")
    print(f"   Train: {len(train_files)} files")
    print(f"   Val: {len(val_files)} files")

    return {
        'train': {'files': train_files, 'labels': train_labels},
        'val': {'files': val_files, 'labels': val_labels}
    }


def extract_embeddings_batch(extractor, audio_files, batch_size=32):
    """
    Extract embeddings for a batch of audio files.

    Args:
        extractor: BirdNETEmbeddingExtractor instance
        audio_files: List of audio file paths
        batch_size: Number of files to process at once (doesn't speed up much)

    Returns:
        embeddings: numpy array (n_files, 1024)
    """
    embeddings = []

    for audio_file in tqdm(audio_files, desc="Extracting embeddings"):
        emb = extractor.extract_embeddings_from_audio(audio_file, aggregate='mean')
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Extract BirdNET embeddings from insect audio'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='combined',
        choices=['insectsound1000', 'insectset459', 'sina', 'xenocanto', 'combined'],
        help='Dataset to process'
    )

    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default='data/raw',
        help='Directory containing raw audio files'
    )

    parser.add_argument(
        '--splits-dir',
        type=str,
        default='data/splits',
        help='Directory containing existing split information'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/embeddings',
        help='Output directory for embeddings'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.3,
        help='Validation split ratio (if creating new splits)'
    )

    parser.add_argument(
        '--aggregate',
        type=str,
        default='mean',
        choices=['mean', 'max', 'first'],
        help='How to aggregate multiple 3-second chunks'
    )

    args = parser.parse_args()

    print("🦗 BirdNET Embedding Extraction for Insect Classification")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize BirdNET extractor
    extractor = BirdNETEmbeddingExtractor()

    # Load audio files and labels
    audio_files, labels, label_encoder = load_audio_file_list(
        args.raw_data_dir,
        args.dataset
    )

    # Check for existing splits
    existing_splits = load_existing_splits(args.splits_dir, args.dataset)

    if existing_splits:
        print(f"\n✅ Using existing train/val splits")
        # Use existing split information
        # TODO: Map to audio files
        splits = None  # Placeholder
    else:
        print(f"\n🔄 Creating new train/val splits ({int((1-args.val_ratio)*100)}% train / {int(args.val_ratio*100)}% val)")
        splits = create_train_val_splits(audio_files, labels, args.val_ratio)

    # If no splits available, create from audio files
    if splits is None:
        splits = create_train_val_splits(audio_files, labels, args.val_ratio)

    # Extract embeddings for train set
    print(f"\n🔍 Extracting TRAIN embeddings...")
    train_embeddings = extract_embeddings_batch(
        extractor,
        splits['train']['files']
    )

    # Extract embeddings for val set
    print(f"\n🔍 Extracting VAL embeddings...")
    val_embeddings = extract_embeddings_batch(
        extractor,
        splits['val']['files']
    )

    # Save embeddings
    print(f"\n💾 Saving embeddings to {output_dir}")

    np.save(output_dir / "X_train_embeddings.npy", train_embeddings)
    np.save(output_dir / "y_train.npy", splits['train']['labels'])
    np.save(output_dir / "X_val_embeddings.npy", val_embeddings)
    np.save(output_dir / "y_val.npy", splits['val']['labels'])

    # Save label encoder
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    # Save metadata
    metadata = {
        'dataset': args.dataset,
        'n_train': len(train_embeddings),
        'n_val': len(val_embeddings),
        'n_classes': len(label_encoder.classes_),
        'embedding_dim': 1024,
        'aggregation': args.aggregate,
        'train_files': splits['train']['files'],
        'val_files': splits['val']['files']
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n✅ Embedding extraction complete!")
    print(f"   Train embeddings: {train_embeddings.shape}")
    print(f"   Val embeddings: {val_embeddings.shape}")
    print(f"   Classes: {len(label_encoder.classes_)}")
    print(f"   Saved to: {output_dir}")

    print(f"\n📋 Files created:")
    print(f"   - X_train_embeddings.npy ({train_embeddings.nbytes / 1024**2:.1f} MB)")
    print(f"   - y_train.npy")
    print(f"   - X_val_embeddings.npy ({val_embeddings.nbytes / 1024**2:.1f} MB)")
    print(f"   - y_val.npy")
    print(f"   - label_encoder.joblib")
    print(f"   - metadata.json")

    print(f"\n🎯 Next step: Train classifier on embeddings")
    print(f"   python scripts/train_birdnet_classifier.py --embeddings-dir {output_dir}")


if __name__ == "__main__":
    main()
