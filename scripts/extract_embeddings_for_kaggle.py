#!/usr/bin/env python3
"""
Extract BirdNET embeddings locally, prepare for Kaggle training.

This script:
1. Extracts BirdNET embeddings from raw audio (slow, ~2-4 hours)
2. Saves as compact numpy files (~120MB vs 6.2GB spectrograms!)
3. Creates a ready-to-upload Kaggle dataset

Then upload to Kaggle and train with GPU in 10-30 minutes!

Usage:
    # Extract embeddings (run locally, one-time)
    python scripts/extract_embeddings_for_kaggle.py --dataset combined

    # Creates ready-to-upload package at:
    # data/embeddings_kaggle/chirpkit-birdnet-embeddings.tar.gz (~120MB)

Then on Kaggle:
    1. Upload the .tar.gz as a dataset
    2. Run the training notebook (provided)
    3. Download trained model
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
import tarfile
import shutil
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def extract_species_from_filename(filename):
    """Extract species name from filename."""
    name = Path(filename).stem

    # Handle insectset459 format
    if '_IN' in name:
        parts = name.split('_IN')[0]
        return parts.replace('_', ' ')

    # Handle xenocanto format
    if name.startswith('XC'):
        parts = name.split('_')[1:-1]
        return ' '.join(parts).replace('-', ' ')

    # Fallback
    parts = name.split('_')[:2]
    return ' '.join(parts)


def collect_audio_files(raw_data_dirs):
    """
    Collect all audio files from multiple directories.

    Args:
        raw_data_dirs: List of directories to scan

    Returns:
        audio_files, species_labels
    """
    audio_files = []
    species_labels = []

    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

    print(f"\n📂 Scanning audio files...")
    for data_dir in raw_data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"⚠️  Warning: {data_dir} not found, skipping...")
            continue

        print(f"   Scanning: {data_path}")
        for audio_path in data_path.rglob('*'):
            if audio_path.suffix.lower() in audio_extensions:
                species = extract_species_from_filename(audio_path.name)
                audio_files.append(str(audio_path))
                species_labels.append(species)

    print(f"✅ Found {len(audio_files)} audio files")
    return audio_files, species_labels


def extract_embeddings_batch(extractor, audio_files, labels, desc="Extracting"):
    """Extract embeddings for a list of audio files."""
    embeddings = []
    valid_labels = []
    failed_files = []

    for audio_path, label in tqdm(list(zip(audio_files, labels)), desc=desc):
        try:
            emb = extractor.extract_embeddings_from_audio(audio_path, aggregate='mean')

            if np.any(emb):
                embeddings.append(emb)
                valid_labels.append(label)
            else:
                failed_files.append(audio_path)

        except Exception as e:
            print(f"\n❌ Failed: {audio_path}: {e}")
            failed_files.append(audio_path)

    return np.array(embeddings, dtype=np.float32), np.array(valid_labels), failed_files


def create_kaggle_package(embeddings_dir, output_dir):
    """
    Create a compressed package ready for Kaggle upload.

    Structure:
        chirpkit-birdnet-embeddings/
        ├── X_train_embeddings.npy
        ├── y_train.npy
        ├── X_val_embeddings.npy
        ├── y_val.npy
        ├── label_encoder.joblib
        ├── metadata.json
        └── train_on_kaggle.py  (training script)
    """
    embeddings_dir = Path(embeddings_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    package_name = "chirpkit-birdnet-embeddings"
    package_dir = output_dir / package_name

    # Clean and recreate package directory
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()

    print(f"\n📦 Creating Kaggle package at {package_dir}...")

    # Copy embedding files
    files_to_copy = [
        'X_train_embeddings.npy',
        'y_train.npy',
        'X_val_embeddings.npy',
        'y_val.npy',
        'label_encoder.joblib',
        'metadata.json'
    ]

    for filename in files_to_copy:
        src = embeddings_dir / filename
        dst = package_dir / filename
        if src.exists():
            shutil.copy(src, dst)
            size_mb = src.stat().st_size / 1024**2
            print(f"   ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   ⚠️  {filename} not found!")

    # Create Kaggle training script
    create_kaggle_training_script(package_dir)

    # Create README
    create_kaggle_readme(package_dir, embeddings_dir / 'metadata.json')

    # Compress to tar.gz
    print(f"\n🗜️  Compressing package...")
    tar_path = output_dir / f"{package_name}.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(package_dir, arcname=package_name)

    compressed_size = tar_path.stat().st_size / 1024**2
    print(f"✅ Created: {tar_path} ({compressed_size:.1f} MB)")

    return tar_path


def create_kaggle_training_script(package_dir):
    """Create standalone training script for Kaggle."""

    script_content = '''#!/usr/bin/env python3
"""
Train BirdNET classifier on Kaggle GPU

This script trains on pre-extracted embeddings for fast GPU training.

Usage on Kaggle:
    1. Add this dataset as input
    2. Create new notebook
    3. Copy this script
    4. Run!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import joblib
import json
from tqdm.notebook import tqdm

# Dataset class
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# Deep MLP Classifier
class DeepMLPClassifier(nn.Module):
    def __init__(self, n_classes, embedding_dim=1024, hidden_dims=[512, 256, 128], dropout=0.4):
        super().__init__()

        self.input_proj = nn.Linear(embedding_dim, hidden_dims[0])

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        self.output = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, x):
        x = torch.relu(self.input_proj(x))
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def train():
    # Configuration
    EPOCHS = 200
    BATCH_SIZE = 512  # Large batch for GPU
    LR = 1e-3
    DROPOUT = 0.4
    PATIENCE = 20

    # Load data
    print("📊 Loading embeddings...")
    data_dir = Path("/kaggle/input/chirpkit-birdnet-embeddings")

    X_train = np.load(data_dir / "X_train_embeddings.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val_embeddings.npy")
    y_val = np.load(data_dir / "y_val.npy")

    # Load metadata
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    n_classes = metadata['n_classes']

    print(f"   Train: {X_train.shape}")
    print(f"   Val: {X_val.shape}")
    print(f"   Classes: {n_classes}")

    # Create datasets
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")

    model = DeepMLPClassifier(n_classes=n_classes, dropout=DROPOUT).to(device)
    print(f"🏗️  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\\n🚀 Starting training...")
    print("=" * 80)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)

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
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        scheduler.step(val_acc)

        print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Gap: {train_acc-val_acc:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, 'best_model.pth')

            print(f"🎉 New best! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️  Early stopping")
                break

    print(f"\\n✅ Training complete!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📈 Improvement over baseline (37%): +{best_val_acc - 37:.2f}%")


if __name__ == "__main__":
    train()
'''

    script_path = package_dir / "train_on_kaggle.py"
    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"   ✓ train_on_kaggle.py")


def create_kaggle_readme(package_dir, metadata_path):
    """Create README for Kaggle dataset."""

    # Load metadata
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    readme_content = f'''# ChirpKit BirdNET Embeddings

Pre-extracted BirdNET embeddings for insect classification.

## Dataset Information

- **Total Samples:** {metadata.get('n_train', 0) + metadata.get('n_val', 0):,}
- **Training:** {metadata.get('n_train', 0):,} samples
- **Validation:** {metadata.get('n_val', 0):,} samples
- **Species:** {metadata.get('n_classes', 0)} classes
- **Embedding Dimension:** 1024 (BirdNET features)
- **Created:** {metadata.get('created_at', 'N/A')}

## Files

- `X_train_embeddings.npy` - Training embeddings (n_train × 1024)
- `y_train.npy` - Training labels
- `X_val_embeddings.npy` - Validation embeddings (n_val × 1024)
- `y_val.npy` - Validation labels
- `label_encoder.joblib` - Species label encoder
- `metadata.json` - Dataset metadata
- `train_on_kaggle.py` - Ready-to-run training script

## Quick Start

```python
import numpy as np

# Load embeddings
X_train = np.load('X_train_embeddings.npy')
y_train = np.load('y_train.npy')

print(f"Training data shape: {{X_train.shape}}")
# Output: Training data shape: ({metadata.get('n_train', 0)}, 1024)
```

## Train on Kaggle (GPU)

```python
# Copy train_on_kaggle.py to your notebook and run:
!python train_on_kaggle.py

# Expected time: 10-30 minutes with GPU
# Expected accuracy: 45-60% (baseline: 37%)
```

## About BirdNET Embeddings

These embeddings were extracted using BirdNET, a pre-trained audio classifier
trained on millions of bird and animal sounds. Transfer learning from BirdNET
provides rich audio features that work well for insect classification.

**Advantages:**
- Pre-trained features (no need to train feature extractor)
- Fast training (10-30 min vs 12+ hours)
- Better generalization (learned from millions of samples)
- Compact storage (~120MB vs 6.2GB spectrograms)

## Baseline Performance

- **CNN-LSTM (from scratch):** 37% accuracy, 12 hours training
- **BirdNET Transfer (frozen):** 45-55% accuracy, 30 minutes training
- **BirdNET + Fine-tuning:** 60-75% accuracy, 4 hours training
- **Target with ensemble:** 70-80% accuracy

## Citation

If you use this dataset, please cite:
- BirdNET: https://github.com/kahst/BirdNET-Analyzer
- ChirpKit: https://github.com/yourusername/chirpkit
'''

    readme_path = package_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"   ✓ README.md")


def main():
    parser = argparse.ArgumentParser(
        description='Extract BirdNET embeddings for Kaggle training'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='combined',
        help='Dataset name (for output directory)'
    )

    parser.add_argument(
        '--raw-data-dirs',
        nargs='+',
        default=['data/raw/insectset459', 'data/raw/xenocanto'],
        help='Raw audio directories to process'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/embeddings',
        help='Directory to save embeddings'
    )

    parser.add_argument(
        '--kaggle-package-dir',
        type=str,
        default='data/embeddings_kaggle',
        help='Directory for Kaggle package'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.3,
        help='Validation split ratio'
    )

    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip extraction if embeddings exist'
    )

    args = parser.parse_args()

    print("🦗 BirdNET Embedding Extraction for Kaggle")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}/{args.dataset}")
    print("=" * 80)

    embeddings_dir = Path(args.output_dir) / args.dataset
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Check if embeddings already exist
    train_emb = embeddings_dir / "X_train_embeddings.npy"
    val_emb = embeddings_dir / "X_val_embeddings.npy"

    if train_emb.exists() and val_emb.exists() and args.skip_extraction:
        print(f"\n✅ Using existing embeddings from {embeddings_dir}")
    else:
        # Initialize BirdNET
        print(f"\n🔧 Initializing BirdNET...")
        extractor = BirdNETEmbeddingExtractor()

        # Collect audio files
        audio_files, species_labels = collect_audio_files(args.raw_data_dirs)

        if len(audio_files) == 0:
            print("❌ No audio files found!")
            print(f"   Searched: {args.raw_data_dirs}")
            print("   Make sure the directories contain .wav, .mp3, or .flac files")
            return

        # Create label encoder
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(species_labels)

        print(f"🦗 {len(label_encoder.classes_)} unique species")

        # Create train/val split
        print(f"\n📊 Creating {int((1-args.val_ratio)*100)}% train / {int(args.val_ratio*100)}% val split...")
        train_files, val_files, train_labels, val_labels = train_test_split(
            audio_files,
            encoded_labels,
            test_size=args.val_ratio,
            random_state=42,
            stratify=encoded_labels
        )

        print(f"   Train: {len(train_files)} samples")
        print(f"   Val: {len(val_files)} samples")

        # Extract embeddings
        start_time = time.time()

        print(f"\n🔍 Extracting TRAIN embeddings (this will take 2-4 hours)...")
        train_embeddings, train_labels_valid, train_failed = extract_embeddings_batch(
            extractor, train_files, train_labels, desc="Train"
        )

        print(f"\n🔍 Extracting VAL embeddings...")
        val_embeddings, val_labels_valid, val_failed = extract_embeddings_batch(
            extractor, val_files, val_labels, desc="Val"
        )

        elapsed = time.time() - start_time

        # Save embeddings
        print(f"\n💾 Saving embeddings...")
        np.save(embeddings_dir / "X_train_embeddings.npy", train_embeddings)
        np.save(embeddings_dir / "y_train.npy", train_labels_valid)
        np.save(embeddings_dir / "X_val_embeddings.npy", val_embeddings)
        np.save(embeddings_dir / "y_val.npy", val_labels_valid)
        joblib.dump(label_encoder, embeddings_dir / "label_encoder.joblib")

        # Save metadata
        metadata = {
            'dataset': args.dataset,
            'n_train': len(train_embeddings),
            'n_val': len(val_embeddings),
            'n_classes': len(label_encoder.classes_),
            'embedding_dim': 1024,
            'extraction_time_hours': elapsed / 3600,
            'created_at': datetime.now().isoformat(),
            'species': label_encoder.classes_.tolist(),
            'train_failed': len(train_failed),
            'val_failed': len(val_failed)
        }

        with open(embeddings_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Extraction complete in {elapsed/3600:.1f} hours")
        print(f"   Train: {train_embeddings.shape}")
        print(f"   Val: {val_embeddings.shape}")
        print(f"   Failed: {len(train_failed) + len(val_failed)} files")

    # Create Kaggle package
    tar_path = create_kaggle_package(embeddings_dir, args.kaggle_package_dir)

    print(f"\n" + "=" * 80)
    print(f"✅ READY FOR KAGGLE!")
    print("=" * 80)
    print(f"📦 Package: {tar_path}")
    print(f"💾 Size: {tar_path.stat().st_size / 1024**2:.1f} MB")
    print(f"\n📋 Next Steps:")
    print(f"1. Go to kaggle.com/datasets")
    print(f"2. Click 'New Dataset'")
    print(f"3. Upload: {tar_path.name}")
    print(f"4. Create notebook and add this dataset")
    print(f"5. Run: !python /kaggle/input/*/train_on_kaggle.py")
    print(f"6. Expected accuracy: 45-60% in 10-30 minutes!")
    print("=" * 80)


if __name__ == "__main__":
    main()
