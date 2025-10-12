#!/usr/bin/env python3
"""
Simple BirdNET embedding extraction that reuses existing preprocessing logic.

This script:
1. Uses your UnifiedDatasetProcessor to get the EXACT same audio files
2. Extracts BirdNET embeddings (highest quality from raw audio)
3. Creates matching train/val splits
4. Packages for Kaggle

Usage:
    python scripts/extract_embeddings_simple.py \
        --datasets insectset459 xenocanto \
        --min-samples 30 \
        --output data/embeddings/combined
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

# Import your existing preprocessing
sys.path.insert(0, str(Path(__file__).parent))
from preprocess_unified import UnifiedDatasetProcessor

from transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description='Extract BirdNET embeddings using existing preprocessing logic'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['insectset459', 'xenocanto'],
        choices=['insectsound1000', 'insectset459', 'sina', 'xenocanto'],
        help='Datasets to process'
    )

    parser.add_argument(
        '--min-samples',
        type=int,
        default=30,
        help='Minimum samples per species (matches your preprocessing)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.3,
        help='Validation split ratio (matches your preprocessing)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/embeddings/combined',
        help='Output directory'
    )

    parser.add_argument(
        '--base-data-dir',
        type=str,
        default='data/raw',
        help='Base directory for raw data'
    )

    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip if embeddings exist'
    )

    parser.add_argument(
        '--create-kaggle-package',
        action='store_true',
        default=True,
        help='Create Kaggle upload package'
    )

    args = parser.parse_args()

    print("🦗 BirdNET Embedding Extraction (Using Existing Preprocessing)")
    print("=" * 80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Min samples per species: {args.min_samples}")
    print(f"Output: {args.output}")
    print("=" * 80)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    if args.skip_extraction:
        train_emb = output_dir / "X_train_embeddings.npy"
        val_emb = output_dir / "X_val_embeddings.npy"
        if train_emb.exists() and val_emb.exists():
            print(f"\n✅ Embeddings already exist, skipping extraction")
            if args.create_kaggle_package:
                print(f"\n📦 Creating Kaggle package...")
                from extract_embeddings_for_kaggle import create_kaggle_package
                tar_path = create_kaggle_package(output_dir, "data/embeddings_kaggle")
                print(f"✅ Package ready: {tar_path}")
            return

    # Initialize your preprocessor
    print(f"\n🔧 Initializing dataset processor...")
    processor = UnifiedDatasetProcessor(base_data_dir=args.base_data_dir, use_enhanced=False)

    # Collect audio files from all datasets
    all_audio_files = []
    all_species = []

    for dataset_name in args.datasets:
        print(f"\n📂 Loading {dataset_name}...")
        try:
            metadata_df = processor.load_metadata(dataset_name)

            # Get audio file paths and species
            for idx, row in metadata_df.iterrows():
                audio_path = row.get('audio_path') or row.get('filepath')
                species = row.get('species') or row.get('scientific_name')

                if audio_path and species:
                    audio_file = Path(args.base_data_dir) / dataset_name / audio_path
                    if audio_file.exists():
                        all_audio_files.append(str(audio_file))
                        all_species.append(species)

            print(f"   Found {len([s for s in all_species if s in metadata_df['species'].unique()])} files")

        except Exception as e:
            print(f"   ⚠️  Error loading {dataset_name}: {e}")
            print(f"   This is expected if dataset doesn't exist locally")
            continue

    if len(all_audio_files) == 0:
        print("\n❌ No audio files found!")
        print(f"   Make sure datasets exist in: {args.base_data_dir}")
        return

    print(f"\n✅ Collected {len(all_audio_files)} total audio files")

    # Filter species with <min_samples
    from collections import Counter
    species_counts = Counter(all_species)

    print(f"\n🔍 Filtering species with <{args.min_samples} samples...")
    print(f"   Before: {len(species_counts)} species")

    valid_species = {sp for sp, count in species_counts.items() if count >= args.min_samples}

    filtered_files = []
    filtered_species = []
    for audio_file, species in zip(all_audio_files, all_species):
        if species in valid_species:
            filtered_files.append(audio_file)
            filtered_species.append(species)

    print(f"   After: {len(valid_species)} species")
    print(f"   Files: {len(filtered_files)}")

    # Create label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(filtered_species)

    # Create train/val split
    print(f"\n📊 Creating train/val split ({int((1-args.val_ratio)*100)}% / {int(args.val_ratio*100)}%)...")
    train_files, val_files, train_labels, val_labels = train_test_split(
        filtered_files,
        encoded_labels,
        test_size=args.val_ratio,
        random_state=42,
        stratify=encoded_labels
    )

    print(f"   Train: {len(train_files)} files")
    print(f"   Val: {len(val_files)} files")

    # Initialize BirdNET
    print(f"\n🔧 Initializing BirdNET extractor...")
    extractor = BirdNETEmbeddingExtractor()

    # Extract embeddings
    print(f"\n🔍 Extracting embeddings (this will take 2-4 hours)...")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")

    start_time = time.time()

    # Train embeddings
    print(f"\n📊 Processing TRAIN set...")
    train_embeddings = []
    train_labels_valid = []
    train_failed = []

    for audio_file, label in tqdm(list(zip(train_files, train_labels)), desc="Train"):
        try:
            emb = extractor.extract_embeddings_from_audio(audio_file, aggregate='mean')
            if np.any(emb):
                train_embeddings.append(emb)
                train_labels_valid.append(label)
            else:
                train_failed.append(audio_file)
        except Exception as e:
            train_failed.append(audio_file)

    # Val embeddings
    print(f"\n📊 Processing VAL set...")
    val_embeddings = []
    val_labels_valid = []
    val_failed = []

    for audio_file, label in tqdm(list(zip(val_files, val_labels)), desc="Val"):
        try:
            emb = extractor.extract_embeddings_from_audio(audio_file, aggregate='mean')
            if np.any(emb):
                val_embeddings.append(emb)
                val_labels_valid.append(label)
            else:
                val_failed.append(audio_file)
        except Exception as e:
            val_failed.append(audio_file)

    elapsed = time.time() - start_time

    # Convert to numpy
    train_embeddings = np.array(train_embeddings, dtype=np.float32)
    train_labels_valid = np.array(train_labels_valid, dtype=np.int64)
    val_embeddings = np.array(val_embeddings, dtype=np.float32)
    val_labels_valid = np.array(val_labels_valid, dtype=np.int64)

    # Save
    print(f"\n💾 Saving embeddings...")
    np.save(output_dir / "X_train_embeddings.npy", train_embeddings)
    np.save(output_dir / "y_train.npy", train_labels_valid)
    np.save(output_dir / "X_val_embeddings.npy", val_embeddings)
    np.save(output_dir / "y_val.npy", val_labels_valid)
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    # Save metadata
    metadata = {
        'datasets': args.datasets,
        'min_samples_per_species': args.min_samples,
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

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Extraction complete!")
    print(f"   Time: {elapsed/3600:.1f} hours")
    print(f"   Train: {train_embeddings.shape}")
    print(f"   Val: {val_embeddings.shape}")
    print(f"   Species: {len(label_encoder.classes_)}")
    print(f"   Failed: {len(train_failed) + len(val_failed)} files")

    # Create Kaggle package
    if args.create_kaggle_package:
        print(f"\n📦 Creating Kaggle package...")
        from extract_embeddings_for_kaggle import create_kaggle_package
        tar_path = create_kaggle_package(output_dir, "data/embeddings_kaggle")

        print(f"\n" + "=" * 80)
        print(f"✅ READY FOR KAGGLE!")
        print("=" * 80)
        print(f"📦 Upload this file to Kaggle:")
        print(f"   {tar_path}")
        print(f"💾 Size: {tar_path.stat().st_size / 1024**2:.1f} MB")
        print(f"\n📋 Then train on Kaggle GPU (10-30 min):")
        print(f"   !python /kaggle/input/*/train_on_kaggle.py")
        print("=" * 80)


if __name__ == "__main__":
    main()
