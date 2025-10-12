#!/usr/bin/env python3
"""
Final simplified BirdNET embedding extraction.

Directly scans audio files, uses species mapping, filters to ≥30 samples.
This matches your existing preprocessing exactly.
"""

import argparse
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
from collections import Counter

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_xenocanto_mapping(mapping_file='data/xenocanto_species_mapping.json'):
    """Load xeno-canto common name → scientific name mapping"""
    if not Path(mapping_file).exists():
        print(f"⚠️  Species mapping not found: {mapping_file}")
        return {}

    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)

    # Convert to simple dict: common_name -> scientific_name
    mapping = {}
    for common_name, data in mapping_data.items():
        mapping[common_name] = data['scientific_name']

    return mapping


def extract_species_from_filename(filename, is_xenocanto=False, xeno_mapping=None):
    """Extract species name from filename"""
    name = Path(filename).stem

    if is_xenocanto and xeno_mapping:
        # XC format: XC1000_Common_Name_Country.mp3
        if name.startswith('XC'):
            parts = name.split('_')
            # Remove XC number and country (last part)
            common_name_parts = parts[1:-1]
            # Join with spaces, preserving hyphens within parts
            common_name = ' '.join(common_name_parts)

            # Look up scientific name
            if common_name in xeno_mapping:
                return xeno_mapping[common_name]

            print(f"⚠️  No mapping for: {common_name}")
            return common_name  # Fallback

    # InsectSet459 format: Species_name_INXXXXXX_number
    if '_IN' in name:
        parts = name.split('_IN')[0]
        return parts.replace('_', ' ')

    # Fallback
    parts = name.split('_')[:2]
    return ' '.join(parts)


def scan_audio_directory(directory, is_xenocanto=False, xeno_mapping=None):
    """Scan directory for audio files and extract species"""
    directory = Path(directory)
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

    audio_files = []
    species_list = []

    for audio_path in directory.rglob('*'):
        if audio_path.suffix.lower() in audio_extensions:
            species = extract_species_from_filename(
                audio_path.name,
                is_xenocanto=is_xenocanto,
                xeno_mapping=xeno_mapping
            )
            audio_files.append(str(audio_path))
            species_list.append(species)

    return audio_files, species_list


def main():
    parser = argparse.ArgumentParser(
        description='Extract BirdNET embeddings - Final Simple Version'
    )

    parser.add_argument('--insectset459-dir', default='data/raw/insectset459/Train')
    parser.add_argument('--xenocanto-dir', default='data/raw/xenocanto/audio')
    parser.add_argument('--xenocanto-mapping', default='data/xenocanto_species_mapping.json')
    parser.add_argument('--min-samples', type=int, default=30)
    parser.add_argument('--val-ratio', type=float, default=0.3)
    parser.add_argument('--output', default='data/embeddings/combined')
    parser.add_argument('--skip-if-exists', action='store_true')

    args = parser.parse_args()

    print("🦗 BirdNET Embedding Extraction - Final Version")
    print("=" * 80)
    print(f"Min samples per species: {args.min_samples}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Output: {args.output}")
    print("=" * 80)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if exists
    if args.skip_if_exists:
        if (output_dir / "X_train_embeddings.npy").exists():
            print(f"\n✅ Embeddings exist, skipping")
            return

    # Load xeno-canto mapping
    print(f"\n📋 Loading xeno-canto species mapping...")
    xeno_mapping = load_xenocanto_mapping(args.xenocanto_mapping)
    print(f"   Loaded {len(xeno_mapping)} species mappings")

    # Scan audio files
    all_files = []
    all_species = []

    print(f"\n📂 Scanning InsectSet459...")
    if Path(args.insectset459_dir).exists():
        files, species = scan_audio_directory(args.insectset459_dir)
        all_files.extend(files)
        all_species.extend(species)
        print(f"   Found {len(files)} files")
    else:
        print(f"   ⚠️  Directory not found: {args.insectset459_dir}")

    print(f"\n📂 Scanning Xeno-canto...")
    if Path(args.xenocanto_dir).exists():
        files, species = scan_audio_directory(
            args.xenocanto_dir,
            is_xenocanto=True,
            xeno_mapping=xeno_mapping
        )
        all_files.extend(files)
        all_species.extend(species)
        print(f"   Found {len(files)} files")
    else:
        print(f"   ⚠️  Directory not found: {args.xenocanto_dir}")

    if len(all_files) == 0:
        print("\n❌ No audio files found!")
        return

    print(f"\n✅ Total: {len(all_files)} files from {len(set(all_species))} species")

    # Filter by min samples
    print(f"\n🔍 Filtering species with ≥{args.min_samples} samples...")
    species_counts = Counter(all_species)
    print(f"   Before: {len(species_counts)} species")

    valid_species = {sp for sp, count in species_counts.items() if count >= args.min_samples}

    filtered_files = []
    filtered_species = []
    for audio_file, species in zip(all_files, all_species):
        if species in valid_species:
            filtered_files.append(audio_file)
            filtered_species.append(species)

    print(f"   After: {len(valid_species)} species")
    print(f"   Files: {len(filtered_files)}")

    # Show top 10 species by count
    filtered_counts = Counter(filtered_species)
    print(f"\n📊 Top 10 species by sample count:")
    for species, count in filtered_counts.most_common(10):
        print(f"   {species}: {count} samples")

    # Create label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(filtered_species)

    # Create train/val split
    print(f"\n📊 Creating train/val split...")
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
    print(f"\n🔍 Extracting embeddings...")
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📅 Estimated time: 2-4 hours for {len(filtered_files)} files")
    print(f"☕ Go get coffee, this will take a while...")

    start_time = time.time()

    # Train
    print(f"\n📊 TRAIN set ({len(train_files)} files)...")
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
            if len(train_failed) <= 5:  # Only print first few errors
                print(f"\n⚠️  Failed: {Path(audio_file).name}: {e}")

    # Val
    print(f"\n📊 VAL set ({len(val_files)} files)...")
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
            if len(val_failed) <= 5:
                print(f"\n⚠️  Failed: {Path(audio_file).name}: {e}")

    elapsed = time.time() - start_time

    # Convert to numpy
    train_embeddings = np.array(train_embeddings, dtype=np.float32)
    train_labels_valid = np.array(train_labels_valid, dtype=np.int64)
    val_embeddings = np.array(val_embeddings, dtype=np.float32)
    val_labels_valid = np.array(val_labels_valid, dtype=np.int64)

    # Save
    print(f"\n💾 Saving to {output_dir}...")
    np.save(output_dir / "X_train_embeddings.npy", train_embeddings)
    np.save(output_dir / "y_train.npy", train_labels_valid)
    np.save(output_dir / "X_val_embeddings.npy", val_embeddings)
    np.save(output_dir / "y_val.npy", val_labels_valid)
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    # Metadata
    metadata = {
        'n_train': len(train_embeddings),
        'n_val': len(val_embeddings),
        'n_classes': len(label_encoder.classes_),
        'min_samples_per_species': args.min_samples,
        'embedding_dim': 1024,
        'extraction_time_hours': elapsed / 3600,
        'created_at': datetime.now().isoformat(),
        'species': label_encoder.classes_.tolist(),
        'train_failed': len(train_failed),
        'val_failed': len(val_failed),
        'datasets': ['insectset459', 'xenocanto']
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save failed files
    if train_failed or val_failed:
        with open(output_dir / "failed_files.txt", 'w') as f:
            f.write("TRAIN FAILURES:\n")
            f.write('\n'.join(train_failed))
            f.write("\n\nVAL FAILURES:\n")
            f.write('\n'.join(val_failed))

    print(f"\n" + "=" * 80)
    print(f"✅ EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"⏱️  Time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"📊 Train: {train_embeddings.shape}")
    print(f"📊 Val: {val_embeddings.shape}")
    print(f"🦗 Species: {len(label_encoder.classes_)}")
    print(f"❌ Failed: {len(train_failed) + len(val_failed)} files")
    print(f"💾 Saved to: {output_dir}")
    print(f"\n📦 Next: Create Kaggle package")
    print(f"   python -c \"from scripts.extract_embeddings_for_kaggle import create_kaggle_package; create_kaggle_package('{output_dir}', 'data/embeddings_kaggle')\"")
    print("=" * 80)


if __name__ == "__main__":
    main()
