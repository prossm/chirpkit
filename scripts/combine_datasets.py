#!/usr/bin/env python3
"""
Combine individual dataset splits into a unified combined dataset
Applies min_samples filtering and creates stratified train/val split
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse

def combine_datasets(datasets=['insectset459', 'sina', 'xenocanto'],
                     min_samples_per_species=30,
                     val_ratio=0.30,
                     output_dir='data/splits/combined'):
    """
    Combine multiple dataset splits into a single unified dataset

    Args:
        datasets: List of dataset names to combine
        min_samples_per_species: Minimum samples required per species
        val_ratio: Validation set ratio (default 0.30 = 30%)
        output_dir: Where to save combined splits
    """

    print("🔗 Combining Datasets into Unified Split")
    print("=" * 80)

    all_features = []
    all_labels = []
    dataset_sources = []

    # Load all datasets
    for dataset_name in datasets:
        splits_dir = Path(f'data/splits/{dataset_name}')

        if not splits_dir.exists():
            print(f"⚠️  Skipping {dataset_name} - directory not found")
            continue

        train_features_path = splits_dir / 'X_train.npy'
        train_labels_path = splits_dir / 'y_train.npy'
        val_features_path = splits_dir / 'X_val.npy'
        val_labels_path = splits_dir / 'y_val.npy'

        if not all(p.exists() for p in [train_features_path, train_labels_path,
                                         val_features_path, val_labels_path]):
            print(f"⚠️  Skipping {dataset_name} - split files not found")
            continue

        print(f"\n📂 Loading {dataset_name}...")

        # Load train and val splits
        X_train = np.load(train_features_path)
        y_train = np.load(train_labels_path, allow_pickle=True)
        X_val = np.load(val_features_path)
        y_val = np.load(val_labels_path, allow_pickle=True)

        # Combine train and val (we'll re-split later)
        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)

        print(f"   Samples: {len(X_all)}")
        print(f"   Species: {len(np.unique(y_all))}")

        all_features.append(X_all)
        all_labels.append(y_all)
        dataset_sources.extend([dataset_name] * len(X_all))

    if not all_features:
        print("❌ No datasets found to combine!")
        return

    # Concatenate all data
    print(f"\n🔀 Merging all datasets...")
    X_combined = np.concatenate(all_features, axis=0)
    y_combined = np.concatenate(all_labels, axis=0)

    print(f"✅ Combined: {len(X_combined)} samples from {len(datasets)} datasets")
    print(f"🦗 Total unique species: {len(np.unique(y_combined))}")

    # Show per-dataset contribution
    print(f"\n📊 Dataset contributions:")
    from collections import Counter
    source_counts = Counter(dataset_sources)
    for ds, count in source_counts.items():
        print(f"   {ds}: {count} samples ({count/len(X_combined)*100:.1f}%)")

    # Filter out species with insufficient samples
    print(f"\n🔍 Filtering species with <{min_samples_per_species} samples...")

    label_counts = Counter(y_combined)
    insufficient_species = [label for label, count in label_counts.items()
                           if count < min_samples_per_species]

    if insufficient_species:
        print(f"⚠️  Found {len(insufficient_species)} species with <{min_samples_per_species} samples")
        print(f"🗑️  Removing under-represented species for better generalization")

        # Show distribution before filtering
        removed_samples = sum(label_counts[label] for label in insufficient_species)
        print(f"   Removing {removed_samples} total samples from {len(insufficient_species)} species")

        # Create mask to filter out insufficient species
        mask = np.array([label not in insufficient_species for label in y_combined])
        X_combined = X_combined[mask]
        y_combined = y_combined[mask]

        print(f"📊 After filtering: {len(X_combined)} samples, {len(np.unique(y_combined))} species")

        # Show new distribution
        new_counts = Counter(y_combined)
        print(f"📊 New distribution:")
        print(f"   Min samples/species: {min(new_counts.values())}")
        print(f"   Max samples/species: {max(new_counts.values())}")
        print(f"   Mean samples/species: {np.mean(list(new_counts.values())):.1f}")
        print(f"   Median samples/species: {np.median(list(new_counts.values())):.1f}")
    else:
        print(f"✅ All species have ≥{min_samples_per_species} samples")

    # Create stratified train/val split
    print(f"\n🔀 Creating train/val split ({int((1-val_ratio)*100)}%/{int(val_ratio*100)}%)...")

    min_class_count = min(Counter(y_combined).values())

    # Use stratified split if all classes have at least 2 samples
    if min_class_count >= 2:
        print(f"✅ Using stratified split (all species have ≥2 samples)")
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_combined,
            test_size=val_ratio,
            stratify=y_combined,
            random_state=42
        )
    else:
        print(f"⚠️  Using random split (some species have only 1 sample)")
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_combined,
            test_size=val_ratio,
            random_state=42
        )

    # Validate validation set distribution
    val_counts = Counter(y_val)
    print(f"\n📊 Validation set statistics:")
    print(f"   Total samples: {len(y_val)}")
    print(f"   Unique species: {len(val_counts)}")
    print(f"   Min samples/species: {min(val_counts.values())}")
    print(f"   Max samples/species: {max(val_counts.values())}")
    print(f"   Mean samples/species: {np.mean(list(val_counts.values())):.1f}")
    print(f"   Median samples/species: {np.median(list(val_counts.values())):.1f}")

    species_with_few_val = sum(1 for c in val_counts.values() if c < 5)
    if species_with_few_val > 0:
        print(f"   ⚠️  {species_with_few_val} species have <5 validation samples")

    # Save combined splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving combined splits to {output_path}...")
    np.save(output_path / 'X_train.npy', X_train)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'X_val.npy', X_val)
    np.save(output_path / 'y_val.npy', y_val)

    print(f"\n✅ Combined dataset created successfully!")
    print(f"📊 Train: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_val))*100:.1f}%)")
    print(f"📊 Val: {len(X_val)} ({len(X_val)/(len(X_train)+len(X_val))*100:.1f}%)")
    print(f"🦗 {len(np.unique(y_combined))} unique species retained")

    # Save metadata about which datasets were combined
    import json
    metadata = {
        'datasets_combined': list(source_counts.keys()),
        'dataset_contributions': dict(source_counts),
        'total_samples': len(X_combined),
        'total_species': len(np.unique(y_combined)),
        'min_samples_per_species': min_samples_per_species,
        'val_ratio': val_ratio,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'species_removed': len(insufficient_species) if insufficient_species else 0,
        'created_at': str(np.datetime64('now'))
    }

    with open(output_path / 'combined_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"📋 Metadata saved to {output_path / 'combined_metadata.json'}")

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Combine individual dataset splits into unified combined dataset')
    parser.add_argument('--datasets', nargs='+',
                       default=['insectset459', 'sina', 'xenocanto'],
                       help='Datasets to combine (default: insectset459 sina xenocanto)')
    parser.add_argument('--min-samples', type=int, default=30,
                       help='Minimum samples per species (default: 30)')
    parser.add_argument('--val-ratio', type=float, default=0.30,
                       help='Validation set ratio (default: 0.30 = 30%%)')
    parser.add_argument('--output-dir', type=str, default='data/splits/combined',
                       help='Output directory (default: data/splits/combined)')

    args = parser.parse_args()

    combine_datasets(
        datasets=args.datasets,
        min_samples_per_species=args.min_samples,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
