#!/usr/bin/env python3
"""
Resplit the combined dataset with a larger validation set for better performance estimates
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

def resplit_combined(val_ratio=0.25, min_samples_per_class=2):
    """
    Resplit combined dataset with better validation ratio

    Args:
        val_ratio: Validation set ratio (0.25 = 25%)
        min_samples_per_class: Minimum samples per class to use stratified split
    """
    # Load current combined data
    data_dir = Path('data/splits/combined')

    print("📂 Loading current combined dataset...")
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy', allow_pickle=True)
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy', allow_pickle=True)

    # Combine all data
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    print(f"✅ Total samples: {len(X_all)}")
    print(f"✅ Unique species: {len(np.unique(y_all))}")

    # Check class distribution
    class_counts = Counter(y_all)
    min_count = min(class_counts.values())

    print(f"\n📊 Class distribution:")
    print(f"  Min samples per species: {min_count}")
    print(f"  Max samples per species: {max(class_counts.values())}")
    print(f"  Mean samples per species: {np.mean(list(class_counts.values())):.1f}")
    print(f"  Median samples per species: {np.median(list(class_counts.values())):.1f}")

    # Determine if we can use stratified split
    if min_count >= min_samples_per_class:
        print(f"\n✅ Using stratified split (min_count={min_count} >= {min_samples_per_class})")
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
            X_all, y_all,
            test_size=val_ratio,
            stratify=y_all,
            random_state=42
        )
    else:
        print(f"\n⚠️  Using random split (min_count={min_count} < {min_samples_per_class})")
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
            X_all, y_all,
            test_size=val_ratio,
            random_state=42
        )

    # Show new split stats
    print(f"\n📊 New split:")
    print(f"  Train: {len(X_train_new)} ({len(X_train_new)/len(X_all)*100:.1f}%)")
    print(f"  Val: {len(X_val_new)} ({len(X_val_new)/len(X_all)*100:.1f}%)")

    # Check validation distribution
    val_counts = Counter(y_val_new)
    print(f"\n📊 Validation set per species:")
    print(f"  Min: {min(val_counts.values())} samples")
    print(f"  Max: {max(val_counts.values())} samples")
    print(f"  Mean: {np.mean(list(val_counts.values())):.1f} samples")
    print(f"  Median: {np.median(list(val_counts.values())):.1f} samples")
    print(f"  Species with <5 val samples: {sum(1 for c in val_counts.values() if c < 5)}")

    # Backup old splits
    backup_dir = data_dir / 'backup_old_split'
    backup_dir.mkdir(exist_ok=True)

    print(f"\n💾 Backing up old splits to {backup_dir}...")
    np.save(backup_dir / 'X_train.npy', X_train)
    np.save(backup_dir / 'y_train.npy', y_train)
    np.save(backup_dir / 'X_val.npy', X_val)
    np.save(backup_dir / 'y_val.npy', y_val)

    # Save new splits
    print(f"💾 Saving new splits to {data_dir}...")
    np.save(data_dir / 'X_train.npy', X_train_new)
    np.save(data_dir / 'y_train.npy', y_train_new)
    np.save(data_dir / 'X_val.npy', X_val_new)
    np.save(data_dir / 'y_val.npy', y_val_new)

    print("\n✅ Done! New splits saved.")
    print(f"   Old splits backed up to: {backup_dir}")
    print(f"\n🎯 Expected improvement:")
    print(f"   - Better validation accuracy estimates")
    print(f"   - More reliable early stopping")
    print(f"   - Each species has ~{np.median(list(val_counts.values())):.0f} validation samples (was ~3)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Resplit combined dataset')
    parser.add_argument('--val-ratio', type=float, default=0.25,
                       help='Validation ratio (default: 0.25 = 25%%)')
    args = parser.parse_args()

    resplit_combined(val_ratio=args.val_ratio)
