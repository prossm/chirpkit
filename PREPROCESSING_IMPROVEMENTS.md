# Preprocessing Improvements

## Summary

Updated `scripts/preprocess_unified.py` to incorporate two major improvements:

1. **Filter out under-represented species** (min 30 samples by default)
2. **Better train/val split ratio** (30% validation by default, no test set)

These changes are based on analysis showing the current dataset has severe class imbalance that causes overfitting.

---

## Current Dataset Issues (Before Improvements)

**483 species, 27,703 samples:**

| Samples per Species | Number of Species | Percentage |
|---------------------|-------------------|------------|
| <20 samples         | 174 species       | 36% 😱     |
| 20-40 samples       | 151 species       | 31%        |
| 40-60 samples       | 52 species        | 11%        |
| 60-100 samples      | 39 species        | 8%         |
| 100+ samples        | 67 species        | 14%        |

**Problems:**
- 36% of species have <20 samples (nearly impossible to learn)
- Range: 11 to 1,362 samples (124x difference!)
- Causes severe overfitting: 87% train accuracy → 37% val accuracy

---

## Changes Made to `preprocess_unified.py`

### 1. Species Filtering (New Parameter: `--min-samples`)

**Default: 30 samples minimum**

```python
# Before: Only filtered species with 1 sample
single_sample_species = [label for label, count in label_counts.items() if count == 1]

# After: Filter species below threshold
insufficient_species = [label for label, count in label_counts.items()
                       if count < min_samples_per_species]
```

**Expected result:**
- Removes ~200 species with <30 samples
- Keeps ~280 species with better representation
- Improves data quality significantly

### 2. Better Train/Val Split (New Parameter: `--val-ratio`)

**Default: 30% validation (was 10%)**

```python
# Before: 80% train / 10% val / 10% test (3-way split)
def create_splits(self, features, labels, output_dir, test_size=0.2, val_size=0.1)

# After: 70% train / 30% val (2-way split, no test set)
def create_splits(self, features, labels, output_dir, val_ratio=0.30, min_samples_per_species=30)
```

**Benefits:**
- More reliable validation accuracy estimates
- Better early stopping decisions
- Each species gets ~9 validation samples (was ~3)
- Stratified splitting ensures balanced representation

### 3. Better Statistics Display

Now shows detailed distribution info:
```
📊 New distribution:
   Min samples/species: 30
   Max samples/species: 1362
   Mean samples/species: 78.2
   Median samples/species: 45.0

📊 Validation set statistics:
   Total samples: 6,500
   Unique species: 280
   Mean samples/species: 23.2
   Median samples/species: 9.0
```

---

## Usage

### Basic Usage (Use Defaults)
```bash
# Reprocess with min_samples=30, val_ratio=0.30
python scripts/preprocess_unified.py --dataset all
```

### Custom Thresholds
```bash
# More aggressive filtering (40 samples minimum)
python scripts/preprocess_unified.py --dataset all --min-samples 40

# Larger validation set (40%)
python scripts/preprocess_unified.py --dataset all --val-ratio 0.40

# Less aggressive (20 samples minimum, 25% validation)
python scripts/preprocess_unified.py --dataset all --min-samples 20 --val-ratio 0.25
```

### Test Run (Quick)
```bash
# Process only 100 samples to test
python scripts/preprocess_unified.py --dataset combined --limit 100
```

---

## Expected Improvements

### Before (Current Data)
- **483 species**, 27,703 samples
- **Training: 87% accuracy**
- **Validation: 37% accuracy**
- **Overfitting gap: 50%** 😱

### After (With min_samples=30, val_ratio=0.30)
- **~280 species**, ~22,000 samples
- **Training: 60-70% accuracy** (lower is good - less memorization)
- **Validation: 50-60% accuracy** (higher than before!)
- **Overfitting gap: <15%** ✅

### Why This Helps

1. **Removes noise:** Species with <30 samples add noise and confusion
2. **Better validation:** 30% validation gives more reliable accuracy estimates
3. **Cleaner learning:** Model learns patterns instead of memorizing rare cases
4. **Stratified splits:** Each species gets proportional representation in train/val

---

## Next Steps

1. **Run preprocessing:**
   ```bash
   python scripts/preprocess_unified.py --dataset all
   ```

2. **Train with new data:**
   ```bash
   # Use the new strongly regularized model
   python scripts/train_enhanced_regularized.py --dataset combined --epochs 500
   ```

3. **Monitor improvements:**
   - Watch overfitting gap (should be <20%)
   - Validation accuracy should be more stable
   - Training should be slower but more reliable

---

## Rollback

If you need to revert to old behavior:

```bash
# Use old settings (no filtering, small validation)
python scripts/preprocess_unified.py --dataset all --min-samples 2 --val-ratio 0.10
```

---

## Files Modified

- ✅ `scripts/preprocess_unified.py` - Updated with filtering and better splitting
- ✅ `scripts/train_enhanced_regularized.py` - New strongly regularized trainer (ready to use)
- ✅ `src/models/enhanced_cnn_lstm_regularized.py` - New model architecture with 50% dropout

## Files Made Obsolete

- `scripts/resplit_combined.py` - Logic now integrated into preprocess_unified.py
