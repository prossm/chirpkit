# ChirpKit Dataset Summary

**Last Updated:** 2025-10-06

---

## Combined Dataset (Ready for Training)

**Location:** `data/splits/combined/`

| Metric | Value |
|--------|-------|
| **Total Samples** | 29,723 |
| **Species** | 255 |
| **Train Samples** | 20,806 (70%) |
| **Val Samples** | 8,917 (30%) |
| **Min Samples/Species** | 30 |
| **Median Val Samples/Species** | 18 |
| **Mean Samples/Species** | 116.6 |

### Source Datasets
- **insectset459:** 16,594 samples (55.8%)
- **xenocanto:** 13,129 samples (44.2%)
- **sina:** Excluded (all 203 species had <30 samples)

---

## Individual Dataset Details

### InsectSet459
| Metric | Value |
|--------|-------|
| **Total Samples** | 16,594 |
| **Species** | 149 |
| **Train Samples** | 11,615 (70%) |
| **Val Samples** | 4,979 (30%) |
| **Min Samples/Species** | 30 |
| **Max Samples/Species** | 718 |
| **Mean Samples/Species** | 111.4 |

**Region:** Global
**Filtered Out:** Species with <30 samples

### Xeno-canto
| Metric | Value |
|--------|-------|
| **Total Samples** | 13,129 |
| **Species** | 130 |
| **Train Samples** | 9,190 (70%) |
| **Val Samples** | 3,939 (30%) |
| **Min Samples/Species** | 30 |
| **Max Samples/Species** | 1,205 |
| **Mean Samples/Species** | 101.0 |

**Region:** Global Community
**Filtered Out:** 285 species with <30 samples

### SINA (Excluded)
| Metric | Value |
|--------|-------|
| **Total Samples** | 265 |
| **Species** | 203 |
| **Max Samples/Species** | 6 |
| **Mean Samples/Species** | 1.3 |

**Region:** North America
**Status:** Completely filtered out - all species had <30 samples
**Why Excluded:** Dataset has excellent species diversity but very few samples per species. Not suitable for deep learning without significant data collection.

---

## Preprocessing Parameters

### Filtering
- **Minimum Samples Per Species:** 30
- **Rationale:** Ensures minimum of 21 train + 9 val samples per species after 70/30 split

### Train/Val Split
- **Ratio:** 70% train / 30% validation
- **Method:** Stratified (proportional representation of each species)
- **Random Seed:** 42 (reproducible)

### Audio Features
- **Sample Rate:** 22,050 Hz (captures up to 11 kHz insect sounds)
- **Duration:** 2.5 seconds (zero-padded or truncated)
- **FFT Size:** 4,096 (5.4 Hz frequency resolution)
- **Hop Length:** 256 (11.6ms temporal resolution)
- **Mel Bins:** 256 (2× standard resolution)
- **MFCCs:** 40 with deltas and delta-deltas

---

## Species Distribution Analysis

### Combined Dataset

**By Sample Count:**
| Range | Species Count | Percentage |
|-------|---------------|------------|
| 30-50 samples | ~140 species | ~55% |
| 51-100 samples | ~60 species | ~24% |
| 101-200 samples | ~35 species | ~14% |
| 201+ samples | ~20 species | ~8% |

**Key Statistics:**
- **Median:** 51.5 samples/species
- **75th Percentile:** ~100 samples/species
- **95th Percentile:** ~300 samples/species

### Validation Set Quality

**Per-Species Validation Samples:**
- **Minimum:** 9 samples
- **Median:** 18 samples
- **Mean:** 35 samples
- **Maximum:** 511 samples

**Quality Assessment:**
- ✅ All species have ≥9 validation samples (statistically significant)
- ✅ Median 18 samples provides good accuracy estimates
- ✅ Stratified split ensures proportional representation

---

## Data Quality Improvements

### Before Filtering (Original Combined)
- **Species:** 483
- **Total Samples:** 27,703
- **Issues:**
  - 174 species with <20 samples (36%)
  - 325 species with <30 samples (67%)
  - Severe class imbalance (11 to 1,362 samples per species)
  - Training: 87% accuracy → Validation: 37% accuracy (50% overfitting gap!)

### After Filtering (Current)
- **Species:** 255 (47% reduction)
- **Total Samples:** 29,723 (7% increase from combining datasets)
- **Improvements:**
  - ✅ All species have ≥30 samples
  - ✅ Better class balance (30 to 1,205 samples per species)
  - ✅ Expected: Train 60-70%, Val 50-60% (10-15% overfitting gap)
  - ✅ More reliable validation accuracy estimates

---

## Training Recommendations

### Expected Performance

**With Current Dataset (255 species, ~117 samples/species average):**
- **Baseline CNN-LSTM:** 40-45% validation accuracy
- **Enhanced CNN-LSTM:** 45-50% validation accuracy
- **Regularized Enhanced:** 50-60% validation accuracy (target)
- **Ensemble Methods:** 55-65% validation accuracy

**Overfitting Goals:**
- Target gap: <15% (was 50%)
- Acceptable gap: <20%
- Warning threshold: >25%

### Training Strategy

1. **Start with `train_enhanced_regularized.py`:**
   ```bash
   python scripts/train_enhanced_regularized.py --dataset combined --epochs 500
   ```

2. **Monitor overfitting gap:**
   - Watch for train_acc - val_acc
   - Should stay below 20% throughout training
   - Early stopping at 50 epochs patience

3. **Expected timeline:**
   - Epochs 1-50: Rapid learning, gap widens
   - Epochs 50-150: Gap narrows, validation improves
   - Epochs 150-300: SWA active, final refinement
   - Best model likely around epoch 200-250

---

## Future Data Collection Priorities

### To Reach 70%+ Accuracy

**Priority 1: Fill Out Existing Species**
- Target: 100+ samples per species
- Focus on species with 30-50 samples currently
- Would add ~7,000 samples

**Priority 2: High-Quality Rare Species**
- Re-evaluate SINA species if more samples collected
- Consider species with 10-29 samples from other datasets
- Potential: +50 species if data quality improves

**Priority 3: New Species (Lower Priority)**
- Only add new species with 50+ samples
- Maintain data quality standards

### Alternative: Transfer Learning
- Use AudioSet (2M samples) for pre-training
- Fine-tune on insect audio
- Could reach 70%+ with current data

---

## File Locations

### Preprocessed Features
```
data/processed/
├── insectset459/
│   ├── features.npy (256 mel bins × time steps)
│   ├── labels.npy
│   └── processed_files.csv
├── xenocanto/
│   ├── features.npy
│   ├── labels.npy
│   └── processed_files.csv
└── sina/ (not used in combined)
    ├── features.npy
    ├── labels.npy
    └── processed_files.csv
```

### Train/Val Splits
```
data/splits/
├── combined/  ← USE THIS FOR TRAINING
│   ├── X_train.npy (20,806 samples)
│   ├── y_train.npy
│   ├── X_val.npy (8,917 samples)
│   ├── y_val.npy
│   └── combined_metadata.json
├── insectset459/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_val.npy
│   └── y_val.npy
└── xenocanto/
    ├── X_train.npy
    ├── y_train.npy
    ├── X_val.npy
    └── y_val.npy
```

---

## Regenerating the Dataset

### Full Preprocessing from Raw Data
```bash
# Reprocess all datasets and auto-combine
python scripts/preprocess_unified.py --dataset all --min-samples 30 --val-ratio 0.30
```

### Just Recombine Existing Splits
```bash
# If individual splits already exist, just recombine them
python scripts/combine_datasets.py --datasets insectset459 xenocanto --min-samples 30
```

### Custom Parameters
```bash
# More aggressive filtering (40 samples minimum)
python scripts/preprocess_unified.py --dataset all --min-samples 40

# Larger validation set (40%)
python scripts/combine_datasets.py --val-ratio 0.40

# Combine specific datasets only
python scripts/combine_datasets.py --datasets insectset459 xenocanto
```

---

## Validation Checklist

After preprocessing, verify:

✅ **Combined dataset exists:**
```bash
ls -lh data/splits/combined/
```

✅ **Check species count:**
```bash
python -c "import numpy as np; y=np.load('data/splits/combined/y_train.npy', allow_pickle=True); print(f'{len(np.unique(y))} species')"
```

✅ **Check sample distribution:**
```bash
python -c "
import numpy as np
from collections import Counter
y_train = np.load('data/splits/combined/y_train.npy', allow_pickle=True)
y_val = np.load('data/splits/combined/y_val.npy', allow_pickle=True)
y_all = np.concatenate([y_train, y_val])
counts = Counter(y_all)
print(f'Min: {min(counts.values())}, Max: {max(counts.values())}, Mean: {np.mean(list(counts.values())):.1f}')
"
```

✅ **Verify metadata:**
```bash
cat data/splits/combined/combined_metadata.json
```

---

## Known Issues & Limitations

### Data Limitations
1. **Class Imbalance:** Still exists (30 to 1,205 samples per species)
2. **Limited Rare Species:** Many interesting species excluded due to <30 samples
3. **Geographic Bias:** Mostly global datasets, some regions underrepresented
4. **Recording Quality:** Variable across datasets

### Mitigation Strategies
1. **Class weighting:** Could be added to loss function
2. **Focal loss:** Alternative to standard cross-entropy
3. **Oversampling:** Rare species could be augmented more heavily
4. **Currently using:** Strong regularization (dropout, mixup, label smoothing)

---

*This dataset represents a balance between data quality and species coverage. Further improvements require either more data collection or advanced techniques like transfer learning.*
