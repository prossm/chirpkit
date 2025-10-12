# BirdNET Transfer Learning for Insect Classification

## Overview

This document describes the BirdNET transfer learning implementation for boosting insect classification accuracy from 37% to an estimated 45-55%.

## Why Transfer Learning?

**Current Situation:**
- 37% validation accuracy on 255 species
- ~81 samples per species (limited data)
- 12+ hour training runs with plateaus
- Overfitting gap reduced to 0.5-1.7% (excellent!)

**The Problem:** Not enough training data per species to learn rich audio features from scratch.

**The Solution:** Use BirdNET's pre-trained feature extractor, which learned from millions of bird/animal sounds.

## Architecture

```
Input Audio (any format/sample rate)
    ↓
Resample to 48kHz (BirdNET requirement)
    ↓
Split into 3-second chunks
    ↓
BirdNET Feature Extractor (FROZEN)
    ↓
1024-dim embeddings
    ↓
Insect Classifier Head (TRAINABLE)
    - Linear (fastest)
    - MLP (recommended)
    - Deep MLP (best accuracy)
    - Attention (experimental)
    - Ensemble (maximum performance)
    ↓
255 species predictions
```

## Expected Performance

| Metric | Current (CNN-LSTM) | Transfer Learning (Est.) | Improvement |
|--------|-------------------|-------------------------|-------------|
| Val Accuracy | 37.0% | 45-55% | +8-18% |
| Training Time | 12+ hours | 10-30 minutes | **40x faster** |
| Overfitting Gap | 0.5-1.7% | 1-3% | Similar |
| Model Size | 50-100MB | 5-10MB | 5-10x smaller |
| Inference Speed | Slower | Faster | Better |

## Implementation Steps

### 1. Install BirdNET-Analyzer ✅

```bash
# Already completed
git clone https://github.com/kahst/BirdNET-Analyzer.git
cd BirdNET-Analyzer
pip install -e .
```

### 2. Extract BirdNET Embeddings

```bash
# Extract embeddings from raw audio files
python scripts/extract_birdnet_embeddings.py \
    --dataset combined \
    --output data/embeddings \
    --val-ratio 0.3 \
    --aggregate mean
```

**What this does:**
- Loads all raw audio files from `data/raw/combined/`
- Resamples to 48kHz
- Splits into 3-second chunks
- Extracts 1024-dim embeddings using BirdNET
- Averages embeddings for multi-chunk audio
- Saves embeddings as numpy arrays

**Output:**
- `data/embeddings/combined/X_train_embeddings.npy` (n_samples, 1024)
- `data/embeddings/combined/y_train.npy` (n_samples,)
- `data/embeddings/combined/X_val_embeddings.npy`
- `data/embeddings/combined/y_val.npy`
- `label_encoder.joblib`
- `metadata.json`

**Time estimate:** 2-4 hours for 29,723 audio files

### 3. Train Classifier on Embeddings

```bash
# Train MLP classifier (recommended)
python scripts/train_birdnet_classifier.py \
    --embeddings-dir data/embeddings/combined \
    --architecture mlp \
    --hidden-dim 512 \
    --dropout 0.4 \
    --epochs 200 \
    --lr 1e-3 \
    --batch-size 256 \
    --patience 20
```

**What this does:**
- Loads pre-extracted embeddings
- Trains lightweight classifier on frozen features
- Uses early stopping
- Saves best model

**Time estimate:** 10-30 minutes

### 4. (Optional) Fine-Tune BirdNET Backbone

**NOTE:** This requires converting BirdNET TFLite to PyTorch, which is non-trivial.

For now, we recommend training on frozen embeddings only.

## File Structure

```
chirpkit/
├── BirdNET-Analyzer/              # Cloned repo (gitignored)
├── data/
│   ├── raw/                       # Original audio files
│   │   └── combined/
│   │       ├── InsectSound1000/
│   │       ├── InsectSet459/
│   │       └── ...
│   ├── embeddings/                # BirdNET embeddings
│   │   └── combined/
│   │       ├── X_train_embeddings.npy
│   │       ├── y_train.npy
│   │       ├── X_val_embeddings.npy
│   │       ├── y_val.npy
│   │       ├── label_encoder.joblib
│   │       └── metadata.json
│   └── splits/                    # Old spectrogram splits
└── src/
    ├── models/
    │   └── birdnet_classifier.py  # Classifier heads
    └── transfer_learning/
        └── birdnet_embeddings.py  # Embedding extraction

scripts/
├── extract_birdnet_embeddings.py  # Step 2
└── train_birdnet_classifier.py    # Step 3 (TODO)
```

## Classifier Architectures

### 1. Linear Classifier
- **Parameters:** ~262K
- **Training time:** 5-10 minutes
- **Expected accuracy:** 42-45%
- **Use case:** Quick baseline

### 2. MLP Classifier (Recommended)
- **Parameters:** ~656K
- **Training time:** 10-20 minutes
- **Expected accuracy:** 45-50%
- **Use case:** Best balance of speed/accuracy

### 3. Deep MLP Classifier
- **Parameters:** ~850K
- **Training time:** 15-30 minutes
- **Expected accuracy:** 47-52%
- **Use case:** Maximum single-model accuracy

### 4. Attention Classifier
- **Parameters:** ~1.3M
- **Training time:** 20-40 minutes
- **Expected accuracy:** 46-51%
- **Use case:** Experimental, may help with feature selection

### 5. Ensemble Classifier
- **Parameters:** ~1.8M
- **Training time:** 25-50 minutes
- **Expected accuracy:** 48-55%
- **Use case:** Maximum performance

## Advantages Over CNN-LSTM

1. **Faster Training:** 10-30 min vs 12+ hours
2. **Better Features:** BirdNET learned from millions of samples
3. **Smaller Models:** 5-10MB vs 50-100MB
4. **Easier to Experiment:** Quick iteration on classifier architectures
5. **Transfer Learning:** Leverages knowledge from related audio domain

## Disadvantages

1. **Two-Stage Process:** Must extract embeddings first
2. **Fixed Features:** Can't adapt BirdNET backbone (without fine-tuning)
3. **Dependency:** Requires TensorFlow for embedding extraction
4. **Storage:** Need to store embeddings (~120MB for 30K samples)

## Expected Results

Based on transfer learning research and your current performance:

- **Conservative estimate:** 42-47% (+5-10% over baseline)
- **Realistic estimate:** 45-50% (+8-13% over baseline)
- **Optimistic estimate:** 48-55% (+11-18% over baseline)

The improvement comes from:
- BirdNET's rich pre-trained features (trained on millions of samples)
- Better generalization to new species
- Reduced overfitting (fewer parameters to train)

## Limitations

Transfer learning won't solve fundamental data limitations:
- Still only ~81 samples per species
- 255 species is a lot for this amount of data
- Some species may have very similar calls

**To reach 60%+:** Would need more data or semi-supervised learning approaches.

## Next Steps

1. ✅ Clone and install BirdNET
2. ✅ Implement embedding extraction module
3. ✅ Implement classifier architectures
4. ⏳ Extract embeddings from all audio (2-4 hours)
5. ⏳ Train classifier (10-30 minutes)
6. ⏳ Evaluate on validation set
7. ⏳ Compare to CNN-LSTM baseline (37%)
8. ⏳ If successful, run on Kaggle with longer audio

## Troubleshooting

### Audio Files Not Found
- Ensure `data/raw/combined/` contains original audio files
- Check that species directories are named correctly

### TensorFlow/BirdNET Errors
- Make sure BirdNET-Analyzer is installed: `pip install -e BirdNET-Analyzer`
- Check TensorFlow version: `pip list | grep tensorflow`

### Out of Memory
- Reduce batch size in training
- Extract embeddings in smaller batches
- Use memory-mapped numpy arrays

### Poor Accuracy
- Try different classifier architectures
- Tune hyperparameters (dropout, learning rate)
- Check if embeddings are being extracted correctly
- Ensure train/val splits match original data

## References

- BirdNET: https://github.com/kahst/BirdNET-Analyzer
- Transfer Learning for Audio: https://arxiv.org/abs/1912.06670
- Few-Shot Audio Classification: https://arxiv.org/abs/2007.15484
