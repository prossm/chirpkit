# ChirpKit BirdNET Embeddings

Pre-extracted BirdNET embeddings for insect classification.

## Dataset Information

- **Total Samples:** 25,140
- **Training:** 17,598 samples
- **Validation:** 7,542 samples
- **Species:** 231 classes
- **Embedding Dimension:** 1024 (BirdNET features)
- **Created:** 2025-10-11T21:00:40.466499

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

print(f"Training data shape: {X_train.shape}")
# Output: Training data shape: (17598, 1024)
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
