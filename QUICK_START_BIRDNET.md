# Quick Start: BirdNET Transfer Learning

## Goal: Reach 80% Validation Accuracy

**Current baseline:** 37% (CNN-LSTM after 12 hours training)
**Target:** 80% with BirdNET transfer learning
**Expected:** 45-60% (realistic), 80% (optimistic with ensemble/fine-tuning)

## 🔥 Recommended: Extract Locally, Train on Kaggle (FASTEST!)

```bash
# 1. Extract embeddings locally (2-4 hours, one-time)
python scripts/extract_embeddings_for_kaggle.py --dataset combined

# 2. Upload to Kaggle (5 min)
# → https://kaggle.com/datasets → Upload chirpkit-birdnet-embeddings.tar.gz

# 3. Train on Kaggle GPU (10-30 min)
# Create notebook, add dataset, run:
!python /kaggle/input/*/train_on_kaggle.py

# See KAGGLE_WORKFLOW.md for detailed instructions
```

**Total time:** ~3-5 hours (mostly unattended)
**Cost:** $0 (all free!)

---

## Alternative: Train Everything Locally

```bash
# Complete pipeline: Extract embeddings + Train classifier
python scripts/train_birdnet_transfer.py \
    --dataset combined \
    --architecture deep_mlp \
    --epochs 200 \
    --batch-size 256 \
    --patience 20
```

**Time estimate:**
- Embedding extraction: 2-4 hours (one-time, cached)
- Training: 15-30 minutes per architecture

## What Happens

### Phase 1: Embedding Extraction (2-4 hours)
1. Scans `data/raw/insectset459/` and `data/raw/xenocanto/`
2. Extracts species names from filenames
3. Creates train/val split matching your existing splits (30% val)
4. Processes each audio file through BirdNET:
   - Resample to 48kHz
   - Split into 3-second chunks
   - Extract 1024-dim embeddings
   - Average across chunks
5. Saves to `data/embeddings/combined/`

**Output files:**
```
data/embeddings/combined/
├── X_train_embeddings.npy  (20,806 × 1024) ~85MB
├── y_train.npy             (20,806,)
├── X_val_embeddings.npy    (8,917 × 1024) ~36MB
├── y_val.npy               (8,917,)
├── label_encoder.joblib
└── metadata.json
```

### Phase 2: Classifier Training (15-30 minutes)
1. Loads pre-extracted embeddings
2. Trains deep MLP classifier:
   - Input: 1024-dim BirdNET features (frozen)
   - Hidden: [512, 256, 128] with ReLU + Dropout
   - Output: 255 species
3. Uses techniques:
   - Label smoothing (0.1)
   - Weight decay (1e-4)
   - Learning rate scheduling
   - Early stopping (patience 20)
4. Saves best model when validation improves

**Output:**
```
models/birdnet_transfer/
└── best_deep_mlp_classifier.pth
```

## Architecture Options

### 1. Deep MLP (Recommended for 80% target)
```bash
--architecture deep_mlp
```
- **Layers:** 1024 → 512 → 256 → 128 → 255
- **Parameters:** ~850K
- **Training time:** 15-30 min
- **Expected accuracy:** 47-55%

### 2. MLP (Fast baseline)
```bash
--architecture mlp
```
- **Layers:** 1024 → 512 → 256 → 255
- **Parameters:** ~656K
- **Training time:** 10-20 min
- **Expected accuracy:** 45-50%

### 3. Ensemble (Maximum performance)
```bash
--architecture ensemble
```
- **Combines:** Linear + MLP + Deep MLP
- **Parameters:** ~1.8M
- **Training time:** 25-50 min
- **Expected accuracy:** 48-60%

## Running on Kaggle

For longer training with GPU:

```bash
# 1. Upload your embeddings to Kaggle (much smaller than spectrograms!)
# 2. Create notebook with:

!pip install -q torch torchvision

# Upload train_birdnet_transfer.py and run:
!python train_birdnet_transfer.py \
    --embeddings-dir /kaggle/input/insect-embeddings/combined \
    --architecture ensemble \
    --epochs 500 \
    --batch-size 512 \
    --lr 1e-3 \
    --patience 50 \
    --device cuda
```

## Incremental Workflow

### Step 1: Extract embeddings only (run once)
```bash
python scripts/train_birdnet_transfer.py \
    --dataset combined \
    --extract-only
```

### Step 2: Try different architectures (fast!)
```bash
# Linear baseline (5-10 min)
python scripts/train_birdnet_transfer.py \
    --embeddings-dir data/embeddings/combined \
    --architecture linear \
    --epochs 100

# MLP (10-20 min)
python scripts/train_birdnet_transfer.py \
    --embeddings-dir data/embeddings/combined \
    --architecture mlp \
    --epochs 200

# Deep MLP (15-30 min)
python scripts/train_birdnet_transfer.py \
    --embeddings-dir data/embeddings/combined \
    --architecture deep_mlp \
    --epochs 200

# Ensemble (25-50 min)
python scripts/train_birdnet_transfer.py \
    --embeddings-dir data/embeddings/combined \
    --architecture ensemble \
    --epochs 300
```

### Step 3: Hyperparameter tuning
```bash
# Higher dropout for regularization
python scripts/train_birdnet_transfer.py \
    --embeddings-dir data/embeddings/combined \
    --architecture deep_mlp \
    --dropout 0.5 \
    --epochs 200

# Lower learning rate for stability
python scripts/train_birdnet_transfer.py \
    --embeddings-dir data/embeddings/combined \
    --architecture deep_mlp \
    --lr 5e-4 \
    --epochs 200

# Larger batch size for faster training
python scripts/train_birdnet_transfer.py \
    --embeddings-dir data/embeddings/combined \
    --architecture deep_mlp \
    --batch-size 512 \
    --epochs 200
```

## Expected Results Path to 80%

| Approach | Expected Accuracy | Time | Status |
|----------|------------------|------|--------|
| **Phase 1: Frozen Features** | | | |
| Linear baseline | 42-45% | 10 min | ⏳ Start here |
| MLP | 45-50% | 20 min | ⏳ Quick wins |
| Deep MLP | 47-55% | 30 min | ⏳ Best single model |
| Ensemble (3 models) | 48-60% | 50 min | ⏳ Combine strengths |
| **Phase 2: Advanced Techniques** | | | |
| + Data augmentation | +2-5% | 60 min | 📋 If needed |
| + Semi-supervised learning | +3-8% | 2 hours | 📋 Use unlabeled data |
| + Fine-tune BirdNET layers | +5-10% | 4 hours | 📋 Unfreeze backbone |
| **Target** | **80%** | **~8 hours** | 🎯 Goal |

## Monitoring Training

The script prints:
```
Epoch  50/200 | Train Loss: 2.1234 Acc: 58.42% | Val Loss: 2.4567 Acc: 52.31% | Gap: 6.11%
🎉 New best! Val Acc: 52.31% (Gap: 6.11%)
```

**Key metrics:**
- **Val Acc:** Your validation accuracy (target: 80%)
- **Gap:** Train - Val accuracy (should be <10% ideally)
- **🎉 New best!:** Model improved and was saved

## Troubleshooting

### Embeddings extraction is slow
- Normal! BirdNET processes each 3-second chunk
- ~30K files × 3 seconds = 25 hours of audio
- Should take 2-4 hours on modern CPU
- Run overnight or on Kaggle

### "FileNotFoundError: data/raw/combined"
```bash
# The script expects raw audio in a combined directory
# You have insectset459 and xenocanto separately

# Option 1: Modify script to process both datasets
# Option 2: Create symbolic link
mkdir -p data/raw/combined
ln -s ../insectset459 data/raw/combined/
ln -s ../xenocanto data/raw/combined/
```

### "Out of memory" during training
```bash
# Reduce batch size
python scripts/train_birdnet_transfer.py \
    --batch-size 128 \
    --embeddings-dir data/embeddings/combined
```

### Accuracy plateaus at 50-55%
This is expected for frozen features. To reach 80%:

1. **Try ensemble** (usually +3-5%)
2. **Add data augmentation** during embedding extraction
3. **Fine-tune BirdNET layers** (requires more complex setup)
4. **Use semi-supervised learning** with unlabeled data
5. **Collect more training data** (most effective!)

### Model overfitting (large train-val gap)
```bash
# Increase dropout
python scripts/train_birdnet_transfer.py \
    --dropout 0.5 \
    --embeddings-dir data/embeddings/combined

# Add more regularization
# (Modify script to increase weight_decay from 1e-4 to 1e-3)
```

## Next Steps After Training

### 1. Evaluate on test set
```python
import torch
import numpy as np

# Load model
checkpoint = torch.load('models/birdnet_transfer/best_deep_mlp_classifier.pth')
print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

# Use for inference on new audio
# (See BIRDNET_TRANSFER_LEARNING.md for inference code)
```

### 2. Compare to CNN-LSTM baseline
```
Baseline (CNN-LSTM): 37% (12 hours training)
BirdNET Transfer: __% (30 min training)
Improvement: +__% (40x faster!)
```

### 3. If accuracy < 80%, try:
- Ensemble of multiple architectures
- Fine-tune BirdNET backbone
- Semi-supervised learning
- More training data

## Summary

```bash
# Complete pipeline (one command)
python scripts/train_birdnet_transfer.py --dataset combined

# Expected output:
# ✅ Extracted 20,806 train embeddings (2-4 hours)
# ✅ Extracted 8,917 val embeddings
# 🎯 Training deep_mlp classifier...
# Epoch 100/200 | Val Acc: 52.31%
# 🎉 New best! Val Acc: 52.31%
# ✅ Training complete! Best: 52.31%
# 📈 Improvement over baseline: +15.31%
```

**Next:** If 52% isn't enough, try ensemble or fine-tuning for 80% target!
