# Kaggle Training Workflow

**Perfect for:** Extract embeddings locally (slow), train on Kaggle GPU (fast)

## Why This Workflow?

✅ **Extract locally** (2-4 hours, one-time)
- Run on your machine with your audio files
- No upload limits (Kaggle has 20GB dataset limit)
- BirdNET uses CPU anyway (TensorFlow Lite)

✅ **Train on Kaggle** (10-30 minutes, unlimited)
- Free GPU (faster training)
- Try multiple architectures quickly
- Easy to share and reproduce

## Complete Workflow

### Step 1: Extract Embeddings Locally (2-4 hours, one-time)

```bash
python scripts/extract_embeddings_for_kaggle.py --dataset combined
```

**What happens:**
1. Scans `data/raw/insectset459/` and `data/raw/xenocanto/`
2. Extracts BirdNET embeddings (1024-dim)
3. Creates train/val splits (70/30)
4. Saves as numpy arrays
5. Packages everything into `chirpkit-birdnet-embeddings.tar.gz` (~120MB)

**Output:**
```
data/embeddings_kaggle/
└── chirpkit-birdnet-embeddings.tar.gz  (~120MB)
    ├── X_train_embeddings.npy (20,806 × 1024)
    ├── y_train.npy
    ├── X_val_embeddings.npy (8,917 × 1024)
    ├── y_val.npy
    ├── label_encoder.joblib
    ├── metadata.json
    ├── train_on_kaggle.py  (ready-to-run training script!)
    └── README.md
```

### Step 2: Upload to Kaggle (5 minutes)

1. Go to https://kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload `chirpkit-birdnet-embeddings.tar.gz`
4. Title: "ChirpKit BirdNET Embeddings"
5. Make it **Private** (or Public if you want)
6. Click **"Create"**

### Step 3: Train on Kaggle (10-30 minutes)

**Option A: Use provided training script**

1. Create new Kaggle Notebook
2. Add your dataset as input
3. Settings → Accelerator → **GPU T4 x2** (free!)
4. New cell:

```python
# Extract dataset
!tar -xzf /kaggle/input/chirpkit-birdnet-embeddings/chirpkit-birdnet-embeddings.tar.gz

# Run training
!python chirpkit-birdnet-embeddings/train_on_kaggle.py
```

5. Click **Run All**
6. Download `best_model.pth` when done

**Option B: Custom notebook (more control)**

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load embeddings
X_train = np.load('/kaggle/input/chirpkit-birdnet-embeddings/X_train_embeddings.npy')
y_train = np.load('/kaggle/input/chirpkit-birdnet-embeddings/y_train.npy')
X_val = np.load('/kaggle/input/chirpkit-birdnet-embeddings/X_val_embeddings.npy')
y_val = np.load('/kaggle/input/chirpkit-birdnet-embeddings/y_val.npy')

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Define model (see train_on_kaggle.py for full code)
# Train with GPU
# Download best_model.pth
```

### Step 4: Download Trained Model

1. Click **"Save Version"** in notebook
2. Go to **Output** tab
3. Download `best_model.pth`
4. Use locally for inference!

## Time & Cost Breakdown

| Task | Where | Time | Cost |
|------|-------|------|------|
| **Extract embeddings** | Local | 2-4 hours | $0 (uses CPU) |
| **Upload to Kaggle** | Browser | 5 min | $0 (free) |
| **Train classifier** | Kaggle GPU | 10-30 min | $0 (free tier) |
| **Download model** | Browser | 1 min | $0 |
| **TOTAL** | | ~3-5 hours | **$0** |

Compare to training everything on Kaggle:
- Upload 30K audio files: Not possible (20GB limit)
- Or upload spectrograms: 6.2GB (takes forever)
- Extract + train on GPU: 4-6 hours

## Advantages

| Aspect | This Workflow | All on Kaggle |
|--------|---------------|---------------|
| **Storage** | 120MB | 6.2GB (spectrograms) |
| **Upload time** | 5 min | 2-3 hours |
| **Flexibility** | Use any audio | Limited by 20GB |
| **Iteration** | Fast (10-30 min) | Slow (4-6 hours) |
| **Cost** | $0 | $0 (but time limit) |

## Re-running Training

Once embeddings are uploaded to Kaggle:

```python
# Try different architecture (3 minutes per)
!python train_on_kaggle.py  # deep_mlp (default)

# Or modify the script for ensemble, etc.
```

No need to re-extract or re-upload! Just train different models.

## Troubleshooting

### "No audio files found"

```bash
# Check your directories
ls data/raw/insectset459/Train/*.mp3 | head -5
ls data/raw/xenocanto/audio/*.mp3 | head -5

# If files are elsewhere, specify:
python scripts/extract_embeddings_for_kaggle.py \
    --raw-data-dirs data/raw/insectset459/Train data/raw/xenocanto/audio
```

### "Out of memory during extraction"

Extraction runs on CPU and shouldn't use much RAM. But if needed:
- Close other applications
- Process one dataset at a time:

```bash
# Extract insectset459 only
python scripts/extract_embeddings_for_kaggle.py \
    --dataset insectset459 \
    --raw-data-dirs data/raw/insectset459

# Then xenocanto
python scripts/extract_embeddings_for_kaggle.py \
    --dataset xenocanto \
    --raw-data-dirs data/raw/xenocanto
```

### "Kaggle notebook times out"

Free tier has 9-hour limit. Training should take 10-30 minutes, so this shouldn't happen. If it does:
- Use smaller batch size (slower but safer)
- Reduce epochs (200 → 100)
- Use simpler architecture (mlp instead of deep_mlp)

### "How do I use the trained model locally?"

```python
import torch
from src.models.birdnet_classifier import DeepMLPClassifier

# Load model
checkpoint = torch.load('best_model.pth')
model = DeepMLPClassifier(n_classes=255)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Model accuracy: {checkpoint['val_acc']:.2f}%")

# Use for inference (see BIRDNET_TRANSFER_LEARNING.md)
```

## Advanced: Try Multiple Architectures

Create multiple versions of `train_on_kaggle.py`:

**1. Ensemble (best accuracy)**
```python
# Modify train_on_kaggle.py
# Replace DeepMLPClassifier with EnsembleClassifier
# Expected: 50-60% accuracy
```

**2. Different hyperparameters**
```python
DROPOUT = 0.5  # More regularization
LR = 5e-4      # Lower learning rate
BATCH_SIZE = 256  # Smaller batches
```

**3. Longer training**
```python
EPOCHS = 500  # More epochs
PATIENCE = 50  # More patience
```

## Expected Results on Kaggle

With GPU training (10-30 minutes):

| Architecture | Expected Accuracy | Time |
|--------------|------------------|------|
| Linear | 42-45% | 5 min |
| MLP | 45-50% | 10 min |
| Deep MLP | 47-55% | 20 min |
| Ensemble | 50-60% | 30 min |

All faster than 12-hour CNN-LSTM training!

## Full Example Commands

```bash
# 1. Extract embeddings locally (one-time, 2-4 hours)
python scripts/extract_embeddings_for_kaggle.py --dataset combined

# Output: data/embeddings_kaggle/chirpkit-birdnet-embeddings.tar.gz

# 2. Upload to Kaggle (web browser, 5 min)
#    https://kaggle.com/datasets → New Dataset → Upload tar.gz

# 3. Train on Kaggle (notebook, 10-30 min)
!tar -xzf /kaggle/input/*/chirpkit-birdnet-embeddings.tar.gz
!python chirpkit-birdnet-embeddings/train_on_kaggle.py

# 4. Download best_model.pth (web browser, 1 min)
#    Output tab → Download

# 5. Use locally for inference
# (See BIRDNET_TRANSFER_LEARNING.md)
```

## Achieved Results (v6.0 - Oct 12, 2024)

### 7-Model Ensemble + Test-Time Augmentation

**Training Command:**
```python
!python train_ensemble_on_kaggle.py
```

**Final Performance:**
```
Individual Model Accuracies:
   Model 1: 77.04%
   Model 2: 77.38%
   Model 3: 77.00%
   Model 4: 76.92%
   Model 5: 76.48%
   Model 6: 77.62%
   Model 7: 77.33%

Average Individual: 77.11%

Ensemble (no TTA):  79.63%
Ensemble (with TTA): 79.73%  ← PRODUCTION MODEL

TTA Improvement: +0.09%
Total Improvement over baseline (37%): +42.73%
```

**Configuration:**
- **Architecture:** Deep MLP (1024 → 512 → 256 → 128 → 231)
- **Ensemble size:** 7 models with different random seeds
- **Test-time augmentation:** 10 rounds with 1% Gaussian noise
- **Training time:** ~7 minutes total (7 models)
- **Model size:** 2.7MB per model (19MB total)

**Deployment Options:**

| Configuration | Accuracy | Inference Time | Use Case |
|--------------|----------|----------------|----------|
| Best single model | 77.62% | ~10ms | Real-time mobile |
| 7-model ensemble (no TTA) | 79.63% | ~35ms | Production API |
| 7-model ensemble + TTA | 79.73% | ~70ms | Highest accuracy |

**Files:**
- Location: `/models/trained/chirpkit-ensemble/`
- Models: `ensemble_model_1.pth` through `ensemble_model_7.pth`
- Metadata: `ensemble_info.json`
- Total size: ~19MB (all 7 models)

---

## Summary

This workflow gives you:
- ✅ **Flexibility:** Extract from any audio source locally
- ✅ **Speed:** Fast GPU training on Kaggle (7 min for ensemble)
- ✅ **Free:** No cloud compute costs
- ✅ **Easy:** One command to extract, one to train
- ✅ **Reproducible:** Share embeddings dataset, others can train
- ✅ **Production-ready:** 79.73% accuracy achieved

**Total time:** ~3-5 hours (mostly unattended extraction)
**Total cost:** $0
**Achieved accuracy:** 79.73% (vs 37% baseline) - **+42.73% improvement**
