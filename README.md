# 🦗 ChirpKit: Multi-Species Insect Sound Classifier

A comprehensive Python system for identifying **231 insect species** from audio recordings using deep learning. Features a 7-model ensemble trained on BirdNET embeddings, achieving **79.7% accuracy** with Test-Time Augmentation.

**Current Model:** v6.0 Ensemble (October 2024)
- **Accuracy:** 79.7% on 231 species (79.6% without TTA)
- **Architecture:** 7 DeepMLP models trained on BirdNET embeddings
- **Training:** 7 minutes on Kaggle GPU
- **Species:** All 231 species have ≥30 samples for reliable training

## 🆕 v6.0 Ensemble: 79.7% Accuracy Achieved! 🎉

ChirpKit v6.0 uses a powerful ensemble approach with BirdNET transfer learning:

```bash
# Use the pre-trained ensemble (recommended)
python simple_ui.py
```

**Performance Journey:**
- v4.0 CNN-LSTM (255 species): 37% accuracy, 12 hours training
- v5.0 BirdNET Single (231 species): 77% accuracy, 2 minutes training
- v5.1 BirdNET 5-Ensemble (231 species): 79.4% accuracy, 5 minutes training
- **v6.0 BirdNET 7-Ensemble + TTA (231 species): 79.7% accuracy, 7 minutes training** ✅

**Key Features:**
- ⚡ **Extremely fast training** (7 min vs 12 hours)
- 📈 **Production-ready accuracy** (79.7%)
- 🎯 **Robust predictions** (7-model ensemble with TTA)
- 💾 **Compact models** (19MB total for all 7 models)
- 🔄 **Flexible deployment** (single model, ensemble, or ensemble+TTA)

See **[KAGGLE_WORKFLOW.md](KAGGLE_WORKFLOW.md)** for training guide and **v6.0 Ensemble** section below for usage.

## ✨ Key Features

- **🎯 231 High-Quality Species**: Trained on carefully curated datasets with minimum 30 samples per species
- **🧠 v6.0 Ensemble**: 7 DeepMLP models with Test-Time Augmentation for robust predictions
- **🚀 BirdNET Transfer Learning**: Leverages pre-trained embeddings from millions of animal sounds
- **🎤 Real-time Recording**: Web UI supports both live audio recording and file uploads
- **📊 Smart Confidence Display**: Context-aware confidence ratings with visual star system
- **🔍 Species Browser**: Searchable modal with all 231 supported species
- **📖 Wikipedia Integration**: Automatic fetching of species info, images, and descriptions
- **⚡ Fast Training**: 7 minutes on free Kaggle GPU vs 12 hours for CNN-LSTM from scratch

## 📊 Dataset Information

ChirpKit uses high-quality insect audio datasets with aggressive quality filtering:

### Current Training Dataset (255 Species, 29,723 Samples)
The current model uses carefully filtered datasets with minimum 30 samples per species:

- **InsectSet459**: 149 species retained (16,594 samples after filtering) - 111GB raw
- **Xeno-canto**: 130 species retained (13,129 samples after filtering) - 574GB raw
- **SINA**: ❌ Excluded (203 species had <30 samples each, insufficient for deep learning)
- **Combined Total**: 255 unique species, 29,723 samples
  - Train: 20,806 samples (70%)
  - Validation: 8,917 samples (30%)
- **Quality Guarantee**: Every species has ≥30 total samples (min 21 train + 9 validation)

**Why 30 samples minimum?** Research shows deep learning models require 20-50 examples per class for basic generalization. With <30 samples, models severely overfit. See `SUPPORTING_RESEARCH.md` for scientific justification.

### Optional Dataset
- **InsectSound1000**: 12 European species (165,982 samples, subsampled to 1,000 for balance) - ~99GB
  - *Note: Not used in current pre-trained models to avoid European geographical bias*
  - *Available for custom training if desired regional focus*

### Available Optional Dataset
- **InsectSound1000**: European species (not used due to geographic bias concerns)
  - Contains 1000 species but heavily biased toward European fauna
  - Can be included in custom training if regional focus is desired

### Species Distribution
- **By Sample Count:**
  - 30-50 samples: ~140 species (55%)
  - 51-100 samples: ~60 species (24%)
  - 101-200 samples: ~35 species (14%)
  - 200+ samples: ~20 species (8%)
- **Median:** 52 samples per species
- **Geographic Coverage:** Global (North America, Europe, Asia, Africa, South America)

### Storage Requirements
- **Raw datasets**: ~685GB (InsectSet459 + Xeno-canto)
- **Preprocessed features**: ~8GB (256 mel bins, high resolution)
- **Combined splits**: ~6.1GB (ready for training)

## 🚀 Quick Start

### Installation

ChirpKit includes all necessary model files via git LFS:

```bash
# Clone the repository (includes models via git LFS)
git clone https://github.com/patrickmetzger/chirpkit.git
cd chirpkit

# Install dependencies
pip install -e .

# Platform-specific optimizations:
# macOS with Apple Silicon/Intel
pip install -e .[full]

# Linux/Windows with optional GPU support
pip install -e .[tensorflow-gpu,torch,viz]

# Development installation
pip install -e .[dev]
```

**What's included:**
- ✅ v6.0 Ensemble models (7 × 2.7MB = 19MB total)
- ✅ BirdNET embedding extractor (25MB model file)
- ✅ Label encoder for 231 species
- ✅ All necessary Python code

**No additional downloads required!** The pre-trained models work out of the box.

**Platform-Specific Recommendations:**
- **macOS**: `pip install .[full]` (includes tensorflow-macos with Metal GPU support)
- **Linux**: `pip install .[tensorflow-gpu,torch]` (with CUDA support)
- **Windows**: `pip install .[tensorflow,torch]`

### Verify Installation

```bash
# Check installation health
chirpkit-doctor

# Get platform-specific installation guide
chirpkit install-guide

# Auto-fix common issues
chirpkit-fix
```

### Option 1: Use v6.0 Ensemble Model (Recommended)

```bash
# Launch the web interface with v6.0 ensemble
python simple_ui.py
```

The UI will automatically load the v6.0 ensemble from `models/trained/chirpkit-ensemble/`.

Access the web UI at `http://localhost:7860` to:
- 🎤 Record insect sounds directly in your browser
- 📁 Upload audio files (.wav, .mp3, .m4a, .flac)
- 🔍 Browse all 231 supported species
- 📖 View species information and Wikipedia photos
- 🎯 Get predictions from 7-model ensemble with 79.7% accuracy

### Option 2: Train v6.0 Ensemble (Fast & Accurate)

Train your own v6.0 ensemble on Kaggle GPU (free):

```bash
# Step 1: Extract BirdNET embeddings locally (one-time, 2-4 hours)
python scripts/extract_embeddings_final.py \
    --insectset459-dir data/raw/insectset459/Train \
    --xenocanto-dir data/raw/xenocanto/audio \
    --min-samples 30 \
    --output data/embeddings/combined

# Step 2: Create Kaggle package
python -c "from scripts.extract_embeddings_for_kaggle import create_kaggle_package; \
    create_kaggle_package('data/embeddings/combined', 'data/embeddings_kaggle')"

# Step 3: Upload to Kaggle (via web interface)
# Go to https://kaggle.com/datasets → New Dataset
# Upload: data/embeddings_kaggle/chirpkit-birdnet-embeddings.tar.gz

# Step 4: Train on Kaggle GPU (7 minutes, free!)
# Create new notebook, add your dataset, enable GPU, run:
# !tar -xzf /kaggle/input/chirpkit-birdnet-embeddings/chirpkit-birdnet-embeddings.tar.gz
# !python chirpkit-birdnet-embeddings/train_ensemble_on_kaggle.py

# Step 5: Download trained models
# Download the 7 .pth files to models/trained/chirpkit-ensemble/
```

**Expected results:**
- Individual models: 76-77% accuracy
- Ensemble: 79.6% accuracy
- Ensemble + TTA: 79.7% accuracy

See [KAGGLE_WORKFLOW.md](KAGGLE_WORKFLOW.md) for detailed step-by-step instructions.

### Option 3: Train CNN-LSTM from Scratch (Legacy)

Train the older CNN-LSTM model (not recommended, much slower):

```bash
# Download datasets
python scripts/download_insectset459.py
python scripts/download_xenocanto.py    # Requires Xeno-canto account (see below)

# Preprocess and combine datasets
python scripts/preprocess_unified.py --dataset all --min-samples 30 --val-ratio 0.30

# Train with strong regularization (12+ hours)
python scripts/train_enhanced_regularized.py --dataset combined --epochs 500
```

### Xeno-canto Dataset Setup

The Xeno-canto dataset requires a free account and API token:

1. **Create Account**: Register at https://xeno-canto.org/auth/register
2. **Verify Email**: Check your email and verify your account
3. **Get API Token**: Once verified, you'll receive an API token for downloads
4. **Update Script**: Add your API token to `scripts/download_xenocanto.py`

Note: Download speeds will be limited based on the Xeno-canto servers, so you should plan for the full dataset to take several days to download. The files are of variable sizes, so some take longer than others. Currently, the code is set up to download only files that include an insect name in the filename (some that are not included have simply an ID with no name, or are generic "soundscapes").

**Species Name Mapping**: After downloading Xeno-canto, run the species mapping script to standardize common names to scientific names:
```bash
# Map Xeno-canto common names to scientific names (required for preprocessing)
python scripts/map_xenocanto_names.py
```
This creates `data/xenocanto_species_mapping.json` which enables cross-dataset training with standardized species names.

**Resume Downloads**: If downloads are interrupted, you can resume from where you left off:
```bash
# Check how many files are already downloaded
find data/raw/xenocanto/audio -name "*.mp3" | wc -l

# Resume from specific page (each page = 100 files)
# If you have 6,700 files, start from page 68: (6700/100 = 67, so start page 68)
python scripts/download_xenocanto.py --start-page 68
```

## 🎯 Model Performance

### v6.0 Ensemble (231 Species) - **Current Production Model** ✅
- **Validation Accuracy**: 79.7% with TTA, 79.6% without TTA
- **Architecture**: 7 DeepMLP models (1024 → 512 → 256 → 128 → 231)
- **Training**: BirdNET embeddings + ensemble + test-time augmentation
- **Training Time**: 7 minutes on Kaggle GPU P100 (free!)
- **Inference Speed**:
  - Single model: ~10ms (77% accuracy)
  - 7-model ensemble: ~35ms (79.6% accuracy)
  - Ensemble + TTA: ~70ms (79.7% accuracy)
- **Model Size**: 2.7MB per model, 19MB total
- **Random Baseline**: 0.43% (1/231)
- **Improvement**: 185× better than random

### Performance Evolution
| Version | Species | Accuracy | Training Time | Method |
|---------|---------|----------|---------------|--------|
| v4.0 | 255 | 37% | 12 hours | CNN-LSTM from scratch |
| v5.0 | 231 | 77.0% | 2 minutes | BirdNET single model |
| v5.1 | 231 | 79.4% | 5 minutes | BirdNET 5-model ensemble |
| **v6.0** | **231** | **79.7%** | **7 minutes** | **BirdNET 7-model ensemble + TTA** |

### Deployment Options (v6.0)
| Mode | Accuracy | Inference Time | Use Case |
|------|----------|----------------|----------|
| Single | 77.6% | ~10ms | Real-time mobile apps |
| Ensemble | 79.6% | ~35ms | Production API |
| Ensemble + TTA | 79.7% | ~70ms | Maximum accuracy |

### Confidence Interpretation (231 species):
- **⭐⭐⭐ Very High** (>10%): Highly reliable identification
- **⭐⭐☆ High** (5-10%): Good confidence, likely correct
- **⭐☆☆ Moderate** (2-5%): Reasonable guess, consider alternatives
- **☆☆☆ Low** (<2%): Uncertain, verify with expert

## 🖥️ Web Interface Features

### Audio Input
- **Live Recording**: Record insect sounds directly in your browser
- **File Upload**: Support for common audio formats
- **Recording Tips**: Built-in guidance for optimal audio capture

### Species Identification
- **v6.0 Ensemble**: Uses 7-model ensemble with 79.7% accuracy
- **Real-time Processing**: Get results in ~1 second
- **Rich Results**: Shows common name, scientific name, confidence, and model info
- **Wikipedia Integration**: Automatic species photos and descriptions
- **Top 5 Predictions**: See alternative identifications with confidence scores

### Species Browser
- **Complete Catalog**: Browse all 231 high-quality species
- **Fast Search**: Real-time filtering by scientific name
- **Mobile Friendly**: Touch-optimized interface

## 🏗️ Project Structure

```
chirpkit/
├── simple_ui.py                       # Web interface (v6.0 ensemble)
├── src/
│   ├── models/
│   │   ├── chirpkit_ensemble.py       # v6.0 Ensemble classifier
│   │   ├── birdnet_classifier.py      # DeepMLP architectures
│   │   └── simple_cnn_lstm.py         # Legacy CNN-LSTM
│   ├── transfer_learning/
│   │   └── birdnet_embeddings.py      # BirdNET embedding extractor
│   └── data/
│       ├── preprocessing.py           # Audio preprocessing utilities
│       └── augmentation.py            # Data augmentation pipeline
├── scripts/
│   ├── extract_embeddings_final.py    # Extract BirdNET embeddings
│   ├── extract_embeddings_for_kaggle.py  # Package for Kaggle upload
│   └── download_*.py                  # Dataset download scripts
├── models/
│   └── trained/
│       └── chirpkit-ensemble/         # v6.0 Production model (231 species)
│           ├── ensemble_model_1.pth through ensemble_model_7.pth
│           └── ensemble_info.json
├── data/
│   ├── embeddings_kaggle/            # Kaggle training packages
│   │   └── chirpkit-birdnet-embeddings/
│   │       ├── train_ensemble_on_kaggle.py
│   │       ├── X_train_embeddings.npy
│   │       └── X_val_embeddings.npy
│   └── raw/                          # Original audio files
│       ├── insectset459/
│       └── xenocanto/
└── BirdNET-Analyzer/                 # BirdNET submodule (for embeddings)
```

## 🔧 Technical Details

### v6.0 Ensemble Architecture (Current)
- **Base Model**: DeepMLP classifier trained on BirdNET embeddings
- **Architecture per model**: 1024 (embedding) → 512 → 256 → 128 → 231 (species)
- **Ensemble size**: 7 models with different random seeds [42, 123, 456, 789, 2024, 3141, 5678]
- **Dropout**: 0.4 throughout
- **Test-Time Augmentation**: 10 rounds with 1% Gaussian noise
- **Features**: BirdNET 1024-dimensional embeddings (pre-trained on millions of bird/animal sounds)
- **Training**:
  - Optimizer: AdamW with weight decay 1e-4
  - Loss: CrossEntropyLoss with label smoothing 0.1
  - Learning rate: 1e-3 with ReduceLROnPlateau
  - Early stopping: patience 50, max epochs 300
- **Inference Modes**:
  - Single model: Single forward pass (~10ms)
  - Ensemble: Average 7 model predictions (~35ms)
  - Ensemble + TTA: Average across 7 models × 10 TTA rounds (~70ms)

### Legacy CNN-LSTM Architecture (v4.0 and earlier)
- **Multi-Scale CNN**: 3 parallel paths (3×3, 5×5, 7×7 kernels) for different temporal scales
- **CNN Depth**: 5 convolutional blocks with batch normalization and 50% dropout
- **LSTM**: 3-layer bidirectional LSTM (256 hidden units per direction, dropout enabled)
- **Attention**: Multi-head attention (8 heads) + species-specific attention
- **Classifier**: 3-layer MLP (1024→512→231) with 50% dropout and batch normalization
- **Features**: High-resolution mel spectrograms (256 mel bins, 2.5-second segments, 22kHz)
- **Note**: Slower training (12 hours) and lower accuracy (37%) - use v6.0 instead

### Training Configuration (Enhanced Regularized)
- **Optimizer**: AdamW with strong weight decay (2e-4) and differential learning rates
- **Learning Rate**: Cosine annealing with warm restarts (T_0=15, T_mult=2)
- **Batch Size**: 16 with gradient accumulation (4 steps = effective batch 64)
- **Early Stopping**: Patience of 50 epochs
- **Regularization**:
  - 50% dropout throughout
  - MixUp augmentation (α=0.3, 60% probability)
  - Label smoothing (0.15)
  - Stochastic Weight Averaging (starts epoch 150)
  - Aggressive data augmentation (80% probability)

### Audio Processing
- **Sample Rate**: 22,050 Hz (captures insect sounds up to 11kHz)
- **Segment Length**: 2.5 seconds (padded/cropped as needed)
- **FFT Size**: 4096 (5.4 Hz frequency resolution)
- **Hop Length**: 256 (11.6ms temporal resolution)
- **Features**: 256-bin mel spectrograms (2× standard resolution)
- **MFCCs**: 40 coefficients with deltas and delta-deltas
- **Normalization**: Log-scale power spectrograms

## 🔧 Troubleshooting

### Common Installation Issues

#### TensorFlow Issues

**Problem**: `AttributeError: module 'tensorflow' has no attribute '__version__'`
```bash
# Corrupted TensorFlow installation
pip uninstall tensorflow tensorflow-macos keras -y
pip cache purge
pip install tensorflow-macos  # macOS
# OR
pip install tensorflow        # Linux/Windows
```

**Problem**: Dependency solver failures, version conflicts
```bash
# Don't mix conda and pip for ML packages
# Use virtual environments with pip exclusively:
python -m venv chirpkit_env
source chirpkit_env/bin/activate  # Linux/macOS
chirpkit_env\Scripts\activate     # Windows
pip install chirpkit[full]
```

#### Platform-Specific Solutions

**macOS Users:**
- ✅ Use `tensorflow-macos` (includes Metal GPU support)
- ✅ Don't install `tensorflow-metal` separately (built-in for TF 2.16+)
- ✅ CPU-only operation is normal and sufficient for most use cases

**Linux Users:**
- ✅ Use standard `tensorflow` package  
- ✅ For GPU: Ensure CUDA drivers installed first
- ✅ Check GPU availability: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

**Windows Users:**
- ✅ Use standard `tensorflow` package
- ✅ For GPU: Install CUDA toolkit and cuDNN
- ✅ Consider using WSL2 for better compatibility

#### NumPy Version Conflicts

**Problem**: NumPy 2.x compatibility issues
```bash
# Downgrade to compatible version
pip install "numpy>=1.21.0,<2.0.0"
```

### Environment Detection

ChirpKit automatically detects your environment and suggests optimal installation:

```bash
chirpkit install-guide
```

Example output:
```
ChirpKit Installation Recommendations
====================================
Platform: Darwin arm64
Python: 3.11

Recommended Installation:
  pip install chirpkit[tensorflow-macos]

Notes:
  • Apple Silicon detected - using tensorflow-macos
  • GPU acceleration available via Metal Performance Shaders
  • Consider installing with: pip install chirpkit[full]
```

### Diagnostic Tools

```bash
# Comprehensive health check
chirpkit-doctor

# Auto-fix critical issues
chirpkit-fix

# Manual dependency check
python -c "import chirpkit; chirpkit.DependencyManager.validate_installation()"
```

### Virtual Environment Best Practices

**Recommended Setup:**
```bash
# Create isolated environment
python -m venv chirpkit_env
source chirpkit_env/bin/activate

# Install chirpkit with appropriate extras
pip install chirpkit[full]  # Complete installation

# Verify installation
chirpkit-doctor
```

**Avoid These Patterns:**
```bash
# ❌ Don't mix package managers
conda install tensorflow-deps
pip install chirpkit

# ❌ Don't use system Python
sudo pip install chirpkit

# ❌ Don't ignore version constraints  
pip install tensorflow==2.6.0 chirpkit  # May conflict
```

## 📋 Requirements

ChirpKit uses flexible dependency management with platform-specific optimizations:

**Core Dependencies:**
```
numpy>=1.21.0,<2.0.0
librosa>=0.9.0
scikit-learn>=1.0.0
pandas>=1.3.0
soundfile>=0.10.0
```

**Backend Options (choose one):**
```bash
# TensorFlow (recommended)
pip install chirpkit[tensorflow-macos]  # macOS
pip install chirpkit[tensorflow]        # Linux/Windows
pip install chirpkit[tensorflow-gpu]    # With CUDA

# PyTorch (optional)
pip install chirpkit[torch]

# Complete installation
pip install chirpkit[full]
```

## 🎨 Usage Examples

### Command Line Training
```bash
# Recommended: Train with strong regularization (best for 255 species)
python scripts/train_enhanced_regularized.py --dataset combined --epochs 500 --patience 50

# Alternative: Standard training (may overfit)
python scripts/train_unified.py --dataset combined --epochs 500 --patience 50

# Custom min_samples threshold (more/less species)
python scripts/preprocess_unified.py --dataset all --min-samples 40  # Fewer species, higher quality
python scripts/preprocess_unified.py --dataset all --min-samples 20  # More species, lower quality

# Train on single dataset only
python scripts/train_enhanced_regularized.py --dataset xenocanto --epochs 500
```

### Python API (Advanced)
```python
from src.models.enhanced_cnn_lstm_regularized import RegularizedEnhancedCNNLSTMClassifier
import torch
import joblib

# Load pre-trained model (255 species)
model = RegularizedEnhancedCNNLSTMClassifier(n_classes=255, dropout=0.5)
model.load_state_dict(torch.load('models/trained/insect_classifier_enhanced_255species.pth'))
label_encoder = joblib.load('models/trained/insect_classifier_enhanced_255species_label_encoder.joblib')

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(audio_tensor)
    top_k_probs, top_k_indices = torch.topk(torch.softmax(predictions[0], dim=0), k=5)

# Get species names
for prob, idx in zip(top_k_probs, top_k_indices):
    species = label_encoder.inverse_transform([idx.item()])[0]
    print(f"{species}: {prob.item()*100:.2f}% confidence")
```

## 📈 Performance Benchmarks

### Current Model (255 Species, Quality-Filtered)
| Metric | Value |
|--------|-------|
| **Species Coverage** | 255 unique species (all with ≥30 samples) |
| **Training Samples** | 20,806 audio recordings |
| **Validation Samples** | 8,917 audio recordings |
| **Target Val Accuracy** | 50-60% |
| **Overfitting Gap** | Target <15% (was 50% with rare classes) |
| **Random Baseline** | 0.39% (1/255) |
| **Expected Improvement** | 130-155× better than random |
| **Inference Speed** | ~0.5-1.0 seconds per sample (larger model) |
| **Model Size** | ~25MB (.pth file, enhanced architecture) |

### Comparison to Previous Model
| Metric | Previous (471 species) | Current (255 species) |
|--------|------------------------|----------------------|
| Species Count | 471 | 255 |
| Min Samples/Species | 1 | 30 |
| Training Accuracy | 87% | Target 60-70% |
| Validation Accuracy | 37% | Target 50-60% |
| Overfitting Gap | 50% ❌ | Target <15% ✅ |
| Data Quality | Mixed | High |

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional insect species datasets
- Model architecture optimizations  
- Web interface enhancements
- Mobile app development
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📦 Repository Contents

### Included in Git (via Git LFS)
- ✅ **Production Models**: `models/trained/chirpkit-ensemble/` (v6.0, 7 models, 19MB total)
- ✅ **Model Archive**: `models/trained/archive/` (historical models and documentation)
- ✅ **BirdNET Model**: `BirdNET-Analyzer/.../BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite` (25MB)
- ✅ **BirdNET Core**: Essential Python modules for embedding extraction
- ✅ **Label Encoders**: Species mappings for 231 species
- ✅ **Kaggle Package**: Ready-to-upload training package in `data/embeddings_kaggle/`

### Excluded from Git (Too Large)
- ❌ **Raw Audio**: `data/raw/` (685GB - download separately if training)
- ❌ **Embeddings**: `data/embeddings/combined/` (regeneratable from raw audio)
- ❌ **Preprocessed Data**: `data/processed/` and `data/splits/` (regeneratable)
- ❌ **Training Checkpoints**: Temporary files from training runs

### Total Repository Size
- **With LFS**: ~50MB (all models included)
- **Without LFS pointers**: ~5MB (just code)

## 📚 Additional Documentation

- **[KAGGLE_WORKFLOW.md](KAGGLE_WORKFLOW.md)**: Complete guide to training v6.0 ensemble
- **[DATASET_SUMMARY.md](DATASET_SUMMARY.md)**: Complete dataset statistics and quality analysis
- **[MODEL_HISTORY.md](models/trained/archive/MODEL_HISTORY.md)**: Evolution from v1.0 to v6.0
- **[SUPPORTING_RESEARCH.md](SUPPORTING_RESEARCH.md)**: Scientific justification with 76 references

## 🙏 Acknowledgments

- **InsectSet459 Dataset**: 16,594 samples (149 species retained) - Global coverage
- **Xeno-canto Dataset**: 13,129 samples (130 species retained) - Community contributions
- **SINA Dataset**: 265 samples (excluded due to limited samples per species)
- **Research Papers**: 57 peer-reviewed papers supporting our techniques (see SUPPORTING_RESEARCH.md)
- **Wikipedia API**: Species information and images
- **Gradio**: Web interface framework
- **PyTorch**: Deep learning framework

---

**Ready to identify insects from their sounds!** 🌟

Launch the web interface with `python simple_ui.py` and start classifying!