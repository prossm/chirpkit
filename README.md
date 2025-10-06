# 🦗 ChirpKit: Multi-Species Insect Sound Classifier

A comprehensive Python system for identifying **255 high-quality insect species** from audio recordings using deep learning. Features a CNN-LSTM neural network trained on globally balanced datasets with strong regularization, comprehensive training pipeline, and an intuitive web interface for real-time species identification.

**Dataset Quality:** All 255 species have ≥30 training samples, ensuring reliable model performance and avoiding overfitting issues common with rare classes.

## ✨ Key Features

- **🎯 255 High-Quality Species**: Trained on carefully curated datasets with minimum 30 samples per species
- **🧠 Enhanced CNN-LSTM**: Multi-scale feature extraction with attention mechanisms and aggressive regularization
- **🎤 Real-time Recording**: Web UI supports both live audio recording and file uploads
- **📊 Smart Confidence Display**: Context-aware confidence ratings with visual star system
- **🔍 Species Browser**: Searchable modal with all 255 supported species
- **📖 Wikipedia Integration**: Automatic fetching of species info, images, and descriptions
- **⚡ Optimized Training**: Advanced regularization techniques (50% dropout, MixUp, SWA) for better generalization

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

ChirpKit supports flexible installation with platform-specific optimizations:

```bash
# Clone the repository
git clone https://github.com/patrickmetzger/chirpkit.git
cd chirpkit

# Basic installation (CPU-only, universal)
pip install .

# macOS with Apple Silicon/Intel optimization
pip install .[full]

# Linux/Windows with optional GPU support
pip install .[tensorflow-gpu,torch,viz]

# Development installation
pip install .[dev]
```

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

### Option 1: Use Pre-trained Model (Recommended)

```bash
# Launch the web interface
python simple_ui.py
```

Access the web UI at `http://localhost:7860` to:
- 🎤 Record insect sounds directly in your browser
- 📁 Upload audio files (.wav, .mp3, .m4a, .flac)
- 🔍 Browse all 255 supported species
- 📖 View species information and Wikipedia photos

### Option 2: Train Your Own Model

```bash
# Download datasets
python scripts/download_insectset459.py
python scripts/download_xenocanto.py    # Requires Xeno-canto account (see below)
python scripts/download_sina.py         # Optional - will be filtered out (<30 samples/species)

# Preprocess and combine datasets (auto-filters to 255 species with ≥30 samples)
python scripts/preprocess_unified.py --dataset all --min-samples 30 --val-ratio 0.30

# Train with strong regularization (recommended)
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

### Current Model (255 Species)
- **Target Validation Accuracy**: 50-60% (vs. 0.39% random baseline = 130-155× improvement)
- **Architecture**: Enhanced CNN-LSTM with multi-scale features, attention, and species-specific focus
- **Regularization**: 50% dropout, MixUp augmentation, label smoothing, SWA
- **Training**: ~300 epochs with early stopping (patience=50)
- **Overfitting Gap**: Target <15% (train_acc - val_acc)

### Previous Model (471 Species, Lower Quality)
- **Validation Accuracy**: 71.6% on 471 species
- **Issues**: Severe overfitting (87% train / 37% val = 50% gap) due to rare classes
- **Lesson Learned**: More species ≠ better model without sufficient data per species

### Confidence Interpretation:
- **⭐⭐⭐ Very High** (>15%): Highly reliable identification
- **⭐⭐☆ High** (8-15%): Good confidence, likely correct
- **⭐☆☆ Moderate** (3-8%): Reasonable guess, consider alternatives
- **☆☆☆ Low** (<3%): Uncertain, verify with expert

## 🖥️ Web Interface Features

### Audio Input
- **Live Recording**: Record insect sounds directly in your browser
- **File Upload**: Support for common audio formats
- **Recording Tips**: Built-in guidance for optimal audio capture

### Species Identification
- **Real-time Processing**: Get results in seconds
- **Rich Results**: Shows common name, scientific name, and confidence
- **Wikipedia Integration**: Automatic species photos and descriptions
- **Top 5 Predictions**: See alternative identifications

### Species Browser
- **Complete Catalog**: Browse all 255 high-quality species
- **Fast Search**: Real-time filtering by scientific name
- **Mobile Friendly**: Touch-optimized interface

## 🏗️ Project Structure

```
chirpkit/
├── simple_ui.py                 # Web interface for species identification
├── src/
│   ├── models/
│   │   └── simple_cnn_lstm.py   # CNN-LSTM model architecture
│   └── data/
│       ├── preprocessing.py      # Audio preprocessing utilities
│       └── augmentation.py       # Data augmentation pipeline
├── scripts/
│   ├── train_enhanced_regularized.py  # Recommended: Strong regularization training
│   ├── train_unified.py               # Alternative: Standard training
│   ├── preprocess_unified.py          # Data preprocessing (auto-combines datasets)
│   ├── combine_datasets.py            # Manually combine preprocessed datasets
│   └── download_*.py                  # Dataset download scripts
├── models/
│   └── trained/                       # Pre-trained models and metadata
│       ├── insect_classifier_enhanced_255species.pth
│       ├── insect_classifier_enhanced_255species_label_encoder.joblib
│       └── insect_classifier_enhanced_255species_info.json
└── data/                        # Dataset storage (not included in repo)
    ├── raw/                     # Original audio files
    ├── processed/               # Preprocessed features
    └── splits/                  # Train/validation/test splits
```

## 🔧 Technical Details

### Model Architecture (Enhanced Regularized)
- **Multi-Scale CNN**: 3 parallel paths (3×3, 5×5, 7×7 kernels) for different temporal scales
- **CNN Depth**: 5 convolutional blocks with batch normalization and 50% dropout
- **LSTM**: 3-layer bidirectional LSTM (256 hidden units per direction, dropout enabled)
- **Attention**: Multi-head attention (8 heads) + species-specific attention
- **Classifier**: 3-layer MLP (1024→512→255) with 50% dropout and batch normalization
- **Features**: High-resolution mel spectrograms (256 mel bins, 2.5-second segments, 22kHz)

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

## 📚 Additional Documentation

- **[DATASET_SUMMARY.md](DATASET_SUMMARY.md)**: Complete dataset statistics and quality analysis
- **[PREPROCESSING_IMPROVEMENTS.md](PREPROCESSING_IMPROVEMENTS.md)**: What changed and why
- **[SUPPORTING_RESEARCH.md](SUPPORTING_RESEARCH.md)**: Scientific justification for all techniques (57 references)

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