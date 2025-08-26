# 🦗 ChirpKit: Multi-Species Insect Sound Classifier

A comprehensive Python system for identifying **471 insect species** from audio recordings using deep learning. Features a CNN-LSTM neural network with attention mechanisms, comprehensive training pipeline, and an intuitive web interface for real-time species identification.

## ✨ Key Features

- **🎯 471 Species Classification**: Trained on combined datasets totaling 176,532 audio samples
- **🧠 Advanced CNN-LSTM Architecture**: Hybrid model with multi-head attention for temporal audio analysis  
- **🎤 Real-time Recording**: Web UI supports both live audio recording and file uploads
- **📊 Smart Confidence Display**: Context-aware confidence ratings with visual star system
- **🔍 Species Browser**: Searchable modal with all 471 supported species
- **📖 Wikipedia Integration**: Automatic fetching of species info, images, and descriptions
- **⚡ Fast Training**: Optimized pipeline with resume capability and adaptive learning rates

## 📊 Dataset Information

ChirpKit is trained on two comprehensive insect audio datasets:

- **InsectSound1000**: 165,982 samples across 12 common species
- **InsectSet459**: 10,550 samples across 459 diverse species  
- **Combined Total**: 176,532 audio samples representing 471 unique species

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
- 🔍 Browse all 471 supported species
- 📖 View species information and Wikipedia photos

### Option 2: Train Your Own Model

```bash
# Download and preprocess datasets
python scripts/download_insectsound1000.py
python scripts/download_insectset459.py
python scripts/preprocess_unified.py --dataset both

# Train the unified model on both datasets
python scripts/train_unified.py --dataset combined
```

## 🎯 Model Performance

- **Validation Accuracy**: 71.6% on 471 species (358x better than random)
- **Architecture**: CNN-LSTM with bidirectional processing and multi-head attention
- **Training Time**: ~200 epochs with early stopping and adaptive learning rate
- **Confidence Calibration**: Context-aware confidence scoring with visual ratings

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
- **Complete Catalog**: Browse all 471 supported species
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
│   ├── train_unified.py         # Unified training for both datasets
│   ├── preprocess_unified.py    # Unified data preprocessing
│   ├── download_*.py            # Dataset download scripts
│   └── preprocess_data.py       # Legacy preprocessing (single dataset)
├── models/
│   └── trained/                 # Pre-trained models and metadata
│       ├── insect_classifier_471species.pth
│       ├── insect_classifier_471species_label_encoder.joblib
│       └── insect_classifier_471species_info.json
└── data/                        # Dataset storage (not included in repo)
    ├── raw/                     # Original audio files
    ├── processed/               # Preprocessed features
    └── splits/                  # Train/validation/test splits
```

## 🔧 Technical Details

### Model Architecture
- **CNN Layers**: 4 convolutional blocks (32→64→128→256 channels)
- **LSTM**: Bidirectional LSTM with 256 hidden units per direction
- **Attention**: Multi-head attention mechanism (8 heads, 512 dimensions)
- **Classifier**: 3-layer MLP with dropout regularization
- **Features**: Mel spectrograms (128 mel bins, 2.5-second audio segments)

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Adaptive scheduling with ReduceLROnPlateau
- **Batch Size**: 32 samples
- **Early Stopping**: Patience of 15 epochs
- **Data Augmentation**: Optional audio augmentation pipeline

### Audio Processing
- **Sample Rate**: 16 kHz
- **Segment Length**: 2.5 seconds (padded/cropped as needed)
- **Features**: 128-bin mel spectrograms
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
# Train on combined datasets with custom parameters
python scripts/train_unified.py --dataset combined --epochs 200 --lr 1e-4

# Train with data augmentation
python scripts/train_unified.py --dataset combined --epochs 150
```

### Python API (Advanced)
```python
from src.models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
import torch
import joblib

# Load pre-trained model
model = SimpleCNNLSTMInsectClassifier(n_classes=471)
model.load_state_dict(torch.load('models/trained/insect_classifier_471species.pth'))
label_encoder = joblib.load('models/trained/insect_classifier_471species_label_encoder.joblib')

# Make predictions
predictions = model(audio_tensor)
species = label_encoder.inverse_transform([torch.argmax(predictions).item()])[0]
```

## 📈 Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Species Coverage** | 471 unique species |
| **Training Samples** | 176,532 audio recordings |
| **Validation Accuracy** | 71.6% |
| **Inference Speed** | ~0.5 seconds per sample |
| **Model Size** | ~17MB (.pth file) |
| **vs Random Baseline** | 358x improvement |

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional insect species datasets
- Model architecture optimizations  
- Web interface enhancements
- Mobile app development
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **InsectSound1000 Dataset**: 165,982 samples across 12 species
- **InsectSet459 Dataset**: 10,550 samples across 459 species  
- **Wikipedia API**: Species information and images
- **Gradio**: Web interface framework
- **PyTorch**: Deep learning framework

---

**Ready to identify insects from their sounds!** 🌟

Launch the web interface with `python simple_ui.py` and start classifying!