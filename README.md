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

### Option 1: Use Pre-trained Model (Recommended)

The repository includes a pre-trained model ready for immediate use:

```bash
# Clone the repository
git clone https://github.com/yourusername/chirpkit.git
cd chirpkit

# Install dependencies
pip install -r requirements.txt

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

## 📋 Requirements

```
torch>=1.9.0
librosa>=0.9.0
gradio>=3.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
requests>=2.25.0
joblib>=1.0.0
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