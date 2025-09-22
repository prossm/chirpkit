# Insect Sound Classification System - Project Setup Guide

## Project Overview

This document provides comprehensive instructions for creating a dedicated insect sound classification system using the InsectSound1000 dataset, designed to integrate with the SoundCurious MoE (Mixture of Experts) architecture as a specialized expert classifier.

## 🎯 Project Goals

1. **Train a high-accuracy insect sound classifier** using state-of-the-art deep learning techniques
2. **Support 12+ insect species** from the InsectSound1000 dataset (169,000+ samples)
3. **Create a production-ready API** compatible with the SoundCurious MoE system
4. **Achieve >85% accuracy** on clean insect sounds (based on research benchmarks)
5. **Enable real-time inference** with optimized model deployment

## 📊 Dataset Information

### InsectSound1000 Dataset
- **Size**: 169,000+ labeled sound samples
- **Species**: 12 insect types
- **Format**: 4-channel WAV files, 2500ms length, 16kHz sample rate, 32-bit resolution
- **Source**: High-quality anechoic chamber recordings
- **Access**: Available via research institutions or Kaggle

### Expected Insect Categories
Based on research literature, likely includes:
- Crickets (various species)
- Cicadas
- Grasshoppers
- Bees/Wasps
- Flies
- Mosquitoes
- Beetles
- Moths
- Other orthoptera and hemiptera

## 🏗️ Technical Architecture

### Recommended Model Architecture
Based on InsectSound1000 research papers:

1. **Hybrid CNN-LSTM Model** (Primary recommendation)
   - **CNN layers**: Extract spatial features from spectrograms
   - **LSTM layers**: Capture temporal patterns and periodicity
   - **Attention mechanism**: Focus on important frequency bands
   - **Target accuracy**: 94.5% (from research)

2. **LEAF-based Model** (Advanced option)
   - **Learnable Audio Frontend**: Adaptive feature extraction
   - **Better than spectrograms**: Automatically optimized features
   - **Research advantage**: Significantly better classification performance

3. **Dual-Tower Network** (Alternative)
   - **Temporal tower**: Time-domain features
   - **Spectral tower**: Frequency-domain features
   - **Fusion module**: Combines both representations

## 🛠️ Implementation Plan

### Phase 1: Environment Setup

#### 1.1 Create New Repository
```bash
mkdir insect-sound-classifier
cd insect-sound-classifier
git init
```

#### 1.2 Python Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install core dependencies
pip install torch torchvision torchaudio
pip install tensorflow-gpu  # or tensorflow for CPU
pip install librosa soundfile
pip install scikit-learn pandas numpy
pip install matplotlib seaborn
pip install wandb  # for experiment tracking
pip install fastapi uvicorn  # for API deployment
pip install pytest  # for testing
```

#### 1.3 Project Structure
```
insect-sound-classifier/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── data_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── metadata/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_lstm.py
│   │   ├── leaf_model.py
│   │   ├── dual_tower.py
│   │   └── base_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── postprocess.py
│   └── utils/
│       ├── __init__.py
│       ├── audio_utils.py
│       ├── viz_utils.py
│       └── config_utils.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── routes.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_evaluation.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── scripts/
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── deploy_model.py
└── models/
    └── trained/
        ├── cnn_lstm_best.pth
        ├── leaf_model_best.pth
        └── metadata.json
```

### Phase 2: Data Acquisition and Preprocessing

#### 2.1 Download InsectSound1000 Dataset
```python
# scripts/download_data.py
import requests
import zipfile
from pathlib import Path

def download_insect_data():
    """Download InsectSound1000 dataset"""
    # Note: Replace with actual dataset URL
    dataset_url = "https://path-to-insect-sound-1000.zip"
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and extract
    response = requests.get(dataset_url)
    with open(data_dir / "insect_sound_1000.zip", "wb") as f:
        f.write(response.content)
    
    with zipfile.ZipFile(data_dir / "insect_sound_1000.zip", 'r') as zip_ref:
        zip_ref.extractall(data_dir)

if __name__ == "__main__":
    download_insect_data()
```

#### 2.2 Data Preprocessing Pipeline
```python
# src/data/preprocessing.py
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict

class InsectAudioPreprocessor:
    """Preprocessor for insect sound data"""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 duration: float = 2.5,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128):
        self.target_sr = target_sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def load_and_preprocess(self, audio_path: Path) -> Dict:
        """Load and preprocess audio file"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        
        # Normalize duration
        target_length = int(self.target_sr * self.duration)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Extract features
        features = self.extract_features(audio)
        
        return {
            'waveform': audio,
            'spectrogram': features['spectrogram'],
            'mfcc': features['mfcc'],
            'chroma': features['chroma'],
            'spectral_centroid': features['spectral_centroid'],
            'zero_crossing_rate': features['zcr']
        }
    
    def extract_features(self, audio: np.ndarray) -> Dict:
        """Extract comprehensive audio features"""
        # Spectrogram
        spec = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        spec_db = librosa.amplitude_to_db(np.abs(spec))
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.target_sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel_spec)
        
        # Additional features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.target_sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        return {
            'spectrogram': mel_db,
            'mfcc': mfcc,
            'chroma': chroma,
            'spectral_centroid': spectral_centroid,
            'zcr': zcr
        }
```

#### 2.3 Data Augmentation Strategy
```python
# src/data/augmentation.py
import numpy as np
import librosa

class InsectAudioAugmenter:
    """Audio augmentation for insect sounds"""
    
    def __init__(self):
        pass
    
    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """Time stretching augmentation"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, sr: int, n_steps: float = 0.0) -> np.ndarray:
        """Pitch shifting augmentation"""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add background noise"""
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    def frequency_mask(self, spectrogram: np.ndarray, num_masks: int = 1, 
                      mask_param: int = 10) -> np.ndarray:
        """SpecAugment frequency masking"""
        spec = spectrogram.copy()
        for _ in range(num_masks):
            f = np.random.uniform(low=0.0, high=mask_param)
            f = int(f)
            f0 = np.random.uniform(low=0.0, high=spec.shape[0] - f)
            f0 = int(f0)
            spec[f0:f0 + f, :] = 0
        return spec
```

### Phase 3: Model Development

#### 3.1 CNN-LSTM Hybrid Model
```python
# src/models/cnn_lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMInsectClassifier(nn.Module):
    """CNN-LSTM hybrid model for insect classification"""
    
    def __init__(self, 
                 n_classes: int = 12,
                 cnn_channels: list = [32, 64, 128, 256],
                 lstm_hidden: int = 256,
                 lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.n_classes = n_classes
        
        # CNN Feature Extractor
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in cnn_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            ))
            in_channels = out_channels
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=8,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Prepare for LSTM (batch, time, features)
        x = x.mean(dim=2)  # Global average pooling across frequency
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        lstm_out = lstm_out.transpose(0, 1)  # (time, batch, features)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = attended.transpose(0, 1)  # (batch, time, features)
        
        # Global average pooling over time
        features = attended.mean(dim=1)
        
        # Classification
        output = self.classifier(features)
        return output
```

#### 3.2 Training Pipeline
```python
# src/training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from typing import Dict, Tuple

class InsectClassifierTrainer:
    """Training pipeline for insect classifier"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 device: torch.device,
                 config: Dict):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize wandb for experiment tracking
        if config.get('use_wandb', True):
            wandb.init(project="insect-sound-classification", config=config)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Per-class accuracy
                for i in range(len(targets)):
                    label = targets[i].item()
                    class_correct[label] = class_correct.get(label, 0) + \
                                          (predicted[i] == targets[i]).item()
                    class_total[label] = class_total.get(label, 0) + 1
        
        # Calculate per-class accuracies
        class_accuracies = {
            cls: 100. * class_correct.get(cls, 0) / class_total.get(cls, 1)
            for cls in class_total.keys()
        }
        
        return (val_loss / len(self.val_loader), 
                100. * correct / total, 
                class_accuracies)
    
    def train(self, epochs: int) -> None:
        """Full training loop"""
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, class_accs = self.validate()
            
            # Learning rate scheduling
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Logging
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            # Add per-class accuracies
            for cls, acc in class_accs.items():
                metrics[f'val_acc_class_{cls}'] = acc
            
            if self.config.get('use_wandb', True):
                wandb.log(metrics)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, 'models/trained/best_model.pth')
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
```

### Phase 4: API Development

#### 4.1 FastAPI Service
```python
# api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, List
import json

from src.models.cnn_lstm import CNNLSTMInsectClassifier
from src.data.preprocessing import InsectAudioPreprocessor
from api.models import InsectPrediction, InsectResult

app = FastAPI(
    title="Insect Sound Classification API",
    description="Deep learning-based insect species identification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
class_names = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, preprocessor, class_names
    
    # Load trained model
    model_path = Path("models/trained/best_model.pth")
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    model = CNNLSTMInsectClassifier(n_classes=config['n_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize preprocessor
    preprocessor = InsectAudioPreprocessor()
    
    # Load class names
    with open("data/metadata/class_names.json", 'r') as f:
        class_names = json.load(f)
    
    print("✅ Insect classifier loaded successfully")

@app.post("/classify", response_model=InsectResult)
async def classify_insect(audio: UploadFile = File(...)):
    """Classify insect sound from audio file"""
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(400, "File must be audio format")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Preprocess audio
        features = preprocessor.load_and_preprocess(Path(tmp_path))
        
        # Prepare input tensor
        spectrogram = torch.FloatTensor(features['spectrogram']).unsqueeze(0).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = model(spectrogram)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=5)
            
            predictions = [
                InsectPrediction(
                    species=class_names[str(idx.item())],
                    confidence=float(prob.item()),
                    index=idx.item()
                )
                for prob, idx in zip(top_probs, top_indices)
            ]
        
        # Cleanup
        Path(tmp_path).unlink()
        
        return InsectResult(
            predictions=predictions,
            top_prediction=predictions[0],
            model_info={
                "name": "CNN-LSTM Insect Classifier",
                "version": "1.0.0",
                "species_count": len(class_names)
            }
        )
        
    except Exception as e:
        raise HTTPException(500, f"Classification failed: {str(e)}")

@app.get("/species")
async def get_species_list():
    """Get list of supported insect species"""
    return {"species": list(class_names.values())}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}
```

#### 4.2 Response Models
```python
# api/models.py
from pydantic import BaseModel
from typing import List, Dict, Optional

class InsectPrediction(BaseModel):
    species: str
    confidence: float
    index: int

class InsectResult(BaseModel):
    predictions: List[InsectPrediction]
    top_prediction: InsectPrediction
    model_info: Dict[str, any]
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
```

### Phase 5: Integration with SoundCurious MoE

#### 5.1 MoE Expert Interface
```python
# integration/moe_expert.py
import requests
import asyncio
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class InsectExpertClassifier:
    """Insect expert for MoE system integration"""
    
    def __init__(self, api_url: str = "http://localhost:8081"):
        self.api_url = api_url
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize connection to insect classifier API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200 and response.json()["model_loaded"]:
                self.is_initialized = True
                logger.info("✅ Insect expert classifier initialized")
            else:
                logger.error("❌ Insect classifier API not ready")
        except Exception as e:
            logger.error(f"❌ Failed to connect to insect classifier: {e}")
    
    async def classify(self, processed_audio) -> Dict[str, Any]:
        """Classify audio for insect species"""
        if not self.is_initialized:
            return self._create_unavailable_result()
        
        try:
            # Convert processed audio to file-like object
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, processed_audio.waveform, processed_audio.sample_rate)
                
                # Call insect classifier API
                with open(tmp.name, 'rb') as audio_file:
                    files = {'audio': audio_file}
                    response = requests.post(f"{self.api_url}/classify", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    return self._format_result(result)
                else:
                    return self._create_error_result(f"API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Insect classification failed: {e}")
            return self._create_error_result(str(e))
    
    def _format_result(self, api_result: Dict) -> Dict[str, Any]:
        """Format API result for MoE compatibility"""
        top_pred = api_result["top_prediction"]
        
        return {
            'species': top_pred['species'],
            'confidence': top_pred['confidence'],
            'source': 'insect-expert',
            'predictions': api_result['predictions'][:5],
            'model_info': api_result['model_info'],
            'domain': 'insect',
            'expert_type': 'insect_specialist'
        }
    
    def _create_unavailable_result(self) -> Dict[str, Any]:
        """Create result when insect expert is unavailable"""
        return {
            'species': 'Insect Expert Unavailable',
            'confidence': 0.0,
            'source': 'insect-expert-unavailable',
            'error': 'Insect classifier not initialized'
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create result for classification errors"""
        return {
            'species': 'Classification Error',
            'confidence': 0.0,
            'source': 'insect-expert-error',
            'error': error_msg
        }
    
    def is_available(self) -> bool:
        """Check if insect expert is available"""
        return self.is_initialized
```

## 🚀 Deployment and Testing

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8081

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8081"]
```

### Testing Strategy
1. **Unit Tests**: Test individual components (preprocessing, model inference)
2. **Integration Tests**: Test API endpoints and data flow
3. **Performance Tests**: Measure inference speed and memory usage
4. **Accuracy Tests**: Validate model performance on held-out test set

### Performance Benchmarks
Target metrics based on research:
- **Accuracy**: >85% on clean recordings
- **Inference Time**: <500ms per sample
- **Memory Usage**: <2GB RAM
- **Species Coverage**: 12+ insect types from InsectSound1000

## 📈 Future Enhancements

1. **Extended Species Support**: Integrate additional datasets (InsectSet459, etc.)
2. **Real-time Processing**: Optimize for streaming audio classification
3. **Environmental Robustness**: Train on noisy/field recordings
4. **Mobile Deployment**: Create TensorFlow Lite/ONNX versions
5. **Multi-modal Features**: Incorporate temporal and environmental context

## 📚 Research References

1. **InsectSound1000**: "An insect sound dataset for deep learning based acoustic insect recognition" (Nature Scientific Data, 2024)
2. **LEAF Architecture**: "Adaptive Representations of Sound for Automatic Insect Recognition" (PLOS Computational Biology)
3. **CNN-LSTM Performance**: "Audio-Based Classification of Insect Species Using Machine Learning Models" (ArXiv, 2024)
4. **SpecAugment**: "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" (Google, 2019)

## 🎯 Success Criteria

The project will be considered successful when:
1. ✅ Model achieves >85% accuracy on InsectSound1000 test set
2. ✅ API can process audio files in <500ms
3. ✅ Successfully integrates with SoundCurious MoE system
4. ✅ Handles common insect species (cricket, cicada, bee, mosquito, etc.)
5. ✅ Provides confidence scores and multiple predictions
6. ✅ Robust to different audio qualities and lengths

---

## 🔄 Integration Back to SoundCurious

Once the insect classifier is trained and deployed:

1. **Add to MoE Config**: Add insect expert to the expert list
2. **Update Router**: Train router to route insect sounds to the insect expert
3. **Enhance Kimi**: Use insect classifier results to improve Kimi's domain insights
4. **API Integration**: Add insect expert results to the main classification response

This dedicated insect classification system will significantly enhance the SoundCurious platform's ability to identify insect sounds with high accuracy and confidence.