"""
Main InsectClassifier class for inference-only usage
"""
import torch
import numpy as np
import librosa
import joblib
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from .models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
except ImportError:
    # Fallback for direct execution
    from models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier


class InsectClassifier:
    """
    Main interface for insect sound classification inference.
    
    Usage:
        classifier = InsectClassifier()
        if classifier.load_model():
            prediction = classifier.predict_audio("audio_file.wav")
            print(f"Species: {prediction['species']}")
            print(f"Confidence: {prediction['confidence']:.2%}")
    """
    
    def __init__(self, model_dir: str = "models/trained"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = None
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.duration = 2.5
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
    
    def load_model(self, model_path: Optional[str] = None, label_encoder_path: Optional[str] = None) -> bool:
        """
        Load the trained model and label encoder.
        
        Args:
            model_path: Optional explicit path to model file
            label_encoder_path: Optional explicit path to label encoder file
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            if label_encoder_path is None:
                # Auto-detect latest label encoder
                encoder_files = list(self.model_dir.glob("*_label_encoder.joblib"))
                if not encoder_files:
                    print(f"No label encoder found in {self.model_dir}")
                    return False
                
                # Use the one with most species (highest number)
                label_encoder_path = max(encoder_files, key=lambda x: int(str(x).split('_')[-3].replace('species', '')))
            
            print(f"Loading label encoder: {label_encoder_path}")
            self.label_encoder = joblib.load(label_encoder_path)
            self.n_classes = len(self.label_encoder.classes_)
            print(f"Loaded {self.n_classes} species classes")
            
            if model_path is None:
                # Auto-detect corresponding model file
                encoder_name = Path(label_encoder_path).name
                model_name = encoder_name.replace('_label_encoder.joblib', '.pth')
                model_path = self.model_dir / model_name
            
            if not Path(model_path).exists():
                print(f"Model file not found: {model_path}")
                return False
                
            print(f"Loading model: {model_path}")
            self.model = SimpleCNNLSTMInsectClassifier(n_classes=self.n_classes)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _preprocess_audio(self, audio_file: str) -> torch.Tensor:
        """
        Preprocess audio file for model input.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            torch.Tensor: Preprocessed audio tensor ready for model
        """
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        # Ensure correct length
        target_length = int(self.sample_rate * self.duration)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel_spec)
        
        # Convert to tensor and add batch/channel dimensions
        input_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return input_tensor.to(self.device)
    
    def predict_audio(self, audio_file: str, top_k: int = 5) -> Dict:
        """
        Predict insect species from audio file.
        
        Args:
            audio_file: Path to audio file
            top_k: Number of top predictions to return
            
        Returns:
            Dict containing:
                - species: Predicted species name
                - confidence: Prediction confidence (0-1)
                - top_predictions: List of (species, confidence) tuples for top_k predictions
        """
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess audio
            input_tensor = self._preprocess_audio(audio_file)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            # Get predicted species
            predicted_species = self.label_encoder.classes_[predicted_idx]
            
            # Get top-k predictions
            probs = probabilities[0].cpu().numpy()
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_predictions = [(self.label_encoder.classes_[i], float(probs[i])) for i in top_indices]
            
            return {
                'species': predicted_species,
                'confidence': float(confidence),
                'top_predictions': top_predictions,
                'scientific_name': predicted_species.replace('_', ' ')
            }
            
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")
    
    def get_species_list(self) -> List[str]:
        """
        Get list of all species the model can classify.
        
        Returns:
            List[str]: List of species names
        """
        if self.label_encoder is None:
            raise RuntimeError("Label encoder not loaded. Call load_model() first.")
        
        return list(self.label_encoder.classes_)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict with model information
        """
        return {
            'n_classes': self.n_classes,
            'device': str(self.device),
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'model_loaded': self.model is not None,
            'label_encoder_loaded': self.label_encoder is not None
        }