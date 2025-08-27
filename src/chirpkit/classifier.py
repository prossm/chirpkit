"""
ChirpKit Insect Classifier - Neural network-based insect sound identification
"""

import logging
import numpy as np
import tempfile
import soundfile as sf
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Use our dependency management system
from .dependencies import DependencyManager, requires_torch

logger = logging.getLogger(__name__)

class InsectClassifier:
    """Neural network-based insect sound classifier using CNN-LSTM architecture"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the InsectClassifier
        
        Args:
            model_path: Optional path to pre-trained model. If None, uses default model.
        """
        self.model = None
        self.is_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.model_path = model_path
        self.species_labels = []
        self.label_encoder = None
        self.n_classes = 471  # Default from our trained model
        self.torch = None  # Will be set during initialization
        self.device = None  # Will be set during initialization
        
        # Audio processing parameters that match training
        self.sample_rate = 16000
        self.duration = 2.5
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
    @requires_torch
    async def initialize(self):
        """Initialize the classifier with PyTorch model"""
        if self.is_initialized:
            return

        try:
            self.torch = DependencyManager.get_torch()
            if self.torch is None:
                raise RuntimeError("PyTorch not available")

            # Set device
            self.device = self._get_device(self.torch)

            # Load the model and species labels
            await self._load_model()
            await self._load_species_labels()

            self.is_initialized = True
            logger.info("✅ ChirpKit InsectClassifier initialized with PyTorch backend")

        except Exception as e:
            logger.error(f"❌ Failed to initialize InsectClassifier: {e}")
            raise

    async def _load_model(self):
        """Load the CNN-LSTM PyTorch model"""
        torch = DependencyManager.get_torch()
        
        # Import the model architecture from existing codebase
        try:
            from ...models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
        except ImportError:
            try:
                # Fallback import path
                from src.models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
            except ImportError:
                try:
                    # Another fallback for different project structures
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
                    from models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
                except ImportError:
                    # Final fallback - direct import
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("simple_cnn_lstm", "src/models/simple_cnn_lstm.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    SimpleCNNLSTMInsectClassifier = module.SimpleCNNLSTMInsectClassifier

        # Use model manager to find models
        from .models import find_any_model
        
        if self.model_path is None:
            # Try to find any available model
            model_files = find_any_model()
            if model_files:
                self.model_path, encoder_path, info_path = model_files
                logger.info(f"Using discovered model: {self.model_path}")
            else:
                raise FileNotFoundError("No trained model found. Run training or download a pre-trained model.")

        # Load model info to get n_classes
        info_path = str(self.model_path).replace('.pth', '_info.json')
        if Path(info_path).exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
                self.n_classes = model_info['n_classes']
                logger.info(f"Model supports {self.n_classes} species")
        else:
            # Try to infer from encoder
            encoder_path = str(self.model_path).replace('.pth', '_label_encoder.joblib')
            if Path(encoder_path).exists():
                import joblib
                encoder = joblib.load(encoder_path)
                self.n_classes = len(encoder.classes_)
                logger.info(f"Inferred {self.n_classes} species from encoder")
            else:
                logger.warning("Could not determine number of classes, using default 471")

        # Create and load model with exact architecture from training
        self.model = SimpleCNNLSTMInsectClassifier(
            n_classes=self.n_classes,
            dropout=0.3  # Match training parameters
        )
        
        device = self._get_device(torch)
        
        # Load state dict with error handling
        try:
            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _get_device(self, torch):
        """Get appropriate device for computation."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    async def _load_species_labels(self):
        """Load species labels mapping"""
        # Load the label encoder
        encoder_path = str(self.model_path).replace('.pth', '_label_encoder.joblib')
        if not Path(encoder_path).exists():
            # Fall back to info file
            info_path = str(self.model_path).replace('.pth', '_info.json')
            if Path(info_path).exists():
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                    self.species_labels = model_info.get('species_list', [])
                    logger.info(f"Loaded {len(self.species_labels)} species from info file")
                    return
            else:
                raise FileNotFoundError(f"Neither {encoder_path} nor {info_path} found")
        
        # Load joblib label encoder
        import joblib
        self.label_encoder = joblib.load(encoder_path)
        self.species_labels = list(self.label_encoder.classes_)
        logger.info(f"Loaded {len(self.species_labels)} species from label encoder")

    async def classify(self, processed_audio, detailed: bool = True) -> Dict[str, Any]:
        """
        Classify processed audio for insect species
        
        Args:
            processed_audio: ProcessedAudio object from audio processor
            detailed: Whether to return detailed predictions
            
        Returns:
            Classification results compatible with MoE system
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # Run classification in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._classify_sync,
                processed_audio,
                detailed
            )
            return result

        except Exception as e:
            logger.error(f"ChirpKit classification failed: {e}")
            return self._create_error_result(str(e))

    def _classify_sync(self, processed_audio, detailed: bool) -> Dict[str, Any]:
        """Synchronous classification"""
        torch = DependencyManager.get_torch()
        
        # Extract features exactly as in training pipeline
        features = self._extract_features(processed_audio)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            probs_numpy = probabilities[0].cpu().numpy()

        # Get top predictions
        top_indices = np.argsort(probs_numpy)[::-1][:5]
        
        results = []
        for i, idx in enumerate(top_indices):
            species_name = self.species_labels[idx] if idx < len(self.species_labels) else f"Species_{idx}"
            results.append({
                'species': species_name,
                'confidence': float(probs_numpy[idx]),
                'rank': i + 1
            })

        top_result = results[0]

        return {
            'model': 'ChirpKit-CNN-LSTM',
            'classification': {
                'is_insect': top_result['confidence'] > 0.01,  # Lower threshold for 471 species
                'species': top_result['species'],
                'confidence': top_result['confidence'],
                'family': self._get_family_from_species(top_result['species'])
            },
            'confidence': top_result['confidence'],
            'predictions': results if detailed else results[:3],
            'features': {
                'chirpkit_powered': True,
                'neural_network': True,
                'cnn_lstm_architecture': True,
                'total_species': len(self.species_labels),
                'pytorch_backend': True
            }
        }

    def _extract_features(self, processed_audio):
        """Extract features exactly matching training pipeline"""
        torch = DependencyManager.get_torch()
        import librosa

        # Get audio data
        if hasattr(processed_audio, 'waveform'):
            audio = processed_audio.waveform
            sr = processed_audio.sample_rate
        else:
            # Fallback for different audio object formats
            audio = processed_audio
            sr = self.sample_rate

        # Ensure correct sample rate
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Ensure correct length (exactly as in training)
        target_length = int(self.sample_rate * self.duration)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

        # Create mel spectrogram (exactly matching training parameters)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Convert to PyTorch tensor and add batch/channel dimensions
        features = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Move to same device as model
        if self.model:
            features = features.to(next(self.model.parameters()).device)

        return features

    def _get_family_from_species(self, species_name: str) -> str:
        """Map species to family based on taxonomic knowledge"""
        # Common insect family mappings based on genus
        family_mapping = {
            # Crickets (Gryllidae)
            'Gryllus': 'Gryllidae',
            'Acheta': 'Gryllidae', 
            'Nemobius': 'Gryllidae',
            'Allonemobius': 'Gryllidae',
            'Anaxipha': 'Gryllidae',
            'Oecanthus': 'Gryllidae',
            'Ornebius': 'Gryllidae',
            'Teleogryllus': 'Gryllidae',
            'Gryllodes': 'Gryllidae',
            'Eumodicogryllus': 'Gryllidae',
            'Eunemobius': 'Gryllidae',
            
            # Mole crickets (Gryllotalpidae) 
            'Gryllotalpa': 'Gryllotalpidae',
            'Neocurtilla': 'Gryllotalpidae',
            
            # Katydids/Bush-crickets (Tettigoniidae)
            'Tettigonia': 'Tettigoniidae',
            'Conocephalus': 'Tettigoniidae',
            'Microcentrum': 'Tettigoniidae',
            'Phaneroptera': 'Tettigoniidae',
            'Leptophyes': 'Tettigoniidae',
            'Isophya': 'Tettigoniidae',
            'Poecilimon': 'Tettigoniidae',
            'Barbitistes': 'Tettigoniidae',
            'Ephippiger': 'Tettigoniidae',
            'Decticus': 'Tettigoniidae',
            'Platycleis': 'Tettigoniidae',
            'Metrioptera': 'Tettigoniidae',
            'Roeseliana': 'Tettigoniidae',
            'Pholidoptera': 'Tettigoniidae',
            'Tylopsis': 'Tettigoniidae',
            'Orchelimum': 'Tettigoniidae',
            
            # Grasshoppers (Acrididae)
            'Chorthippus': 'Acrididae',
            'Omocestus': 'Acrididae',
            'Pseudochorthippus': 'Acrididae',
            'Stenobothrus': 'Acrididae',
            'Gomphocerippus': 'Acrididae',
            'Chrysochraon': 'Acrididae',
            'Euchorthippus': 'Acrididae',
            
            # Cicadas (Cicadidae)
            'Cicada': 'Cicadidae',
            'Cicadatra': 'Cicadidae',
            'Cicadetta': 'Cicadidae',
            'Magicicada': 'Cicadidae',
            'Neotibicen': 'Cicadidae',
            'Diceroprocta': 'Cicadidae',
            'Tibicina': 'Cicadidae',
            'Lyristes': 'Cicadidae',
            'Cryptotympana': 'Cicadidae',
            'Platypleura': 'Cicadidae',
            'Megatibicen': 'Cicadidae',
            'Okanagana': 'Cicadidae',
            'Quesada': 'Cicadidae',
            'Psaltoda': 'Cicadidae',
            'Thopha': 'Cicadidae',
            
            # Other families
            'Anabrus': 'Anabridae',  # Mormon crickets
        }

        genus = species_name.split('_')[0] if '_' in species_name else species_name
        family = family_mapping.get(genus, 'Unknown')
        
        return family

    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'model': 'ChirpKit-Error',
            'classification': {
                'is_insect': False,
                'species': 'Error',
                'confidence': 0.0,
                'family': 'unknown'
            },
            'confidence': 0.0,
            'error': error_msg,
            'features': {
                'chirpkit_powered': False,
                'error': True
            }
        }

    def is_available(self) -> bool:
        """Check if classifier is available"""
        return self.is_initialized and self.model is not None

    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.is_initialized = False
        logger.info("ChirpKit classifier cleanup completed")

    # Compatibility methods for existing codebase
    def load_model(self, model_path: Optional[str] = None, label_encoder_path: Optional[str] = None) -> bool:
        """Synchronous model loading for compatibility"""
        try:
            if model_path:
                self.model_path = model_path
                
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in a loop, we need to run in a thread
                import threading
                import concurrent.futures
                
                def run_init():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        new_loop.run_until_complete(self.initialize())
                        return True
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_init)
                    return future.result()
                    
            except RuntimeError:
                # No loop running, we can create our own
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.initialize())
                    return True
                finally:
                    loop.close()
                    
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict_audio(self, audio_file: str, top_k: int = 5) -> Dict:
        """Predict insect species from audio file (compatibility method)"""
        import librosa
        
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        # Create a simple audio object
        class SimpleAudio:
            def __init__(self, waveform, sample_rate):
                self.waveform = waveform
                self.sample_rate = sample_rate
        
        processed_audio = SimpleAudio(audio, sr)
        
        # Run classification (handle asyncio properly)
        try:
            loop = asyncio.get_running_loop()
            # If we're in a loop, run in thread
            import concurrent.futures
            
            def run_classify():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.classify(processed_audio, detailed=True))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_classify)
                result = future.result()
                
        except RuntimeError:
            # No loop running, create our own
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.classify(processed_audio, detailed=True))
            finally:
                loop.close()
        
        # Convert to expected format
        if 'error' in result:
            raise RuntimeError(f"Classification error: {result['error']}")
            
        predictions = result['predictions'][:top_k]
        top_prediction = predictions[0]
        
        return {
            'species': top_prediction['species'],
            'confidence': float(top_prediction['confidence']),
            'top_predictions': [(p['species'], p['confidence']) for p in predictions],
            'scientific_name': top_prediction['species'].replace('_', ' ')
        }

    def get_species_list(self) -> List[str]:
        """Get list of all species the model can classify"""
        if not self.species_labels:
            raise RuntimeError("Species labels not loaded. Call initialize() first.")
        return list(self.species_labels)

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        torch = DependencyManager.get_torch()
        device = next(self.model.parameters()).device if self.model else 'unknown'
        
        return {
            'n_classes': self.n_classes,
            'device': str(device),
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'model_loaded': self.model is not None,
            'species_loaded': len(self.species_labels) > 0,
            'backend': 'pytorch',
            'total_species': len(self.species_labels)
        }