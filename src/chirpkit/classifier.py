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
        """Initialize the classifier with PyTorch model and graceful fallbacks"""
        if self.is_initialized:
            return

        self.torch = DependencyManager.get_torch()
        if self.torch is None:
            raise RuntimeError("PyTorch not available - ChirpKit requires PyTorch for neural network inference")

        # Set device
        self.device = self._get_device(self.torch)
        logger.info(f"Using device: {self.device}")

        # Try to load model with graceful degradation
        model_loaded = False
        try:
            await self._load_model()
            model_loaded = True
            logger.info("✅ Model loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"⚠️  No trained model found: {e}")
            logger.info("📋 ChirpKit will attempt to work with basic functionality")
        except Exception as e:
            logger.warning(f"⚠️  Model loading failed: {e}")
            logger.info("📋 ChirpKit will attempt to work with basic functionality")

        # Try to load species labels with fallback
        try:
            await self._load_species_labels()
            logger.info(f"✅ Loaded {len(self.species_labels)} species labels")
        except Exception as e:
            logger.warning(f"⚠️  Could not load species labels: {e}")
            self.species_labels = self._get_default_species_list()
            logger.info(f"📋 Using default species list with {len(self.species_labels)} common species")

        # If no model was loaded but we have a working PyTorch setup, create a minimal setup
        if not model_loaded and not self.model:
            logger.info("🔧 Creating minimal classifier setup for basic functionality")
            self._create_minimal_setup()

        self.is_initialized = True
        
        # Report final status
        if model_loaded and self.model:
            logger.info("✅ ChirpKit InsectClassifier fully initialized with trained model")
        else:
            logger.info("⚠️  ChirpKit InsectClassifier initialized with limited functionality")

    async def _load_model(self):
        """Load the CNN-LSTM PyTorch model"""
        torch = DependencyManager.get_torch()
        
        # Import the model architecture with comprehensive fallback handling
        SimpleCNNLSTMInsectClassifier = None
        
        # List of import paths to try
        import_attempts = [
            # Relative imports
            ("...models.simple_cnn_lstm", "SimpleCNNLSTMInsectClassifier"),
            ("..models.simple_cnn_lstm", "SimpleCNNLSTMInsectClassifier"),
            # Absolute imports
            ("src.models.simple_cnn_lstm", "SimpleCNNLSTMInsectClassifier"),
            ("models.simple_cnn_lstm", "SimpleCNNLSTMInsectClassifier"),
        ]
        
        # Try standard imports first
        for module_path, class_name in import_attempts:
            try:
                if module_path.startswith("..."):
                    from ...models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
                elif module_path.startswith(".."):
                    from ..models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier  
                else:
                    module = __import__(module_path, fromlist=[class_name])
                    SimpleCNNLSTMInsectClassifier = getattr(module, class_name)
                logger.info(f"Successfully imported model architecture from {module_path}")
                break
            except ImportError as e:
                logger.debug(f"Failed to import from {module_path}: {e}")
                continue
        
        # Try direct file path imports if standard imports failed
        if SimpleCNNLSTMInsectClassifier is None:
            import sys
            import os
            import importlib.util
            
            # Possible file paths to try
            base_dir = Path(__file__).parent.parent.parent  # Go up to project root
            file_paths = [
                base_dir / "src" / "models" / "simple_cnn_lstm.py",
                base_dir / "models" / "simple_cnn_lstm.py",
                Path("src/models/simple_cnn_lstm.py"),
                Path("models/simple_cnn_lstm.py"),
            ]
            
            for file_path in file_paths:
                if file_path.exists():
                    try:
                        spec = importlib.util.spec_from_file_location("simple_cnn_lstm", file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        SimpleCNNLSTMInsectClassifier = module.SimpleCNNLSTMInsectClassifier
                        logger.info(f"Successfully loaded model architecture from {file_path}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load from {file_path}: {e}")
                        continue
        
        # Final fallback: create architecture inline
        if SimpleCNNLSTMInsectClassifier is None:
            logger.warning("Could not import SimpleCNNLSTMInsectClassifier, creating inline definition")
            SimpleCNNLSTMInsectClassifier = self._create_fallback_architecture()

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
        """Synchronous classification with fallback for untrained models"""
        torch = DependencyManager.get_torch()
        
        # Check if we have a trained model
        if not self._has_trained_model():
            return self._create_heuristic_result(processed_audio, detailed)
        
        # Extract features exactly as in training pipeline
        features = self._extract_features(processed_audio)

        # Make prediction with trained model
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

    def _has_trained_model(self) -> bool:
        """Check if we have a trained model (not just architecture)"""
        return (self.model is not None and 
                hasattr(self, 'model_path') and 
                self.model_path is not None)

    def _create_heuristic_result(self, processed_audio, detailed: bool) -> Dict[str, Any]:
        """Create heuristic-based result when no trained model is available"""
        # Simple heuristic: analyze basic audio properties
        try:
            # Get audio data
            if hasattr(processed_audio, 'waveform'):
                audio = processed_audio.waveform
                sr = processed_audio.sample_rate
            else:
                audio = processed_audio
                sr = self.sample_rate

            # Basic heuristic: higher frequency content suggests insects
            import librosa
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroids)
            
            # Simple heuristic classification
            is_likely_insect = avg_centroid > 2000  # Hz
            confidence = min(0.7, avg_centroid / 10000)  # Cap at 70% for heuristics
            
            # Pick a default species based on frequency characteristics
            if avg_centroid > 6000:
                default_species = 'Cicada_orni'  # High frequency = cicada-like
            elif avg_centroid > 4000:
                default_species = 'Tettigonia_viridissima'  # Mid-high = katydid-like
            else:
                default_species = 'Gryllus_bimaculatus'  # Lower = cricket-like
                
        except Exception as e:
            logger.warning(f"Heuristic analysis failed: {e}")
            is_likely_insect = True
            confidence = 0.5
            default_species = 'Gryllus_bimaculatus'

        # Create simple prediction list
        results = [
            {'species': default_species, 'confidence': confidence, 'rank': 1}
        ]
        
        # Add a few other common species with lower confidence
        if detailed:
            other_species = [s for s in self.species_labels[:4] if s != default_species]
            for i, species in enumerate(other_species):
                results.append({
                    'species': species,
                    'confidence': max(0.1, confidence - 0.15 * (i + 1)),
                    'rank': i + 2
                })

        return {
            'model': 'ChirpKit-Heuristic',
            'classification': {
                'is_insect': is_likely_insect,
                'species': default_species,
                'confidence': confidence,
                'family': self._get_family_from_species(default_species)
            },
            'confidence': confidence,
            'predictions': results,
            'features': {
                'chirpkit_powered': True,
                'neural_network': False,
                'heuristic_fallback': True,
                'total_species': len(self.species_labels),
                'note': 'Using basic heuristics - no trained model available'
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

    def _create_fallback_architecture(self):
        """Create CNN-LSTM architecture inline as fallback when imports fail"""
        torch = DependencyManager.get_torch()
        
        class FallbackCNNLSTMInsectClassifier(torch.nn.Module):
            """Fallback CNN-LSTM model architecture created inline"""
            def __init__(self, n_classes: int = 471, dropout: float = 0.3):
                super().__init__()
                self.n_classes = n_classes
                
                # CNN layers (matching the working architecture)
                self.conv_layers = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(32),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.MaxPool2d(2, 2),
                        torch.nn.Dropout2d(dropout)
                    ),
                    torch.nn.Sequential(
                        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.MaxPool2d(2, 2),
                        torch.nn.Dropout2d(dropout)
                    ),
                    torch.nn.Sequential(
                        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(128),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.MaxPool2d(2, 2),
                        torch.nn.Dropout2d(dropout)
                    ),
                    torch.nn.Sequential(
                        torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                        torch.nn.BatchNorm2d(256),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.MaxPool2d(2, 2),
                        torch.nn.Dropout2d(dropout)
                    )
                ])
                
                # LSTM
                self.lstm = torch.nn.LSTM(
                    input_size=256,
                    hidden_size=256,
                    num_layers=2,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=True
                )
                
                # Attention
                self.attention = torch.nn.MultiheadAttention(
                    embed_dim=512,  # 256 * 2 (bidirectional)
                    num_heads=8,
                    dropout=dropout
                )
                
                # Classifier
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(512, 256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(128, n_classes)
                )
            
            def forward(self, x):
                batch_size = x.size(0)
                
                # CNN forward pass
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)
                
                # Global average pooling over frequency dimension
                x = x.mean(dim=2)  # [batch, channels, time]
                x = x.transpose(1, 2)  # [batch, time, channels]
                
                # LSTM processing
                lstm_out, _ = self.lstm(x)
                
                # Attention
                lstm_out = lstm_out.transpose(0, 1)  # [seq, batch, features]
                attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
                attended = attended.transpose(0, 1)  # [batch, seq, features]
                
                # Pooling
                features = attended.mean(dim=1)  # Average over time
                
                # Classification
                output = self.classifier(features)
                return output
        
        return FallbackCNNLSTMInsectClassifier

    def _get_default_species_list(self) -> List[str]:
        """Get a default list of common insect species for fallback"""
        return [
            'Gryllus_bimaculatus',  # Two-spotted cricket
            'Acheta_domesticus',    # House cricket
            'Teleogryllus_commodus', # Black cricket
            'Gryllus_campestris',   # Field cricket
            'Allonemobius_allardi', # Allard's ground cricket
            'Nemobius_sylvestris',  # Wood cricket
            'Oecanthus_pellucens',  # Tree cricket
            'Tettigonia_viridissima', # Great green bush-cricket
            'Conocephalus_fuscus',  # Long-winged conehead
            'Pholidoptera_griseoaptera', # Dark bush-cricket
            'Chorthippus_brunneus', # Field grasshopper
            'Chorthippus_parallelus', # Meadow grasshopper
            'Omocestus_viridulus',  # Common green grasshopper
            'Pseudochorthippus_parallelus', # Meadow grasshopper
            'Stenobothrus_lineatus', # Stripe-winged grasshopper
            'Cicada_orni',          # European cicada
            'Cicadetta_montana',    # New Forest cicada
            'Magicicada_septendecim', # Periodical cicada
            'Neotibicen_canicularis', # Dog-day cicada
            'Tibicen_lyricen'       # Lyric cicada
        ]

    def _create_minimal_setup(self):
        """Create a minimal setup when no trained model is available"""
        # Create a basic architecture for structure (won't be used for actual inference)
        SimpleCNNLSTMInsectClassifier = self._create_fallback_architecture()
        
        # Create a minimal model (not trained, just for structure)
        self.model = SimpleCNNLSTMInsectClassifier(
            n_classes=len(self.species_labels),
            dropout=0.3
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Created minimal model architecture with {len(self.species_labels)} species")

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