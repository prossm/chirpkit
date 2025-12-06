"""
ChirpKit Insect Classifier

Neural network-based insect sound identification using BirdNET embeddings
and a 7-model ensemble for robust predictions.

Model Version: v6.0 (ensemble architecture)
Package Version: See __version__ in _version.py
"""

import numpy as np
import json
import asyncio
import requests
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from .dependencies import DependencyManager, requires_torch
from ._version import __version__, __model_version__
from .model_downloader import get_default_cache_dir
from .utils import get_chirpkit_logger

logger = get_chirpkit_logger(__name__)

class InsectClassifier:
    """Neural network-based insect sound classifier using ensemble model"""

    def __init__(
        self,
        model_root: Optional[str] = None,
        birdnet_model_path: Optional[str] = None,
        ensemble_path: Optional[str] = None,
        mode: str = "ensemble_tta",
        auto_download: bool = False,
        validate_compatibility: bool = True,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the InsectClassifier with flexible model configuration

        Args:
            model_root: Root directory for all models (if not using specific paths)
            birdnet_model_path: Explicit path to BirdNET model file
            ensemble_path: Explicit path to ensemble model directory
            mode: Deployment mode - 'single', 'ensemble', or 'ensemble_tta' (default)
            auto_download: Whether to auto-download missing models (default: False)
            validate_compatibility: Whether to validate model compatibility (default: True)
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detection)
            **kwargs: Additional configuration options

        Environment Variables:
            CHIRPKIT_ROOT_DIR: Root directory for all models
            CHIRPKIT_BIRDNET_MODEL: Path to BirdNET model
            CHIRPKIT_ENSEMBLE_DIR: Path to ensemble directory
            CHIRPKIT_MODEL_DIR: Legacy - model root directory
            CHIRPKIT_HOME: Legacy - ChirpKit home directory

        Configuration File Support:
            ~/.chirpkit/config.yaml or ~/.chirpkit/config.json
            ./chirpkit.config.yaml or ./chirpkit.config.json
        """
        # Import configuration system
        from .config import ConfigurationManager, ModelDiscovery, ModelValidator

        # Combine all user-provided configuration
        user_config = {
            'root_directory': model_root,
            'birdnet_model_path': birdnet_model_path,
            'ensemble_path': ensemble_path,
            'mode': mode,
            'auto_download': auto_download,
            'validate_compatibility': validate_compatibility,
            'device': device,
            **kwargs
        }
        
        # Remove None values to allow lower-priority sources to provide values
        user_config = {k: v for k, v in user_config.items() if v is not None}

        # Resolve final configuration
        config_manager = ConfigurationManager(user_config)
        self.config = config_manager.resolve_configuration()

        # Resolve model paths from configuration
        self.model_path, self.birdnet_path = self._resolve_model_paths()

        self.mode = self.config.mode
        self.is_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Will be set during initialization
        self.ensemble_classifier = None
        self.birdnet_extractor = None
        self.species_labels = []
        self.label_encoder = None
        self.n_classes = 231  # Model ensemble classes
        self.torch = None
        self.device = None

        # Wikipedia integration
        self.species_cache = {}
        self.cache_file = Path("species_cache.json")
        self.enable_enrichment = True
        
    def _resolve_model_paths(self) -> Tuple[str, Optional[str]]:
        """
        Resolve model paths from configuration
        
        Returns:
            Tuple of (ensemble_path, birdnet_path)
        """
        from .config import ModelDiscovery, ModelValidator
        from .model_downloader import get_default_cache_dir
        
        ensemble_path = self.config.ensemble_path
        birdnet_path = self.config.birdnet_model_path
        
        # If explicit paths are provided, use them
        if ensemble_path and birdnet_path:
            logger.info(f"Using explicit model paths:")
            logger.info(f"  Ensemble: {ensemble_path}")
            logger.info(f"  BirdNET: {birdnet_path}")
            
            if self.config.validate_compatibility:
                ModelValidator.validate_model_compatibility(birdnet_path, ensemble_path)
            
            return ensemble_path, birdnet_path
        
        # If root directory is provided, discover models
        if self.config.root_directory:
            root_path = Path(self.config.root_directory)
            logger.info(f"Discovering models in root directory: {root_path}")
            
            discovered = ModelDiscovery.find_models(root_path)
            selection = ModelDiscovery.select_best_models(discovered)
            
            # Use discovered paths, falling back to explicit paths if provided
            final_ensemble_path = ensemble_path or selection.get('ensemble_path')
            final_birdnet_path = birdnet_path or selection.get('birdnet_model_path')
            
            if final_ensemble_path and final_birdnet_path:
                logger.info(f"Discovered model paths:")
                logger.info(f"  Ensemble: {final_ensemble_path}")
                logger.info(f"  BirdNET: {final_birdnet_path}")
                
                if self.config.validate_compatibility:
                    ModelValidator.validate_model_compatibility(final_birdnet_path, final_ensemble_path)
                
                return final_ensemble_path, final_birdnet_path
        
        # Fallback to legacy behavior
        if self.config.fallback_to_default:
            logger.info("Falling back to default model paths")
            
            # Try development mode first
            dev_model_path = Path("models/trained/chirpkit-ensemble")
            if dev_model_path.exists():
                default_ensemble_path = str(dev_model_path)
                default_birdnet_path = None  # Let BirdNET extractor handle this
            else:
                # Use environment-aware cache directory
                cache_dir = get_default_cache_dir()
                default_ensemble_path = str(cache_dir / "trained" / "chirpkit-ensemble")
                default_birdnet_path = None
            
            logger.info(f"Default ensemble path: {default_ensemble_path}")
            return default_ensemble_path, default_birdnet_path
        
        # No models found and fallback disabled
        raise RuntimeError(
            "No model paths could be resolved. Please either:\n"
            "1. Specify explicit paths (birdnet_model_path, ensemble_path)\n"
            "2. Set a root directory (model_root) containing models\n"
            "3. Enable auto_download=True to download models automatically\n"
            "4. Set fallback_to_default=True to use default locations"
        )

    def is_available(self) -> bool:
        """
        Check if the classifier is available and ready to use.

        Returns:
            bool: True if classifier is initialized and models are loaded
        """
        return self.is_initialized

    @requires_torch
    async def initialize(self):
        """Initialize the ensemble classifier"""
        if self.is_initialized:
            return

        logger.info(f"🚀 Initializing ChirpKit v{__version__} (Model v{__model_version__})...")

        self.torch = DependencyManager.get_torch()
        if self.torch is None:
            raise RuntimeError("PyTorch not available - ChirpKit requires PyTorch")

        # Set device
        self.device = self._get_device(self.torch)
        logger.info(f"🖥️  Using device: {self.device}")

        # Load ensemble model
        model_loaded = await self._load_ensemble()

        if not model_loaded:
            raise RuntimeError(f"Failed to load ChirpKit ensemble model v{__model_version__}")

        # Load species cache for Wikipedia data
        await self._load_species_cache()
        logger.info(f"📚 Loaded species cache with {len(self.species_cache)} entries")

        self.is_initialized = True
        logger.info(f"✅ ChirpKit initialized ({self.n_classes} species, {self.mode} mode)")

    async def _load_ensemble(self):
        """Load ensemble classifier and BirdNET extractor"""
        try:
            # Import models and transfer_learning from chirpkit package
            from .models.chirpkit_ensemble import ChirpKitEnsembleClassifier
            from .transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor

            logger.info("📦 Loading BirdNET embedding extractor...")
            # Pass resolved BirdNET path if available
            birdnet_path = self.birdnet_path if hasattr(self, 'birdnet_path') else None
            self.birdnet_extractor = BirdNETEmbeddingExtractor(
                model_path=birdnet_path,
                auto_download=self.config.auto_download
            )

            logger.info(f"📦 Loading ChirpKit ensemble ({self.mode} mode)...")
            self.ensemble_classifier = ChirpKitEnsembleClassifier(
                model_dir=self.model_path,
                mode=self.mode,
                device=self.device,
                auto_download=self.config.auto_download
            )
            self.ensemble_classifier.load_models()

            # Update class info
            self.n_classes = self.ensemble_classifier.n_classes
            self.label_encoder = self.ensemble_classifier.label_encoder
            self.species_labels = list(self.label_encoder.classes_)

            logger.info(f"✅ Ensemble loaded: {self.n_classes} species, {self.ensemble_classifier.ensemble_info['num_models']} models")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load ensemble model v{__model_version__}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_device(self, torch):
        """Determine the best available device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    async def classify(self, audio_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Classify insect species from audio file

        Args:
            audio_path: Path to audio file
            top_k: Number of top predictions to return

        Returns:
            Dictionary with classification results
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # Extract BirdNET embedding
            logger.info(f"🔍 Extracting embeddings from {audio_path}")
            embedding = self.birdnet_extractor.extract_embeddings_from_audio(
                audio_path,
                aggregate='mean'
            )

            # Get ensemble prediction
            logger.info("🎯 Running ensemble prediction...")
            result = self.ensemble_classifier.predict(embedding, top_k=top_k)

            # Enrich with Wikipedia data if enabled
            if self.enable_enrichment:
                await self._enrich_predictions(result['predictions'])

            # Format response
            return {
                'success': True,
                'species': result['top_prediction']['species'],
                'scientific_name': result['top_prediction']['scientific_name'],
                'confidence': result['top_prediction']['confidence'],
                'top_predictions': [
                    {
                        'rank': p['rank'],
                        'species': p['species'],
                        'scientific_name': p['scientific_name'],
                        'confidence': p['confidence'],
                        'common_name': p.get('common_name', p['scientific_name']),
                        'description': p.get('description', ''),
                        'image_url': p.get('image_url', ''),
                        'wikipedia_url': p.get('wikipedia_url', '')
                    }
                    for p in result['predictions']
                ],
                'model_info': {
                    'package_version': __version__,
                    'model_version': __model_version__,
                    'mode': result['mode'],
                    'num_models': result['num_models'],
                    'tta_rounds': result.get('tta_rounds', 0),
                    'total_species': self.n_classes
                }
            }

        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'species': None,
                'confidence': 0.0
            }

    async def _enrich_predictions(self, predictions: List[Dict]):
        """Enrich predictions with Wikipedia data"""
        for pred in predictions:
            species_name = pred['species']
            if species_name in self.species_cache:
                info = self.species_cache[species_name]
                pred['common_name'] = info.get('common_name', pred['scientific_name'])
                pred['description'] = info.get('description', '')
                pred['image_url'] = info.get('image_url', '')
                pred['wikipedia_url'] = info.get('wikipedia_url', '')
            else:
                # Fetch from Wikipedia
                info = await self._fetch_species_info(species_name)
                pred['common_name'] = info.get('common_name', pred['scientific_name'])
                pred['description'] = info.get('description', '')
                pred['image_url'] = info.get('image_url', '')
                pred['wikipedia_url'] = info.get('wikipedia_url', '')

    async def _fetch_species_info(self, scientific_name: str) -> Dict[str, str]:
        """Fetch species info from Wikipedia"""
        if scientific_name in self.species_cache:
            return self.species_cache[scientific_name]

        search_name = scientific_name.replace('_', ' ')

        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                self.executor,
                self._fetch_species_info_sync,
                search_name
            )

            self.species_cache[scientific_name] = info
            await self._save_species_cache()
            return info

        except Exception as e:
            logger.debug(f"Could not fetch Wikipedia info for {scientific_name}: {e}")
            return {
                'common_name': search_name,
                'description': '',
                'image_url': '',
                'wikipedia_url': ''
            }

    def _fetch_species_info_sync(self, search_name: str) -> Dict[str, str]:
        """Synchronous Wikipedia fetch (runs in thread pool)"""
        try:
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(search_name)}"
            headers = {
                'User-Agent': f'ChirpKit/{__version__} (https://github.com/prossm/chirpkit; contact@chirpkit.ai)'
            }
            response = requests.get(search_url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                common_name = data.get('title', search_name)
                description = data.get('extract', '')
                image_url = data.get('thumbnail', {}).get('source', '')

                # Try to extract common name from description
                if description and ',' in description:
                    potential_common = description.split(',')[0].strip()
                    if len(potential_common) < 50 and not potential_common.startswith('The'):
                        common_name = potential_common

                return {
                    'common_name': common_name,
                    'description': description[:200] + '...' if len(description) > 200 else description,
                    'image_url': image_url,
                    'wikipedia_url': f"https://en.wikipedia.org/wiki/{urllib.parse.quote(search_name)}"
                }

        except Exception as e:
            logger.debug(f"Wikipedia fetch failed: {e}")

        return {
            'common_name': search_name,
            'description': '',
            'image_url': '',
            'wikipedia_url': ''
        }

    async def _load_species_cache(self):
        """Load species info cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.species_cache = json.load(f)
            except Exception as e:
                logger.debug(f"Could not load species cache: {e}")
                self.species_cache = {}
        else:
            self.species_cache = {}

    async def _save_species_cache(self):
        """Save species info cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.species_cache, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save species cache: {e}")

    def load_model(self) -> bool:
        """
        Synchronous wrapper for initialize() for backward compatibility

        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to get existing loop
            try:
                loop = asyncio.get_running_loop()
                # Loop is running - can't block here
                logger.warning("Event loop already running - use 'await classifier.initialize()' instead")
                # Schedule initialization but don't wait
                asyncio.create_task(self.initialize())
                return True
            except RuntimeError:
                # No running loop - create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.initialize())
                    return True
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, audio_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Synchronous wrapper for classify() for backward compatibility

        Args:
            audio_path: Path to audio file
            top_k: Number of top predictions

        Returns:
            Classification results
        """
        try:
            # Try to get existing loop
            try:
                loop = asyncio.get_running_loop()
                # Loop is running - can't block here
                raise RuntimeError("Use 'await classifier.classify()' in async context, not 'classifier.predict()'")
            except RuntimeError as e:
                if "async context" in str(e):
                    raise
                # No running loop - create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.classify(audio_path, top_k))
                finally:
                    loop.close()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'species': None,
                'confidence': 0.0
            }
