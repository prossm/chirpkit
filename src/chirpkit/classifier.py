"""
ChirpKit Insect Classifier - v6.0 Ensemble

Neural network-based insect sound identification using BirdNET embeddings
and a 7-model ensemble for robust predictions.
"""

import logging
import numpy as np
import json
import asyncio
import requests
import urllib.parse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from .dependencies import DependencyManager, requires_torch
from ._version import __version__

logger = logging.getLogger(__name__)

class InsectClassifier:
    """Neural network-based insect sound classifier using v6.0 ensemble"""

    def __init__(self, model_path: Optional[str] = None, mode: str = "ensemble_tta"):
        """
        Initialize the InsectClassifier with v6.0 ensemble

        Args:
            model_path: Optional path to ensemble model directory.
                       Defaults to "models/trained/chirpkit-ensemble"
            mode: Deployment mode - 'single', 'ensemble', or 'ensemble_tta' (default)
        """
        self.model_path = model_path or "models/trained/chirpkit-ensemble"
        self.mode = mode
        self.is_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Will be set during initialization
        self.ensemble_classifier = None
        self.birdnet_extractor = None
        self.species_labels = []
        self.label_encoder = None
        self.n_classes = 231  # v6.0 ensemble
        self.torch = None
        self.device = None

        # Wikipedia integration
        self.species_cache = {}
        self.cache_file = Path("species_cache.json")
        self.enable_enrichment = True

    @requires_torch
    async def initialize(self):
        """Initialize the v6.0 ensemble classifier"""
        if self.is_initialized:
            return

        logger.info("🚀 Initializing ChirpKit v6.0 Ensemble...")

        self.torch = DependencyManager.get_torch()
        if self.torch is None:
            raise RuntimeError("PyTorch not available - ChirpKit requires PyTorch")

        # Set device
        self.device = self._get_device(self.torch)
        logger.info(f"🖥️  Using device: {self.device}")

        # Load v6.0 ensemble
        model_loaded = await self._load_ensemble()

        if not model_loaded:
            raise RuntimeError("Failed to load ChirpKit v6.0 ensemble")

        # Load species cache for Wikipedia data
        await self._load_species_cache()
        logger.info(f"📚 Loaded species cache with {len(self.species_cache)} entries")

        self.is_initialized = True
        logger.info(f"✅ ChirpKit v6.0 initialized ({self.n_classes} species, {self.mode} mode)")

    async def _load_ensemble(self):
        """Load v6.0 ensemble classifier and BirdNET extractor"""
        try:
            from models.chirpkit_ensemble import ChirpKitEnsembleClassifier
            from transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor

            logger.info("📦 Loading BirdNET embedding extractor...")
            self.birdnet_extractor = BirdNETEmbeddingExtractor()

            logger.info(f"📦 Loading ChirpKit ensemble ({self.mode} mode)...")
            self.ensemble_classifier = ChirpKitEnsembleClassifier(
                model_dir=self.model_path,
                mode=self.mode,
                device=self.device
            )
            self.ensemble_classifier.load_models()

            # Update class info
            self.n_classes = self.ensemble_classifier.n_classes
            self.label_encoder = self.ensemble_classifier.label_encoder
            self.species_labels = list(self.label_encoder.classes_)

            logger.info(f"✅ Ensemble loaded: {self.n_classes} species, {self.ensemble_classifier.ensemble_info['num_models']} models")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load v6.0 ensemble: {e}")
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
                    'version': '6.0',
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
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, schedule initialization
                future = asyncio.ensure_future(self.initialize())
                return True
            else:
                # Run in new event loop
                loop.run_until_complete(self.initialize())
                return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create task
                future = asyncio.ensure_future(self.classify(audio_path, top_k))
                # This is tricky - we can't block in an async context
                # Return a pending result that the caller needs to await
                raise RuntimeError("Use await classify() in async context")
            else:
                # Run in event loop
                return loop.run_until_complete(self.classify(audio_path, top_k))
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'species': None,
                'confidence': 0.0
            }
