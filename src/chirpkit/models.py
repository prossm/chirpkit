"""
Model management utilities for ChirpKit
"""

import logging
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Tuple
from .dependencies import DependencyManager

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ChirpKit model downloading, loading, and caching"""
    
    DEFAULT_MODEL_DIR = Path("models/trained")
    REMOTE_MODEL_BASE_URL = "https://github.com/patrickmetzger/chirpkit/releases/download/v0.1.0/"
    
    DEFAULT_MODELS = {
        "471species": {
            "model_file": "insect_classifier_471species.pth",
            "encoder_file": "insect_classifier_471species_label_encoder.joblib", 
            "info_file": "insect_classifier_471species_info.json",
            "description": "471 insect species classifier (71.6% accuracy)",
            "size_mb": 17.2,
            "n_classes": 471
        }
    }
    
    @classmethod
    def get_default_model_path(cls, model_name: str = "471species") -> Optional[Tuple[Path, Path, Path]]:
        """
        Get paths to default pre-trained model files
        
        Args:
            model_name: Name of the model to get paths for
            
        Returns:
            Tuple of (model_path, encoder_path, info_path) or None if not found
        """
        if model_name not in cls.DEFAULT_MODELS:
            logger.error(f"Unknown model: {model_name}. Available: {list(cls.DEFAULT_MODELS.keys())}")
            return None
            
        model_info = cls.DEFAULT_MODELS[model_name]
        
        model_path = cls.DEFAULT_MODEL_DIR / model_info["model_file"]
        encoder_path = cls.DEFAULT_MODEL_DIR / model_info["encoder_file"]  
        info_path = cls.DEFAULT_MODEL_DIR / model_info["info_file"]
        
        # Check if all files exist
        if all(p.exists() for p in [model_path, encoder_path, info_path]):
            return model_path, encoder_path, info_path
        else:
            missing = [str(p) for p in [model_path, encoder_path, info_path] if not p.exists()]
            logger.warning(f"Missing model files: {missing}")
            return None

    @classmethod
    def find_any_model(cls) -> Optional[Tuple[Path, Path, Path]]:
        """
        Find the specific 471-species model and its related files
        
        Returns:
            Tuple of (model_path, encoder_path, info_path) or None if not found
        """
        # Specific model we're looking for
        target_model_name = "insect_classifier_471species"
        
        # Search in a few key locations
        search_paths = [
            cls.DEFAULT_MODEL_DIR,  # models/trained/
            Path("models") / "trained",
            Path(".") / "models" / "trained",
        ]
        
        for search_path in search_paths:
            try:
                if not search_path.exists():
                    logger.debug(f"Search path does not exist: {search_path}")
                    continue
                    
                # Look specifically for the 471-species model
                model_path = search_path / f"{target_model_name}.pth"
                encoder_path = search_path / f"{target_model_name}_label_encoder.joblib"
                info_path = search_path / f"{target_model_name}_info.json"
                
                if model_path.exists() and encoder_path.exists():
                    logger.info(f"✅ Found 471-species model in {search_path}")
                    logger.info(f"   Model: {model_path}")
                    logger.info(f"   Encoder: {encoder_path}")
                    
                    # Create info file if missing
                    if not info_path.exists():
                        logger.info("Creating missing info file...")
                        info_path = cls._create_minimal_info_file(model_path, encoder_path)
                    else:
                        logger.info(f"   Info: {info_path}")
                    
                    return model_path, encoder_path, info_path
                    
            except Exception as e:
                logger.debug(f"Error searching {search_path}: {e}")
                continue
        
        logger.warning(f"❌ Could not find {target_model_name} model in any search location")
        logger.info("Expected files:")
        logger.info(f"  - {target_model_name}.pth")
        logger.info(f"  - {target_model_name}_label_encoder.joblib")
        logger.info(f"  - {target_model_name}_info.json (optional)")
        
        return None

    @classmethod
    def _create_minimal_info_file(cls, model_path: Path, encoder_path: Path) -> Path:
        """Create a minimal info file for a model"""
        try:
            import joblib
            encoder = joblib.load(encoder_path)
            n_classes = len(encoder.classes_)
            species_list = list(encoder.classes_)
        except Exception as e:
            logger.warning(f"Could not load encoder: {e}, using defaults")
            n_classes = 471
            species_list = []
            
        info = {
            "model_name": model_path.stem,
            "n_classes": n_classes,
            "species_list": species_list,
            "created_by": "chirpkit_model_manager",
            "notes": "Auto-generated info file"
        }
        
        info_path = model_path.parent / f"{model_path.stem}_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        logger.info(f"Created minimal info file: {info_path}")
        return info_path

    @classmethod
    async def download_pretrained_model(cls, model_name: str = "471species", force: bool = False) -> Optional[Path]:
        """
        Download pre-trained model if not available locally
        
        Args:
            model_name: Name of the model to download
            force: Whether to force re-download even if files exist
            
        Returns:
            Path to downloaded model file or None if failed
        """
        if model_name not in cls.DEFAULT_MODELS:
            logger.error(f"Unknown model: {model_name}")
            return None
            
        model_info = cls.DEFAULT_MODELS[model_name]
        model_dir = cls.DEFAULT_MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists and not forcing
        if not force:
            existing = cls.get_default_model_path(model_name)
            if existing:
                logger.info(f"Model {model_name} already exists")
                return existing[0]
        
        logger.info(f"Downloading model {model_name} (~{model_info['size_mb']}MB)...")
        
        # Download all required files
        files_to_download = [
            model_info["model_file"],
            model_info["encoder_file"], 
            model_info["info_file"]
        ]
        
        downloaded_files = []
        
        for filename in files_to_download:
            url = cls.REMOTE_MODEL_BASE_URL + filename
            local_path = model_dir / filename
            
            try:
                logger.info(f"Downloading {filename}...")
                await cls._download_file(url, local_path)
                downloaded_files.append(local_path)
                logger.info(f"✓ Downloaded {filename}")
                
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
                # Clean up partial downloads
                for downloaded in downloaded_files:
                    if downloaded.exists():
                        downloaded.unlink()
                return None
                
        logger.info(f"✅ Successfully downloaded model {model_name}")
        return model_dir / model_info["model_file"]

    @classmethod
    async def _download_file(cls, url: str, local_path: Path) -> None:
        """Download a file from URL to local path"""
        import aiohttp
        import aiofiles
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {url}")
                
                async with aiofiles.open(local_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)

    @classmethod
    def get_model_info(cls, model_path: Path) -> Dict:
        """Load model information from info file"""
        info_path = model_path.parent / f"{model_path.stem}_info.json"
        
        if not info_path.exists():
            logger.warning(f"No info file found for {model_path}")
            return {
                "model_name": model_path.stem,
                "n_classes": "unknown",
                "notes": "No info file available"
            }
            
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model info: {e}")
            return {
                "model_name": model_path.stem,
                "error": str(e)
            }

    @classmethod
    def list_available_models(cls) -> Dict[str, Dict]:
        """List all available models (local and remote)"""
        models = {}
        
        # Add remote models
        for name, info in cls.DEFAULT_MODELS.items():
            local_paths = cls.get_default_model_path(name)
            models[name] = {
                **info,
                "status": "available_locally" if local_paths else "available_remote",
                "local_paths": local_paths
            }
            
        # Add any other local models
        if cls.DEFAULT_MODEL_DIR.exists():
            local_models = list(cls.DEFAULT_MODEL_DIR.glob("*.pth"))
            for model_path in local_models:
                model_name = model_path.stem
                if model_name not in models:
                    models[model_name] = {
                        "model_file": model_path.name,
                        "description": "Local model",
                        "status": "local_only",
                        "n_classes": "unknown"
                    }
                    
        return models

    @classmethod
    def validate_model_files(cls, model_path: Path, encoder_path: Path, info_path: Path) -> bool:
        """Validate that model files are complete and loadable"""
        try:
            # Check file existence
            if not all(p.exists() for p in [model_path, encoder_path, info_path]):
                logger.error("Some model files are missing")
                return False
                
            # Try to load encoder
            import joblib
            encoder = joblib.load(encoder_path)
            logger.info(f"Encoder loaded: {len(encoder.classes_)} classes")
            
            # Try to load info
            with open(info_path, 'r') as f:
                info = json.load(f)
            logger.info(f"Info loaded: {info.get('n_classes', 'unknown')} classes")
            
            # Basic consistency check
            if hasattr(encoder, 'classes_') and 'n_classes' in info:
                if len(encoder.classes_) != info['n_classes']:
                    logger.warning("Mismatch between encoder and info file class counts")
                    
            # Try to load model structure (without full load)
            torch = DependencyManager.get_torch()
            if torch:
                try:
                    # Just check if the file can be opened
                    checkpoint = torch.load(model_path, map_location='cpu')
                    logger.info("Model file appears valid")
                except Exception as e:
                    logger.error(f"Model file validation failed: {e}")
                    return False
            else:
                logger.warning("PyTorch not available, skipping model file validation")
                
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False


# Convenience functions
def get_default_model_path(model_name: str = "471species") -> Optional[Tuple[Path, Path, Path]]:
    """Get path to default pre-trained model"""
    return ModelManager.get_default_model_path(model_name)


def find_any_model() -> Optional[Tuple[Path, Path, Path]]:
    """Find any available trained model"""
    return ModelManager.find_any_model()


async def download_pretrained_model(model_name: str = "471species") -> Optional[Path]:
    """Download pre-trained model if not available locally"""
    return await ModelManager.download_pretrained_model(model_name)


def list_models() -> Dict[str, Dict]:
    """List available models"""
    return ModelManager.list_available_models()