"""
Model management utilities for ChirpKit
"""

from pathlib import Path
from typing import Optional, Tuple
from ._version import __version__, __model_version__
from .utils import get_chirpkit_logger

logger = get_chirpkit_logger(__name__)

class ModelManager:
    """Manages ChirpKit ensemble model paths"""

    DEFAULT_MODEL_DIR = Path("models/trained/chirpkit-ensemble")

    @classmethod
    def get_default_model_path(cls) -> Path:
        """
        Get path to ensemble model directory

        Returns:
            Path to ensemble model directory
        """
        return cls.DEFAULT_MODEL_DIR

    @classmethod
    def verify_ensemble_files(cls, model_dir: Optional[Path] = None) -> bool:
        """
        Verify that all required ensemble files exist

        Args:
            model_dir: Directory to check. Defaults to DEFAULT_MODEL_DIR

        Returns:
            True if all files present, False otherwise
        """
        if model_dir is None:
            model_dir = cls.DEFAULT_MODEL_DIR

        required_files = [
            'ensemble_info.json',
            'ensemble_model_1.pth',
            'ensemble_model_2.pth',
            'ensemble_model_3.pth',
            'ensemble_model_4.pth',
            'ensemble_model_5.pth',
            'ensemble_model_6.pth',
            'ensemble_model_7.pth',
        ]

        for filename in required_files:
            file_path = model_dir / filename
            if not file_path.exists():
                logger.warning(f"Missing ensemble file: {file_path}")
                return False

        return True


def find_any_model() -> Optional[Tuple[Path, Path, Path]]:
    """
    Legacy function - returns None as old models are no longer supported
    Use ensemble model instead

    Returns:
        None (ensemble model uses directory structure, not single files)
    """
    logger.info(f"ChirpKit v{__version__} uses ensemble models (model v{__model_version__})")
    logger.info("Old single-model format no longer supported")
    return None


def list_models() -> list:
    """
    List available models

    Returns:
        List with ensemble model info
    """
    ensemble_dir = ModelManager.get_default_model_path()

    models = []
    if ModelManager.verify_ensemble_files(ensemble_dir):
        models.append({
            'name': f'chirpkit-ensemble-v{__model_version__}',
            'path': str(ensemble_dir),
            'type': 'ensemble',
            'num_models': 7,
            'species': 231,
            'accuracy': '79.7%',
            'package_version': __version__,
            'model_version': __model_version__
        })
    else:
        logger.info(f"Ensemble model v{__model_version__} not found - will auto-download on first use")

    return models
