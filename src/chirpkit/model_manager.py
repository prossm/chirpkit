"""
Model management utilities for ChirpKit v6.0
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages ChirpKit v6.0 ensemble model paths"""

    DEFAULT_MODEL_DIR = Path("models/trained/chirpkit-ensemble")

    @classmethod
    def get_default_model_path(cls) -> Path:
        """
        Get path to v6.0 ensemble model directory

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
    Use v6.0 ensemble instead

    Returns:
        None (v6.0 uses ensemble, not single model files)
    """
    logger.info("ChirpKit v6.0 uses ensemble models")
    logger.info("Old single-model format no longer supported")
    return None


def list_models() -> list:
    """
    List available models

    Returns:
        List with v6.0 ensemble info
    """
    ensemble_dir = ModelManager.get_default_model_path()

    models = []
    if ModelManager.verify_ensemble_files(ensemble_dir):
        models.append({
            'name': 'chirpkit-v6.0-ensemble',
            'path': str(ensemble_dir),
            'type': 'ensemble',
            'num_models': 7,
            'species': 231,
            'accuracy': '79.7%'
        })
    else:
        logger.info("v6.0 ensemble not found - will auto-download on first use")

    return models
