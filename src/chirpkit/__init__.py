"""
ChirpKit - A robust toolkit for insect sound classification and analysis.

This package provides tools for:
- Audio preprocessing and feature extraction
- Machine learning model training and inference
- Species classification from audio recordings
- Installation diagnostics and environment setup

The package is designed to work across different ML backends (TensorFlow, PyTorch)
with graceful fallback handling for missing dependencies.
"""

from .dependencies import (
    get_tensorflow,
    get_torch, 
    get_gradio,
    get_wandb,
    DependencyManager,
    requires_tensorflow,
    requires_torch,
    requires_ui,
    warn_about_missing_gpu
)

try:
    from .classifier import InsectClassifier
    from .models import ModelManager, find_any_model, list_models
    from .cli import classify_audio_file, get_classifier_instance
except ImportError as e:
    # Classifier may not be available if ML dependencies are missing
    logger.warning(f"Some ChirpKit modules not available: {e}")
    InsectClassifier = None
    ModelManager = None
    find_any_model = None
    list_models = None
    classify_audio_file = None
    get_classifier_instance = None

__version__ = "0.1.0"
__author__ = "Patrick Metzger"

# Check core dependencies on import
import logging
logger = logging.getLogger(__name__)

# Perform basic dependency validation
_missing_deps = []

try:
    import numpy as np
    import librosa
    import soundfile as sf
    import sklearn
except ImportError as e:
    _missing_deps.append(str(e))

if _missing_deps:
    logger.warning(
        f"Some core dependencies are missing: {', '.join(_missing_deps)}. "
        "Run 'chirpkit-doctor' to check installation health."
    )

# Optional GPU availability check
try:
    warn_about_missing_gpu()
except Exception:
    # Silently ignore GPU check failures
    pass

__all__ = [
    'get_tensorflow',
    'get_torch',
    'get_gradio', 
    'get_wandb',
    'DependencyManager',
    'requires_tensorflow',
    'requires_torch',
    'requires_ui',
    'warn_about_missing_gpu',
    'InsectClassifier',
    'ModelManager',
    'find_any_model',
    'list_models',
    'classify_audio_file',
    'get_classifier_instance'
]