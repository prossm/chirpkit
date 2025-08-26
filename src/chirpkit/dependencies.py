"""
Dependency management utilities for graceful handling of optional imports.
Provides fallback mechanisms and helpful error messages for missing dependencies.
"""

import logging
import importlib
import functools
from typing import Optional, Any, Callable, Dict
import warnings

logger = logging.getLogger(__name__)


class DependencyManager:
    """Manages optional dependencies with graceful fallbacks and helpful error messages."""
    
    _import_cache: Dict[str, Any] = {}
    _failed_imports: Dict[str, str] = {}
    
    @classmethod
    def get_tensorflow(cls) -> Optional[Any]:
        """Import TensorFlow with graceful fallback handling."""
        if 'tensorflow' in cls._import_cache:
            return cls._import_cache['tensorflow']
        
        try:
            import tensorflow as tf
            
            # Check for corrupted installation
            if not hasattr(tf, '__version__'):
                error_msg = (
                    "Corrupted TensorFlow installation detected. "
                    "Run: chirpkit-fix to repair the installation"
                )
                cls._failed_imports['tensorflow'] = error_msg
                logger.error(error_msg)
                return None
            
            # Test basic functionality
            try:
                tf.constant([1, 2, 3])
            except Exception as e:
                error_msg = (
                    f"TensorFlow runtime error: {e}. "
                    "Run: chirpkit-fix to repair the installation"
                )
                cls._failed_imports['tensorflow'] = error_msg
                logger.error(error_msg)
                return None
            
            cls._import_cache['tensorflow'] = tf
            logger.info(f"TensorFlow {tf.__version__} loaded successfully")
            return tf
            
        except ImportError as e:
            error_msg = (
                "TensorFlow not installed. Install with: "
                "pip install chirpkit[tensorflow-macos] (macOS) or "
                "pip install chirpkit[tensorflow] (Linux/Windows)"
            )
            cls._failed_imports['tensorflow'] = error_msg
            logger.warning(error_msg)
            return None
    
    @classmethod
    def get_torch(cls) -> Optional[Any]:
        """Import PyTorch with graceful fallback handling."""
        if 'torch' in cls._import_cache:
            return cls._import_cache['torch']
        
        try:
            import torch
            
            # Test basic functionality
            try:
                torch.tensor([1, 2, 3])
            except Exception as e:
                error_msg = f"PyTorch runtime error: {e}"
                cls._failed_imports['torch'] = error_msg
                logger.error(error_msg)
                return None
            
            cls._import_cache['torch'] = torch
            logger.info(f"PyTorch {torch.__version__} loaded successfully")
            return torch
            
        except ImportError:
            error_msg = (
                "PyTorch not installed. Install with: "
                "pip install chirpkit[torch] or pip install torch torchvision torchaudio"
            )
            cls._failed_imports['torch'] = error_msg
            logger.warning(error_msg)
            return None
    
    @classmethod
    def get_gradio(cls) -> Optional[Any]:
        """Import Gradio with graceful fallback handling."""
        if 'gradio' in cls._import_cache:
            return cls._import_cache['gradio']
        
        try:
            import gradio as gr
            cls._import_cache['gradio'] = gr
            return gr
        except ImportError:
            error_msg = "Gradio not installed. Install with: pip install chirpkit[ui]"
            cls._failed_imports['gradio'] = error_msg
            logger.warning(error_msg)
            return None
    
    @classmethod
    def get_wandb(cls) -> Optional[Any]:
        """Import Weights & Biases with graceful fallback handling."""
        if 'wandb' in cls._import_cache:
            return cls._import_cache['wandb']
        
        try:
            import wandb
            cls._import_cache['wandb'] = wandb
            return wandb
        except ImportError:
            error_msg = "Weights & Biases not installed. Install with: pip install chirpkit[viz]"
            cls._failed_imports['wandb'] = error_msg
            logger.warning(error_msg)
            return None
    
    @classmethod
    def check_required_dependencies(cls, required: list) -> bool:
        """Check if all required dependencies are available."""
        missing = []
        for dep in required:
            method_name = f"get_{dep.replace('-', '_')}"
            if hasattr(cls, method_name):
                if getattr(cls, method_name)() is None:
                    missing.append(dep)
            else:
                logger.warning(f"Unknown dependency check: {dep}")
        
        if missing:
            logger.error(f"Missing required dependencies: {', '.join(missing)}")
            return False
        return True
    
    @classmethod
    def get_failed_imports(cls) -> Dict[str, str]:
        """Get information about failed imports."""
        return cls._failed_imports.copy()


def requires_tensorflow(func: Callable) -> Callable:
    """Decorator to check TensorFlow availability before function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tf = DependencyManager.get_tensorflow()
        if tf is None:
            failed_msg = DependencyManager.get_failed_imports().get('tensorflow', 'TensorFlow not available')
            raise RuntimeError(f"TensorFlow required but not available: {failed_msg}")
        return func(*args, **kwargs)
    return wrapper


def requires_torch(func: Callable) -> Callable:
    """Decorator to check PyTorch availability before function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch = DependencyManager.get_torch()
        if torch is None:
            failed_msg = DependencyManager.get_failed_imports().get('torch', 'PyTorch not available')
            raise RuntimeError(f"PyTorch required but not available: {failed_msg}")
        return func(*args, **kwargs)
    return wrapper


def requires_ui(func: Callable) -> Callable:
    """Decorator to check UI dependencies (Gradio) availability."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        gr = DependencyManager.get_gradio()
        if gr is None:
            failed_msg = DependencyManager.get_failed_imports().get('gradio', 'Gradio not available')
            raise RuntimeError(f"UI components required but not available: {failed_msg}")
        return func(*args, **kwargs)
    return wrapper


def optional_import(module_name: str, fallback_message: Optional[str] = None):
    """
    Import a module optionally, with a helpful error message if not available.
    
    Args:
        module_name: Name of the module to import
        fallback_message: Custom message to show if import fails
    
    Returns:
        The imported module or None if import failed
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        if fallback_message:
            logger.warning(fallback_message)
        else:
            logger.warning(f"Optional dependency '{module_name}' not available: {e}")
        return None


def warn_about_missing_gpu():
    """Warn user about missing GPU acceleration."""
    tf = DependencyManager.get_tensorflow()
    if tf is not None:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                warnings.warn(
                    "No GPU devices found. Computations will run on CPU. "
                    "For better performance, ensure CUDA (Linux/Windows) or "
                    "Metal (macOS) acceleration is properly configured.",
                    UserWarning
                )
            else:
                logger.info(f"Found {len(gpus)} GPU device(s) for TensorFlow")
        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}")
    
    torch = DependencyManager.get_torch()
    if torch is not None:
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA available with {torch.cuda.device_count()} device(s)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("PyTorch MPS (Apple Silicon) backend available")
        else:
            warnings.warn(
                "No GPU acceleration available for PyTorch. "
                "Computations will run on CPU.",
                UserWarning
            )


# Convenience functions for common imports
def get_tensorflow():
    """Get TensorFlow module if available."""
    return DependencyManager.get_tensorflow()


def get_torch():
    """Get PyTorch module if available."""
    return DependencyManager.get_torch()


def get_gradio():
    """Get Gradio module if available."""
    return DependencyManager.get_gradio()


def get_wandb():
    """Get Weights & Biases module if available."""
    return DependencyManager.get_wandb()