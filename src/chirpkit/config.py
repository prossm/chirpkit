"""
ChirpKit Configuration Management

Provides flexible model configuration with support for:
- Constructor-level parameters
- Configuration files (YAML/JSON)
- Environment variables
- Model discovery and validation
- Backwards compatibility
"""

import os
import yaml
import json
import glob
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


class ModelCompatibilityError(Exception):
    """Raised when model compatibility validation fails"""
    pass


@dataclass
class ModelConfiguration:
    """
    Comprehensive model configuration for ChirpKit
    
    Supports multiple ways to specify model paths and settings:
    1. Explicit paths for full control
    2. Root directory for automatic discovery
    3. Configuration files
    4. Environment variables
    """
    
    # Root directory containing all models
    root_directory: Optional[str] = None
    
    # Specific model paths (take precedence over root_directory)
    birdnet_model_path: Optional[str] = None
    birdnet_labels_path: Optional[str] = None
    ensemble_path: Optional[str] = None
    
    # Deployment settings
    mode: str = "ensemble_tta"  # single, ensemble, ensemble_tta
    auto_download: bool = True
    validate_compatibility: bool = True
    fallback_to_default: bool = True
    
    # Advanced settings
    tta_rounds: int = 10
    tta_noise_std: float = 0.01
    device: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization"""
        if self.mode not in ["single", "ensemble", "ensemble_tta"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of: single, ensemble, ensemble_tta")
        
        # Convert string paths to Path objects for easier handling
        if self.root_directory:
            self.root_directory = str(Path(self.root_directory).expanduser().absolute())
        if self.birdnet_model_path:
            self.birdnet_model_path = str(Path(self.birdnet_model_path).expanduser().absolute())
        if self.birdnet_labels_path:
            self.birdnet_labels_path = str(Path(self.birdnet_labels_path).expanduser().absolute())
        if self.ensemble_path:
            self.ensemble_path = str(Path(self.ensemble_path).expanduser().absolute())


class ConfigurationManager:
    """
    Manages ChirpKit configuration resolution and validation
    
    Resolution priority (highest to lowest):
    1. Explicit constructor parameters
    2. Environment variables
    3. Configuration file
    4. Defaults
    """
    
    DEFAULT_CONFIG_PATHS = [
        "~/.chirpkit/config.yaml",
        "~/.chirpkit/config.json", 
        "./chirpkit.config.yaml",
        "./chirpkit.config.json"
    ]
    
    def __init__(self, user_config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager
        
        Args:
            user_config: User-provided configuration dictionary
        """
        self.user_config = user_config or {}
        self._resolved_config = None
        
    def resolve_configuration(self) -> ModelConfiguration:
        """
        Resolve final configuration from all sources
        
        Returns:
            ModelConfiguration: Resolved configuration object
        """
        if self._resolved_config is not None:
            return self._resolved_config
            
        # Start with defaults
        config = self._load_default_config()
        
        # Layer on configuration file
        file_config = self._load_config_file()
        config.update(file_config)
        
        # Layer on environment variables
        env_config = self._load_env_config()
        config.update(env_config)
        
        # Layer on user-provided config (highest priority)
        config.update(self.user_config)
        
        # Create and validate configuration object
        self._resolved_config = ModelConfiguration(**config)
        
        logger.debug(f"Resolved configuration: {asdict(self._resolved_config)}")
        return self._resolved_config
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'mode': 'ensemble_tta',
            'auto_download': True,
            'validate_compatibility': True,
            'fallback_to_default': True,
            'tta_rounds': 10,
            'tta_noise_std': 0.01
        }
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from file"""
        for config_path in self.DEFAULT_CONFIG_PATHS:
            expanded_path = Path(config_path).expanduser()
            if expanded_path.exists():
                logger.info(f"Loading configuration from {expanded_path}")
                return self._parse_config_file(expanded_path)
        
        logger.debug("No configuration file found")
        return {}
    
    def _parse_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Parse YAML or JSON configuration file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f) or {}
                else:
                    data = json.load(f) or {}
            
            # Handle nested structure (models: key)
            if 'models' in data:
                models_config = data['models']
                
                # Extract model-specific settings
                config = {}
                
                if 'root_directory' in models_config:
                    config['root_directory'] = models_config['root_directory']
                    
                if 'birdnet' in models_config:
                    birdnet_config = models_config['birdnet']
                    if 'model_path' in birdnet_config:
                        if birdnet_config['model_path'].startswith('/'):
                            # Absolute path
                            config['birdnet_model_path'] = birdnet_config['model_path']
                        else:
                            # Relative to root_directory
                            root_dir = models_config.get('root_directory', '')
                            config['birdnet_model_path'] = str(Path(root_dir) / birdnet_config['model_path'])
                    if 'labels_path' in birdnet_config:
                        if birdnet_config['labels_path'].startswith('/'):
                            config['birdnet_labels_path'] = birdnet_config['labels_path']
                        else:
                            root_dir = models_config.get('root_directory', '')
                            config['birdnet_labels_path'] = str(Path(root_dir) / birdnet_config['labels_path'])
                
                if 'ensemble' in models_config:
                    ensemble_config = models_config['ensemble']
                    if 'path' in ensemble_config:
                        if ensemble_config['path'].startswith('/'):
                            config['ensemble_path'] = ensemble_config['path']
                        else:
                            root_dir = models_config.get('root_directory', '')
                            config['ensemble_path'] = str(Path(root_dir) / ensemble_config['path'])
                    if 'mode' in ensemble_config:
                        config['mode'] = ensemble_config['mode']
                
                if 'download' in models_config:
                    download_config = models_config['download']
                    if 'auto_download' in download_config:
                        config['auto_download'] = download_config['auto_download']
                    if 'fallback_to_default' in download_config:
                        config['fallback_to_default'] = download_config['fallback_to_default']
                        
                # Add other top-level settings
                if 'validate_compatibility' in models_config:
                    config['validate_compatibility'] = models_config['validate_compatibility']
                
                return config
            else:
                # Flat structure - use as-is
                return data
                
        except Exception as e:
            logger.warning(f"Failed to parse configuration file {config_path}: {e}")
            return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        
        # Legacy environment variables (backwards compatibility)
        if env_dir := os.environ.get('CHIRPKIT_MODEL_DIR'):
            config['root_directory'] = env_dir
        elif chirpkit_home := os.environ.get('CHIRPKIT_HOME'):
            config['root_directory'] = str(Path(chirpkit_home) / 'models')
        
        # New comprehensive environment variables
        if env_root := os.environ.get('CHIRPKIT_ROOT_DIR'):
            config['root_directory'] = env_root
        
        if env_birdnet := os.environ.get('CHIRPKIT_BIRDNET_MODEL'):
            config['birdnet_model_path'] = env_birdnet
            
        if env_birdnet_labels := os.environ.get('CHIRPKIT_BIRDNET_LABELS'):
            config['birdnet_labels_path'] = env_birdnet_labels
        
        if env_ensemble := os.environ.get('CHIRPKIT_ENSEMBLE_DIR'):
            config['ensemble_path'] = env_ensemble
        
        if env_mode := os.environ.get('CHIRPKIT_MODE'):
            config['mode'] = env_mode
        
        if env_auto_download := os.environ.get('CHIRPKIT_AUTO_DOWNLOAD'):
            config['auto_download'] = env_auto_download.lower() in ['true', '1', 'yes', 'on']
        
        if env_validate := os.environ.get('CHIRPKIT_VALIDATE_COMPATIBILITY'):
            config['validate_compatibility'] = env_validate.lower() in ['true', '1', 'yes', 'on']
        
        if env_device := os.environ.get('CHIRPKIT_DEVICE'):
            config['device'] = env_device
        
        logger.debug(f"Environment configuration: {config}")
        return config


class ModelDiscovery:
    """
    Intelligent model discovery and path resolution
    """
    
    @staticmethod
    def find_models(root_directory: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Discover compatible models in directory structure
        
        Args:
            root_directory: Root directory to search
            
        Returns:
            Dictionary with discovered model paths by type
        """
        root_path = Path(root_directory)
        discovered = {
            'birdnet_models': [],
            'ensemble_models': [],
            'label_encoders': []
        }
        
        if not root_path.exists():
            logger.warning(f"Root directory does not exist: {root_path}")
            return discovered
        
        # Look for BirdNET models
        birdnet_patterns = [
            "**/BirdNET*.tflite",
            "**/birdnet/*.tflite",
            "**/*birdnet*.tflite"
        ]
        
        for pattern in birdnet_patterns:
            for model_path in root_path.glob(pattern):
                if model_path.is_file():
                    discovered['birdnet_models'].append(str(model_path))
        
        # Look for ensemble models
        ensemble_patterns = [
            "**/ensemble_model_*.pth",
            "**/trained/*/ensemble_model_*.pth", 
            "**/chirpkit-ensemble/ensemble_model_*.pth",
            "**/*ensemble*/ensemble_model_*.pth"
        ]
        
        ensemble_dirs = set()
        for pattern in ensemble_patterns:
            for model_path in root_path.glob(pattern):
                if model_path.is_file():
                    ensemble_dirs.add(str(model_path.parent))
        
        discovered['ensemble_models'] = list(ensemble_dirs)
        
        # Look for label encoders
        encoder_patterns = [
            "**/label_encoder.joblib",
            "**/*label_encoder*.joblib"
        ]
        
        for pattern in encoder_patterns:
            for encoder_path in root_path.glob(pattern):
                if encoder_path.is_file():
                    discovered['label_encoders'].append(str(encoder_path))
        
        logger.debug(f"Discovered models in {root_path}: {discovered}")
        return discovered
    
    @staticmethod
    def select_best_models(discovered: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
        """
        Select best models from discovered options
        
        Args:
            discovered: Dictionary from find_models()
            
        Returns:
            Dictionary with selected model paths
        """
        selection = {
            'birdnet_model_path': None,
            'ensemble_path': None,
            'label_encoder_path': None
        }
        
        # Select BirdNET model (prefer v2.4, then latest)
        if discovered['birdnet_models']:
            # Prefer v2.4 models
            v24_models = [p for p in discovered['birdnet_models'] if 'V2.4' in p or 'v2.4' in p]
            if v24_models:
                # Prefer FP16 over FP32
                fp16_models = [p for p in v24_models if 'FP16' in p]
                selection['birdnet_model_path'] = fp16_models[0] if fp16_models else v24_models[0]
            else:
                selection['birdnet_model_path'] = discovered['birdnet_models'][0]
        
        # Select ensemble directory (prefer chirpkit-ensemble)
        if discovered['ensemble_models']:
            chirpkit_ensembles = [p for p in discovered['ensemble_models'] if 'chirpkit-ensemble' in p]
            selection['ensemble_path'] = chirpkit_ensembles[0] if chirpkit_ensembles else discovered['ensemble_models'][0]
        
        # Select label encoder (prefer one near ensemble)
        if discovered['label_encoders'] and selection['ensemble_path']:
            # Look for encoder in same directory as ensemble
            ensemble_dir = Path(selection['ensemble_path'])
            for encoder_path in discovered['label_encoders']:
                if Path(encoder_path).parent == ensemble_dir:
                    selection['label_encoder_path'] = encoder_path
                    break
            else:
                # Use first available
                selection['label_encoder_path'] = discovered['label_encoders'][0]
        
        logger.info(f"Selected models: {selection}")
        return selection


class ModelValidator:
    """
    Validates model compatibility and configuration
    """
    
    @staticmethod
    def validate_configuration(config: ModelConfiguration) -> None:
        """
        Validate model configuration for compatibility issues
        
        Args:
            config: Configuration to validate
            
        Raises:
            ModelCompatibilityError: If validation fails
        """
        errors = []
        
        # Check if paths exist (when not using auto-download)
        if not config.auto_download:
            if config.birdnet_model_path and not Path(config.birdnet_model_path).exists():
                errors.append(f"BirdNET model not found: {config.birdnet_model_path}")
            
            if config.ensemble_path and not Path(config.ensemble_path).exists():
                errors.append(f"Ensemble path not found: {config.ensemble_path}")
        
        # Validate ensemble directory structure
        if config.ensemble_path:
            ensemble_dir = Path(config.ensemble_path)
            if ensemble_dir.exists():
                ensemble_info = ensemble_dir / "ensemble_info.json"
                if not ensemble_info.exists():
                    errors.append(f"Ensemble info file missing: {ensemble_info}")
                
                # Check for at least one model file
                model_files = list(ensemble_dir.glob("ensemble_model_*.pth"))
                if not model_files:
                    errors.append(f"No ensemble model files found in: {ensemble_dir}")
        
        if errors:
            raise ModelCompatibilityError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        logger.info("Configuration validation passed")
    
    @staticmethod
    def validate_model_compatibility(birdnet_path: Optional[str], ensemble_path: Optional[str]) -> None:
        """
        Validate that BirdNET and ensemble models are compatible
        
        Args:
            birdnet_path: Path to BirdNET model
            ensemble_path: Path to ensemble directory
            
        Raises:
            ModelCompatibilityError: If models are incompatible
        """
        if not (birdnet_path and ensemble_path):
            logger.info("Skipping compatibility check - missing model paths")
            return
        
        try:
            # Get BirdNET output dimensions (this would require loading the model)
            # For now, we'll assume BirdNET v2.4 outputs 1024-dimensional embeddings
            birdnet_dims = 1024
            
            # Get ensemble input dimensions from ensemble_info.json
            ensemble_info_path = Path(ensemble_path) / "ensemble_info.json"
            if ensemble_info_path.exists():
                with open(ensemble_info_path, 'r') as f:
                    ensemble_info = json.load(f)
                
                # Assume ensemble expects 1024-dimensional embeddings
                # (This could be read from the model architecture in ensemble_info)
                ensemble_input_dims = ensemble_info.get('input_dimensions', 1024)
                
                if birdnet_dims != ensemble_input_dims:
                    raise ModelCompatibilityError(
                        f"Model dimension mismatch: "
                        f"BirdNET outputs {birdnet_dims}D embeddings but "
                        f"ensemble expects {ensemble_input_dims}D inputs"
                    )
            
            logger.info("Model compatibility check passed")
            
        except Exception as e:
            if isinstance(e, ModelCompatibilityError):
                raise
            else:
                logger.warning(f"Could not validate model compatibility: {e}")


def create_example_config_file(config_path: str = "~/.chirpkit/config.yaml") -> None:
    """
    Create an example configuration file
    
    Args:
        config_path: Path where to create the config file
    """
    config_path = Path(config_path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    example_config = {
        'models': {
            'root_directory': "/models/chirpkit",
            'birdnet': {
                'model_path': "birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite",
                'labels_path': "birdnet/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
            },
            'ensemble': {
                'path': "trained/chirpkit-ensemble",
                'mode': "ensemble_tta"
            },
            'download': {
                'auto_download': False,
                'fallback_to_default': True
            },
            'validate_compatibility': True
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Example configuration created at: {config_path}")


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing ChirpKit Configuration System")
    print("=" * 60)
    
    # Test configuration resolution
    config_manager = ConfigurationManager()
    config = config_manager.resolve_configuration()
    
    print(f"Resolved configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    
    # Test model discovery
    if config.root_directory:
        print(f"\n🔍 Discovering models in: {config.root_directory}")
        discovered = ModelDiscovery.find_models(config.root_directory)
        selection = ModelDiscovery.select_best_models(discovered)
        
        print(f"Discovered: {discovered}")
        print(f"Selected: {selection}")
    
    # Test validation
    try:
        ModelValidator.validate_configuration(config)
        print("\n✅ Configuration validation passed")
    except ModelCompatibilityError as e:
        print(f"\n❌ Configuration validation failed: {e}")
    
    print("\n" + "=" * 60)