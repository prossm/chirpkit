"""
Configuration management for the Insect Classifier
"""
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json
import yaml

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    n_classes: int = 12
    cnn_channels: List[int] = None
    lstm_hidden: int = 256
    lstm_layers: int = 2
    dropout: float = 0.3
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128, 256]

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4
    
    # Scheduler settings
    scheduler_type: str = "cosine_annealing"  # cosine_annealing, reduce_on_plateau, step
    scheduler_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.scheduler_params is None:
            if self.scheduler_type == "cosine_annealing":
                self.scheduler_params = {"T_max": self.max_epochs, "eta_min": 1e-6}
            elif self.scheduler_type == "reduce_on_plateau":
                self.scheduler_params = {"patience": 5, "factor": 0.5, "verbose": True}
            elif self.scheduler_type == "step":
                self.scheduler_params = {"step_size": 30, "gamma": 0.1}

@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    use_augmentation: bool = True
    augmentation_prob: float = 0.6
    time_stretch_range: tuple = (0.8, 1.2)
    frequency_shift_max: int = 5
    noise_factor_range: tuple = (0.001, 0.01)
    freq_mask_param: int = 15
    time_mask_param: int = 20

@dataclass
class DataConfig:
    """Data paths and preprocessing configuration"""
    train_features_path: str = 'data/splits/X_train.npy'
    train_labels_path: str = 'data/splits/y_train.npy'
    val_features_path: str = 'data/splits/X_val.npy'
    val_labels_path: str = 'data/splits/y_val.npy'
    test_features_path: str = 'data/splits/X_test.npy'
    test_labels_path: str = 'data/splits/y_test.npy'
    
    # Preprocessing
    target_sr: int = 16000
    duration: float = 2.5
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    data: DataConfig
    
    # Paths
    model_save_dir: str = 'models/trained'
    checkpoint_dir: str = 'models/checkpoints'
    log_dir: str = 'runs'
    evaluation_dir: str = 'models/evaluation'
    
    # System
    device: str = "auto"  # auto, cuda, cpu
    resume_training: bool = True
    seed: int = 42
    
    # Experiment tracking
    experiment_name: str = "insect_classifier_enhanced"
    notes: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["audio", "classification", "insects", "cnn-lstm"]

class ConfigManager:
    """Configuration management utilities"""
    
    @staticmethod
    def create_default_config() -> ExperimentConfig:
        """Create default configuration"""
        return ExperimentConfig(
            model=ModelConfig(),
            training=TrainingConfig(),
            augmentation=AugmentationConfig(),
            data=DataConfig()
        )
    
    @staticmethod
    def save_config(config: ExperimentConfig, path: str):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config_dict = asdict(config)
        
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def load_config(path: str) -> ExperimentConfig:
        """Load configuration from file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                config_dict = json.load(f)
        
        return ExperimentConfig(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            augmentation=AugmentationConfig(**config_dict['augmentation']),
            data=DataConfig(**config_dict['data'])
        )
    
    @staticmethod
    def get_device(device_config: str = "auto") -> str:
        """Get appropriate device"""
        import torch
        
        if device_config == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return device_config
    
    @staticmethod
    def setup_directories(config: ExperimentConfig):
        """Setup all necessary directories"""
        directories = [
            config.model_save_dir,
            config.checkpoint_dir,
            config.log_dir,
            config.evaluation_dir,
            'models/validation'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"✅ Created directories: {', '.join(directories)}")
    
    @staticmethod
    def setup_experiment(config_path: Optional[str] = None) -> ExperimentConfig:
        """Setup complete experiment with configuration"""
        if config_path and os.path.exists(config_path):
            print(f"📁 Loading configuration from {config_path}")
            config = ConfigManager.load_config(config_path)
        else:
            print("🔧 Using default configuration")
            config = ConfigManager.create_default_config()
            
            # Save default config for reference
            default_config_path = 'config/default_config.json'
            ConfigManager.save_config(config, default_config_path)
            print(f"💾 Default configuration saved to {default_config_path}")
        
        # Setup directories
        ConfigManager.setup_directories(config)
        
        # Set random seeds for reproducibility
        import torch
        import numpy as np
        import random
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        print(f"🎯 Experiment: {config.experiment_name}")
        print(f"🏷️ Tags: {', '.join(config.tags)}")
        if config.notes:
            print(f"📝 Notes: {config.notes}")
        
        return config

# Convenience function for backwards compatibility
def get_default_config():
    """Get default configuration - backwards compatible"""
    return ConfigManager.create_default_config()

if __name__ == "__main__":
    # Example usage
    config = ConfigManager.create_default_config()
    ConfigManager.save_config(config, 'config/example_config.json')
    print("✅ Example configuration saved to config/example_config.json")