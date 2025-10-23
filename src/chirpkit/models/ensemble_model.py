"""
Ensemble model for combining multiple approaches to reach 85%+ accuracy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class InsectEnsemble(nn.Module):
    """Ensemble combining CNN-LSTM, Enhanced CNN-LSTM, and Bayesian models"""

    def __init__(self, model_configs, ensemble_method='weighted_average'):
        super().__init__()

        self.models = nn.ModuleList()
        self.model_weights = nn.Parameter(torch.ones(len(model_configs)))
        self.ensemble_method = ensemble_method

        # Load individual models
        for config in model_configs:
            model = self._load_model(config)
            self.models.append(model)

        # Meta-learner for intelligent ensemble
        if ensemble_method == 'meta_learning':
            feature_dim = sum(config['feature_dim'] for config in model_configs)
            self.meta_learner = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, model_configs[0]['n_classes'])
            )

    def _load_model(self, config):
        """Load a pre-trained model"""
        model_type = config['type']

        if model_type == 'enhanced_cnn_lstm':
            from .enhanced_cnn_lstm import EnhancedCNNLSTMClassifier
            model = EnhancedCNNLSTMClassifier(
                n_classes=config['n_classes'],
                lstm_hidden=config.get('lstm_hidden', 256),
                dropout=config.get('dropout', 0.3)
            )
        elif model_type == 'bayesian_cnn_lstm':
            from .bayesian_cnn_lstm import BayesianInsectClassifier
            model = BayesianInsectClassifier(
                n_classes=config['n_classes'],
                dropout=config.get('dropout', 0.3)
            )
        elif model_type == 'simple_cnn_lstm':
            from .simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
            model = SimpleCNNLSTMInsectClassifier(
                n_classes=config['n_classes']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load pre-trained weights if available
        if 'weights_path' in config and Path(config['weights_path']).exists():
            state_dict = torch.load(config['weights_path'], map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded weights for {model_type} from {config['weights_path']}")

        return model

    def forward(self, x, return_individual=False):
        individual_outputs = []
        individual_features = []

        # Get predictions from each model
        for i, model in enumerate(self.models):
            if hasattr(model, 'forward') and 'return_features' in model.forward.__code__.co_varnames:
                # Enhanced models that return features
                try:
                    logits, features, aux_logits = model(x, return_features=True)
                    individual_outputs.append(logits)
                    individual_features.append(features)
                except:
                    # Fallback if model doesn't support return_features
                    logits = model(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    individual_outputs.append(logits)
                    individual_features.append(None)
            else:
                # Simple models
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                individual_outputs.append(logits)
                individual_features.append(None)

        # Ensemble the predictions
        if self.ensemble_method == 'simple_average':
            ensemble_logits = torch.stack(individual_outputs).mean(dim=0)

        elif self.ensemble_method == 'weighted_average':
            weights = F.softmax(self.model_weights, dim=0)
            weighted_outputs = []
            for i, output in enumerate(individual_outputs):
                weighted_outputs.append(weights[i] * output)
            ensemble_logits = torch.stack(weighted_outputs).sum(dim=0)

        elif self.ensemble_method == 'max_confidence':
            # Use prediction from most confident model
            confidences = [F.softmax(output, dim=1).max(dim=1)[0] for output in individual_outputs]
            max_conf_indices = torch.stack(confidences).argmax(dim=0)

            batch_size = x.size(0)
            ensemble_logits = torch.zeros_like(individual_outputs[0])
            for b in range(batch_size):
                ensemble_logits[b] = individual_outputs[max_conf_indices[b]][b]

        elif self.ensemble_method == 'meta_learning' and all(f is not None for f in individual_features):
            # Use meta-learner to combine features
            combined_features = torch.cat([f for f in individual_features if f is not None], dim=1)
            ensemble_logits = self.meta_learner(combined_features)

        else:
            # Default to simple average
            ensemble_logits = torch.stack(individual_outputs).mean(dim=0)

        if return_individual:
            return ensemble_logits, individual_outputs
        else:
            return ensemble_logits

    def update_weights(self, individual_accuracies):
        """Update ensemble weights based on individual model performance"""
        accuracies = torch.tensor(individual_accuracies, device=self.model_weights.device)
        # Higher accuracy models get higher weights
        self.model_weights.data = F.softmax(accuracies * 5.0, dim=0)  # Temperature scaling

class AdaptiveEnsemble(nn.Module):
    """Ensemble that adapts based on input characteristics"""

    def __init__(self, model_configs, n_classes):
        super().__init__()

        self.models = nn.ModuleList()
        for config in model_configs:
            model = self._load_model(config)
            self.models.append(model)

        # Input analyzer to determine which models to use
        self.input_analyzer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, len(model_configs)),  # Output weights for each model
            nn.Sigmoid()
        )

        self.n_classes = n_classes

    def _load_model(self, config):
        """Load model same as InsectEnsemble"""
        # Same implementation as InsectEnsemble._load_model
        model_type = config['type']

        if model_type == 'enhanced_cnn_lstm':
            from .enhanced_cnn_lstm import EnhancedCNNLSTMClassifier
            model = EnhancedCNNLSTMClassifier(
                n_classes=config['n_classes'],
                lstm_hidden=config.get('lstm_hidden', 256),
                dropout=config.get('dropout', 0.3)
            )
        elif model_type == 'bayesian_cnn_lstm':
            from .bayesian_cnn_lstm import BayesianInsectClassifier
            model = BayesianInsectClassifier(
                n_classes=config['n_classes'],
                dropout=config.get('dropout', 0.3)
            )
        elif model_type == 'simple_cnn_lstm':
            from .simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
            model = SimpleCNNLSTMInsectClassifier(
                n_classes=config['n_classes']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if 'weights_path' in config and Path(config['weights_path']).exists():
            state_dict = torch.load(config['weights_path'], map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        return model

    def forward(self, x):
        # Analyze input to determine model weights
        adaptive_weights = self.input_analyzer(x)  # (batch, n_models)

        # Get predictions from each model
        individual_outputs = []
        for model in self.models:
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            individual_outputs.append(logits)

        # Weighted combination based on input analysis
        batch_size = x.size(0)
        ensemble_logits = torch.zeros(batch_size, self.n_classes, device=x.device)

        for b in range(batch_size):
            weighted_sum = torch.zeros(self.n_classes, device=x.device)
            total_weight = 0

            for i, output in enumerate(individual_outputs):
                weight = adaptive_weights[b, i]
                weighted_sum += weight * output[b]
                total_weight += weight

            ensemble_logits[b] = weighted_sum / (total_weight + 1e-8)

        return ensemble_logits