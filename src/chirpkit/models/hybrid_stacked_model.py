"""
Hybrid Stacked Generalization: CNN-LSTM Feature Extractor + Bayesian Classifier
Combines the feature learning power of CNN-LSTM with Bayesian uncertainty quantification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib

try:
    from .simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
    from .bayesian_cnn_lstm import BayesianInsectClassifier
except ImportError:
    from simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
    from bayesian_cnn_lstm import BayesianInsectClassifier


class CNNLSTMFeatureExtractor(nn.Module):
    """Feature extractor based on pre-trained CNN-LSTM model"""

    def __init__(self, pretrained_model_path: str, feature_dim: int = 256):
        super().__init__()

        # Load the pre-trained CNN-LSTM model
        self.pretrained_cnn_lstm = SimpleCNNLSTMInsectClassifier()
        state_dict = torch.load(pretrained_model_path, map_location='cpu')
        self.pretrained_cnn_lstm.load_state_dict(state_dict)

        # Freeze the feature extraction layers
        for param in self.pretrained_cnn_lstm.parameters():
            param.requires_grad = False

        # Remove the final classification layer and replace with feature extractor
        # The CNN-LSTM has: conv layers -> lstm -> attention -> classifier
        # We want to extract features right before the final classifier

        self.cnn_layers = self.pretrained_cnn_lstm.cnn_layers
        self.lstm = self.pretrained_cnn_lstm.lstm
        self.attention = self.pretrained_cnn_lstm.attention

        # Add a feature projection layer
        self.feature_dim = feature_dim
        lstm_output_size = self.pretrained_cnn_lstm.lstm.hidden_size * 2  # bidirectional
        self.feature_projector = nn.Sequential(
            nn.Linear(lstm_output_size, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim)
        )

        self.eval()  # Set to eval mode for feature extraction

    def forward(self, x):
        """Extract features from audio spectrograms"""
        with torch.no_grad():  # No gradients needed for frozen layers
            # CNN feature extraction
            x = self.cnn_layers(x)

            # Prepare for LSTM: (batch, channels, freq, time) -> (batch, time, features)
            batch_size, channels, freq, time = x.size()
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.view(batch_size, time, channels * freq)

            # LSTM processing
            lstm_out, (hidden, cell) = self.lstm(x)

            # Attention mechanism
            context_vector = self.attention(lstm_out)

        # Project to feature space (this layer can be trained)
        features = self.feature_projector(context_vector)
        return features


class CompactBayesianClassifier(nn.Module):
    """Lightweight Bayesian classifier for CNN-LSTM features"""

    def __init__(self, feature_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()

        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.dropout = dropout

        # Compact Bayesian classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, n_classes)
        )

        # Bayesian components
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_calibrated = False
        self.uncertainty_method = 'monte_carlo'
        self.n_mc_samples = 50

    def forward(self, features):
        """Forward pass through compact classifier"""
        return self.classifier(features)

    def forward_with_uncertainty(self, features, n_samples=50):
        """Bayesian inference with uncertainty quantification"""
        self.train()  # Enable dropout for MC sampling

        predictions_list = []

        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.forward(features)
                predictions_list.append(F.softmax(logits, dim=1))

        predictions_tensor = torch.stack(predictions_list)  # [n_samples, batch, n_classes]

        # Compute statistics
        mean_predictions = predictions_tensor.mean(dim=0)
        prediction_variance = predictions_tensor.var(dim=0)

        # Total uncertainty (entropy of mean prediction)
        total_uncertainty = -torch.sum(mean_predictions * torch.log(mean_predictions + 1e-8), dim=1)

        # Aleatoric uncertainty (expected entropy of individual predictions)
        individual_entropies = -torch.sum(predictions_tensor * torch.log(predictions_tensor + 1e-8), dim=2)
        aleatoric_uncertainty = individual_entropies.mean(dim=0)

        # Epistemic uncertainty (difference between total and aleatoric)
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

        return {
            'predictions': mean_predictions,
            'prediction_variance': prediction_variance,
            'total_uncertainty': total_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'mc_predictions': predictions_tensor
        }

    def calibrate_temperature(self, val_loader, device='cpu'):
        """Temperature scaling calibration"""
        logits_list = []
        labels_list = []

        self.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = self.forward(features)
                logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.is_calibrated = True

        return self.temperature.item()


class HybridStackedModel(nn.Module):
    """Complete hybrid model: CNN-LSTM features + Bayesian uncertainty"""

    def __init__(self, pretrained_cnn_lstm_path: str, n_classes: int, feature_dim: int = 256):
        super().__init__()

        self.feature_extractor = CNNLSTMFeatureExtractor(pretrained_cnn_lstm_path, feature_dim)
        self.bayesian_classifier = CompactBayesianClassifier(feature_dim, n_classes)

        self.n_classes = n_classes
        self.feature_dim = feature_dim

    def forward(self, x):
        """Standard forward pass"""
        features = self.feature_extractor(x)
        return self.bayesian_classifier(features)

    def forward_with_uncertainty(self, x, n_samples=50, return_features=False):
        """Full uncertainty-aware prediction"""
        # Extract features
        features = self.feature_extractor(x)

        # Bayesian prediction with uncertainty
        uncertainty_data = self.bayesian_classifier.forward_with_uncertainty(features, n_samples)

        if return_features:
            uncertainty_data['features'] = features

        return uncertainty_data

    def freeze_feature_extractor(self):
        """Freeze feature extractor for faster Bayesian training"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_projector(self):
        """Allow feature projector to adapt during Bayesian training"""
        for param in self.feature_extractor.feature_projector.parameters():
            param.requires_grad = True

    def save_model(self, save_path: str):
        """Save complete hybrid model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'feature_dim': self.feature_dim,
            'n_classes': self.n_classes,
            'is_calibrated': self.bayesian_classifier.is_calibrated,
            'temperature': self.bayesian_classifier.temperature.item() if self.bayesian_classifier.is_calibrated else None
        }, save_path)

    def load_model(self, model_path: str):
        """Load complete hybrid model"""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint.get('is_calibrated', False):
            self.bayesian_classifier.is_calibrated = True
            self.bayesian_classifier.temperature.data = torch.tensor([checkpoint['temperature']])

        return checkpoint


def create_feature_dataset(pretrained_model_path: str, data_loader, device='cpu', save_path=None):
    """Extract features from existing dataset using pre-trained CNN-LSTM"""

    # Create feature extractor
    feature_extractor = CNNLSTMFeatureExtractor(pretrained_model_path)
    feature_extractor.to(device)
    feature_extractor.eval()

    features_list = []
    labels_list = []

    print(f"🔄 Extracting features using {pretrained_model_path}...")

    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device)
            features = feature_extractor(X_batch)

            features_list.append(features.cpu())
            labels_list.append(y_batch)

            if i % 100 == 0:
                print(f"   Processed {i}/{len(data_loader)} batches...")

    # Combine all features
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    print(f"✅ Extracted {all_features.shape[0]} feature vectors of dimension {all_features.shape[1]}")

    if save_path:
        torch.save({
            'features': all_features,
            'labels': all_labels,
            'feature_dim': all_features.shape[1]
        }, save_path)
        print(f"💾 Features saved to: {save_path}")

    return all_features, all_labels