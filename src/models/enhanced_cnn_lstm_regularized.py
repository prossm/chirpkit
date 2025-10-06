"""
Enhanced CNN-LSTM with AGGRESSIVE regularization for limited data
Optimized for 483 species with only ~40 samples per species
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple temporal and frequency scales with STRONG dropout"""

    def __init__(self, input_channels=1, dropout=0.5):
        super().__init__()

        # Three parallel paths for different scales
        # Path 1: Fine-grained (small kernels, high resolution)
        self.fine_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),  # Lighter dropout early
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)  # Stronger dropout after pooling
        )

        # Path 2: Medium-scale (moderate kernels)
        self.medium_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )

        # Path 3: Coarse-grained (large kernels, capture long patterns)
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(32, 64, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )

        # Feature fusion with dropout
        self.fusion_conv = nn.Conv2d(192, 128, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(128)
        self.fusion_dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        fine = self.fine_conv(x)
        medium = self.medium_conv(x)
        coarse = self.coarse_conv(x)

        # Concatenate along channel dimension
        fused = torch.cat([fine, medium, coarse], dim=1)
        fused = self.fusion_conv(fused)
        fused = self.fusion_bn(fused)
        fused = F.relu(fused)
        fused = self.fusion_dropout(fused)

        return fused

class MultiHeadAttention(nn.Module):
    """Multi-head attention for sequence modeling with increased dropout"""

    def __init__(self, d_model, n_heads=8, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout * 0.7)  # Additional dropout on output
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        residual = x

        # Multi-head attention
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Output projection and residual connection
        output = self.w_o(attn_output)
        output = self.output_dropout(output)  # Extra dropout before residual
        output = self.layer_norm(output + residual)

        return output

class SpeciesSpecificAttention(nn.Module):
    """Learn species-specific attention patterns with L2 regularization"""

    def __init__(self, feature_dim, n_classes, dropout=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes

        # Species-specific attention weights (will be L2 regularized)
        self.species_attention = nn.Parameter(torch.randn(n_classes, feature_dim) * 0.01)  # Small init
        self.temperature = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, predictions=None):
        # features: (batch, seq_len, feature_dim)
        # predictions: (batch, n_classes) - class probabilities

        if predictions is not None:
            # Use predicted class probabilities to weight species attention
            weighted_attention = torch.matmul(predictions, self.species_attention)  # (batch, feature_dim)
            weighted_attention = weighted_attention.unsqueeze(1)  # (batch, 1, feature_dim)

            # Apply attention to features
            attention_scores = torch.matmul(features, weighted_attention.transpose(1, 2))  # (batch, seq_len, 1)
            attention_weights = F.softmax(attention_scores / self.temperature, dim=1)

            attended_features = features * attention_weights
            attended_features = self.dropout(attended_features)  # Dropout on attended features
            return attended_features.sum(dim=1)  # (batch, feature_dim)
        else:
            # During initial forward pass, use mean pooling
            return features.mean(dim=1)

class RegularizedEnhancedCNNLSTMClassifier(nn.Module):
    """Enhanced CNN-LSTM with AGGRESSIVE regularization for limited data"""

    def __init__(self, n_classes, input_channels=1, lstm_hidden=256, dropout=0.5):
        super().__init__()

        self.n_classes = n_classes
        self.dropout_rate = dropout

        # Multi-scale feature extraction with strong dropout
        self.feature_extractor = MultiScaleFeatureExtractor(input_channels, dropout=dropout)

        # Additional CNN layers with MORE dropout
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.6),  # Dropout before pooling
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),  # Dropout after pooling

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.6),  # Dropout before pooling
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)  # Dropout after pooling
        )

        # We'll calculate the LSTM input size dynamically
        self.lstm_hidden = lstm_hidden
        self.lstm_dropout = dropout
        self.lstm = None
        self._lstm_initialized = False

        lstm_output_dim = lstm_hidden * 2  # Bidirectional

        # Multi-head attention with strong dropout
        self.attention = MultiHeadAttention(lstm_output_dim, n_heads=8, dropout=dropout)

        # Manual dropout between LSTM and attention (since PyTorch LSTM dropout is disabled for 2 layers)
        self.lstm_output_dropout = nn.Dropout(dropout)

        # Species-specific attention with dropout
        self.species_attention = SpeciesSpecificAttention(lstm_output_dim, n_classes, dropout=dropout)

        # Classification head with STRONG dropout
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),  # 50% dropout

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),  # 50% dropout

            nn.Linear(512, n_classes)
        )

        # Auxiliary classifier for regularization
        self.aux_classifier = nn.Sequential(
            nn.Dropout(dropout * 0.7),  # Dropout before aux classifier too
            nn.Linear(lstm_output_dim, n_classes)
        )

    def forward(self, x, return_features=False):
        batch_size = x.size(0)

        # Multi-scale feature extraction (already has dropout inside)
        features = self.feature_extractor(x)  # (batch, 128, H, W)
        features = self.cnn_layers(features)  # (batch, 512, H', W')

        # Get actual dimensions after CNN layers
        _, channels, height, width = features.shape
        feature_dim = channels * height * width

        # Initialize LSTM on first forward pass if needed (3 layers to enable dropout!)
        if not self._lstm_initialized:
            print(f"🔧 Initializing 3-layer LSTM with input size: {feature_dim}, dropout: {self.lstm_dropout}")
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=self.lstm_hidden,
                num_layers=3,  # 3 layers to enable LSTM dropout!
                batch_first=True,
                dropout=self.lstm_dropout,  # Now this will work!
                bidirectional=True
            ).to(features.device)
            self._lstm_initialized = True

        # Reshape for LSTM: treat spatial dimensions as sequence
        features = features.view(batch_size, -1, feature_dim)  # (batch, seq_len, feature_dim)

        # LSTM processing (now has internal dropout!)
        lstm_out, _ = self.lstm(features)  # (batch, seq_len, lstm_hidden*2)

        # Additional dropout after LSTM output
        lstm_out = self.lstm_output_dropout(lstm_out)

        # Multi-head attention (has dropout inside)
        attended_features = self.attention(lstm_out)  # (batch, seq_len, lstm_hidden*2)

        # Initial classification for species-specific attention
        pooled_features = attended_features.mean(dim=1)
        aux_logits = self.aux_classifier(pooled_features)
        aux_probs = F.softmax(aux_logits, dim=1)

        # Species-specific attention (has dropout inside)
        final_features = self.species_attention(attended_features, aux_probs)

        # Final classification
        logits = self.classifier(final_features)

        if return_features:
            return logits, final_features, aux_logits
        else:
            return logits, aux_logits

    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        with torch.no_grad():
            logits, aux_logits = self.forward(x)
            return logits.mean(dim=1, keepdim=True)

    def get_l2_regularization_loss(self, weight_decay=1e-4):
        """Compute L2 regularization on species-specific attention weights"""
        # Extra penalty on species attention to prevent overfitting
        return weight_decay * torch.norm(self.species_attention.species_attention, p=2)

class EnhancedLoss(nn.Module):
    """Enhanced loss with auxiliary supervision and STRONG label smoothing"""

    def __init__(self, n_classes, alpha=0.4, label_smoothing=0.15):
        """
        Args:
            alpha: Weight for auxiliary loss (increased from 0.3 to 0.4)
            label_smoothing: Increased from 0.1 to 0.15 for stronger regularization
        """
        super().__init__()
        self.alpha = alpha
        self.main_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, main_logits, aux_logits, targets):
        main_loss = self.main_criterion(main_logits, targets)
        aux_loss = self.aux_criterion(aux_logits, targets)

        total_loss = main_loss + self.alpha * aux_loss
        return total_loss, main_loss, aux_loss
