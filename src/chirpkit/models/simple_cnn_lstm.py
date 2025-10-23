import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNLSTMInsectClassifier(nn.Module):
    """Enhanced CNN-LSTM model for 80% target accuracy on 483 species"""
    def __init__(self, n_classes: int = 12, dropout: float = 0.3, input_height: int = 256):
        """
        Enhanced architecture for insect classification

        Args:
            n_classes: Number of insect species
            dropout: Dropout rate
            input_height: Input spectrogram height (256 mel bins for enhanced features)
        """
        super().__init__()
        self.n_classes = n_classes
        self.input_height = input_height

        # Enhanced CNN layers: 5 layers for richer feature extraction
        # Handles 256 mel bins (vs 128 previously)
        self.conv_layers = nn.ModuleList([
            # Layer 1: 256x time -> 128x time/2
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            ),
            # Layer 2: 128x time/2 -> 64x time/4
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            ),
            # Layer 3: 64x time/4 -> 32x time/8
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            ),
            # Layer 4: 32x time/8 -> 16x time/16
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            ),
            # Layer 5: 16x time/16 -> 8x time/32
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            )
        ])

        # Enhanced LSTM: 3 stacked layers with 512 hidden units
        # Deeper network to learn complex temporal patterns in insect calls
        self.lstm = nn.LSTM(
            input_size=512,   # From final CNN layer
            hidden_size=512,  # Increased from 256
            num_layers=3,     # Increased from 1 (deeper for complex patterns)
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,  # Dropout between LSTM layers
            bidirectional=True
        )

        # Attention mechanism: helps model focus on diagnostic parts of insect calls
        self.attention_weights = nn.Linear(1024, 1)  # 1024 from bidirectional 512
        self.attention_dropout = nn.Dropout(dropout)

        # Enhanced classifier: wider network for 483-class discrimination
        # Architecture: 1024 → 1024 → 1024 → 512 → n_classes
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),  # Keep full bidirectional features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),  # Wide hidden layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),   # Compress
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)  # Final classification
        )
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through enhanced architecture

        Args:
            x: Input tensor [batch, 1, freq, time] - freq is 256 mel bins
            return_attention: Return attention weights for diversity loss

        Returns:
            output: Classification logits [batch, n_classes]
            attention_weights: (optional) Attention weights [batch, seq_len]
        """
        batch_size = x.size(0)

        # Enhanced CNN forward pass (5 layers)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Global average pooling over frequency dimension
        x = x.mean(dim=2)  # [batch, channels, time]
        x = x.transpose(1, 2)  # [batch, time, channels]

        # Enhanced LSTM processing (3 layers, bidirectional)
        lstm_out, _ = self.lstm(x)  # [batch, time, 1024] (512*2 from bidirectional)

        # Attention mechanism to focus on diagnostic parts of insect calls
        # Calculate attention scores using a linear layer
        attention_scores = self.attention_weights(lstm_out)  # [batch, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch, seq_len]

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len]

        # Apply attention weights to LSTM outputs
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [batch, seq_len, 1]
        weighted_features = lstm_out * attention_weights_expanded  # [batch, seq_len, 1024]

        # Sum across time dimension (weighted sum)
        features = weighted_features.sum(dim=1)  # [batch, 1024]
        features = self.attention_dropout(features)

        # Enhanced classification (4-layer MLP)
        output = self.classifier(features)

        if return_attention:
            return output, attention_weights
        return output
    
    def compute_attention_diversity_loss(self, attention_weights):
        """
        Compute attention diversity loss to encourage exploration of different time steps

        Args:
            attention_weights: [batch_size, seq_len] attention weights

        Returns:
            diversity_loss: scalar tensor encouraging attention to spread across time steps
        """
        # Calculate entropy of attention weights to encourage diversity
        # Higher entropy = more diverse attention (better)
        # We want to maximize entropy, so we minimize negative entropy

        # Add small epsilon to avoid log(0)
        eps = 1e-8
        attention_weights_safe = attention_weights + eps

        # Calculate entropy: -sum(p * log(p))
        entropy = -(attention_weights_safe * torch.log(attention_weights_safe)).sum(dim=1)

        # We want high entropy (diverse attention), so minimize negative entropy
        diversity_loss = -entropy.mean()

        return diversity_loss