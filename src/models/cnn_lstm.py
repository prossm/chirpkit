import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like attention"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)

class CNNLSTMInsectClassifier(nn.Module):
    """Enhanced CNN-LSTM hybrid model for insect classification"""
    def __init__(self, n_classes: int = 12, cnn_channels: list = [32, 64, 128, 256], lstm_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes
        
        # Enhanced CNN with residual connections and squeeze-excitation
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for i, out_channels in enumerate(cnn_channels):
            # Main conv block
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels)
            )
            
            # Squeeze-and-Excitation block
            se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 16, out_channels, 1),
                nn.Sigmoid()
            )
            
            # Residual connection (if dimensions match)
            if in_channels == out_channels:
                residual = nn.Identity()
            else:
                residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
            self.conv_layers.append(nn.ModuleDict({
                'conv': conv_block,
                'se': se_block,
                'residual': residual,
                'pool': nn.MaxPool2d(2, 2),
                'dropout': nn.Dropout2d(dropout)
            }))
            in_channels = out_channels
        
        # Enhanced LSTM with layer normalization
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Layer normalization for LSTM output
        self.lstm_norm = nn.LayerNorm(lstm_hidden * 2)
        
        # Positional encoding for attention
        self.pos_encoding = PositionalEncoding(lstm_hidden * 2)
        
        # Multi-head self-attention with multiple layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=lstm_hidden * 2,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)
        ])
        
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(lstm_hidden * 2) for _ in range(2)
        ])
        
        # Enhanced classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        batch_size = x.size(0)
        
        # Enhanced CNN forward pass with residual connections and SE
        for conv_dict in self.conv_layers:
            residual_input = x
            
            # Main convolution
            x = conv_dict['conv'](x)
            
            # Squeeze-and-Excitation
            se_weights = conv_dict['se'](x)
            x = x * se_weights
            
            # Residual connection
            residual = conv_dict['residual'](residual_input)
            x = F.relu(x + residual)
            
            # Pooling and dropout
            x = conv_dict['pool'](x)
            x = conv_dict['dropout'](x)
        
        # Global average pooling over frequency dimension
        x = x.mean(dim=2)  # [batch, channels, time]
        x = x.transpose(1, 2)  # [batch, time, channels]
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Add positional encoding
        lstm_out = self.pos_encoding(lstm_out)
        
        # Multi-layer self-attention with residual connections
        attended = lstm_out
        for attn_layer, norm_layer in zip(self.attention_layers, self.attention_norms):
            residual = attended
            attended, _ = attn_layer(attended, attended, attended)
            attended = norm_layer(attended + residual)  # Residual connection
        
        # Global attention pooling - weighted average instead of simple mean
        attention_weights = torch.softmax(
            torch.sum(attended * attended, dim=-1), dim=1
        ).unsqueeze(-1)
        features = torch.sum(attended * attention_weights, dim=1)
        
        # Classification
        output = self.classifier(features)
        return output
