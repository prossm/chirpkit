import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNLSTMInsectClassifier(nn.Module):
    """Simple CNN-LSTM model that actually works"""
    def __init__(self, n_classes: int = 12, dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes
        
        # Simple CNN layers (like your original working model)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            )
        ])
        
        # Simple LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Simple attention (just one layer)
        self.attention = nn.MultiheadAttention(
            embed_dim=512,  # 256 * 2 (bidirectional)
            num_heads=8,
            dropout=dropout
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x, return_attention=False):
        batch_size = x.size(0)
        
        # Simple CNN forward pass
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global average pooling over frequency dimension
        x = x.mean(dim=2)  # [batch, channels, time]
        x = x.transpose(1, 2)  # [batch, time, channels]
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Simple attention with weights returned
        lstm_out = lstm_out.transpose(0, 1)  # [seq, batch, features]
        attended, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attended = attended.transpose(0, 1)  # [batch, seq, features]
        
        # Simple pooling
        features = attended.mean(dim=1)  # Average over time
        
        # Classification
        output = self.classifier(features)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def compute_attention_diversity_loss(self, attention_weights):
        """
        Compute attention diversity loss to encourage different heads to focus on different regions.
        
        Args:
            attention_weights: Tensor of shape [batch_size, num_heads, seq_len, seq_len]
        
        Returns:
            diversity_loss: Scalar tensor encouraging attention head diversity
        """
        # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Average attention across the sequence dimension for each head
        # Focus on how much each head attends to each position
        head_attention_patterns = attention_weights.mean(dim=-1)  # [batch, heads, seq_len]
        
        # Compute similarity between attention heads
        head_similarities = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                # Cosine similarity between attention patterns of heads i and j
                head_i = head_attention_patterns[:, i, :]  # [batch, seq_len]
                head_j = head_attention_patterns[:, j, :]  # [batch, seq_len]
                
                # Normalize for cosine similarity
                head_i_norm = F.normalize(head_i, p=2, dim=1)
                head_j_norm = F.normalize(head_j, p=2, dim=1)
                
                # Cosine similarity (higher = more similar = bad for diversity)
                similarity = (head_i_norm * head_j_norm).sum(dim=1)  # [batch]
                head_similarities.append(similarity)
        
        # Stack all pairwise similarities
        if head_similarities:
            similarities = torch.stack(head_similarities, dim=1)  # [batch, num_pairs]
            
            # Diversity loss: penalize high similarities (encourage low similarity)
            diversity_loss = similarities.mean()
        else:
            # No pairs to compare (single head)
            diversity_loss = torch.tensor(0.0, device=attention_weights.device)
        
        return diversity_loss