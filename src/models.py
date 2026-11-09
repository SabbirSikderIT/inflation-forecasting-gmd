"""Model definitions: CNN-LSTM-Attention and baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Temporal attention over LSTM outputs."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, lstm_out):
        scores = self.attn_weights(lstm_out)          # (B, T, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_out, dim=1)  # (B, H)
        return context, weights


class CNN_LSTM_Attention(nn.Module):
    """
    Hybrid architecture:
      CNN (local patterns) -> LSTM (temporal) -> Attention -> Dense
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        cnn_channels=(32, 64),
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        
        # CNN block
        self.conv1 = nn.Conv1d(n_features, cnn_channels[0], 3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], 3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        self.dropout = nn.Dropout(dropout)
        
        # LSTM block
        self.lstm = nn.LSTM(
            cnn_channels[1], lstm_hidden, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Attention + head
        self.attention = TemporalAttention(lstm_hidden)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (B, T, F) -> CNN wants (B, F, T)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Back to (B, T, C) for LSTM
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        
        # Attention
        context, attn_weights = self.attention(lstm_out)
        
        # Dense head
        out = self.fc(context).squeeze(-1)
        return out, attn_weights