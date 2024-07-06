import math
import torch
from torch import nn


class LearnableEncoder(nn.Module):
    def __init__(self, n_patches, embed_dim=16, dropout=0.1):
        super(LearnableEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.positional = nn.Parameter(
            torch.empty(1, n_patches, embed_dim).normal_(std=0.02)
        )

    def forward(self, x):
        x = x + self.positional
        return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len=200, embed_dim=16, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        positional = torch.zeros(max_seq_len, 1, embed_dim)
        positional[:, 0, 0::2] = torch.sin(position * div_term)
        positional[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional', positional)

    def forward(self, x):
        x = x + self.positional[:x.shape[0]]
        return self.dropout(x)
