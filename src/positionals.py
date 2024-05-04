import torch
from torch import nn


class LearnableEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=16):
        super(LearnableEncoder, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positional = nn.Parameter(
            torch.randn(input_dim, embed_dim).to(device))

    def forward(self, x):
        b, n, _ = x.shape
        pos = self.positional.repeat(b, 1, 1)
        x = x + pos[:, :n, :]
        return x
