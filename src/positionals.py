from torch import nn, randn


class LearnableEncoder(nn.Module):
    def __init__(self, embed_dim=16):
        super(LearnableEncoder, self).__init__()
        self.positional = nn.Parameter(randn(embed_dim))

    def forward(self, x):
        return x + self.positional
