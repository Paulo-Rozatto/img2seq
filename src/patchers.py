from torch import nn


class Patcher(nn.Module):
    """
    Create patches and linear maps them
    """

    def __init__(self, input_dim, embed_dim=16, bias=False):
        super(Patcher, self).__init__()

        self.input_dim = input_dim
        self.linear_map = nn.Linear(self.input_dim, embed_dim, bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.input_dim)
        return self.linear_map(x)
