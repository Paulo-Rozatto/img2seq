from torch import nn


class Patcher(nn.Module):
    """
    Create patches and linear maps them
    """

    def __init__(self, img_shape, patch_size, embed_dim=16, bias=False):
        super(Patcher, self).__init__()

        c, _, _ = img_shape
        input_dim = patch_size * patch_size * c

        self.input_dim = input_dim
        self.linear_map = nn.Linear(self.input_dim, embed_dim, bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.input_dim)
        return self.linear_map(x)


class ConvPatcher(nn.Module):
    def __init__(self, img_shape, patch_size, embed_dim, bias=False):
        super(ConvPatcher, self).__init__()

        c, _, _ = img_shape
        self.proj = nn.Conv2d(
            c, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x
