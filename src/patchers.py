from torch import nn


class Patcher(nn.Module):
    """
    Create patches and linear maps them
    """

    def __init__(self, img_shape=(1, 28, 28), patch_size=4, embed_dim=16, bias=False):
        super(Patcher, self).__init__()

        c, h, w = img_shape

        assert h == w, "Image has to have the same width and height"
        assert h % patch_size == 0, f"Can't divide image dimension {h} by patch size {patch_size}"

        self.input_dim = patch_size * patch_size * c
        self.linear_map = nn.Linear(self.input_dim, embed_dim, bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.input_dim)
        return self.linear_map(x)
