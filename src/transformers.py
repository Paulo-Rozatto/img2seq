from torch import nn

# from attention import Block
from .attention import Block
from .patchers import Patcher
from .positionals import LearnableEncoder


class ViT(nn.Module):
    """
    Vision Transformer
    """

    def __init__(
        self,
        patcher=Patcher,
        positional_encoder=LearnableEncoder,
        block=Block,
        embed_dim=16,
        n_blocks=2,
        n_heads=4,
        img_shape=(1, 28, 28),
        patch_size=4,
        mlp_ratio=4,
        attention_bias=False,
        patch_bias=False
    ):
        super(ViT, self).__init__()

        self.patcher = patcher(img_shape, patch_size, embed_dim, patch_bias)
        self.positional_encoder = positional_encoder(embed_dim)
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            self.blocks.append(
                block(embed_dim, n_heads, mlp_ratio, attention_bias)
            )

    def forward(self, x):
        x = self.patcher(x)
        x = self.positional_encoder(x)

        for block in self.blocks:
            x = block(x)

        return x
