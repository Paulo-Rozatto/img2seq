import torch
from torch import nn

from .attention import Block, DecoderBlock
from .patchers import ConvPatcher
from .positionals import LearnableEncoder, PositionalEncoding


class ViT(nn.Module):
    """
    Vision Transformer
    """

    def __init__(
        self,
        patcher=ConvPatcher,
        positional_encoder=LearnableEncoder,
        block=Block,
        embed_dim=16,
        n_blocks=2,
        n_heads=4,
        img_shape=(1, 28, 28),
        patch_size=7,
        mlp_ratio=4,
        attention_bias=False,
        patch_bias=False
    ):
        super(ViT, self).__init__()
    
        self.class_token = nn.Parameter(
            torch.empty(1, embed_dim).normal_(std=0.02)
        )

        _, h, w = img_shape
        n_patches = h * w // (patch_size * patch_size)

        self.patcher = patcher(img_shape, patch_size, embed_dim, patch_bias)
        self.positional_encoder = positional_encoder(n_patches + 1, embed_dim)
        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            self.blocks.append(
                block(embed_dim, n_heads, mlp_ratio, attention_bias)
            )

    def forward(self, x):
        x = self.patcher(x)

        token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((token, x), 1)

        x = self.positional_encoder(x)

        for block in self.blocks:
            x = block(x)

        return x


class Decoder(nn.Module):
    """
    Vision Transformer
    """

    def __init__(
        self,
        positional_encoder=PositionalEncoding,
        block=DecoderBlock,
        embed_dim=16,
        n_blocks=2,
        n_heads=1,
        seq_len=12,
        mlp_ratio=4,
        attention_bias=False
    ):
        super(Decoder, self).__init__()

        self.positional_encoder = positional_encoder(seq_len, embed_dim)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                block(embed_dim, embed_dim, n_heads,
                      mlp_ratio, attention_bias)
            )

    def forward(self, x, encoder_embed, mask=None, pad_mask=None):
        x = self.positional_encoder(x)

        for block in self.blocks:
            x = block(x, encoder_embed, mask, pad_mask)

        return x
