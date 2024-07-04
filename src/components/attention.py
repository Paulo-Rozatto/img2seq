import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    Multihead Attention
    """

    def __init__(self, embed_dim=16, n_heads=4, attention_bias=False, proj_bias=False) -> None:
        super(SelfAttention, self).__init__()

        assert embed_dim % n_heads == 0, f"Can't divide embed size {embed_dim} by number of heads {n_heads}"

        self.n_heads = n_heads
        self.scaling_factor = n_heads ** -0.5
        self.linear_proj = nn.Linear(embed_dim, embed_dim, proj_bias)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, attention_bias)

    def forward(self, x, mask=None, pad_mask=None):
        # b: batch size, n: number of patches
        b, n, _ = x.shape

        # calculate qkv in the same tensor
        # then, reshape to split query, keys and values and to
        # distribute values across heads
        x = self.qkv(x)

        # if pad_mask is not None:
        #     x[:, 1:][~pad_mask] = torch.zeros_like(x[0, 0])

        q, k, v = x.reshape(b, n, 3, self.n_heads, -1) \
            .permute(2, 0, 3, 1, 4) \
            .reshape(3, b * self.n_heads, n, -1) \
            .unbind(0)
        
        # if pad_mask is not None:
        #     k[:, 1:][~pad_mask] = torch.zeros_like(k[0, 0])


        attention = (q @ k.transpose(-2, -1)) * self.scaling_factor

        if pad_mask is not None:
            attention[:, :, 1:].transpose(-2, -1)[~pad_mask] = float("-inf")
            # attention[:, 1:, :][~pad_mask] = float("-inf")
            # attention = attention.masked_fill(attention == 0.0, float("-inf"))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = attention.softmax(dim=-1) @ v
        attention = attention.reshape(b, n, -1)

        return self.linear_proj(attention)


class Attention(nn.Module):
    """
    Multihead Attention
    """

    def __init__(self, embed_dim=16, n_heads=4, proj_bias=False) -> None:
        super(Attention, self).__init__()

        assert embed_dim % n_heads == 0, f"Can't divide embed size {embed_dim} by number of heads {n_heads}"

        self.n_heads = n_heads
        self.scaling_factor = n_heads ** -0.5
        self.linear_proj = nn.Linear(embed_dim, embed_dim, proj_bias)

    def forward(self, qvk, mask=None):
        # b: batch size, n: number of patches
        b, n, _, _ = qvk.shape

        q, k, v = qvk.reshape(b, n, 3, self.n_heads, -1) \
            .permute(2, 0, 3, 1, 4) \
            .reshape(3, b * self.n_heads, n, -1) \
            .unbind(0)

        attention = (q @ k.transpose(-2, -1)) * self.scaling_factor

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = attention.softmax(dim=-1) @ v
        attention = attention.reshape(b, n, -1)

        return self.linear_proj(attention)


class Block(nn.Module):
    def __init__(self, embed_dim=16, n_heads=4, mlp_ratio=4, attention_bias=False):
        super(Block, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.1)

        self.attention = SelfAttention(embed_dim, n_heads, attention_bias)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        x2 = self.attention(self.norm1(x))
        x2 = self.dropout(x2)
        x = x + x2
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim=3, encoder_dim=4, n_heads=1, mlp_ratio=4, attention_bias=False):
        super(DecoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(encoder_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.1)

        self.q = nn.Linear(encoder_dim, encoder_dim, bias=attention_bias)
        self.kv = nn.Linear(encoder_dim, 2 * encoder_dim)

        self.selfAttention = SelfAttention(embed_dim, n_heads, attention_bias)
        self.attention = Attention(encoder_dim, n_heads, attention_bias)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.ReLU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.ReLU()
        )

    def forward(self, x, encoder_embed, mask=None, pad_mask=None):
        b, n, _ = x.shape

        x2 = self.selfAttention(x, mask, pad_mask)
        x2 = self.dropout(x)
        x = self.norm1(x + x2)

        q = self.q(x).view(b, n, 1, -1)

        vk = self.kv(encoder_embed).view(b, 1, 2, -1)
        qkv = torch.cat((q, vk.repeat((1, n, 1, 1))), dim=2)

        x = self.norm2(x + self.attention(qkv))
        x = self.norm3(x + self.mlp(x))

        return x
