from torch import nn


class Attention(nn.Module):
    """
    Multihead Attention
    """

    def __init__(self, embed_dim=16, n_heads=4, attention_bias=False, proj_bias=False) -> None:
        super(Attention, self).__init__()

        assert embed_dim % n_heads == 0, f"Can't divide embed size {embed_dim} by number of heads {n_heads}"

        self.n_heads = n_heads
        self.scaling_factor = n_heads ** -0.5
        self.linear_proj = nn.Linear(embed_dim, embed_dim, proj_bias)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, attention_bias)

    def forward(self, x):
        # b: batch size, n: number of patches
        b, n, _ = x.shape

        # calculate qkv in the same tensor
        # then, reshape to split query, keys and values and to
        # distribute values across heads
        x = self.qkv(x)
        q, k, v = x.reshape(b, n, 3, self.n_heads, -1) \
            .permute(2, 0, 3, 1, 4) \
            .reshape(3, b * self.n_heads, n, -1) \
            .unbind(0)

        attention = (q @ k.transpose(-2, -1)) * self.scaling_factor
        attention = attention.softmax(dim=-1) @ v
        attention = attention.reshape(b, n, -1)

        return self.linear_proj(attention)


class Block(nn.Module):
    def __init__(self, embed_dim=16, n_heads=4, mlp_ratio=4, attention_bias=False):
        super(Block, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.attention = Attention(embed_dim, n_heads, attention_bias)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
