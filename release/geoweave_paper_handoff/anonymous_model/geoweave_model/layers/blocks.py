from torch import nn

from .sparse_attention import GeoWeaveAttention


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class GeoWeaveBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dependency_heads: int,
        dependency_head_dim: int,
        topk: int,
        use_dependency_selection: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GeoWeaveAttention(
            dim=dim,
            num_heads=num_heads,
            dependency_heads=dependency_heads,
            dependency_head_dim=dependency_head_dim,
            topk=topk,
            use_dependency_selection=use_dependency_selection,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
