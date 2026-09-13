import torch
from torch import nn


class DependencyScorer(nn.Module):
    """Lightweight query-conditioned scorer for reliable token dependencies."""

    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim)
        self.k_proj = nn.Linear(dim, n_heads * head_dim)
        self.gate = nn.Linear(dim, n_heads)

        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, token_count, _ = x.shape
        q = self.q_proj(x).view(bsz, token_count, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, token_count, self.n_heads, self.head_dim).transpose(1, 2)
        gate = self.gate(x).transpose(1, 2).unsqueeze(2)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        scores = torch.relu(scores) * gate
        return scores.mean(dim=1)

    def topk(self, x: torch.Tensor, k: int) -> torch.Tensor:
        scores = self.forward(x)
        keep = min(max(1, int(k)), scores.shape[-1])
        return torch.topk(scores, k=keep, dim=-1).indices
