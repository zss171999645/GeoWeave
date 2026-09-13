import torch
from torch import nn

from .dependency_scorer import DependencyScorer


class GeoWeaveAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dependency_heads: int,
        dependency_head_dim: int,
        topk: int,
        use_dependency_selection: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk = int(topk)
        self.use_dependency_selection = use_dependency_selection

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dependency_scorer = DependencyScorer(dim, dependency_heads, dependency_head_dim)

    def _project_qkv(self, x: torch.Tensor):
        bsz, token_count, dim = x.shape
        qkv = self.qkv(x).view(bsz, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _dense_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return torch.matmul(attn, v)

    def _selected_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        bsz, heads, token_count, head_dim = q.shape
        keep = selected.shape[-1]
        gather_index = selected[:, None, :, :, None].expand(bsz, heads, token_count, keep, head_dim)
        k_selected = torch.gather(
            k[:, :, None, :, :].expand(bsz, heads, token_count, token_count, head_dim),
            dim=3,
            index=gather_index,
        )
        v_selected = torch.gather(
            v[:, :, None, :, :].expand(bsz, heads, token_count, token_count, head_dim),
            dim=3,
            index=gather_index,
        )
        logits = (q.unsqueeze(3) * k_selected).sum(dim=-1) * self.scale
        attn = logits.softmax(dim=-1)
        return (attn.unsqueeze(-1) * v_selected).sum(dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        if self.use_dependency_selection:
            selected = self.dependency_scorer.topk(x, self.topk)
            out = self._selected_attention(q, k, v, selected)
        else:
            out = self._dense_attention(q, k, v)
        out = out.transpose(1, 2).reshape(x.shape)
        return self.proj(out)
