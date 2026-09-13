import json
import os
import torch
from torch import nn
from typing import Optional, Tuple, Union

from easyvolcap.utils.dist_utils import get_rank, is_main_process
from easyvolcap.utils.custom_indexer.streaming_kl_autograd import streaming_kl_autograd_loss
from easyvolcap.utils.custom_indexer.topk_support_autograd import topk_support_autograd_loss
try:
    from easyvolcap.utils.custom_indexer import sparse_topk_indexer_func, sparse_topk_indexer_inference_func
except Exception:
    sparse_topk_indexer_func = None
    sparse_topk_indexer_inference_func = None

class LightningIndexer(nn.Module):
    """Lightning indexer for DeepSeek-style sparse attention.

    Computes index scores I_{t,s} with a lightweight multi-head dot product and ReLU,
    used to select top-k tokens for sparse attention.
    """
    _mem_snapshot_count = 0
    _mem_snapshot_peak_alloc = 0
    _mem_snapshot_peak_reserved = 0
    _kl_debug_count = 0
    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        head_dim: int = 64,
        rope=None,
        qk_norm: bool = False,
        score_dtype: Optional[Union[str, torch.dtype]] = None,
        score_head_chunk_size: Optional[int] = 0,
        score_key_chunk_size: Optional[int] = 0,
        streaming_kl_autograd: bool = False,
        use_topk_kernel: bool = False,
        topk_block: int = 256,
        topk_merge_blocks: int = 0,
        soft_view_bias_training_kernel_enabled: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope = rope
        self.layer_idx = None
        self.score_dtype = self._parse_dtype(score_dtype)
        self.score_head_chunk_size = self._parse_chunk_size(score_head_chunk_size)
        self.score_key_chunk_size = self._parse_chunk_size(score_key_chunk_size)
        self.streaming_kl_autograd = bool(streaming_kl_autograd)
        self.use_topk_kernel = bool(use_topk_kernel)
        self.topk_block = int(topk_block)
        self.topk_merge_blocks = int(topk_merge_blocks)
        self.soft_view_bias_training_kernel_enabled = bool(soft_view_bias_training_kernel_enabled)

        self.q_proj = nn.Linear(dim, n_heads * head_dim)
        self.k_proj = nn.Linear(dim, n_heads * head_dim)
        self.w_proj = nn.Linear(dim, n_heads)
        self._reset_score_gate()
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else nn.Identity()
        self.scale = head_dim ** -0.5

    def _reset_score_gate(self) -> None:
        # w_proj is a multiplicative score gate, not a standalone predictor.
        # Starting near zero makes scratch indexer KL nearly uniform with tiny gradients.
        nn.init.zeros_(self.w_proj.weight)
        if self.w_proj.bias is not None:
            nn.init.ones_(self.w_proj.bias)

    @staticmethod
    def _mib(bytes_value: int) -> float:
        return bytes_value / 2**20

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        if num_bytes >= 2**30:
            return f"{num_bytes / 2**30:.2f} GiB"
        return f"{num_bytes / 2**20:.1f} MiB"

    @staticmethod
    def _parse_dtype(value: Optional[object]) -> Optional[torch.dtype]:
        if value is None:
            return None
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("", "none"):
                return None
            if text in ("fp16", "float16", "half", "f16"):
                return torch.float16
            if text in ("bf16", "bfloat16"):
                return torch.bfloat16
            if text in ("fp32", "float32", "f32"):
                return torch.float32
        return None

    @staticmethod
    def _parse_chunk_size(value: Optional[object]) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _resolve_chunk_size(chunk_size: int, full_size: int) -> int:
        if full_size <= 0:
            return 0
        if chunk_size <= 0:
            return int(full_size)
        return max(1, min(int(chunk_size), int(full_size)))

    def _scale_like(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(self.scale)

    @staticmethod
    def _env_flag(name: str) -> bool:
        value = os.getenv(name, "")
        return value.lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return int(default)

    def _mem_snapshot_rank_ok(self) -> bool:
        if self._env_flag("VGGT_MEM_TRACE_ALL_RANKS"):
            return True
        return is_main_process()

    def _mem_snapshot_enabled(self) -> bool:
        return self._env_flag("VGGT_MEM_SNAPSHOT")

    @staticmethod
    def _mem_snapshot_max() -> int:
        try:
            return int(os.getenv("VGGT_MEM_SNAPSHOT_MAX", "20"))
        except ValueError:
            return 20

    @staticmethod
    def _mem_snapshot_min_delta_bytes() -> int:
        try:
            return int(float(os.getenv("VGGT_MEM_SNAPSHOT_MIN_DELTA_MIB", "0")) * 2**20)
        except ValueError:
            return 0

    @staticmethod
    def _mem_snapshot_dir() -> str:
        output_dir = os.getenv("VGGT_MEM_SNAPSHOT_DIR") or os.getenv("CUDA_MEM_TRACE_OUTPUT_DIR") or "cuda_mem"
        return os.path.abspath(output_dir)

    @staticmethod
    def _kl_debug_max() -> int:
        try:
            return int(os.getenv("VGGT_INDEXER_KL_DEBUG_MAX", "8"))
        except ValueError:
            return 8

    def _maybe_mem_snapshot(self, tag: str) -> None:
        if not self._mem_snapshot_enabled():
            return
        if not torch.cuda.is_available() or not self._mem_snapshot_rank_ok():
            return
        max_snapshots = self._mem_snapshot_max()
        if max_snapshots > 0 and self.__class__._mem_snapshot_count >= max_snapshots:
            return
        if self._env_flag("VGGT_MEM_SNAPSHOT_SYNC"):
            torch.cuda.synchronize()
        cur_alloc = int(torch.cuda.memory_allocated())
        cur_reserved = int(torch.cuda.memory_reserved())
        min_delta = self._mem_snapshot_min_delta_bytes()
        if (
            cur_alloc <= self.__class__._mem_snapshot_peak_alloc
            and cur_reserved <= self.__class__._mem_snapshot_peak_reserved
        ):
            return
        if (
            cur_alloc - self.__class__._mem_snapshot_peak_alloc < min_delta
            and cur_reserved - self.__class__._mem_snapshot_peak_reserved < min_delta
        ):
            return
        self.__class__._mem_snapshot_peak_alloc = max(self.__class__._mem_snapshot_peak_alloc, cur_alloc)
        self.__class__._mem_snapshot_peak_reserved = max(
            self.__class__._mem_snapshot_peak_reserved, cur_reserved
        )

        snapshot_fn = getattr(torch.cuda, "memory_snapshot", None)
        if snapshot_fn is None:
            return
        try:
            snapshot = snapshot_fn()
        except Exception as exc:
            return

        if self._env_flag("VGGT_MEM_SNAPSHOT_SAVE"):
            out_dir = self._mem_snapshot_dir()
            os.makedirs(out_dir, exist_ok=True)
            filename = f"snapshot_indexer_{tag}_rank{get_rank()}_{self.__class__._mem_snapshot_count:03d}.json"
            path = os.path.join(out_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=True)

        self.__class__._mem_snapshot_count += 1

    def _maybe_debug_kl_loss(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w_by_head: torch.Tensor,
        p: torch.Tensor,
        norm_mask: Optional[torch.Tensor],
        score_head_chunk_size: int,
        score_key_chunk_size: int,
        scale: torch.Tensor,
        eps: float,
        view_bias_data: Optional[dict] = None,
    ) -> None:
        if not self._env_flag("VGGT_INDEXER_KL_DEBUG"):
            return
        if not is_main_process():
            return
        max_debug = self._kl_debug_max()
        if max_debug >= 0 and self.__class__._kl_debug_count >= max_debug:
            return

        bsz, tgt_len, src_len = p.shape
        accum_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        row_sum_p = torch.zeros((bsz, tgt_len), device=p.device, dtype=accum_dtype)
        p_log_p = torch.zeros((bsz, tgt_len), device=p.device, dtype=accum_dtype)
        expected_score = torch.zeros((bsz, tgt_len), device=p.device, dtype=accum_dtype)
        log_denom = torch.full((bsz, tgt_len), float("-inf"), device=p.device, dtype=accum_dtype)
        score_sum = torch.zeros((), device=p.device, dtype=torch.float32)
        score_sq_sum = torch.zeros((), device=p.device, dtype=torch.float32)
        score_count = 0

        with torch.no_grad():
            for s_start in range(0, src_len, score_key_chunk_size):
                s_end = min(s_start + score_key_chunk_size, src_len)
                score_chunk = self._compute_score_chunk(
                    q=q,
                    k=k,
                    w_by_head=w_by_head,
                    s_start=s_start,
                    s_end=s_end,
                    score_head_chunk_size=score_head_chunk_size,
                    scale=scale,
                    view_bias_data=view_bias_data,
                )
                if norm_mask is not None:
                    score_chunk = score_chunk + norm_mask[:, :, s_start:s_end]

                score_float = score_chunk.float()
                finite_score = torch.isfinite(score_float)
                if finite_score.any():
                    finite_values = score_float[finite_score]
                    score_sum = score_sum + finite_values.sum()
                    score_sq_sum = score_sq_sum + (finite_values * finite_values).sum()
                    score_count += int(finite_values.numel())

                p_chunk = p[:, :, s_start:s_end]
                p_chunk_acc = p_chunk.to(accum_dtype)
                row_sum_p = row_sum_p + p_chunk_acc.sum(dim=-1)
                p_log_p = p_log_p + (p_chunk_acc * torch.log(p_chunk_acc + float(eps))).sum(dim=-1)
                expected_score = expected_score + (p_chunk * score_chunk).sum(dim=-1, dtype=accum_dtype)
                log_denom = torch.logaddexp(log_denom, torch.logsumexp(score_chunk, dim=-1).to(accum_dtype))

            loss = (p_log_p - expected_score + row_sum_p * log_denom).mean().float()
            entropy = (-p_log_p).mean().float()
            uniform_entropy = torch.log(p.new_tensor(float(max(src_len, 1)))).float()
            row_mean = row_sum_p.mean().float()
            row_min = row_sum_p.min().float()
            row_max = row_sum_p.max().float()
            if score_count > 0:
                score_mean = score_sum / float(score_count)
                score_var = torch.clamp(score_sq_sum / float(score_count) - score_mean * score_mean, min=0.0)
                score_std = torch.sqrt(score_var)
            else:
                score_mean = score_sum
                score_std = score_sum

            print(
                "[indexer-kl-debug] "
                f"layer={self.layer_idx} shape=({bsz},{tgt_len},{src_len}) "
                f"loss={float(loss.item()):.9e} entropy={float(entropy.item()):.9e} "
                f"uniform_entropy={float(uniform_entropy.item()):.9e} "
                f"row_sum_mean={float(row_mean.item()):.9e} "
                f"row_sum_min={float(row_min.item()):.9e} "
                f"row_sum_max={float(row_max.item()):.9e} "
                f"score_mean={float(score_mean.item()):.9e} "
                f"score_std={float(score_std.item()):.9e} "
                f"q_requires_grad={q.requires_grad} k_requires_grad={k.requires_grad} w_requires_grad={w_by_head.requires_grad}",
                flush=True,
            )

        self.__class__._kl_debug_count += 1

    def project(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_norm(k)
        w = self.w_proj(x).view(bsz, seq_len, self.n_heads)

        if self.rope is not None and pos is not None:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            q = self.rope(q, pos)
            k = self.rope(k, pos)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)

        if self.score_dtype is not None and (
            q.dtype != self.score_dtype or k.dtype != self.score_dtype or w.dtype != self.score_dtype
        ):
            q = q.to(self.score_dtype)
            k = k.to(self.score_dtype)
            w = w.to(self.score_dtype)

        return q, k, w

    def project_query(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)

        if self.rope is not None and pos is not None:
            q = q.permute(0, 2, 1, 3)
            q = self.rope(q, pos)
            q = q.permute(0, 2, 1, 3)

        if self.score_dtype is not None and self.score_dtype != q.dtype:
            q = q.to(self.score_dtype)
        return q

    def project_key_weight(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_norm(k)
        w = self.w_proj(x).view(bsz, seq_len, self.n_heads)

        if self.rope is not None and pos is not None:
            k = k.permute(0, 2, 1, 3)
            k = self.rope(k, pos)
            k = k.permute(0, 2, 1, 3)

        if self.score_dtype is not None and (
            k.dtype != self.score_dtype or w.dtype != self.score_dtype
        ):
            k = k.to(self.score_dtype)
            w = w.to(self.score_dtype)
        return k, w

    @staticmethod
    def _normalize_mask(mask: Optional[torch.Tensor], bsz: int, tgt_len: int, src_len: int) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3 and mask.size(0) == 1 and bsz > 1:
            mask = mask.expand(bsz, -1, -1)
        if mask.dim() != 3:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        if mask.size(-2) != tgt_len or mask.size(-1) != src_len:
            raise ValueError(f"Mask shape {mask.shape} does not match ({bsz}, {tgt_len}, {src_len})")
        return mask

    def _compute_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        mask: Optional[torch.Tensor],
        view_bias_data: Optional[dict] = None,
    ) -> torch.Tensor:
        bsz, tgt_len, n_heads, _ = q.shape
        src_len = int(k.shape[1])
        score_head_chunk_size = self._resolve_chunk_size(self.score_head_chunk_size, n_heads)
        score_key_chunk_size = self._resolve_chunk_size(self.score_key_chunk_size, src_len)
        scale = self._scale_like(q)
        w_by_head = w.permute(0, 2, 1)
        norm_view_bias = self._normalize_view_bias_data(view_bias_data, bsz, tgt_len, src_len, q.device, q.dtype)
        scores = q.new_zeros((bsz, tgt_len, src_len))
        for s_start in range(0, src_len, score_key_chunk_size):
            s_end = min(s_start + score_key_chunk_size, src_len)
            scores[:, :, s_start:s_end] = self._compute_score_chunk(
                q=q,
                k=k,
                w_by_head=w_by_head,
                s_start=s_start,
                s_end=s_end,
                score_head_chunk_size=score_head_chunk_size,
                scale=scale,
                view_bias_data=norm_view_bias,
            )
        if mask is not None:
            if mask.dtype != q.dtype:
                mask = mask.to(q.dtype)
            scores = scores + mask
        if (
            self._mem_snapshot_enabled()
            and self._env_flag("VGGT_MEM_SNAPSHOT_BWD")
            and scores.requires_grad
        ):
            layer_idx = self.layer_idx
            seq_val = tgt_len
            src_val = src_len
            def _scores_bwd_hook(grad: torch.Tensor) -> torch.Tensor:
                _ = grad
                self._maybe_mem_snapshot(f"bwd_scores_layer{layer_idx}_{seq_val}x{src_val}")
                return grad
            scores.register_hook(_scores_bwd_hook)
        return scores

    def _compute_score_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w_by_head: torch.Tensor,
        s_start: int,
        s_end: int,
        score_head_chunk_size: int,
        scale: torch.Tensor,
        view_bias_data: Optional[dict] = None,
    ) -> torch.Tensor:
        bsz, tgt_len, n_heads, _ = q.shape
        k_chunk = k[:, s_start:s_end]
        w_chunk = w_by_head[:, :, s_start:s_end]
        score_chunk = None

        for h_start in range(0, n_heads, score_head_chunk_size):
            h_end = min(h_start + score_head_chunk_size, n_heads)
            q_h = q[:, :, h_start:h_end]
            k_h = k_chunk[:, :, h_start:h_end]
            w_h = w_chunk[:, h_start:h_end]

            head_scores = torch.einsum("bthd,bshd->bths", q_h, k_h) * scale
            head_scores = torch.relu(head_scores)
            head_scores = head_scores * w_h.unsqueeze(1)
            head_scores = head_scores.sum(dim=2, dtype=q.dtype)
            score_chunk = head_scores if score_chunk is None else score_chunk + head_scores

        if score_chunk is None:
            score_chunk = q.new_zeros((bsz, tgt_len, s_end - s_start))
        if view_bias_data is not None:
            score_chunk = score_chunk + self._compute_view_bias_chunk(
                view_bias_data=view_bias_data,
                s_start=s_start,
                s_end=s_end,
            )
        return score_chunk

    @staticmethod
    def _normalize_view_bias_data(
        view_bias_data: Optional[dict],
        bsz: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[dict]:
        if view_bias_data is None:
            return None
        q_view_ids = view_bias_data.get("q_view_ids", None)
        s_view_ids = view_bias_data.get("s_view_ids", None)
        view_bias = view_bias_data.get("view_bias", None)
        if q_view_ids is None or s_view_ids is None or view_bias is None:
            return None
        if q_view_ids.dim() == 1:
            q_view_ids = q_view_ids.unsqueeze(0)
        if s_view_ids.dim() == 1:
            s_view_ids = s_view_ids.unsqueeze(0)
        if view_bias.dim() == 2:
            view_bias = view_bias.unsqueeze(0)
        if q_view_ids.size(0) == 1 and bsz > 1:
            q_view_ids = q_view_ids.expand(bsz, -1)
        if s_view_ids.size(0) == 1 and bsz > 1:
            s_view_ids = s_view_ids.expand(bsz, -1)
        if view_bias.size(0) == 1 and bsz > 1:
            view_bias = view_bias.expand(bsz, -1, -1)
        if q_view_ids.shape != (bsz, tgt_len):
            raise ValueError(f"q_view_ids shape {tuple(q_view_ids.shape)} mismatches ({bsz}, {tgt_len})")
        if s_view_ids.shape != (bsz, src_len):
            raise ValueError(f"s_view_ids shape {tuple(s_view_ids.shape)} mismatches ({bsz}, {src_len})")
        if view_bias.dim() != 3 or view_bias.shape[0] != bsz:
            raise ValueError(f"view_bias shape {tuple(view_bias.shape)} mismatches batch {bsz}")
        return dict(
            q_view_ids=q_view_ids.to(device=device, dtype=torch.long),
            s_view_ids=s_view_ids.to(device=device, dtype=torch.long),
            view_bias=view_bias.to(device=device, dtype=dtype),
        )

    @staticmethod
    def _compute_view_bias_chunk(
        view_bias_data: dict,
        s_start: int,
        s_end: int,
    ) -> torch.Tensor:
        q_view_ids = view_bias_data["q_view_ids"]
        s_view_ids = view_bias_data["s_view_ids"][:, s_start:s_end]
        view_bias = view_bias_data["view_bias"]
        bsz = int(q_view_ids.shape[0])
        batch_idx = torch.arange(bsz, device=view_bias.device, dtype=torch.long).view(bsz, 1, 1)
        return view_bias[batch_idx, q_view_ids.unsqueeze(-1), s_view_ids.unsqueeze(1)]

    def select_topk_projected(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
        return_scores: bool = True,
        view_bias_data: Optional[dict] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if q.dim() != 4 or k.dim() != 4 or w.dim() != 3:
            raise ValueError("q/k/w must be [B,T,H,D], [B,S,H,D], [B,S,H]")
        bsz, tgt_len, n_heads, head_dim = q.shape
        if k.shape[0] != bsz or k.shape[2] != n_heads or k.shape[3] != head_dim:
            raise ValueError(f"projected k shape {tuple(k.shape)} mismatches q {tuple(q.shape)}")
        if w.shape[0] != bsz or w.shape[1] != k.shape[1] or w.shape[2] != n_heads:
            raise ValueError(f"projected w shape {tuple(w.shape)} mismatches q {tuple(q.shape)} and k {tuple(k.shape)}")

        norm_mask = self._normalize_mask(mask, bsz, tgt_len, int(k.shape[1]))
        if norm_mask is not None and norm_mask.dtype != q.dtype:
            norm_mask = norm_mask.to(q.dtype)

        if topk is None:
            return self._compute_scores(q, k, w, norm_mask)

        topk = min(int(topk), int(k.shape[1]))
        use_soft_view_bias_training_kernel = (
            view_bias_data is not None
            and self.soft_view_bias_training_kernel_enabled
            and (
                torch.is_grad_enabled()
                or ((not q.requires_grad) and (not k.requires_grad) and (not w.requires_grad))
            )
        )
        use_inference_topk_direct = (
            sparse_topk_indexer_inference_func is not None
            and q.is_cuda
            and (not torch.is_grad_enabled())
            and (not q.requires_grad)
            and (not k.requires_grad)
            and (not w.requires_grad)
            and view_bias_data is None
            and self._env_flag("VGGT_INDEXER_TOPK_INFER_DIRECT")
        )
        if (
            use_inference_topk_direct
            and (self.use_topk_kernel or self._env_flag("VGGT_INDEXER_TOPK_KERNEL"))
        ):
            mask_tensor = norm_mask if norm_mask is not None else q.new_empty(0)
            topk_indices, topk_scores = sparse_topk_indexer_inference_func(
                q,
                k,
                w,
                mask_tensor,
                topk,
                float(self.scale),
                self.topk_block,
                self.topk_merge_blocks,
                outer_query_chunk=self._env_int("VGGT_INDEXER_TOPK_INFER_OUTER_QUERY_CHUNK", 0),
                return_scores=return_scores,
            )
        elif (
            sparse_topk_indexer_func is not None
            and q.is_cuda
            and (view_bias_data is None or use_soft_view_bias_training_kernel)
            and (self.use_topk_kernel or self._env_flag("VGGT_INDEXER_TOPK_KERNEL"))
        ):
            mask_tensor = norm_mask if norm_mask is not None else q.new_empty(0)
            topk_indices, topk_scores = sparse_topk_indexer_func(
                q,
                k,
                w,
                mask_tensor,
                topk,
                float(self.scale),
                self.topk_block,
                self.topk_merge_blocks,
                view_bias_data=view_bias_data if use_soft_view_bias_training_kernel else None,
            )
        else:
            scores = self._compute_scores(q, k, w, norm_mask, view_bias_data=view_bias_data)
            topk_indices = scores.topk(topk, dim=-1, sorted=False)[1]
            topk_scores = scores.gather(-1, topk_indices)
        return (topk_indices, topk_scores) if return_scores else topk_indices

    def compute_kl_loss(
        self,
        p: torch.Tensor,
        eps: float = 1e-6,
        x: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        projected: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        view_bias_data: Optional[dict] = None,
    ) -> torch.Tensor:
        if projected is None:
            if x is None:
                raise ValueError("Either x or projected must be provided for compute_kl_loss.")
            q, k, w = self.project(x, pos=pos)
        else:
            q, k, w = projected

        if p.dim() != 3:
            raise ValueError(f"Expected p shape [B, T, S], got {tuple(p.shape)}")
        bsz, tgt_len, src_len = p.shape
        if q.shape[0] != bsz or q.shape[1] != tgt_len:
            raise ValueError(f"q shape {tuple(q.shape)} mismatches p shape {tuple(p.shape)}")
        if k.shape[0] != bsz or k.shape[1] != src_len:
            raise ValueError(f"k shape {tuple(k.shape)} mismatches p shape {tuple(p.shape)}")
        if w.shape[0] != bsz or w.shape[1] != src_len or w.shape[2] != q.shape[2]:
            raise ValueError(f"w shape {tuple(w.shape)} mismatches q {tuple(q.shape)} and p {tuple(p.shape)}")

        norm_mask = self._normalize_mask(mask, bsz, tgt_len, src_len)
        if norm_mask is not None and norm_mask.dtype != q.dtype:
            norm_mask = norm_mask.to(q.dtype)

        n_heads = q.shape[2]
        score_head_chunk_size = self._resolve_chunk_size(self.score_head_chunk_size, n_heads)
        score_key_chunk_size = self._resolve_chunk_size(self.score_key_chunk_size, src_len)
        scale = self._scale_like(q)
        w_by_head = w.permute(0, 2, 1)
        accum_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        use_streaming_kl_autograd = self.streaming_kl_autograd or self._env_flag("VGGT_STREAMING_KL_AUTOGRAD")
        norm_view_bias = self._normalize_view_bias_data(view_bias_data, bsz, tgt_len, src_len, q.device, q.dtype)
        self._maybe_debug_kl_loss(
            q=q,
            k=k,
            w_by_head=w_by_head,
            p=p,
            norm_mask=norm_mask,
            score_head_chunk_size=score_head_chunk_size,
            score_key_chunk_size=score_key_chunk_size,
            scale=scale,
            eps=float(eps),
            view_bias_data=norm_view_bias,
        )

        if use_streaming_kl_autograd and q.requires_grad:
            return streaming_kl_autograd_loss(
                q=q,
                k=k,
                w=w,
                p=p,
                mask=norm_mask,
                scale=float(self.scale),
                score_head_chunk_size=score_head_chunk_size,
                score_key_chunk_size=score_key_chunk_size,
                eps=float(eps),
            )

        row_sum_p = torch.zeros((bsz, tgt_len), device=p.device, dtype=accum_dtype)
        p_log_p = torch.zeros((bsz, tgt_len), device=p.device, dtype=accum_dtype)
        expected_score = torch.zeros((bsz, tgt_len), device=p.device, dtype=accum_dtype)
        log_denom = torch.full((bsz, tgt_len), float("-inf"), device=p.device, dtype=accum_dtype)

        for s_start in range(0, src_len, score_key_chunk_size):
            s_end = min(s_start + score_key_chunk_size, src_len)
            score_chunk = self._compute_score_chunk(
                q=q,
                k=k,
                w_by_head=w_by_head,
                s_start=s_start,
                s_end=s_end,
                score_head_chunk_size=score_head_chunk_size,
                scale=scale,
                view_bias_data=norm_view_bias,
            )
            if norm_mask is not None:
                score_chunk = score_chunk + norm_mask[:, :, s_start:s_end]

            p_chunk = p[:, :, s_start:s_end]
            p_chunk_acc = p_chunk.to(accum_dtype)
            row_sum_p = row_sum_p + p_chunk_acc.sum(dim=-1)
            p_log_p = p_log_p + (p_chunk_acc * torch.log(p_chunk_acc + float(eps))).sum(dim=-1)
            expected_score = expected_score + (p_chunk * score_chunk).sum(dim=-1, dtype=accum_dtype)
            log_denom = torch.logaddexp(log_denom, torch.logsumexp(score_chunk, dim=-1).to(accum_dtype))

        return (p_log_p - expected_score + row_sum_p * log_denom).mean()

    @staticmethod
    def _compute_view_bias_selected(
        view_bias_data: dict,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        q_view_ids = view_bias_data["q_view_ids"]
        s_view_ids = view_bias_data["s_view_ids"]
        view_bias = view_bias_data["view_bias"]
        bsz = int(indices.shape[0])
        batch_idx = torch.arange(bsz, device=indices.device, dtype=torch.long).view(bsz, 1, 1)
        selected_source_views = torch.gather(
            s_view_ids,
            dim=1,
            index=indices.reshape(bsz, -1),
        ).view_as(indices)
        return view_bias[batch_idx, q_view_ids.unsqueeze(-1), selected_source_views]

    def _compute_selected_score_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        indices: torch.Tensor,
        score_head_chunk_size: int,
        scale: torch.Tensor,
        view_bias_data: Optional[dict] = None,
    ) -> torch.Tensor:
        bsz, tgt_len, n_heads, _ = q.shape
        support = int(indices.shape[-1])
        batch_idx = torch.arange(bsz, device=indices.device, dtype=torch.long).view(bsz, 1, 1)
        k_sel = k[batch_idx, indices]  # [B, T, K, H, D]
        w_sel = w[batch_idx, indices]  # [B, T, K, H]
        score_chunk = None
        for h_start in range(0, n_heads, score_head_chunk_size):
            h_end = min(h_start + score_head_chunk_size, n_heads)
            q_h = q[:, :, h_start:h_end]
            k_h = k_sel[:, :, :, h_start:h_end]
            w_h = w_sel[:, :, :, h_start:h_end].permute(0, 1, 3, 2)
            head_scores = torch.einsum("bthd,btkhd->bthk", q_h, k_h) * scale
            head_scores = torch.relu(head_scores)
            head_scores = head_scores * w_h
            head_scores = head_scores.sum(dim=2, dtype=q.dtype)
            score_chunk = head_scores if score_chunk is None else score_chunk + head_scores
        if score_chunk is None:
            score_chunk = q.new_zeros((bsz, tgt_len, support))
        if view_bias_data is not None:
            score_chunk = score_chunk + self._compute_view_bias_selected(view_bias_data, indices)
        return score_chunk

    def compute_topk_support_loss(
        self,
        p: torch.Tensor,
        support_topk: int,
        eps: float = 1e-6,
        x: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        projected: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        view_bias_data: Optional[dict] = None,
        support_chunk_size: int = 128,
        query_chunk_size: int = 256,
        query_sample_size: int = 0,
    ) -> torch.Tensor:
        """Maximize selector probability mass on teacher top-k support.

        The teacher support is defined from dense attention ``p`` by taking its
        top ``support_topk`` source tokens per query. The loss is
        ``logsumexp(score_all) - logsumexp(score_teacher_topk)``, so it ignores
        teacher probabilities inside the support set and only trains the
        selector as a support retriever.
        """
        if projected is None:
            if x is None:
                raise ValueError("Either x or projected must be provided for compute_topk_support_loss.")
            q, k, w = self.project(x, pos=pos)
        else:
            q, k, w = projected

        if p.dim() != 3:
            raise ValueError(f"Expected p shape [B, T, S], got {tuple(p.shape)}")
        bsz, tgt_len, src_len = p.shape
        if q.shape[0] != bsz or q.shape[1] != tgt_len:
            raise ValueError(f"q shape {tuple(q.shape)} mismatches p shape {tuple(p.shape)}")
        if k.shape[0] != bsz or k.shape[1] != src_len:
            raise ValueError(f"k shape {tuple(k.shape)} mismatches p shape {tuple(p.shape)}")
        if w.shape[0] != bsz or w.shape[1] != src_len or w.shape[2] != q.shape[2]:
            raise ValueError(f"w shape {tuple(w.shape)} mismatches q {tuple(q.shape)} and p {tuple(p.shape)}")

        norm_mask = self._normalize_mask(mask, bsz, tgt_len, src_len)
        if norm_mask is not None and norm_mask.dtype != q.dtype:
            norm_mask = norm_mask.to(q.dtype)

        query_sample_size = int(query_sample_size or 0)
        sampled_view_bias_data = view_bias_data
        if 0 < query_sample_size < tgt_len:
            with torch.no_grad():
                query_indices = torch.randperm(tgt_len, device=p.device)[:query_sample_size]
                query_indices = query_indices.sort()[0]
            q = q.index_select(1, query_indices)
            p = p.index_select(1, query_indices)
            if norm_mask is not None:
                norm_mask = norm_mask.index_select(1, query_indices)
            if view_bias_data is not None and view_bias_data.get("q_view_ids", None) is not None:
                sampled_view_bias_data = dict(view_bias_data)
                q_view_ids = sampled_view_bias_data["q_view_ids"]
                sampled_view_bias_data["q_view_ids"] = q_view_ids.index_select(
                    0 if q_view_ids.dim() == 1 else 1,
                    query_indices.to(q_view_ids.device),
                )
            tgt_len = int(query_sample_size)

        support_topk = max(1, min(int(support_topk), int(src_len)))
        support_chunk_size = self._resolve_chunk_size(int(support_chunk_size or 0), support_topk)
        query_chunk_size = self._resolve_chunk_size(int(query_chunk_size or 0), tgt_len)
        with torch.no_grad():
            _, target_indices = torch.topk(p, k=support_topk, dim=-1, largest=True, sorted=False)
            target_indices = target_indices.to(torch.long)

        n_heads = q.shape[2]
        score_head_chunk_size = self._resolve_chunk_size(self.score_head_chunk_size, n_heads)
        score_key_chunk_size = self._resolve_chunk_size(self.score_key_chunk_size, src_len)
        scale = self._scale_like(q)
        w_by_head = w.permute(0, 2, 1)
        accum_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        norm_view_bias = self._normalize_view_bias_data(sampled_view_bias_data, bsz, tgt_len, src_len, q.device, q.dtype)

        use_topk_support_autograd = (
            (self.streaming_kl_autograd or self._env_flag("VGGT_TOPK_SUPPORT_AUTOGRAD"))
            and q.requires_grad
            and norm_view_bias is None
        )
        if use_topk_support_autograd:
            mask_tensor = norm_mask if norm_mask is not None else None
            return topk_support_autograd_loss(
                q=q,
                k=k,
                w=w,
                p=p,
                mask=mask_tensor,
                support_topk=support_topk,
                scale=float(self.scale),
                score_head_chunk_size=score_head_chunk_size,
                score_key_chunk_size=score_key_chunk_size,
                support_chunk_size=support_chunk_size,
                query_chunk_size=query_chunk_size,
            )

        log_denom = torch.full((bsz, tgt_len), float("-inf"), device=p.device, dtype=accum_dtype)
        for s_start in range(0, src_len, score_key_chunk_size):
            s_end = min(s_start + score_key_chunk_size, src_len)
            score_chunk = self._compute_score_chunk(
                q=q,
                k=k,
                w_by_head=w_by_head,
                s_start=s_start,
                s_end=s_end,
                score_head_chunk_size=score_head_chunk_size,
                scale=scale,
                view_bias_data=norm_view_bias,
            )
            if norm_mask is not None:
                score_chunk = score_chunk + norm_mask[:, :, s_start:s_end]
            log_denom = torch.logaddexp(log_denom, torch.logsumexp(score_chunk, dim=-1).to(accum_dtype))

        support_log_denom = torch.full((bsz, tgt_len), float("-inf"), device=p.device, dtype=accum_dtype)
        for q_start in range(0, tgt_len, query_chunk_size):
            q_end = min(q_start + query_chunk_size, tgt_len)
            q_chunk = q[:, q_start:q_end]
            q_support_log = torch.full((bsz, q_end - q_start), float("-inf"), device=p.device, dtype=accum_dtype)
            chunk_view_bias = None
            if norm_view_bias is not None:
                chunk_view_bias = dict(norm_view_bias)
                chunk_view_bias["q_view_ids"] = norm_view_bias["q_view_ids"][:, q_start:q_end]
            for k_start in range(0, support_topk, support_chunk_size):
                k_end = min(k_start + support_chunk_size, support_topk)
                indices = target_indices[:, q_start:q_end, k_start:k_end]
                selected_scores = self._compute_selected_score_chunk(
                    q=q_chunk,
                    k=k,
                    w=w,
                    indices=indices,
                    score_head_chunk_size=score_head_chunk_size,
                    scale=scale,
                    view_bias_data=chunk_view_bias,
                )
                if norm_mask is not None:
                    selected_scores = selected_scores + torch.gather(
                        norm_mask[:, q_start:q_end],
                        dim=-1,
                        index=indices,
                    )
                q_support_log = torch.logaddexp(
                    q_support_log,
                    torch.logsumexp(selected_scores, dim=-1).to(accum_dtype),
                )
            support_log_denom[:, q_start:q_end] = q_support_log

        return (log_denom - support_log_denom).mean()

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
        return_scores: bool = True,
        projected: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        view_bias_data: Optional[dict] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, N, C]
        bsz, seq_len, _ = x.shape
        if projected is None:
            q, k, w = self.project(x, pos=pos)
        else:
            q, k, w = projected
            if q.dim() != 4 or k.dim() != 4 or w.dim() != 3:
                raise ValueError("projected q/k/w must be [B,T,H,D], [B,T,H,D], [B,T,H]")
            if q.shape[0] != bsz or q.shape[1] != seq_len:
                raise ValueError(
                    f"projected q shape {tuple(q.shape)} mismatches input {(bsz, seq_len)}"
                )
            if k.shape[:2] != q.shape[:2] or k.shape[2:] != q.shape[2:]:
                raise ValueError(f"projected k shape {tuple(k.shape)} mismatches q {tuple(q.shape)}")
            if w.shape[0] != bsz or w.shape[1] != seq_len or w.shape[2] != q.shape[2]:
                raise ValueError(
                    f"projected w shape {tuple(w.shape)} mismatches q {tuple(q.shape)}"
                )
            if self.score_dtype is not None and self.score_dtype != q.dtype:
                q = q.to(self.score_dtype)
                k = k.to(self.score_dtype)
                w = w.to(self.score_dtype)

        norm_mask = self._normalize_mask(mask, bsz, seq_len, seq_len)
        if norm_mask is not None and norm_mask.dtype != q.dtype:
            norm_mask = norm_mask.to(q.dtype)

        scores = None
        if topk is None:
            scores = self._compute_scores(q, k, w, norm_mask, view_bias_data=view_bias_data)
            return scores
        return self.select_topk_projected(
            q,
            k,
            w,
            mask=norm_mask,
            topk=topk,
            return_scores=return_scores,
            view_bias_data=view_bias_data,
        )
