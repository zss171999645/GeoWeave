import json
import os
import gc
import math
import torch
from collections import Counter
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from typing import Optional, Tuple, Union

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import log
from easyvolcap.utils.dist_utils import get_rank, is_main_process
from easyvolcap.official_vggt.utils.epipolar_selector import (
    build_depth_reprojection_support_mask,
    build_epipolar_band_mask,
    build_downsampled_patch_indices,
    build_patch_token_centers,
    compute_query_to_source_fundamental_matrices,
    compute_epipolar_band_logsumexp_terms,
    build_image_token_indices,
)
try:
    from easyvolcap.utils.custom_indexer.indexer_kl_fused import sparse_indexer_kl_loss
except Exception:
    sparse_indexer_kl_loss = None
from .indexer import LightningIndexer
try:
    from easyvolcap.utils.custom_flash_attn.sparse_index_flash_attn import (
        sparse_index_flash_attn_func,
        sparse_index_flash_attn_bhtd_func,
        sparse_index_flash_attn_value_gate_bhtd_func,
        dense_index_flash_attn_func,
        sparse_index_flash_attn_inference_func,
        sparse_index_flash_attn_inference_bhtd_func,
    )
except Exception:
    sparse_index_flash_attn_func = None
    sparse_index_flash_attn_bhtd_func = None
    sparse_index_flash_attn_value_gate_bhtd_func = None
    dense_index_flash_attn_func = None
    sparse_index_flash_attn_inference_func = None
    sparse_index_flash_attn_inference_bhtd_func = None


class DSAAttention(nn.Module):
    """Attention with DeepSeek-style lightning indexer and top-k token selection."""
    _mem_snapshot_peak_alloc = 0
    _mem_snapshot_peak_reserved = 0
    _mem_snapshot_count = 0
    _mem_snapshot_cache_cleared = False
    _sparse_loss_warned = False
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        indexer_cfg: Optional[dotdict] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.kv_n_heads = self.num_heads

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.indexer_cfg = dotdict(
            enabled=False,
            n_heads=4,
            head_dim=64,
            topk=256,
            use_topk_kernel=True,
            use_sparse_flash_attn=True,
            use_dense_flash_attn_warmup_kernel=False,
            topk_block=256,
            topk_merge_blocks=0,
            loss_weight=1.0,
            eps=1e-6,
            detach_input=True,
            head_chunk_size=0,
            score_head_chunk_size=0,
            score_key_chunk_size=0,
            streaming_kl_loss=False,
            streaming_kl_autograd=False,
            warmup_indexer_loss_mode="kl",
            warmup_topk_coverage_k=1024,
            warmup_topk_coverage_chunk_size=128,
            warmup_topk_coverage_query_chunk_size=256,
            warmup_topk_coverage_query_sample_size=0,
            score_dtype="",
            sparse_use_mask=False,
            sparse_flash_sort_kv=False,
            sparse_flash_disable_attn_sum=False,
            force_keep_special_tokens=False,
            force_keep_special_in_topk_budget=False,
            force_keep_register_tokens=False,
            force_keep_self_view_tokens=False,
            objective_value_gate_enabled=False,
            objective_value_gate_scale=0.0,
            objective_value_gate_tau=1.0,
            objective_value_gate_eps=1e-6,
            objective_value_gate_query_chunk_size=64,
            objective_value_gate_value_chunk_size=0,
            shared_kv_n_heads=0,
            source_downsample_enabled=False,
            source_downsample_factor=2,
            source_downsample_query_chunk=4096,
            source_downsample_coarse_topk=0,
            source_downsample_coarse_ratio=1.1,
            source_downsample_pad_window=64,
            source_downsample_representative_points=False,
            source_downsample_strategy="legacy",
            source_downsample_recall_growth=1.5,
            source_downsample_recall_max_steps=3,
            source_downsample_rerank_query_chunk=256,
            soft_view_bias_enabled=False,
            soft_view_bias_training_kernel_enabled=False,
            soft_view_bias_descriptor="camera",
            soft_view_bias_hidden_dim=64,
            soft_view_bias_init_scale=0.0,
            epipolar_selector_band_enabled=False,
            epipolar_selector_band_weight=0.1,
            epipolar_selector_band_px=12.0,
            epipolar_selector_kl_enabled=False,
            epipolar_selector_kl_weight=0.1,
            epipolar_selector_sigma_px=12.0,
            epipolar_selector_query_stride=8,
            epipolar_selector_query_chunk_size=256,
            epipolar_selector_source_view_chunk_size=12,
            epipolar_selector_loss_downsample_factor=1,
            epipolar_selector_loss_mode="full",
            geometry_support_enabled=False,
            geometry_support_type="auto",
            geometry_support_weight=0.1,
            geometry_support_depth_radius_px=0.0,
            geometry_support_min_depth=1e-4,
            geometry_support_anchor_patch_radius=0,
            geometry_support_anchor_samples_per_view=8,
        )
        if indexer_cfg is not None:
            self.indexer_cfg.update(indexer_cfg)
        self.indexer_cfg.pop("debug_log", None)
        self.indexer_cfg.pop("mem_log", None)
        self.kv_n_heads = self._resolve_kv_n_heads(self.indexer_cfg.get("shared_kv_n_heads", 0))
        self.qkv = nn.Linear(dim, self._attention_q_dim + 2 * self._attention_kv_dim, bias=qkv_bias)

        self.indexer = LightningIndexer(
            dim=dim,
            n_heads=self.indexer_cfg.n_heads,
            head_dim=self.indexer_cfg.head_dim,
            rope=rope,
            qk_norm=qk_norm,
            score_dtype=self.indexer_cfg.get("score_dtype", None),
            score_head_chunk_size=self.indexer_cfg.get("score_head_chunk_size", 0),
            score_key_chunk_size=self.indexer_cfg.get("score_key_chunk_size", 0),
            streaming_kl_autograd=self.indexer_cfg.get("streaming_kl_autograd", False),
            use_topk_kernel=self.indexer_cfg.get("use_topk_kernel", False),
            topk_block=self.indexer_cfg.get("topk_block", 256),
            topk_merge_blocks=self.indexer_cfg.get("topk_merge_blocks", 0),
            soft_view_bias_training_kernel_enabled=self.indexer_cfg.get("soft_view_bias_training_kernel_enabled", False),
        ) if self.indexer_cfg.enabled else None

        self.indexer_state = dotdict(enabled=False)
        self.last_topk_indices = None
        self.last_topk_scores = None
        self.last_epipolar_selector_stats = None
        self._fused_proj_params_cache = None
        self._stream_proj_params_cache = {}
        self._async_indexer_streams = {}
        self._fullchain_graph_cache = {}
        self._fullchain_graph_failures = set()
        self._source_downsample_layout_cache = {}
        self.soft_view_bias_q_proj = None
        self.soft_view_bias_k_proj = None
        if bool(self.indexer_cfg.get("soft_view_bias_enabled", False)):
            bias_dim = max(int(self.indexer_cfg.get("soft_view_bias_hidden_dim", 64) or 64), 1)
            self.soft_view_bias_q_proj = nn.Linear(dim, bias_dim, bias=True)
            self.soft_view_bias_k_proj = nn.Linear(dim, bias_dim, bias=True)
            descriptor_mode = str(self.indexer_cfg.get("soft_view_bias_descriptor", "camera") or "camera").strip().lower()
            if descriptor_mode == "camera":
                self.register_buffer("soft_view_bias_scale", torch.tensor(0.1), persistent=False)
            else:
                self.soft_view_bias_scale = nn.Parameter(
                    torch.tensor(float(self.indexer_cfg.get("soft_view_bias_init_scale", 0.0) or 0.0))
                )

    @property
    def _attention_q_dim(self) -> int:
        return self.dim

    @property
    def _attention_kv_dim(self) -> int:
        return self.kv_n_heads * self.head_dim

    def _resolve_kv_n_heads(self, value: object) -> int:
        try:
            kv_n_heads = int(value)
        except (TypeError, ValueError):
            kv_n_heads = 0
        if kv_n_heads <= 0:
            return self.num_heads
        if self.num_heads % kv_n_heads != 0:
            raise ValueError(f"shared_kv_n_heads={kv_n_heads} must divide num_heads={self.num_heads}")
        return kv_n_heads

    def _expand_attention_kv_heads(self, x: Tensor) -> Tensor:
        if x.shape[1] == self.num_heads:
            return x
        repeat = self.num_heads // x.shape[1]
        return x.repeat_interleave(repeat, dim=1)

    def _project_attention_qkv_from_flat(self, qkv: Tensor, bsz: int, tgt_len: int) -> Tuple[Tensor, Tensor, Tensor]:
        if self.kv_n_heads == self.num_heads:
            qkv = qkv.reshape(bsz, tgt_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            return qkv.unbind(0)

        q_flat, k_flat, v_flat = torch.split(
            qkv,
            [self._attention_q_dim, self._attention_kv_dim, self._attention_kv_dim],
            dim=-1,
        )
        q = q_flat.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k_flat.view(bsz, tgt_len, self.kv_n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_flat.view(bsz, tgt_len, self.kv_n_heads, self.head_dim).permute(0, 2, 1, 3)
        return q, k, v

    def _project_attention_qkv(self, x: Tensor, bsz: int, tgt_len: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self._project_attention_qkv_from_flat(self.qkv(x), bsz, tgt_len)

    def _adapt_dense_qkv_weight_for_shared_kv(self, weight: Tensor) -> Tensor:
        if self.kv_n_heads == self.num_heads:
            return weight
        if weight.dim() != 2 or weight.shape[0] != 3 * self.dim or weight.shape[1] != self.dim:
            return weight

        group = self.num_heads // self.kv_n_heads
        q_weight, k_weight, v_weight = weight.split(self.dim, dim=0)
        k_weight = k_weight.view(self.num_heads, self.head_dim, self.dim)
        v_weight = v_weight.view(self.num_heads, self.head_dim, self.dim)
        k_weight = k_weight.view(self.kv_n_heads, group, self.head_dim, self.dim).mean(dim=1)
        v_weight = v_weight.view(self.kv_n_heads, group, self.head_dim, self.dim).mean(dim=1)
        return torch.cat(
            [
                q_weight,
                k_weight.reshape(self._attention_kv_dim, self.dim),
                v_weight.reshape(self._attention_kv_dim, self.dim),
            ],
            dim=0,
        )

    def _adapt_dense_qkv_bias_for_shared_kv(self, bias: Tensor) -> Tensor:
        if self.kv_n_heads == self.num_heads:
            return bias
        if bias.dim() != 1 or bias.shape[0] != 3 * self.dim:
            return bias

        group = self.num_heads // self.kv_n_heads
        q_bias, k_bias, v_bias = bias.split(self.dim, dim=0)
        k_bias = k_bias.view(self.num_heads, self.head_dim).view(self.kv_n_heads, group, self.head_dim).mean(dim=1)
        v_bias = v_bias.view(self.num_heads, self.head_dim).view(self.kv_n_heads, group, self.head_dim).mean(dim=1)
        return torch.cat(
            [
                q_bias,
                k_bias.reshape(self._attention_kv_dim),
                v_bias.reshape(self._attention_kv_dim),
            ],
            dim=0,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.kv_n_heads != self.num_heads:
            weight_key = f"{prefix}qkv.weight"
            bias_key = f"{prefix}qkv.bias"
            if weight_key in state_dict:
                state_dict[weight_key] = self._adapt_dense_qkv_weight_for_shared_kv(state_dict[weight_key])
            if bias_key in state_dict:
                state_dict[bias_key] = self._adapt_dense_qkv_bias_for_shared_kv(state_dict[bias_key])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _mib(bytes_value: int) -> float:
        return bytes_value / 2**20

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        if num_bytes >= 2**30:
            return f"{num_bytes / 2**30:.2f} GiB"
        return f"{num_bytes / 2**20:.1f} MiB"

    @staticmethod
    def _env_flag(name: str) -> bool:
        value = os.getenv(name, "")
        return value.lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.getenv(name, "")
        try:
            return int(value) if value else int(default)
        except ValueError:
            return int(default)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name, "")
        try:
            return float(value) if value else float(default)
        except ValueError:
            return float(default)

    @staticmethod
    def _env_str(name: str, default: str = "") -> str:
        value = os.getenv(name, "")
        return value if value else default

    def _should_use_sparse_bhtd_autograd(self, state: dotdict, use_inference_sparse: bool) -> bool:
        if use_inference_sparse or sparse_index_flash_attn_bhtd_func is None:
            return False
        if self._env_flag("VGGT_SPARSE_FLASH_BHTD_AUTOGRAD"):
            return True
        policy = self._env_str("VGGT_SPARSE_FLASH_BHTD_AUTOGRAD_POLICY", "off").strip().lower()
        if policy in ("on", "force"):
            return True
        if policy == "auto":
            # Training/grad path can skip repeated [B,H,T,D] <-> [B,T,H,D] transposes.
            return self.training or torch.is_grad_enabled() or bool(state.get("compute_loss", False))
        return False

    def _mem_snapshot_enabled(self, state: dotdict) -> bool:
        return self._env_flag("VGGT_MEM_SNAPSHOT") or bool(state.get("mem_snapshot", False))

    @staticmethod
    def _mem_snapshot_topk() -> int:
        try:
            return int(os.getenv("VGGT_MEM_SNAPSHOT_TOPK", "30"))
        except ValueError:
            return 30

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
    def _format_snapshot_location(block: dict) -> str:
        frames = block.get("frames") or block.get("stack") or block.get("stack_trace") or block.get("traceback") or []
        if isinstance(frames, dict):
            frames = frames.get("frames", []) or []
        if isinstance(frames, str):
            return frames
        if frames:
            frame = frames[0]
            filename = frame.get("filename", "unknown")
            line = frame.get("line", 0)
            return f"{filename}:{line}"
        return "unknown"

    def _summarize_snapshot(self, snapshot: object, topk: int) -> dotdict:
        totals_by_state = Counter()
        totals_by_segment_type = Counter()
        active_bytes = 0
        active_requested_bytes = 0
        by_location = Counter()
        unknown_sizes = Counter()

        if isinstance(snapshot, dict):
            segments = snapshot.get("segments") or snapshot.get("data") or []
        elif isinstance(snapshot, list):
            segments = snapshot
        else:
            segments = []

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            seg_type = segment.get("segment_type", segment.get("type", "unknown"))
            totals_by_segment_type[seg_type] += int(segment.get("total_size", 0) or 0)
            for block in segment.get("blocks", []) or []:
                if not isinstance(block, dict):
                    continue
                state = block.get("state", "unknown")
                size = int(block.get("size", block.get("allocated_size", 0)) or 0)
                requested = int(block.get("requested_size", block.get("requested", size)) or size)
                totals_by_state[state] += size
                if str(state).startswith("active"):
                    active_bytes += size
                    active_requested_bytes += requested
                if state == "active_allocated":
                    location = self._format_snapshot_location(block)
                    by_location[location] += size
                    if location == "unknown":
                        unknown_sizes[size] += 1

        top_allocations = []
        if topk > 0:
            for location, size in by_location.most_common(topk):
                top_allocations.append(dotdict(location=location, bytes=size))

        return dotdict(
            totals_by_state=dotdict({k: int(v) for k, v in totals_by_state.items()}),
            totals_by_segment_type=dotdict({k: int(v) for k, v in totals_by_segment_type.items()}),
            active_bytes=int(active_bytes),
            active_requested_bytes=int(active_requested_bytes),
            top_allocations=top_allocations,
            unknown_sizes=unknown_sizes,
        )

    def _maybe_mem_snapshot(self, state: dotdict, tag: str) -> None:
        if not self._mem_snapshot_enabled(state):
            return
        if not torch.cuda.is_available() or not self._mem_trace_rank_ok():
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
            filename = f"snapshot_{tag}_rank{get_rank()}_{self.__class__._mem_snapshot_count:03d}.json"
            path = os.path.join(out_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=True)

        self.__class__._mem_snapshot_count += 1

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

    def _resolve_score_dtype(self, state: dotdict, fallback: torch.dtype) -> torch.dtype:
        env_dtype = self._parse_dtype(os.getenv("VGGT_SCORE_DTYPE", None))
        if env_dtype is not None:
            return env_dtype
        dtype = self._parse_dtype(state.get("score_dtype", None))
        if dtype is None:
            dtype = self._parse_dtype(self.indexer_cfg.get("score_dtype", None))
        return dtype or fallback

    def _scale_like(self, ref: Tensor) -> Tensor:
        return ref.new_tensor(self.scale)

    def _should_use_fused_proj(self, state: dotdict, x: Tensor) -> bool:
        if not self._env_flag("VGGT_DSA_FUSE_QKV_INDEXER_PROJ"):
            return False
        if self.indexer is None or not bool(state.get("enabled", False)) or not bool(state.get("sparse", False)):
            return False
        in_train_or_grad = self.training or torch.is_grad_enabled()
        if in_train_or_grad and (not self._env_flag("VGGT_DSA_FUSE_QKV_INDEXER_PROJ_TRAIN")):
            return False
        # Fused projection requires qkv and indexer to share the same input tensor.
        if bool(state.get("detach_input", False)):
            return False
        return x.is_cuda

    def _get_fused_proj_params(self, *, cache: bool = True):
        idx = self.indexer
        if idx is None:
            return None
        linears = (self.qkv, idx.q_proj, idx.k_proj, idx.w_proj)
        sig = None
        if cache:
            sig = tuple(
                (
                    int(lin.weight.data_ptr()),
                    int(lin.bias.data_ptr()) if lin.bias is not None else 0,
                    str(lin.weight.dtype),
                    str(lin.weight.device),
                )
                for lin in linears
            )
            cached = self._fused_proj_params_cache
            if cached is not None and cached.get("sig", None) == sig:
                return cached["weight"], cached["bias"], cached["splits"]

        weight = torch.cat([lin.weight for lin in linears], dim=0)
        bias_parts = []
        for lin in linears:
            if lin.bias is None:
                bias_parts.append(lin.weight.new_zeros((lin.out_features,)))
            else:
                bias_parts.append(lin.bias)
        bias = torch.cat(bias_parts, dim=0)
        splits = [lin.out_features for lin in linears]
        if cache:
            self._fused_proj_params_cache = {
                "sig": sig,
                "weight": weight,
                "bias": bias,
                "splits": splits,
            }
        return weight, bias, splits

    def _project_qkv_and_indexer_fused(
        self,
        x: Tensor,
        pos: Optional[Tensor],
    ):
        cache_params = not (self.training or torch.is_grad_enabled())
        params = self._get_fused_proj_params(cache=cache_params)
        if params is None or self.indexer is None:
            return None
        weight, bias, splits = params
        fused_out = F.linear(x, weight, bias)
        qkv_dim, idx_q_dim, idx_k_dim, idx_w_dim = splits
        qkv_flat, idx_q_flat, idx_k_flat, idx_w_flat = torch.split(
            fused_out,
            [qkv_dim, idx_q_dim, idx_k_dim, idx_w_dim],
            dim=-1,
        )

        bsz, tgt_len, _ = x.shape

        idx = self.indexer
        idx_q = idx_q_flat.view(bsz, tgt_len, idx.n_heads, idx.head_dim)
        idx_k = idx_k_flat.view(bsz, tgt_len, idx.n_heads, idx.head_dim)
        idx_w = idx_w_flat.view(bsz, tgt_len, idx.n_heads)
        idx_k = idx.k_norm(idx_k)

        if idx.rope is not None and pos is not None:
            idx_q = idx_q.permute(0, 2, 1, 3)
            idx_k = idx_k.permute(0, 2, 1, 3)
            idx_q = idx.rope(idx_q, pos)
            idx_k = idx.rope(idx_k, pos)
            idx_q = idx_q.permute(0, 2, 1, 3)
            idx_k = idx_k.permute(0, 2, 1, 3)

        if idx.score_dtype is not None and idx.score_dtype != idx_q.dtype:
            idx_q = idx_q.to(idx.score_dtype)
            idx_k = idx_k.to(idx.score_dtype)
            idx_w = idx_w.to(idx.score_dtype)

        return qkv_flat, (idx_q, idx_k, idx_w)

    def _get_sparse_stream_proj_params(self, mode: str, *, cache: bool = True):
        idx = self.indexer
        if idx is None:
            return None
        if mode == "q_idxq":
            linears = (self.qkv, idx.q_proj)
        elif mode == "kv_idxkw":
            linears = (self.qkv, idx.k_proj, idx.w_proj)
        else:
            raise ValueError(f"Unsupported sparse stream projection mode: {mode}")

        sig = None
        cache_key = None
        if cache:
            sig = tuple(
                (
                    int(lin.weight.data_ptr()),
                    int(lin.bias.data_ptr()) if lin.bias is not None else 0,
                    str(lin.weight.dtype),
                    str(lin.weight.device),
                )
                for lin in linears
            )
            cache_key = (mode, sig)
            cached = self._stream_proj_params_cache.get(cache_key, None)
            if cached is not None:
                return cached["weight"], cached["bias"], cached["splits"]

        q_dim = int(self._attention_q_dim)
        kv_dim = int(self._attention_kv_dim)
        qkv_weight = self.qkv.weight
        qkv_bias = self.qkv.bias
        if mode == "q_idxq":
            weight_parts = [
                qkv_weight[:q_dim],
                idx.q_proj.weight,
            ]
            bias_parts = [
                qkv_bias[:q_dim] if qkv_bias is not None else qkv_weight.new_zeros((q_dim,)),
                idx.q_proj.bias if idx.q_proj.bias is not None else idx.q_proj.weight.new_zeros((idx.q_proj.out_features,)),
            ]
            splits = [q_dim, idx.q_proj.out_features]
        else:
            kv_total = 2 * kv_dim
            weight_parts = [
                qkv_weight[q_dim:q_dim + kv_total],
                idx.k_proj.weight,
                idx.w_proj.weight,
            ]
            bias_parts = [
                qkv_bias[q_dim:q_dim + kv_total] if qkv_bias is not None else qkv_weight.new_zeros((kv_total,)),
                idx.k_proj.bias if idx.k_proj.bias is not None else idx.k_proj.weight.new_zeros((idx.k_proj.out_features,)),
                idx.w_proj.bias if idx.w_proj.bias is not None else idx.w_proj.weight.new_zeros((idx.w_proj.out_features,)),
            ]
            splits = [kv_total, idx.k_proj.out_features, idx.w_proj.out_features]

        weight = torch.cat(weight_parts, dim=0)
        bias = torch.cat(bias_parts, dim=0)
        if cache and cache_key is not None:
            self._stream_proj_params_cache[cache_key] = {
                "weight": weight,
                "bias": bias,
                "splits": splits,
            }
        return weight, bias, splits

    def _project_sparse_stream_q_and_indexer_query_fused(
        self,
        x: Tensor,
        pos: Optional[Tensor],
        *,
        score_dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        params = self._get_sparse_stream_proj_params("q_idxq", cache=not (self.training or torch.is_grad_enabled()))
        if params is None or self.indexer is None:
            raise RuntimeError("Sparse stream q/indexer fused projection requires an active indexer")
        weight, bias, splits = params
        fused = F.linear(x, weight, bias)
        q_dim, idx_q_dim = splits
        q_flat, idx_q_flat = torch.split(fused, [q_dim, idx_q_dim], dim=-1)

        bsz, tgt_len, _ = x.shape
        q = q_flat.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
        if q.dtype != score_dtype:
            q = q.to(score_dtype)

        idx = self.indexer
        idx_q = idx_q_flat.view(bsz, tgt_len, idx.n_heads, idx.head_dim)
        if idx.rope is not None and pos is not None:
            idx_q = idx_q.permute(0, 2, 1, 3)
            idx_q = idx.rope(idx_q, pos)
            idx_q = idx_q.permute(0, 2, 1, 3)
        if idx.score_dtype is not None and idx.score_dtype != idx_q.dtype:
            idx_q = idx_q.to(idx.score_dtype)
        return q, idx_q

    def _project_sparse_stream_kv_and_indexer_kw_fused(
        self,
        x: Tensor,
        pos: Optional[Tensor],
        *,
        score_dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        params = self._get_sparse_stream_proj_params("kv_idxkw", cache=not (self.training or torch.is_grad_enabled()))
        if params is None or self.indexer is None:
            raise RuntimeError("Sparse stream kv/indexer fused projection requires an active indexer")
        weight, bias, splits = params
        fused = F.linear(x, weight, bias)
        kv_dim, idx_k_dim, idx_w_dim = splits
        kv_flat, idx_k_flat, idx_w_flat = torch.split(fused, [kv_dim, idx_k_dim, idx_w_dim], dim=-1)

        bsz, tgt_len, _ = x.shape
        k_flat, v_flat = torch.split(kv_flat, [self._attention_kv_dim, self._attention_kv_dim], dim=-1)
        k = k_flat.view(bsz, tgt_len, self.kv_n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_flat.view(bsz, tgt_len, self.kv_n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_norm(k)
        if self.rope is not None and pos is not None:
            k = self.rope(k, pos)
        if k.dtype != score_dtype:
            k = k.to(score_dtype)
            v = v.to(score_dtype)

        idx = self.indexer
        idx_k = idx_k_flat.view(bsz, tgt_len, idx.n_heads, idx.head_dim)
        idx_k = idx.k_norm(idx_k)
        idx_w = idx_w_flat.view(bsz, tgt_len, idx.n_heads)
        if idx.rope is not None and pos is not None:
            idx_k = idx_k.permute(0, 2, 1, 3)
            idx_k = idx.rope(idx_k, pos)
            idx_k = idx_k.permute(0, 2, 1, 3)
        if idx.score_dtype is not None and idx.score_dtype != idx_k.dtype:
            idx_k = idx_k.to(idx.score_dtype)
            idx_w = idx_w.to(idx.score_dtype)
        return k, v, idx_k, idx_w

    def _should_use_async_indexer(self, state: dotdict, x: Tensor) -> bool:
        if not self._env_flag("VGGT_DSA_ASYNC_INDEXER_PREFETCH"):
            return False
        if self.indexer is None or not bool(state.get("enabled", False)) or not bool(state.get("sparse", False)):
            return False
        if self.training or torch.is_grad_enabled():
            return False
        return x.is_cuda

    def _get_async_indexer_stream(self, device: torch.device) -> torch.cuda.Stream:
        key = str(device)
        stream = self._async_indexer_streams.get(key, None)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._async_indexer_streams[key] = stream
        return stream

    def _should_use_fullchain_fastpath(self, state: dotdict, x: Tensor, mask: Optional[Tensor]) -> bool:
        if not self._env_flag("VGGT_DSA_FULLCHAIN_FASTPATH"):
            return False
        if self.indexer is None or not bool(state.get("enabled", False)) or not bool(state.get("sparse", False)):
            return False
        if self._objective_value_gate_enabled(state):
            return False
        if self.training or torch.is_grad_enabled() or (not x.is_cuda):
            return False
        if bool(state.get("compute_loss", False)):
            return False
        if mask is not None:
            return False
        if bool(state.get("sparse_use_mask", self.indexer_cfg.get("sparse_use_mask", False))):
            return False
        if state.get("reuse_topk_indices", None) is not None:
            return False
        if bool(state.get("force_keep_special_tokens", False)):
            return False
        if bool(state.get("force_keep_register_tokens", False)):
            return False
        if bool(state.get("force_keep_self_view_tokens", False)):
            return False
        if bool(state.get("force_keep_special_in_topk_budget", False)):
            return False
        if sparse_index_flash_attn_inference_func is None:
            return False
        return True

    def _fullchain_graph_key(self, x: Tensor, pos: Optional[Tensor], state: dotdict) -> tuple:
        topk = min(int(state.get("topk", self.indexer_cfg.topk)), int(x.shape[1]))
        pos_sig = ("none",) if pos is None else (tuple(pos.shape), str(pos.dtype))
        return (
            str(x.device),
            str(x.dtype),
            int(x.shape[1]),
            int(topk),
            int(self.kv_n_heads),
            str(self._resolve_score_dtype(state, x.dtype)),
            bool(self._env_flag("VGGT_DSA_FUSE_QKV_INDEXER_PROJ")),
            bool(self._env_flag("VGGT_SPARSE_FLASH_NO_QKV_CONTIG")),
            bool(self._env_flag("VGGT_SPARSE_FLASH_NO_KV_CONTIG")),
            pos_sig,
        )

    def _should_use_sparse_stream_fastpath(self, state: dotdict, x: Tensor, mask: Optional[Tensor]) -> bool:
        if not self._env_flag("VGGT_DSA_SPARSE_STREAM_FASTPATH"):
            return False
        if self.indexer is None or not bool(state.get("enabled", False)) or not bool(state.get("sparse", False)):
            return False
        if self._objective_value_gate_enabled(state):
            return False
        if self.training or torch.is_grad_enabled() or (not x.is_cuda):
            return False
        if bool(state.get("compute_loss", False)):
            return False
        if mask is not None:
            return False
        if bool(state.get("sparse_use_mask", self.indexer_cfg.get("sparse_use_mask", False))):
            return False
        if state.get("reuse_topk_indices", None) is not None:
            return False
        if bool(state.get("force_keep_special_tokens", False)):
            return False
        if bool(state.get("force_keep_register_tokens", False)):
            return False
        if bool(state.get("force_keep_self_view_tokens", False)):
            return False
        if bool(state.get("force_keep_special_in_topk_budget", False)):
            return False
        if sparse_index_flash_attn_inference_func is None:
            return False
        return True

    def _resolve_sparse_stream_query_chunk(self, tgt_len: int, topk: int) -> int:
        chunk = self._env_int("VGGT_DSA_SPARSE_STREAM_QCHUNK", 0)
        if chunk <= 0:
            return 0
        chunk = max(1, min(int(chunk), int(tgt_len)))
        if chunk >= int(tgt_len):
            return 0
        if chunk < int(topk):
            chunk = int(topk)
        return min(chunk, int(tgt_len))

    def _resolve_sparse_stream_consume_query_chunk(self, query_chunk: int) -> int:
        consume_chunk = self._env_int("VGGT_DSA_SPARSE_STREAM_CONSUME_QCHUNK", 0)
        if consume_chunk <= 0:
            return max(1, int(query_chunk))
        return max(1, min(int(query_chunk), int(consume_chunk)))

    def _resolve_sparse_stream_qk_sym_x2_query_chunk(
        self,
        state: dotdict,
        *,
        tgt_len: int,
        base_query_chunk: int,
    ) -> int:
        qk_query_chunk = self._resolve_source_downsample_query_chunk(state, int(tgt_len))
        if qk_query_chunk <= 0:
            qk_query_chunk = int(base_query_chunk)
        qk_query_chunk = max(int(qk_query_chunk), 1)
        return min(int(tgt_len), int(qk_query_chunk))

    def _should_use_sparse_stream_chunk_q_contig_only(self, use_bhtd: bool) -> bool:
        if not use_bhtd:
            return False
        return self._env_flag("VGGT_DSA_SPARSE_STREAM_CHUNK_Q_CONTIG_ONLY")

    def _should_use_sparse_stream_sort_kv(self) -> bool:
        return self._env_flag("VGGT_DSA_SPARSE_STREAM_SORT_KV") or self._env_flag("VGGT_SPARSE_FLASH_SORT_KV")

    def _should_force_sparse_stream_k_contig(self) -> bool:
        return self._env_flag("VGGT_DSA_SPARSE_STREAM_FORCE_K_CONTIG")

    def _should_force_sparse_stream_v_contig(self) -> bool:
        return self._env_flag("VGGT_DSA_SPARSE_STREAM_FORCE_V_CONTIG")

    def _should_use_sparse_stream_pack_kv(self, use_bhtd: bool, no_qkv_contig: bool) -> bool:
        if not use_bhtd or (not no_qkv_contig):
            return False
        return self._env_flag("VGGT_DSA_SPARSE_STREAM_PACK_KV")

    def _should_use_sparse_stream_ima_guard(
        self,
        *,
        tgt_len: int,
        topk: int,
        use_bhtd: bool,
        no_qkv_contig: bool,
        force_v_contig: bool,
        stream_indexer_q_chunk: bool,
    ) -> bool:
        if os.getenv("VGGT_DSA_SPARSE_STREAM_IMA_GUARD", "1").strip().lower() in ("0", "false", "no", "off", ""):
            return False
        if (not use_bhtd) or (not no_qkv_contig) or (int(topk) < 2048):
            return False
        tgt_len = int(tgt_len)
        vcontig_min_tokens = max(
            int(topk),
            self._env_int("VGGT_DSA_SPARSE_STREAM_IMA_GUARD_VCONTIG_MIN_TOKENS", 700 * 1374),
        )
        qproj_min_tokens = max(
            int(topk),
            self._env_int("VGGT_DSA_SPARSE_STREAM_IMA_GUARD_QPROJ_MIN_TOKENS", 1000 * 1374),
        )
        if force_v_contig and tgt_len >= int(vcontig_min_tokens):
            return True
        if stream_indexer_q_chunk and tgt_len >= int(qproj_min_tokens):
            return True
        return False

    def _resolve_sparse_stream_ima_guard_qchunk(self, tgt_len: int, topk: int, query_chunk: int) -> int:
        safe_qchunk = max(
            int(topk),
            self._env_int("VGGT_DSA_SPARSE_STREAM_IMA_GUARD_QCHUNK", 8192),
        )
        safe_qchunk = min(int(tgt_len), int(safe_qchunk))
        if int(query_chunk) <= 0:
            return int(safe_qchunk)
        return min(int(query_chunk), int(safe_qchunk))

    def _should_use_sparse_stream_indexer_q_chunk_proj(self) -> bool:
        return self._env_flag("VGGT_DSA_SPARSE_STREAM_INDEXER_Q_CHUNK_PROJ")

    def _should_use_sparse_stream_attn_q_chunk_proj(self) -> bool:
        return self._env_flag("VGGT_DSA_SPARSE_STREAM_ATTN_Q_CHUNK_PROJ")

    def _pack_sparse_stream_kv_chunk(
        self,
        k_bhtd: Tensor,
        v_bhtd: Tensor,
        topk_indices: Tensor,
        *,
        seqlen_k: int,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        if topk_indices.numel() == 0:
            return None
        batch, q_chunk, topk = topk_indices.shape
        max_ratio = max(0.0, self._env_float("VGGT_DSA_SPARSE_STREAM_PACK_KV_MAX_RATIO", 1.0))
        min_unique = max(1, self._env_int("VGGT_DSA_SPARSE_STREAM_PACK_KV_MIN_UNIQUE", 1))
        local_uniques = []
        local_inverse = []
        max_local = 0
        for bidx in range(batch):
            flat = topk_indices[bidx].reshape(-1)
            uniq, inverse = torch.unique(flat, sorted=True, return_inverse=True)
            local_count = int(uniq.numel())
            if local_count < min_unique:
                return None
            if (max_ratio > 0.0) and (float(local_count) > float(seqlen_k) * max_ratio):
                return None
            local_uniques.append(uniq)
            local_inverse.append(inverse.reshape(q_chunk, topk))
            max_local = max(max_local, local_count)
        if max_local <= 0:
            return None

        packed_k = k_bhtd.new_zeros((batch, k_bhtd.shape[1], max_local, k_bhtd.shape[-1]))
        packed_v = v_bhtd.new_zeros((batch, v_bhtd.shape[1], max_local, v_bhtd.shape[-1]))
        local_pos = torch.full((batch, q_chunk, topk), -1, device=topk_indices.device, dtype=torch.int32)
        for bidx, uniq in enumerate(local_uniques):
            local_count = int(uniq.numel())
            packed_k[bidx, :, :local_count] = k_bhtd[bidx].index_select(1, uniq)
            packed_v[bidx, :, :local_count] = v_bhtd[bidx].index_select(1, uniq)
            local_pos[bidx] = local_inverse[bidx].to(dtype=torch.int32)
        return packed_k, packed_v, local_pos

    def _debug_sparse_stream(self, **info) -> None:
        if not self._env_flag("VGGT_DSA_SPARSE_STREAM_DEBUG"):
            return
        if getattr(self, "_sparse_stream_debug_printed", False):
            return
        self._sparse_stream_debug_printed = True
        extras = " ".join(f"{k}={v}" for k, v in info.items())
        print(f"[DSA_SPARSE_STREAM] {extras}", flush=True)

    def _debug_sparse_stream_mem(self, label: str, *, q_start: Optional[int] = None) -> None:
        if not self._env_flag("VGGT_DSA_SPARSE_STREAM_MEM_DEBUG"):
            return
        if q_start is not None and q_start != 0:
            return
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        print(
            "[DSA_SPARSE_STREAM_MEM]",
            f"label={label}",
            f"q_start={-1 if q_start is None else int(q_start)}",
            f"alloc_gib={alloc / (1024 ** 3):.3f}",
            f"reserved_gib={reserved / (1024 ** 3):.3f}",
            f"peak_gib={peak / (1024 ** 3):.3f}",
            flush=True,
        )

    def _run_sparse_streaming_direct(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        indexer_input: Tensor,
        projected_indexer: Optional[Tuple[Tensor, Tensor, Tensor]],
        pos: Optional[Tensor],
        state: dotdict,
        *,
        orig_dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if self.indexer is None:
            return None

        bsz, _, tgt_len, _ = q.shape
        topk = min(int(state.get("topk", self.indexer_cfg.topk)), int(tgt_len))
        query_chunk = self._resolve_sparse_stream_query_chunk(int(tgt_len), int(topk))
        if query_chunk <= 0:
            return None
        consume_query_chunk = self._resolve_sparse_stream_consume_query_chunk(int(query_chunk))

        prefer_qk_sym_x2_selector = (
            bool(state.get("source_downsample_enabled", False))
            and int(state.get("source_downsample_factor", 1) or 1) == 2
            and str(state.get("source_downsample_strategy", self.indexer_cfg.get("source_downsample_strategy", "legacy")) or "legacy").strip().lower() == "qk_sym_x2_broadcast"
        )
        stream_indexer_q_chunk = (projected_indexer is None) and self._should_use_sparse_stream_indexer_q_chunk_proj()
        if stream_indexer_q_chunk or prefer_qk_sym_x2_selector:
            idx_q = None
            idx_k, idx_w = self.indexer.project_key_weight(indexer_input, pos=pos)
            self._debug_sparse_stream_mem("project_kw_full")
        else:
            idx_q, idx_k, idx_w = projected_indexer if projected_indexer is not None else self.indexer.project(indexer_input, pos=pos)
            self._debug_sparse_stream_mem("project_qkw_full")

        force_kv_contig = not self._env_flag("VGGT_SPARSE_FLASH_NO_KV_CONTIG")
        no_qkv_contig = self._env_flag("VGGT_SPARSE_FLASH_NO_QKV_CONTIG")
        use_bhtd = self._env_flag("VGGT_SPARSE_FLASH_DIRECT_BHTD") and (
            sparse_index_flash_attn_inference_bhtd_func is not None
        )

        q_attn = q
        k_attn = k
        v_attn = v
        if q_attn.dtype not in (torch.float16, torch.bfloat16):
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q_attn = q_attn.to(target_dtype)
            k_attn = k_attn.to(target_dtype)
            v_attn = v_attn.to(target_dtype)
        if k_attn.shape[1] != q_attn.shape[1]:
            k_attn = self._expand_attention_kv_heads(k_attn)
            v_attn = self._expand_attention_kv_heads(v_attn)
        chunk_q_contig_only = self._should_use_sparse_stream_chunk_q_contig_only(use_bhtd)
        force_k_contig = self._should_force_sparse_stream_k_contig()
        force_v_contig = self._should_force_sparse_stream_v_contig()
        pack_local_kv = self._should_use_sparse_stream_pack_kv(use_bhtd, no_qkv_contig)
        ima_guard_pack_kv = self._should_use_sparse_stream_ima_guard(
            tgt_len=int(tgt_len),
            topk=int(topk),
            use_bhtd=bool(use_bhtd),
            no_qkv_contig=bool(no_qkv_contig),
            force_v_contig=bool(force_v_contig),
            stream_indexer_q_chunk=bool(stream_indexer_q_chunk),
        )
        if ima_guard_pack_kv:
            query_chunk = self._resolve_sparse_stream_ima_guard_qchunk(int(tgt_len), int(topk), int(query_chunk))
            force_k_contig = False
            force_v_contig = False
            pack_local_kv = True
        self._debug_sparse_stream(
            tgt_len=int(tgt_len),
            topk=int(topk),
            query_chunk=int(query_chunk),
            consume_query_chunk=int(consume_query_chunk),
            use_bhtd=int(use_bhtd),
            no_qkv_contig=int(no_qkv_contig),
            force_k_contig=int(force_k_contig),
            force_v_contig=int(force_v_contig),
            chunk_q_contig_only=int(chunk_q_contig_only),
            pack_local_kv=int(pack_local_kv),
            ima_guard_pack_kv=int(ima_guard_pack_kv),
        )
        if not no_qkv_contig:
            if not chunk_q_contig_only:
                q_attn = q_attn.contiguous()
            k_attn = k_attn.contiguous()
            v_attn = v_attn.contiguous()
        else:
            if force_k_contig:
                k_attn = k_attn.contiguous()
            if force_v_contig:
                v_attn = v_attn.contiguous()

        k_bthd = None
        v_bthd = None
        if not use_bhtd:
            k_bthd = k_attn.transpose(1, 2)
            v_bthd = v_attn.transpose(1, 2)
            if not no_qkv_contig:
                k_bthd = k_bthd.contiguous()
                v_bthd = v_bthd.contiguous()

        out = torch.empty_like(q_attn)
        collect_last = self._env_flag("VGGT_DSA_SPARSE_STREAM_RECORD_LAST")
        sort_kv = self._should_use_sparse_stream_sort_kv()
        last_topk_parts = [] if collect_last else None
        source_downsample_prepared = None
        qk_sym_x2_selector = False
        if self._should_use_source_downsample_selector(state, idx_k, idx_k, None, False):
            source_downsample_prepared = self._prepare_source_downsample_projected(idx_k, idx_w, state)
            if source_downsample_prepared is not None:
                qk_sym_x2_selector = self._resolve_source_downsample_strategy(state, source_downsample_prepared.layout) == "qk_sym_x2_broadcast"
                if qk_sym_x2_selector:
                    query_chunk = self._resolve_sparse_stream_qk_sym_x2_query_chunk(
                        state,
                        tgt_len=int(tgt_len),
                        base_query_chunk=int(query_chunk),
                    )
                    consume_query_chunk = min(int(consume_query_chunk), int(query_chunk))

        for q_start in range(0, int(tgt_len), int(query_chunk)):
            q_end = min(q_start + int(query_chunk), int(tgt_len))
            idx_q_outer = None
            topk_outer = None
            query_projected_prepared_outer = None
            if not qk_sym_x2_selector:
                idx_q_outer = (
                    self.indexer.project_query(
                        indexer_input[:, q_start:q_end],
                        pos=None if pos is None else pos[:, q_start:q_end],
                    )
                    if stream_indexer_q_chunk
                    else idx_q[:, q_start:q_end]
                )
                self._debug_sparse_stream_mem("project_q_chunk", q_start=q_start)
            else:
                query_projected_prepared_outer = self._prepare_query_downsample_projected_chunk(
                    indexer_input[:, q_start:q_end],
                    None if pos is None else pos[:, q_start:q_end],
                    source_downsample_prepared.layout,
                    int(q_start),
                )
                self._debug_sparse_stream_mem("project_q_chunk", q_start=q_start)
                topk_outer = self._select_source_downsample_qk_sym_x2_coarse(
                    source_downsample_prepared,
                    state,
                    int(topk),
                    coarse_topk=self._resolve_source_downsample_coarse_topk(
                        state,
                        source_downsample_prepared.layout,
                        int(topk),
                    ),
                    query_prepared=query_projected_prepared_outer,
                )
                self._debug_sparse_stream_mem("select_topk_done", q_start=q_start)
                if sort_kv:
                    topk_outer = torch.sort(topk_outer, dim=-1)[0]
            outer_rows = int(q_end - q_start)
            for inner_start in range(0, outer_rows, int(consume_query_chunk)):
                inner_end = min(inner_start + int(consume_query_chunk), outer_rows)
                sub_q_start = q_start + inner_start
                sub_q_end = q_start + inner_end
                if qk_sym_x2_selector:
                    topk_indices = self._broadcast_source_downsample_chunk_to_queries(
                        topk_outer,
                        query_projected_prepared_outer,
                        fine_q_len=outer_rows,
                        fine_start=inner_start,
                        fine_end=inner_end,
                    )
                else:
                    idx_q_chunk = idx_q_outer[:, inner_start:inner_end]
                    topk_indices = self._select_topk_projected_inference(
                        idx_q_chunk,
                        idx_k,
                        idx_w,
                        state,
                        mask=None,
                        topk=topk,
                        return_scores=False,
                        prepared=source_downsample_prepared,
                        q_offset=sub_q_start,
                    )
                    self._debug_sparse_stream_mem("select_topk_done", q_start=sub_q_start)
                    if sort_kv:
                        topk_indices = torch.sort(topk_indices, dim=-1)[0]
                if not qk_sym_x2_selector and sort_kv:
                    topk_indices = torch.sort(topk_indices, dim=-1)[0]
                kv_positions = topk_indices if topk_indices.dtype == torch.int32 else topk_indices.to(torch.int32)
                if collect_last:
                    last_topk_parts.append(kv_positions.detach())
                self._debug_sparse_stream_mem("kv_positions_ready", q_start=sub_q_start)
                if force_kv_contig and (not kv_positions.is_contiguous()):
                    kv_positions = kv_positions.contiguous()
                    self._debug_sparse_stream_mem("kv_positions_contig", q_start=sub_q_start)

                local_k_attn = k_attn
                local_v_attn = v_attn
                local_kv_positions = kv_positions
                if pack_local_kv:
                    packed = self._pack_sparse_stream_kv_chunk(
                        k_attn,
                        v_attn,
                        kv_positions,
                        seqlen_k=int(tgt_len),
                    )
                    if packed is not None:
                        local_k_attn, local_v_attn, local_kv_positions = packed
                        self._debug_sparse_stream_mem("pack_kv_done", q_start=sub_q_start)
                del topk_indices

                if use_bhtd:
                    q_chunk = q_attn[:, :, sub_q_start:sub_q_end]
                    if (chunk_q_contig_only or (not no_qkv_contig)) and (not q_chunk.is_contiguous()):
                        q_chunk = q_chunk.contiguous()
                        self._debug_sparse_stream_mem("q_chunk_contig", q_start=sub_q_start)
                    out_chunk, _ = sparse_index_flash_attn_inference_bhtd_func(
                        q_chunk,
                        local_k_attn,
                        local_v_attn,
                        local_kv_positions,
                        None,
                        False,
                        float(self.scale),
                        False,
                    )
                    out[:, :, sub_q_start:sub_q_end] = out_chunk
                    self._debug_sparse_stream_mem("sparse_attn_done", q_start=sub_q_start)
                else:
                    q_chunk = q_attn[:, :, sub_q_start:sub_q_end].transpose(1, 2)
                    if (chunk_q_contig_only or (not no_qkv_contig)) and (not q_chunk.is_contiguous()):
                        q_chunk = q_chunk.contiguous()
                        self._debug_sparse_stream_mem("q_chunk_contig", q_start=sub_q_start)
                    out_chunk, _ = sparse_index_flash_attn_inference_func(
                        q_chunk,
                        k_bthd,
                        v_bthd,
                        local_kv_positions,
                        None,
                        False,
                        float(self.scale),
                        False,
                    )
                    out[:, :, sub_q_start:sub_q_end] = out_chunk.transpose(1, 2)
                    self._debug_sparse_stream_mem("sparse_attn_done", q_start=sub_q_start)
                del kv_positions
                del local_kv_positions
            if idx_q_outer is not None:
                del idx_q_outer
            if topk_outer is not None:
                del topk_outer

        if collect_last:
            self.last_topk_indices = torch.cat(last_topk_parts, dim=1)
            self.last_topk_scores = None
        else:
            self.last_topk_indices = None
            self.last_topk_scores = None

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        out = out.transpose(1, 2).reshape(bsz, tgt_len, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def _run_sparse_streaming_attn_q_chunk_direct(
        self,
        x: Tensor,
        indexer_input: Tensor,
        pos: Optional[Tensor],
        state: dotdict,
    ) -> Optional[Tensor]:
        if self.indexer is None:
            return None

        bsz, tgt_len, _ = x.shape
        topk = min(int(state.get("topk", self.indexer_cfg.topk)), int(tgt_len))
        query_chunk = self._resolve_sparse_stream_query_chunk(int(tgt_len), int(topk))
        if query_chunk <= 0:
            return None
        consume_query_chunk = self._resolve_sparse_stream_consume_query_chunk(int(query_chunk))

        orig_dtype = x.dtype
        score_dtype = self._resolve_score_dtype(state, orig_dtype)
        stream_indexer_q_chunk = self._should_use_sparse_stream_indexer_q_chunk_proj()

        k_attn, v_attn, idx_k, idx_w = self._project_sparse_stream_kv_and_indexer_kw_fused(
            indexer_input,
            pos,
            score_dtype=score_dtype,
        )
        self._debug_sparse_stream_mem("project_kv_kw_full")

        if k_attn.shape[1] != self.num_heads:
            k_attn = self._expand_attention_kv_heads(k_attn)
            v_attn = self._expand_attention_kv_heads(v_attn)

        force_kv_contig = not self._env_flag("VGGT_SPARSE_FLASH_NO_KV_CONTIG")
        no_qkv_contig = self._env_flag("VGGT_SPARSE_FLASH_NO_QKV_CONTIG")
        use_bhtd = self._env_flag("VGGT_SPARSE_FLASH_DIRECT_BHTD") and (
            sparse_index_flash_attn_inference_bhtd_func is not None
        )
        if not use_bhtd:
            return None
        chunk_q_contig_only = self._should_use_sparse_stream_chunk_q_contig_only(use_bhtd)
        force_k_contig = self._should_force_sparse_stream_k_contig()
        force_v_contig = self._should_force_sparse_stream_v_contig()
        pack_local_kv = self._should_use_sparse_stream_pack_kv(use_bhtd, no_qkv_contig)
        ima_guard_pack_kv = self._should_use_sparse_stream_ima_guard(
            tgt_len=int(tgt_len),
            topk=int(topk),
            use_bhtd=bool(use_bhtd),
            no_qkv_contig=bool(no_qkv_contig),
            force_v_contig=bool(force_v_contig),
            stream_indexer_q_chunk=bool(stream_indexer_q_chunk),
        )
        if ima_guard_pack_kv:
            query_chunk = self._resolve_sparse_stream_ima_guard_qchunk(int(tgt_len), int(topk), int(query_chunk))
            consume_query_chunk = min(int(consume_query_chunk), int(query_chunk))
            force_k_contig = False
            force_v_contig = False
            pack_local_kv = True
        self._debug_sparse_stream(
            tgt_len=int(tgt_len),
            topk=int(topk),
            query_chunk=int(query_chunk),
            consume_query_chunk=int(consume_query_chunk),
            use_bhtd=int(use_bhtd),
            no_qkv_contig=int(no_qkv_contig),
            force_k_contig=int(force_k_contig),
            force_v_contig=int(force_v_contig),
            chunk_q_contig_only=int(chunk_q_contig_only),
            pack_local_kv=int(pack_local_kv),
            ima_guard_pack_kv=int(ima_guard_pack_kv),
            attn_q_chunk_proj=int(use_fused_q_idxq),
        )

        if not no_qkv_contig:
            k_attn = k_attn.contiguous()
            v_attn = v_attn.contiguous()
        else:
            if force_k_contig:
                k_attn = k_attn.contiguous()
            if force_v_contig:
                v_attn = v_attn.contiguous()

        out = torch.empty((bsz, int(tgt_len), self.dim), device=x.device, dtype=orig_dtype)
        collect_last = self._env_flag("VGGT_DSA_SPARSE_STREAM_RECORD_LAST")
        sort_kv = self._should_use_sparse_stream_sort_kv()
        last_topk_parts = [] if collect_last else None
        source_downsample_prepared = None
        qk_sym_x2_selector = False
        if self._should_use_source_downsample_selector(state, idx_k, idx_k, None, False):
            source_downsample_prepared = self._prepare_source_downsample_projected(idx_k, idx_w, state)
            if source_downsample_prepared is not None:
                qk_sym_x2_selector = self._resolve_source_downsample_strategy(state, source_downsample_prepared.layout) == "qk_sym_x2_broadcast"
        use_fused_q_idxq = self._should_use_sparse_stream_attn_q_chunk_proj() and (not qk_sym_x2_selector)

        for q_start in range(0, int(tgt_len), int(query_chunk)):
            q_end = min(q_start + int(query_chunk), int(tgt_len))
            for sub_q_start in range(q_start, q_end, int(consume_query_chunk)):
                sub_q_end = min(sub_q_start + int(consume_query_chunk), q_end)
                x_chunk = x[:, sub_q_start:sub_q_end]
                pos_chunk = None if pos is None else pos[:, sub_q_start:sub_q_end]
                query_projected_prepared = None
                if use_fused_q_idxq:
                    q_chunk, idx_q_chunk = self._project_sparse_stream_q_and_indexer_query_fused(
                        x_chunk,
                        pos_chunk,
                        score_dtype=score_dtype,
                    )
                else:
                    q_chunk = F.linear(
                        x_chunk,
                        self.qkv.weight[:self._attention_q_dim],
                        None if self.qkv.bias is None else self.qkv.bias[:self._attention_q_dim],
                    )
                    q_chunk = q_chunk.view(bsz, sub_q_end - sub_q_start, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                    q_chunk = self.q_norm(q_chunk)
                    if self.rope is not None and pos_chunk is not None:
                        q_chunk = self.rope(q_chunk, pos_chunk)
                    if q_chunk.dtype != score_dtype:
                        q_chunk = q_chunk.to(score_dtype)
                    if qk_sym_x2_selector:
                        query_projected_prepared = self._prepare_query_downsample_projected_chunk(
                            indexer_input[:, sub_q_start:sub_q_end],
                            pos_chunk,
                            source_downsample_prepared.layout,
                            int(sub_q_start),
                        )
                        idx_q_chunk = query_projected_prepared.coarse_q
                    else:
                        idx_q_chunk = self.indexer.project_query(
                            indexer_input[:, sub_q_start:sub_q_end],
                            pos=pos_chunk,
                        )
                self._debug_sparse_stream_mem("project_q_idxq_chunk", q_start=sub_q_start)

                if qk_sym_x2_selector:
                    topk_indices = self._select_topk_source_downsample_chunk(
                        idx_q_chunk,
                        source_downsample_prepared,
                        state,
                        int(topk),
                        q_offset=int(sub_q_start),
                        query_prepared=query_projected_prepared,
                    )
                else:
                    topk_indices = self._select_topk_projected_inference(
                        idx_q_chunk,
                        idx_k,
                        idx_w,
                        state,
                        mask=None,
                        topk=topk,
                        return_scores=False,
                        prepared=source_downsample_prepared,
                        q_offset=sub_q_start,
                    )
                self._debug_sparse_stream_mem("select_topk_done", q_start=sub_q_start)
                if sort_kv:
                    topk_indices = torch.sort(topk_indices, dim=-1)[0]
                kv_positions = topk_indices if topk_indices.dtype == torch.int32 else topk_indices.to(torch.int32)
                if collect_last:
                    last_topk_parts.append(kv_positions.detach())
                self._debug_sparse_stream_mem("kv_positions_ready", q_start=sub_q_start)
                if force_kv_contig and (not kv_positions.is_contiguous()):
                    kv_positions = kv_positions.contiguous()
                    self._debug_sparse_stream_mem("kv_positions_contig", q_start=sub_q_start)

                local_k_attn = k_attn
                local_v_attn = v_attn
                local_kv_positions = kv_positions
                if pack_local_kv:
                    packed = self._pack_sparse_stream_kv_chunk(
                        k_attn,
                        v_attn,
                        kv_positions,
                        seqlen_k=int(tgt_len),
                    )
                    if packed is not None:
                        local_k_attn, local_v_attn, local_kv_positions = packed
                        self._debug_sparse_stream_mem("pack_kv_done", q_start=sub_q_start)
                del topk_indices

                if (chunk_q_contig_only or (not no_qkv_contig)) and (not q_chunk.is_contiguous()):
                    q_chunk = q_chunk.contiguous()
                    self._debug_sparse_stream_mem("q_chunk_contig", q_start=sub_q_start)
                out_chunk, _ = sparse_index_flash_attn_inference_bhtd_func(
                    q_chunk,
                    local_k_attn,
                    local_v_attn,
                    local_kv_positions,
                    None,
                    False,
                    float(self.scale),
                    False,
                )
                self._debug_sparse_stream_mem("sparse_attn_done", q_start=sub_q_start)
                if out_chunk.dtype != orig_dtype:
                    out_chunk = out_chunk.to(orig_dtype)
                out_chunk = out_chunk.transpose(1, 2).reshape(bsz, sub_q_end - sub_q_start, -1)
                out_chunk = self.proj(out_chunk)
                out_chunk = self.proj_drop(out_chunk)
                out[:, sub_q_start:sub_q_end] = out_chunk
                del kv_positions
                del local_kv_positions

        if collect_last:
            self.last_topk_indices = torch.cat(last_topk_parts, dim=1)
            self.last_topk_scores = None
        else:
            self.last_topk_indices = None
            self.last_topk_scores = None
        return out

    def _run_fullchain_direct(
        self,
        x: Tensor,
        pos: Optional[Tensor],
        state: dotdict,
        *,
        record_last: bool,
    ) -> Tensor:
        bsz, tgt_len, _ = x.shape
        indexer_input = x.detach() if state.get("detach_input", False) else x
        projected_indexer = None

        use_fused_proj = self._should_use_fused_proj(state, x)
        if use_fused_proj:
            fused_proj = self._project_qkv_and_indexer_fused(indexer_input, pos)
            if fused_proj is not None:
                qkv, projected_indexer = fused_proj
            else:
                qkv = self.qkv(x)
        else:
            qkv = self.qkv(x)

        q, k, v = self._project_attention_qkv_from_flat(qkv, bsz, tgt_len)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        orig_dtype = q.dtype
        score_dtype = self._resolve_score_dtype(state, orig_dtype)
        if score_dtype != orig_dtype:
            q = q.to(score_dtype)
            k = k.to(score_dtype)
            v = v.to(score_dtype)

        topk = min(int(state.get("topk", self.indexer_cfg.topk)), tgt_len)
        if projected_indexer is None:
            projected_indexer = self.indexer.project(indexer_input, pos=pos)
        source_downsample_prepared = None
        if self._should_use_source_downsample_selector(
            state,
            projected_indexer[0],
            projected_indexer[1],
            None,
            False,
        ):
            source_downsample_prepared = self._prepare_source_downsample_projected(
                projected_indexer[1],
                projected_indexer[2],
                state,
            )
        topk_indices = self._select_topk_projected_inference(
            projected_indexer[0],
            projected_indexer[1],
            projected_indexer[2],
            state,
            mask=None,
            topk=topk,
            return_scores=False,
            prepared=source_downsample_prepared,
            q_offset=0,
        )
        if record_last:
            self.last_topk_indices = topk_indices.detach()
            self.last_topk_scores = None

        kv_positions = topk_indices.to(torch.int32)
        force_kv_contig = not self._env_flag("VGGT_SPARSE_FLASH_NO_KV_CONTIG")
        if force_kv_contig and (not kv_positions.is_contiguous()):
            kv_positions = kv_positions.contiguous()

        if q.dtype not in (torch.float16, torch.bfloat16):
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)

        q_in = q.transpose(1, 2)
        k_in = k.transpose(1, 2)
        v_in = v.transpose(1, 2)
        no_qkv_contig = self._env_flag("VGGT_SPARSE_FLASH_NO_QKV_CONTIG")
        if not no_qkv_contig:
            q_in = q_in.contiguous()
            k_in = k_in.contiguous()
            v_in = v_in.contiguous()

        out, _ = sparse_index_flash_attn_inference_func(
            q_in,
            k_in,
            v_in,
            kv_positions,
            None,
            False,
            float(self.scale),
            False,
        )
        out = out.transpose(1, 2)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        out = out.transpose(1, 2).reshape(bsz, tgt_len, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def _run_fullchain_graph(
        self,
        x: Tensor,
        pos: Optional[Tensor],
        state: dotdict,
    ) -> Optional[Tensor]:
        key = self._fullchain_graph_key(x, pos, state)
        if key in self._fullchain_graph_failures:
            return None

        entry = self._fullchain_graph_cache.get(key, None)
        if entry is None:
            try:
                static_x = torch.empty_like(x)
                static_pos = torch.empty_like(pos) if pos is not None else None
                warmup = max(1, self._env_int("VGGT_DSA_FULLCHAIN_GRAPH_WARMUP", 1))
                with torch.no_grad():
                    for _ in range(warmup):
                        _ = self._run_fullchain_direct(static_x, static_pos, state, record_last=False)
                    torch.cuda.synchronize(x.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        static_out = self._run_fullchain_direct(static_x, static_pos, state, record_last=False)
                    torch.cuda.synchronize(x.device)

                entry = {
                    "graph": graph,
                    "static_x": static_x,
                    "static_pos": static_pos,
                    "static_out": static_out,
                }
                max_entries = max(1, self._env_int("VGGT_DSA_FULLCHAIN_GRAPH_MAX_ENTRIES", 32))
                if len(self._fullchain_graph_cache) >= max_entries:
                    oldest_key = next(iter(self._fullchain_graph_cache))
                    old_entry = self._fullchain_graph_cache.pop(oldest_key, None)
                    if old_entry is not None:
                        del old_entry
                        if self._env_flag("VGGT_DSA_FULLCHAIN_GRAPH_EVICT_EMPTY_CACHE"):
                            gc.collect()
                            torch.cuda.empty_cache()
                self._fullchain_graph_cache[key] = entry
            except Exception as exc:
                self._fullchain_graph_failures.add(key)
                if self._env_flag("VGGT_DSA_FULLCHAIN_DEBUG"):
                    log(f"[dsa] fullchain graph build fallback: {repr(exc)}")
                return None

        try:
            entry["static_x"].copy_(x)
            if pos is not None and entry["static_pos"] is not None:
                entry["static_pos"].copy_(pos)
            entry["graph"].replay()
            return entry["static_out"].clone()
        except Exception as exc:
            self._fullchain_graph_failures.add(key)
            self._fullchain_graph_cache.pop(key, None)
            if self._env_flag("VGGT_DSA_FULLCHAIN_DEBUG"):
                log(f"[dsa] fullchain graph replay fallback: {repr(exc)}")
            return None

    def _run_fullchain_fastpath(
        self,
        x: Tensor,
        pos: Optional[Tensor],
        state: dotdict,
    ) -> Optional[Tensor]:
        try:
            if self._env_flag("VGGT_DSA_FULLCHAIN_CUDAGRAPH"):
                out = self._run_fullchain_graph(x, pos, state)
                if out is not None:
                    return out
            return self._run_fullchain_direct(x, pos, state, record_last=True)
        except Exception as exc:
            if self._env_flag("VGGT_DSA_FULLCHAIN_DEBUG"):
                log(f"[dsa] fullchain fastpath fallback: {repr(exc)}")
            return None

    @staticmethod
    def _mask_fill_value(dtype: torch.dtype) -> float:
        if dtype.is_floating_point:
            return float(torch.finfo(dtype).min)
        return -1e9

    def _mem_trace_rank_ok(self) -> bool:
        if self._env_flag("VGGT_MEM_TRACE_ALL_RANKS"):
            return True
        return is_main_process()

    def _state(self):
        state = dotdict(self.indexer_cfg)
        state.update(getattr(self, "indexer_state", dotdict()))
        return state

    def _soft_view_bias_enabled(self, state: dotdict) -> bool:
        return bool(
            state.get("soft_view_bias_enabled", False)
            and self.soft_view_bias_q_proj is not None
            and self.soft_view_bias_k_proj is not None
            and getattr(self, "soft_view_bias_scale", None) is not None
        )

    @staticmethod
    def _objective_value_gate_enabled(state: dotdict) -> bool:
        return bool(state.get("objective_value_gate_enabled", False))

    def _resolve_objective_value_gate_query_chunk(self, state: dotdict, tgt_len: int) -> int:
        query_chunk = int(
            state.get(
                "objective_value_gate_query_chunk_size",
                self.indexer_cfg.get("objective_value_gate_query_chunk_size", 64),
            )
            or 0
        )
        if query_chunk <= 0:
            query_chunk = self._env_int("VGGT_OBJECTIVE_VALUE_GATE_QCHUNK", 64)
        query_chunk = max(int(query_chunk), 1)
        return min(int(tgt_len), int(query_chunk))

    def _resolve_objective_value_gate_value_chunk(self, state: dotdict, topk: int) -> int:
        value_chunk = int(
            state.get(
                "objective_value_gate_value_chunk_size",
                self.indexer_cfg.get("objective_value_gate_value_chunk_size", 0),
            )
            or 0
        )
        if value_chunk <= 0:
            value_chunk = self._env_int("VGGT_OBJECTIVE_VALUE_GATE_VCHUNK", 0)
        if value_chunk <= 0:
            return int(topk)
        return max(1, min(int(value_chunk), int(topk)))

    def _resolve_objective_value_gate_gather_chunk(self, state: dotdict, topk: int) -> int:
        gather_chunk = int(
            state.get(
                "objective_value_gate_gather_chunk_size",
                self.indexer_cfg.get("objective_value_gate_gather_chunk_size", 0),
            )
            or 0
        )
        if gather_chunk <= 0:
            gather_chunk = self._env_int("VGGT_OBJECTIVE_VALUE_GATE_GATHER_CHUNK", 0)
        if gather_chunk <= 0:
            return int(topk)
        return max(1, min(int(gather_chunk), int(topk)))

    def _build_objective_value_gate(
        self,
        topk_scores: Tensor,
        state: dotdict,
        *,
        target_dtype: torch.dtype,
    ) -> Tensor:
        score_scale = float(state.get("objective_value_gate_scale", self.indexer_cfg.get("objective_value_gate_scale", 0.0)) or 0.0)
        if score_scale == 0.0:
            return torch.ones(
                (topk_scores.shape[0], 1, topk_scores.shape[1], topk_scores.shape[2]),
                device=topk_scores.device,
                dtype=target_dtype,
            )

        tau = float(state.get("objective_value_gate_tau", self.indexer_cfg.get("objective_value_gate_tau", 1.0)) or 1.0)
        tau = max(abs(tau), 1e-6)
        eps = float(state.get("objective_value_gate_eps", self.indexer_cfg.get("objective_value_gate_eps", 1e-6)) or 1e-6)
        eps = max(eps, 1e-12)

        gate_scores = topk_scores.to(torch.float32)
        gate_scores = gate_scores - gate_scores.mean(dim=-1, keepdim=True)
        gate_var = gate_scores.square().mean(dim=-1, keepdim=True)
        gate_scores = gate_scores * torch.rsqrt(gate_var + eps)
        gate = 1.0 + score_scale * torch.tanh(gate_scores / tau)
        gate = torch.clamp_min(gate, eps)
        return gate[:, None].to(dtype=target_dtype)

    @staticmethod
    def _apply_attention_to_values(attn: Tensor, v_sel: Tensor, key_chunk_size: int = 0) -> Tensor:
        topk = int(attn.shape[-1])
        key_chunk = int(key_chunk_size or 0)
        if key_chunk <= 0:
            return torch.matmul(attn.unsqueeze(-2), v_sel).squeeze(-2)
        key_chunk = max(1, min(key_chunk, topk))

        out = None
        for k_start in range(0, topk, key_chunk):
            k_end = min(k_start + key_chunk, topk)
            attn_chunk = attn[..., k_start:k_end]
            v_chunk = v_sel[..., k_start:k_end, :]
            out_chunk = torch.bmm(
                attn_chunk.reshape(-1, 1, k_end - k_start),
                v_chunk.reshape(-1, k_end - k_start, v_chunk.shape[-1]),
            ).reshape(*attn_chunk.shape[:-1], v_chunk.shape[-1])
            if out is None:
                out = out_chunk
            else:
                out.add_(out_chunk)
        return out

    @staticmethod
    def _compute_sparse_scores(q: Tensor, k_sel: Tensor, scale: float, key_chunk_size: int = 0) -> Tensor:
        topk = int(k_sel.shape[-2])
        key_chunk = int(key_chunk_size or 0)
        if key_chunk <= 0:
            return torch.matmul(q.unsqueeze(-2), k_sel.transpose(-1, -2)).squeeze(-2) * scale
        key_chunk = max(1, min(key_chunk, topk))

        q_flat = q.reshape(-1, 1, q.shape[-1])
        score_chunks = []
        for k_start in range(0, topk, key_chunk):
            k_end = min(k_start + key_chunk, topk)
            k_chunk = k_sel[..., k_start:k_end, :]
            score_chunks.append(
                torch.bmm(
                    q_flat,
                    k_chunk.reshape(-1, k_end - k_start, k_chunk.shape[-1]).transpose(1, 2),
                ).reshape(*q.shape[:-1], k_end - k_start)
            )
        return torch.cat(score_chunks, dim=-1) * scale

    @staticmethod
    def _build_view_ids(tokens_per_view: int, total_tokens: int, device: torch.device) -> Tensor:
        return torch.arange(total_tokens, device=device, dtype=torch.long) // int(tokens_per_view)

    @staticmethod
    def _slice_view_bias_data(
        view_bias_data: Optional[dict],
        q_offset: int,
        q_len: int,
        s_view_ids: Optional[Tensor] = None,
    ) -> Optional[dict]:
        if view_bias_data is None:
            return None
        sliced = dict(
            q_view_ids=view_bias_data["q_view_ids"][:, int(q_offset): int(q_offset) + int(q_len)],
            s_view_ids=view_bias_data["s_view_ids"] if s_view_ids is None else s_view_ids,
            view_bias=view_bias_data["view_bias"],
        )
        return sliced

    def _build_soft_view_bias_data(
        self,
        indexer_input: Tensor,
        state: dotdict,
    ) -> Optional[dict]:
        if not self._soft_view_bias_enabled(state):
            return None
        tokens_per_view = int(state.get("tokens_per_view", 0) or 0)
        num_views = int(state.get("num_views", 0) or 0)
        if tokens_per_view <= 0 or num_views <= 0:
            return None
        total_tokens = tokens_per_view * num_views
        if int(indexer_input.shape[1]) != total_tokens:
            return None

        bsz, _, dim = indexer_input.shape
        tokens_by_view = indexer_input.reshape(bsz, num_views, tokens_per_view, dim)
        descriptor_mode = str(
            state.get("soft_view_bias_descriptor", self.indexer_cfg.get("soft_view_bias_descriptor", "camera"))
            or "camera"
        ).strip().lower()
        if descriptor_mode == "camera":
            desc = tokens_by_view[:, :, 0, :]
        elif descriptor_mode in ("image_pool", "image", "patch_pool"):
            patch_start_idx = int(state.get("patch_start_idx", 0) or 0)
            if tokens_per_view <= patch_start_idx:
                return None
            desc = tokens_by_view[:, :, patch_start_idx:, :].mean(dim=2)
        else:
            raise ValueError(f"Unsupported soft_view_bias_descriptor: {descriptor_mode}")

        q_desc = F.normalize(self.soft_view_bias_q_proj(desc), dim=-1, eps=1e-6)
        k_desc = F.normalize(self.soft_view_bias_k_proj(desc), dim=-1, eps=1e-6)
        view_bias = torch.einsum("bvd,bwd->bvw", q_desc, k_desc)
        view_bias = view_bias * self.soft_view_bias_scale.to(dtype=view_bias.dtype)
        view_ids = self._build_view_ids(tokens_per_view, total_tokens, indexer_input.device).view(1, total_tokens).expand(bsz, -1)
        return dict(
            q_view_ids=view_ids,
            s_view_ids=view_ids,
            view_bias=view_bias,
        )

    def _epipolar_selector_band_enabled(self, state: dotdict) -> bool:
        return bool(
            (
                state.get("geometry_support_enabled", False)
                or state.get("epipolar_selector_band_enabled", False)
                or state.get("epipolar_selector_kl_enabled", False)
            )
            and state.get("compute_loss", False)
            and state.get("sparse", False)
            and state.get("extrinsics", None) is not None
            and state.get("intrinsics", None) is not None
            and int(state.get("image_height", 0) or 0) > 0
            and int(state.get("image_width", 0) or 0) > 0
        )

    @staticmethod
    def _epipolar_selector_loss_weight(state: dotdict) -> float:
        if bool(state.get("geometry_support_enabled", False)):
            return float(state.get("geometry_support_weight", 0.1) or 0.1)
        return float(state.get("epipolar_selector_band_weight", state.get("epipolar_selector_kl_weight", 0.1)) or 0.1)

    @staticmethod
    def _geometry_support_mode(state: dotdict) -> str:
        if bool(state.get("geometry_support_enabled", False)):
            mode = str(state.get("geometry_support_type", "auto") or "auto").strip().lower()
            aliases = {
                "depth": "depth_reprojection",
                "depth_reproj": "depth_reprojection",
                "reprojection": "depth_reprojection",
                "pose": "epipolar",
                "epi": "epipolar",
            }
            return aliases.get(mode, mode)
        return "epipolar"

    @staticmethod
    def _geometry_support_depth_available(state: dotdict) -> bool:
        depths = state.get("depths", None)
        return depths is not None and torch.is_tensor(depths) and depths.dim() == 4

    @staticmethod
    def _geometry_support_depth_radius_px(state: dotdict) -> float:
        radius = float(state.get("geometry_support_depth_radius_px", 0.0) or 0.0)
        if radius > 0.0:
            return radius
        return float(state.get("epipolar_selector_band_px", state.get("epipolar_selector_sigma_px", 12.0)) or 12.0)

    def _build_selector_support_mask(
        self,
        *,
        query_token_indices: Tensor,
        state: dotdict,
        num_views: int,
        tokens_per_view: int,
        patch_start_idx: int,
        patch_grid_height: int,
        patch_grid_width: int,
        image_height: int,
        image_width: int,
        extrinsics: Tensor,
        intrinsics: Tensor,
        source_mask: Optional[Tensor],
        source_view_indices: Tensor,
        source_patch_indices: Tensor,
        band_px: float,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mode = self._geometry_support_mode(state)
        query_indices = query_token_indices.view(1, -1).expand(extrinsics.shape[0], -1)

        if mode not in ("auto", "depth_reprojection", "epipolar"):
            raise ValueError(f"Unsupported geometry_support_type: {mode}")

        use_depth = mode in ("auto", "depth_reprojection") and self._geometry_support_depth_available(state)
        if use_depth:
            depth_support_mask, depth_valid_mask, source_token_indices, depth_query_valid = build_depth_reprojection_support_mask(
                query_token_indices=query_indices,
                num_views=num_views,
                tokens_per_view=tokens_per_view,
                patch_start_idx=patch_start_idx,
                patch_grid_height=patch_grid_height,
                patch_grid_width=patch_grid_width,
                image_height=image_height,
                image_width=image_width,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                depths=state.depths,
                point_masks=state.get("point_masks", None),
                radius_px=self._geometry_support_depth_radius_px(state),
                min_depth=float(state.get("geometry_support_min_depth", 1e-4) or 1e-4),
                exclude_self_view=True,
                source_mask=source_mask,
                source_view_indices=source_view_indices,
                source_patch_indices=source_patch_indices,
            )
            if mode == "depth_reprojection":
                return depth_support_mask, depth_valid_mask, source_token_indices
            if bool(depth_query_valid.all()):
                return depth_support_mask, depth_valid_mask, source_token_indices
            fallback_query_mask = (~depth_query_valid).any(dim=0)
            if not bool(fallback_query_mask.any()):
                return depth_support_mask, depth_valid_mask, source_token_indices
            epipolar_query_token_indices = query_token_indices[fallback_query_mask]
        else:
            epipolar_query_token_indices = query_token_indices

        if mode == "depth_reprojection":
            source_token_indices = build_image_token_indices(
                num_views=num_views,
                tokens_per_view=tokens_per_view,
                patch_start_idx=patch_start_idx,
                device=query_token_indices.device,
                view_indices=source_view_indices,
                patch_indices=source_patch_indices,
            )
            empty = torch.zeros(
                (extrinsics.shape[0], int(query_token_indices.numel()), int(source_token_indices.numel())),
                device=query_token_indices.device,
                dtype=torch.bool,
            )
            return empty, empty, source_token_indices

        epipolar_support_mask, epipolar_valid_mask, source_token_indices = build_epipolar_band_mask(
            query_token_indices=epipolar_query_token_indices.view(1, -1).expand(extrinsics.shape[0], -1),
            num_views=num_views,
            tokens_per_view=tokens_per_view,
            patch_start_idx=patch_start_idx,
            patch_grid_height=patch_grid_height,
            patch_grid_width=patch_grid_width,
            image_height=image_height,
            image_width=image_width,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            band_px=band_px,
            exclude_self_view=True,
            source_mask=source_mask,
            source_view_indices=source_view_indices,
            source_patch_indices=source_patch_indices,
        )
        if not use_depth:
            return epipolar_support_mask, epipolar_valid_mask, source_token_indices

        support_mask = depth_support_mask.clone()
        valid_mask = depth_valid_mask.clone()
        fallback_depth_valid = depth_query_valid[:, fallback_query_mask, None]
        support_mask[:, fallback_query_mask] = torch.where(
            fallback_depth_valid,
            depth_support_mask[:, fallback_query_mask],
            epipolar_support_mask,
        )
        valid_mask[:, fallback_query_mask] = torch.where(
            fallback_depth_valid,
            depth_valid_mask[:, fallback_query_mask],
            epipolar_valid_mask,
        )
        return support_mask, valid_mask, source_token_indices

    @staticmethod
    def _gather_view_bias_data(
        view_bias_data: Optional[dict],
        query_indices: Tensor,
        source_indices: Tensor,
        tokens_per_view: int,
        bsz: int,
    ) -> Optional[dict]:
        if view_bias_data is None:
            return None
        query_view_ids = torch.div(query_indices, int(tokens_per_view), rounding_mode="floor")
        source_view_ids = torch.div(source_indices, int(tokens_per_view), rounding_mode="floor")
        return dict(
            q_view_ids=query_view_ids.view(1, -1).expand(bsz, -1),
            s_view_ids=source_view_ids.view(1, -1).expand(bsz, -1),
            view_bias=view_bias_data["view_bias"],
        )

    def _epipolar_selector_loss_mode(self, state: dotdict) -> str:
        return str(
            state.get(
                "epipolar_selector_loss_mode",
                self.indexer_cfg.get("epipolar_selector_loss_mode", "full"),
            )
            or "full"
        ).strip().lower()

    @staticmethod
    def _candidate_index_mask(
        index_mask: Optional[Tensor],
        query_token_indices: Tensor,
        source_token_indices: Tensor,
        fill_threshold: float,
    ) -> Optional[Tensor]:
        if index_mask is None:
            return None
        source_clamped = source_token_indices.clamp(min=0, max=max(int(index_mask.shape[-1]) - 1, 0)).to(torch.long)
        query_mask = index_mask[:, query_token_indices]
        gathered = torch.gather(query_mask, dim=-1, index=source_clamped)
        if gathered.dtype.is_floating_point:
            return gathered > fill_threshold
        return gathered.bool()

    def _build_selector_support_for_source_tokens(
        self,
        *,
        query_token_indices: Tensor,
        source_token_indices: Tensor,
        state: dotdict,
        num_views: int,
        tokens_per_view: int,
        patch_start_idx: int,
        patch_grid_height: int,
        patch_grid_width: int,
        image_height: int,
        image_width: int,
        extrinsics: Tensor,
        intrinsics: Tensor,
        index_mask: Optional[Tensor],
        band_px: float,
        patch_centers: Optional[Tensor] = None,
        fundamental: Optional[Tensor] = None,
        intrinsics_inv: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if source_token_indices.dim() != 3:
            raise ValueError(f"Expected source_token_indices [B,Tq,K], got {tuple(source_token_indices.shape)}")
        device = source_token_indices.device
        bsz, tgt_len, candidate_count = source_token_indices.shape
        patch_token_count = int(tokens_per_view) - int(patch_start_idx)
        if candidate_count <= 0 or patch_token_count <= 0:
            empty = torch.zeros((bsz, tgt_len, candidate_count), device=device, dtype=torch.bool)
            return empty, empty

        query_indices = query_token_indices.to(device=device, dtype=torch.long).view(1, -1).expand(bsz, -1)
        source_indices = source_token_indices.to(device=device, dtype=torch.long)
        source_indices_clamped = source_indices.clamp(min=0, max=max(int(num_views) * int(tokens_per_view) - 1, 0))
        query_view_ids = torch.div(query_indices, int(tokens_per_view), rounding_mode="floor")
        source_view_ids = torch.div(source_indices_clamped, int(tokens_per_view), rounding_mode="floor")
        source_local_patch = torch.remainder(source_indices_clamped, int(tokens_per_view)) - int(patch_start_idx)
        source_image = (
            (source_indices >= 0)
            & (source_view_ids >= 0)
            & (source_view_ids < int(num_views))
            & (source_local_patch >= 0)
            & (source_local_patch < patch_token_count)
        )
        valid_mask = source_image & (source_view_ids != query_view_ids[..., None])
        candidate_index_mask = self._candidate_index_mask(
            index_mask=index_mask,
            query_token_indices=query_token_indices.to(device=device, dtype=torch.long),
            source_token_indices=source_indices,
            fill_threshold=self._mask_fill_value(index_mask.dtype) * 0.5 if index_mask is not None and index_mask.dtype.is_floating_point else 0.0,
        )
        if candidate_index_mask is not None:
            valid_mask = valid_mask & candidate_index_mask

        if patch_centers is None:
            patch_centers = build_patch_token_centers(
                image_height=image_height,
                image_width=image_width,
                patch_grid_height=patch_grid_height,
                patch_grid_width=patch_grid_width,
                device=device,
                dtype=torch.float32,
            )
        else:
            patch_centers = patch_centers.to(device=device, dtype=torch.float32)
        query_local_patch = torch.remainder(query_indices, int(tokens_per_view)) - int(patch_start_idx)
        query_local_patch = query_local_patch.clamp(min=0, max=max(patch_token_count - 1, 0))
        query_centers = patch_centers[query_local_patch.reshape(-1)].reshape(bsz, tgt_len, 3)
        source_centers = patch_centers[source_local_patch.clamp(min=0, max=max(patch_token_count - 1, 0)).reshape(-1)]
        source_centers = source_centers.reshape(bsz, tgt_len, candidate_count, 3)

        extrinsics = extrinsics.to(device=device, dtype=torch.float32)
        intrinsics = intrinsics.to(device=device, dtype=torch.float32)
        source_view_ids_clamped = source_view_ids.clamp(min=0, max=max(int(num_views) - 1, 0))

        if fundamental is None:
            fundamental = compute_query_to_source_fundamental_matrices(extrinsics=extrinsics, intrinsics=intrinsics)
        else:
            fundamental = fundamental.to(device=device, dtype=torch.float32)
        batch_q = torch.arange(bsz, device=device)[:, None, None].expand(bsz, tgt_len, candidate_count)
        query_f = fundamental[
            batch_q,
            query_view_ids[..., None].expand(-1, -1, candidate_count),
            source_view_ids_clamped,
        ]
        lines = torch.einsum("btkij,btj->btki", query_f, query_centers)
        numer = torch.abs((source_centers * lines).sum(dim=-1))
        denom = lines[..., :2].square().sum(dim=-1).sqrt()
        epipolar_valid = valid_mask & torch.isfinite(denom) & (denom > 1e-6)
        epipolar_support = epipolar_valid & ((numer / denom.clamp_min(1e-6)) <= float(band_px))

        mode = self._geometry_support_mode(state)
        if mode not in ("auto", "depth_reprojection", "epipolar"):
            raise ValueError(f"Unsupported geometry_support_type: {mode}")
        use_depth = mode in ("auto", "depth_reprojection") and self._geometry_support_depth_available(state)
        if not use_depth:
            if mode == "depth_reprojection":
                empty = torch.zeros_like(valid_mask)
                return empty, empty
            return epipolar_support, epipolar_valid

        depths = state.depths.to(device=device, dtype=torch.float32)
        depth_h, depth_w = int(depths.shape[-2]), int(depths.shape[-1])
        sample_x = query_centers[..., 0] * (float(depth_w) / float(image_width))
        sample_y = query_centers[..., 1] * (float(depth_h) / float(image_height))
        sample_x = sample_x.round().long().clamp(min=0, max=max(depth_w - 1, 0))
        sample_y = sample_y.round().long().clamp(min=0, max=max(depth_h - 1, 0))
        batch_t = torch.arange(bsz, device=device)[:, None].expand(bsz, tgt_len)
        query_depth = depths[batch_t, query_view_ids, sample_y, sample_x]
        query_depth_valid = torch.isfinite(query_depth) & (query_depth > float(state.get("geometry_support_min_depth", 1e-4) or 1e-4))
        point_masks = state.get("point_masks", None)
        if point_masks is not None:
            point_masks = point_masks.to(device=device, dtype=torch.bool)
            query_depth_valid = query_depth_valid & point_masks[batch_t, query_view_ids, sample_y, sample_x]

        if intrinsics_inv is None:
            intrinsics_inv = torch.linalg.inv(intrinsics)
        else:
            intrinsics_inv = intrinsics_inv.to(device=device, dtype=torch.float32)
        query_k_inv = intrinsics_inv[batch_t, query_view_ids]
        query_rays = torch.matmul(query_k_inv, query_centers.unsqueeze(-1)).squeeze(-1)
        query_cam_points = query_rays * query_depth.clamp_min(float(state.get("geometry_support_min_depth", 1e-4) or 1e-4)).unsqueeze(-1)
        rot = extrinsics[..., :3]
        trans = extrinsics[..., 3]
        query_rot = rot[batch_t, query_view_ids]
        query_trans = trans[batch_t, query_view_ids]
        world_points = torch.matmul(
            query_rot.transpose(-1, -2),
            (query_cam_points - query_trans).unsqueeze(-1),
        ).squeeze(-1)

        source_rot = rot[batch_q, source_view_ids_clamped]
        source_trans = trans[batch_q, source_view_ids_clamped]
        source_cam_points = torch.matmul(source_rot, world_points[:, :, None, :, None]).squeeze(-1) + source_trans
        source_z = source_cam_points[..., 2]
        source_intrinsics = intrinsics[batch_q, source_view_ids_clamped]
        source_pixels_h = torch.matmul(source_intrinsics, source_cam_points.unsqueeze(-1)).squeeze(-1)
        projected_xy = source_pixels_h[..., :2] / source_pixels_h[..., 2:].clamp_min(float(state.get("geometry_support_min_depth", 1e-4) or 1e-4))
        projected_inside = (
            query_depth_valid[..., None]
            & torch.isfinite(projected_xy).all(dim=-1)
            & (source_z > float(state.get("geometry_support_min_depth", 1e-4) or 1e-4))
            & (projected_xy[..., 0] >= 0.0)
            & (projected_xy[..., 0] <= float(image_width) - 1.0)
            & (projected_xy[..., 1] >= 0.0)
            & (projected_xy[..., 1] <= float(image_height) - 1.0)
        )
        dist_sq = (source_centers[..., 0] - projected_xy[..., 0]).square()
        dist_sq = dist_sq + (source_centers[..., 1] - projected_xy[..., 1]).square()
        depth_valid = valid_mask & projected_inside
        depth_support = depth_valid & (dist_sq <= self._geometry_support_depth_radius_px(state) ** 2)
        if mode == "depth_reprojection":
            return depth_support, depth_valid
        use_depth_query = query_depth_valid[..., None]
        support = torch.where(use_depth_query, depth_support, epipolar_support)
        valid = torch.where(use_depth_query, depth_valid, epipolar_valid)
        return support, valid

    def _build_depth_anchor_indices(
        self,
        *,
        query_token_indices: Tensor,
        state: dotdict,
        num_views: int,
        tokens_per_view: int,
        patch_start_idx: int,
        patch_grid_height: int,
        patch_grid_width: int,
        image_height: int,
        image_width: int,
        extrinsics: Tensor,
        intrinsics: Tensor,
        patch_centers: Optional[Tensor] = None,
        intrinsics_inv: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        device = query_token_indices.device
        if not self._geometry_support_depth_available(state):
            empty_idx = torch.empty((extrinsics.shape[0], int(query_token_indices.numel()), 0), device=device, dtype=torch.long)
            empty_mask = torch.zeros_like(empty_idx, dtype=torch.bool)
            return empty_idx, empty_mask
        bsz = int(extrinsics.shape[0])
        tgt_len = int(query_token_indices.numel())
        query_indices = query_token_indices.to(device=device, dtype=torch.long).view(1, -1).expand(bsz, -1)
        query_view_ids = torch.div(query_indices, int(tokens_per_view), rounding_mode="floor")
        patch_token_count = int(tokens_per_view) - int(patch_start_idx)
        if patch_centers is None:
            patch_centers = build_patch_token_centers(
                image_height=image_height,
                image_width=image_width,
                patch_grid_height=patch_grid_height,
                patch_grid_width=patch_grid_width,
                device=device,
                dtype=torch.float32,
            )
        else:
            patch_centers = patch_centers.to(device=device, dtype=torch.float32)
        query_local_patch = torch.remainder(query_indices, int(tokens_per_view)) - int(patch_start_idx)
        query_centers = patch_centers[query_local_patch.clamp(min=0, max=max(patch_token_count - 1, 0)).reshape(-1)]
        query_centers = query_centers.reshape(bsz, tgt_len, 3)

        depths = state.depths.to(device=device, dtype=torch.float32)
        depth_h, depth_w = int(depths.shape[-2]), int(depths.shape[-1])
        sample_x = query_centers[..., 0] * (float(depth_w) / float(image_width))
        sample_y = query_centers[..., 1] * (float(depth_h) / float(image_height))
        sample_x = sample_x.round().long().clamp(min=0, max=max(depth_w - 1, 0))
        sample_y = sample_y.round().long().clamp(min=0, max=max(depth_h - 1, 0))
        batch_t = torch.arange(bsz, device=device)[:, None].expand(bsz, tgt_len)
        query_depth = depths[batch_t, query_view_ids, sample_y, sample_x]
        min_depth = float(state.get("geometry_support_min_depth", 1e-4) or 1e-4)
        query_depth_valid = torch.isfinite(query_depth) & (query_depth > min_depth)
        point_masks = state.get("point_masks", None)
        if point_masks is not None:
            point_masks = point_masks.to(device=device, dtype=torch.bool)
            query_depth_valid = query_depth_valid & point_masks[batch_t, query_view_ids, sample_y, sample_x]

        extrinsics = extrinsics.to(device=device, dtype=torch.float32)
        intrinsics = intrinsics.to(device=device, dtype=torch.float32)
        if intrinsics_inv is None:
            intrinsics_inv = torch.linalg.inv(intrinsics)
        else:
            intrinsics_inv = intrinsics_inv.to(device=device, dtype=torch.float32)
        query_k_inv = intrinsics_inv[batch_t, query_view_ids]
        query_rays = torch.matmul(query_k_inv, query_centers.unsqueeze(-1)).squeeze(-1)
        query_cam_points = query_rays * query_depth.clamp_min(min_depth).unsqueeze(-1)
        rot = extrinsics[..., :3]
        trans = extrinsics[..., 3]
        query_rot = rot[batch_t, query_view_ids]
        query_trans = trans[batch_t, query_view_ids]
        world_points = torch.matmul(
            query_rot.transpose(-1, -2),
            (query_cam_points - query_trans).unsqueeze(-1),
        ).squeeze(-1)

        source_view_ids = torch.arange(int(num_views), device=device, dtype=torch.long)
        source_rot = rot[:, source_view_ids]
        source_trans = trans[:, source_view_ids]
        source_cam_points = torch.einsum("bvij,btj->btvi", source_rot, world_points) + source_trans[:, None]
        source_z = source_cam_points[..., 2]
        source_intrinsics = intrinsics[:, source_view_ids]
        source_pixels_h = torch.einsum("bvij,btvj->btvi", source_intrinsics, source_cam_points)
        projected_xy = source_pixels_h[..., :2] / source_pixels_h[..., 2:].clamp_min(min_depth)
        projected_inside = (
            query_depth_valid[..., None]
            & torch.isfinite(projected_xy).all(dim=-1)
            & (source_z > min_depth)
            & (projected_xy[..., 0] >= 0.0)
            & (projected_xy[..., 0] <= float(image_width) - 1.0)
            & (projected_xy[..., 1] >= 0.0)
            & (projected_xy[..., 1] <= float(image_height) - 1.0)
            & (source_view_ids.view(1, 1, -1) != query_view_ids[..., None])
        )
        cols = torch.floor(projected_xy[..., 0] * float(patch_grid_width) / float(image_width)).long()
        rows = torch.floor(projected_xy[..., 1] * float(patch_grid_height) / float(image_height)).long()
        radius = max(int(state.get("geometry_support_anchor_patch_radius", 0) or 0), 0)
        offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        anchor_indices = []
        anchor_valid = []
        view_offsets = source_view_ids.view(1, 1, -1) * int(tokens_per_view)
        for dy, dx in offsets:
            rr = rows + int(dy)
            cc = cols + int(dx)
            inside_patch = (rr >= 0) & (rr < int(patch_grid_height)) & (cc >= 0) & (cc < int(patch_grid_width))
            patch_local = rr.clamp(min=0, max=max(int(patch_grid_height) - 1, 0)) * int(patch_grid_width)
            patch_local = patch_local + cc.clamp(min=0, max=max(int(patch_grid_width) - 1, 0))
            anchor_indices.append(view_offsets + int(patch_start_idx) + patch_local)
            anchor_valid.append(projected_inside & inside_patch)
        return torch.cat([idx.reshape(bsz, tgt_len, -1) for idx in anchor_indices], dim=-1), torch.cat(
            [mask.reshape(bsz, tgt_len, -1) for mask in anchor_valid],
            dim=-1,
        )

    def _build_epipolar_anchor_indices(
        self,
        *,
        query_token_indices: Tensor,
        state: dotdict,
        num_views: int,
        tokens_per_view: int,
        patch_start_idx: int,
        patch_grid_height: int,
        patch_grid_width: int,
        image_height: int,
        image_width: int,
        extrinsics: Tensor,
        intrinsics: Tensor,
        patch_centers: Optional[Tensor] = None,
        fundamental: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        samples_per_view = max(int(state.get("geometry_support_anchor_samples_per_view", 8) or 0), 0)
        device = query_token_indices.device
        bsz = int(extrinsics.shape[0])
        tgt_len = int(query_token_indices.numel())
        if samples_per_view <= 0:
            empty_idx = torch.empty((bsz, tgt_len, 0), device=device, dtype=torch.long)
            return empty_idx, torch.zeros_like(empty_idx, dtype=torch.bool)

        query_indices = query_token_indices.to(device=device, dtype=torch.long).view(1, -1).expand(bsz, -1)
        query_view_ids = torch.div(query_indices, int(tokens_per_view), rounding_mode="floor")
        patch_token_count = int(tokens_per_view) - int(patch_start_idx)
        if patch_centers is None:
            patch_centers = build_patch_token_centers(
                image_height=image_height,
                image_width=image_width,
                patch_grid_height=patch_grid_height,
                patch_grid_width=patch_grid_width,
                device=device,
                dtype=torch.float32,
            )
        else:
            patch_centers = patch_centers.to(device=device, dtype=torch.float32)
        query_local_patch = torch.remainder(query_indices, int(tokens_per_view)) - int(patch_start_idx)
        query_centers = patch_centers[query_local_patch.clamp(min=0, max=max(patch_token_count - 1, 0)).reshape(-1)]
        query_centers = query_centers.reshape(bsz, tgt_len, 3)

        extrinsics = extrinsics.to(device=device, dtype=torch.float32)
        intrinsics = intrinsics.to(device=device, dtype=torch.float32)
        if fundamental is None:
            fundamental = compute_query_to_source_fundamental_matrices(extrinsics=extrinsics, intrinsics=intrinsics)
        else:
            fundamental = fundamental.to(device=device, dtype=torch.float32)
        source_view_ids = torch.arange(int(num_views), device=device, dtype=torch.long)
        batch_ids = torch.arange(bsz, device=device)[:, None, None]
        source_ids = source_view_ids.view(1, 1, -1).expand(bsz, tgt_len, -1)
        query_f = fundamental[
            batch_ids,
            query_view_ids[..., None].expand(-1, -1, int(num_views)),
            source_ids,
        ]
        lines = torch.einsum("btvij,btj->btvi", query_f, query_centers)
        sample_x = torch.linspace(0.5, max(float(image_width) - 0.5, 0.5), samples_per_view, device=device)
        sample_y = torch.linspace(0.5, max(float(image_height) - 0.5, 0.5), samples_per_view, device=device)
        a, b, c = lines[..., 0], lines[..., 1], lines[..., 2]
        use_x = b.abs() >= a.abs()
        safe_a = torch.where(a.abs() < 1e-6, torch.where(a >= 0, a.new_tensor(1e-6), a.new_tensor(-1e-6)), a)
        safe_b = torch.where(b.abs() < 1e-6, torch.where(b >= 0, b.new_tensor(1e-6), b.new_tensor(-1e-6)), b)
        y_from_x = -(a.unsqueeze(-1) * sample_x.view(1, 1, 1, -1) + c.unsqueeze(-1)) / safe_b.unsqueeze(-1)
        x_from_y = -(b.unsqueeze(-1) * sample_y.view(1, 1, 1, -1) + c.unsqueeze(-1)) / safe_a.unsqueeze(-1)
        xs = torch.where(use_x.unsqueeze(-1), sample_x.view(1, 1, 1, -1).expand_as(y_from_x), x_from_y)
        ys = torch.where(use_x.unsqueeze(-1), y_from_x, sample_y.view(1, 1, 1, -1).expand_as(x_from_y))
        inside = (
            torch.isfinite(xs)
            & torch.isfinite(ys)
            & (xs >= 0.0)
            & (xs <= float(image_width) - 1.0)
            & (ys >= 0.0)
            & (ys <= float(image_height) - 1.0)
            & (source_view_ids.view(1, 1, -1, 1) != query_view_ids[..., None, None])
            & lines[..., :2].square().sum(dim=-1).sqrt().unsqueeze(-1).gt(1e-6)
        )
        cols = torch.floor(xs * float(patch_grid_width) / float(image_width)).long().clamp(min=0, max=max(int(patch_grid_width) - 1, 0))
        rows = torch.floor(ys * float(patch_grid_height) / float(image_height)).long().clamp(min=0, max=max(int(patch_grid_height) - 1, 0))
        patch_local = rows * int(patch_grid_width) + cols
        anchors = source_view_ids.view(1, 1, -1, 1) * int(tokens_per_view) + int(patch_start_idx) + patch_local
        return anchors.reshape(bsz, tgt_len, -1), inside.reshape(bsz, tgt_len, -1)

    def _build_geometry_anchor_indices(
        self,
        *,
        query_token_indices: Tensor,
        state: dotdict,
        num_views: int,
        tokens_per_view: int,
        patch_start_idx: int,
        patch_grid_height: int,
        patch_grid_width: int,
        image_height: int,
        image_width: int,
        extrinsics: Tensor,
        intrinsics: Tensor,
        patch_centers: Optional[Tensor] = None,
        fundamental: Optional[Tensor] = None,
        intrinsics_inv: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        mode = self._geometry_support_mode(state)
        depth_indices = depth_valid = None
        if mode in ("auto", "depth_reprojection") and self._geometry_support_depth_available(state):
            depth_indices, depth_valid = self._build_depth_anchor_indices(
                query_token_indices=query_token_indices,
                state=state,
                num_views=num_views,
                tokens_per_view=tokens_per_view,
                patch_start_idx=patch_start_idx,
                patch_grid_height=patch_grid_height,
                patch_grid_width=patch_grid_width,
                image_height=image_height,
                image_width=image_width,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                patch_centers=patch_centers,
                intrinsics_inv=intrinsics_inv,
            )
            if mode == "depth_reprojection":
                return depth_indices, depth_valid
        if mode == "depth_reprojection":
            empty_idx = torch.empty((extrinsics.shape[0], int(query_token_indices.numel()), 0), device=query_token_indices.device, dtype=torch.long)
            return empty_idx, torch.zeros_like(empty_idx, dtype=torch.bool)

        epi_indices, epi_valid = self._build_epipolar_anchor_indices(
            query_token_indices=query_token_indices,
            state=state,
            num_views=num_views,
            tokens_per_view=tokens_per_view,
            patch_start_idx=patch_start_idx,
            patch_grid_height=patch_grid_height,
            patch_grid_width=patch_grid_width,
            image_height=image_height,
            image_width=image_width,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            patch_centers=patch_centers,
            fundamental=fundamental,
        )
        if depth_indices is None or depth_valid is None:
            return epi_indices, epi_valid
        no_depth_anchor = ~depth_valid.any(dim=-1, keepdim=True)
        epi_valid = epi_valid & no_depth_anchor
        return torch.cat([depth_indices, epi_indices], dim=-1), torch.cat([depth_valid, epi_valid], dim=-1)

    def _score_projected_gathered(
        self,
        *,
        q_subset: Tensor,
        k_proj: Tensor,
        w_proj: Tensor,
        source_indices: Tensor,
        query_indices: Tensor,
        tokens_per_view: int,
        view_bias_data: Optional[dict],
    ) -> Tensor:
        bsz, tgt_len, _, _ = q_subset.shape
        if source_indices.shape[-1] <= 0:
            return q_subset.new_zeros((bsz, tgt_len, 0))
        source_indices = source_indices.to(device=q_subset.device, dtype=torch.long)
        source_clamped = source_indices.clamp(min=0, max=max(int(k_proj.shape[1]) - 1, 0))
        batch_ids = torch.arange(bsz, device=q_subset.device)[:, None, None].expand_as(source_clamped)
        k_sel = k_proj[batch_ids, source_clamped]
        w_sel = w_proj[batch_ids, source_clamped]
        scale = q_subset.new_tensor(getattr(self.indexer, "scale", self.indexer_cfg.head_dim ** -0.5))
        per_head = (q_subset.unsqueeze(2) * k_sel).sum(dim=-1) * scale
        scores = (torch.relu(per_head) * w_sel).sum(dim=-1, dtype=q_subset.dtype)
        if view_bias_data is not None:
            q_view_ids = torch.div(query_indices.to(device=q_subset.device, dtype=torch.long), int(tokens_per_view), rounding_mode="floor")
            q_view_ids = q_view_ids.view(1, -1, 1).expand_as(source_clamped)
            s_view_ids = torch.div(source_clamped, int(tokens_per_view), rounding_mode="floor")
            view_bias = view_bias_data["view_bias"]
            batch_ids = torch.arange(bsz, device=q_subset.device)[:, None, None].expand_as(source_clamped)
            scores = scores + view_bias[batch_ids, q_view_ids, s_view_ids].to(dtype=scores.dtype)
        return scores

    @staticmethod
    def _projected_indexer_graph_zero(projected_indexer: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        q_proj, k_proj, w_proj = projected_indexer
        zero = q_proj.reshape(-1)[:1].to(torch.float32).sum() * 0.0
        zero = zero + k_proj.reshape(-1)[:1].to(torch.float32).sum() * 0.0
        zero = zero + w_proj.reshape(-1)[:1].to(torch.float32).sum() * 0.0
        return zero

    def _compute_topk_anchor_selector_loss(
        self,
        *,
        projected_indexer: Tuple[Tensor, Tensor, Tensor],
        state: dotdict,
        index_mask: Optional[Tensor],
        view_bias_data: Optional[dict],
        topk_indices: Optional[Tensor],
        topk_scores: Optional[Tensor],
    ) -> Optional[Tensor]:
        if topk_indices is None:
            return None
        q_proj, k_proj, w_proj = projected_indexer
        device = q_proj.device
        tokens_per_view = int(state.get("tokens_per_view", 0) or 0)
        num_views = int(state.get("num_views", 0) or 0)
        patch_start_idx = int(state.get("patch_start_idx", 0) or 0)
        patch_grid_height = int(state.get("patch_grid_height", 0) or 0)
        patch_grid_width = int(state.get("patch_grid_width", 0) or 0)
        image_height = int(state.get("image_height", 0) or 0)
        image_width = int(state.get("image_width", 0) or 0)
        if num_views < 2 or tokens_per_view <= patch_start_idx or patch_grid_height <= 0 or patch_grid_width <= 0:
            return None

        loss_downsample_factor = max(int(state.get("epipolar_selector_loss_downsample_factor", 1) or 0), 1)
        loss_patch_indices = build_downsampled_patch_indices(
            patch_grid_height=patch_grid_height,
            patch_grid_width=patch_grid_width,
            factor=loss_downsample_factor,
            device=device,
        )
        image_token_indices = build_image_token_indices(
            num_views=num_views,
            tokens_per_view=tokens_per_view,
            patch_start_idx=patch_start_idx,
            device=device,
            patch_indices=loss_patch_indices,
        )
        if image_token_indices.numel() == 0:
            return None
        query_stride = max(int(state.get("epipolar_selector_query_stride", 8) or 0), 1)
        query_token_indices = image_token_indices[::query_stride]
        if query_token_indices.numel() == 0:
            query_token_indices = image_token_indices[:1]

        band_px = float(state.get("epipolar_selector_band_px", state.get("epipolar_selector_sigma_px", 12.0)) or 12.0)
        query_chunk_size = max(int(state.get("epipolar_selector_query_chunk_size", 256) or 0), 1)
        extrinsics = state.extrinsics.to(device=device, dtype=torch.float32)
        intrinsics = state.intrinsics.to(device=device, dtype=torch.float32)
        patch_centers = build_patch_token_centers(
            image_height=image_height,
            image_width=image_width,
            patch_grid_height=patch_grid_height,
            patch_grid_width=patch_grid_width,
            device=device,
            dtype=torch.float32,
        )
        fundamental = compute_query_to_source_fundamental_matrices(extrinsics=extrinsics, intrinsics=intrinsics)
        support_mode = self._geometry_support_mode(state)
        needs_depth_projection = support_mode in ("auto", "depth_reprojection") and self._geometry_support_depth_available(state)
        intrinsics_inv = torch.linalg.inv(intrinsics) if needs_depth_projection else None
        total_loss_sum = q_proj.new_tensor(0.0, dtype=torch.float32)
        total_valid_queries = q_proj.new_tensor(0.0, dtype=torch.float32)
        total_topk_candidate_queries = q_proj.new_tensor(0.0, dtype=torch.float32)
        total_topk_positive_queries = q_proj.new_tensor(0.0, dtype=torch.float32)

        topk_indices = topk_indices.to(device=device, dtype=torch.long)
        if topk_scores is not None:
            topk_scores = topk_scores.to(device=device)

        for q_start in range(0, int(query_token_indices.numel()), int(query_chunk_size)):
            chunk_query_indices = query_token_indices[q_start : q_start + int(query_chunk_size)]
            q_subset = q_proj[:, chunk_query_indices]
            chunk_topk_indices = topk_indices[:, chunk_query_indices]
            if topk_scores is None:
                chunk_topk_scores = self._score_projected_gathered(
                    q_subset=q_subset,
                    k_proj=k_proj,
                    w_proj=w_proj,
                    source_indices=chunk_topk_indices,
                    query_indices=chunk_query_indices,
                    tokens_per_view=tokens_per_view,
                    view_bias_data=view_bias_data,
                )
            else:
                chunk_topk_scores = topk_scores[:, chunk_query_indices]
            topk_support, topk_valid = self._build_selector_support_for_source_tokens(
                query_token_indices=chunk_query_indices,
                source_token_indices=chunk_topk_indices,
                state=state,
                num_views=num_views,
                tokens_per_view=tokens_per_view,
                patch_start_idx=patch_start_idx,
                patch_grid_height=patch_grid_height,
                patch_grid_width=patch_grid_width,
                image_height=image_height,
                image_width=image_width,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                index_mask=index_mask,
                band_px=band_px,
                patch_centers=patch_centers,
                fundamental=fundamental,
                intrinsics_inv=intrinsics_inv,
            )
            candidate_scores = chunk_topk_scores
            candidate_valid = topk_valid
            positive_mask = topk_support
            total_topk_candidate_queries = total_topk_candidate_queries + topk_valid.any(dim=-1).to(torch.float32).sum()
            total_topk_positive_queries = total_topk_positive_queries + topk_support.any(dim=-1).to(torch.float32).sum()

            if self._epipolar_selector_loss_mode(state) == "topk_anchor":
                anchor_indices, anchor_valid = self._build_geometry_anchor_indices(
                    query_token_indices=chunk_query_indices,
                    state=state,
                    num_views=num_views,
                    tokens_per_view=tokens_per_view,
                    patch_start_idx=patch_start_idx,
                    patch_grid_height=patch_grid_height,
                    patch_grid_width=patch_grid_width,
                    image_height=image_height,
                    image_width=image_width,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    patch_centers=patch_centers,
                    fundamental=fundamental,
                    intrinsics_inv=intrinsics_inv,
                )
                if anchor_indices.shape[-1] > 0:
                    duplicate_anchor = (anchor_indices.unsqueeze(-2) == chunk_topk_indices.unsqueeze(-1)).any(dim=-2)
                    anchor_valid = anchor_valid & (~duplicate_anchor)
                    anchor_index_mask = self._candidate_index_mask(
                        index_mask=index_mask,
                        query_token_indices=chunk_query_indices,
                        source_token_indices=anchor_indices,
                        fill_threshold=self._mask_fill_value(index_mask.dtype) * 0.5 if index_mask is not None and index_mask.dtype.is_floating_point else 0.0,
                    )
                    if anchor_index_mask is not None:
                        anchor_valid = anchor_valid & anchor_index_mask
                    anchor_scores = self._score_projected_gathered(
                        q_subset=q_subset,
                        k_proj=k_proj,
                        w_proj=w_proj,
                        source_indices=anchor_indices,
                        query_indices=chunk_query_indices,
                        tokens_per_view=tokens_per_view,
                        view_bias_data=view_bias_data,
                    ).to(dtype=candidate_scores.dtype)
                    candidate_scores = torch.cat([candidate_scores, anchor_scores], dim=-1)
                    candidate_valid = torch.cat([candidate_valid, anchor_valid], dim=-1)
                    positive_mask = torch.cat([positive_mask, anchor_valid], dim=-1)

            finite_scores = torch.isfinite(candidate_scores)
            candidate_valid = candidate_valid & finite_scores
            positive_mask = positive_mask & candidate_valid
            valid_queries = candidate_valid.any(dim=-1) & positive_mask.any(dim=-1)
            fill_value = torch.finfo(candidate_scores.dtype).min
            masked_scores = candidate_scores.masked_fill(~candidate_valid, fill_value)
            positive_scores = masked_scores.masked_fill(~positive_mask, fill_value)
            loss_per_query = (
                torch.logsumexp(masked_scores, dim=-1).to(torch.float32)
                - torch.logsumexp(positive_scores, dim=-1).to(torch.float32)
            )
            loss_per_query = torch.where(valid_queries, loss_per_query, torch.zeros_like(loss_per_query))
            valid_queries_float = valid_queries.to(torch.float32)
            total_loss_sum = total_loss_sum + (loss_per_query * valid_queries_float).sum()
            total_valid_queries = total_valid_queries + valid_queries_float.sum()

        zero = self._projected_indexer_graph_zero(projected_indexer)
        loss = total_loss_sum / total_valid_queries.clamp_min(1.0)
        valid_query_ratio = total_topk_positive_queries / total_topk_candidate_queries.clamp_min(1.0)
        self.last_epipolar_selector_stats = dotdict(
            epipolar_selector_valid_queries=total_topk_positive_queries.detach(),
            epipolar_selector_candidate_queries=total_topk_candidate_queries.detach(),
            epipolar_selector_valid_query_ratio=valid_query_ratio.detach(),
        )
        return torch.where((total_valid_queries > 0) & torch.isfinite(loss), loss, zero)

    def _compute_epipolar_selector_band_loss(
        self,
        projected_indexer: Optional[Tuple[Tensor, Tensor, Tensor]],
        indexer_input: Tensor,
        pos: Optional[Tensor],
        state: dotdict,
        index_mask: Optional[Tensor],
        view_bias_data: Optional[dict],
        topk_indices: Optional[Tensor] = None,
        topk_scores: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if not self._epipolar_selector_band_enabled(state):
            return None
        if self.indexer is None:
            return None
        if projected_indexer is None:
            projected_indexer = self.indexer.project(indexer_input, pos=pos)
        loss_mode = self._epipolar_selector_loss_mode(state)
        if loss_mode in ("topk", "topk_anchor"):
            topk_loss = self._compute_topk_anchor_selector_loss(
                projected_indexer=projected_indexer,
                state=state,
                index_mask=index_mask,
                view_bias_data=view_bias_data,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
            )
            if topk_loss is not None:
                return topk_loss

        tokens_per_view = int(state.get("tokens_per_view", 0) or 0)
        num_views = int(state.get("num_views", 0) or 0)
        patch_start_idx = int(state.get("patch_start_idx", 0) or 0)
        patch_grid_height = int(state.get("patch_grid_height", 0) or 0)
        patch_grid_width = int(state.get("patch_grid_width", 0) or 0)
        image_height = int(state.get("image_height", 0) or 0)
        image_width = int(state.get("image_width", 0) or 0)
        if num_views < 2 or tokens_per_view <= patch_start_idx or patch_grid_height <= 0 or patch_grid_width <= 0:
            return None

        q_proj, k_proj, w_proj = projected_indexer
        device = q_proj.device
        loss_downsample_factor = max(int(state.get("epipolar_selector_loss_downsample_factor", 1) or 0), 1)
        loss_patch_indices = build_downsampled_patch_indices(
            patch_grid_height=patch_grid_height,
            patch_grid_width=patch_grid_width,
            factor=loss_downsample_factor,
            device=device,
        )
        image_token_indices = build_image_token_indices(
            num_views=num_views,
            tokens_per_view=tokens_per_view,
            patch_start_idx=patch_start_idx,
            device=device,
            patch_indices=loss_patch_indices,
        )
        if image_token_indices.numel() == 0:
            return None

        query_stride = max(int(state.get("epipolar_selector_query_stride", 8) or 0), 1)
        query_token_indices = image_token_indices[::query_stride]
        if query_token_indices.numel() == 0:
            query_token_indices = image_token_indices[:1]

        band_px = float(
            state.get("epipolar_selector_band_px", state.get("epipolar_selector_sigma_px", 12.0)) or 12.0
        )
        query_chunk_size = max(int(state.get("epipolar_selector_query_chunk_size", 256) or 0), 1)
        source_view_chunk_size = max(int(state.get("epipolar_selector_source_view_chunk_size", 12) or 0), 1)
        use_block_checkpoint = bool(state.get("epipolar_selector_checkpoint_blocks", True))
        extrinsics = state.extrinsics.to(device=device, dtype=torch.float32)
        intrinsics = state.intrinsics.to(device=device, dtype=torch.float32)
        all_source_view_ids = torch.arange(num_views, device=device, dtype=torch.long)
        query_chunks = []
        for q_start in range(0, int(query_token_indices.numel()), int(query_chunk_size)):
            chunk_query_indices = query_token_indices[q_start : q_start + int(query_chunk_size)]
            q_subset = q_proj[:, chunk_query_indices, :, :]
            query_chunks.append(
                {
                    "indices": chunk_query_indices,
                    "q": q_subset,
                    "total_lse": torch.full(
                        (q_proj.shape[0], int(chunk_query_indices.numel())),
                        float("-inf"),
                        device=device,
                        dtype=torch.float32,
                    ),
                    "band_lse": torch.full(
                        (q_proj.shape[0], int(chunk_query_indices.numel())),
                        float("-inf"),
                        device=device,
                        dtype=torch.float32,
                    ),
                    "has_valid_source": torch.zeros(
                        (q_proj.shape[0], int(chunk_query_indices.numel())),
                        device=device,
                        dtype=torch.bool,
                    ),
                    "has_band_source": torch.zeros(
                        (q_proj.shape[0], int(chunk_query_indices.numel())),
                        device=device,
                        dtype=torch.bool,
                    ),
                }
            )

        any_source_chunk = False
        for s_start in range(0, num_views, int(source_view_chunk_size)):
            chunk_source_view_ids = all_source_view_ids[s_start : s_start + int(source_view_chunk_size)]
            chunk_source_indices = build_image_token_indices(
                num_views=num_views,
                tokens_per_view=tokens_per_view,
                patch_start_idx=patch_start_idx,
                device=device,
                view_indices=chunk_source_view_ids,
                patch_indices=loss_patch_indices,
            )
            if chunk_source_indices.numel() <= 0:
                continue
            any_source_chunk = True
            k_image = k_proj[:, chunk_source_indices, :, :]
            w_image = w_proj[:, chunk_source_indices, :]

            for chunk in query_chunks:
                chunk_query_indices = chunk["indices"]
                q_subset = chunk["q"]
                epipolar_view_bias_data = self._gather_view_bias_data(
                    view_bias_data=view_bias_data,
                    query_indices=chunk_query_indices,
                    source_indices=chunk_source_indices,
                    tokens_per_view=tokens_per_view,
                    bsz=q_proj.shape[0],
                )
                if index_mask is not None:
                    image_mask = index_mask[:, chunk_query_indices][:, :, chunk_source_indices]
                    valid_source_mask = torch.isfinite(image_mask) if image_mask.dtype.is_floating_point else image_mask.bool()
                    if image_mask.dtype.is_floating_point:
                        valid_source_mask = image_mask > (self._mask_fill_value(image_mask.dtype) * 0.5)
                else:
                    image_mask = None
                    valid_source_mask = None

                def build_block_support(
                    block_query_indices: Tensor = chunk_query_indices,
                    block_source_view_ids: Tensor = chunk_source_view_ids,
                    block_source_mask: Optional[Tensor] = valid_source_mask,
                ):
                    return self._build_selector_support_mask(
                        query_token_indices=block_query_indices,
                        state=state,
                        num_views=num_views,
                        tokens_per_view=tokens_per_view,
                        patch_start_idx=patch_start_idx,
                        patch_grid_height=patch_grid_height,
                        patch_grid_width=patch_grid_width,
                        image_height=image_height,
                        image_width=image_width,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                        source_mask=block_source_mask,
                        source_view_indices=block_source_view_ids,
                        source_patch_indices=loss_patch_indices,
                        band_px=band_px,
                    )

                can_checkpoint = (
                    use_block_checkpoint
                    and epipolar_view_bias_data is None
                    and torch.is_grad_enabled()
                    and (q_subset.requires_grad or k_image.requires_grad or w_image.requires_grad)
                )
                recompute_support_in_checkpoint = can_checkpoint and image_mask is None

                if recompute_support_in_checkpoint:
                    with torch.no_grad():
                        band_mask_any, valid_band_mask_any, _ = build_block_support()
                        block_valid_any = valid_band_mask_any.any(dim=-1)
                        block_band_any = band_mask_any.any(dim=-1)
                    del band_mask_any, valid_band_mask_any
                else:
                    band_mask, valid_band_mask, _ = build_block_support()
                    block_valid_any = valid_band_mask.any(dim=-1)
                    block_band_any = band_mask.any(dim=-1)

                def compute_block_terms(
                    q_tensor: Tensor,
                    k_tensor: Tensor,
                    w_tensor: Tensor,
                    image_mask_tensor: Optional[Tensor] = image_mask,
                    recompute_support: bool = recompute_support_in_checkpoint,
                    support_builder=build_block_support,
                    block_band_mask: Optional[Tensor] = None if recompute_support_in_checkpoint else band_mask,
                    block_valid_mask: Optional[Tensor] = None if recompute_support_in_checkpoint else valid_band_mask,
                    block_view_bias_data: Optional[dict] = epipolar_view_bias_data,
                ):
                    if recompute_support:
                        band_mask_tensor, valid_band_mask_tensor, _ = support_builder()
                    else:
                        band_mask_tensor = block_band_mask
                        valid_band_mask_tensor = block_valid_mask
                    student_scores = self.indexer.select_topk_projected(
                        q_tensor,
                        k_tensor,
                        w_tensor,
                        mask=None,
                        topk=None,
                        view_bias_data=block_view_bias_data,
                    )
                    if image_mask_tensor is not None:
                        student_scores = student_scores + image_mask_tensor.to(dtype=student_scores.dtype)
                    block_total_lse, block_band_lse, _ = compute_epipolar_band_logsumexp_terms(
                        student_scores=student_scores,
                        band_mask=band_mask_tensor,
                        valid_mask=valid_band_mask_tensor,
                    )
                    return block_total_lse, block_band_lse

                if can_checkpoint:
                    block_total_lse, block_band_lse = activation_checkpoint(
                        compute_block_terms,
                        q_subset,
                        k_image,
                        w_image,
                        use_reentrant=False,
                    )
                else:
                    block_total_lse, block_band_lse = compute_block_terms(
                        q_subset,
                        k_image,
                        w_image,
                    )
                chunk["total_lse"] = torch.logaddexp(chunk["total_lse"], block_total_lse)
                chunk["band_lse"] = torch.logaddexp(chunk["band_lse"], block_band_lse)
                chunk["has_valid_source"] = chunk["has_valid_source"] | block_valid_any
                chunk["has_band_source"] = chunk["has_band_source"] | block_band_any

        if not any_source_chunk:
            return None

        total_loss_sum = q_proj.new_tensor(0.0, dtype=torch.float32)
        total_valid_queries = q_proj.new_tensor(0.0, dtype=torch.float32)
        for chunk in query_chunks:
            valid_queries = chunk["has_valid_source"] & chunk["has_band_source"]
            loss_per_query = chunk["total_lse"] - chunk["band_lse"]
            loss_per_query = torch.where(valid_queries, loss_per_query, torch.zeros_like(loss_per_query))
            valid_queries_float = valid_queries.to(torch.float32)
            total_loss_sum = total_loss_sum + (loss_per_query * valid_queries_float).sum()
            total_valid_queries = total_valid_queries + valid_queries_float.sum()

        zero = self._projected_indexer_graph_zero(projected_indexer)
        loss = total_loss_sum / total_valid_queries.clamp_min(1.0)
        return torch.where((total_valid_queries > 0) & torch.isfinite(loss), loss, zero)

    @staticmethod
    def _normalize_keep_mask(mask: Optional[Tensor], bsz: int) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.size(0) == 1 and bsz > 1:
            mask = mask.expand(bsz, -1, -1, -1)
        return mask

    @staticmethod
    def _special_token_span(state: dotdict, patch_start_idx: int) -> int:
        if patch_start_idx <= 0:
            return 0
        if bool(state.get("force_keep_register_tokens", False)):
            return patch_start_idx
        camera_tokens_per_view = int(state.get("camera_tokens_per_view", 1) or 1)
        camera_tokens_per_view = max(camera_tokens_per_view, 1)
        return min(camera_tokens_per_view, patch_start_idx)

    @staticmethod
    def _build_force_keep_mask(state: dotdict, tgt_len: int, device: torch.device) -> Optional[Tensor]:
        keep_special = bool(state.get("force_keep_special_tokens", False))
        keep_self_view = bool(state.get("force_keep_self_view_tokens", False))
        if not (keep_special or keep_self_view):
            return None

        tokens_per_view = int(state.get("tokens_per_view", 0) or 0)
        num_views = int(state.get("num_views", 0) or 0)
        patch_start_idx = int(state.get("patch_start_idx", 0) or 0)
        if tokens_per_view <= 0 or num_views <= 0:
            return None
        if tokens_per_view * num_views != tgt_len:
            return None

        idx = torch.arange(tgt_len, device=device)
        keep_mask = None
        if keep_self_view:
            view_ids = idx // tokens_per_view
            keep_mask = view_ids[:, None].eq(view_ids[None, :])
        forced_span = DSAAttention._special_token_span(state, patch_start_idx)
        if keep_special and forced_span > 0:
            special = (idx % tokens_per_view) < forced_span
            special_mask = special.unsqueeze(0).expand(tgt_len, tgt_len)
            keep_mask = special_mask if keep_mask is None else (keep_mask | special_mask)
        return keep_mask

    @staticmethod
    def _build_special_token_indices(state: dotdict, tgt_len: int, device: torch.device) -> Optional[Tensor]:
        tokens_per_view = int(state.get("tokens_per_view", 0) or 0)
        num_views = int(state.get("num_views", 0) or 0)
        patch_start_idx = int(state.get("patch_start_idx", 0) or 0)
        if tokens_per_view <= 0 or num_views <= 0 or patch_start_idx <= 0:
            return None
        if tokens_per_view * num_views != tgt_len:
            return None
        forced_span = DSAAttention._special_token_span(state, patch_start_idx)
        if forced_span <= 0:
            return None
        idx = torch.arange(tgt_len, device=device, dtype=torch.long)
        special = (idx % tokens_per_view) < forced_span
        if not torch.any(special):
            return None
        return idx[special]

    @staticmethod
    def _infer_patch_grid(state: dotdict, image_tokens_per_view: int) -> Tuple[int, int]:
        grid_h = int(state.get("patch_grid_height", 0) or 0)
        grid_w = int(state.get("patch_grid_width", 0) or 0)
        if grid_h > 0 and grid_w > 0 and grid_h * grid_w == image_tokens_per_view:
            return grid_h, grid_w
        side = math.isqrt(max(int(image_tokens_per_view), 0))
        if side > 0 and side * side == image_tokens_per_view:
            return side, side
        return 0, 0

    def _should_use_source_downsample_selector(
        self,
        state: dotdict,
        q: Tensor,
        k: Tensor,
        mask: Optional[Tensor],
        return_scores: bool,
    ) -> bool:
        if self.indexer is None:
            return False
        if not bool(state.get("source_downsample_enabled", False)):
            return False
        if return_scores:
            return False
        if mask is not None:
            return False
        if self.training or torch.is_grad_enabled() or bool(state.get("compute_loss", False)):
            return False
        if q.dim() != 4 or k.dim() != 4:
            return False
        if q.shape[0] != k.shape[0]:
            return False
        factor = int(state.get("source_downsample_factor", 1) or 1)
        return factor > 1

    def _get_source_downsample_layout(self, state: dotdict, src_len: int, device: torch.device) -> Optional[dotdict]:
        tokens_per_view = int(state.get("tokens_per_view", 0) or 0)
        num_views = int(state.get("num_views", 0) or 0)
        patch_start_idx = int(state.get("patch_start_idx", 0) or 0)
        factor = max(int(state.get("source_downsample_factor", 1) or 1), 1)
        if tokens_per_view <= 0 or num_views <= 0 or tokens_per_view * num_views != int(src_len):
            return None
        if patch_start_idx < 0 or patch_start_idx >= tokens_per_view:
            return None
        image_tokens_per_view = tokens_per_view - patch_start_idx
        if image_tokens_per_view <= 0:
            return None
        grid_h, grid_w = self._infer_patch_grid(state, image_tokens_per_view)
        if grid_h <= 0 or grid_w <= 0:
            return None
        use_representative_points = factor > 2 and bool(
            state.get(
                "source_downsample_representative_points",
                self.indexer_cfg.get("source_downsample_representative_points", False),
            )
        )

        key = (
            device.type,
            device.index,
            tokens_per_view,
            num_views,
            patch_start_idx,
            grid_h,
            grid_w,
            factor,
            use_representative_points,
        )
        cached = self._source_downsample_layout_cache.get(key)
        if cached is not None:
            return cached

        pooled_h = (grid_h // factor) * factor
        pooled_w = (grid_w // factor) * factor
        coarse_h = pooled_h // factor if pooled_h > 0 else 0
        coarse_w = pooled_w // factor if pooled_w > 0 else 0
        if not use_representative_points:
            representative_offsets = [(yy, xx) for yy in range(factor) for xx in range(factor)]
            expand_round_robin = True
        else:
            quarter = max(int(factor // 4), 0)
            low = min(max(quarter, 0), factor - 1)
            high = max(min(factor - 1 - quarter, factor - 1), 0)
            representative_offsets = []
            for item in ((low, low), (low, high), (high, low), (high, high)):
                if item not in representative_offsets:
                    representative_offsets.append(item)
            expand_round_robin = False
        max_block_tokens = len(representative_offsets)
        special_count = num_views * patch_start_idx
        edge_per_view = image_tokens_per_view - pooled_h * pooled_w
        edge_count = num_views * edge_per_view
        coarse_count = num_views * coarse_h * coarse_w
        special_indices = torch.empty((special_count,), device=device, dtype=torch.long)
        if special_count > 0:
            view_offsets = torch.arange(num_views, device=device, dtype=torch.long) * tokens_per_view
            local_special = torch.arange(patch_start_idx, device=device, dtype=torch.long)
            special_indices = (view_offsets[:, None] + local_special[None, :]).reshape(-1)

        source_len = special_count + edge_count + coarse_count
        source_to_fine = torch.full((source_len, max_block_tokens), -1, device=device, dtype=torch.long)
        source_valid = torch.zeros((source_len, max_block_tokens), device=device, dtype=torch.bool)
        edge_full_indices = torch.empty((edge_count,), device=device, dtype=torch.long)
        source_idx = 0
        if special_count > 0:
            source_to_fine[:special_count, 0] = special_indices
            source_valid[:special_count, 0] = True
            source_idx = special_count

        edge_write = 0
        for view_idx in range(num_views):
            view_base = view_idx * tokens_per_view + patch_start_idx
            for y in range(grid_h):
                for x in range(grid_w):
                    if y < pooled_h and x < pooled_w:
                        continue
                    fine_idx = view_base + y * grid_w + x
                    edge_full_indices[edge_write] = fine_idx
                    source_to_fine[source_idx, 0] = fine_idx
                    source_valid[source_idx, 0] = True
                    edge_write += 1
                    source_idx += 1

        for view_idx in range(num_views):
            view_base = view_idx * tokens_per_view + patch_start_idx
            for coarse_y in range(coarse_h):
                y_start = coarse_y * factor
                for coarse_x in range(coarse_w):
                    x_start = coarse_x * factor
                    fine_tokens = []
                    for yy_off, xx_off in representative_offsets:
                        yy = y_start + yy_off
                        xx = x_start + xx_off
                        if yy >= grid_h or xx >= grid_w:
                            continue
                        fine_tokens.append(view_base + yy * grid_w + xx)
                    block_len = len(fine_tokens)
                    if block_len > 0:
                        source_to_fine[source_idx, :block_len] = torch.tensor(
                            fine_tokens,
                            device=device,
                            dtype=torch.long,
                        )
                        source_valid[source_idx, :block_len] = True
                    source_idx += 1

        if source_idx != source_len or edge_write != edge_count:
            raise RuntimeError("Invalid source-downsample layout construction.")

        row_token_counts = source_valid.sum(dim=-1, dtype=torch.long)
        fine_to_source_row = torch.full((src_len,), -1, device=device, dtype=torch.long)
        fine_to_source_slot = torch.full((src_len,), -1, device=device, dtype=torch.long)
        valid_flat = source_valid.reshape(-1)
        if bool(valid_flat.any()):
            row_ids = torch.arange(source_len, device=device, dtype=torch.long).view(-1, 1).expand(source_len, max_block_tokens)
            slot_ids = torch.arange(max_block_tokens, device=device, dtype=torch.long).view(1, -1).expand(source_len, max_block_tokens)
            fine_to_source_row[source_to_fine.reshape(-1)[valid_flat]] = row_ids.reshape(-1)[valid_flat]
            fine_to_source_slot[source_to_fine.reshape(-1)[valid_flat]] = slot_ids.reshape(-1)[valid_flat]
        if (not use_representative_points) and bool((fine_to_source_row < 0).any()):
            raise RuntimeError("Invalid source-downsample fine-to-row mapping construction.")

        mapped_token_count = special_count + edge_count + coarse_count * max_block_tokens
        non_special_source_len = source_len - special_count
        non_special_mapped_token_count = mapped_token_count - special_count
        layout = dotdict(
            tokens_per_view=tokens_per_view,
            num_views=num_views,
            patch_start_idx=patch_start_idx,
            image_tokens_per_view=image_tokens_per_view,
            grid_h=grid_h,
            grid_w=grid_w,
            factor=factor,
            pooled_h=pooled_h,
            pooled_w=pooled_w,
            coarse_h=coarse_h,
            coarse_w=coarse_w,
            special_count=special_count,
            edge_count=edge_count,
            edge_per_view=edge_per_view,
            coarse_count=coarse_count,
            coarse_source_offset=special_count + edge_count,
            source_len=source_len,
            max_block_tokens=max_block_tokens,
            representative_offsets=representative_offsets,
            use_representative_points=use_representative_points,
            expand_round_robin=expand_round_robin,
            avg_expand_tokens=float(mapped_token_count) / float(source_len),
            mapped_token_count=mapped_token_count,
            non_special_source_len=non_special_source_len,
            non_special_avg_expand_tokens=(
                float(non_special_mapped_token_count) / float(non_special_source_len)
                if non_special_source_len > 0
                else 0.0
            ),
            edge_full_indices=edge_full_indices,
            source_to_fine=source_to_fine,
            source_valid=source_valid,
            row_token_counts=row_token_counts,
            fine_to_source_row=fine_to_source_row,
            fine_to_source_slot=fine_to_source_slot,
        )
        if factor >= 4 and max_block_tokens > 1 and coarse_count > 0:
            group_ids = self._get_source_downsample_subcell_group_ids(layout)
            subcell_count = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 1
            tokens_per_subcell = 0
            for group_id in range(subcell_count):
                tokens_per_subcell = max(tokens_per_subcell, int(group_ids.eq(group_id).sum().item()))
            coarse_rows = source_to_fine[int(layout.coarse_source_offset): int(layout.coarse_source_offset) + coarse_count]
            coarse_valid_rows = source_valid[int(layout.coarse_source_offset): int(layout.coarse_source_offset) + coarse_count]
            coarse_subcell_to_fine = torch.full(
                (coarse_count, subcell_count, max(tokens_per_subcell, 1)),
                -1,
                device=device,
                dtype=torch.long,
            )
            coarse_subcell_valid = torch.zeros_like(coarse_subcell_to_fine, dtype=torch.bool)
            for group_id in range(subcell_count):
                group_pos = torch.nonzero(group_ids.eq(group_id), as_tuple=False).flatten()
                if group_pos.numel() <= 0:
                    continue
                coarse_subcell_to_fine[:, group_id, :group_pos.numel()] = coarse_rows.index_select(dim=1, index=group_pos)
                coarse_subcell_valid[:, group_id, :group_pos.numel()] = coarse_valid_rows.index_select(dim=1, index=group_pos)
            layout.coarse_subcell_to_fine = coarse_subcell_to_fine
            layout.coarse_subcell_valid = coarse_subcell_valid
        self._source_downsample_layout_cache[key] = layout
        return layout

    def _resolve_source_downsample_query_chunk(self, state: dotdict, q_len: int) -> int:
        query_chunk = int(
            state.get(
                "source_downsample_query_chunk",
                self.indexer_cfg.get("source_downsample_query_chunk", 4096),
            ) or 0
        )
        if query_chunk <= 0:
            query_chunk = self._env_int("VGGT_DSA_SOURCE_DOWNSAMPLE_QCHUNK", 4096)
        query_chunk = max(int(query_chunk), 1)
        return min(query_chunk, int(q_len))

    def _resolve_source_downsample_strategy(self, state: dotdict, layout: dotdict) -> str:
        strategy = str(
            state.get(
                "source_downsample_strategy",
                self.indexer_cfg.get("source_downsample_strategy", "legacy"),
            )
            or "legacy"
        ).strip().lower()
        factor = int(layout.factor)
        if strategy == "qk_sym_x2_broadcast":
            return "qk_sym_x2_broadcast" if factor == 2 else "legacy"
        if factor < 4 and strategy != "legacy":
            return "legacy"
        if strategy not in {"legacy", "exact_rerank", "subcell_top1", "subcell_static_pack4", "qk_sym_x2_broadcast"}:
            return "legacy"
        return strategy

    def _get_source_downsample_effective_avg_expand(self, layout: dotdict, strategy: str) -> float:
        if strategy in {"subcell_top1", "subcell_static_pack4"}:
            subcell_size = 2
            per_block = max(int(layout.factor) // subcell_size, 1) ** 2
            mapped_token_count = int(layout.special_count) + int(layout.edge_count) + int(layout.coarse_count) * per_block
            return max(float(mapped_token_count) / float(max(int(layout.source_len), 1)), 1.0)
        return max(float(layout.avg_expand_tokens), 1.0)

    def _resolve_source_downsample_recall_growth(self, state: dotdict) -> float:
        growth = float(
            state.get(
                "source_downsample_recall_growth",
                self.indexer_cfg.get("source_downsample_recall_growth", 1.5),
            )
            or 1.5
        )
        return max(growth, 1.1)

    def _resolve_source_downsample_recall_max_steps(self, state: dotdict) -> int:
        max_steps = int(
            state.get(
                "source_downsample_recall_max_steps",
                self.indexer_cfg.get("source_downsample_recall_max_steps", 3),
            )
            or 3
        )
        return max(max_steps, 1)

    def _resolve_source_downsample_rerank_query_chunk(self, state: dotdict, q_len: int) -> int:
        rerank_qchunk = int(
            state.get(
                "source_downsample_rerank_query_chunk",
                self.indexer_cfg.get("source_downsample_rerank_query_chunk", 256),
            )
            or 256
        )
        return min(max(rerank_qchunk, 1), int(q_len))

    def _resolve_source_downsample_coarse_topk(self, state: dotdict, layout: dotdict, topk: int) -> int:
        strategy = self._resolve_source_downsample_strategy(state, layout)
        coarse_topk = int(
            state.get(
                "source_downsample_coarse_topk",
                self.indexer_cfg.get("source_downsample_coarse_topk", 0),
            ) or 0
        )
        if coarse_topk <= 0:
            coarse_ratio = float(
                state.get(
                    "source_downsample_coarse_ratio",
                    self.indexer_cfg.get("source_downsample_coarse_ratio", 1.1),
                ) or 1.1
            )
            if strategy == "subcell_static_pack4":
                special_count = min(int(layout.special_count), int(topk))
                remaining_topk = max(int(topk) - int(special_count), 0)
                pack_tokens = max(int(layout.factor) // 2, 1) ** 2
                coarse_topk = int(
                    int(special_count)
                    + math.ceil((float(remaining_topk) / float(max(pack_tokens, 1))) * max(coarse_ratio, 1.0))
                )
                return min(max(int(coarse_topk), 1), int(layout.source_len))
            avg_expand = self._get_source_downsample_effective_avg_expand(layout, strategy)
            coarse_topk = int(math.ceil((float(topk) / avg_expand) * max(coarse_ratio, 1.0)))
            special_count = min(int(layout.special_count), int(topk))
            non_special_source_len = int(layout.get("non_special_source_len", 0) or 0)
            non_special_avg_expand = max(float(layout.get("non_special_avg_expand_tokens", 0.0) or 0.0), 1.0)
            if strategy in {"subcell_top1", "subcell_static_pack4"}:
                factor = max(int(layout.factor), 1)
                non_special_avg_expand = max(non_special_avg_expand / float(max(factor, 1) / 2.0) ** 2, 1.0)
            # When the auto coarse budget is smaller than the available special tokens,
            # factor-4 layouts can collapse into an almost all-special selection and then
            # spend the fixed-K path on padding. Reserve room for the singleton specials
            # first, then budget the remaining patch tokens using the non-special average.
            if (
                special_count > 0
                and special_count < int(topk)
                and int(coarse_topk) < int(special_count)
                and non_special_source_len > 0
            ):
                remaining_topk = int(topk) - int(special_count)
                remaining_budget = int(
                    math.ceil((float(remaining_topk) / max(non_special_avg_expand, 1.0)) * max(coarse_ratio, 1.0))
                )
                coarse_topk = max(int(coarse_topk), int(special_count) + max(int(remaining_budget), 1))
        return min(max(int(coarse_topk), 1), int(layout.source_len))

    def _resolve_source_downsample_pad_window(self, state: dotdict, target_topk: int) -> int:
        pad_window = int(
            state.get(
                "source_downsample_pad_window",
                self.indexer_cfg.get("source_downsample_pad_window", 64),
            ) or 64
        )
        return max(1, min(int(pad_window), int(target_topk)))

    def _prepare_source_downsample_projected(
        self,
        k: Tensor,
        w: Tensor,
        state: dotdict,
    ) -> Optional[dotdict]:
        layout = self._get_source_downsample_layout(state, int(k.shape[1]), k.device)
        if layout is None:
            return None

        bsz, _, n_heads, head_dim = k.shape
        tokens_per_view = int(layout.tokens_per_view)
        num_views = int(layout.num_views)
        patch_start_idx = int(layout.patch_start_idx)
        grid_h = int(layout.grid_h)
        grid_w = int(layout.grid_w)
        factor = int(layout.factor)
        pooled_h = int(layout.pooled_h)
        pooled_w = int(layout.pooled_w)
        coarse_h = int(layout.coarse_h)
        coarse_w_tokens = int(layout.coarse_w)
        coarse_count = int(layout.coarse_count)
        edge_count = int(layout.edge_count)

        k_views = k.reshape(bsz, num_views, tokens_per_view, n_heads, head_dim)
        w_views = w.reshape(bsz, num_views, tokens_per_view, n_heads)

        special_k = k_views.new_empty((bsz, 0, n_heads, head_dim))
        special_w = w_views.new_empty((bsz, 0, n_heads))
        if patch_start_idx > 0:
            special_k = k_views[:, :, :patch_start_idx].reshape(bsz, num_views * patch_start_idx, n_heads, head_dim)
            special_w = w_views[:, :, :patch_start_idx].reshape(bsz, num_views * patch_start_idx, n_heads)

        edge_k = k_views.new_empty((bsz, 0, n_heads, head_dim))
        edge_w = w_views.new_empty((bsz, 0, n_heads))
        if edge_count > 0:
            edge_k = k.index_select(dim=1, index=layout.edge_full_indices)
            edge_w = w.index_select(dim=1, index=layout.edge_full_indices)

        coarse_k = k_views.new_empty((bsz, 0, n_heads, head_dim))
        coarse_w = w_views.new_empty((bsz, 0, n_heads))
        coarse_subcell_k = None
        coarse_subcell_w = None
        if coarse_count > 0:
            image_k = k_views[:, :, patch_start_idx:].reshape(
                bsz * num_views, grid_h, grid_w, n_heads * head_dim
            ).permute(0, 3, 1, 2)
            image_w = w_views[:, :, patch_start_idx:].reshape(
                bsz * num_views, grid_h, grid_w, n_heads
            ).permute(0, 3, 1, 2)
            if not bool(layout.get("use_representative_points", False)):
                coarse_k_2d = F.avg_pool2d(
                    image_k[:, :, :pooled_h, :pooled_w],
                    kernel_size=factor,
                    stride=factor,
                    ceil_mode=False,
                    count_include_pad=False,
                )
                coarse_w_2d = F.avg_pool2d(
                    image_w[:, :, :pooled_h, :pooled_w],
                    kernel_size=factor,
                    stride=factor,
                    ceil_mode=False,
                    count_include_pad=False,
                )
            else:
                coarse_k_parts = []
                coarse_w_parts = []
                for yy_off, xx_off in layout.representative_offsets:
                    coarse_k_parts.append(image_k[:, :, yy_off:pooled_h:factor, xx_off:pooled_w:factor])
                    coarse_w_parts.append(image_w[:, :, yy_off:pooled_h:factor, xx_off:pooled_w:factor])
                coarse_k_2d = torch.stack(coarse_k_parts, dim=0).mean(dim=0)
                coarse_w_2d = torch.stack(coarse_w_parts, dim=0).mean(dim=0)
            coarse_k = coarse_k_2d.permute(0, 2, 3, 1).reshape(
                bsz, num_views * coarse_h * coarse_w_tokens, n_heads, head_dim
            )
            coarse_w = coarse_w_2d.permute(0, 2, 3, 1).reshape(
                bsz, num_views * coarse_h * coarse_w_tokens, n_heads
            )
            if factor >= 4 and not bool(layout.get("use_representative_points", False)):
                subcell_k_2d = F.avg_pool2d(
                    image_k[:, :, :pooled_h, :pooled_w],
                    kernel_size=2,
                    stride=2,
                    ceil_mode=False,
                    count_include_pad=False,
                )
                subcell_w_2d = F.avg_pool2d(
                    image_w[:, :, :pooled_h, :pooled_w],
                    kernel_size=2,
                    stride=2,
                    ceil_mode=False,
                    count_include_pad=False,
                )
                coarse_subcell_k = subcell_k_2d.permute(0, 2, 3, 1).reshape(
                    bsz,
                    num_views,
                    coarse_h,
                    2,
                    coarse_w_tokens,
                    2,
                    n_heads,
                    head_dim,
                ).permute(0, 1, 2, 4, 3, 5, 6, 7).reshape(
                    bsz, num_views * coarse_h * coarse_w_tokens, 4, n_heads, head_dim
                )
                coarse_subcell_w = subcell_w_2d.permute(0, 2, 3, 1).reshape(
                    bsz,
                    num_views,
                    coarse_h,
                    2,
                    coarse_w_tokens,
                    2,
                    n_heads,
                ).permute(0, 1, 2, 4, 3, 5, 6).reshape(
                    bsz, num_views * coarse_h * coarse_w_tokens, 4, n_heads
                )
        source_k = torch.cat([special_k, edge_k, coarse_k], dim=1)
        source_w = torch.cat([special_w, edge_w, coarse_w], dim=1)
        static_subcell_to_fine = None
        static_subcell_valid = None
        if coarse_subcell_w is not None and hasattr(layout, "coarse_subcell_to_fine") and hasattr(layout, "coarse_subcell_valid"):
            pack_width = int(layout.coarse_subcell_to_fine.shape[-1])
            source_len = int(layout.source_len)
            static_subcell_to_fine = torch.full((bsz, source_len, pack_width), -1, device=k.device, dtype=torch.long)
            static_subcell_valid = torch.zeros_like(static_subcell_to_fine, dtype=torch.bool)
            source_to_fine_prefix = layout.source_to_fine[:, :1]
            source_valid_prefix = layout.source_valid[:, :1]
            prefix_len = min(int(layout.coarse_source_offset), source_len)
            if prefix_len > 0:
                static_subcell_to_fine[:, :prefix_len, :1] = source_to_fine_prefix[:prefix_len].unsqueeze(0)
                static_subcell_valid[:, :prefix_len, :1] = source_valid_prefix[:prefix_len].unsqueeze(0)
            coarse_count = int(layout.coarse_count)
            if coarse_count > 0:
                static_subcell_score = coarse_subcell_w.mean(dim=-1)
                best_subcell = static_subcell_score.argmax(dim=-1)
                coarse_subcell_to_fine = layout.coarse_subcell_to_fine.unsqueeze(0).expand(bsz, -1, -1, -1)
                coarse_subcell_valid = layout.coarse_subcell_valid.unsqueeze(0).expand(bsz, -1, -1, -1)
                gather_index = best_subcell.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, pack_width)
                selected_coarse_fine = coarse_subcell_to_fine.gather(dim=2, index=gather_index).squeeze(2)
                selected_coarse_valid = coarse_subcell_valid.gather(dim=2, index=gather_index).squeeze(2)
                start = int(layout.coarse_source_offset)
                end = start + coarse_count
                static_subcell_to_fine[:, start:end] = selected_coarse_fine
                static_subcell_valid[:, start:end] = selected_coarse_valid
        return dotdict(
            layout=layout,
            source_k=source_k,
            source_w=source_w,
            coarse_subcell_k=coarse_subcell_k,
            coarse_subcell_w=coarse_subcell_w,
            static_subcell_to_fine=static_subcell_to_fine,
            static_subcell_valid=static_subcell_valid,
            fine_k=k,
            fine_w=w,
            query_chunk=self._resolve_source_downsample_query_chunk(state, int(k.shape[1])),
        )

    def _gather_source_downsample_blocks(self, coarse_indices: Tensor, layout: dotdict) -> Tuple[Tensor, Tensor]:
        bsz, q_len, coarse_topk = coarse_indices.shape
        max_block_tokens = int(layout.max_block_tokens)
        fine_blocks = layout.source_to_fine.index_select(0, coarse_indices.reshape(-1)).reshape(
            bsz, q_len, coarse_topk, max_block_tokens
        )
        fine_valid = layout.source_valid.index_select(0, coarse_indices.reshape(-1)).reshape(
            bsz, q_len, coarse_topk, max_block_tokens
        )
        return fine_blocks, fine_valid

    def _score_source_downsample_candidates(
        self,
        q: Tensor,
        k: Tensor,
        w: Tensor,
        candidate_indices: Tensor,
        candidate_valid: Tensor,
        state: dotdict,
    ) -> Tensor:
        if candidate_indices.shape != candidate_valid.shape:
            raise ValueError(
                f"candidate_indices shape {tuple(candidate_indices.shape)} mismatches candidate_valid {tuple(candidate_valid.shape)}"
            )
        bsz, q_len, cand_len = candidate_indices.shape
        if cand_len <= 0:
            return q.new_empty((bsz, q_len, 0))
        n_heads = int(q.shape[2])
        head_dim = int(q.shape[3])
        rerank_query_chunk = self._resolve_source_downsample_rerank_query_chunk(state, int(q_len))
        flat_k = k.reshape(bsz * int(k.shape[1]), n_heads, head_dim)
        flat_w = w.reshape(bsz * int(w.shape[1]), n_heads)
        batch_offsets = torch.arange(bsz, device=q.device, dtype=torch.long).view(bsz, 1, 1) * int(k.shape[1])
        safe_indices = candidate_indices.clamp(min=0) + batch_offsets
        scale = q.new_tensor(float(self.indexer.scale))
        score_parts = []
        for q_start in range(0, q_len, rerank_query_chunk):
            q_end = min(q_start + rerank_query_chunk, q_len)
            q_part = q[:, q_start:q_end]
            idx_part = safe_indices[:, q_start:q_end]
            valid_part = candidate_valid[:, q_start:q_end]
            cand_k = flat_k.index_select(0, idx_part.reshape(-1)).reshape(
                bsz, q_end - q_start, cand_len, n_heads, head_dim
            )
            cand_w = flat_w.index_select(0, idx_part.reshape(-1)).reshape(
                bsz, q_end - q_start, cand_len, n_heads
            )
            head_scores = torch.einsum("bqhd,bqchd->bqhc", q_part, cand_k) * scale
            head_scores = torch.relu(head_scores)
            head_scores = head_scores * cand_w.permute(0, 1, 3, 2)
            score_part = head_scores.sum(dim=2, dtype=q.dtype)
            score_part = torch.where(valid_part, score_part, torch.full_like(score_part, float("-inf")))
            score_parts.append(score_part)
        return torch.cat(score_parts, dim=1) if len(score_parts) > 1 else score_parts[0]

    def _select_source_downsample_exact_rerank(
        self,
        q: Tensor,
        prepared: dotdict,
        state: dotdict,
        topk: int,
        coarse_topk: int,
    ) -> Optional[Tensor]:
        fine_src_len = int(state.get("tokens_per_view", 0) or 0) * int(state.get("num_views", 0) or 0)
        target_topk = min(int(topk), int(fine_src_len))
        if target_topk <= 0:
            return q.new_empty((int(q.shape[0]), int(q.shape[1]), 0), dtype=torch.long)
        coarse_indices = self.indexer.select_topk_projected(
            q,
            prepared.source_k,
            prepared.source_w,
            mask=None,
            topk=coarse_topk,
            return_scores=False,
        )
        fine_blocks, fine_valid = self._gather_source_downsample_blocks(coarse_indices, prepared.layout)
        candidate_counts = fine_valid.sum(dim=-1, dtype=torch.int64).sum(dim=-1)
        if bool((candidate_counts < target_topk).any()):
            return None
        candidate_indices = fine_blocks.reshape(int(q.shape[0]), int(q.shape[1]), -1)
        candidate_valid = fine_valid.reshape(int(q.shape[0]), int(q.shape[1]), -1)
        candidate_scores = self._score_source_downsample_candidates(
            q,
            prepared.fine_k,
            prepared.fine_w,
            candidate_indices,
            candidate_valid,
            state,
        )
        topk_local = candidate_scores.topk(target_topk, dim=-1, sorted=False)[1]
        return candidate_indices.gather(-1, topk_local)

    def _get_source_downsample_subcell_group_ids(self, layout: dotdict) -> Tensor:
        factor = int(layout.factor)
        if factor < 4 or int(layout.max_block_tokens) <= 1:
            return torch.zeros((int(layout.max_block_tokens),), device=layout.source_to_fine.device, dtype=torch.long)
        subcell_size = 2
        sub_w = max(factor // subcell_size, 1)
        group_ids = []
        for yy_off, xx_off in layout.representative_offsets:
            group_ids.append((yy_off // subcell_size) * sub_w + (xx_off // subcell_size))
        return torch.tensor(group_ids, device=layout.source_to_fine.device, dtype=torch.long)

    def _select_source_downsample_subcell_top1(
        self,
        q: Tensor,
        prepared: dotdict,
        state: dotdict,
        topk: int,
        coarse_topk: int,
    ) -> Optional[Tensor]:
        fine_src_len = int(state.get("tokens_per_view", 0) or 0) * int(state.get("num_views", 0) or 0)
        target_topk = min(int(topk), int(fine_src_len))
        if target_topk <= 0:
            return q.new_empty((int(q.shape[0]), int(q.shape[1]), 0), dtype=torch.long)
        layout = prepared.layout
        if prepared.coarse_subcell_k is None or prepared.coarse_subcell_w is None:
            return None
        if not hasattr(layout, "coarse_subcell_to_fine") or not hasattr(layout, "coarse_subcell_valid"):
            return None

        coarse_indices, coarse_scores = self.indexer.select_topk_projected(
            q,
            prepared.source_k,
            prepared.source_w,
            mask=None,
            topk=coarse_topk,
            return_scores=True,
        )
        bsz, q_len, row_count = coarse_indices.shape
        source_offset = int(layout.coarse_source_offset)
        singleton_mask = coarse_indices < source_offset
        block_mask = ~singleton_mask
        pack_width = int(layout.coarse_subcell_to_fine.shape[-1])

        coarse_rows = layout.source_to_fine.index_select(0, coarse_indices.reshape(-1)).reshape(
            bsz, q_len, row_count, int(layout.max_block_tokens)
        )
        coarse_valid = layout.source_valid.index_select(0, coarse_indices.reshape(-1)).reshape(
            bsz, q_len, row_count, int(layout.max_block_tokens)
        )
        singleton_unit_indices = coarse_rows.new_full((bsz, q_len, row_count, 1, pack_width), -1)
        singleton_unit_valid = torch.zeros_like(singleton_unit_indices, dtype=torch.bool)
        singleton_unit_indices[..., 0, 0] = coarse_rows[..., 0]
        singleton_unit_valid[..., 0, 0] = coarse_valid[..., 0] & singleton_mask

        coarse_count = int(prepared.coarse_subcell_k.shape[1])
        batch_offsets = torch.arange(bsz, device=q.device, dtype=torch.long).view(bsz, 1, 1) * coarse_count
        block_ids = (coarse_indices - source_offset).clamp(min=0)
        safe_block_ids = block_ids + batch_offsets

        flat_subcell_k = prepared.coarse_subcell_k.reshape(
            bsz * coarse_count,
            int(prepared.coarse_subcell_k.shape[2]),
            int(q.shape[2]),
            int(q.shape[3]),
        )
        flat_subcell_w = prepared.coarse_subcell_w.reshape(
            bsz * coarse_count,
            int(prepared.coarse_subcell_w.shape[2]),
            int(q.shape[2]),
        )
        rerank_query_chunk = self._resolve_source_downsample_rerank_query_chunk(state, int(q_len))
        subcell_scores_parts = []
        for q_start in range(0, q_len, rerank_query_chunk):
            q_end = min(q_start + rerank_query_chunk, q_len)
            q_part = q[:, q_start:q_end]
            safe_block_ids_part = safe_block_ids[:, q_start:q_end]
            block_mask_part = block_mask[:, q_start:q_end]
            subcell_k_part = flat_subcell_k.index_select(0, safe_block_ids_part.reshape(-1)).reshape(
                bsz,
                q_end - q_start,
                row_count,
                int(prepared.coarse_subcell_k.shape[2]),
                int(q.shape[2]),
                int(q.shape[3]),
            )
            subcell_w_part = flat_subcell_w.index_select(0, safe_block_ids_part.reshape(-1)).reshape(
                bsz,
                q_end - q_start,
                row_count,
                int(prepared.coarse_subcell_w.shape[2]),
                int(q.shape[2]),
            )
            head_scores = torch.einsum("bqhd,bqcshd->bqcsh", q_part, subcell_k_part) * q.new_tensor(float(self.indexer.scale))
            head_scores = torch.relu(head_scores)
            subcell_scores_part = (head_scores * subcell_w_part).sum(dim=-1, dtype=q.dtype)
            subcell_scores_part = torch.where(
                block_mask_part.unsqueeze(-1),
                subcell_scores_part,
                torch.full_like(subcell_scores_part, float("-inf")),
            )
            subcell_scores_parts.append(subcell_scores_part)
        subcell_scores = (
            torch.cat(subcell_scores_parts, dim=1)
            if len(subcell_scores_parts) > 1
            else subcell_scores_parts[0]
        )

        block_unit_indices = layout.coarse_subcell_to_fine.index_select(0, block_ids.reshape(-1)).reshape(
            bsz,
            q_len,
            row_count,
            int(layout.coarse_subcell_to_fine.shape[1]),
            pack_width,
        )
        block_unit_valid = layout.coarse_subcell_valid.index_select(0, block_ids.reshape(-1)).reshape(
            bsz,
            q_len,
            row_count,
            int(layout.coarse_subcell_valid.shape[1]),
            pack_width,
        ) & block_mask.unsqueeze(-1).unsqueeze(-1)
        singleton_tokens = singleton_unit_indices[..., 0, 0]
        singleton_valid = singleton_unit_valid[..., 0, 0]
        singleton_counts = singleton_valid.sum(dim=-1, dtype=torch.int64)

        flat_subcell_scores = subcell_scores.reshape(bsz, q_len, -1)
        flat_block_unit_indices = block_unit_indices.reshape(bsz, q_len, -1, pack_width)
        flat_block_unit_valid = block_unit_valid.reshape(bsz, q_len, -1, pack_width)
        remain_topk = (torch.full_like(singleton_counts, target_topk) - singleton_counts).clamp_min(0)
        subcell_keep = torch.div(remain_topk + (pack_width - 1), pack_width, rounding_mode="floor")
        max_subcell_keep = int(subcell_keep.max().item()) if subcell_keep.numel() > 0 else 0

        selected_block_indices = flat_block_unit_indices.new_empty((bsz, q_len, 0, pack_width))
        selected_block_valid = torch.zeros_like(selected_block_indices, dtype=torch.bool)
        if max_subcell_keep > 0:
            subcell_keep = torch.minimum(
                subcell_keep,
                flat_block_unit_valid.any(dim=-1).sum(dim=-1, dtype=torch.int64),
            )
            max_subcell_keep = int(subcell_keep.max().item()) if subcell_keep.numel() > 0 else 0
        if max_subcell_keep > 0:
            top_subcell_scores, top_subcell_pos = flat_subcell_scores.topk(max_subcell_keep, dim=-1, sorted=False)
            gather_pos = top_subcell_pos.unsqueeze(-1).expand(-1, -1, -1, pack_width)
            selected_block_indices = flat_block_unit_indices.gather(dim=2, index=gather_pos)
            selected_block_valid = flat_block_unit_valid.gather(dim=2, index=gather_pos)
            keep_mask = (
                torch.arange(max_subcell_keep, device=q.device, dtype=torch.int64).view(1, 1, -1)
                < subcell_keep.unsqueeze(-1)
            ) & top_subcell_scores.isfinite()
            selected_block_valid = selected_block_valid & keep_mask.unsqueeze(-1)

        candidate_counts = singleton_counts + selected_block_valid.sum(dim=-1, dtype=torch.int64).sum(dim=-1)
        if bool((candidate_counts < target_topk).any()):
            return None

        flat_indices = torch.cat(
            [
                singleton_tokens,
                selected_block_indices.reshape(bsz, q_len, -1),
            ],
            dim=-1,
        )
        flat_valid = torch.cat(
            [
                singleton_valid,
                selected_block_valid.reshape(bsz, q_len, -1),
            ],
            dim=-1,
        )
        rank = flat_valid.to(torch.int64).cumsum(dim=-1) - 1
        invalid_rank = flat_indices.shape[-1] + torch.arange(
            flat_indices.shape[-1],
            device=flat_indices.device,
            dtype=torch.int64,
        ).view(1, 1, -1)
        gather_rank = torch.where(flat_valid, rank, invalid_rank)
        compact_order = torch.argsort(gather_rank, dim=-1)
        return flat_indices.gather(-1, compact_order[..., :target_topk])

    def _prepare_query_downsample_inputs_chunk(
        self,
        x: Tensor,
        pos: Optional[Tensor],
        layout: dotdict,
        q_offset: int,
    ) -> dotdict:
        bsz, q_len, feat_dim = x.shape
        max_block_tokens = int(layout.max_block_tokens)
        if q_len <= 0:
            empty_x = x.new_empty((bsz, 0, feat_dim))
            empty_pos = None if pos is None else pos.new_empty((bsz, 0, pos.shape[-1]))
            empty_map = torch.empty((0,), device=x.device, dtype=torch.long)
            return dotdict(coarse_x=empty_x, coarse_pos=empty_pos, fine_to_local_coarse_row=empty_map)

        global_fine = torch.arange(int(q_offset), int(q_offset) + int(q_len), device=x.device, dtype=torch.long)
        row_ids = layout.fine_to_source_row.index_select(0, global_fine)
        if bool((row_ids < 0).any()):
            raise RuntimeError("Invalid query downsample row mapping.")
        unique_rows, inverse = torch.unique(row_ids, sorted=True, return_inverse=True)
        if int(unique_rows.numel()) <= 0:
            empty_x = x.new_empty((bsz, 0, feat_dim))
            empty_pos = None if pos is None else pos.new_empty((bsz, 0, pos.shape[-1]))
            empty_map = torch.empty((0,), device=x.device, dtype=torch.long)
            return dotdict(coarse_x=empty_x, coarse_pos=empty_pos, fine_to_local_coarse_row=empty_map)

        row_local_counts = torch.bincount(inverse, minlength=int(unique_rows.numel()))
        row_full_counts = layout.row_token_counts.index_select(0, unique_rows).to(dtype=row_local_counts.dtype)
        full_row_mask = row_local_counts == row_full_counts
        token_full_mask = full_row_mask.index_select(0, inverse)

        coarse_x_parts = []
        coarse_pos_parts = [] if pos is not None else None
        fine_to_local_coarse_row = torch.full((q_len,), -1, device=x.device, dtype=torch.long)
        coarse_row_offset = 0

        if bool(full_row_mask.any()):
            full_rows = unique_rows[full_row_mask]
            full_local = layout.source_to_fine.index_select(0, full_rows).to(dtype=torch.long) - int(q_offset)
            full_valid = layout.source_valid.index_select(0, full_rows)
            safe_full_local = full_local.clamp(min=0)

            gathered_x = x.index_select(dim=1, index=safe_full_local.reshape(-1)).reshape(
                bsz,
                int(full_rows.shape[0]),
                max_block_tokens,
                feat_dim,
            )
            gathered_x = gathered_x * full_valid.view(1, int(full_rows.shape[0]), max_block_tokens, 1).to(x.dtype)
            denom = layout.row_token_counts.index_select(0, full_rows).clamp(min=1).view(
                1, int(full_rows.shape[0]), 1
            ).to(x.dtype)
            coarse_x_parts.append(gathered_x.sum(dim=2) / denom)

            if coarse_pos_parts is not None:
                # RoPE positions must stay integer-valued. Use the first valid token
                # in each fully covered block as the anchor position instead of
                # averaging coordinates into floats.
                first_valid_slot = full_valid.to(torch.long).argmax(dim=-1, keepdim=True)
                anchor_local = safe_full_local.gather(dim=1, index=first_valid_slot).reshape(-1)
                coarse_pos_parts.append(
                    pos.index_select(dim=1, index=anchor_local).reshape(
                        bsz,
                        int(full_rows.shape[0]),
                        int(pos.shape[-1]),
                    )
                )

            full_row_positions = torch.cumsum(full_row_mask.to(torch.long), dim=0) - 1
            fine_to_local_coarse_row[token_full_mask] = full_row_positions.index_select(
                0, inverse[token_full_mask]
            ) + int(coarse_row_offset)
            coarse_row_offset += int(full_rows.shape[0])

        partial_token_pos = torch.nonzero(~token_full_mask, as_tuple=False).reshape(-1)
        if int(partial_token_pos.numel()) > 0:
            coarse_x_parts.append(x.index_select(dim=1, index=partial_token_pos.to(dtype=torch.long)))
            if coarse_pos_parts is not None:
                coarse_pos_parts.append(pos.index_select(dim=1, index=partial_token_pos.to(dtype=torch.long)))
            fine_to_local_coarse_row[partial_token_pos] = int(coarse_row_offset) + torch.arange(
                int(partial_token_pos.numel()),
                device=x.device,
                dtype=torch.long,
            )

        if len(coarse_x_parts) == 0 or bool((fine_to_local_coarse_row < 0).any()):
            raise RuntimeError("Invalid query downsample coarse-input construction.")

        coarse_x = torch.cat(coarse_x_parts, dim=1)
        coarse_pos = None
        if coarse_pos_parts is not None:
            coarse_pos = torch.cat(coarse_pos_parts, dim=1) if len(coarse_pos_parts) > 0 else pos.new_empty((bsz, 0, pos.shape[-1]))
        return dotdict(coarse_x=coarse_x, coarse_pos=coarse_pos, fine_to_local_coarse_row=fine_to_local_coarse_row)

    def _prepare_query_downsample_projected_chunk(
        self,
        x: Tensor,
        pos: Optional[Tensor],
        layout: dotdict,
        q_offset: int,
    ) -> dotdict:
        if self.indexer is None:
            raise RuntimeError("Query downsample projection requires an active indexer.")
        prepared = self._prepare_query_downsample_inputs_chunk(x, pos, layout, q_offset)
        if int(prepared.coarse_x.shape[1]) <= 0:
            empty_q = x.new_empty((int(x.shape[0]), 0, int(self.indexer.n_heads), int(self.indexer.head_dim)))
            return dotdict(coarse_q=empty_q, fine_to_local_coarse_row=prepared.fine_to_local_coarse_row)
        coarse_q = self.indexer.project_query(prepared.coarse_x, pos=prepared.coarse_pos)
        return dotdict(coarse_q=coarse_q, fine_to_local_coarse_row=prepared.fine_to_local_coarse_row)

    def _prepare_query_downsample_chunk(
        self,
        q: Tensor,
        layout: dotdict,
        q_offset: int,
    ) -> dotdict:
        bsz, q_len, n_heads, head_dim = q.shape
        max_block_tokens = int(layout.max_block_tokens)
        if q_len <= 0:
            empty_q = q.new_empty((bsz, 0, n_heads, head_dim))
            empty_map = torch.empty((0,), device=q.device, dtype=torch.long)
            return dotdict(coarse_q=empty_q, fine_to_local_coarse_row=empty_map)

        global_fine = torch.arange(int(q_offset), int(q_offset) + int(q_len), device=q.device, dtype=torch.long)
        row_ids = layout.fine_to_source_row.index_select(0, global_fine)
        if bool((row_ids < 0).any()):
            raise RuntimeError("Invalid query downsample row mapping.")
        unique_rows, inverse = torch.unique(row_ids, sorted=True, return_inverse=True)
        if int(unique_rows.numel()) <= 0:
            empty_q = q.new_empty((bsz, 0, n_heads, head_dim))
            empty_map = torch.empty((0,), device=q.device, dtype=torch.long)
            return dotdict(coarse_q=empty_q, fine_to_local_coarse_row=empty_map)

        row_local_counts = torch.bincount(inverse, minlength=int(unique_rows.numel()))
        row_full_counts = layout.row_token_counts.index_select(0, unique_rows).to(dtype=row_local_counts.dtype)
        full_row_mask = row_local_counts == row_full_counts
        token_full_mask = full_row_mask.index_select(0, inverse)

        coarse_parts = []
        fine_to_local_coarse_row = torch.full((q_len,), -1, device=q.device, dtype=torch.long)
        coarse_row_offset = 0

        if bool(full_row_mask.any()):
            full_rows = unique_rows[full_row_mask]
            full_local = layout.source_to_fine.index_select(0, full_rows).to(dtype=torch.long) - int(q_offset)
            full_valid = layout.source_valid.index_select(0, full_rows)
            safe_full_local = full_local.clamp(min=0)
            gathered_q = q.index_select(dim=1, index=safe_full_local.reshape(-1)).reshape(
                bsz,
                int(full_rows.shape[0]),
                max_block_tokens,
                n_heads,
                head_dim,
            )
            gathered_q = gathered_q * full_valid.view(1, int(full_rows.shape[0]), max_block_tokens, 1, 1).to(q.dtype)
            denom = layout.row_token_counts.index_select(0, full_rows).clamp(min=1).view(
                1, int(full_rows.shape[0]), 1, 1
            ).to(q.dtype)
            coarse_parts.append(gathered_q.sum(dim=2) / denom)

            full_row_positions = torch.cumsum(full_row_mask.to(torch.long), dim=0) - 1
            fine_to_local_coarse_row[token_full_mask] = full_row_positions.index_select(
                0, inverse[token_full_mask]
            ) + int(coarse_row_offset)
            coarse_row_offset += int(full_rows.shape[0])

        partial_token_pos = torch.nonzero(~token_full_mask, as_tuple=False).reshape(-1)
        if int(partial_token_pos.numel()) > 0:
            coarse_parts.append(q.index_select(dim=1, index=partial_token_pos.to(dtype=torch.long)))
            fine_to_local_coarse_row[partial_token_pos] = int(coarse_row_offset) + torch.arange(
                int(partial_token_pos.numel()),
                device=q.device,
                dtype=torch.long,
            )

        if len(coarse_parts) == 0 or bool((fine_to_local_coarse_row < 0).any()):
            raise RuntimeError("Invalid query downsample coarse-row construction.")

        coarse_q = torch.cat(coarse_parts, dim=1)
        return dotdict(coarse_q=coarse_q, fine_to_local_coarse_row=fine_to_local_coarse_row)

    def _broadcast_source_downsample_chunk_to_queries(
        self,
        coarse_indices: Tensor,
        query_prepared: dotdict,
        fine_q_len: int,
        fine_start: int = 0,
        fine_end: Optional[int] = None,
    ) -> Tensor:
        _, _, _ = coarse_indices.shape
        fine_q_len = int(fine_q_len)
        fine_to_local = query_prepared.fine_to_local_coarse_row.to(dtype=torch.long)
        if int(fine_to_local.numel()) != fine_q_len:
            raise RuntimeError(
                f"Query downsample broadcast size mismatch: fine_q_len={fine_q_len} mapped={int(fine_to_local.numel())}"
            )
        fine_start = max(int(fine_start), 0)
        fine_end = fine_q_len if fine_end is None else min(int(fine_end), fine_q_len)
        if fine_end < fine_start:
            fine_end = fine_start
        if fine_start > 0 or fine_end < fine_q_len:
            fine_to_local = fine_to_local[fine_start:fine_end]
        return coarse_indices.index_select(dim=1, index=fine_to_local)

    def _select_source_downsample_qk_sym_x2_broadcast(
        self,
        q: Tensor,
        prepared: dotdict,
        state: dotdict,
        topk: int,
        coarse_topk: int,
        q_offset: int,
        query_prepared: Optional[dotdict] = None,
    ) -> Tensor:
        query_prepared = (
            query_prepared
            if query_prepared is not None
            else self._prepare_query_downsample_chunk(q, prepared.layout, int(q_offset))
        )
        if int(query_prepared.coarse_q.shape[1]) <= 0:
            return q.new_empty((int(q.shape[0]), int(q.shape[1]), 0), dtype=torch.long)
        fine_q_len = (
            int(query_prepared.fine_to_local_coarse_row.numel())
            if query_prepared is not None
            else int(q.shape[1])
        )
        fine_source_indices = self._select_source_downsample_qk_sym_x2_coarse(
            prepared,
            state,
            int(topk),
            int(coarse_topk),
            query_prepared,
        )
        return self._broadcast_source_downsample_chunk_to_queries(
            fine_source_indices,
            query_prepared,
            fine_q_len=fine_q_len,
        )

    def _select_source_downsample_qk_sym_x2_coarse(
        self,
        prepared: dotdict,
        state: dotdict,
        topk: int,
        coarse_topk: int,
        query_prepared: dotdict,
    ) -> Tensor:
        fine_src_len = int(state.get("tokens_per_view", 0) or 0) * int(state.get("num_views", 0) or 0)
        coarse_indices = self.indexer.select_topk_projected(
            query_prepared.coarse_q,
            prepared.source_k,
            prepared.source_w,
            mask=None,
            topk=coarse_topk,
            return_scores=False,
        )
        return self._expand_source_downsample_chunk(
            coarse_indices,
            prepared.layout,
            state,
            int(topk),
            fine_src_len,
        )

    def _expand_source_downsample_chunk(
        self,
        coarse_indices: Tensor,
        layout: dotdict,
        state: dotdict,
        topk: int,
        fine_src_len: int,
    ) -> Tensor:
        bsz, q_len, coarse_topk = coarse_indices.shape
        max_block_tokens = int(layout.max_block_tokens)
        target_topk = min(int(topk), int(fine_src_len))
        if max_block_tokens <= 0 or target_topk <= 0:
            return coarse_indices.new_empty((bsz, q_len, 0))

        fine_blocks = layout.source_to_fine.index_select(0, coarse_indices.reshape(-1)).reshape(
            bsz, q_len, coarse_topk, max_block_tokens
        )
        fine_valid = layout.source_valid.index_select(0, coarse_indices.reshape(-1)).reshape(
            bsz, q_len, coarse_topk, max_block_tokens
        )
        if bool(layout.get("expand_round_robin", True)):
            # 2x path: preserve more source coverage under fixed-K truncation.
            flat_blocks = fine_blocks.permute(0, 1, 3, 2).reshape(bsz, q_len, coarse_topk * max_block_tokens)
            flat_valid = fine_valid.permute(0, 1, 3, 2).reshape(bsz, q_len, coarse_topk * max_block_tokens)
        else:
            # 4x path: direct representative-point expansion, no cross-block round-robin.
            flat_blocks = fine_blocks.reshape(bsz, q_len, coarse_topk * max_block_tokens)
            flat_valid = fine_valid.reshape(bsz, q_len, coarse_topk * max_block_tokens)
        rank = flat_valid.to(torch.int64).cumsum(dim=-1) - 1
        invalid_rank = flat_blocks.shape[-1] + torch.arange(
            flat_blocks.shape[-1],
            device=flat_blocks.device,
            dtype=torch.int64,
        ).view(1, 1, -1)
        gather_rank = torch.where(flat_valid, rank, invalid_rank)
        order = torch.argsort(gather_rank, dim=-1)
        compact = flat_blocks.gather(-1, order[..., :target_topk])

        valid_counts = flat_valid.sum(dim=-1).clamp(max=target_topk)
        if bool((valid_counts < target_topk).any()):
            safe_counts = valid_counts.clamp(min=1)
            keep = torch.arange(target_topk, device=flat_blocks.device, dtype=torch.int64).view(1, 1, -1)
            pad_window = self._resolve_source_downsample_pad_window(state, target_topk)
            safe_window = torch.minimum(safe_counts, torch.full_like(safe_counts, pad_window))
            start = safe_counts - safe_window
            pad_offsets = (keep - safe_counts.unsqueeze(-1)).clamp_min(0)
            repeat_idx = start.unsqueeze(-1) + torch.remainder(pad_offsets, safe_window.unsqueeze(-1))
            repeat_values = compact.gather(-1, repeat_idx)
            compact = torch.where(keep < safe_counts.unsqueeze(-1), compact, repeat_values)
        return compact

    def _expand_source_downsample_chunk_batched(
        self,
        coarse_indices: Tensor,
        source_to_fine: Tensor,
        source_valid: Tensor,
        *,
        expand_round_robin: bool,
        state: dotdict,
        topk: int,
        fine_src_len: int,
    ) -> Tensor:
        bsz, q_len, coarse_topk = coarse_indices.shape
        max_block_tokens = int(source_to_fine.shape[-1])
        target_topk = min(int(topk), int(fine_src_len))
        if max_block_tokens <= 0 or target_topk <= 0:
            return coarse_indices.new_empty((bsz, q_len, 0))

        source_len = int(source_to_fine.shape[1])
        batch_offsets = torch.arange(bsz, device=coarse_indices.device, dtype=torch.long).view(bsz, 1, 1) * source_len
        flat_source_to_fine = source_to_fine.reshape(bsz * source_len, max_block_tokens)
        flat_source_valid = source_valid.reshape(bsz * source_len, max_block_tokens)
        safe_indices = coarse_indices + batch_offsets
        fine_blocks = flat_source_to_fine.index_select(0, safe_indices.reshape(-1)).reshape(
            bsz, q_len, coarse_topk, max_block_tokens
        )
        fine_valid = flat_source_valid.index_select(0, safe_indices.reshape(-1)).reshape(
            bsz, q_len, coarse_topk, max_block_tokens
        )
        if expand_round_robin:
            flat_blocks = fine_blocks.permute(0, 1, 3, 2).reshape(bsz, q_len, coarse_topk * max_block_tokens)
            flat_valid = fine_valid.permute(0, 1, 3, 2).reshape(bsz, q_len, coarse_topk * max_block_tokens)
        else:
            flat_blocks = fine_blocks.reshape(bsz, q_len, coarse_topk * max_block_tokens)
            flat_valid = fine_valid.reshape(bsz, q_len, coarse_topk * max_block_tokens)
        rank = flat_valid.to(torch.int64).cumsum(dim=-1) - 1
        invalid_rank = flat_blocks.shape[-1] + torch.arange(
            flat_blocks.shape[-1],
            device=flat_blocks.device,
            dtype=torch.int64,
        ).view(1, 1, -1)
        gather_rank = torch.where(flat_valid, rank, invalid_rank)
        order = torch.argsort(gather_rank, dim=-1)
        compact = flat_blocks.gather(-1, order[..., :target_topk])

        valid_counts = flat_valid.sum(dim=-1).clamp(max=target_topk)
        if bool((valid_counts < target_topk).any()):
            safe_counts = valid_counts.clamp(min=1)
            keep = torch.arange(target_topk, device=flat_blocks.device, dtype=torch.int64).view(1, 1, -1)
            pad_window = self._resolve_source_downsample_pad_window(state, target_topk)
            safe_window = torch.minimum(safe_counts, torch.full_like(safe_counts, pad_window))
            start = safe_counts - safe_window
            pad_offsets = (keep - safe_counts.unsqueeze(-1)).clamp_min(0)
            repeat_idx = start.unsqueeze(-1) + torch.remainder(pad_offsets, safe_window.unsqueeze(-1))
            repeat_values = compact.gather(-1, repeat_idx)
            compact = torch.where(keep < safe_counts.unsqueeze(-1), compact, repeat_values)
        return compact

    def _select_topk_source_downsample_chunk(
        self,
        q: Tensor,
        prepared: dotdict,
        state: dotdict,
        topk: int,
        q_offset: int = 0,
        query_prepared: Optional[dotdict] = None,
    ) -> Tensor:
        strategy = self._resolve_source_downsample_strategy(state, prepared.layout)
        coarse_topk = self._resolve_source_downsample_coarse_topk(state, prepared.layout, int(topk))
        if strategy == "qk_sym_x2_broadcast":
            return self._select_source_downsample_qk_sym_x2_broadcast(
                q,
                prepared,
                state,
                topk=int(topk),
                coarse_topk=coarse_topk,
                q_offset=int(q_offset),
                query_prepared=query_prepared,
            )
        if strategy == "legacy":
            fine_src_len = int(state.get("tokens_per_view", 0) or 0) * int(state.get("num_views", 0) or 0)
            coarse_indices = self.indexer.select_topk_projected(
                q,
                prepared.source_k,
                prepared.source_w,
                mask=None,
                topk=coarse_topk,
                return_scores=False,
            )
            return self._expand_source_downsample_chunk(
                coarse_indices,
                prepared.layout,
                state,
                int(topk),
                fine_src_len,
            )
        if strategy == "subcell_static_pack4":
            fine_src_len = int(state.get("tokens_per_view", 0) or 0) * int(state.get("num_views", 0) or 0)
            if prepared.static_subcell_to_fine is None or prepared.static_subcell_valid is None:
                return self.indexer.select_topk_projected(
                    q,
                    prepared.fine_k,
                    prepared.fine_w,
                    mask=None,
                    topk=topk,
                    return_scores=False,
                )
            coarse_indices = self.indexer.select_topk_projected(
                q,
                prepared.source_k,
                prepared.source_w,
                mask=None,
                topk=coarse_topk,
                return_scores=False,
            )
            return self._expand_source_downsample_chunk_batched(
                coarse_indices,
                prepared.static_subcell_to_fine,
                prepared.static_subcell_valid,
                expand_round_robin=True,
                state=state,
                topk=int(topk),
                fine_src_len=fine_src_len,
            )

        recall_growth = self._resolve_source_downsample_recall_growth(state)
        recall_max_steps = self._resolve_source_downsample_recall_max_steps(state)
        attempt_topk = min(max(int(coarse_topk), 1), int(prepared.layout.source_len))
        selector = (
            self._select_source_downsample_exact_rerank
            if strategy == "exact_rerank"
            else self._select_source_downsample_subcell_top1
        )
        for _ in range(recall_max_steps):
            selected = selector(q, prepared, state, int(topk), int(attempt_topk))
            if selected is not None:
                return selected
            if attempt_topk >= int(prepared.layout.source_len):
                break
            next_topk = max(int(math.ceil(float(attempt_topk) * recall_growth)), int(attempt_topk) + 1)
            attempt_topk = min(next_topk, int(prepared.layout.source_len))
        return self.indexer.select_topk_projected(
            q,
            prepared.fine_k,
            prepared.fine_w,
            mask=None,
            topk=topk,
            return_scores=False,
        )

    def _select_topk_projected_inference(
        self,
        q: Tensor,
        k: Tensor,
        w: Tensor,
        state: dotdict,
        *,
        mask: Optional[Tensor],
        topk: int,
        return_scores: bool,
        prepared: Optional[dotdict] = None,
        q_offset: int = 0,
        view_bias_data: Optional[dict] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        use_source_downsample = self._should_use_source_downsample_selector(state, q, k, mask, return_scores)
        if use_source_downsample:
            prepared = prepared if prepared is not None else self._prepare_source_downsample_projected(k, w, state)
            if prepared is not None:
                query_chunk = max(int(prepared.query_chunk), 1)
                if q.shape[1] <= query_chunk:
                    return self._select_topk_source_downsample_chunk(
                        q,
                        prepared,
                        state,
                        int(topk),
                        q_offset=int(q_offset),
                    )
                else:
                    parts = []
                    for q_start in range(0, int(q.shape[1]), query_chunk):
                        q_end = min(q_start + query_chunk, int(q.shape[1]))
                        part = self._select_topk_source_downsample_chunk(
                            q[:, q_start:q_end],
                            prepared,
                            state,
                            int(topk),
                            q_offset=int(q_offset) + int(q_start),
                        )
                        parts.append(part)
                    if len(parts) > 0:
                        return torch.cat(parts, dim=1)
        chunk_view_bias = self._slice_view_bias_data(view_bias_data, q_offset=int(q_offset), q_len=int(q.shape[1]))
        return self.indexer.select_topk_projected(
            q,
            k,
            w,
            mask=mask,
            topk=topk,
            return_scores=return_scores,
            view_bias_data=chunk_view_bias,
        )

    @staticmethod
    def _select_remaining_from_candidates(
        cand_indices: Tensor,
        cand_scores: Optional[Tensor],
        tokens_per_view: int,
        forced_span: int,
        remain_k: int,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, tgt_len, _ = cand_indices.shape
        if remain_k <= 0:
            empty_idx = cand_indices.new_empty((bsz, tgt_len, 0))
            empty_scores = None
            if cand_scores is not None:
                empty_scores = cand_scores.new_empty((bsz, tgt_len, 0))
            return empty_idx, empty_scores

        is_forced = (cand_indices % tokens_per_view) < forced_span
        non_forced = ~is_forced
        rank = non_forced.to(torch.int64).cumsum(dim=-1) - 1
        take = non_forced & (rank < remain_k)
        counts = take.sum(dim=-1)
        expected = torch.full_like(counts, remain_k)
        if not torch.equal(counts, expected):
            raise RuntimeError("Failed to collect enough non-special top-k candidates.")

        rem_indices = cand_indices.masked_select(take).view(bsz, tgt_len, remain_k)
        rem_scores = None
        if cand_scores is not None:
            rem_scores = cand_scores.masked_select(take).view(bsz, tgt_len, remain_k)
        return rem_indices, rem_scores

    def _compute_forced_scores(
        self,
        indexer_input: Tensor,
        pos: Optional[Tensor],
        index_mask: Optional[Tensor],
        forced_indices: Tensor,
        view_bias_data: Optional[dict] = None,
    ) -> Tensor:
        # Compute only the forced-token scores to avoid materializing full [T, T] score maps.
        q, k, w = self.indexer.project(indexer_input, pos=pos)
        k_forced = k.index_select(dim=1, index=forced_indices)
        w_forced = w.index_select(dim=1, index=forced_indices)
        scores = torch.einsum("bthd,bfhd->btfh", q, k_forced) * self.indexer._scale_like(q)
        # Keep this path out-of-place to avoid autograd version mismatch.
        scores = torch.relu(scores)
        scores = scores * w_forced.unsqueeze(1)
        scores = scores.sum(dim=-1, dtype=q.dtype)
        if index_mask is not None:
            if index_mask.dtype != q.dtype:
                index_mask = index_mask.to(q.dtype)
            gather_idx = forced_indices.view(1, 1, -1).expand(scores.shape[0], scores.shape[1], -1)
            scores = scores + index_mask.gather(-1, gather_idx)
        if view_bias_data is not None:
            q_view_ids = view_bias_data["q_view_ids"]
            s_view_ids = view_bias_data["s_view_ids"].index_select(dim=1, index=forced_indices)
            view_bias = view_bias_data["view_bias"].to(dtype=q.dtype)
            batch_idx = torch.arange(scores.shape[0], device=scores.device, dtype=torch.long).view(scores.shape[0], 1, 1)
            scores = scores + view_bias[batch_idx, q_view_ids.unsqueeze(-1), s_view_ids.unsqueeze(1)]
        return scores

    def _build_topk_mask(
        self,
        topk_indices: Tensor,
        base_mask: Optional[Tensor],
        dtype: torch.dtype,
        extra_keep_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # scatter/gather expects int64 indices; some indexers may emit int32 to save memory.
        if topk_indices.dtype != torch.long:
            topk_indices = topk_indices.to(torch.long)
        bsz, tgt_len, topk = topk_indices.shape
        src_len = tgt_len
        fill = self._mask_fill_value(dtype)
        attn_mask = torch.full(
            (bsz, 1, tgt_len, src_len),
            fill,
            device=topk_indices.device,
            dtype=dtype,
        )
        attn_mask.scatter_(dim=-1, index=topk_indices.unsqueeze(1), value=0.0)
        if extra_keep_mask is not None:
            extra_keep_mask = self._normalize_keep_mask(extra_keep_mask, bsz)
            if extra_keep_mask is not None:
                extra_keep_mask = extra_keep_mask.to(device=attn_mask.device, dtype=torch.bool)
                attn_mask = attn_mask.masked_fill(extra_keep_mask, 0.0)
        if base_mask is None:
            return attn_mask
        if base_mask.dtype != dtype:
            base_mask = base_mask.to(dtype)
        return attn_mask + base_mask

    @staticmethod
    def _use_no_grad_attn(state: dotdict) -> bool:
        return bool(
            state.get("warmup", False)
            and state.get("compute_loss", False)
            and state.get("warmup_only_indexer_loss", True)
            and state.get("detach_input", False)
            and state.get("warmup_no_grad_attn", True)
        )

    @staticmethod
    def _normalize_mask(mask: Optional[Tensor], bsz: int) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.size(0) == 1 and bsz > 1:
            mask = mask.expand(bsz, -1, -1, -1)
        return mask

    def _dense_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        scale = self._scale_like(q)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if mask is not None:
            scores = scores + mask
        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        return out, attn

    def _dense_attention_with_loss(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        index_scores: Optional[Tensor],
        state: dotdict,
        indexer_input: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        index_mask: Optional[Tensor] = None,
        projected_indexer: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        view_bias_data: Optional[dict] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, n_heads, tgt_len, head_dim = q.shape
        src_len = k.shape[2]
        use_no_grad_attn = self._use_no_grad_attn(state)
        layer_idx = state.get("layer_idx", None)
        scale = self._scale_like(q)

        p_accum = None

        def _run_materialized_nograd_attention(head_chunk: int) -> Tuple[Tensor, Optional[Tensor]]:
            if head_chunk >= n_heads:
                scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
                if mask is not None:
                    scores = scores + mask
                attn = scores.softmax(dim=-1)
                if self.attn_drop.p > 0 and self.training:
                    attn = self.attn_drop(attn)
                self._maybe_mem_snapshot(
                    state,
                    tag=f"nograd_layer{layer_idx}_h0_{tgt_len}x{src_len}",
                )
                out_local = torch.einsum("bhqk,bhkd->bhqd", attn, v)
                p_local = attn.sum(dim=1) if state.get("compute_loss", False) else None
                return out_local, p_local

            out_local = torch.zeros((bsz, n_heads, tgt_len, head_dim), device=q.device, dtype=q.dtype)
            p_local = (
                torch.zeros((bsz, tgt_len, src_len), device=q.device, dtype=q.dtype)
                if state.get("compute_loss", False)
                else None
            )
            for h_start in range(0, n_heads, head_chunk):
                h_end = min(h_start + head_chunk, n_heads)
                q_h = q[:, h_start:h_end]
                k_h = k[:, h_start:h_end]
                v_h = v[:, h_start:h_end]
                scores = torch.einsum("bhqd,bhkd->bhqk", q_h, k_h) * self._scale_like(q_h)
                if mask is not None:
                    scores = scores + mask
                attn = scores.softmax(dim=-1)
                if self.attn_drop.p > 0 and self.training:
                    attn = self.attn_drop(attn)
                if h_start == 0:
                    self._maybe_mem_snapshot(
                        state,
                        tag=f"nograd_layer{layer_idx}_h0_{tgt_len}x{src_len}",
                    )
                out_local[:, h_start:h_end] = torch.einsum("bhqk,bhkd->bhqd", attn, v_h)
                if p_local is not None:
                    p_local += attn.sum(dim=1)
            return out_local, p_local

        def _valid_attention_summary(summary: Optional[Tensor]) -> bool:
            if summary is None:
                return False
            if summary.numel() == 0:
                return False
            row_sum_mean = float(summary.detach().float().sum(dim=-1).mean().item())
            if not math.isfinite(row_sum_mean):
                return False
            return row_sum_mean > max(float(n_heads) * 0.5, float(state.eps))

        if use_no_grad_attn:
            head_chunk = int(state.get("head_chunk_size", 0) or n_heads)
            head_chunk = max(min(head_chunk, n_heads), 1)
            use_dense_warmup_kernel = (
                dense_index_flash_attn_func is not None
                and q.is_cuda
                and q.dtype in (torch.float16, torch.bfloat16)
                and (not self.training or self.attn_drop.p == 0.0)
                and (
                    bool(
                        state.get(
                            "use_dense_flash_attn_warmup_kernel",
                            self.indexer_cfg.get("use_dense_flash_attn_warmup_kernel", False),
                        )
                    )
                    or self._env_flag("VGGT_DENSE_FLASH_ATTN_WARMUP")
                )
            )
            with torch.no_grad():
                if use_dense_warmup_kernel:
                    dense_bias = mask
                    if dense_bias is not None and dense_bias.dtype != q.dtype:
                        dense_bias = dense_bias.to(q.dtype)
                    out_dense, attn_sum = dense_index_flash_attn_func(
                        q.transpose(1, 2).contiguous(),
                        k.transpose(1, 2).contiguous(),
                        v.transpose(1, 2).contiguous(),
                        dense_bias,
                        False,
                        float(self.scale),
                    )
                    out = out_dense.transpose(1, 2)
                    if state.get("compute_loss", False):
                        if _valid_attention_summary(attn_sum):
                            p_accum = attn_sum.to(q.dtype)
                        else:
                            out, p_accum = _run_materialized_nograd_attention(head_chunk)
                    self._maybe_mem_snapshot(
                        state,
                        tag=f"nograd_dense_kernel_layer{layer_idx}_h0_{tgt_len}x{src_len}",
                    )
                elif head_chunk >= n_heads:
                    out, p_accum = _run_materialized_nograd_attention(head_chunk)
                else:
                    out, p_accum = _run_materialized_nograd_attention(head_chunk)
        else:
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
            if mask is not None:
                scores = scores + mask
            if (
                self._mem_snapshot_enabled(state)
                and self._env_flag("VGGT_MEM_SNAPSHOT_BWD")
                and scores.requires_grad
            ):
                def _scores_bwd_hook(grad: Tensor, _layer_idx=layer_idx) -> Tensor:
                    self._maybe_mem_snapshot(
                        state,
                        tag=f"bwd_scores_layer{_layer_idx}_h0_{tgt_len}x{src_len}",
                    )
                    return grad

                scores.register_hook(_scores_bwd_hook)
            attn = scores.softmax(dim=-1)
            if self.attn_drop.p > 0 and self.training:
                attn = self.attn_drop(attn)
            self._maybe_mem_snapshot(
                state,
                tag=f"dense_layer{layer_idx}_h0_{tgt_len}x{src_len}",
            )
            out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
            if p_accum is not None:
                p_accum = attn.sum(dim=1)

        indexer_loss = None
        if state.get("compute_loss", False):
            if p_accum is None:
                raise RuntimeError(
                    "DSA attention did not produce a dense attention summary for indexer_loss "
                    f"(layer_idx={layer_idx}, warmup={state.get('warmup', False)})."
                )
            p = p_accum / float(n_heads)
            p = p / (p.sum(dim=-1, keepdim=True) + state.eps)
            use_streaming_kl = (
                bool(state.get("streaming_kl_loss", self.indexer_cfg.get("streaming_kl_loss", False)))
                and (not p.requires_grad)
            )
            if use_streaming_kl:
                if self.indexer is None:
                    raise RuntimeError("streaming_kl_loss requires an initialized indexer.")
                warmup_loss_mode = str(
                    state.get(
                        "warmup_indexer_loss_mode",
                        self.indexer_cfg.get("warmup_indexer_loss_mode", "kl"),
                    )
                    or "kl"
                ).strip().lower()
                if warmup_loss_mode in ("topk_coverage", "topk-support", "support_coverage", "coverage"):
                    indexer_loss = self.indexer.compute_topk_support_loss(
                        p=p,
                        support_topk=int(
                            state.get(
                                "warmup_topk_coverage_k",
                                self.indexer_cfg.get("warmup_topk_coverage_k", state.get("topk", 1024)),
                            )
                        ),
                        eps=float(state.eps),
                        x=indexer_input,
                        pos=pos,
                        mask=index_mask,
                        projected=projected_indexer,
                        view_bias_data=view_bias_data,
                        support_chunk_size=int(
                            state.get(
                                "warmup_topk_coverage_chunk_size",
                                self.indexer_cfg.get("warmup_topk_coverage_chunk_size", 128),
                            )
                        ),
                        query_chunk_size=int(
                            state.get(
                                "warmup_topk_coverage_query_chunk_size",
                                self.indexer_cfg.get("warmup_topk_coverage_query_chunk_size", 256),
                            )
                        ),
                        query_sample_size=int(
                            state.get(
                                "warmup_topk_coverage_query_sample_size",
                                self.indexer_cfg.get("warmup_topk_coverage_query_sample_size", 0),
                            )
                        ),
                    )
                elif warmup_loss_mode in ("kl", "dense_kl"):
                    indexer_loss = self.indexer.compute_kl_loss(
                        p=p,
                        eps=float(state.eps),
                        x=indexer_input,
                        pos=pos,
                        mask=index_mask,
                        projected=projected_indexer,
                        view_bias_data=view_bias_data,
                    )
                else:
                    raise ValueError(f"Unsupported warmup_indexer_loss_mode={warmup_loss_mode}")
            else:
                if index_scores is None:
                    if self.indexer is None:
                        raise RuntimeError("index_scores is None but indexer is not available.")
                    if indexer_input is None:
                        raise RuntimeError("indexer_input is required when materializing index scores.")
                    index_scores = self.indexer(
                        indexer_input,
                        pos=pos,
                        mask=index_mask,
                        projected=projected_indexer,
                        view_bias_data=view_bias_data,
                    )
                indexer_loss = self._compute_indexer_loss_from_p(p, index_scores, state)
        return out, indexer_loss

    def _compute_indexer_loss_from_p(
        self,
        p: Tensor,
        index_scores: Tensor,
        state: dotdict,
    ) -> Tensor:
        if state.get("detach_input", False):
            p = p.detach()
        fused_flag = os.getenv("VGGT_INDEXER_LOSS_FUSED", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        use_fused_loss = (
            sparse_indexer_kl_loss is not None
            and fused_flag
            and p.is_cuda
            and index_scores.is_cuda
            and p.dim() == 3
            and index_scores.dim() == 3
            and tuple(p.shape) == tuple(index_scores.shape)
        )
        if use_fused_loss:
            return sparse_indexer_kl_loss(p, index_scores, eps=float(state.eps))
        log_q = F.log_softmax(index_scores, dim=-1)
        return (p * (torch.log(p + state.eps) - log_q)).sum(dim=-1).mean()

    def _gather_selected(self, x: Tensor, idx: Tensor) -> Tensor:
        bsz, n_heads, src_len, dim = x.shape
        tgt_len, topk = idx.shape[1], idx.shape[2]
        x_flat = x.reshape(bsz * n_heads, src_len, dim)
        idx_flat = idx[:, None, :, :].expand(bsz, n_heads, tgt_len, topk).reshape(
            bsz * n_heads,
            tgt_len * topk,
        )
        gathered = x_flat.gather(1, idx_flat.unsqueeze(-1).expand(-1, -1, dim))
        return gathered.reshape(bsz, n_heads, tgt_len, topk, dim)

    def _gather_kv(self, k: Tensor, v: Tensor, idx: Tensor) -> Tuple[Tensor, Tensor]:
        return self._gather_selected(k, idx), self._gather_selected(v, idx)

    def _sparse_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        topk_indices: Tensor,
        mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        k_sel = self._gather_selected(k, topk_indices)
        scale = self._scale_like(q)
        scores = self._compute_sparse_scores(q, k_sel, scale)
        del k_sel
        if mask is not None:
            mask = mask.squeeze(1)
            mask_sel = mask.gather(-1, topk_indices)
            scores = scores + mask_sel.unsqueeze(1)
        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v_sel = self._gather_selected(v, topk_indices)
        if self._env_flag("VGGT_SPARSE_MEM_SNAPSHOT"):
            self._maybe_mem_snapshot(self._state(), f"sparse_preout_layer{self.indexer_state.get('layer_idx', None)}_{attn.shape[2]}x{attn.shape[3]}")
        out = self._apply_attention_to_values(attn, v_sel)
        return out, attn

    def _sparse_attention_with_objective_value_gate(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        topk_indices: Tensor,
        topk_scores: Tensor,
        mask: Optional[Tensor],
        state: dotdict,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, _, tgt_len, _ = q.shape
        query_chunk = self._resolve_objective_value_gate_query_chunk(state, int(tgt_len))
        value_chunk = self._resolve_objective_value_gate_value_chunk(state, int(topk_indices.shape[-1]))
        gather_chunk = self._resolve_objective_value_gate_gather_chunk(state, int(topk_indices.shape[-1]))
        total_heads = int(q.shape[1])
        head_chunk = int(state.get("head_chunk_size", total_heads) or total_heads)
        head_chunk = max(1, min(head_chunk, total_heads))
        gate = self._build_objective_value_gate(topk_scores, state, target_dtype=v.dtype)
        scale = self._scale_like(q)
        out_chunks = []
        p_chunks = [] if bool(state.get("compute_loss", False)) else None

        mask_src = mask.squeeze(1) if mask is not None else None
        force_inner_chunk_reduce = int(value_chunk) < int(topk_indices.shape[-1])
        for q_start in range(0, int(tgt_len), int(query_chunk)):
            q_end = min(q_start + int(query_chunk), int(tgt_len))
            q_chunk = q[:, :, q_start:q_end]
            idx_chunk = topk_indices[:, q_start:q_end]
            gate_chunk = gate[:, :, q_start:q_end].to(dtype=v.dtype)
            out_head_chunks = []
            p_chunk = None
            mask_sel = mask_src[:, q_start:q_end].gather(-1, idx_chunk) if mask_src is not None else None

            for h_start in range(0, total_heads, head_chunk):
                h_end = min(h_start + head_chunk, total_heads)
                q_head = q_chunk[:, h_start:h_end]
                k_head = k[:, h_start:h_end]
                v_head = v[:, h_start:h_end]
                score_chunks = []
                for k_start in range(0, int(idx_chunk.shape[-1]), value_chunk):
                    k_end = min(k_start + value_chunk, int(idx_chunk.shape[-1]))
                    idx_sub = idx_chunk[:, :, k_start:k_end]
                    k_sel = self._gather_selected(k_head, idx_sub)
                    score_chunks.append(
                        self._compute_sparse_scores(
                            q_head,
                            k_sel,
                            scale,
                            key_chunk_size=(k_end - k_start) if force_inner_chunk_reduce else 0,
                        )
                    )
                scores = torch.cat(score_chunks, dim=-1)
                if mask_sel is not None:
                    scores = scores + mask_sel.unsqueeze(1)
                attn = scores.softmax(dim=-1)
                attn = self.attn_drop(attn)
                out_head = None
                for k_start in range(0, int(idx_chunk.shape[-1]), value_chunk):
                    k_end = min(k_start + value_chunk, int(idx_chunk.shape[-1]))
                    idx_sub = idx_chunk[:, :, k_start:k_end]
                    gate_sub = gate_chunk[:, :, :, k_start:k_end]
                    out_chunk = None
                    gather_subchunk = min(gather_chunk, int(idx_sub.shape[-1]))
                    for gather_start in range(0, int(idx_sub.shape[-1]), gather_subchunk):
                        gather_end = min(gather_start + gather_subchunk, int(idx_sub.shape[-1]))
                        idx_inner = idx_sub[:, :, gather_start:gather_end]
                        gate_inner = gate_sub[:, :, :, gather_start:gather_end]
                        v_sel = self._gather_selected(v_head, idx_inner)
                        v_sel.mul_(gate_inner.unsqueeze(-1))
                        inner_chunk = self._apply_attention_to_values(
                            attn[..., k_start + gather_start:k_start + gather_end],
                            v_sel,
                            key_chunk_size=(gather_end - gather_start) if force_inner_chunk_reduce else 0,
                        )
                        if out_chunk is None:
                            out_chunk = inner_chunk
                        else:
                            out_chunk.add_(inner_chunk)
                    if out_head is None:
                        out_head = out_chunk
                    else:
                        out_head.add_(out_chunk)
                out_head_chunks.append(out_head)
                if p_chunks is not None:
                    attn_sum = attn.sum(dim=1)
                    p_chunk = attn_sum if p_chunk is None else (p_chunk + attn_sum)

            out_chunks.append(torch.cat(out_head_chunks, dim=1))
            if p_chunks is not None:
                p_chunks.append(p_chunk)

        out = torch.cat(out_chunks, dim=2)
        p = torch.cat(p_chunks, dim=1) if p_chunks is not None else None
        return out, p

    def forward(self, x: Tensor, pos: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, tgt_len, _ = x.shape
        self.last_topk_indices = None
        self.last_topk_scores = None
        self.last_epipolar_selector_stats = None
        state = self._state()
        layer_idx = state.get("layer_idx", None)
        mask = self._normalize_mask(state.get("attn_mask", None), bsz)
        index_mask = mask.squeeze(1) if mask is not None else None
        indexer_input = x.detach() if state.get("detach_input", False) else x
        projected_indexer = None
        async_topk_event = None
        async_topk_result = None
        source_downsample_prepared = None
        indexer_loss = None
        topk_indices = None
        topk_scores = None
        view_bias_data = self._build_soft_view_bias_data(indexer_input, state)

        if self.indexer is not None:
            self.indexer.layer_idx = layer_idx

        if self._should_use_fullchain_fastpath(state, x, mask):
            fast_out = self._run_fullchain_fastpath(x, pos, state)
            if fast_out is not None:
                return fast_out, None

        if state.get("sparse", False) and self._should_use_sparse_stream_fastpath(state, x, mask) and self._should_use_sparse_stream_attn_q_chunk_proj():
            stream_out = self._run_sparse_streaming_attn_q_chunk_direct(
                x,
                indexer_input,
                pos,
                state,
            )
            if stream_out is not None:
                return stream_out, indexer_loss

        use_fused_proj = self._should_use_fused_proj(state, x)
        use_async_prefetch = self._should_use_async_indexer(state, x) and (not use_fused_proj)
        needs_objective_value_gate = self._objective_value_gate_enabled(state)
        needs_topk_scores = bool(state.get("compute_loss", False)) or needs_objective_value_gate
        if use_async_prefetch:
            prefetch_blocked = (
                state.get("reuse_topk_indices", None) is not None
                or bool(state.get("force_keep_special_tokens", False))
                or bool(state.get("force_keep_self_view_tokens", False))
                or bool(state.get("force_keep_special_in_topk_budget", False))
                or bool(state.get("source_downsample_enabled", False))
            )
            if not prefetch_blocked:
                prefetch_topk = min(int(state.get("topk", self.indexer_cfg.topk)), tgt_len)
                prefetch_need_scores = needs_topk_scores
                prefetch_stream = self._get_async_indexer_stream(x.device)
                with torch.cuda.stream(prefetch_stream):
                    if prefetch_need_scores:
                        async_topk_result = self.indexer(
                            indexer_input,
                            pos=pos,
                            mask=index_mask,
                            topk=prefetch_topk,
                            return_scores=True,
                            projected=projected_indexer,
                            view_bias_data=view_bias_data,
                        )
                    else:
                        prefetch_indices = self.indexer(
                            indexer_input,
                            pos=pos,
                            mask=index_mask,
                            topk=prefetch_topk,
                            return_scores=False,
                            projected=projected_indexer,
                            view_bias_data=view_bias_data,
                        )
                        async_topk_result = (prefetch_indices, None)
                async_topk_event = torch.cuda.Event(blocking=False)
                async_topk_event.record(prefetch_stream)

        if use_fused_proj:
            fused_proj = self._project_qkv_and_indexer_fused(indexer_input, pos)
            if fused_proj is not None:
                qkv, projected_indexer = fused_proj
            else:
                qkv = self.qkv(x)
        else:
            qkv = self.qkv(x)
        q, k, v = self._project_attention_qkv_from_flat(qkv, bsz, tgt_len)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        orig_dtype = q.dtype
        score_dtype = self._resolve_score_dtype(state, orig_dtype)
        if score_dtype != orig_dtype:
            q = q.to(score_dtype)
            k = k.to(score_dtype)
            v = v.to(score_dtype)
        if mask is not None and mask.dtype != q.dtype:
            mask = mask.to(q.dtype)
        source_downsample_candidate = (
            self.indexer is not None
            and bool(state.get("enabled", False))
            and bool(state.get("sparse", False))
            and bool(state.get("source_downsample_enabled", False))
            and (not self.training)
            and (not torch.is_grad_enabled())
            and (not bool(state.get("compute_loss", False)))
            and index_mask is None
        )
        if source_downsample_candidate:
            if projected_indexer is None:
                projected_indexer = self.indexer.project(indexer_input, pos=pos)
            source_downsample_prepared = self._prepare_source_downsample_projected(
                projected_indexer[1],
                projected_indexer[2],
                state,
            )
        expanded_k = None
        expanded_v = None

        def _dense_kv() -> Tuple[Tensor, Tensor]:
            nonlocal expanded_k, expanded_v
            if expanded_k is None or expanded_v is None:
                expanded_k = self._expand_attention_kv_heads(k)
                expanded_v = self._expand_attention_kv_heads(v)
            return expanded_k, expanded_v

        if not state.enabled or self.indexer is None:
            k_dense, v_dense = _dense_kv()
            if self.fused_attn:
                out = F.scaled_dot_product_attention(
                    q,
                    k_dense,
                    v_dense,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                out, _ = self._dense_attention(q, k_dense, v_dense, mask)
            out = out.transpose(1, 2).reshape(bsz, tgt_len, -1)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out, indexer_loss

        if state.get("sparse", False):
            topk = min(int(state.get("topk", self.indexer_cfg.topk)), tgt_len)
            keep_mask = None
            reuse_applied = False

            if self._should_use_sparse_stream_fastpath(state, x, mask):
                stream_out = self._run_sparse_streaming_direct(
                    q,
                    k,
                    v,
                    indexer_input,
                    projected_indexer,
                    pos,
                    state,
                    orig_dtype=orig_dtype,
                )
                if stream_out is not None:
                    return stream_out, indexer_loss

            reuse_topk_indices = state.get("reuse_topk_indices", None)
            reuse_topk_scores = state.get("reuse_topk_scores", None)
            reuse_allow_loss = bool(state.get("reuse_topk_allow_loss", False))
            needs_loss = bool(state.get("compute_loss", False))
            if reuse_topk_indices is not None:
                if (
                    isinstance(reuse_topk_indices, torch.Tensor)
                    and reuse_topk_indices.dim() == 3
                    and reuse_topk_indices.shape[0] == bsz
                    and reuse_topk_indices.shape[1] == tgt_len
                    and reuse_topk_indices.shape[2] >= topk
                    and (not needs_topk_scores or (reuse_allow_loss and isinstance(reuse_topk_scores, torch.Tensor)))
                ):
                    topk_indices = reuse_topk_indices[..., :topk]
                    if topk_indices.dtype != torch.long:
                        topk_indices = topk_indices.to(torch.long)
                    if needs_topk_scores and reuse_allow_loss:
                        if (
                            reuse_topk_scores.dim() == 3
                            and reuse_topk_scores.shape[0] == bsz
                            and reuse_topk_scores.shape[1] == tgt_len
                            and reuse_topk_scores.shape[2] >= topk
                        ):
                            topk_scores = reuse_topk_scores[..., :topk]
                        else:
                            topk_scores = None
                    reuse_applied = True

            keep_special = bool(state.get("force_keep_special_tokens", False))
            keep_self_view = bool(state.get("force_keep_self_view_tokens", False))
            keep_special_in_budget = bool(state.get("force_keep_special_in_topk_budget", False))
            if (not reuse_applied) and async_topk_event is not None and async_topk_result is not None:
                torch.cuda.current_stream().wait_event(async_topk_event)
                topk_indices = async_topk_result[0]
                if topk_indices.dtype != torch.long:
                    topk_indices = topk_indices.to(torch.long)
                if needs_topk_scores:
                    topk_scores = async_topk_result[1]
                reuse_applied = True

            if not reuse_applied:
                special_indices = None
                if keep_special and keep_special_in_budget:
                    special_indices = self._build_special_token_indices(state, tgt_len, q.device)

                if special_indices is not None:
                    special_count = int(special_indices.numel())
                    forced_k = min(topk, special_count)
                    remain_k = topk - forced_k
                    forced_indices = special_indices[:forced_k].view(1, 1, forced_k).expand(bsz, tgt_len, forced_k)
                    rem_indices = forced_indices.new_empty((bsz, tgt_len, 0))
                    rem_scores = None
                    if remain_k > 0:
                        candidate_k = min(tgt_len, remain_k + special_count)
                        if needs_topk_scores:
                            cand_indices, cand_scores = self.indexer(
                                indexer_input,
                                pos=pos,
                                mask=index_mask,
                                topk=candidate_k,
                                return_scores=True,
                                projected=projected_indexer,
                                view_bias_data=view_bias_data,
                            )
                        else:
                            if projected_indexer is not None:
                                cand_indices = self._select_topk_projected_inference(
                                    projected_indexer[0],
                                    projected_indexer[1],
                                    projected_indexer[2],
                                    state,
                                    mask=index_mask,
                                    topk=candidate_k,
                                    return_scores=False,
                                    prepared=source_downsample_prepared,
                                    q_offset=0,
                                    view_bias_data=view_bias_data,
                                )
                            else:
                                cand_indices = self.indexer(
                                    indexer_input,
                                    pos=pos,
                                    mask=index_mask,
                                    topk=candidate_k,
                                    return_scores=False,
                                    projected=projected_indexer,
                                    view_bias_data=view_bias_data,
                                )
                            cand_scores = None
                        rem_indices, rem_scores = self._select_remaining_from_candidates(
                            cand_indices,
                            cand_scores,
                            int(state.get("tokens_per_view", 0) or 0),
                            self._special_token_span(state, int(state.get("patch_start_idx", 0) or 0)),
                            remain_k,
                        )
                    topk_indices = forced_indices if remain_k <= 0 else torch.cat([forced_indices, rem_indices], dim=-1)
                    if needs_topk_scores:
                        forced_scores = self._compute_forced_scores(
                            indexer_input=indexer_input,
                            pos=pos,
                            index_mask=index_mask,
                            forced_indices=special_indices[:forced_k],
                            view_bias_data=view_bias_data,
                        )
                        if rem_scores is None:
                            topk_scores = forced_scores
                        elif forced_k <= 0:
                            topk_scores = rem_scores
                        else:
                            topk_scores = torch.cat([forced_scores, rem_scores], dim=-1)
                    else:
                        topk_scores = None

                    # Special tokens are already inside top-k budget; only keep self-view extras (if requested).
                    if keep_self_view:
                        keep_state = dotdict(state)
                        keep_state.force_keep_special_tokens = False
                        keep_mask = self._build_force_keep_mask(keep_state, tgt_len, q.device)
                    else:
                        keep_mask = None
                else:
                    keep_mask = self._build_force_keep_mask(state, tgt_len, q.device)
                    if needs_topk_scores:
                        topk_indices, topk_scores = self.indexer(
                            indexer_input,
                            pos=pos,
                            mask=index_mask,
                            topk=topk,
                            return_scores=True,
                            projected=projected_indexer,
                            view_bias_data=view_bias_data,
                        )
                    else:
                        if projected_indexer is not None:
                            topk_indices = self._select_topk_projected_inference(
                                projected_indexer[0],
                                projected_indexer[1],
                                projected_indexer[2],
                                state,
                                mask=index_mask,
                                topk=topk,
                                return_scores=False,
                                prepared=source_downsample_prepared,
                                q_offset=0,
                                view_bias_data=view_bias_data,
                            )
                        else:
                            topk_indices = self.indexer(
                                indexer_input,
                                pos=pos,
                                mask=index_mask,
                                topk=topk,
                                return_scores=False,
                                projected=projected_indexer,
                                view_bias_data=view_bias_data,
                            )
                        topk_scores = None
            else:
                keep_mask = self._build_force_keep_mask(state, tgt_len, q.device)

            if needs_topk_scores and topk_scores is None:
                if projected_indexer is None:
                    projected_indexer = self.indexer.project(indexer_input, pos=pos)
                query_indices = torch.arange(tgt_len, device=q.device, dtype=torch.long)
                topk_scores = self._score_projected_gathered(
                    q_subset=projected_indexer[0],
                    k_proj=projected_indexer[1],
                    w_proj=projected_indexer[2],
                    source_indices=topk_indices,
                    query_indices=query_indices,
                    tokens_per_view=max(int(state.get("tokens_per_view", 0) or 0), 1),
                    view_bias_data=view_bias_data,
                )

            self.last_topk_indices = topk_indices.detach()
            self.last_topk_scores = topk_scores.detach() if isinstance(topk_scores, torch.Tensor) else None

            use_mask = bool(state.get("sparse_use_mask", self.indexer_cfg.get("sparse_use_mask", False))) or (
                keep_mask is not None
            )
            use_objective_value_gate = needs_objective_value_gate and (keep_mask is None) and (not use_mask)
            use_sparse_kernel = bool(state.get("use_sparse_flash_attn", False)) or self._env_flag(
                "VGGT_SPARSE_FLASH_ATTN"
            )
            if use_objective_value_gate:
                use_compute_loss = bool(state.get("compute_loss", False))
                use_value_gate_sparse_kernel = (
                    sparse_index_flash_attn_value_gate_bhtd_func is not None and q.is_cuda and use_sparse_kernel
                )
                if use_value_gate_sparse_kernel:
                    allow_sort_kv_train = use_compute_loss and self._env_flag("VGGT_SPARSE_FLASH_SORT_KV_TRAIN_SAFE")
                    sort_kv = ((not use_compute_loss) or allow_sort_kv_train) and (
                        bool(state.get("sparse_flash_sort_kv", self.indexer_cfg.get("sparse_flash_sort_kv", False)))
                        or self._env_flag("VGGT_SPARSE_FLASH_SORT_KV")
                    )
                    kv_indices = topk_indices
                    if sort_kv:
                        kv_indices, sort_order = torch.sort(kv_indices, dim=-1)
                        if topk_scores is not None:
                            topk_scores = torch.gather(topk_scores, dim=-1, index=sort_order)
                    kv_positions = kv_indices.to(torch.int32)
                    force_kv_contig = not self._env_flag("VGGT_SPARSE_FLASH_NO_KV_CONTIG")
                    if force_kv_contig and (not kv_positions.is_contiguous()):
                        kv_positions = kv_positions.contiguous()
                    if q.dtype not in (torch.float16, torch.bfloat16):
                        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        q = q.to(target_dtype)
                        k = k.to(target_dtype)
                        v = v.to(target_dtype)
                    gate = self._build_objective_value_gate(topk_scores, state, target_dtype=v.dtype)
                    allow_train_no_qkv_contig = self._env_flag("VGGT_SPARSE_FLASH_NO_QKV_CONTIG_TRAIN")
                    no_qkv_contig = self._env_flag("VGGT_SPARSE_FLASH_NO_QKV_CONTIG") and (
                        ((not self.training) and (not torch.is_grad_enabled())) or allow_train_no_qkv_contig
                    )
                    q_bhtd = q
                    k_bhtd = k
                    v_bhtd = v
                    if not no_qkv_contig:
                        q_bhtd = q_bhtd.contiguous()
                        k_bhtd = k_bhtd.contiguous()
                        v_bhtd = v_bhtd.contiguous()
                    out, attn = sparse_index_flash_attn_value_gate_bhtd_func(
                        q_bhtd,
                        k_bhtd,
                        v_bhtd,
                        kv_positions,
                        gate,
                        float(self.scale),
                        use_compute_loss,
                    )
                    if attn is not None and attn.numel() == 0:
                        attn = None
                    elif attn is not None and attn.dtype != q.dtype:
                        attn = attn.to(q.dtype)
                else:
                    k_dense, v_dense = _dense_kv()
                    out, attn = self._sparse_attention_with_objective_value_gate(
                        q,
                        k_dense,
                        v_dense,
                        topk_indices,
                        topk_scores,
                        mask,
                        state,
                    )
            elif use_mask:
                k_dense, v_dense = _dense_kv()
                attn_mask = self._build_topk_mask(topk_indices, mask, q.dtype, keep_mask)
                if self.fused_attn:
                    out = F.scaled_dot_product_attention(
                        q,
                        k_dense,
                        v_dense,
                        attn_mask=attn_mask,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                    )
                    attn = None
                else:
                    out, attn = self._dense_attention(q, k_dense, v_dense, attn_mask)
            else:
                if sparse_index_flash_attn_func is not None and q.is_cuda and use_sparse_kernel:
                    use_compute_loss = bool(state.get("compute_loss", False))
                    allow_sort_kv_train = use_compute_loss and self._env_flag("VGGT_SPARSE_FLASH_SORT_KV_TRAIN_SAFE")
                    sort_kv = ((not use_compute_loss) or allow_sort_kv_train) and (
                        bool(state.get("sparse_flash_sort_kv", self.indexer_cfg.get("sparse_flash_sort_kv", False)))
                        or self._env_flag("VGGT_SPARSE_FLASH_SORT_KV")
                    )
                    kv_indices = topk_indices
                    sort_order = None
                    if sort_kv:
                        kv_indices, sort_order = torch.sort(kv_indices, dim=-1)
                        if use_compute_loss and topk_scores is not None:
                            topk_scores = torch.gather(topk_scores, dim=-1, index=sort_order)
                    kv_positions = kv_indices.to(torch.int32)
                    force_kv_contig = not self._env_flag("VGGT_SPARSE_FLASH_NO_KV_CONTIG")
                    if force_kv_contig and (not kv_positions.is_contiguous()):
                        kv_positions = kv_positions.contiguous()
                    bias = None
                    if mask is not None:
                        mask_sel = mask.squeeze(1).gather(-1, kv_indices)
                        bias = mask_sel.unsqueeze(1).contiguous()
                    if q.dtype not in (torch.float16, torch.bfloat16):
                        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        q = q.to(target_dtype)
                        k = k.to(target_dtype)
                        v = v.to(target_dtype)
                        if bias is not None and bias.dtype != target_dtype:
                            bias = bias.to(target_dtype)
                    elif bias is not None and bias.dtype != q.dtype:
                        bias = bias.to(q.dtype)
                    allow_train_no_qkv_contig = self._env_flag("VGGT_SPARSE_FLASH_NO_QKV_CONTIG_TRAIN")
                    no_qkv_contig = self._env_flag("VGGT_SPARSE_FLASH_NO_QKV_CONTIG") and (
                        ((not self.training) and (not torch.is_grad_enabled())) or allow_train_no_qkv_contig
                    )
                    q_in = q.transpose(1, 2)
                    k_in = k.transpose(1, 2)
                    v_in = v.transpose(1, 2)
                    if not no_qkv_contig:
                        q_in = q_in.contiguous()
                        k_in = k_in.contiguous()
                        v_in = v_in.contiguous()
                    disable_attn_sum = (not use_compute_loss) and (
                        bool(
                            state.get(
                                "sparse_flash_disable_attn_sum",
                                self.indexer_cfg.get("sparse_flash_disable_attn_sum", False),
                            )
                        )
                        or self._env_flag("VGGT_SPARSE_FLASH_DISABLE_ATTN_SUM")
                    )
                    use_inference_sparse = (
                        disable_attn_sum
                        and sparse_index_flash_attn_inference_func is not None
                        and (not self.training)
                        and (not torch.is_grad_enabled())
                    )
                    use_inference_sparse_bhtd = (
                        use_inference_sparse
                        and sparse_index_flash_attn_inference_bhtd_func is not None
                        and self._env_flag("VGGT_SPARSE_FLASH_DIRECT_BHTD")
                    )
                    use_autograd_sparse_bhtd = self._should_use_sparse_bhtd_autograd(state, use_inference_sparse)
                    out_is_bhtd = False
                    if use_inference_sparse:
                        if use_inference_sparse_bhtd:
                            q_bhtd = q
                            k_bhtd = k
                            v_bhtd = v
                            if not no_qkv_contig:
                                q_bhtd = q_bhtd.contiguous()
                                k_bhtd = k_bhtd.contiguous()
                                v_bhtd = v_bhtd.contiguous()
                            out, attn = sparse_index_flash_attn_inference_bhtd_func(
                                q_bhtd,
                                k_bhtd,
                                v_bhtd,
                                kv_positions,
                                bias,
                                False,
                                float(self.scale),
                                False,
                            )
                            out_is_bhtd = True
                        else:
                            out, attn = sparse_index_flash_attn_inference_func(
                                q_in,
                                k_in,
                                v_in,
                                kv_positions,
                                bias,
                                False,
                                float(self.scale),
                                False,
                            )
                    elif use_autograd_sparse_bhtd:
                        q_bhtd = q
                        k_bhtd = k
                        v_bhtd = v
                        if not no_qkv_contig:
                            q_bhtd = q_bhtd.contiguous()
                            k_bhtd = k_bhtd.contiguous()
                            v_bhtd = v_bhtd.contiguous()
                        out, attn = sparse_index_flash_attn_bhtd_func(
                            q_bhtd,
                            k_bhtd,
                            v_bhtd,
                            kv_positions,
                            bias,
                            False,
                            float(self.scale),
                        )
                        out_is_bhtd = True
                    else:
                        out, attn = sparse_index_flash_attn_func(
                            q_in,
                            k_in,
                            v_in,
                            kv_positions,
                            bias,
                            False,
                            float(self.scale),
                        )
                    if not out_is_bhtd:
                        out = out.transpose(1, 2)
                    if attn is not None and attn.dtype != q.dtype:
                        attn = attn.to(q.dtype)
                else:
                    k_dense, v_dense = _dense_kv()
                    out, attn = self._sparse_attention(q, k_dense, v_dense, topk_indices, mask)
            if state.get("compute_loss", False) and attn is not None:
                p = attn if attn.dim() == 3 else attn.sum(dim=1)
                if topk_scores is not None and p.shape[-1] != topk_scores.shape[-1]:
                    # In mask-based sparse path, attn can be full [B, H, T, T].
                    # Align p to top-k domain so KL compares distributions on identical support.
                    p = p.gather(-1, topk_indices)
                p = p / (p.sum(dim=-1, keepdim=True) + state.eps)
                indexer_loss = self._compute_indexer_loss_from_p(p, topk_scores, state)
            elif state.get("compute_loss", False) and attn is None and not self.__class__._sparse_loss_warned:
                if is_main_process():
                    log("[dsa] sparse_use_mask enabled: skip indexer_loss (no attn weights).")
                self.__class__._sparse_loss_warned = True
        else:
            if state.get("compute_loss", False):
                need_streaming_kl = (
                    bool(state.get("streaming_kl_loss", self.indexer_cfg.get("streaming_kl_loss", False)))
                    and self._use_no_grad_attn(state)
                )
                index_scores = None
                if not need_streaming_kl:
                    index_scores = self.indexer(
                        indexer_input,
                        pos=pos,
                        mask=index_mask,
                        projected=projected_indexer,
                        view_bias_data=view_bias_data,
                    )
                k_dense, v_dense = _dense_kv()
                out, indexer_loss = self._dense_attention_with_loss(
                    q,
                    k_dense,
                    v_dense,
                    mask,
                    index_scores,
                    state,
                    indexer_input=indexer_input,
                    pos=pos,
                    index_mask=index_mask,
                    projected_indexer=projected_indexer,
                    view_bias_data=view_bias_data,
                )
            else:
                k_dense, v_dense = _dense_kv()
                out, _ = self._dense_attention(q, k_dense, v_dense, mask)

        if state.get("sparse", False):
            epipolar_selector_loss = self._compute_epipolar_selector_band_loss(
                projected_indexer=projected_indexer,
                indexer_input=indexer_input,
                pos=pos,
                state=state,
                index_mask=index_mask,
                view_bias_data=view_bias_data,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
            )
            if epipolar_selector_loss is not None:
                epi_weight = self._epipolar_selector_loss_weight(state)
                if indexer_loss is None:
                    indexer_loss = epipolar_selector_loss * epi_weight
                else:
                    indexer_loss = indexer_loss + epipolar_selector_loss * epi_weight

        if indexer_loss is not None:
            indexer_loss = indexer_loss * state.loss_weight
            if indexer_loss.dtype != orig_dtype:
                indexer_loss = indexer_loss.to(orig_dtype)

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        out = out.transpose(1, 2).reshape(bsz, tgt_len, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, indexer_loss
