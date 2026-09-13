import math
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_INT32_MAX = (1 << 31) - 1
_SEGMENT_TOPK_ERROR_COUNT = 0

try:
    from .merge_two_topk_cuda import is_merge_two_topk_available as _is_merge_two_topk_available
    from .merge_two_topk_cuda import merge_two_topk_inplace as _merge_two_topk_inplace
except Exception:
    _is_merge_two_topk_available = None
    _merge_two_topk_inplace = None

try:
    from .topk_exact_cuda import is_topk_exact_available as _is_topk_exact_available
    from .topk_exact_cuda import topk_exact_select as _topk_exact_select
except Exception:
    _is_topk_exact_available = None
    _topk_exact_select = None

try:
    from .score_topk_fused_cuda import is_score_topk_fused_available as _is_score_topk_fused_available
    from .score_topk_fused_cuda import score_topk_fused_select as _score_topk_fused_select
except Exception:
    _is_score_topk_fused_available = None
    _score_topk_fused_select = None

_WORKSPACE_CACHE = {}
_ARANGE_CACHE = {}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    try:
        return int(value) if value else default
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "")
    if not value:
        return default
    return str(value).strip()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if not value:
        return bool(default)
    return value.lower() in ("1", "true", "yes", "y", "on")


def _alloc_workspace(name: str, shape, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not _env_flag("VGGT_INDEXER_TOPK_PERSIST_WORKSPACE", False):
        return torch.empty(shape, device=device, dtype=dtype)
    mode = _env_str("VGGT_INDEXER_TOPK_PERSIST_WORKSPACE_MODE", "exact").lower()
    if mode in ("pool", "max", "largest"):
        # Pool mode keeps one grow-only flat buffer per (name, device, dtype),
        # then returns shaped views. This avoids shape-cardinality explosion.
        need_numel = int(math.prod(int(x) for x in shape))
        key = (name, str(device), str(dtype), "pool")
        cached = _WORKSPACE_CACHE.get(key)
        if cached is None or cached.numel() < need_numel or cached.device != device or cached.dtype != dtype:
            cached = torch.empty((need_numel,), device=device, dtype=dtype)
            _WORKSPACE_CACHE[key] = cached
        return cached[:need_numel].view(shape)

    key = (name, tuple(int(x) for x in shape), str(device), str(dtype), "exact")
    cached = _WORKSPACE_CACHE.get(key)
    if cached is None or tuple(cached.shape) != tuple(shape) or cached.device != device or cached.dtype != dtype:
        cached = torch.empty(shape, device=device, dtype=dtype)
        _WORKSPACE_CACHE[key] = cached
    return cached


def _get_cached_arange(length: int, *, device: torch.device) -> torch.Tensor:
    if not _env_flag("VGGT_INDEXER_TOPK_CACHE_ARANGE", False):
        return torch.arange(length, device=device, dtype=torch.int32)
    key = (int(length), str(device))
    cached = _ARANGE_CACHE.get(key)
    if cached is None or cached.device != device or cached.numel() != int(length):
        cached = torch.arange(length, device=device, dtype=torch.int32)
        _ARANGE_CACHE[key] = cached
        # Keep cache tiny to avoid unbounded growth on shape sweeps.
        max_entries = max(1, _env_int("VGGT_INDEXER_TOPK_ARANGE_CACHE_MAX", 8))
        if len(_ARANGE_CACHE) > max_entries:
            oldest = next(iter(_ARANGE_CACHE))
            if oldest != key:
                _ARANGE_CACHE.pop(oldest, None)
    return cached


def _needs_pending_topk_indices_buf(use_merge_index_kernel: bool) -> bool:
    return not bool(use_merge_index_kernel)


def clear_runtime_cache() -> None:
    _WORKSPACE_CACHE.clear()
    _ARANGE_CACHE.clear()


def _resolve_block_merge_policy(
    seq_len: int,
    topk: int,
    block_k: int,
    merge_blocks: int,
) -> Tuple[int, int]:
    policy = _env_str("VGGT_INDEXER_TOPK_BLOCK_MERGE_POLICY", "off").lower()
    if policy in ("", "0", "off", "none"):
        return int(block_k), int(merge_blocks)

    if policy in ("auto_512_v1", "v1"):
        if int(topk) == 512:
            # 40 * 1374 tokens is a practical knee point from local profiling.
            min_tokens = _env_int("VGGT_INDEXER_TOPK_POLICY_MIN_TOKENS", 54960)
            if int(seq_len) >= int(min_tokens):
                block_k = _env_int("VGGT_INDEXER_TOPK_POLICY_BLOCK", 768)
                merge_blocks = _env_int("VGGT_INDEXER_TOPK_POLICY_MERGE", 32)

    return int(block_k), int(merge_blocks)


def _resolve_runtime_bucket_policy(
    seq_len: int,
    topk: int,
    block_k: int,
    merge_blocks: int,
    query_chunk: int,
    merge_index_block_k: int,
) -> Tuple[int, int, int, int]:
    policy = _env_str("VGGT_INDEXER_TOPK_BUCKET_POLICY", "off").lower()
    if policy in ("", "0", "off", "none"):
        return int(block_k), int(merge_blocks), int(query_chunk), int(merge_index_block_k)

    if policy in ("auto_512_v2", "v2") and int(topk) == 512:
        # Token buckets follow src_view count (tokens_per_view ~= 1374).
        tok_per_view = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_TOKENS_PER_VIEW", 1374))
        b1_views = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_V1_MAX_VIEWS", 30))
        b2_views = max(b1_views, _env_int("VGGT_INDEXER_TOPK_BUCKET_V2_MAX_VIEWS", 70))
        b3_views = max(b2_views, _env_int("VGGT_INDEXER_TOPK_BUCKET_V3_MAX_VIEWS", 110))
        b1_tokens = b1_views * tok_per_view
        b2_tokens = b2_views * tok_per_view
        b3_tokens = b3_views * tok_per_view

        v1_block = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_V1_BLOCK", 896))
        v1_merge = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_V1_MERGE", 20))
        v1_qchunk = max(0, _env_int("VGGT_INDEXER_TOPK_BUCKET_V1_QCHUNK", 0))
        v1_merge_idx = max(32, _env_int("VGGT_INDEXER_TOPK_BUCKET_V1_MERGE_INDEX_BLOCK", 256))

        v2_block = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_V2_BLOCK", 896))
        v2_merge = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_V2_MERGE", 24))
        v2_qchunk = max(0, _env_int("VGGT_INDEXER_TOPK_BUCKET_V2_QCHUNK", 0))
        v2_merge_idx = max(32, _env_int("VGGT_INDEXER_TOPK_BUCKET_V2_MERGE_INDEX_BLOCK", 256))

        v3_block = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_V3_BLOCK", 768))
        v3_merge = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_V3_MERGE", 28))
        v3_qchunk = max(0, _env_int("VGGT_INDEXER_TOPK_BUCKET_V3_QCHUNK", 0))
        v3_merge_idx = max(32, _env_int("VGGT_INDEXER_TOPK_BUCKET_V3_MERGE_INDEX_BLOCK", 128))

        v4_block = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_V4_BLOCK", 768))
        v4_merge = max(1, _env_int("VGGT_INDEXER_TOPK_BUCKET_V4_MERGE", 34))
        v4_qchunk = max(0, _env_int("VGGT_INDEXER_TOPK_BUCKET_V4_QCHUNK", 0))
        v4_merge_idx = max(32, _env_int("VGGT_INDEXER_TOPK_BUCKET_V4_MERGE_INDEX_BLOCK", 128))

        if seq_len <= b1_tokens:
            block_k, merge_blocks, query_chunk, merge_index_block_k = (
                v1_block,
                v1_merge,
                v1_qchunk,
                max(v1_merge_idx, merge_index_block_k),
            )
        elif seq_len <= b2_tokens:
            block_k, merge_blocks, query_chunk, merge_index_block_k = (
                v2_block,
                v2_merge,
                v2_qchunk,
                max(v2_merge_idx, merge_index_block_k),
            )
        elif seq_len <= b3_tokens:
            block_k, merge_blocks, query_chunk, merge_index_block_k = (
                v3_block,
                v3_merge,
                v3_qchunk,
                max(v3_merge_idx, merge_index_block_k),
            )
        else:
            block_k, merge_blocks, query_chunk, merge_index_block_k = (
                v4_block,
                v4_merge,
                v4_qchunk,
                max(v4_merge_idx, merge_index_block_k),
            )

    return int(block_k), int(merge_blocks), int(query_chunk), int(merge_index_block_k)


def _resolve_outer_query_chunk_policy(
    *,
    tgt_len: int,
    src_len: int,
    topk: int,
    user_chunk: int,
) -> int:
    if int(user_chunk) > 0:
        return max(1, min(int(user_chunk), int(tgt_len)))

    policy = _env_str("VGGT_INDEXER_TOPK_OUTER_QUERY_POLICY", "auto_512_v1").lower()
    if policy in ("", "0", "off", "none"):
        return 0

    if policy in ("auto_512_v1", "v1", "auto"):
        if int(topk) != 512:
            return 0
        min_tokens = max(1, _env_int("VGGT_INDEXER_TOPK_OUTER_QUERY_POLICY_MIN_TOKENS", 70 * 1374))
        if int(src_len) < int(min_tokens):
            return 0
        chunk = max(1, _env_int("VGGT_INDEXER_TOPK_OUTER_QUERY_POLICY_CHUNK", 49152))
        return min(int(tgt_len), int(chunk))

    return 0


def _resolve_merge_two_policy(
    *,
    src_len: int,
    topk: int,
    explicit_want: bool,
    q_requires_grad: bool,
) -> bool:
    if bool(explicit_want):
        return True

    if bool(q_requires_grad) and (not _env_flag("VGGT_INDEXER_TOPK_MERGE_TWO_POLICY_ALLOW_GRAD", False)):
        return False

    # Keep strict numerical behavior by default; enable via env for opt-in inference speed.
    policy = _env_str("VGGT_INDEXER_TOPK_MERGE_TWO_POLICY", "off").lower()
    if policy in ("", "0", "off", "none"):
        return False

    if policy in ("auto_512_v1", "v1", "auto"):
        if int(topk) != 512:
            return False
        min_tokens = max(1, _env_int("VGGT_INDEXER_TOPK_MERGE_TWO_POLICY_MIN_TOKENS", 70 * 1374))
        return int(src_len) >= int(min_tokens)

    return False


def _resolve_score_kernel_policy(
    src_len: int,
    head_dim: int,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> Tuple[int, int, int, int]:
    policy = _env_str("VGGT_INDEXER_SCORE_BUCKET_POLICY", "off").lower()
    if policy in ("", "0", "off", "none"):
        return int(block_m), int(block_n), int(num_warps), int(num_stages)

    if policy in ("auto_tc_v1", "tc_v1"):
        # Tuned for tensor-core path on long-seq sparse indexer.
        tok_per_view = max(1, _env_int("VGGT_INDEXER_SCORE_POLICY_TOKENS_PER_VIEW", 1374))
        small_views = max(1, _env_int("VGGT_INDEXER_SCORE_POLICY_SMALL_VIEWS", 70))
        small_tokens = small_views * tok_per_view
        small_bm = max(16, _env_int("VGGT_INDEXER_SCORE_POLICY_SMALL_BLOCK_M", 128))
        small_bn = max(16, _env_int("VGGT_INDEXER_SCORE_POLICY_SMALL_BLOCK_N", 128))
        small_warps = max(1, _env_int("VGGT_INDEXER_SCORE_POLICY_SMALL_WARPS", 8))
        small_stages = max(1, _env_int("VGGT_INDEXER_SCORE_POLICY_SMALL_STAGES", 1))
        large_bm = max(16, _env_int("VGGT_INDEXER_SCORE_POLICY_LARGE_BLOCK_M", 64))
        large_bn = max(16, _env_int("VGGT_INDEXER_SCORE_POLICY_LARGE_BLOCK_N", 256))
        large_warps = max(1, _env_int("VGGT_INDEXER_SCORE_POLICY_LARGE_WARPS", 8))
        large_stages = max(1, _env_int("VGGT_INDEXER_SCORE_POLICY_LARGE_STAGES", 1))
        if int(src_len) <= int(small_tokens):
            block_m, block_n, num_warps, num_stages = small_bm, small_bn, small_warps, small_stages
        else:
            block_m, block_n, num_warps, num_stages = large_bm, large_bn, large_warps, large_stages

    return int(block_m), int(block_n), int(num_warps), int(num_stages)


def _resolve_topk_query_chunk(scores: torch.Tensor, user_chunk: int) -> int:
    # Keep each top-k call under int32 indexing bound to avoid slow large-index kernels.
    bsz, rows, cols = scores.shape
    if user_chunk > 0:
        return min(int(user_chunk), int(rows))

    if not _env_flag("VGGT_INDEXER_TOPK_AUTO_QUERY_CHUNK", True):
        return 0

    if bsz <= 0 or rows <= 0 or cols <= 0:
        return 0

    max_rows_i32 = _INT32_MAX // max(1, int(bsz) * int(cols))
    if max_rows_i32 <= 0 or rows <= max_rows_i32:
        return 0

    # Mild alignment tends to give stabler top-k kernel throughput.
    if max_rows_i32 > 1024:
        max_rows_i32 = (max_rows_i32 // 1024) * 1024
    return max(1, min(int(rows), int(max_rows_i32)))


def _topk_lastdim_torch(
    scores: torch.Tensor,
    k: int,
    *,
    sorted: bool,
    out_scores: Optional[torch.Tensor] = None,
    out_pos: Optional[torch.Tensor] = None,
    query_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 3:
        raise ValueError(f"Expected 3D scores tensor for top-k, got shape: {tuple(scores.shape)}")

    bsz, rows, _ = scores.shape
    if bsz == 0 or rows == 0:
        if out_scores is not None and out_pos is not None:
            return out_scores, out_pos
        return torch.topk(scores, k, dim=-1, sorted=sorted)

    chunk_rows = _resolve_topk_query_chunk(scores, int(query_chunk))
    if chunk_rows <= 0 or chunk_rows >= rows:
        if out_scores is not None and out_pos is not None:
            torch.topk(scores, k, dim=-1, sorted=sorted, out=(out_scores, out_pos))
            return out_scores, out_pos
        return torch.topk(scores, k, dim=-1, sorted=sorted)

    if out_scores is not None and out_pos is not None:
        for q_start in range(0, rows, chunk_rows):
            q_end = min(q_start + chunk_rows, rows)
            torch.topk(
                scores[:, q_start:q_end],
                k,
                dim=-1,
                sorted=sorted,
                out=(out_scores[:, q_start:q_end], out_pos[:, q_start:q_end]),
            )
        return out_scores, out_pos

    chunks_scores = []
    chunks_pos = []
    for q_start in range(0, rows, chunk_rows):
        q_end = min(q_start + chunk_rows, rows)
        cur_scores, cur_pos = torch.topk(scores[:, q_start:q_end], k, dim=-1, sorted=sorted)
        chunks_scores.append(cur_scores)
        chunks_pos.append(cur_pos)
    return torch.cat(chunks_scores, dim=1), torch.cat(chunks_pos, dim=1)


def _topk_lastdim_exact_cuda(
    scores: torch.Tensor,
    k: int,
    *,
    out_scores: Optional[torch.Tensor] = None,
    out_pos: Optional[torch.Tensor] = None,
    query_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k != 512:
        raise RuntimeError("topk_exact supports k=512 only")
    if _topk_exact_select is None or _is_topk_exact_available is None or not _is_topk_exact_available():
        raise RuntimeError("topk_exact extension unavailable")
    if scores.ndim != 3:
        raise ValueError(f"Expected 3D scores tensor for exact top-k, got shape: {tuple(scores.shape)}")
    if not scores.is_cuda:
        raise ValueError("scores must be CUDA tensor")

    bsz, rows, cols = scores.shape
    if bsz == 0 or rows == 0:
        if out_scores is not None and out_pos is not None:
            return out_scores, out_pos
        return torch.topk(scores, k, dim=-1, sorted=False)
    if cols < k:
        return _topk_lastdim_torch(
            scores,
            k,
            sorted=False,
            out_scores=out_scores,
            out_pos=out_pos,
            query_chunk=query_chunk,
        )

    chunk_rows = _resolve_topk_query_chunk(scores, int(query_chunk))
    if chunk_rows <= 0 or chunk_rows >= rows:
        return _topk_exact_select(scores, out_scores=out_scores, out_pos=out_pos)

    if out_scores is None:
        out_scores = torch.empty((bsz, rows, k), device=scores.device, dtype=scores.dtype)
    if out_pos is None:
        out_pos = torch.empty((bsz, rows, k), device=scores.device, dtype=torch.long)
    for q_start in range(0, rows, chunk_rows):
        q_end = min(q_start + chunk_rows, rows)
        _topk_exact_select(
            scores[:, q_start:q_end],
            out_scores=out_scores[:, q_start:q_end],
            out_pos=out_pos[:, q_start:q_end],
        )
    return out_scores, out_pos


def _segmented_topk_lastdim(
    scores: torch.Tensor,
    k: int,
    *,
    sorted: bool,
    out_scores: Optional[torch.Tensor] = None,
    out_pos: Optional[torch.Tensor] = None,
    query_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 3:
        raise ValueError(f"Expected 3D scores tensor for segmented top-k, got shape: {tuple(scores.shape)}")

    bsz, rows, cols = scores.shape
    if bsz == 0 or rows == 0:
        if out_scores is not None and out_pos is not None:
            return out_scores, out_pos
        return torch.topk(scores, k, dim=-1, sorted=sorted)

    segment_cols = max(k + 1, _env_int("VGGT_INDEXER_TOPK_SEGMENT_COLS", 2048))
    segment_min_cols = max(segment_cols + 1, _env_int("VGGT_INDEXER_TOPK_SEGMENT_MIN_COLS", segment_cols * 2))
    if cols < segment_min_cols or segment_cols >= cols:
        return _topk_lastdim_torch(
            scores,
            k,
            sorted=sorted,
            out_scores=out_scores,
            out_pos=out_pos,
            query_chunk=query_chunk,
        )

    # Large [rows, cols] segmented top-k needs explicit row chunking to avoid
    # allocating giant temporary index buffers.
    row_chunk = max(1, _env_int("VGGT_INDEXER_TOPK_SEGMENT_ROW_CHUNK", 8192))
    if query_chunk > 0:
        row_chunk = min(row_chunk, int(query_chunk))
    row_chunk = min(row_chunk, rows)

    if out_scores is None or out_pos is None:
        out_scores = torch.empty((bsz, rows, k), device=scores.device, dtype=scores.dtype)
        out_pos = torch.empty((bsz, rows, k), device=scores.device, dtype=torch.long)

    debug = _env_flag("VGGT_INDEXER_TOPK_SEGMENT_DEBUG", False)
    use_vectorized = _env_flag("VGGT_INDEXER_TOPK_SEGMENT_VECTORIZE", True)
    num_segments = cols // segment_cols
    tail = cols - num_segments * segment_cols
    if use_vectorized and num_segments >= 2 and tail == 0:
        offsets = (torch.arange(num_segments, device=scores.device, dtype=torch.long) * int(segment_cols)).view(
            1, 1, num_segments, 1
        )
        for q_start in range(0, rows, row_chunk):
            q_end = min(q_start + row_chunk, rows)
            q_rows = q_end - q_start
            q_scores = scores[:, q_start:q_end]
            q_scores_4d = q_scores.reshape(bsz, q_rows, num_segments, segment_cols)
            seg_scores, seg_pos = torch.topk(q_scores_4d, k, dim=-1, sorted=sorted)
            seg_pos = seg_pos + offsets

            cand_scores = seg_scores.reshape(bsz, q_rows, num_segments * k)
            cand_pos = seg_pos.reshape(bsz, q_rows, num_segments * k)

            q_out_scores = out_scores[:, q_start:q_end]
            q_out_pos = out_pos[:, q_start:q_end]
            merge_sel = torch.empty_like(q_out_pos)
            torch.topk(cand_scores, k, dim=-1, sorted=sorted, out=(q_out_scores, merge_sel))
            torch.gather(cand_pos, -1, merge_sel, out=q_out_pos)
    else:
        for q_start in range(0, rows, row_chunk):
            q_end = min(q_start + row_chunk, rows)
            q_scores = scores[:, q_start:q_end]

            cur_scores, cur_pos = torch.topk(q_scores[:, :, :segment_cols], k, dim=-1, sorted=sorted)
            # Ping-pong running buffers so merge updates avoid reallocations.
            run_scores_a = cur_scores
            run_pos_a = cur_pos
            run_scores_b = torch.empty_like(run_scores_a)
            run_pos_b = torch.empty_like(run_pos_a)
            use_a_as_src = True

            candidate_scores = torch.empty((bsz, q_end - q_start, k * 2), device=scores.device, dtype=scores.dtype)
            candidate_pos = torch.empty((bsz, q_end - q_start, k * 2), device=scores.device, dtype=torch.long)
            seg_scores_buf = torch.empty_like(run_scores_a)
            seg_pos_buf = torch.empty_like(run_pos_a)
            merge_sel_buf = torch.empty_like(run_pos_a)

            for start in range(segment_cols, cols, segment_cols):
                end = min(start + segment_cols, cols)
                seg_len = end - start
                src_scores = run_scores_a if use_a_as_src else run_scores_b
                src_pos = run_pos_a if use_a_as_src else run_pos_b
                dst_scores = run_scores_b if use_a_as_src else run_scores_a
                dst_pos = run_pos_b if use_a_as_src else run_pos_a

                if seg_len >= k:
                    torch.topk(
                        q_scores[:, :, start:end],
                        k,
                        dim=-1,
                        sorted=sorted,
                        out=(seg_scores_buf, seg_pos_buf),
                    )
                    if start != 0:
                        seg_pos_buf.add_(int(start))
                    candidate_scores[:, :, :k] = src_scores
                    candidate_scores[:, :, k:] = seg_scores_buf
                    candidate_pos[:, :, :k] = src_pos
                    candidate_pos[:, :, k:] = seg_pos_buf
                    torch.topk(
                        candidate_scores,
                        k,
                        dim=-1,
                        sorted=sorted,
                        out=(dst_scores, merge_sel_buf),
                    )
                    torch.gather(candidate_pos, -1, merge_sel_buf, out=dst_pos)
                else:
                    seg_scores, seg_pos = torch.topk(
                        q_scores[:, :, start:end],
                        seg_len,
                        dim=-1,
                        sorted=sorted,
                    )
                    if start != 0:
                        seg_pos = seg_pos + int(start)
                    keep = k + seg_len
                    cand_scores_view = candidate_scores[:, :, :keep]
                    cand_pos_view = candidate_pos[:, :, :keep]
                    cand_scores_view[:, :, :k] = src_scores
                    cand_scores_view[:, :, k:keep] = seg_scores
                    cand_pos_view[:, :, :k] = src_pos
                    cand_pos_view[:, :, k:keep] = seg_pos
                    torch.topk(
                        cand_scores_view,
                        k,
                        dim=-1,
                        sorted=sorted,
                        out=(dst_scores, merge_sel_buf),
                    )
                    torch.gather(cand_pos_view, -1, merge_sel_buf, out=dst_pos)

                use_a_as_src = not use_a_as_src

            final_scores = run_scores_a if use_a_as_src else run_scores_b
            final_pos = run_pos_a if use_a_as_src else run_pos_b
            out_scores[:, q_start:q_end] = final_scores
            out_pos[:, q_start:q_end] = final_pos

    if debug:
        print(
            f"[sparse_topk_indexer] segmented_topk rows={rows} cols={cols} "
            f"k={k} segment_cols={segment_cols} row_chunk={row_chunk}"
        )
    return out_scores, out_pos


def _topk_lastdim(
    scores: torch.Tensor,
    k: int,
    *,
    sorted: bool,
    out_scores: Optional[torch.Tensor] = None,
    out_pos: Optional[torch.Tensor] = None,
    query_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    use_exact = _env_flag("VGGT_INDEXER_TOPK_EXACT_KERNEL", False)
    if use_exact and int(k) == 512 and scores.is_cuda:
        try:
            return _topk_lastdim_exact_cuda(
                scores,
                k,
                out_scores=out_scores,
                out_pos=out_pos,
                query_chunk=query_chunk,
            )
        except Exception as exc:
            if _env_flag("VGGT_INDEXER_TOPK_EXACT_DEBUG", False):
                print("[sparse_topk_indexer] exact_topk fallback:", repr(exc))

    use_segmented = _env_flag("VGGT_INDEXER_TOPK_SEGMENTED", False)
    if not use_segmented:
        return _topk_lastdim_torch(
            scores,
            k,
            sorted=sorted,
            out_scores=out_scores,
            out_pos=out_pos,
            query_chunk=query_chunk,
        )
    try:
        return _segmented_topk_lastdim(
            scores,
            k,
            sorted=sorted,
            out_scores=out_scores,
            out_pos=out_pos,
            query_chunk=query_chunk,
        )
    except Exception as exc:
        # Keep segmented path as optional acceleration only: any failure
        # immediately falls back to baseline top-k.
        global _SEGMENT_TOPK_ERROR_COUNT
        _SEGMENT_TOPK_ERROR_COUNT += 1
        if _env_flag("VGGT_INDEXER_TOPK_SEGMENT_DEBUG", False) and _SEGMENT_TOPK_ERROR_COUNT <= 8:
            print("[sparse_topk_indexer] segmented_topk fallback:", repr(exc))
        return _topk_lastdim_torch(
            scores,
            k,
            sorted=sorted,
            out_scores=out_scores,
            out_pos=out_pos,
            query_chunk=query_chunk,
        )


def _normalize_view_bias_data(
    view_bias_data: Optional[dict],
    *,
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
    if tuple(q_view_ids.shape) != (bsz, tgt_len):
        raise ValueError(f"q_view_ids shape {tuple(q_view_ids.shape)} mismatches ({bsz}, {tgt_len})")
    if tuple(s_view_ids.shape) != (bsz, src_len):
        raise ValueError(f"s_view_ids shape {tuple(s_view_ids.shape)} mismatches ({bsz}, {src_len})")
    if view_bias.dim() != 3 or int(view_bias.shape[0]) != int(bsz):
        raise ValueError(f"view_bias shape {tuple(view_bias.shape)} mismatches batch {bsz}")
    return dict(
        q_view_ids=q_view_ids.to(device=device, dtype=torch.long),
        s_view_ids=s_view_ids.to(device=device, dtype=torch.long),
        view_bias=view_bias.to(device=device, dtype=dtype),
    )


def _slice_view_bias_data(
    view_bias_data: Optional[dict],
    *,
    q_start: int,
    q_end: int,
) -> Optional[dict]:
    if view_bias_data is None:
        return None
    return dict(
        q_view_ids=view_bias_data["q_view_ids"][:, int(q_start): int(q_end)],
        s_view_ids=view_bias_data["s_view_ids"],
        view_bias=view_bias_data["view_bias"],
    )


def _compute_view_bias_chunk(
    view_bias_data: dict,
    *,
    s_start: int,
    s_end: int,
) -> torch.Tensor:
    q_view_ids = view_bias_data["q_view_ids"]
    s_view_ids = view_bias_data["s_view_ids"][:, int(s_start): int(s_end)]
    view_bias = view_bias_data["view_bias"]
    bsz = int(q_view_ids.shape[0])
    batch_idx = torch.arange(bsz, device=view_bias.device, dtype=torch.long).view(bsz, 1, 1)
    return view_bias[batch_idx, q_view_ids.unsqueeze(-1), s_view_ids.unsqueeze(1)]


def _recompute_selected_topk_scores(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    mask: Optional[torch.Tensor],
    view_bias_data: Optional[dict],
) -> torch.Tensor:
    topk_indices_long = topk_indices.to(torch.long)
    bsz, tgt_len, num_topk = topk_indices_long.shape
    _, _, n_heads, head_dim = q.shape
    src_len = int(k.shape[1])
    batch_offsets = torch.arange(bsz, device=topk_indices_long.device, dtype=torch.long).view(bsz, 1, 1) * src_len
    flat_k = k.reshape(bsz * src_len, n_heads, head_dim)
    flat_w = w.reshape(bsz * src_len, n_heads)
    flat_mask = mask.reshape(bsz * tgt_len, src_len) if mask is not None else None
    flat_s_view_ids = view_bias_data["s_view_ids"].reshape(bsz * src_len) if view_bias_data is not None else None
    q_view_ids = view_bias_data["q_view_ids"] if view_bias_data is not None else None
    view_bias = view_bias_data["view_bias"] if view_bias_data is not None else None
    scores = q.new_empty((bsz, tgt_len, num_topk))

    target_chunk_size = max(1, min(tgt_len, _env_int("VGGT_INDEXER_RECOMPUTE_TGT_CHUNK", 16)))
    topk_chunk_size = max(1, min(num_topk, _env_int("VGGT_INDEXER_RECOMPUTE_TOPK_CHUNK", 128)))
    for t_start in range(0, tgt_len, target_chunk_size):
        t_end = min(t_start + target_chunk_size, tgt_len)
        q_chunk = q[:, t_start:t_end]
        idx_chunk = topk_indices_long[:, t_start:t_end]
        if flat_mask is not None:
            flat_mask_chunk = flat_mask[t_start * bsz:(t_end) * bsz] if bsz == 1 else None
            if flat_mask_chunk is None:
                batch_t_offsets = torch.arange(bsz, device=idx_chunk.device, dtype=torch.long).view(bsz, 1, 1) * tgt_len
                mask_rows = (torch.arange(t_start, t_end, device=idx_chunk.device, dtype=torch.long).view(1, -1, 1) + batch_t_offsets).reshape(-1)
                gathered_mask = flat_mask.index_select(0, mask_rows).view(bsz, t_end - t_start, src_len)
            else:
                gathered_mask = flat_mask_chunk.view(bsz, t_end - t_start, src_len)
        else:
            gathered_mask = None

        q_view_chunk = q_view_ids[:, t_start:t_end] if q_view_ids is not None else None
        batch_idx = torch.arange(bsz, device=view_bias.device, dtype=torch.long).view(bsz, 1, 1) if view_bias is not None else None

        for k_start in range(0, num_topk, topk_chunk_size):
            k_end = min(k_start + topk_chunk_size, num_topk)
            idx_sub = idx_chunk[:, :, k_start:k_end]
            global_idx = idx_sub + batch_offsets
            gathered_k = flat_k.index_select(0, global_idx.reshape(-1)).view(bsz, t_end - t_start, k_end - k_start, n_heads, head_dim)
            gathered_w = flat_w.index_select(0, global_idx.reshape(-1)).view(bsz, t_end - t_start, k_end - k_start, n_heads)

            chunk_scores = torch.einsum("bthd,btkhd->btkh", q_chunk, gathered_k) * float(softmax_scale)
            chunk_scores = torch.relu(chunk_scores)
            chunk_scores = (chunk_scores * gathered_w).sum(dim=-1)

            if gathered_mask is not None:
                chunk_scores = chunk_scores + torch.gather(gathered_mask, 2, idx_sub)

            if flat_s_view_ids is not None and q_view_chunk is not None and view_bias is not None and batch_idx is not None:
                selected_s_view_ids = flat_s_view_ids.index_select(0, global_idx.reshape(-1)).view(bsz, t_end - t_start, k_end - k_start)
                chunk_scores = chunk_scores + view_bias[batch_idx, q_view_chunk.unsqueeze(-1), selected_s_view_ids]

            scores[:, t_start:t_end, k_start:k_end] = chunk_scores

    return scores


def _backward_selected_topk_qkw(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seq_len, n_heads, head_dim = q.shape
    num_topk = topk_indices.shape[-1]

    use_lowp_acc = _env_flag("VGGT_INDEXER_TOPK_BWD_ACC_LOWP", False)
    dq_acc_dtype = q.dtype if use_lowp_acc else torch.float32
    dk_acc_dtype = k.dtype if use_lowp_acc else torch.float32
    dw_acc_dtype = w.dtype if use_lowp_acc else torch.float32
    dq = torch.empty(q.shape, device=q.device, dtype=dq_acc_dtype)
    dk = torch.zeros(k.shape, device=k.device, dtype=dk_acc_dtype)
    dw = torch.zeros(w.shape, device=w.device, dtype=dw_acc_dtype)

    block_d = max(triton.next_power_of_2(head_dim), 16)
    bwd_block_k_default = 16 if use_lowp_acc else 32
    bwd_num_warps_default = 8 if (use_lowp_acc and head_dim <= 64) else (4 if head_dim <= 64 else 8)
    bwd_num_warps = max(1, _env_int("VGGT_INDEXER_TOPK_BWD_NUM_WARPS", bwd_num_warps_default))
    bwd_num_stages = max(1, _env_int("VGGT_INDEXER_TOPK_BWD_NUM_STAGES", 1))
    bwd_block_k = max(8, _env_int("VGGT_INDEXER_TOPK_BWD_BLOCK_K", bwd_block_k_default))
    grid = (bsz * seq_len, n_heads)

    use_fast_topk_bwd = (
        _env_flag("VGGT_INDEXER_TOPK_BWD_FAST_TOPK512", False)
        and _env_flag("VGGT_INDEXER_TOPK_BWD_ASSUME_VALID_IDX", False)
        and int(num_topk) == 512
        and (512 % int(bwd_block_k) == 0)
    )

    if use_fast_topk_bwd:
        _indexer_topk_bwd_fast_topk_kernel[grid](
            q,
            k,
            w,
            topk_indices,
            grad_scores,
            dq,
            dk,
            dw,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            topk_indices.stride(0),
            topk_indices.stride(1),
            topk_indices.stride(2),
            grad_scores.stride(0),
            grad_scores.stride(1),
            grad_scores.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dw.stride(0),
            dw.stride(1),
            dw.stride(2),
            seq_len,
            n_heads,
            head_dim,
            float(softmax_scale),
            NUM_TOPK=512,
            BLOCK_D=block_d,
            BLOCK_K=bwd_block_k,
            num_warps=bwd_num_warps,
            num_stages=bwd_num_stages,
        )
    else:
        _indexer_topk_bwd_kernel[grid](
            q,
            k,
            w,
            topk_indices,
            grad_scores,
            dq,
            dk,
            dw,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            topk_indices.stride(0),
            topk_indices.stride(1),
            topk_indices.stride(2),
            grad_scores.stride(0),
            grad_scores.stride(1),
            grad_scores.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dw.stride(0),
            dw.stride(1),
            dw.stride(2),
            seq_len,
            n_heads,
            head_dim,
            num_topk,
            float(softmax_scale),
            BLOCK_D=block_d,
            BLOCK_K=bwd_block_k,
            num_warps=bwd_num_warps,
            num_stages=bwd_num_stages,
        )

    if dq.dtype != q.dtype:
        dq = dq.to(q.dtype)
    if dk.dtype != k.dtype:
        dk = dk.to(k.dtype)
    if dw.dtype != w.dtype:
        dw = dw.to(w.dtype)
    return dq, dk, dw


def _scatter_view_bias_grad(
    *,
    topk_indices: torch.Tensor,
    grad_scores: torch.Tensor,
    q_view_ids: torch.Tensor,
    s_view_ids: torch.Tensor,
    view_bias_shape: Tuple[int, int, int],
    view_bias_dtype: torch.dtype,
) -> torch.Tensor:
    bsz, tgt_len, _ = topk_indices.shape
    src_len = int(s_view_ids.shape[1])
    _, num_q_views, num_s_views = view_bias_shape
    grad_view_bias = torch.zeros(view_bias_shape, device=grad_scores.device, dtype=torch.float32)
    flat_grad = grad_view_bias.view(-1)
    flat_s_view_ids = s_view_ids.reshape(bsz * src_len)
    src_batch_offsets = (
        torch.arange(bsz, device=topk_indices.device, dtype=torch.long).view(bsz, 1, 1) * src_len
    )
    bias_batch_offsets = (
        torch.arange(bsz, device=topk_indices.device, dtype=torch.long).view(bsz, 1, 1) * (num_q_views * num_s_views)
    )
    tgt_chunk_size = max(1, min(tgt_len, _env_int("VGGT_INDEXER_VIEW_BIAS_BWD_TGT_CHUNK", 64)))

    for t_start in range(0, tgt_len, tgt_chunk_size):
        t_end = min(t_start + tgt_chunk_size, tgt_len)
        idx_chunk = topk_indices[:, t_start:t_end].to(torch.long)
        global_idx = idx_chunk + src_batch_offsets
        selected_s_view_ids = flat_s_view_ids.index_select(0, global_idx.reshape(-1)).view_as(idx_chunk)
        selected_q_view_ids = q_view_ids[:, t_start:t_end].to(torch.long).unsqueeze(-1)
        linear_idx = bias_batch_offsets + selected_q_view_ids * num_s_views + selected_s_view_ids
        flat_grad.scatter_add_(0, linear_idx.reshape(-1), grad_scores[:, t_start:t_end].reshape(-1).to(torch.float32))

    if view_bias_dtype != torch.float32:
        grad_view_bias = grad_view_bias.to(view_bias_dtype)
    return grad_view_bias


def _compute_scores_block_einsum(
    q: torch.Tensor,
    k_block: torch.Tensor,
    w_block: torch.Tensor,
    mask_block: Optional[torch.Tensor],
    view_bias_block: Optional[torch.Tensor],
    softmax_scale: float,
) -> torch.Tensor:
    scores_block = torch.einsum("bthd,bshd->bths", q, k_block) * softmax_scale
    scores_block = torch.relu(scores_block)
    scores_block = scores_block * w_block.permute(0, 2, 1).unsqueeze(1)
    scores_block = scores_block.sum(dim=2)
    if mask_block is not None:
        scores_block = scores_block + mask_block
    if view_bias_block is not None:
        scores_block = scores_block + view_bias_block
    return scores_block


def _compute_scores_block_headwise(
    q: torch.Tensor,
    k_block: torch.Tensor,
    w_block: torch.Tensor,
    mask_block: Optional[torch.Tensor],
    view_bias_block: Optional[torch.Tensor],
    softmax_scale: float,
) -> torch.Tensor:
    # Avoid materializing [B, T, H, K] in forward selection. This lowers memory traffic
    # and launch overhead for the common H=4 indexer setup.
    bsz, tgt_len, n_heads, _ = q.shape
    src_len = k_block.shape[1]
    scores_block = torch.zeros((bsz, tgt_len, src_len), device=q.device, dtype=q.dtype)
    for h in range(n_heads):
        q_h = q[:, :, h, :]
        k_h_t = k_block[:, :, h, :].transpose(1, 2)
        s_h = torch.matmul(q_h, k_h_t) * softmax_scale
        s_h = torch.relu(s_h)
        w_h = w_block[:, :, h].unsqueeze(1)
        scores_block.add_(s_h * w_h)
    if mask_block is not None:
        scores_block = scores_block + mask_block
    if view_bias_block is not None:
        scores_block = scores_block + view_bias_block
    return scores_block


@triton.jit
def _score_block_kernel(
    Q,
    K,
    W,
    OUT,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_wb,
    stride_wt,
    stride_wh,
    stride_ob,
    stride_ot,
    stride_os,
    n_heads,
    head_dim,
    tgt_len,
    src_len,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_H: tl.constexpr,
    USE_TENSOR_CORE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Use int64 offsets for pointer arithmetic to avoid overflow when
    # writing into strided output views at large sequence lengths.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)

    m_mask = offs_m < tgt_len
    n_mask = offs_n < src_len
    d_mask = offs_d < head_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for h in range(MAX_H):
        h_valid = h < n_heads
        h_idx = tl.where(h_valid, h, 0).to(tl.int64)

        q_ptrs = (
            Q
            + pid_b * stride_qb
            + offs_m[:, None] * stride_qt
            + h_idx * stride_qh
            + offs_d[None, :] * stride_qd
        )
        k_ptrs = (
            K
            + pid_b * stride_kb
            + offs_n[:, None] * stride_kt
            + h_idx * stride_kh
            + offs_d[None, :] * stride_kd
        )
        q_raw = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        k_raw = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        if USE_TENSOR_CORE:
            dot = tl.dot(q_raw, tl.trans(k_raw), out_dtype=tl.float32)
        else:
            q_val = q_raw.to(tl.float32)
            k_val = k_raw.to(tl.float32)
            dot = tl.dot(q_val, tl.trans(k_val))
        dot = tl.maximum(dot * softmax_scale, 0.0)

        w_ptrs = W + pid_b * stride_wb + offs_n * stride_wt + h_idx * stride_wh
        w_val = tl.load(w_ptrs, mask=n_mask, other=0.0).to(tl.float32)
        h_scale = tl.where(h_valid, 1.0, 0.0)
        acc += dot * (w_val[None, :] * h_scale)

    out_ptrs = OUT + pid_b * stride_ob + offs_m[:, None] * stride_ot + offs_n[None, :] * stride_os
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _merge_topk_indices_kernel(
    TOPK_PREV,
    NEW_POS,
    PENDING_POS,
    OUT,
    stride_tb,
    stride_tt,
    stride_tk,
    stride_nb,
    stride_nt,
    stride_nk,
    stride_pb,
    stride_pt,
    stride_pk,
    stride_ob,
    stride_ot,
    stride_ok,
    seq_len,
    topk,
    pending_start,
    HAS_PENDING_MAP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    b = pid_row // seq_len
    t = pid_row - b * seq_len
    if t >= seq_len:
        return

    offs_k = pid_blk * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < topk

    new_ptrs = NEW_POS + b * stride_nb + t * stride_nt + offs_k * stride_nk
    pos = tl.load(new_ptrs, mask=k_mask, other=0).to(tl.int32)

    left_mask = pos < topk

    left_pos = tl.where(left_mask, pos, 0)
    left_ptrs = TOPK_PREV + b * stride_tb + t * stride_tt + left_pos * stride_tk
    left_idx = tl.load(left_ptrs, mask=k_mask, other=0).to(tl.int32)

    right_rel = tl.where(left_mask, 0, pos - topk)
    if HAS_PENDING_MAP:
        pending_ptrs = PENDING_POS + b * stride_pb + t * stride_pt + right_rel * stride_pk
        right_rel = tl.load(pending_ptrs, mask=k_mask, other=0).to(tl.int32)

    right_idx = right_rel + pending_start
    out_idx = tl.where(left_mask, left_idx, right_idx)

    out_ptrs = OUT + b * stride_ob + t * stride_ot + offs_k * stride_ok
    tl.store(out_ptrs, out_idx, mask=k_mask)


def _compute_scores_block_triton(
    q: torch.Tensor,
    k_block: torch.Tensor,
    w_block: torch.Tensor,
    mask_block: Optional[torch.Tensor],
    view_bias_block: Optional[torch.Tensor],
    softmax_scale: float,
    total_src_len: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz, tgt_len, n_heads, head_dim = q.shape
    src_len = k_block.shape[1]

    if out is None:
        out = torch.empty((bsz, tgt_len, src_len), device=q.device, dtype=q.dtype)
    else:
        if out.shape != (bsz, tgt_len, src_len):
            raise ValueError(f"Invalid triton score out shape: got {tuple(out.shape)}, expected {(bsz, tgt_len, src_len)}")

    block_m = max(16, _env_int("VGGT_INDEXER_SCORE_BLOCK_M", 64))
    block_n = max(16, _env_int("VGGT_INDEXER_SCORE_BLOCK_N", 64))
    max_h = max(1, _env_int("VGGT_INDEXER_SCORE_MAX_H", max(4, int(n_heads))))
    block_d = max(triton.next_power_of_2(head_dim), 16)
    num_warps = _env_int("VGGT_INDEXER_SCORE_NUM_WARPS", 4 if head_dim <= 64 else 8)
    num_stages = _env_int("VGGT_INDEXER_SCORE_NUM_STAGES", 1)
    use_tensor_core = _env_flag("VGGT_INDEXER_SCORE_USE_TC", False)
    policy_src_len = int(total_src_len) if int(total_src_len) > 0 else int(src_len)
    block_m, block_n, num_warps, num_stages = _resolve_score_kernel_policy(
        src_len=policy_src_len,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    grid = (triton.cdiv(tgt_len, block_m), triton.cdiv(src_len, block_n), bsz)
    _score_block_kernel[grid](
        q,
        k_block,
        w_block,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_block.stride(0),
        k_block.stride(1),
        k_block.stride(2),
        k_block.stride(3),
        w_block.stride(0),
        w_block.stride(1),
        w_block.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        n_heads,
        head_dim,
        tgt_len,
        src_len,
        softmax_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        MAX_H=max_h,
        USE_TENSOR_CORE=use_tensor_core,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if mask_block is not None:
        out.add_(mask_block)
    if view_bias_block is not None:
        out.add_(view_bias_block)
    return out


def _compute_scores_block_dispatch(
    *,
    fwd_mode: str,
    q: torch.Tensor,
    k_block: torch.Tensor,
    w_block: torch.Tensor,
    mask_block: Optional[torch.Tensor],
    view_bias_block: Optional[torch.Tensor],
    softmax_scale: float,
    total_src_len: int,
    use_triton_direct_out: bool,
    direct_out: Optional[torch.Tensor],
) -> torch.Tensor:
    if fwd_mode == "headwise":
        return _compute_scores_block_headwise(
            q=q,
            k_block=k_block,
            w_block=w_block,
            mask_block=mask_block,
            view_bias_block=view_bias_block,
            softmax_scale=softmax_scale,
        )
    if fwd_mode == "triton":
        return _compute_scores_block_triton(
            q=q,
            k_block=k_block,
            w_block=w_block,
            mask_block=mask_block,
            view_bias_block=view_bias_block,
            softmax_scale=softmax_scale,
            total_src_len=total_src_len,
            out=direct_out if use_triton_direct_out else None,
        )
    return _compute_scores_block_einsum(
        q=q,
        k_block=k_block,
        w_block=w_block,
        mask_block=mask_block,
        view_bias_block=view_bias_block,
        softmax_scale=softmax_scale,
    )


def _blockwise_topk_scores_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    mask: Optional[torch.Tensor],
    view_bias_data: Optional[dict],
    topk: int,
    softmax_scale: float,
    block_k: int,
    merge_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    use_score_topk_fused = _env_flag("VGGT_INDEXER_SCORE_TOPK_FUSED_CUDA", False)
    fused_max_src_len = _env_int("VGGT_INDEXER_SCORE_TOPK_FUSED_MAX_SRC_LEN", 0)
    fused_max_tgt_len = _env_int("VGGT_INDEXER_SCORE_TOPK_FUSED_MAX_TGT_LEN", 0)
    fused_allow_shape = (
        (fused_max_src_len <= 0 or int(k.shape[1]) <= fused_max_src_len)
        and (fused_max_tgt_len <= 0 or int(q.shape[1]) <= fused_max_tgt_len)
    )
    if use_score_topk_fused and (not fused_allow_shape) and _env_flag("VGGT_INDEXER_SCORE_TOPK_FUSED_DEBUG", False):
        print(
            "[sparse_topk_indexer] score_topk_fused skip by shape:",
            f"tgt_len={int(q.shape[1])}",
            f"src_len={int(k.shape[1])}",
            f"max_tgt={fused_max_tgt_len}",
            f"max_src={fused_max_src_len}",
        )
    if use_score_topk_fused and int(topk) == 512 and mask is None and q.is_cuda and k.is_cuda and w.is_cuda and fused_allow_shape:
        try:
            if (
                _score_topk_fused_select is not None
                and _is_score_topk_fused_available is not None
                and _is_score_topk_fused_available()
            ):
                fused_scores, fused_pos = _score_topk_fused_select(
                    q,
                    k,
                    w,
                    softmax_scale=float(softmax_scale),
                )
                if fused_pos.dtype != torch.int32:
                    fused_pos = fused_pos.to(torch.int32)
                return fused_pos, fused_scores
        except Exception as exc:
            if _env_flag("VGGT_INDEXER_SCORE_TOPK_FUSED_DEBUG", False):
                print("[sparse_topk_indexer] score_topk_fused fallback:", repr(exc))

    bsz, tgt_len, _, _ = q.shape
    src_len = k.shape[1]
    topk = min(int(topk), src_len)
    block_k = max(1, int(block_k))

    device = q.device
    dtype = q.dtype

    topk_scores = torch.full((bsz, tgt_len, topk), -float("inf"), device=device, dtype=dtype)
    topk_indices = torch.zeros((bsz, tgt_len, topk), device=device, dtype=torch.int32)
    reuse_topk_out = _env_flag("VGGT_INDEXER_TOPK_REUSE_OUT", True)
    if reuse_topk_out:
        topk_scores_buf = _alloc_workspace("topk_scores_buf", (bsz, tgt_len, topk), device=device, dtype=dtype)
        topk_indices_buf = _alloc_workspace("topk_indices_buf", (bsz, tgt_len, topk), device=device, dtype=torch.int32)
        topk_pos_buf = _alloc_workspace("topk_pos_buf", (bsz, tgt_len, topk), device=device, dtype=torch.long)
    else:
        topk_scores_buf = None
        topk_indices_buf = None
        topk_pos_buf = None

    if mask is not None and mask.dtype != dtype:
        mask = mask.to(dtype)

    # `sorted=False` avoids unnecessary ordering work while keeping identical top-k set.
    sort_results = _env_int("VGGT_INDEXER_TOPK_SORTED", 0) != 0
    fwd_mode = _env_str("VGGT_INDEXER_TOPK_FWD_MODE", "einsum").lower()
    use_triton_direct_out = _env_flag("VGGT_INDEXER_SCORE_DIRECT_OUT", True)
    use_reuse_buffer = _env_flag("VGGT_INDEXER_TOPK_REUSE_BUFFER", True)
    use_merge_index_kernel = (
        use_reuse_buffer
        and reuse_topk_out
        and _env_flag("VGGT_INDEXER_TOPK_MERGE_INDEX_KERNEL", True)
        and q.is_cuda
    )
    explicit_want_merge_two_kernel = _env_flag("VGGT_INDEXER_TOPK_MERGE_TWO_KERNEL", False) or _env_flag(
        "VGGT_INDEXER_TOPK_MERGE_TWO_CUDA", False
    )
    want_merge_two_kernel = _resolve_merge_two_policy(
        src_len=int(src_len),
        topk=int(topk),
        explicit_want=explicit_want_merge_two_kernel,
        q_requires_grad=bool(q.requires_grad),
    )
    merge_two_available = (
        want_merge_two_kernel
        and _merge_two_topk_inplace is not None
        and _is_merge_two_topk_available is not None
        and _is_merge_two_topk_available()
    )
    use_merge_two_kernel = (
        use_reuse_buffer
        and reuse_topk_out
        and merge_two_available
        and q.is_cuda
    )
    merge_two_debug = _env_flag("VGGT_INDEXER_MERGE_TWO_DEBUG", False)
    merge_two_attempts = 0
    merge_two_success = 0
    merge_index_block_k = max(32, _env_int("VGGT_INDEXER_TOPK_MERGE_INDEX_BLOCK_K", 256))
    topk_query_chunk = _env_int("VGGT_INDEXER_TOPK_QUERY_CHUNK", 0)
    if merge_blocks <= 0:
        merge_blocks = _env_int("VGGT_INDEXER_TOPK_MERGE_BLOCKS", 1)
    block_k, merge_blocks = _resolve_block_merge_policy(
        seq_len=src_len,
        topk=topk,
        block_k=block_k,
        merge_blocks=merge_blocks,
    )
    block_k, merge_blocks, topk_query_chunk, merge_index_block_k = _resolve_runtime_bucket_policy(
        seq_len=src_len,
        topk=topk,
        block_k=block_k,
        merge_blocks=merge_blocks,
        query_chunk=topk_query_chunk,
        merge_index_block_k=merge_index_block_k,
    )
    block_k = max(1, int(block_k))
    merge_blocks = max(1, int(merge_blocks))
    pending_cap = block_k * merge_blocks
    all_indices = _get_cached_arange(src_len, device=device)
    use_stream_fuse = _env_flag("VGGT_INDEXER_TOPK_STREAM_FUSE", False)

    if use_stream_fuse:
        # Experimental path: per-block score->select->merge without pending score staging.
        # Mathematically exact for top-k set (iterative merge of block top-k).
        candidate_scores = _alloc_workspace("stream_candidate_scores", (bsz, tgt_len, topk * 2), device=device, dtype=dtype)
        candidate_indices = _alloc_workspace(
            "stream_candidate_indices",
            (bsz, tgt_len, topk * 2),
            device=device,
            dtype=torch.int32,
        )
        block_topk_scores = _alloc_workspace("stream_block_topk_scores", (bsz, tgt_len, topk), device=device, dtype=dtype)
        block_topk_pos = _alloc_workspace("stream_block_topk_pos", (bsz, tgt_len, topk), device=device, dtype=torch.long)
        block_topk_indices = _alloc_workspace(
            "stream_block_topk_indices",
            (bsz, tgt_len, topk),
            device=device,
            dtype=torch.int32,
        )

        for start in range(0, src_len, block_k):
            end = min(start + block_k, src_len)
            cur_block = end - start
            k_block = k[:, start:end]
            w_block = w[:, start:end]
            mask_block = None if mask is None else mask[:, :, start:end]
            view_bias_block = None if view_bias_data is None else _compute_view_bias_chunk(
                view_bias_data,
                s_start=start,
                s_end=end,
            )
            scores_block = _compute_scores_block_dispatch(
                fwd_mode=fwd_mode,
                q=q,
                k_block=k_block,
                w_block=w_block,
                mask_block=mask_block,
                view_bias_block=view_bias_block,
                softmax_scale=softmax_scale,
                total_src_len=src_len,
                use_triton_direct_out=False,
                direct_out=None,
            )

            if cur_block > topk:
                _topk_lastdim(
                    scores_block,
                    topk,
                    sorted=sort_results,
                    out_scores=block_topk_scores,
                    out_pos=block_topk_pos,
                    query_chunk=topk_query_chunk,
                )
                block_topk_indices.copy_(block_topk_pos)
                if start != 0:
                    block_topk_indices.add_(int(start))
                block_scores_view = block_topk_scores
                block_indices_view = block_topk_indices
                block_pos_view = block_topk_pos
                block_keep = topk
            else:
                block_scores_view = scores_block
                block_indices_view = all_indices[start:end].view(1, 1, cur_block)
                block_pos_view = None
                block_keep = cur_block

            merged_by_cuda = False
            if (
                use_merge_two_kernel
                and topk == 512
                and block_keep == topk
                and block_pos_view is not None
                and topk_scores_buf is not None
                and topk_indices_buf is not None
            ):
                merged_by_cuda = bool(
                    _merge_two_topk_inplace(
                        topk_scores=topk_scores,
                        topk_indices=topk_indices,
                        pending_scores=block_scores_view,
                        pending_pos=block_pos_view,
                        pending_start=int(start),
                        out_scores=topk_scores_buf,
                        out_indices=topk_indices_buf,
                    )
                )
                if merged_by_cuda:
                    topk_scores, topk_scores_buf = topk_scores_buf, topk_scores
                    topk_indices, topk_indices_buf = topk_indices_buf, topk_indices

            if merged_by_cuda:
                continue

            candidate_len = topk + block_keep
            candidate_scores[:, :, :topk] = topk_scores
            candidate_scores[:, :, topk:candidate_len] = block_scores_view[:, :, :block_keep]
            candidate_indices[:, :, :topk] = topk_indices
            if block_indices_view.shape[0] == 1 and block_indices_view.shape[1] == 1:
                candidate_indices[:, :, topk:candidate_len] = block_indices_view
            else:
                candidate_indices[:, :, topk:candidate_len] = block_indices_view[:, :, :block_keep]

            if reuse_topk_out and topk_scores_buf is not None and topk_pos_buf is not None and topk_indices_buf is not None:
                _topk_lastdim(
                    candidate_scores[:, :, :candidate_len],
                    topk,
                    sorted=sort_results,
                    out_scores=topk_scores_buf,
                    out_pos=topk_pos_buf,
                    query_chunk=topk_query_chunk,
                )
                torch.gather(candidate_indices[:, :, :candidate_len], -1, topk_pos_buf, out=topk_indices_buf)
                topk_scores, topk_scores_buf = topk_scores_buf, topk_scores
                topk_indices, topk_indices_buf = topk_indices_buf, topk_indices
            else:
                new_scores, new_pos = _topk_lastdim(
                    candidate_scores[:, :, :candidate_len],
                    topk,
                    sorted=sort_results,
                    query_chunk=topk_query_chunk,
                )
                topk_scores = new_scores
                topk_indices = torch.gather(candidate_indices[:, :, :candidate_len], -1, new_pos)

        return topk_indices, topk_scores

    if use_reuse_buffer:
        # Two-stage merge:
        # 1) top-k inside pending window
        # 2) merge with running top-k
        # This avoids repeatedly copying giant [topk + pending_cap] candidate buffers.
        candidate_scores = _alloc_workspace("candidate_scores", (bsz, tgt_len, topk * 2), device=device, dtype=dtype)
        candidate_indices = (
            None
            if use_merge_index_kernel
            else _alloc_workspace("candidate_indices", (bsz, tgt_len, topk * 2), device=device, dtype=torch.int32)
        )
        pending_scores_buf = _alloc_workspace("pending_scores_buf", (bsz, tgt_len, pending_cap), device=device, dtype=dtype)
        if reuse_topk_out:
            pending_topk_scores_buf = _alloc_workspace(
                "pending_topk_scores_buf",
                (bsz, tgt_len, topk),
                device=device,
                dtype=dtype,
            )
            pending_topk_pos_buf = _alloc_workspace(
                "pending_topk_pos_buf",
                (bsz, tgt_len, topk),
                device=device,
                dtype=torch.long,
            )
            if _needs_pending_topk_indices_buf(use_merge_index_kernel):
                pending_topk_indices_buf = _alloc_workspace(
                    "pending_topk_indices_buf",
                    (bsz, tgt_len, topk),
                    device=device,
                    dtype=torch.int32,
                )
            else:
                pending_topk_indices_buf = None
        else:
            pending_topk_scores_buf = None
            pending_topk_pos_buf = None
            pending_topk_indices_buf = None
    else:
        candidate_scores = None
        candidate_indices = None
        pending_scores_buf = None
        pending_topk_scores_buf = None
        pending_topk_pos_buf = None
        pending_topk_indices_buf = None

    pending_scores = []
    pending_indices = []
    pending_start = 0
    pending_len = 0

    for start in range(0, src_len, block_k):
        end = min(start + block_k, src_len)
        k_block = k[:, start:end]
        w_block = w[:, start:end]
        mask_block = None if mask is None else mask[:, :, start:end]
        view_bias_block = None if view_bias_data is None else _compute_view_bias_chunk(
            view_bias_data,
            s_start=start,
            s_end=end,
        )
        cur_block = end - start
        direct_out = None
        if use_reuse_buffer and pending_scores_buf is not None:
            if pending_len == 0:
                pending_start = start
            next_len = pending_len + cur_block
            if fwd_mode == "triton" and use_triton_direct_out:
                direct_out = pending_scores_buf[:, :, pending_len:next_len]
        scores_block = _compute_scores_block_dispatch(
            fwd_mode=fwd_mode,
            q=q,
            k_block=k_block,
            w_block=w_block,
            mask_block=mask_block,
            view_bias_block=view_bias_block,
            softmax_scale=softmax_scale,
            total_src_len=src_len,
            use_triton_direct_out=use_triton_direct_out,
            direct_out=direct_out,
        )

        if use_reuse_buffer and pending_scores_buf is not None:
            if direct_out is None:
                pending_scores_buf[:, :, pending_len:next_len] = scores_block
            pending_len = next_len
        else:
            block_indices = all_indices[start:end].view(1, 1, cur_block)
            pending_scores.append(scores_block)
            pending_indices.append(block_indices.expand(bsz, tgt_len, cur_block))
            pending_len += cur_block

        need_flush = (pending_len >= pending_cap) or (end >= src_len)
        if not need_flush:
            continue

        candidate_len = topk + pending_len
        pending_pos_for_kernel = None
        if use_reuse_buffer and candidate_scores is not None and candidate_indices is not None:
            pending_scores_view = pending_scores_buf[:, :, :pending_len]
            pending_index_base = all_indices[pending_start:pending_start + pending_len].view(1, 1, pending_len)
            if pending_len > topk:
                if (
                    reuse_topk_out
                    and pending_topk_scores_buf is not None
                    and pending_topk_pos_buf is not None
                    and pending_topk_indices_buf is not None
                ):
                    _topk_lastdim(
                        pending_scores_view,
                        topk,
                        sorted=sort_results,
                        out_scores=pending_topk_scores_buf,
                        out_pos=pending_topk_pos_buf,
                        query_chunk=topk_query_chunk,
                    )
                    pending_scores_view = pending_topk_scores_buf
                    # Pending window indices are contiguous: [pending_start, pending_start + pending_len).
                    # Map relative top-k positions to absolute indices via offset add (avoid gather).
                    pending_topk_indices_buf.copy_(pending_topk_pos_buf)
                    if pending_start != 0:
                        pending_topk_indices_buf.add_(int(pending_start))
                    pending_indices_view = pending_topk_indices_buf
                else:
                    pending_scores_view, pending_pos = _topk_lastdim(
                        pending_scores_view,
                        topk,
                        sorted=sort_results,
                        query_chunk=topk_query_chunk,
                    )
                    pending_indices_view = pending_pos.to(torch.int32)
                    if pending_start != 0:
                        pending_indices_view = pending_indices_view + int(pending_start)
                pending_keep = topk
            else:
                pending_indices_view = pending_index_base
                pending_keep = pending_len

            candidate_len = topk + pending_keep
            candidate_scores[:, :, :topk] = topk_scores
            candidate_scores[:, :, topk:candidate_len] = pending_scores_view[:, :, :pending_keep]
            candidate_indices[:, :, :topk] = topk_indices
            if pending_indices_view.shape[0] == 1 and pending_indices_view.shape[1] == 1:
                candidate_indices[:, :, topk:candidate_len] = pending_indices_view
            else:
                candidate_indices[:, :, topk:candidate_len] = pending_indices_view[:, :, :pending_keep]
            scores_view = candidate_scores[:, :, :candidate_len]
            indices_view = candidate_indices[:, :, :candidate_len]
        elif use_reuse_buffer and candidate_scores is not None:
            pending_scores_view = pending_scores_buf[:, :, :pending_len]
            if pending_len > topk:
                if reuse_topk_out and pending_topk_scores_buf is not None and pending_topk_pos_buf is not None:
                    _topk_lastdim(
                        pending_scores_view,
                        topk,
                        sorted=sort_results,
                        out_scores=pending_topk_scores_buf,
                        out_pos=pending_topk_pos_buf,
                        query_chunk=topk_query_chunk,
                    )
                    pending_scores_view = pending_topk_scores_buf
                    pending_pos_for_kernel = pending_topk_pos_buf
                else:
                    pending_scores_view, pending_pos = _topk_lastdim(
                        pending_scores_view,
                        topk,
                        sorted=sort_results,
                        query_chunk=topk_query_chunk,
                    )
                    pending_pos_for_kernel = pending_pos
                pending_keep = topk
            else:
                pending_keep = pending_len
            candidate_len = topk + pending_keep
            candidate_scores[:, :, :topk] = topk_scores
            candidate_scores[:, :, topk:candidate_len] = pending_scores_view[:, :, :pending_keep]
            scores_view = candidate_scores[:, :, :candidate_len]
            indices_view = None
        else:
            if pending_scores:
                pending_scores_cat = torch.cat(pending_scores, dim=-1)
                pending_indices_cat = torch.cat(pending_indices, dim=-1)
                scores_view = torch.cat([topk_scores, pending_scores_cat], dim=-1)
                indices_view = torch.cat([topk_indices, pending_indices_cat], dim=-1)
            else:
                scores_view = topk_scores
                indices_view = topk_indices

        if reuse_topk_out and topk_scores_buf is not None and topk_pos_buf is not None and topk_indices_buf is not None:
            use_merge_two = (
                use_merge_two_kernel
                and use_reuse_buffer
                and candidate_scores is not None
                and candidate_indices is None
                and pending_len > topk
                and pending_pos_for_kernel is not None
                and topk == 512
            )

            merged_by_cuda = False
            if use_merge_two:
                merge_two_attempts += 1
                pending_scores_for_merge = pending_scores_view[:, :, :topk]
                if pending_scores_for_merge.is_contiguous():
                    merged_by_cuda = bool(
                        _merge_two_topk_inplace(
                            topk_scores=topk_scores,
                            topk_indices=topk_indices,
                            pending_scores=pending_scores_for_merge,
                            pending_pos=pending_pos_for_kernel,
                            pending_start=int(pending_start),
                            out_scores=topk_scores_buf,
                            out_indices=topk_indices_buf,
                        )
                    )
                elif merge_two_debug:
                    print("[sparse_topk_indexer] skip merge_two due non-contiguous pending_scores")
                if merged_by_cuda:
                    merge_two_success += 1

            if not merged_by_cuda:
                _topk_lastdim(
                    scores_view,
                    topk,
                    sorted=sort_results,
                    out_scores=topk_scores_buf,
                    out_pos=topk_pos_buf,
                    query_chunk=topk_query_chunk,
                )
                if use_reuse_buffer and candidate_scores is not None and candidate_indices is None:
                    has_pending_map = pending_len > topk and pending_pos_for_kernel is not None
                    pending_pos_tensor = pending_pos_for_kernel if has_pending_map else topk_pos_buf
                    grid = (bsz * tgt_len, triton.cdiv(topk, merge_index_block_k))
                    _merge_topk_indices_kernel[grid](
                        topk_indices,
                        topk_pos_buf,
                        pending_pos_tensor,
                        topk_indices_buf,
                        topk_indices.stride(0),
                        topk_indices.stride(1),
                        topk_indices.stride(2),
                        topk_pos_buf.stride(0),
                        topk_pos_buf.stride(1),
                        topk_pos_buf.stride(2),
                        pending_pos_tensor.stride(0),
                        pending_pos_tensor.stride(1),
                        pending_pos_tensor.stride(2),
                        topk_indices_buf.stride(0),
                        topk_indices_buf.stride(1),
                        topk_indices_buf.stride(2),
                        tgt_len,
                        topk,
                        int(pending_start),
                        HAS_PENDING_MAP=has_pending_map,
                        BLOCK_K=merge_index_block_k,
                        num_warps=4,
                        num_stages=1,
                    )
                else:
                    torch.gather(indices_view, -1, topk_pos_buf, out=topk_indices_buf)
            topk_scores, topk_scores_buf = topk_scores_buf, topk_scores
            topk_indices, topk_indices_buf = topk_indices_buf, topk_indices
        else:
            new_scores, new_pos = _topk_lastdim(
                scores_view,
                topk,
                sorted=sort_results,
                query_chunk=topk_query_chunk,
            )
            topk_scores = new_scores
            topk_indices = torch.gather(indices_view, -1, new_pos)
        pending_start = 0
        pending_len = 0
        pending_scores.clear()
        pending_indices.clear()

    if merge_two_debug and use_merge_two_kernel:
        print(
            f"[sparse_topk_indexer] merge_two attempts={merge_two_attempts} "
            f"success={merge_two_success}"
        )

    return topk_indices, topk_scores


def _blockwise_topk_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    mask: Optional[torch.Tensor],
    view_bias_data: Optional[dict],
    topk: int,
    softmax_scale: float,
    block_k: int,
    merge_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, tgt_len, _, _ = q.shape
    src_len = k.shape[1]
    outer_query_chunk = _resolve_outer_query_chunk_policy(
        tgt_len=int(tgt_len),
        src_len=int(src_len),
        topk=int(topk),
        user_chunk=_env_int("VGGT_INDEXER_TOPK_OUTER_QUERY_CHUNK", 0),
    )
    if outer_query_chunk <= 0:
        return _blockwise_topk_scores_impl(
            q=q,
            k=k,
            w=w,
            mask=mask,
            view_bias_data=view_bias_data,
            topk=topk,
            softmax_scale=softmax_scale,
            block_k=block_k,
            merge_blocks=merge_blocks,
        )

    outer_query_chunk = max(1, min(int(outer_query_chunk), int(tgt_len)))
    if outer_query_chunk >= tgt_len:
        return _blockwise_topk_scores_impl(
            q=q,
            k=k,
            w=w,
            mask=mask,
            view_bias_data=view_bias_data,
            topk=topk,
            softmax_scale=softmax_scale,
            block_k=block_k,
            merge_blocks=merge_blocks,
        )

    topk = min(int(topk), int(k.shape[1]))
    out_scores = torch.empty((bsz, tgt_len, topk), device=q.device, dtype=q.dtype)
    out_indices = torch.empty((bsz, tgt_len, topk), device=q.device, dtype=torch.int32)
    for q_start in range(0, tgt_len, outer_query_chunk):
        q_end = min(q_start + outer_query_chunk, tgt_len)
        mask_chunk = None if mask is None else mask[:, q_start:q_end, :]
        sub_indices, sub_scores = _blockwise_topk_scores_impl(
            q=q[:, q_start:q_end],
            k=k,
            w=w,
            mask=mask_chunk,
            view_bias_data=_slice_view_bias_data(view_bias_data, q_start=q_start, q_end=q_end),
            topk=topk,
            softmax_scale=softmax_scale,
            block_k=block_k,
            merge_blocks=merge_blocks,
        )
        out_indices[:, q_start:q_end] = sub_indices
        out_scores[:, q_start:q_end] = sub_scores
    return out_indices, out_scores


@triton.jit
def _indexer_topk_bwd_kernel(
    Q,
    K,
    W,
    TOPK_IDX,
    G,
    DQ,
    DK,
    DW,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_wb,
    stride_wt,
    stride_wh,
    stride_ib,
    stride_it,
    stride_ik,
    stride_gb,
    stride_gt,
    stride_gk,
    stride_dqb,
    stride_dqt,
    stride_dqh,
    stride_dqd,
    stride_dkb,
    stride_dkt,
    stride_dkh,
    stride_dkd,
    stride_dwb,
    stride_dwt,
    stride_dwh,
    seq_len,
    n_heads,
    head_dim,
    num_topk,
    softmax_scale,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = pid_q // seq_len
    t = pid_q - b * seq_len
    if t >= seq_len:
        return
    h = pid_h
    if h >= n_heads:
        return

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    q_ptrs = Q + b * stride_qb + t * stride_qt + h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    dq_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for start_k in range(0, num_topk, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)
        k_mask = (start_k + offs_k) < num_topk

        idx_ptrs = TOPK_IDX + b * stride_ib + t * stride_it + (start_k + offs_k) * stride_ik
        idx = tl.load(idx_ptrs, mask=k_mask, other=0).to(tl.int32)
        idx_safe = tl.where(k_mask, idx, 0)

        g_ptrs = G + b * stride_gb + t * stride_gt + (start_k + offs_k) * stride_gk
        g = tl.load(g_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        w_ptrs = W + b * stride_wb + idx_safe * stride_wt + h * stride_wh
        w_val = tl.load(w_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        k_ptrs = K + b * stride_kb + idx_safe[:, None] * stride_kt + h * stride_kh + offs_d[None] * stride_kd
        k_val = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None], other=0.0).to(tl.float32)

        z = tl.sum(k_val * q[None, :], axis=1) * softmax_scale
        relu_mask = z > 0
        relu_z = tl.where(relu_mask, z, 0.0)

        relu_mask_f = tl.where(relu_mask, 1.0, 0.0)
        factor = g * w_val * relu_mask_f * softmax_scale

        dq_acc += tl.sum(factor[:, None] * k_val, axis=0)

        dk_ptrs = DK + b * stride_dkb + idx_safe[:, None] * stride_dkt + h * stride_dkh + offs_d[None] * stride_dkd
        tl.atomic_add(dk_ptrs, factor[:, None] * q[None, :], mask=k_mask[:, None] & d_mask[None])

        dw_ptrs = DW + b * stride_dwb + idx_safe * stride_dwt + h * stride_dwh
        tl.atomic_add(dw_ptrs, g * relu_z, mask=k_mask)

    dq_ptrs = DQ + b * stride_dqb + t * stride_dqt + h * stride_dqh + offs_d * stride_dqd
    tl.store(dq_ptrs, dq_acc, mask=d_mask)


@triton.jit
def _indexer_topk_bwd_fast_topk_kernel(
    Q,
    K,
    W,
    TOPK_IDX,
    G,
    DQ,
    DK,
    DW,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_wb,
    stride_wt,
    stride_wh,
    stride_ib,
    stride_it,
    stride_ik,
    stride_gb,
    stride_gt,
    stride_gk,
    stride_dqb,
    stride_dqt,
    stride_dqh,
    stride_dqd,
    stride_dkb,
    stride_dkt,
    stride_dkh,
    stride_dkd,
    stride_dwb,
    stride_dwt,
    stride_dwh,
    seq_len,
    n_heads,
    head_dim,
    softmax_scale,
    NUM_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Fast path for fixed topk (e.g. 512): removes per-iteration topk tail masks.
    # Requires topk indices to be valid [0, seq_len).
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = pid_q // seq_len
    t = pid_q - b * seq_len
    if t >= seq_len:
        return
    h = pid_h
    if h >= n_heads:
        return

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    q_ptrs = Q + b * stride_qb + t * stride_qt + h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    dq_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for start_k in range(0, NUM_TOPK, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)

        idx_ptrs = TOPK_IDX + b * stride_ib + t * stride_it + offs_k * stride_ik
        idx = tl.load(idx_ptrs).to(tl.int32)

        g_ptrs = G + b * stride_gb + t * stride_gt + offs_k * stride_gk
        g = tl.load(g_ptrs).to(tl.float32)

        w_ptrs = W + b * stride_wb + idx * stride_wt + h * stride_wh
        w_val = tl.load(w_ptrs).to(tl.float32)

        k_ptrs = K + b * stride_kb + idx[:, None] * stride_kt + h * stride_kh + offs_d[None] * stride_kd
        k_val = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)

        z = tl.sum(k_val * q[None, :], axis=1) * softmax_scale
        relu_mask = z > 0
        relu_z = tl.where(relu_mask, z, 0.0)

        relu_mask_f = tl.where(relu_mask, 1.0, 0.0)
        factor = g * w_val * relu_mask_f * softmax_scale

        dq_acc += tl.sum(factor[:, None] * k_val, axis=0)

        dk_ptrs = DK + b * stride_dkb + idx[:, None] * stride_dkt + h * stride_dkh + offs_d[None] * stride_dkd
        tl.atomic_add(dk_ptrs, factor[:, None] * q[None, :], mask=d_mask[None, :])

        dw_ptrs = DW + b * stride_dwb + idx * stride_dwt + h * stride_dwh
        tl.atomic_add(dw_ptrs, g * relu_z)

    dq_ptrs = DQ + b * stride_dqb + t * stride_dqt + h * stride_dqh + offs_d * stride_dqd
    tl.store(dq_ptrs, dq_acc, mask=d_mask)


class SparseTopkIndexerFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        mask: Optional[torch.Tensor],
        topk: int,
        softmax_scale: Optional[float],
        block_k: int,
        merge_blocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.contiguous()
        k = k.contiguous()
        w = w.contiguous()
        if mask is not None and mask.numel() == 0:
            mask = None

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        if block_k <= 0:
            block_k = _env_int("VGGT_INDEXER_TOPK_BLOCK", 256)

        with torch.no_grad():
            topk_indices, topk_scores = _blockwise_topk_scores(
                q=q,
                k=k,
                w=w,
                mask=mask,
                view_bias_data=None,
                topk=topk,
                softmax_scale=softmax_scale,
                block_k=block_k,
                merge_blocks=int(merge_blocks),
            )

        ctx.save_for_backward(q, k, w, topk_indices)
        ctx.softmax_scale = float(softmax_scale)
        ctx.block_k = int(block_k)
        return topk_indices, topk_scores

    @staticmethod
    def backward(ctx, grad_indices: Optional[torch.Tensor], grad_scores: Optional[torch.Tensor]):
        q, k, w, topk_indices = ctx.saved_tensors
        if grad_scores is None:
            return None, None, None, None, None, None, None, None

        if grad_scores.stride(-1) != 1:
            grad_scores = grad_scores.contiguous()
        dq, dk, dw = _backward_selected_topk_qkw(
            q=q,
            k=k,
            w=w,
            topk_indices=topk_indices,
            grad_scores=grad_scores,
            softmax_scale=ctx.softmax_scale,
        )
        return dq, dk, dw, None, None, None, None, None


class SparseTopkIndexerViewBiasFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        mask: Optional[torch.Tensor],
        q_view_ids: torch.Tensor,
        s_view_ids: torch.Tensor,
        view_bias: torch.Tensor,
        topk: int,
        softmax_scale: Optional[float],
        block_k: int,
        merge_blocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.contiguous()
        k = k.contiguous()
        w = w.contiguous()
        q_view_ids = q_view_ids.contiguous()
        s_view_ids = s_view_ids.contiguous()
        view_bias = view_bias.contiguous()
        if mask is not None and mask.numel() == 0:
            mask = None

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        if block_k <= 0:
            block_k = _env_int("VGGT_INDEXER_TOPK_BLOCK", 256)

        norm_view_bias = dict(q_view_ids=q_view_ids, s_view_ids=s_view_ids, view_bias=view_bias)
        with torch.no_grad():
            topk_indices, topk_scores = _blockwise_topk_scores(
                q=q,
                k=k,
                w=w,
                mask=mask,
                view_bias_data=norm_view_bias,
                topk=topk,
                softmax_scale=float(softmax_scale),
                block_k=block_k,
                merge_blocks=int(merge_blocks),
            )

        ctx.save_for_backward(q, k, w, topk_indices, q_view_ids, s_view_ids)
        ctx.softmax_scale = float(softmax_scale)
        ctx.view_bias_shape = tuple(int(x) for x in view_bias.shape)
        ctx.view_bias_dtype = view_bias.dtype
        return topk_indices, topk_scores

    @staticmethod
    def backward(ctx, grad_indices: Optional[torch.Tensor], grad_scores: Optional[torch.Tensor]):
        q, k, w, topk_indices, q_view_ids, s_view_ids = ctx.saved_tensors
        if grad_scores is None:
            return None, None, None, None, None, None, None, None, None, None, None

        if grad_scores.stride(-1) != 1:
            grad_scores = grad_scores.contiguous()

        dq, dk, dw = _backward_selected_topk_qkw(
            q=q,
            k=k,
            w=w,
            topk_indices=topk_indices,
            grad_scores=grad_scores,
            softmax_scale=ctx.softmax_scale,
        )
        grad_view_bias = _scatter_view_bias_grad(
            topk_indices=topk_indices,
            grad_scores=grad_scores,
            q_view_ids=q_view_ids,
            s_view_ids=s_view_ids,
            view_bias_shape=ctx.view_bias_shape,
            view_bias_dtype=ctx.view_bias_dtype,
        )
        return dq, dk, dw, None, None, None, grad_view_bias, None, None, None, None


def sparse_topk_indexer_func(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    mask: Optional[torch.Tensor],
    topk: int,
    softmax_scale: Optional[float],
    block_k: int,
    merge_blocks: int = 0,
    view_bias_data: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mask is not None and mask.numel() == 0:
        mask = None

    if view_bias_data is None:
        return SparseTopkIndexerFunc.apply(q, k, w, mask, topk, softmax_scale, block_k, merge_blocks)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if block_k <= 0:
        block_k = _env_int("VGGT_INDEXER_TOPK_BLOCK", 256)

    norm_view_bias = _normalize_view_bias_data(
        view_bias_data,
        bsz=int(q.shape[0]),
        tgt_len=int(q.shape[1]),
        src_len=int(k.shape[1]),
        device=q.device,
        dtype=q.dtype,
    )
    if not q.is_cuda:
        with torch.no_grad():
            topk_indices, _ = _blockwise_topk_scores(
                q=q.contiguous(),
                k=k.contiguous(),
                w=w.contiguous(),
                mask=mask,
                view_bias_data=norm_view_bias,
                topk=topk,
                softmax_scale=float(softmax_scale),
                block_k=block_k,
                merge_blocks=int(merge_blocks),
            )
        topk_scores = _recompute_selected_topk_scores(
            q=q,
            k=k,
            w=w,
            topk_indices=topk_indices,
            softmax_scale=float(softmax_scale),
            mask=mask,
            view_bias_data=norm_view_bias,
        )
        return topk_indices, topk_scores
    return SparseTopkIndexerViewBiasFunc.apply(
        q,
        k,
        w,
        mask,
        norm_view_bias["q_view_ids"],
        norm_view_bias["s_view_ids"],
        norm_view_bias["view_bias"],
        topk,
        float(softmax_scale),
        block_k,
        int(merge_blocks),
    )
