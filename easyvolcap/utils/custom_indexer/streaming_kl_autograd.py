import os
from typing import Optional

import torch
try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

try:
    from .sparse_topk_indexer import _compute_scores_block_triton, _resolve_score_kernel_policy
except Exception:
    _compute_scores_block_triton = None
    _resolve_score_kernel_policy = None


_SCORE_MODE_FALLBACK_COUNT = 0
_FLASH_FWD_FALLBACK_COUNT = 0


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    return value if value else default


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    try:
        return int(value) if value else int(default)
    except ValueError:
        return int(default)


def _resolve_score_mode(q: torch.Tensor) -> str:
    mode = _env_str("VGGT_STREAMING_KL_SCORE_MODE", "auto").strip().lower()
    if mode in ("legacy", "torch", "einsum"):
        return "legacy"
    if mode == "triton":
        return "triton"
    if mode == "auto":
        if _compute_scores_block_triton is not None and q.is_cuda:
            return "triton"
        return "legacy"
    return "legacy"


def _resolve_kl_forward_mode(q: torch.Tensor) -> str:
    mode = _env_str("VGGT_STREAMING_KL_FWD_MODE", "score").strip().lower()
    if mode in ("score", "default", "legacy_score"):
        return "score"
    if mode in ("flash", "fused", "triton_flash"):
        if triton is not None and tl is not None and q.is_cuda:
            return "flash"
        return "score"
    if mode == "auto":
        if triton is not None and tl is not None and q.is_cuda:
            return "flash"
        return "score"
    return "score"


def _resolve_kl_objective() -> str:
    mode = _env_str("VGGT_STREAMING_KL_OBJECTIVE", "kl").strip().lower()
    if mode in ("ce", "cross_entropy", "xent", "flash_ce"):
        return "ce"
    return "kl"


def _split_p_stats_from_flash_kernel() -> bool:
    return _env_flag("VGGT_STREAMING_KL_SPLIT_P_STATS")


def _use_flash_block_ptr_fwd() -> bool:
    return _env_flag("VGGT_STREAMING_KL_FLASH_BLOCK_PTR_FWD")


def _use_h2_flash_fwd_kernel(q: torch.Tensor) -> bool:
    if triton is None or tl is None or (not q.is_cuda):
        return False
    if int(q.shape[2]) > 4 or int(q.shape[3]) > 64:
        return False
    mode = _env_str("VGGT_STREAMING_KL_H2_FWD_KERNEL", "0").strip().lower()
    return mode not in ("0", "false", "off", "no")


def _use_h4_flash_kernels(q: torch.Tensor, kind: str = "") -> bool:
    if triton is None or tl is None or (not q.is_cuda):
        return False
    if int(q.shape[2]) > 4 or int(q.shape[3]) > 64:
        return False
    # Head-specialized kernels remain opt-in only. On dev014 they looked promising
    # in tiny-shape smoke tests, but long-run 16384-token benchmarks still favored
    # the generic flash kernels as the default path.
    suffix = kind.strip().upper()
    default = "0"
    mode = _env_str(f"VGGT_STREAMING_KL_H4_{suffix}_KERNEL", _env_str("VGGT_STREAMING_KL_H4_KERNEL", default)).strip().lower()
    return mode not in ("0", "false", "off", "no")


if triton is not None and tl is not None:
    @triton.jit
    def _streaming_kl_forward_kernel(
        Q,
        K,
        W,
        P,
        MASK,
        ROW_SUM_P,
        P_LOG_P,
        EXPECTED_SCORE,
        LOG_DENOM,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_wb,
        stride_ws,
        stride_wh,
        stride_pb,
        stride_pt,
        stride_ps,
        stride_mb,
        stride_mt,
        stride_ms,
        stride_rsb,
        stride_rst,
        stride_plb,
        stride_plt,
        stride_esb,
        stride_est,
        stride_ldb,
        stride_ldt,
        n_heads,
        head_dim,
        tgt_len,
        src_len,
        softmax_scale,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_H: tl.constexpr,
        USE_TENSOR_CORE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        OBJECTIVE_IS_CE: tl.constexpr,
        TRACK_P_STATS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        neg_inf = float("-inf")
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        m_mask = offs_m < tgt_len

        if TRACK_P_STATS:
            row_sum_p = tl.zeros((BLOCK_M,), dtype=tl.float32)
            p_log_p = tl.zeros((BLOCK_M,), dtype=tl.float32)
        expected_score = tl.zeros((BLOCK_M,), dtype=tl.float32)
        row_max = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
        row_l = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for n_start in range(0, src_len, BLOCK_N):
            offs_n = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
            n_mask = offs_n < src_len
            valid = m_mask[:, None] & n_mask[None, :]
            score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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
                    + offs_n[:, None] * stride_ks
                    + h_idx * stride_kh
                    + offs_d[None, :] * stride_kd
                )
                d_mask = offs_d < head_dim
                q_raw = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
                k_raw = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                if USE_TENSOR_CORE:
                    dot = tl.dot(q_raw, tl.trans(k_raw), out_dtype=tl.float32)
                else:
                    dot = tl.dot(q_raw.to(tl.float32), tl.trans(k_raw.to(tl.float32)))
                dot = tl.maximum(dot * softmax_scale, 0.0)

                w_ptrs = W + pid_b * stride_wb + offs_n * stride_ws + h_idx * stride_wh
                w_val = tl.load(w_ptrs, mask=n_mask, other=0.0).to(tl.float32)
                h_scale = tl.where(h_valid, 1.0, 0.0)
                score += dot * (w_val[None, :] * h_scale)

            if HAS_MASK:
                mask_ptrs = MASK + pid_b * stride_mb + offs_m[:, None] * stride_mt + offs_n[None, :] * stride_ms
                mask_val = tl.load(mask_ptrs, mask=valid, other=neg_inf).to(tl.float32)
                score += mask_val

            p_ptrs = P + pid_b * stride_pb + offs_m[:, None] * stride_pt + offs_n[None, :] * stride_ps
            p_val = tl.load(p_ptrs, mask=valid, other=0.0).to(tl.float32)
            if TRACK_P_STATS:
                row_sum_p += tl.sum(p_val, axis=1)
                p_log_p += tl.sum(p_val * tl.log(p_val + eps), axis=1)
            expected_score += tl.sum(p_val * score, axis=1)

            score = tl.where(valid, score, neg_inf)
            block_max = tl.max(score, axis=1)
            block_max_safe = tl.where(block_max == neg_inf, 0.0, block_max)
            block_exp = tl.where(valid, tl.exp(score - block_max_safe[:, None]), 0.0)
            block_l = tl.where(block_max == neg_inf, 0.0, tl.sum(block_exp, axis=1))

            new_max = tl.maximum(row_max, block_max)
            new_max_safe = tl.where(new_max == neg_inf, 0.0, new_max)
            row_max_safe = tl.where(row_max == neg_inf, 0.0, row_max)
            old_scale = tl.where(row_max == neg_inf, 0.0, tl.exp(row_max_safe - new_max_safe))
            block_scale = tl.where(block_max == neg_inf, 0.0, tl.exp(block_max_safe - new_max_safe))
            row_l = row_l * old_scale + block_l * block_scale
            row_max = new_max

        log_denom = tl.where(row_l > 0, row_max + tl.log(row_l), neg_inf)

        row_ptrs = pid_b * stride_rsb + offs_m * stride_rst
        if TRACK_P_STATS:
            tl.store(ROW_SUM_P + row_ptrs, row_sum_p, mask=m_mask)
            tl.store(P_LOG_P + pid_b * stride_plb + offs_m * stride_plt, p_log_p, mask=m_mask)
        tl.store(EXPECTED_SCORE + pid_b * stride_esb + offs_m * stride_est, expected_score, mask=m_mask)
        tl.store(LOG_DENOM + pid_b * stride_ldb + offs_m * stride_ldt, log_denom, mask=m_mask)


    @triton.jit
    def _streaming_kl_forward_kernel_h2(
        Q,
        K,
        W,
        P,
        MASK,
        ROW_SUM_P,
        P_LOG_P,
        EXPECTED_SCORE,
        LOG_DENOM,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_wb,
        stride_ws,
        stride_wh,
        stride_pb,
        stride_pt,
        stride_ps,
        stride_mb,
        stride_mt,
        stride_ms,
        stride_rsb,
        stride_rst,
        stride_plb,
        stride_plt,
        stride_esb,
        stride_est,
        stride_ldb,
        stride_ldt,
        n_heads,
        head_dim,
        tgt_len,
        src_len,
        softmax_scale,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_TENSOR_CORE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        OBJECTIVE_IS_CE: tl.constexpr,
        TRACK_P_STATS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        neg_inf = float("-inf")
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        m_mask = offs_m < tgt_len
        d_mask = offs_d < head_dim

        h1_valid = n_heads > 1
        h2_valid = n_heads > 2
        h3_valid = n_heads > 3
        h1_idx = tl.where(h1_valid, 1, 0).to(tl.int64)
        h2_idx = tl.where(h2_valid, 2, 0).to(tl.int64)
        h3_idx = tl.where(h3_valid, 3, 0).to(tl.int64)
        h1_scale = tl.where(h1_valid, 1.0, 0.0)
        h2_scale = tl.where(h2_valid, 1.0, 0.0)
        h3_scale = tl.where(h3_valid, 1.0, 0.0)

        q0_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q1_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h1_idx * stride_qh + offs_d[None, :] * stride_qd
        q2_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h2_idx * stride_qh + offs_d[None, :] * stride_qd
        q3_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h3_idx * stride_qh + offs_d[None, :] * stride_qd
        q0 = tl.load(q0_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q1 = tl.load(q1_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q2 = tl.load(q2_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q3 = tl.load(q3_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

        if TRACK_P_STATS:
            row_sum_p = tl.zeros((BLOCK_M,), dtype=tl.float32)
            p_log_p = tl.zeros((BLOCK_M,), dtype=tl.float32)
        expected_score = tl.zeros((BLOCK_M,), dtype=tl.float32)
        row_max = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
        row_l = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for n_start in range(0, src_len, BLOCK_N):
            offs_n = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
            n_mask = offs_n < src_len
            valid = m_mask[:, None] & n_mask[None, :]
            score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            k0_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k1_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h1_idx * stride_kh + offs_d[None, :] * stride_kd
            k0 = tl.load(k0_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            k1 = tl.load(k1_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            if USE_TENSOR_CORE:
                pre0 = tl.dot(q0, tl.trans(k0), out_dtype=tl.float32) * softmax_scale
                pre1 = tl.dot(q1, tl.trans(k1), out_dtype=tl.float32) * softmax_scale
            else:
                pre0 = tl.dot(q0.to(tl.float32), tl.trans(k0.to(tl.float32))) * softmax_scale
                pre1 = tl.dot(q1.to(tl.float32), tl.trans(k1.to(tl.float32))) * softmax_scale
            relu0 = tl.maximum(pre0, 0.0)
            relu1 = tl.maximum(pre1, 0.0)
            w0 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws, mask=n_mask, other=0.0).to(tl.float32)
            w1 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h1_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)
            score += relu0 * w0[None, :]
            score += relu1 * (w1[None, :] * h1_scale)

            k2_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h2_idx * stride_kh + offs_d[None, :] * stride_kd
            k3_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h3_idx * stride_kh + offs_d[None, :] * stride_kd
            k2 = tl.load(k2_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            k3 = tl.load(k3_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            if USE_TENSOR_CORE:
                pre2 = tl.dot(q2, tl.trans(k2), out_dtype=tl.float32) * softmax_scale
                pre3 = tl.dot(q3, tl.trans(k3), out_dtype=tl.float32) * softmax_scale
            else:
                pre2 = tl.dot(q2.to(tl.float32), tl.trans(k2.to(tl.float32))) * softmax_scale
                pre3 = tl.dot(q3.to(tl.float32), tl.trans(k3.to(tl.float32))) * softmax_scale
            relu2 = tl.maximum(pre2, 0.0)
            relu3 = tl.maximum(pre3, 0.0)
            w2 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h2_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)
            w3 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h3_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)
            score += relu2 * (w2[None, :] * h2_scale)
            score += relu3 * (w3[None, :] * h3_scale)

            if HAS_MASK:
                mask_ptrs = MASK + pid_b * stride_mb + offs_m[:, None] * stride_mt + offs_n[None, :] * stride_ms
                score += tl.load(mask_ptrs, mask=valid, other=neg_inf).to(tl.float32)

            p_ptrs = P + pid_b * stride_pb + offs_m[:, None] * stride_pt + offs_n[None, :] * stride_ps
            p_val = tl.load(p_ptrs, mask=valid, other=0.0).to(tl.float32)
            if TRACK_P_STATS:
                row_sum_p += tl.sum(p_val, axis=1)
                p_log_p += tl.sum(p_val * tl.log(p_val + eps), axis=1)
            expected_score += tl.sum(p_val * score, axis=1)

            score = tl.where(valid, score, neg_inf)
            block_max = tl.max(score, axis=1)
            block_max_safe = tl.where(block_max == neg_inf, 0.0, block_max)
            block_exp = tl.where(valid, tl.exp(score - block_max_safe[:, None]), 0.0)
            block_l = tl.where(block_max == neg_inf, 0.0, tl.sum(block_exp, axis=1))

            new_max = tl.maximum(row_max, block_max)
            new_max_safe = tl.where(new_max == neg_inf, 0.0, new_max)
            row_max_safe = tl.where(row_max == neg_inf, 0.0, row_max)
            old_scale = tl.where(row_max == neg_inf, 0.0, tl.exp(row_max_safe - new_max_safe))
            block_scale = tl.where(block_max == neg_inf, 0.0, tl.exp(block_max_safe - new_max_safe))
            row_l = row_l * old_scale + block_l * block_scale
            row_max = new_max

        log_denom = tl.where(row_l > 0, row_max + tl.log(row_l), neg_inf)

        if TRACK_P_STATS:
            tl.store(ROW_SUM_P + pid_b * stride_rsb + offs_m * stride_rst, row_sum_p, mask=m_mask)
            tl.store(P_LOG_P + pid_b * stride_plb + offs_m * stride_plt, p_log_p, mask=m_mask)
        tl.store(EXPECTED_SCORE + pid_b * stride_esb + offs_m * stride_est, expected_score, mask=m_mask)
        tl.store(LOG_DENOM + pid_b * stride_ldb + offs_m * stride_ldt, log_denom, mask=m_mask)


    @triton.jit
    def _streaming_kl_forward_kernel_block_ptr(
        Q,
        K,
        W,
        P,
        MASK,
        ROW_SUM_P,
        P_LOG_P,
        EXPECTED_SCORE,
        LOG_DENOM,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_wb,
        stride_ws,
        stride_wh,
        stride_pb,
        stride_pt,
        stride_ps,
        stride_mb,
        stride_mt,
        stride_ms,
        stride_rsb,
        stride_rst,
        stride_plb,
        stride_plt,
        stride_esb,
        stride_est,
        stride_ldb,
        stride_ldt,
        n_heads,
        head_dim,
        tgt_len,
        src_len,
        softmax_scale,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_H: tl.constexpr,
        USE_TENSOR_CORE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        OBJECTIVE_IS_CE: tl.constexpr,
        TRACK_P_STATS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        neg_inf = float("-inf")
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n_base = tl.arange(0, BLOCK_N).to(tl.int64)
        m_mask = offs_m < tgt_len

        if TRACK_P_STATS:
            row_sum_p = tl.zeros((BLOCK_M,), dtype=tl.float32)
            p_log_p = tl.zeros((BLOCK_M,), dtype=tl.float32)
        expected_score = tl.zeros((BLOCK_M,), dtype=tl.float32)
        row_max = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
        row_l = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for n_start in range(0, src_len, BLOCK_N):
            offs_n = n_start + offs_n_base
            n_mask = offs_n < src_len
            valid = m_mask[:, None] & n_mask[None, :]
            score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for h in range(MAX_H):
                h_valid = h < n_heads
                h_idx = tl.where(h_valid, h, 0).to(tl.int64)
                q_block_ptr = tl.make_block_ptr(
                    base=Q + pid_b * stride_qb + h_idx * stride_qh,
                    shape=(tgt_len, head_dim),
                    strides=(stride_qt, stride_qd),
                    offsets=(pid_m * BLOCK_M, 0),
                    block_shape=(BLOCK_M, BLOCK_D),
                    order=(1, 0),
                )
                k_block_ptr = tl.make_block_ptr(
                    base=K + pid_b * stride_kb + h_idx * stride_kh,
                    shape=(head_dim, src_len),
                    strides=(stride_kd, stride_ks),
                    offsets=(0, n_start),
                    block_shape=(BLOCK_D, BLOCK_N),
                    order=(0, 1),
                )
                q_raw = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
                k_raw = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
                if USE_TENSOR_CORE:
                    dot = tl.dot(q_raw, k_raw, out_dtype=tl.float32)
                else:
                    dot = tl.dot(q_raw.to(tl.float32), k_raw.to(tl.float32))
                dot = tl.maximum(dot * softmax_scale, 0.0)

                w_ptrs = W + pid_b * stride_wb + offs_n * stride_ws + h_idx * stride_wh
                w_val = tl.load(w_ptrs, mask=n_mask, other=0.0).to(tl.float32)
                h_scale = tl.where(h_valid, 1.0, 0.0)
                score += dot * (w_val[None, :] * h_scale)

            if HAS_MASK:
                mask_ptrs = MASK + pid_b * stride_mb + offs_m[:, None] * stride_mt + offs_n[None, :] * stride_ms
                mask_val = tl.load(mask_ptrs, mask=valid, other=neg_inf).to(tl.float32)
                score += mask_val

            p_ptrs = P + pid_b * stride_pb + offs_m[:, None] * stride_pt + offs_n[None, :] * stride_ps
            p_val = tl.load(p_ptrs, mask=valid, other=0.0).to(tl.float32)
            if TRACK_P_STATS:
                row_sum_p += tl.sum(p_val, axis=1)
                p_log_p += tl.sum(p_val * tl.log(p_val + eps), axis=1)
            expected_score += tl.sum(p_val * score, axis=1)

            score = tl.where(valid, score, neg_inf)
            block_max = tl.max(score, axis=1)
            block_max_safe = tl.where(block_max == neg_inf, 0.0, block_max)
            block_exp = tl.where(valid, tl.exp(score - block_max_safe[:, None]), 0.0)
            block_l = tl.where(block_max == neg_inf, 0.0, tl.sum(block_exp, axis=1))

            new_max = tl.maximum(row_max, block_max)
            new_max_safe = tl.where(new_max == neg_inf, 0.0, new_max)
            row_max_safe = tl.where(row_max == neg_inf, 0.0, row_max)
            old_scale = tl.where(row_max == neg_inf, 0.0, tl.exp(row_max_safe - new_max_safe))
            block_scale = tl.where(block_max == neg_inf, 0.0, tl.exp(block_max_safe - new_max_safe))
            row_l = row_l * old_scale + block_l * block_scale
            row_max = new_max

        log_denom = tl.where(row_l > 0, row_max + tl.log(row_l), neg_inf)

        row_ptrs = pid_b * stride_rsb + offs_m * stride_rst
        if TRACK_P_STATS:
            tl.store(ROW_SUM_P + row_ptrs, row_sum_p, mask=m_mask)
            tl.store(P_LOG_P + pid_b * stride_plb + offs_m * stride_plt, p_log_p, mask=m_mask)
        tl.store(EXPECTED_SCORE + pid_b * stride_esb + offs_m * stride_est, expected_score, mask=m_mask)
        tl.store(LOG_DENOM + pid_b * stride_ldb + offs_m * stride_ldt, log_denom, mask=m_mask)


    @triton.jit
    def _streaming_kl_forward_kernel_h4(
        Q,
        K,
        W,
        P,
        MASK,
        ROW_SUM_P,
        P_LOG_P,
        EXPECTED_SCORE,
        LOG_DENOM,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_wb,
        stride_ws,
        stride_wh,
        stride_pb,
        stride_pt,
        stride_ps,
        stride_mb,
        stride_mt,
        stride_ms,
        stride_rsb,
        stride_rst,
        stride_plb,
        stride_plt,
        stride_esb,
        stride_est,
        stride_ldb,
        stride_ldt,
        n_heads,
        head_dim,
        tgt_len,
        src_len,
        softmax_scale,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_TENSOR_CORE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        OBJECTIVE_IS_CE: tl.constexpr,
        TRACK_P_STATS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        neg_inf = float("-inf")
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        m_mask = offs_m < tgt_len
        d_mask = offs_d < head_dim

        h1_valid = n_heads > 1
        h2_valid = n_heads > 2
        h3_valid = n_heads > 3
        h1_idx = tl.where(h1_valid, 1, 0).to(tl.int64)
        h2_idx = tl.where(h2_valid, 2, 0).to(tl.int64)
        h3_idx = tl.where(h3_valid, 3, 0).to(tl.int64)
        h1_scale = tl.where(h1_valid, 1.0, 0.0)
        h2_scale = tl.where(h2_valid, 1.0, 0.0)
        h3_scale = tl.where(h3_valid, 1.0, 0.0)

        q0_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q1_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h1_idx * stride_qh + offs_d[None, :] * stride_qd
        q2_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h2_idx * stride_qh + offs_d[None, :] * stride_qd
        q3_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h3_idx * stride_qh + offs_d[None, :] * stride_qd
        q0 = tl.load(q0_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q1 = tl.load(q1_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q2 = tl.load(q2_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q3 = tl.load(q3_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

        if TRACK_P_STATS:
            row_sum_p = tl.zeros((BLOCK_M,), dtype=tl.float32)
            p_log_p = tl.zeros((BLOCK_M,), dtype=tl.float32)
        expected_score = tl.zeros((BLOCK_M,), dtype=tl.float32)
        row_max = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
        row_l = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for n_start in range(0, src_len, BLOCK_N):
            offs_n = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
            n_mask = offs_n < src_len
            valid = m_mask[:, None] & n_mask[None, :]

            k0_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k1_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h1_idx * stride_kh + offs_d[None, :] * stride_kd
            k2_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h2_idx * stride_kh + offs_d[None, :] * stride_kd
            k3_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h3_idx * stride_kh + offs_d[None, :] * stride_kd
            k0 = tl.load(k0_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            k1 = tl.load(k1_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            k2 = tl.load(k2_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            k3 = tl.load(k3_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            if USE_TENSOR_CORE:
                pre0 = tl.dot(q0, tl.trans(k0), out_dtype=tl.float32) * softmax_scale
                pre1 = tl.dot(q1, tl.trans(k1), out_dtype=tl.float32) * softmax_scale
                pre2 = tl.dot(q2, tl.trans(k2), out_dtype=tl.float32) * softmax_scale
                pre3 = tl.dot(q3, tl.trans(k3), out_dtype=tl.float32) * softmax_scale
            else:
                pre0 = tl.dot(q0.to(tl.float32), tl.trans(k0.to(tl.float32))) * softmax_scale
                pre1 = tl.dot(q1.to(tl.float32), tl.trans(k1.to(tl.float32))) * softmax_scale
                pre2 = tl.dot(q2.to(tl.float32), tl.trans(k2.to(tl.float32))) * softmax_scale
                pre3 = tl.dot(q3.to(tl.float32), tl.trans(k3.to(tl.float32))) * softmax_scale

            relu0 = tl.maximum(pre0, 0.0)
            relu1 = tl.maximum(pre1, 0.0)
            relu2 = tl.maximum(pre2, 0.0)
            relu3 = tl.maximum(pre3, 0.0)

            w0 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws, mask=n_mask, other=0.0).to(tl.float32)
            w1 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h1_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)
            w2 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h2_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)
            w3 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h3_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)

            score = relu0 * w0[None, :]
            score += relu1 * (w1[None, :] * h1_scale)
            score += relu2 * (w2[None, :] * h2_scale)
            score += relu3 * (w3[None, :] * h3_scale)

            if HAS_MASK:
                mask_ptrs = MASK + pid_b * stride_mb + offs_m[:, None] * stride_mt + offs_n[None, :] * stride_ms
                score += tl.load(mask_ptrs, mask=valid, other=neg_inf).to(tl.float32)

            p_ptrs = P + pid_b * stride_pb + offs_m[:, None] * stride_pt + offs_n[None, :] * stride_ps
            p_val = tl.load(p_ptrs, mask=valid, other=0.0).to(tl.float32)
            if TRACK_P_STATS:
                row_sum_p += tl.sum(p_val, axis=1)
                p_log_p += tl.sum(p_val * tl.log(p_val + eps), axis=1)
            expected_score += tl.sum(p_val * score, axis=1)

            score = tl.where(valid, score, neg_inf)
            block_max = tl.max(score, axis=1)
            block_max_safe = tl.where(block_max == neg_inf, 0.0, block_max)
            block_exp = tl.where(valid, tl.exp(score - block_max_safe[:, None]), 0.0)
            block_l = tl.where(block_max == neg_inf, 0.0, tl.sum(block_exp, axis=1))

            new_max = tl.maximum(row_max, block_max)
            new_max_safe = tl.where(new_max == neg_inf, 0.0, new_max)
            row_max_safe = tl.where(row_max == neg_inf, 0.0, row_max)
            old_scale = tl.where(row_max == neg_inf, 0.0, tl.exp(row_max_safe - new_max_safe))
            block_scale = tl.where(block_max == neg_inf, 0.0, tl.exp(block_max_safe - new_max_safe))
            row_l = row_l * old_scale + block_l * block_scale
            row_max = new_max

        log_denom = tl.where(row_l > 0, row_max + tl.log(row_l), neg_inf)

        if TRACK_P_STATS:
            tl.store(ROW_SUM_P + pid_b * stride_rsb + offs_m * stride_rst, row_sum_p, mask=m_mask)
            tl.store(P_LOG_P + pid_b * stride_plb + offs_m * stride_plt, p_log_p, mask=m_mask)
        tl.store(EXPECTED_SCORE + pid_b * stride_esb + offs_m * stride_est, expected_score, mask=m_mask)
        tl.store(LOG_DENOM + pid_b * stride_ldb + offs_m * stride_ldt, log_denom, mask=m_mask)


def _streaming_kl_forward_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    mask: Optional[torch.Tensor],
    scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is unavailable for streaming KL flash forward.")

    bsz, tgt_len, n_heads, head_dim = q.shape
    src_len = k.shape[1]
    accum_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    objective_is_ce = _resolve_kl_objective() == "ce"
    track_p_stats = (not objective_is_ce) and (not _split_p_stats_from_flash_kernel())
    row_sum_p = None if (objective_is_ce or not track_p_stats) else torch.empty((bsz, tgt_len), device=q.device, dtype=accum_dtype)
    p_log_p = None if (objective_is_ce or not track_p_stats) else torch.empty((bsz, tgt_len), device=q.device, dtype=accum_dtype)
    expected_score = torch.empty((bsz, tgt_len), device=q.device, dtype=accum_dtype)
    log_denom = torch.empty((bsz, tgt_len), device=q.device, dtype=accum_dtype)
    row_sum_p_arg = row_sum_p if row_sum_p is not None else p
    p_log_p_arg = p_log_p if p_log_p is not None else p

    block_m = max(16, _env_int("VGGT_STREAMING_KL_FLASH_BLOCK_M", _env_int("VGGT_INDEXER_SCORE_BLOCK_M", 64)))
    block_n = max(16, _env_int("VGGT_STREAMING_KL_FLASH_BLOCK_N", _env_int("VGGT_INDEXER_SCORE_BLOCK_N", 256)))
    max_h = max(1, _env_int("VGGT_STREAMING_KL_FLASH_MAX_H", max(4, int(n_heads))))
    block_d = max(triton.next_power_of_2(head_dim), 16)
    num_warps = _env_int("VGGT_STREAMING_KL_FLASH_NUM_WARPS", _env_int("VGGT_INDEXER_SCORE_NUM_WARPS", 8))
    num_stages = _env_int("VGGT_STREAMING_KL_FLASH_NUM_STAGES", _env_int("VGGT_INDEXER_SCORE_NUM_STAGES", 1))
    use_tensor_core = _env_flag("VGGT_STREAMING_KL_FLASH_USE_TC")
    if not use_tensor_core and _env_str("VGGT_STREAMING_KL_FLASH_USE_TC", "") == "":
        use_tensor_core = _env_flag("VGGT_INDEXER_SCORE_USE_TC")
    if _resolve_score_kernel_policy is not None:
        block_m, block_n, num_warps, num_stages = _resolve_score_kernel_policy(
            src_len=int(src_len),
            head_dim=int(head_dim),
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    grid = (triton.cdiv(tgt_len, block_m), bsz)
    mask_tensor = mask if mask is not None else p
    if _use_h4_flash_kernels(q, "fwd"):
        _streaming_kl_forward_kernel_h4[grid](
            q,
            k,
            w,
            p,
            mask_tensor,
            row_sum_p_arg,
            p_log_p_arg,
            expected_score,
            log_denom,
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
            p.stride(0),
            p.stride(1),
            p.stride(2),
            mask_tensor.stride(0),
            mask_tensor.stride(1),
            mask_tensor.stride(2),
            row_sum_p_arg.stride(0),
            row_sum_p_arg.stride(1),
            p_log_p_arg.stride(0),
            p_log_p_arg.stride(1),
            expected_score.stride(0),
            expected_score.stride(1),
            log_denom.stride(0),
            log_denom.stride(1),
            n_heads,
            head_dim,
            tgt_len,
            src_len,
            float(scale),
            float(eps),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            USE_TENSOR_CORE=use_tensor_core,
            HAS_MASK=mask is not None,
            OBJECTIVE_IS_CE=objective_is_ce,
            TRACK_P_STATS=track_p_stats,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif _use_h2_flash_fwd_kernel(q):
        _streaming_kl_forward_kernel_h2[grid](
            q,
            k,
            w,
            p,
            mask_tensor,
            row_sum_p_arg,
            p_log_p_arg,
            expected_score,
            log_denom,
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
            p.stride(0),
            p.stride(1),
            p.stride(2),
            mask_tensor.stride(0),
            mask_tensor.stride(1),
            mask_tensor.stride(2),
            row_sum_p_arg.stride(0),
            row_sum_p_arg.stride(1),
            p_log_p_arg.stride(0),
            p_log_p_arg.stride(1),
            expected_score.stride(0),
            expected_score.stride(1),
            log_denom.stride(0),
            log_denom.stride(1),
            n_heads,
            head_dim,
            tgt_len,
            src_len,
            float(scale),
            float(eps),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            USE_TENSOR_CORE=use_tensor_core,
            HAS_MASK=mask is not None,
            OBJECTIVE_IS_CE=objective_is_ce,
            TRACK_P_STATS=track_p_stats,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif _use_flash_block_ptr_fwd():
        _streaming_kl_forward_kernel_block_ptr[grid](
            q,
            k,
            w,
            p,
            mask_tensor,
            row_sum_p_arg,
            p_log_p_arg,
            expected_score,
            log_denom,
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
            p.stride(0),
            p.stride(1),
            p.stride(2),
            mask_tensor.stride(0),
            mask_tensor.stride(1),
            mask_tensor.stride(2),
            row_sum_p_arg.stride(0),
            row_sum_p_arg.stride(1),
            p_log_p_arg.stride(0),
            p_log_p_arg.stride(1),
            expected_score.stride(0),
            expected_score.stride(1),
            log_denom.stride(0),
            log_denom.stride(1),
            n_heads,
            head_dim,
            tgt_len,
            src_len,
            float(scale),
            float(eps),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            MAX_H=max_h,
            USE_TENSOR_CORE=use_tensor_core,
            HAS_MASK=mask is not None,
            OBJECTIVE_IS_CE=objective_is_ce,
            TRACK_P_STATS=track_p_stats,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _streaming_kl_forward_kernel[grid](
            q,
            k,
            w,
            p,
            mask_tensor,
            row_sum_p_arg,
            p_log_p_arg,
            expected_score,
            log_denom,
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
            p.stride(0),
            p.stride(1),
            p.stride(2),
            mask_tensor.stride(0),
            mask_tensor.stride(1),
            mask_tensor.stride(2),
            row_sum_p_arg.stride(0),
            row_sum_p_arg.stride(1),
            p_log_p_arg.stride(0),
            p_log_p_arg.stride(1),
            expected_score.stride(0),
            expected_score.stride(1),
            log_denom.stride(0),
            log_denom.stride(1),
            n_heads,
            head_dim,
            tgt_len,
            src_len,
            float(scale),
            float(eps),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            MAX_H=max_h,
            USE_TENSOR_CORE=use_tensor_core,
            HAS_MASK=mask is not None,
            OBJECTIVE_IS_CE=objective_is_ce,
            TRACK_P_STATS=track_p_stats,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if (not objective_is_ce) and (not track_p_stats):
        row_sum_p = p.sum(dim=-1, dtype=accum_dtype)
        p_log_p = (p.to(accum_dtype) * torch.log(p.to(accum_dtype) + float(eps))).sum(dim=-1)
    if row_sum_p is None:
        row_sum_p = log_denom
    if p_log_p is None:
        p_log_p = log_denom
    return row_sum_p, p_log_p, expected_score, log_denom


def _resolve_kl_backward_mode(q: torch.Tensor) -> str:
    mode = _env_str("VGGT_STREAMING_KL_BWD_MODE", "score").strip().lower()
    if mode in ("score", "torch", "legacy"):
        return "score"
    if mode in ("flash", "fused", "triton_flash"):
        if triton is not None and tl is not None and q.is_cuda:
            return "flash"
        return "score"
    if mode == "auto":
        if triton is not None and tl is not None and q.is_cuda:
            return "flash"
        return "score"
    return "score"


if triton is not None and tl is not None:
    @triton.jit
    def _streaming_kl_backward_dq_kernel(
        Q,
        K,
        W,
        P,
        MASK,
        ROW_SUM_P,
        LOG_DENOM,
        DQ,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_wb,
        stride_ws,
        stride_wh,
        stride_pb,
        stride_pt,
        stride_ps,
        stride_mb,
        stride_mt,
        stride_ms,
        stride_rsb,
        stride_rst,
        stride_ldb,
        stride_ldt,
        stride_dqb,
        stride_dqt,
        stride_dqh,
        stride_dqd,
        n_heads,
        head_dim,
        tgt_len,
        src_len,
        softmax_scale,
        grad_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_H: tl.constexpr,
        USE_TENSOR_CORE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        OBJECTIVE_IS_CE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        neg_inf = float("-inf")
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        m_mask = offs_m < tgt_len
        d_mask = offs_d < head_dim

        if not OBJECTIVE_IS_CE:
            row_sum_p = tl.load(ROW_SUM_P + pid_b * stride_rsb + offs_m * stride_rst, mask=m_mask, other=0.0).to(tl.float32)
        log_denom = tl.load(LOG_DENOM + pid_b * stride_ldb + offs_m * stride_ldt, mask=m_mask, other=0.0).to(tl.float32)

        for h in range(MAX_H):
            h_valid = h < n_heads
            h_idx = tl.where(h_valid, h, 0).to(tl.int64)
            dq_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            q_ptrs = (
                Q
                + pid_b * stride_qb
                + offs_m[:, None] * stride_qt
                + h_idx * stride_qh
                + offs_d[None, :] * stride_qd
            )
            q_raw = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

            for n_start in range(0, src_len, BLOCK_N):
                offs_n = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
                n_mask = offs_n < src_len
                valid = m_mask[:, None] & n_mask[None, :]

                score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for hh in range(MAX_H):
                    hh_valid = hh < n_heads
                    hh_idx = tl.where(hh_valid, hh, 0).to(tl.int64)
                    q_all_ptrs = (
                        Q
                        + pid_b * stride_qb
                        + offs_m[:, None] * stride_qt
                        + hh_idx * stride_qh
                        + offs_d[None, :] * stride_qd
                    )
                    k_all_ptrs = (
                        K
                        + pid_b * stride_kb
                        + offs_n[:, None] * stride_ks
                        + hh_idx * stride_kh
                        + offs_d[None, :] * stride_kd
                    )
                    q_all = tl.load(q_all_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
                    k_all = tl.load(k_all_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                    if USE_TENSOR_CORE:
                        dot = tl.dot(q_all, tl.trans(k_all), out_dtype=tl.float32)
                    else:
                        dot = tl.dot(q_all.to(tl.float32), tl.trans(k_all.to(tl.float32)))
                    dot = tl.maximum(dot * softmax_scale, 0.0)
                    w_ptrs = W + pid_b * stride_wb + offs_n * stride_ws + hh_idx * stride_wh
                    w_val = tl.load(w_ptrs, mask=n_mask, other=0.0).to(tl.float32)
                    score += dot * (w_val[None, :] * tl.where(hh_valid, 1.0, 0.0))

                if HAS_MASK:
                    mask_ptrs = MASK + pid_b * stride_mb + offs_m[:, None] * stride_mt + offs_n[None, :] * stride_ms
                    score += tl.load(mask_ptrs, mask=valid, other=neg_inf).to(tl.float32)
                score = tl.where(valid, score, neg_inf)

                p_ptrs = P + pid_b * stride_pb + offs_m[:, None] * stride_pt + offs_n[None, :] * stride_ps
                p_val = tl.load(p_ptrs, mask=valid, other=0.0).to(tl.float32)
                softmax = tl.exp(score - log_denom[:, None])
                if OBJECTIVE_IS_CE:
                    grad_score = softmax - p_val
                else:
                    grad_score = row_sum_p[:, None] * softmax - p_val
                grad_score = grad_score * grad_scale

                k_ptrs = (
                    K
                    + pid_b * stride_kb
                    + offs_n[:, None] * stride_ks
                    + h_idx * stride_kh
                    + offs_d[None, :] * stride_kd
                )
                k_raw = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                if USE_TENSOR_CORE:
                    dot_h = tl.dot(q_raw, tl.trans(k_raw), out_dtype=tl.float32)
                else:
                    dot_h = tl.dot(q_raw.to(tl.float32), tl.trans(k_raw.to(tl.float32)))
                pre_h = dot_h * softmax_scale
                active = valid & (pre_h > 0)
                w_ptrs = W + pid_b * stride_wb + offs_n * stride_ws + h_idx * stride_wh
                w_val = tl.load(w_ptrs, mask=n_mask, other=0.0).to(tl.float32)
                coeff = tl.where(active, grad_score * w_val[None, :], 0.0)
                dq_acc += tl.dot(coeff, k_raw.to(tl.float32)) * softmax_scale

            dq_ptrs = (
                DQ
                + pid_b * stride_dqb
                + offs_m[:, None] * stride_dqt
                + h_idx * stride_dqh
                + offs_d[None, :] * stride_dqd
            )
            store_mask = m_mask[:, None] & d_mask[None, :] & tl.full((BLOCK_M, BLOCK_D), h_valid, tl.int1)
            tl.store(dq_ptrs, dq_acc.to(q_raw.dtype), mask=store_mask)


    @triton.jit
    def _streaming_kl_backward_dkdw_kernel(
        Q,
        K,
        W,
        P,
        MASK,
        ROW_SUM_P,
        LOG_DENOM,
        DK,
        DW,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_wb,
        stride_ws,
        stride_wh,
        stride_pb,
        stride_pt,
        stride_ps,
        stride_mb,
        stride_mt,
        stride_ms,
        stride_rsb,
        stride_rst,
        stride_ldb,
        stride_ldt,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dkd,
        stride_dwb,
        stride_dws,
        stride_dwh,
        n_heads,
        head_dim,
        tgt_len,
        src_len,
        softmax_scale,
        grad_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_H: tl.constexpr,
        USE_TENSOR_CORE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        OBJECTIVE_IS_CE: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)

        neg_inf = float("-inf")
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        n_mask = offs_n < src_len
        d_mask = offs_d < head_dim

        for h in range(MAX_H):
            h_valid = h < n_heads
            h_idx = tl.where(h_valid, h, 0).to(tl.int64)
            dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
            dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

            k_ptrs = (
                K
                + pid_b * stride_kb
                + offs_n[:, None] * stride_ks
                + h_idx * stride_kh
                + offs_d[None, :] * stride_kd
            )
            k_raw = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            w_ptrs = W + pid_b * stride_wb + offs_n * stride_ws + h_idx * stride_wh
            w_val = tl.load(w_ptrs, mask=n_mask, other=0.0).to(tl.float32)

            for m_start in range(0, tgt_len, BLOCK_M):
                offs_m = (m_start + tl.arange(0, BLOCK_M)).to(tl.int64)
                m_mask = offs_m < tgt_len
                valid = m_mask[:, None] & n_mask[None, :]
                if not OBJECTIVE_IS_CE:
                    row_sum_p = tl.load(ROW_SUM_P + pid_b * stride_rsb + offs_m * stride_rst, mask=m_mask, other=0.0).to(tl.float32)
                log_denom = tl.load(LOG_DENOM + pid_b * stride_ldb + offs_m * stride_ldt, mask=m_mask, other=0.0).to(tl.float32)

                score = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for hh in range(MAX_H):
                    hh_valid = hh < n_heads
                    hh_idx = tl.where(hh_valid, hh, 0).to(tl.int64)
                    q_all_ptrs = (
                        Q
                        + pid_b * stride_qb
                        + offs_m[:, None] * stride_qt
                        + hh_idx * stride_qh
                        + offs_d[None, :] * stride_qd
                    )
                    k_all_ptrs = (
                        K
                        + pid_b * stride_kb
                        + offs_n[:, None] * stride_ks
                        + hh_idx * stride_kh
                        + offs_d[None, :] * stride_kd
                    )
                    q_all = tl.load(q_all_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
                    k_all = tl.load(k_all_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
                    if USE_TENSOR_CORE:
                        dot = tl.dot(q_all, tl.trans(k_all), out_dtype=tl.float32)
                    else:
                        dot = tl.dot(q_all.to(tl.float32), tl.trans(k_all.to(tl.float32)))
                    dot = tl.maximum(dot * softmax_scale, 0.0)
                    w_all_ptrs = W + pid_b * stride_wb + offs_n * stride_ws + hh_idx * stride_wh
                    w_all = tl.load(w_all_ptrs, mask=n_mask, other=0.0).to(tl.float32)
                    score += dot * (w_all[None, :] * tl.where(hh_valid, 1.0, 0.0))

                if HAS_MASK:
                    mask_ptrs = MASK + pid_b * stride_mb + offs_m[:, None] * stride_mt + offs_n[None, :] * stride_ms
                    score += tl.load(mask_ptrs, mask=valid, other=neg_inf).to(tl.float32)
                score = tl.where(valid, score, neg_inf)

                p_ptrs = P + pid_b * stride_pb + offs_m[:, None] * stride_pt + offs_n[None, :] * stride_ps
                p_val = tl.load(p_ptrs, mask=valid, other=0.0).to(tl.float32)
                softmax = tl.exp(score - log_denom[:, None])
                if OBJECTIVE_IS_CE:
                    grad_score = softmax - p_val
                else:
                    grad_score = row_sum_p[:, None] * softmax - p_val
                grad_score = grad_score * grad_scale

                q_ptrs = (
                    Q
                    + pid_b * stride_qb
                    + offs_m[:, None] * stride_qt
                    + h_idx * stride_qh
                    + offs_d[None, :] * stride_qd
                )
                q_raw = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
                if USE_TENSOR_CORE:
                    dot_h = tl.dot(q_raw, tl.trans(k_raw), out_dtype=tl.float32)
                else:
                    dot_h = tl.dot(q_raw.to(tl.float32), tl.trans(k_raw.to(tl.float32)))
                pre_h = dot_h * softmax_scale
                relu_h = tl.maximum(pre_h, 0.0)
                active = valid & (pre_h > 0)
                coeff = tl.where(active, grad_score * w_val[None, :], 0.0)
                dk_acc += tl.dot(tl.trans(coeff), q_raw.to(tl.float32)) * softmax_scale
                dw_acc += tl.sum(grad_score * relu_h, axis=0)

            dk_ptrs = (
                DK
                + pid_b * stride_dkb
                + offs_n[:, None] * stride_dks
                + h_idx * stride_dkh
                + offs_d[None, :] * stride_dkd
            )
            dk_mask = n_mask[:, None] & d_mask[None, :] & tl.full((BLOCK_N, BLOCK_D), h_valid, tl.int1)
            tl.store(dk_ptrs, dk_acc.to(k_raw.dtype), mask=dk_mask)

            dw_ptrs = DW + pid_b * stride_dwb + offs_n * stride_dws + h_idx * stride_dwh
            dw_mask = n_mask & tl.full((BLOCK_N,), h_valid, tl.int1)
            tl.store(dw_ptrs, dw_acc.to(w_val.dtype), mask=dw_mask)


    @triton.jit
    def _streaming_kl_backward_dq_kernel_h4(
        Q,
        K,
        W,
        P,
        MASK,
        ROW_SUM_P,
        LOG_DENOM,
        DQ,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_wb,
        stride_ws,
        stride_wh,
        stride_pb,
        stride_pt,
        stride_ps,
        stride_mb,
        stride_mt,
        stride_ms,
        stride_rsb,
        stride_rst,
        stride_ldb,
        stride_ldt,
        stride_dqb,
        stride_dqt,
        stride_dqh,
        stride_dqd,
        n_heads,
        head_dim,
        tgt_len,
        src_len,
        softmax_scale,
        grad_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_TENSOR_CORE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        OBJECTIVE_IS_CE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        neg_inf = float("-inf")
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        m_mask = offs_m < tgt_len
        d_mask = offs_d < head_dim

        h1_valid = n_heads > 1
        h2_valid = n_heads > 2
        h3_valid = n_heads > 3
        h1_idx = tl.where(h1_valid, 1, 0).to(tl.int64)
        h2_idx = tl.where(h2_valid, 2, 0).to(tl.int64)
        h3_idx = tl.where(h3_valid, 3, 0).to(tl.int64)
        h1_scale = tl.where(h1_valid, 1.0, 0.0)
        h2_scale = tl.where(h2_valid, 1.0, 0.0)
        h3_scale = tl.where(h3_valid, 1.0, 0.0)

        if not OBJECTIVE_IS_CE:
            row_sum_p = tl.load(ROW_SUM_P + pid_b * stride_rsb + offs_m * stride_rst, mask=m_mask, other=0.0).to(tl.float32)
        log_denom = tl.load(LOG_DENOM + pid_b * stride_ldb + offs_m * stride_ldt, mask=m_mask, other=0.0).to(tl.float32)

        q0_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q1_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h1_idx * stride_qh + offs_d[None, :] * stride_qd
        q2_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h2_idx * stride_qh + offs_d[None, :] * stride_qd
        q3_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h3_idx * stride_qh + offs_d[None, :] * stride_qd
        q0 = tl.load(q0_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q1 = tl.load(q1_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q2 = tl.load(q2_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
        q3 = tl.load(q3_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

        dq0 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        dq1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        dq2 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        dq3 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for n_start in range(0, src_len, BLOCK_N):
            offs_n = (n_start + tl.arange(0, BLOCK_N)).to(tl.int64)
            n_mask = offs_n < src_len
            valid = m_mask[:, None] & n_mask[None, :]

            k0_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k1_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h1_idx * stride_kh + offs_d[None, :] * stride_kd
            k2_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h2_idx * stride_kh + offs_d[None, :] * stride_kd
            k3_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h3_idx * stride_kh + offs_d[None, :] * stride_kd
            k0 = tl.load(k0_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            k1 = tl.load(k1_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            k2 = tl.load(k2_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
            k3 = tl.load(k3_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            if USE_TENSOR_CORE:
                pre0 = tl.dot(q0, tl.trans(k0), out_dtype=tl.float32) * softmax_scale
                pre1 = tl.dot(q1, tl.trans(k1), out_dtype=tl.float32) * softmax_scale
                pre2 = tl.dot(q2, tl.trans(k2), out_dtype=tl.float32) * softmax_scale
                pre3 = tl.dot(q3, tl.trans(k3), out_dtype=tl.float32) * softmax_scale
            else:
                pre0 = tl.dot(q0.to(tl.float32), tl.trans(k0.to(tl.float32))) * softmax_scale
                pre1 = tl.dot(q1.to(tl.float32), tl.trans(k1.to(tl.float32))) * softmax_scale
                pre2 = tl.dot(q2.to(tl.float32), tl.trans(k2.to(tl.float32))) * softmax_scale
                pre3 = tl.dot(q3.to(tl.float32), tl.trans(k3.to(tl.float32))) * softmax_scale

            relu0 = tl.maximum(pre0, 0.0)
            relu1 = tl.maximum(pre1, 0.0)
            relu2 = tl.maximum(pre2, 0.0)
            relu3 = tl.maximum(pre3, 0.0)

            w0 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws, mask=n_mask, other=0.0).to(tl.float32)
            w1 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h1_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)
            w2 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h2_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)
            w3 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h3_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)

            score = relu0 * w0[None, :]
            score += relu1 * (w1[None, :] * h1_scale)
            score += relu2 * (w2[None, :] * h2_scale)
            score += relu3 * (w3[None, :] * h3_scale)
            if HAS_MASK:
                mask_ptrs = MASK + pid_b * stride_mb + offs_m[:, None] * stride_mt + offs_n[None, :] * stride_ms
                score += tl.load(mask_ptrs, mask=valid, other=neg_inf).to(tl.float32)
            score = tl.where(valid, score, neg_inf)

            p_ptrs = P + pid_b * stride_pb + offs_m[:, None] * stride_pt + offs_n[None, :] * stride_ps
            p_val = tl.load(p_ptrs, mask=valid, other=0.0).to(tl.float32)
            softmax = tl.exp(score - log_denom[:, None])
            if OBJECTIVE_IS_CE:
                grad_score = softmax - p_val
            else:
                grad_score = row_sum_p[:, None] * softmax - p_val
            grad_score = grad_score * grad_scale

            coeff0 = tl.where(valid & (pre0 > 0), grad_score * w0[None, :], 0.0)
            coeff1 = tl.where(valid & (pre1 > 0), grad_score * w1[None, :] * h1_scale, 0.0)
            coeff2 = tl.where(valid & (pre2 > 0), grad_score * w2[None, :] * h2_scale, 0.0)
            coeff3 = tl.where(valid & (pre3 > 0), grad_score * w3[None, :] * h3_scale, 0.0)

            dq0 += tl.dot(coeff0, k0.to(tl.float32)) * softmax_scale
            dq1 += tl.dot(coeff1, k1.to(tl.float32)) * softmax_scale
            dq2 += tl.dot(coeff2, k2.to(tl.float32)) * softmax_scale
            dq3 += tl.dot(coeff3, k3.to(tl.float32)) * softmax_scale

        dq0_ptrs = DQ + pid_b * stride_dqb + offs_m[:, None] * stride_dqt + offs_d[None, :] * stride_dqd
        dq1_ptrs = DQ + pid_b * stride_dqb + offs_m[:, None] * stride_dqt + h1_idx * stride_dqh + offs_d[None, :] * stride_dqd
        dq2_ptrs = DQ + pid_b * stride_dqb + offs_m[:, None] * stride_dqt + h2_idx * stride_dqh + offs_d[None, :] * stride_dqd
        dq3_ptrs = DQ + pid_b * stride_dqb + offs_m[:, None] * stride_dqt + h3_idx * stride_dqh + offs_d[None, :] * stride_dqd
        base_mask = m_mask[:, None] & d_mask[None, :]
        tl.store(dq0_ptrs, dq0.to(q0.dtype), mask=base_mask)
        tl.store(dq1_ptrs, dq1.to(q0.dtype), mask=base_mask & h1_valid)
        tl.store(dq2_ptrs, dq2.to(q0.dtype), mask=base_mask & h2_valid)
        tl.store(dq3_ptrs, dq3.to(q0.dtype), mask=base_mask & h3_valid)


    @triton.jit
    def _streaming_kl_backward_dkdw_kernel_h4(
        Q,
        K,
        W,
        P,
        MASK,
        ROW_SUM_P,
        LOG_DENOM,
        DK,
        DW,
        stride_qb,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kh,
        stride_kd,
        stride_wb,
        stride_ws,
        stride_wh,
        stride_pb,
        stride_pt,
        stride_ps,
        stride_mb,
        stride_mt,
        stride_ms,
        stride_rsb,
        stride_rst,
        stride_ldb,
        stride_ldt,
        stride_dkb,
        stride_dks,
        stride_dkh,
        stride_dkd,
        stride_dwb,
        stride_dws,
        stride_dwh,
        n_heads,
        head_dim,
        tgt_len,
        src_len,
        softmax_scale,
        grad_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_TENSOR_CORE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        OBJECTIVE_IS_CE: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_b = tl.program_id(1)

        neg_inf = float("-inf")
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        n_mask = offs_n < src_len
        d_mask = offs_d < head_dim

        h1_valid = n_heads > 1
        h2_valid = n_heads > 2
        h3_valid = n_heads > 3
        h1_idx = tl.where(h1_valid, 1, 0).to(tl.int64)
        h2_idx = tl.where(h2_valid, 2, 0).to(tl.int64)
        h3_idx = tl.where(h3_valid, 3, 0).to(tl.int64)
        h1_scale = tl.where(h1_valid, 1.0, 0.0)
        h2_scale = tl.where(h2_valid, 1.0, 0.0)
        h3_scale = tl.where(h3_valid, 1.0, 0.0)

        k0_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k1_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h1_idx * stride_kh + offs_d[None, :] * stride_kd
        k2_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h2_idx * stride_kh + offs_d[None, :] * stride_kd
        k3_ptrs = K + pid_b * stride_kb + offs_n[:, None] * stride_ks + h3_idx * stride_kh + offs_d[None, :] * stride_kd
        k0 = tl.load(k0_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        k1 = tl.load(k1_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        k2 = tl.load(k2_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        k3 = tl.load(k3_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        w0 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws, mask=n_mask, other=0.0).to(tl.float32)
        w1 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h1_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)
        w2 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h2_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)
        w3 = tl.load(W + pid_b * stride_wb + offs_n * stride_ws + h3_idx * stride_wh, mask=n_mask, other=0.0).to(tl.float32)

        dk0 = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dk1 = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dk2 = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dk3 = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        dw0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        dw1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        dw2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        dw3 = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for m_start in range(0, tgt_len, BLOCK_M):
            offs_m = (m_start + tl.arange(0, BLOCK_M)).to(tl.int64)
            m_mask = offs_m < tgt_len
            valid = m_mask[:, None] & n_mask[None, :]

            if not OBJECTIVE_IS_CE:
                row_sum_p = tl.load(ROW_SUM_P + pid_b * stride_rsb + offs_m * stride_rst, mask=m_mask, other=0.0).to(tl.float32)
            log_denom = tl.load(LOG_DENOM + pid_b * stride_ldb + offs_m * stride_ldt, mask=m_mask, other=0.0).to(tl.float32)

            q0_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
            q1_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h1_idx * stride_qh + offs_d[None, :] * stride_qd
            q2_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h2_idx * stride_qh + offs_d[None, :] * stride_qd
            q3_ptrs = Q + pid_b * stride_qb + offs_m[:, None] * stride_qt + h3_idx * stride_qh + offs_d[None, :] * stride_qd
            q0 = tl.load(q0_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            q1 = tl.load(q1_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            q2 = tl.load(q2_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
            q3 = tl.load(q3_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

            if USE_TENSOR_CORE:
                pre0 = tl.dot(q0, tl.trans(k0), out_dtype=tl.float32) * softmax_scale
                pre1 = tl.dot(q1, tl.trans(k1), out_dtype=tl.float32) * softmax_scale
                pre2 = tl.dot(q2, tl.trans(k2), out_dtype=tl.float32) * softmax_scale
                pre3 = tl.dot(q3, tl.trans(k3), out_dtype=tl.float32) * softmax_scale
            else:
                pre0 = tl.dot(q0.to(tl.float32), tl.trans(k0.to(tl.float32))) * softmax_scale
                pre1 = tl.dot(q1.to(tl.float32), tl.trans(k1.to(tl.float32))) * softmax_scale
                pre2 = tl.dot(q2.to(tl.float32), tl.trans(k2.to(tl.float32))) * softmax_scale
                pre3 = tl.dot(q3.to(tl.float32), tl.trans(k3.to(tl.float32))) * softmax_scale

            relu0 = tl.maximum(pre0, 0.0)
            relu1 = tl.maximum(pre1, 0.0)
            relu2 = tl.maximum(pre2, 0.0)
            relu3 = tl.maximum(pre3, 0.0)

            score = relu0 * w0[None, :]
            score += relu1 * (w1[None, :] * h1_scale)
            score += relu2 * (w2[None, :] * h2_scale)
            score += relu3 * (w3[None, :] * h3_scale)
            if HAS_MASK:
                mask_ptrs = MASK + pid_b * stride_mb + offs_m[:, None] * stride_mt + offs_n[None, :] * stride_ms
                score += tl.load(mask_ptrs, mask=valid, other=neg_inf).to(tl.float32)
            score = tl.where(valid, score, neg_inf)

            p_ptrs = P + pid_b * stride_pb + offs_m[:, None] * stride_pt + offs_n[None, :] * stride_ps
            p_val = tl.load(p_ptrs, mask=valid, other=0.0).to(tl.float32)
            softmax = tl.exp(score - log_denom[:, None])
            if OBJECTIVE_IS_CE:
                grad_score = softmax - p_val
            else:
                grad_score = row_sum_p[:, None] * softmax - p_val
            grad_score = grad_score * grad_scale

            coeff0 = tl.where(valid & (pre0 > 0), grad_score * w0[None, :], 0.0)
            coeff1 = tl.where(valid & (pre1 > 0), grad_score * w1[None, :] * h1_scale, 0.0)
            coeff2 = tl.where(valid & (pre2 > 0), grad_score * w2[None, :] * h2_scale, 0.0)
            coeff3 = tl.where(valid & (pre3 > 0), grad_score * w3[None, :] * h3_scale, 0.0)

            dk0 += tl.dot(tl.trans(coeff0), q0.to(tl.float32)) * softmax_scale
            dk1 += tl.dot(tl.trans(coeff1), q1.to(tl.float32)) * softmax_scale
            dk2 += tl.dot(tl.trans(coeff2), q2.to(tl.float32)) * softmax_scale
            dk3 += tl.dot(tl.trans(coeff3), q3.to(tl.float32)) * softmax_scale
            dw0 += tl.sum(grad_score * relu0, axis=0)
            dw1 += tl.sum(grad_score * relu1, axis=0) * h1_scale
            dw2 += tl.sum(grad_score * relu2, axis=0) * h2_scale
            dw3 += tl.sum(grad_score * relu3, axis=0) * h3_scale

        base_mask = n_mask[:, None] & d_mask[None, :]
        dk0_ptrs = DK + pid_b * stride_dkb + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dk1_ptrs = DK + pid_b * stride_dkb + offs_n[:, None] * stride_dks + h1_idx * stride_dkh + offs_d[None, :] * stride_dkd
        dk2_ptrs = DK + pid_b * stride_dkb + offs_n[:, None] * stride_dks + h2_idx * stride_dkh + offs_d[None, :] * stride_dkd
        dk3_ptrs = DK + pid_b * stride_dkb + offs_n[:, None] * stride_dks + h3_idx * stride_dkh + offs_d[None, :] * stride_dkd
        tl.store(dk0_ptrs, dk0.to(k0.dtype), mask=base_mask)
        tl.store(dk1_ptrs, dk1.to(k0.dtype), mask=base_mask & h1_valid)
        tl.store(dk2_ptrs, dk2.to(k0.dtype), mask=base_mask & h2_valid)
        tl.store(dk3_ptrs, dk3.to(k0.dtype), mask=base_mask & h3_valid)

        dw0_ptrs = DW + pid_b * stride_dwb + offs_n * stride_dws
        dw1_ptrs = DW + pid_b * stride_dwb + offs_n * stride_dws + h1_idx * stride_dwh
        dw2_ptrs = DW + pid_b * stride_dwb + offs_n * stride_dws + h2_idx * stride_dwh
        dw3_ptrs = DW + pid_b * stride_dwb + offs_n * stride_dws + h3_idx * stride_dwh
        tl.store(dw0_ptrs, dw0.to(w0.dtype), mask=n_mask)
        tl.store(dw1_ptrs, dw1.to(w0.dtype), mask=n_mask & h1_valid)
        tl.store(dw2_ptrs, dw2.to(w0.dtype), mask=n_mask & h2_valid)
        tl.store(dw3_ptrs, dw3.to(w0.dtype), mask=n_mask & h3_valid)


def _streaming_kl_backward_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    mask: Optional[torch.Tensor],
    row_sum_p: torch.Tensor,
    log_denom: torch.Tensor,
    scale: float,
    grad_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is unavailable for streaming KL flash backward.")

    bsz, tgt_len, n_heads, head_dim = q.shape
    src_len = k.shape[1]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)

    block_m = max(16, _env_int("VGGT_STREAMING_KL_BWD_BLOCK_M", _env_int("VGGT_STREAMING_KL_FLASH_BLOCK_M", 64)))
    block_n = max(16, _env_int("VGGT_STREAMING_KL_BWD_BLOCK_N", 64))
    max_h = max(1, _env_int("VGGT_STREAMING_KL_BWD_MAX_H", max(4, int(n_heads))))
    block_d = max(triton.next_power_of_2(head_dim), 16)
    num_warps = _env_int("VGGT_STREAMING_KL_BWD_NUM_WARPS", 4)
    num_stages = _env_int("VGGT_STREAMING_KL_BWD_NUM_STAGES", 1)
    use_tensor_core = _env_flag("VGGT_STREAMING_KL_BWD_USE_TC")
    objective_is_ce = _resolve_kl_objective() == "ce"
    if not use_tensor_core and _env_str("VGGT_STREAMING_KL_BWD_USE_TC", "") == "":
        use_tensor_core = _env_flag("VGGT_STREAMING_KL_FLASH_USE_TC")
        if not use_tensor_core and _env_str("VGGT_STREAMING_KL_FLASH_USE_TC", "") == "":
            use_tensor_core = _env_flag("VGGT_INDEXER_SCORE_USE_TC")

    mask_tensor = mask if mask is not None else p
    dq_grid = (triton.cdiv(tgt_len, block_m), bsz)
    dkdw_grid = (triton.cdiv(src_len, block_n), bsz)
    if _use_h4_flash_kernels(q, "bwd"):
        _streaming_kl_backward_dq_kernel_h4[dq_grid](
            q,
            k,
            w,
            p,
            mask_tensor,
            row_sum_p,
            log_denom,
            dq,
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
            p.stride(0),
            p.stride(1),
            p.stride(2),
            mask_tensor.stride(0),
            mask_tensor.stride(1),
            mask_tensor.stride(2),
            row_sum_p.stride(0),
            row_sum_p.stride(1),
            log_denom.stride(0),
            log_denom.stride(1),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            n_heads,
            head_dim,
            tgt_len,
            src_len,
            float(scale),
            float(grad_scale),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            USE_TENSOR_CORE=use_tensor_core,
            HAS_MASK=mask is not None,
            OBJECTIVE_IS_CE=objective_is_ce,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        _streaming_kl_backward_dkdw_kernel_h4[dkdw_grid](
            q,
            k,
            w,
            p,
            mask_tensor,
            row_sum_p,
            log_denom,
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
            p.stride(0),
            p.stride(1),
            p.stride(2),
            mask_tensor.stride(0),
            mask_tensor.stride(1),
            mask_tensor.stride(2),
            row_sum_p.stride(0),
            row_sum_p.stride(1),
            log_denom.stride(0),
            log_denom.stride(1),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dw.stride(0),
            dw.stride(1),
            dw.stride(2),
            n_heads,
            head_dim,
            tgt_len,
            src_len,
            float(scale),
            float(grad_scale),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            USE_TENSOR_CORE=use_tensor_core,
            HAS_MASK=mask is not None,
            OBJECTIVE_IS_CE=objective_is_ce,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _streaming_kl_backward_dq_kernel[dq_grid](
            q,
            k,
            w,
            p,
            mask_tensor,
            row_sum_p,
            log_denom,
            dq,
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
            p.stride(0),
            p.stride(1),
            p.stride(2),
            mask_tensor.stride(0),
            mask_tensor.stride(1),
            mask_tensor.stride(2),
            row_sum_p.stride(0),
            row_sum_p.stride(1),
            log_denom.stride(0),
            log_denom.stride(1),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            n_heads,
            head_dim,
            tgt_len,
            src_len,
            float(scale),
            float(grad_scale),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            MAX_H=max_h,
            USE_TENSOR_CORE=use_tensor_core,
            HAS_MASK=mask is not None,
            OBJECTIVE_IS_CE=objective_is_ce,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        _streaming_kl_backward_dkdw_kernel[dkdw_grid](
            q,
            k,
            w,
            p,
            mask_tensor,
            row_sum_p,
            log_denom,
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
            p.stride(0),
            p.stride(1),
            p.stride(2),
            mask_tensor.stride(0),
            mask_tensor.stride(1),
            mask_tensor.stride(2),
            row_sum_p.stride(0),
            row_sum_p.stride(1),
            log_denom.stride(0),
            log_denom.stride(1),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dw.stride(0),
            dw.stride(1),
            dw.stride(2),
            n_heads,
            head_dim,
            tgt_len,
            src_len,
            float(scale),
            float(grad_scale),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            MAX_H=max_h,
            USE_TENSOR_CORE=use_tensor_core,
            HAS_MASK=mask is not None,
            OBJECTIVE_IS_CE=objective_is_ce,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return dq, dk, dw


def _score_chunk_forward_legacy(
    q: torch.Tensor,
    k_chunk: torch.Tensor,
    w_chunk: torch.Tensor,
    scale: float,
    score_head_chunk_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    bsz, tgt_len, n_heads, _ = q.shape
    src_chunk = k_chunk.shape[1]
    score_chunk = None
    compute_dtype = torch.promote_types(torch.promote_types(q.dtype, k_chunk.dtype), w_chunk.dtype)

    for h_start in range(0, n_heads, score_head_chunk_size):
        h_end = min(h_start + score_head_chunk_size, n_heads)
        q_h = q[:, :, h_start:h_end].to(compute_dtype)
        k_h = k_chunk[:, :, h_start:h_end].to(compute_dtype)
        w_h = w_chunk[:, h_start:h_end].to(compute_dtype)

        head_scores = torch.einsum("bthd,bshd->bths", q_h, k_h) * scale
        head_scores = torch.relu(head_scores)
        head_scores = head_scores * w_h.unsqueeze(1)
        head_scores = head_scores.sum(dim=2, dtype=out_dtype)
        score_chunk = head_scores if score_chunk is None else score_chunk + head_scores

    if score_chunk is None:
        score_chunk = q.new_zeros((bsz, tgt_len, src_chunk), dtype=out_dtype)
    return score_chunk


def _score_chunk_forward(
    q: torch.Tensor,
    k_chunk: torch.Tensor,
    w_chunk: torch.Tensor,
    scale: float,
    score_head_chunk_size: int,
    out_dtype: torch.dtype,
    total_src_len: int,
    mask_chunk: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mode = _resolve_score_mode(q)
    if mode == "triton":
        try:
            score_chunk = _compute_scores_block_triton(
                q=q,
                k_block=k_chunk,
                w_block=w_chunk,
                mask_block=mask_chunk,
                softmax_scale=float(scale),
                total_src_len=int(total_src_len),
                out=out,
            )
            if score_chunk.dtype != out_dtype:
                score_chunk = score_chunk.to(out_dtype)
            return score_chunk
        except Exception as exc:
            global _SCORE_MODE_FALLBACK_COUNT
            _SCORE_MODE_FALLBACK_COUNT += 1
            if _env_flag("VGGT_STREAMING_KL_SCORE_DEBUG") and _SCORE_MODE_FALLBACK_COUNT <= 8:
                print("[streaming_kl_autograd] triton fallback:", repr(exc))

    score_chunk = _score_chunk_forward_legacy(
        q=q,
        k_chunk=k_chunk,
        w_chunk=w_chunk.permute(0, 2, 1).contiguous(),
        scale=float(scale),
        score_head_chunk_size=int(score_head_chunk_size),
        out_dtype=out_dtype,
    )
    if mask_chunk is not None:
        score_chunk = score_chunk + mask_chunk
    return score_chunk


def _score_chunk_backward(
    q: torch.Tensor,
    k_chunk: torch.Tensor,
    w_chunk: torch.Tensor,
    grad_score_chunk: torch.Tensor,
    scale: float,
    score_head_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, n_heads, _ = q.shape
    src_chunk = k_chunk.shape[1]
    dq = torch.zeros_like(q)
    dk_chunk = torch.zeros_like(k_chunk)
    dw_chunk = torch.zeros_like(w_chunk)
    compute_dtype = torch.promote_types(torch.promote_types(q.dtype, k_chunk.dtype), w_chunk.dtype)
    compute_dtype = torch.promote_types(compute_dtype, grad_score_chunk.dtype)
    grad_score_chunk = grad_score_chunk.to(compute_dtype)

    for h_start in range(0, n_heads, score_head_chunk_size):
        h_end = min(h_start + score_head_chunk_size, n_heads)
        q_h = q[:, :, h_start:h_end].to(compute_dtype)
        k_h = k_chunk[:, :, h_start:h_end].to(compute_dtype)
        w_h = w_chunk[:, :, h_start:h_end].to(compute_dtype)

        pre_act = torch.einsum("bthd,bshd->bths", q_h, k_h) * scale
        relu_scores = torch.relu(pre_act)
        grad_relu = grad_score_chunk.unsqueeze(2) * w_h.permute(0, 2, 1).unsqueeze(1)
        grad_pre = torch.where(pre_act > 0, grad_relu, torch.zeros_like(grad_relu))

        dq_update = torch.einsum(
            "bths,bshd->bthd",
            grad_pre,
            k_h,
        ) * scale
        dk_update = torch.einsum(
            "bths,bthd->bshd",
            grad_pre,
            q_h,
        ) * scale
        dw_update = (
            grad_score_chunk.unsqueeze(2) * relu_scores
        ).sum(dim=1, dtype=compute_dtype).permute(0, 2, 1)

        dq[:, :, h_start:h_end] = dq[:, :, h_start:h_end] + dq_update.to(dq.dtype)
        dk_chunk[:, :, h_start:h_end] = dk_chunk[:, :, h_start:h_end] + dk_update.to(dk_chunk.dtype)
        dw_chunk[:, :, h_start:h_end] = dw_chunk[:, :, h_start:h_end] + dw_update.to(dw_chunk.dtype)

    if src_chunk == 0:
        dk_chunk = q.new_zeros((q.shape[0], 0, n_heads, q.shape[3]))
        dw_chunk = q.new_zeros((q.shape[0], 0, n_heads))
    return dq, dk_chunk, dw_chunk


class _StreamingKLLossAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
        score_head_chunk_size: int,
        score_key_chunk_size: int,
        eps: float,
    ) -> torch.Tensor:
        if p.dim() != 3:
            raise ValueError(f"Expected p shape [B, T, S], got {tuple(p.shape)}")

        bsz, tgt_len, src_len = p.shape
        accum_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        has_mask = mask.numel() > 0
        norm_mask = mask if has_mask else None
        objective = _resolve_kl_objective()
        objective_is_ce = objective == "ce"
        if _resolve_kl_forward_mode(q) == "flash":
            try:
                row_sum_p, p_log_p, expected_score, log_denom = _streaming_kl_forward_flash(
                    q=q,
                    k=k,
                    w=w,
                    p=p,
                    mask=norm_mask,
                    scale=float(scale),
                    eps=float(eps),
                )
                ctx.save_for_backward(q, k, w, p, mask, row_sum_p, log_denom)
                ctx.scale = float(scale)
                ctx.score_head_chunk_size = int(score_head_chunk_size)
                ctx.score_key_chunk_size = int(score_key_chunk_size)
                ctx.eps = float(eps)
                ctx.objective = objective
                if objective_is_ce:
                    return (-expected_score + log_denom).mean()
                return (p_log_p - expected_score + row_sum_p * log_denom).mean()
            except Exception as exc:
                global _FLASH_FWD_FALLBACK_COUNT
                _FLASH_FWD_FALLBACK_COUNT += 1
                if _env_flag("VGGT_STREAMING_KL_FLASH_DEBUG") and _FLASH_FWD_FALLBACK_COUNT <= 8:
                    print("[streaming_kl_autograd] flash forward fallback:", repr(exc))

        score_chunk_buf = None
        if _resolve_score_mode(q) == "triton":
            max_src_chunk = max(1, int(score_key_chunk_size))
            score_chunk_buf = q.new_empty((bsz, tgt_len, max_src_chunk), dtype=q.dtype)

        row_sum_p = None if objective_is_ce else torch.zeros((bsz, tgt_len), device=p.device, dtype=accum_dtype)
        p_log_p = None if objective_is_ce else torch.zeros((bsz, tgt_len), device=p.device, dtype=accum_dtype)
        expected_score = torch.zeros((bsz, tgt_len), device=p.device, dtype=accum_dtype)
        log_denom = torch.full((bsz, tgt_len), float("-inf"), device=p.device, dtype=accum_dtype)

        for s_start in range(0, src_len, score_key_chunk_size):
            s_end = min(s_start + score_key_chunk_size, src_len)
            mask_chunk = None if norm_mask is None else norm_mask[:, :, s_start:s_end]
            out_chunk = None
            if score_chunk_buf is not None:
                out_chunk = score_chunk_buf[:, :, : (s_end - s_start)]
            score_chunk = _score_chunk_forward(
                q=q,
                k_chunk=k[:, s_start:s_end],
                w_chunk=w[:, s_start:s_end],
                scale=float(scale),
                score_head_chunk_size=int(score_head_chunk_size),
                out_dtype=q.dtype,
                total_src_len=src_len,
                mask_chunk=mask_chunk,
                out=out_chunk,
            )

            p_chunk = p[:, :, s_start:s_end]
            p_chunk_acc = p_chunk.to(accum_dtype)
            if not objective_is_ce:
                row_sum_p = row_sum_p + p_chunk_acc.sum(dim=-1)
                p_log_p = p_log_p + (p_chunk_acc * torch.log(p_chunk_acc + float(eps))).sum(dim=-1)
            expected_score = expected_score + (p_chunk * score_chunk).sum(dim=-1, dtype=accum_dtype)
            log_denom = torch.logaddexp(log_denom, torch.logsumexp(score_chunk, dim=-1).to(accum_dtype))

        ctx.save_for_backward(q, k, w, p, mask, row_sum_p if row_sum_p is not None else log_denom, log_denom)
        ctx.scale = float(scale)
        ctx.score_head_chunk_size = int(score_head_chunk_size)
        ctx.score_key_chunk_size = int(score_key_chunk_size)
        ctx.eps = float(eps)
        ctx.objective = objective
        if objective_is_ce:
            return (-expected_score + log_denom).mean()
        return (p_log_p - expected_score + row_sum_p * log_denom).mean()

    @staticmethod
    def backward(ctx, grad_out: Optional[torch.Tensor]):
        q, k, w, p, mask, row_sum_p, log_denom = ctx.saved_tensors
        score_key_chunk_size = int(ctx.score_key_chunk_size)
        score_head_chunk_size = int(ctx.score_head_chunk_size)
        scale = float(ctx.scale)
        objective_is_ce = getattr(ctx, "objective", "kl") == "ce"
        if grad_out is None:
            return (
                torch.zeros_like(q),
                torch.zeros_like(k),
                torch.zeros_like(w),
                None,
                None,
                None,
                None,
                None,
                None,
            )
        grad_scale = grad_out.detach().float() / float(max(1, q.shape[0] * q.shape[1]))

        if _resolve_kl_backward_mode(q) == "flash":
            try:
                dq, dk, dw = _streaming_kl_backward_flash(
                    q=q,
                    k=k,
                    w=w,
                    p=p,
                    mask=mask if mask.numel() > 0 else None,
                    row_sum_p=row_sum_p,
                    log_denom=log_denom,
                    scale=scale,
                    grad_scale=1.0,
                )
                if dq.dtype != grad_scale.dtype:
                    grad_scale_q = grad_scale.to(dq.dtype)
                else:
                    grad_scale_q = grad_scale
                dq = dq * grad_scale_q
                dk = dk * grad_scale_q.to(dk.dtype)
                dw = dw * grad_scale_q.to(dw.dtype)
                return (
                    dq,
                    dk,
                    dw,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            except Exception as exc:
                global _FLASH_FWD_FALLBACK_COUNT
                _FLASH_FWD_FALLBACK_COUNT += 1
                if _env_flag("VGGT_STREAMING_KL_FLASH_DEBUG") and _FLASH_FWD_FALLBACK_COUNT <= 8:
                    print("[streaming_kl_autograd] flash backward fallback:", repr(exc))

        bsz, tgt_len, n_heads, head_dim = q.shape
        src_len = k.shape[1]
        has_mask = mask.numel() > 0
        norm_mask = mask if has_mask else None

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dw = torch.zeros_like(w)
        log_denom_expanded = log_denom.unsqueeze(-1)
        row_sum_p_expanded = None if objective_is_ce else row_sum_p.unsqueeze(-1)
        score_chunk_buf = None
        if _resolve_score_mode(q) == "triton":
            max_src_chunk = max(1, int(score_key_chunk_size))
            score_chunk_buf = q.new_empty((bsz, tgt_len, max_src_chunk), dtype=q.dtype)

        for s_start in range(0, src_len, score_key_chunk_size):
            s_end = min(s_start + score_key_chunk_size, src_len)
            k_chunk = k[:, s_start:s_end]
            w_chunk = w[:, s_start:s_end]
            mask_chunk = None if norm_mask is None else norm_mask[:, :, s_start:s_end]
            out_chunk = None
            if score_chunk_buf is not None:
                out_chunk = score_chunk_buf[:, :, : (s_end - s_start)]
            score_chunk = _score_chunk_forward(
                q=q,
                k_chunk=k_chunk,
                w_chunk=w_chunk,
                scale=scale,
                score_head_chunk_size=score_head_chunk_size,
                out_dtype=q.dtype,
                total_src_len=src_len,
                mask_chunk=mask_chunk,
                out=out_chunk,
            )

            softmax_chunk = torch.exp(score_chunk.to(log_denom.dtype) - log_denom_expanded)
            if objective_is_ce:
                grad_score_chunk = softmax_chunk - p[:, :, s_start:s_end].to(log_denom.dtype)
            else:
                grad_score_chunk = row_sum_p_expanded * softmax_chunk - p[:, :, s_start:s_end].to(log_denom.dtype)
            grad_score_chunk = grad_score_chunk * grad_scale.to(log_denom.dtype)

            dq_chunk, dk_chunk, dw_chunk = _score_chunk_backward(
                q=q,
                k_chunk=k_chunk,
                w_chunk=w_chunk,
                grad_score_chunk=grad_score_chunk,
                scale=scale,
                score_head_chunk_size=score_head_chunk_size,
            )
            dq = dq + dq_chunk
            dk[:, s_start:s_end] = dk[:, s_start:s_end] + dk_chunk
            dw[:, s_start:s_end] = dw[:, s_start:s_end] + dw_chunk

        return (
            dq,
            dk,
            dw,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def streaming_kl_autograd_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    mask: Optional[torch.Tensor],
    scale: float,
    score_head_chunk_size: int,
    score_key_chunk_size: int,
    eps: float,
) -> torch.Tensor:
    mask_tensor = mask if mask is not None else q.new_empty(0)
    return _StreamingKLLossAutogradFunc.apply(
        q,
        k,
        w,
        p,
        mask_tensor,
        float(scale),
        int(score_head_chunk_size),
        int(score_key_chunk_size),
        float(eps),
    )
