"""
*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""

import math
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

try:
    from easyvolcap.utils.custom_flash_attn.sparse_bwd_grouped_cuda import sparse_prob_bwd_grouped
except Exception:
    sparse_prob_bwd_grouped = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    try:
        return int(value) if value else int(default)
    except ValueError:
        return int(default)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "")
    if not value:
        return str(default)
    return str(value).strip()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "")
    if not value:
        return bool(default)
    return value.lower() in ("1", "true", "yes", "y", "on")


def _should_save_prob_cache_for_bwd(
    *,
    batch: int,
    nheads: int,
    seqlen_q: int,
    num_kv_candidates: int,
    has_bias: bool,
) -> bool:
    if has_bias:
        return False
    policy = _env_str("VGGT_SPARSE_FLASH_BWD_P_CACHE_POLICY", "off").lower()
    if policy in ("", "0", "off", "none"):
        return False
    if policy not in ("1", "on", "true", "yes", "auto"):
        return False

    if policy == "auto":
        min_tokens = max(1, _env_int("VGGT_SPARSE_FLASH_BWD_P_CACHE_MIN_TOKENS", 4096))
        if int(seqlen_q) < min_tokens:
            return False

    max_topk_default = 512 if policy == "auto" else 0
    max_topk = max(0, _env_int("VGGT_SPARSE_FLASH_BWD_P_CACHE_MAX_TOPK", max_topk_default))
    if max_topk > 0 and int(num_kv_candidates) > max_topk:
        return False

    max_elems_default = 300_000_000 if policy == "auto" else 0
    max_elems = max(0, _env_int("VGGT_SPARSE_FLASH_BWD_P_CACHE_MAX_ELEMS", max_elems_default))
    if max_elems > 0:
        total_elems = int(batch) * int(nheads) * int(seqlen_q) * int(num_kv_candidates)
        if total_elems > max_elems:
            return False
    return True


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["num_kv_candidates"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    KV_POS,
    Bias,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Attn,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    stride_posb,
    stride_posm,
    stride_ab,
    stride_am,
    stride_an,
    nheads,
    seqlen_q,
    _seqlen_k,
    seqlen_q_rounded,
    num_kv_candidates,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    HAS_ATTN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)  # query的index
    off_hb = tl.program_id(1)  # head&batch的index
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    offs_m = start_m
    if offs_m >= seqlen_q:
        return
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim
    # q, k, v: [B, L, H, C]
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m * stride_qm + offs_d
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_d[None, :]
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + offs_d[None, :]
    # kv pos: [B, L, K]
    kv_pos_ptrs = KV_POS + off_b * stride_posb + offs_m * stride_posm + offs_n

    assert BIAS_TYPE == "none"
    lse_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    assert not IS_CAUSAL
    for start_n in range(0, num_kv_candidates, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_mask = (start_n + offs_n) < num_kv_candidates
        kv_indices = tl.load(kv_pos_ptrs + start_n, mask=kv_mask, other=-1)
        kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < _seqlen_k)
        kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

        _k_ptrs = k_ptrs + kv_indices_safe[:, None] * stride_kn
        _v_ptrs = v_ptrs + kv_indices_safe[:, None] * stride_vn
        k = tl.load(_k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v = tl.load(_v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        qk = tl.sum(k * q[None, :], axis=1)
        qk = tl.where(kv_valid, qk, float("-inf"))
        m_ij = tl.maximum(tl.max(qk, 0) * softmax_scale, m_i)
        p = tl.exp(qk * softmax_scale - m_ij)
        l_ij = tl.sum(p, 0)

        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale
        acc_o += tl.sum(p[:, None] * v, axis=0)

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m + tl.arange(0, 1)
    tl.store(lse_ptrs, lse_i)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_m * stride_om + offs_d
    if EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=d_mask)

    if HAS_ATTN:
        for start_n in range(0, num_kv_candidates, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            kv_mask = (start_n + offs_n) < num_kv_candidates
            kv_indices = tl.load(kv_pos_ptrs + start_n, mask=kv_mask, other=-1)
            kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < _seqlen_k)
            kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

            _k_ptrs = k_ptrs + kv_indices_safe[:, None] * stride_kn
            k = tl.load(_k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            qk = tl.sum(k * q[None, :], axis=1)
            qk = tl.where(kv_valid, qk, float("-inf"))
            p = tl.exp(qk * softmax_scale - lse_i)
            p = tl.where(kv_valid, p, 0.0)

            attn_ptrs = Attn + off_b * stride_ab + offs_m * stride_am + (start_n + offs_n) * stride_an
            tl.atomic_add(attn_ptrs, p, mask=kv_mask)


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["num_kv_candidates"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_value_gate_bhtd_kernel(
    Q,
    K,
    V,
    KV_POS,
    Gate,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_gb,
    stride_gh,
    stride_gm,
    stride_gn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_posb,
    stride_posm,
    nheads,
    seqlen_q,
    _seqlen_k,
    seqlen_q_rounded,
    num_kv_candidates,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    SHARED_GATE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m
    if offs_m >= seqlen_q:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim
    gate_head = 0 if SHARED_GATE else off_h

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m * stride_qm + offs_d
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_d[None, :]
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + offs_d[None, :]
    kv_pos_ptrs = KV_POS + off_b * stride_posb + offs_m * stride_posm + offs_n
    gate_ptrs = Gate + off_b * stride_gb + gate_head * stride_gh + offs_m * stride_gm + offs_n * stride_gn

    lse_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    for start_n in range(0, num_kv_candidates, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_mask = (start_n + offs_n) < num_kv_candidates
        kv_indices = tl.load(kv_pos_ptrs + start_n, mask=kv_mask, other=-1)
        kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < _seqlen_k)
        kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

        _k_ptrs = k_ptrs + kv_indices_safe[:, None] * stride_kn
        _v_ptrs = v_ptrs + kv_indices_safe[:, None] * stride_vn
        k = tl.load(_k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v = tl.load(_v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        g = tl.load(gate_ptrs + start_n * stride_gn, mask=kv_mask, other=1.0).to(tl.float32)

        qk = tl.sum(k * q[None, :], axis=1)
        qk = tl.where(kv_valid, qk, float("-inf"))
        m_ij = tl.maximum(tl.max(qk, 0) * softmax_scale, m_i)
        p = tl.exp(qk * softmax_scale - m_ij)
        l_ij = tl.sum(p, 0)

        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale
        acc_o += tl.sum((p * g)[:, None] * v, axis=0)

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m + tl.arange(0, 1)
    tl.store(lse_ptrs, lse_i)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + offs_m * stride_om + offs_d
    if EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o)
    else:
        tl.store(out_ptrs, acc_o, mask=d_mask)


@triton.jit
def _attn_sum_kernel(
    Q,
    K,
    KV_POS,
    Lse,
    Attn,
    Prob,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_posb,
    stride_posm,
    stride_ab,
    stride_am,
    stride_an,
    stride_pb,
    stride_ph,
    stride_pm,
    stride_pn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    num_kv_candidates,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    HAS_PROB: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    if start_m >= seqlen_q:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + start_m * stride_qm + offs_d
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    lse_ptr = Lse + off_hb * seqlen_q_rounded + start_m
    lse = tl.load(lse_ptr).to(tl.float32)

    for start_n in range(0, num_kv_candidates, BLOCK_N):
        kv_idx_ptrs = KV_POS + off_b * stride_posb + start_m * stride_posm + start_n + offs_n
        kv_mask = (start_n + offs_n) < num_kv_candidates
        kv_indices = tl.load(kv_idx_ptrs, mask=kv_mask, other=-1)
        kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < seqlen_k)
        kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices_safe[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        qk = tl.sum(k * q[None, :], axis=1)
        qk = tl.where(kv_valid, qk, float("-inf"))

        p = tl.exp(qk * softmax_scale - lse)
        p = tl.where(kv_valid, p, 0.0)

        attn_ptrs = (
            Attn
            + off_b * stride_ab
            + start_m * stride_am
            + (start_n + offs_n) * stride_an
        )
        tl.atomic_add(attn_ptrs, p, mask=kv_mask)
        if HAS_PROB:
            prob_ptrs = (
                Prob
                + off_b * stride_pb
                + off_h * stride_ph
                + start_m * stride_pm
                + (start_n + offs_n) * stride_pn
            )
            tl.store(prob_ptrs, p, mask=kv_mask)


@triton.jit
def _attn_prob_kernel(
    Q,
    K,
    KV_POS,
    Lse,
    Prob,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_posb,
    stride_posm,
    stride_pb,
    stride_ph,
    stride_pm,
    stride_pn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    num_kv_candidates,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    if start_m >= seqlen_q:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + start_m * stride_qm + offs_d
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    lse_ptr = Lse + off_hb * seqlen_q_rounded + start_m
    lse = tl.load(lse_ptr).to(tl.float32)

    for start_n in range(0, num_kv_candidates, BLOCK_N):
        kv_idx_ptrs = KV_POS + off_b * stride_posb + start_m * stride_posm + start_n + offs_n
        kv_mask = (start_n + offs_n) < num_kv_candidates
        kv_indices = tl.load(kv_idx_ptrs, mask=kv_mask, other=-1)
        kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < seqlen_k)
        kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices_safe[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        qk = tl.sum(k * q[None, :], axis=1)
        qk = tl.where(kv_valid, qk, float("-inf"))

        p = tl.exp(qk * softmax_scale - lse)
        p = tl.where(kv_valid, p, 0.0)

        prob_ptrs = (
            Prob
            + off_b * stride_pb
            + off_h * stride_ph
            + start_m * stride_pm
            + (start_n + offs_n) * stride_pn
        )
        tl.store(prob_ptrs, p, mask=kv_mask)


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_sparse_index_kernel(
    Q,
    K,
    V,
    KV_POS,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    DATTN,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_posb,
    stride_posm,
    stride_dab,
    stride_dam,
    stride_dan,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    num_kv_candidates,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    HAS_DATTN: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m
    if offs_m >= seqlen_q:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m * stride_qm + offs_d
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + offs_m * stride_dom + offs_d
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + offs_m * stride_dqm + offs_d

    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    lse_ptr = LSE + off_hb * seqlen_q_rounded + offs_m
    delta_ptr = D + off_hb * seqlen_q_rounded + offs_m
    lse = tl.load(lse_ptr).to(tl.float32)
    delta = tl.load(delta_ptr).to(tl.float32)

    sum_dattn_p = tl.zeros([1], dtype=tl.float32)
    if HAS_DATTN:
        for start_n in range(0, num_kv_candidates, BLOCK_N):
            kv_idx_ptrs = KV_POS + off_b * stride_posb + offs_m * stride_posm + start_n + offs_n
            kv_mask = (start_n + offs_n) < num_kv_candidates
            kv_indices = tl.load(kv_idx_ptrs, mask=kv_mask, other=-1)
            kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < seqlen_k)
            kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

            k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices_safe[:, None] * stride_kn + offs_d[None, :]
            k = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            qk = tl.sum(k * q[None, :], axis=1)
            qk = tl.where(kv_valid, qk, float("-inf"))

            if BIAS_TYPE == "matrix":
                b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_m * stride_bm + start_n + offs_n
                bias = tl.load(b_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
                qk = qk + bias
            elif BIAS_TYPE == "vector":
                b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + start_n + offs_n
                bias = tl.load(b_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
                qk = qk + bias

            p = tl.exp(qk * softmax_scale - lse)
            p = tl.where(kv_valid, p, 0.0)

            dattn_ptrs = DATTN + off_b * stride_dab + offs_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            sum_dattn_p += tl.sum(dattn * p)

    total = delta + (sum_dattn_p if HAS_DATTN else 0.0)
    dq_accum = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, num_kv_candidates, BLOCK_N):
        kv_idx_ptrs = KV_POS + off_b * stride_posb + offs_m * stride_posm + start_n + offs_n
        kv_mask = (start_n + offs_n) < num_kv_candidates
        kv_indices = tl.load(kv_idx_ptrs, mask=kv_mask, other=-1)
        kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < seqlen_k)
        kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices_safe[:, None] * stride_kn + offs_d[None, :]
        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + kv_indices_safe[:, None] * stride_vn + offs_d[None, :]

        k = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        qk = tl.sum(k * q[None, :], axis=1)
        qk = tl.where(kv_valid, qk, float("-inf"))

        if BIAS_TYPE == "matrix":
            b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_m * stride_bm + start_n + offs_n
            bias = tl.load(b_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            qk = qk + bias
        elif BIAS_TYPE == "vector":
            b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + start_n + offs_n
            bias = tl.load(b_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            qk = qk + bias

        p = tl.exp(qk * softmax_scale - lse)
        p = tl.where(kv_valid, p, 0.0)

        dp = tl.sum(v * do[None, :], axis=1)
        if HAS_DATTN:
            dattn_ptrs = DATTN + off_b * stride_dab + offs_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            dp = dp + dattn

        ds = (dp - total) * p
        dq_accum += tl.sum(ds[:, None] * k, axis=0)

        dk = ds[:, None] * q[None, :]
        dv = p[:, None] * do[None, :]
        dk = dk * softmax_scale

        dk_ptrs = DK + off_b * stride_dkb + off_h * stride_dkh + kv_indices_safe[:, None] * stride_dkn + offs_d[None, :]
        dv_ptrs = DV + off_b * stride_dvb + off_h * stride_dvh + kv_indices_safe[:, None] * stride_dvn + offs_d[None, :]

        tl.atomic_add(dk_ptrs, dk, mask=kv_valid[:, None] & d_mask[None, :])
        tl.atomic_add(dv_ptrs, dv, mask=kv_valid[:, None] & d_mask[None, :])

    dq_accum = dq_accum * softmax_scale
    tl.store(dq_ptrs, dq_accum, mask=d_mask)


@triton.jit
def _bwd_sparse_index_cached_p_kernel(
    Q,
    K,
    V,
    KV_POS,
    DO,
    DQ,
    DK,
    DV,
    D,
    Prob,
    DATTN,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_posb,
    stride_posm,
    stride_pb,
    stride_ph,
    stride_pm,
    stride_pn,
    stride_dab,
    stride_dam,
    stride_dan,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    num_kv_candidates,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    HAS_DATTN: tl.constexpr,
    LOWP_DKDV: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    if start_m >= seqlen_q:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + start_m * stride_qm + offs_d
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + start_m * stride_dom + offs_d
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + start_m * stride_dqm + offs_d

    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    delta_ptr = D + off_hb * seqlen_q_rounded + start_m
    delta = tl.load(delta_ptr).to(tl.float32)

    sum_dattn_p = tl.zeros([1], dtype=tl.float32)
    if HAS_DATTN:
        for start_n in range(0, num_kv_candidates, BLOCK_N):
            kv_mask = (start_n + offs_n) < num_kv_candidates
            prob_ptrs = (
                Prob
                + off_b * stride_pb
                + off_h * stride_ph
                + start_m * stride_pm
                + (start_n + offs_n) * stride_pn
            )
            p = tl.load(prob_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            dattn_ptrs = DATTN + off_b * stride_dab + start_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            sum_dattn_p += tl.sum(dattn * p)

    total = delta + (sum_dattn_p if HAS_DATTN else 0.0)
    dq_accum = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, num_kv_candidates, BLOCK_N):
        kv_idx_ptrs = KV_POS + off_b * stride_posb + start_m * stride_posm + start_n + offs_n
        kv_mask = (start_n + offs_n) < num_kv_candidates
        kv_indices = tl.load(kv_idx_ptrs, mask=kv_mask, other=-1)
        kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < seqlen_k)
        kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

        prob_ptrs = (
            Prob
            + off_b * stride_pb
            + off_h * stride_ph
            + start_m * stride_pm
            + (start_n + offs_n) * stride_pn
        )
        p = tl.load(prob_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        p = tl.where(kv_valid, p, 0.0)

        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + kv_indices_safe[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        dp = tl.sum(v * do[None, :], axis=1)
        if HAS_DATTN:
            dattn_ptrs = DATTN + off_b * stride_dab + start_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            dp = dp + dattn

        ds = (dp - total) * p

        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices_safe[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        dq_accum += tl.sum(ds[:, None] * k, axis=0)

        if LOWP_DKDV:
            ds_h = ds.to(tl.float16)
            p_h = p.to(tl.float16)
            q_h = q.to(tl.float16)
            do_h = do.to(tl.float16)
            dk = (ds_h[:, None] * q_h[None, :]) * softmax_scale
            dv = p_h[:, None] * do_h[None, :]
        else:
            dk = ds[:, None] * q[None, :]
            dk = dk * softmax_scale
            dv = p[:, None] * do[None, :]

        dk_ptrs = DK + off_b * stride_dkb + off_h * stride_dkh + kv_indices_safe[:, None] * stride_dkn + offs_d[None, :]
        dv_ptrs = DV + off_b * stride_dvb + off_h * stride_dvh + kv_indices_safe[:, None] * stride_dvn + offs_d[None, :]

        tl.atomic_add(dk_ptrs, dk, mask=kv_valid[:, None] & d_mask[None, :])
        tl.atomic_add(dv_ptrs, dv, mask=kv_valid[:, None] & d_mask[None, :])

    dq_accum = dq_accum * softmax_scale
    tl.store(dq_ptrs, dq_accum, mask=d_mask)


@triton.jit
def _bwd_sparse_index_cached_p_fast_kernel(
    Q,
    K,
    V,
    KV_POS,
    DO,
    DQ,
    DK,
    DV,
    D,
    Prob,
    DATTN,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_posb,
    stride_posm,
    stride_pb,
    stride_ph,
    stride_pm,
    stride_pn,
    stride_dab,
    stride_dam,
    stride_dan,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    num_kv_candidates,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    HAS_DATTN: tl.constexpr,
    LOWP_DKDV: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Fast path: assumes KV_POS values are always valid [0, seqlen_k).
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    if start_m >= seqlen_q:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + start_m * stride_qm + offs_d
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + start_m * stride_dom + offs_d
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + start_m * stride_dqm + offs_d

    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    delta_ptr = D + off_hb * seqlen_q_rounded + start_m
    delta = tl.load(delta_ptr).to(tl.float32)

    sum_dattn_p = tl.zeros([1], dtype=tl.float32)
    if HAS_DATTN:
        for start_n in range(0, num_kv_candidates, BLOCK_N):
            kv_mask = (start_n + offs_n) < num_kv_candidates
            prob_ptrs = (
                Prob
                + off_b * stride_pb
                + off_h * stride_ph
                + start_m * stride_pm
                + (start_n + offs_n) * stride_pn
            )
            p = tl.load(prob_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            dattn_ptrs = DATTN + off_b * stride_dab + start_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            sum_dattn_p += tl.sum(dattn * p)

    total = delta + (sum_dattn_p if HAS_DATTN else 0.0)
    dq_accum = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, num_kv_candidates, BLOCK_N):
        kv_mask = (start_n + offs_n) < num_kv_candidates
        kv_idx_ptrs = KV_POS + off_b * stride_posb + start_m * stride_posm + start_n + offs_n
        kv_indices = tl.load(kv_idx_ptrs, mask=kv_mask, other=0).to(tl.int32)

        prob_ptrs = (
            Prob
            + off_b * stride_pb
            + off_h * stride_ph
            + start_m * stride_pm
            + (start_n + offs_n) * stride_pn
        )
        p = tl.load(prob_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + kv_indices[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        dp = tl.sum(v * do[None, :], axis=1)
        if HAS_DATTN:
            dattn_ptrs = DATTN + off_b * stride_dab + start_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            dp = dp + dattn

        ds = (dp - total) * p

        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        dq_accum += tl.sum(ds[:, None] * k, axis=0)

        if LOWP_DKDV:
            ds_h = ds.to(tl.float16)
            p_h = p.to(tl.float16)
            q_h = q.to(tl.float16)
            do_h = do.to(tl.float16)
            dk = (ds_h[:, None] * q_h[None, :]) * softmax_scale
            dv = p_h[:, None] * do_h[None, :]
        else:
            dk = (ds[:, None] * q[None, :]) * softmax_scale
            dv = p[:, None] * do[None, :]

        dk_ptrs = DK + off_b * stride_dkb + off_h * stride_dkh + kv_indices[:, None] * stride_dkn + offs_d[None, :]
        dv_ptrs = DV + off_b * stride_dvb + off_h * stride_dvh + kv_indices[:, None] * stride_dvn + offs_d[None, :]
        tl.atomic_add(dk_ptrs, dk, mask=kv_mask[:, None] & d_mask[None, :])
        tl.atomic_add(dv_ptrs, dv, mask=kv_mask[:, None] & d_mask[None, :])

    dq_accum = dq_accum * softmax_scale
    tl.store(dq_ptrs, dq_accum, mask=d_mask)


@triton.jit
def _bwd_sparse_index_cached_p_fast_topk_kernel(
    Q,
    K,
    V,
    KV_POS,
    DO,
    DQ,
    DK,
    DV,
    D,
    Prob,
    DATTN,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_posb,
    stride_posm,
    stride_pb,
    stride_ph,
    stride_pm,
    stride_pn,
    stride_dab,
    stride_dam,
    stride_dan,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    NUM_TOPK: tl.constexpr,
    HAS_DATTN: tl.constexpr,
    LOWP_DKDV: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Fastest path for fixed topk (e.g. 512):
    # - KV_POS assumed valid [0, seqlen_k)
    # - NUM_TOPK fixed, so no tail mask branch in loop
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    if start_m >= seqlen_q:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + start_m * stride_qm + offs_d
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + start_m * stride_dom + offs_d
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + start_m * stride_dqm + offs_d

    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    delta_ptr = D + off_hb * seqlen_q_rounded + start_m
    delta = tl.load(delta_ptr).to(tl.float32)

    sum_dattn_p = tl.zeros([1], dtype=tl.float32)
    if HAS_DATTN:
        for start_n in range(0, NUM_TOPK, BLOCK_N):
            prob_ptrs = (
                Prob
                + off_b * stride_pb
                + off_h * stride_ph
                + start_m * stride_pm
                + (start_n + offs_n) * stride_pn
            )
            p = tl.load(prob_ptrs).to(tl.float32)
            dattn_ptrs = DATTN + off_b * stride_dab + start_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs).to(tl.float32)
            sum_dattn_p += tl.sum(dattn * p)

    total = delta + (sum_dattn_p if HAS_DATTN else 0.0)
    dq_accum = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, NUM_TOPK, BLOCK_N):
        kv_idx_ptrs = KV_POS + off_b * stride_posb + start_m * stride_posm + start_n + offs_n
        kv_indices = tl.load(kv_idx_ptrs).to(tl.int32)

        prob_ptrs = (
            Prob
            + off_b * stride_pb
            + off_h * stride_ph
            + start_m * stride_pm
            + (start_n + offs_n) * stride_pn
        )
        p = tl.load(prob_ptrs).to(tl.float32)

        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + kv_indices[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
        dp = tl.sum(v * do[None, :], axis=1)
        if HAS_DATTN:
            dattn_ptrs = DATTN + off_b * stride_dab + start_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs).to(tl.float32)
            dp = dp + dattn

        ds = (dp - total) * p

        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
        dq_accum += tl.sum(ds[:, None] * k, axis=0)

        if LOWP_DKDV:
            ds_h = ds.to(tl.float16)
            p_h = p.to(tl.float16)
            q_h = q.to(tl.float16)
            do_h = do.to(tl.float16)
            dk = (ds_h[:, None] * q_h[None, :]) * softmax_scale
            dv = p_h[:, None] * do_h[None, :]
        else:
            dk = (ds[:, None] * q[None, :]) * softmax_scale
            dv = p[:, None] * do[None, :]

        dk_ptrs = DK + off_b * stride_dkb + off_h * stride_dkh + kv_indices[:, None] * stride_dkn + offs_d[None, :]
        dv_ptrs = DV + off_b * stride_dvb + off_h * stride_dvh + kv_indices[:, None] * stride_dvn + offs_d[None, :]
        tl.atomic_add(dk_ptrs, dk, mask=d_mask[None, :])
        tl.atomic_add(dv_ptrs, dv, mask=d_mask[None, :])

    dq_accum = dq_accum * softmax_scale
    tl.store(dq_ptrs, dq_accum, mask=d_mask)


@triton.jit
def _bwd_sparse_index_value_gate_kernel(
    Q,
    K,
    V,
    KV_POS,
    Gate,
    DO,
    DQ,
    DK,
    DV,
    DGATE,
    LSE,
    D,
    DATTN,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_gb,
    stride_gh,
    stride_gm,
    stride_gn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dgb,
    stride_dgh,
    stride_dgm,
    stride_dgn,
    stride_posb,
    stride_posm,
    stride_dab,
    stride_dam,
    stride_dan,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    num_kv_candidates,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    SHARED_GATE: tl.constexpr,
    HAS_DATTN: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m
    if offs_m >= seqlen_q:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim
    gate_head = 0 if SHARED_GATE else off_h

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + offs_m * stride_qm + offs_d
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + offs_m * stride_dom + offs_d
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + offs_m * stride_dqm + offs_d

    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    lse_ptr = LSE + off_hb * seqlen_q_rounded + offs_m
    delta_ptr = D + off_hb * seqlen_q_rounded + offs_m
    lse = tl.load(lse_ptr).to(tl.float32)
    delta = tl.load(delta_ptr).to(tl.float32)

    sum_dattn_p = tl.zeros([1], dtype=tl.float32)
    if HAS_DATTN:
        for start_n in range(0, num_kv_candidates, BLOCK_N):
            kv_idx_ptrs = KV_POS + off_b * stride_posb + offs_m * stride_posm + start_n + offs_n
            kv_mask = (start_n + offs_n) < num_kv_candidates
            kv_indices = tl.load(kv_idx_ptrs, mask=kv_mask, other=-1)
            kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < seqlen_k)
            kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

            k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices_safe[:, None] * stride_kn + offs_d[None, :]
            k = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            qk = tl.sum(k * q[None, :], axis=1)
            qk = tl.where(kv_valid, qk, float("-inf"))
            p = tl.exp(qk * softmax_scale - lse)
            p = tl.where(kv_valid, p, 0.0)

            dattn_ptrs = DATTN + off_b * stride_dab + offs_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            sum_dattn_p += tl.sum(dattn * p)

    total = delta + (sum_dattn_p if HAS_DATTN else 0.0)
    dq_accum = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, num_kv_candidates, BLOCK_N):
        kv_idx_ptrs = KV_POS + off_b * stride_posb + offs_m * stride_posm + start_n + offs_n
        kv_mask = (start_n + offs_n) < num_kv_candidates
        kv_indices = tl.load(kv_idx_ptrs, mask=kv_mask, other=-1)
        kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < seqlen_k)
        kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

        gate_ptrs = Gate + off_b * stride_gb + gate_head * stride_gh + offs_m * stride_gm + (start_n + offs_n) * stride_gn
        dgate_ptrs = DGATE + off_b * stride_dgb + gate_head * stride_dgh + offs_m * stride_dgm + (start_n + offs_n) * stride_dgn
        g = tl.load(gate_ptrs, mask=kv_mask, other=1.0).to(tl.float32)

        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices_safe[:, None] * stride_kn + offs_d[None, :]
        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + kv_indices_safe[:, None] * stride_vn + offs_d[None, :]

        k = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        qk = tl.sum(k * q[None, :], axis=1)
        qk = tl.where(kv_valid, qk, float("-inf"))
        p = tl.exp(qk * softmax_scale - lse)
        p = tl.where(kv_valid, p, 0.0)

        dp_base = tl.sum(v * do[None, :], axis=1)
        dp = dp_base * g
        if HAS_DATTN:
            dattn_ptrs = DATTN + off_b * stride_dab + offs_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            dp = dp + dattn

        ds = (dp - total) * p
        dq_accum += tl.sum(ds[:, None] * k, axis=0)

        dk = ds[:, None] * q[None, :]
        dv = (p * g)[:, None] * do[None, :]
        dg = p * dp_base
        dk = dk * softmax_scale

        dk_ptrs = DK + off_b * stride_dkb + off_h * stride_dkh + kv_indices_safe[:, None] * stride_dkn + offs_d[None, :]
        dv_ptrs = DV + off_b * stride_dvb + off_h * stride_dvh + kv_indices_safe[:, None] * stride_dvn + offs_d[None, :]

        tl.atomic_add(dk_ptrs, dk, mask=kv_valid[:, None] & d_mask[None, :])
        tl.atomic_add(dv_ptrs, dv, mask=kv_valid[:, None] & d_mask[None, :])
        tl.atomic_add(dgate_ptrs, dg, mask=kv_mask)

    dq_accum = dq_accum * softmax_scale
    tl.store(dq_ptrs, dq_accum, mask=d_mask)


@triton.jit
def _bwd_sparse_index_cached_p_value_gate_kernel(
    Q,
    K,
    V,
    KV_POS,
    Gate,
    DO,
    DQ,
    DK,
    DV,
    DGATE,
    D,
    Prob,
    DATTN,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_gb,
    stride_gh,
    stride_gm,
    stride_gn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dgb,
    stride_dgh,
    stride_dgm,
    stride_dgn,
    stride_posb,
    stride_posm,
    stride_pb,
    stride_ph,
    stride_pm,
    stride_pn,
    stride_dab,
    stride_dam,
    stride_dan,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    num_kv_candidates,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    SHARED_GATE: tl.constexpr,
    HAS_DATTN: tl.constexpr,
    LOWP_DKDV: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    if start_m >= seqlen_q:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    d_mask = offs_d < headdim
    gate_head = 0 if SHARED_GATE else off_h

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + start_m * stride_qm + offs_d
    do_ptrs = DO + off_b * stride_dob + off_h * stride_doh + start_m * stride_dom + offs_d
    dq_ptrs = DQ + off_b * stride_dqb + off_h * stride_dqh + start_m * stride_dqm + offs_d

    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    delta_ptr = D + off_hb * seqlen_q_rounded + start_m
    delta = tl.load(delta_ptr).to(tl.float32)

    sum_dattn_p = tl.zeros([1], dtype=tl.float32)
    if HAS_DATTN:
        for start_n in range(0, num_kv_candidates, BLOCK_N):
            kv_mask = (start_n + offs_n) < num_kv_candidates
            prob_ptrs = (
                Prob
                + off_b * stride_pb
                + off_h * stride_ph
                + start_m * stride_pm
                + (start_n + offs_n) * stride_pn
            )
            p = tl.load(prob_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            dattn_ptrs = DATTN + off_b * stride_dab + start_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            sum_dattn_p += tl.sum(dattn * p)

    total = delta + (sum_dattn_p if HAS_DATTN else 0.0)
    dq_accum = tl.zeros([BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, num_kv_candidates, BLOCK_N):
        kv_idx_ptrs = KV_POS + off_b * stride_posb + start_m * stride_posm + start_n + offs_n
        kv_mask = (start_n + offs_n) < num_kv_candidates
        kv_indices = tl.load(kv_idx_ptrs, mask=kv_mask, other=-1)
        kv_valid = (kv_indices >= 0) & kv_mask & (kv_indices < seqlen_k)
        kv_indices_safe = tl.where(kv_valid, kv_indices, 0)

        gate_ptrs = Gate + off_b * stride_gb + gate_head * stride_gh + start_m * stride_gm + (start_n + offs_n) * stride_gn
        dgate_ptrs = DGATE + off_b * stride_dgb + gate_head * stride_dgh + start_m * stride_dgm + (start_n + offs_n) * stride_dgn
        g = tl.load(gate_ptrs, mask=kv_mask, other=1.0).to(tl.float32)

        prob_ptrs = (
            Prob
            + off_b * stride_pb
            + off_h * stride_ph
            + start_m * stride_pm
            + (start_n + offs_n) * stride_pn
        )
        p = tl.load(prob_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        p = tl.where(kv_valid, p, 0.0)

        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + kv_indices_safe[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        dp_base = tl.sum(v * do[None, :], axis=1)
        dp = dp_base * g
        if HAS_DATTN:
            dattn_ptrs = DATTN + off_b * stride_dab + start_m * stride_dam + start_n + offs_n
            dattn = tl.load(dattn_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            dp = dp + dattn

        ds = (dp - total) * p

        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + kv_indices_safe[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        dq_accum += tl.sum(ds[:, None] * k, axis=0)

        if LOWP_DKDV:
            ds_h = ds.to(tl.float16)
            p_h = p.to(tl.float16)
            g_h = g.to(tl.float16)
            q_h = q.to(tl.float16)
            do_h = do.to(tl.float16)
            dk = (ds_h[:, None] * q_h[None, :]) * softmax_scale
            dv = (p_h * g_h)[:, None] * do_h[None, :]
        else:
            dk = (ds[:, None] * q[None, :]) * softmax_scale
            dv = (p * g)[:, None] * do[None, :]
        dg = p * dp_base

        dk_ptrs = DK + off_b * stride_dkb + off_h * stride_dkh + kv_indices_safe[:, None] * stride_dkn + offs_d[None, :]
        dv_ptrs = DV + off_b * stride_dvb + off_h * stride_dvh + kv_indices_safe[:, None] * stride_dvn + offs_d[None, :]
        tl.atomic_add(dk_ptrs, dk, mask=kv_valid[:, None] & d_mask[None, :])
        tl.atomic_add(dv_ptrs, dv, mask=kv_valid[:, None] & d_mask[None, :])
        tl.atomic_add(dgate_ptrs, dg, mask=kv_mask)

    dq_accum = dq_accum * softmax_scale
    tl.store(dq_ptrs, dq_accum, mask=d_mask)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == "none":
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        # compute dq
        if not (
            EVEN_M & EVEN_HEADDIM
        ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _flash_attn_forward(
    q,
    k,
    v,
    kv_positions,
    bias=None,
    causal=False,
    softmax_scale=None,
    return_attn=True,
    save_prob_for_bwd: bool = False,
    q_layout: str = "bthd",
):
    # shape constraints
    if q_layout == "bthd":
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        assert k.shape == (batch, seqlen_k, nheads, d)
        assert v.shape == (batch, seqlen_k, nheads, d)
        q_stride_h, q_stride_m = q.stride(2), q.stride(1)
        k_stride_h, k_stride_n = k.stride(2), k.stride(1)
        v_stride_h, v_stride_n = v.stride(2), v.stride(1)
    elif q_layout == "bhtd":
        batch, nheads, seqlen_q, d = q.shape
        _, _, seqlen_k, _ = k.shape
        assert k.shape == (batch, nheads, seqlen_k, d)
        assert v.shape == (batch, nheads, seqlen_k, d)
        q_stride_h, q_stride_m = q.stride(1), q.stride(2)
        k_stride_h, k_stride_n = k.stride(1), k.stride(2)
        v_stride_h, v_stride_n = v.stride(1), v.stride(2)
    else:
        raise ValueError(f"Unsupported q_layout: {q_layout}")

    num_kv_candidates = kv_positions.shape[-1]
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert kv_positions.dtype == torch.int32
    assert kv_positions.shape == (batch, seqlen_q, num_kv_candidates)

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    # Keep output contiguous to avoid pathological preserved strides when q/k/v are
    # non-contiguous views (e.g. train no_qkv_contig fast path).
    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    attn_sum = None
    prob_cache = None
    use_fused_attn_sum = return_attn and (_env_int("VGGT_SPARSE_FLASH_FUSED_ATTN_SUM", 0) != 0)
    if return_attn:
        attn_sum = torch.zeros(
            (batch, seqlen_q, num_kv_candidates),
            device=q.device,
            dtype=torch.float32,
        )
    if save_prob_for_bwd:
        prob_cache = torch.empty(
            (batch, nheads, seqlen_q, num_kv_candidates),
            device=q.device,
            dtype=q.dtype,
        )
    if use_fused_attn_sum:
        attn_ptr = attn_sum
        attn_strides = (attn_sum.stride(0), attn_sum.stride(1), attn_sum.stride(2))
    else:
        # Placeholder pointer/strides when fused attn_sum path is disabled.
        attn_ptr = q
        attn_strides = (0, 0, 0)
    if q_layout == "bthd":
        o_stride_h, o_stride_m = o.stride(2), o.stride(1)
    else:
        o_stride_h, o_stride_m = o.stride(1), o.stride(2)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    block_n = max(32, _env_int("VGGT_SPARSE_FLASH_FWD_BLOCK_N", 128))
    num_warps_default = 4 if d <= 64 else 8
    num_warps = max(1, _env_int("VGGT_SPARSE_FLASH_FWD_NUM_WARPS", num_warps_default))
    num_stages = max(1, _env_int("VGGT_SPARSE_FLASH_FWD_NUM_STAGES", 1))
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        kv_positions,
        bias,
        o,
        lse,
        tmp,
        attn_ptr,
        softmax_scale,
        q.stride(0),
        q_stride_h,
        q_stride_m,
        k.stride(0),
        k_stride_h,
        k_stride_n,
        v.stride(0),
        v_stride_h,
        v_stride_n,
        *bias_strides,
        o.stride(0),
        o_stride_h,
        o_stride_m,
        kv_positions.stride(0),
        kv_positions.stride(1),
        *attn_strides,
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        num_kv_candidates,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        HAS_ATTN=use_fused_attn_sum,
        BLOCK_M=1,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if return_attn and (not use_fused_attn_sum):
        grid_attn = lambda META: (seqlen_q, batch * nheads)
        attn_block_n = max(32, _env_int("VGGT_SPARSE_FLASH_ATTN_BLOCK_N", 128))
        attn_num_warps = max(1, _env_int("VGGT_SPARSE_FLASH_ATTN_NUM_WARPS", num_warps_default))
        attn_num_stages = max(1, _env_int("VGGT_SPARSE_FLASH_ATTN_NUM_STAGES", 1))
        has_prob_inline = prob_cache is not None
        prob_inline = prob_cache if has_prob_inline else q
        prob_inline_strides = (
            prob_cache.stride(0),
            prob_cache.stride(1),
            prob_cache.stride(2),
            prob_cache.stride(3),
        ) if has_prob_inline else (0, 0, 0, 0)
        _attn_sum_kernel[grid_attn](
            q,
            k,
            kv_positions,
            lse,
            attn_sum,
            prob_inline,
            softmax_scale,
            q.stride(0),
            q_stride_h,
            q_stride_m,
            k.stride(0),
            k_stride_h,
            k_stride_n,
            kv_positions.stride(0),
            kv_positions.stride(1),
            attn_sum.stride(0),
            attn_sum.stride(1),
            attn_sum.stride(2),
            *prob_inline_strides,
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            num_kv_candidates,
            d,
            seqlen_q // 32,
            seqlen_k // 32,
            HAS_PROB=has_prob_inline,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_N=attn_block_n,
            num_warps=attn_num_warps,
            num_stages=attn_num_stages,
        )
    # When attn_sum is produced by a dedicated kernel, prob_cache can be written inline there.
    # Fallback to a standalone prob kernel only when attn_sum uses fused fwd path (or attn is skipped).
    if prob_cache is not None and (use_fused_attn_sum or (not return_attn)):
        grid_prob = lambda META: (seqlen_q, batch * nheads)
        prob_block_n = max(32, _env_int("VGGT_SPARSE_FLASH_PROB_BLOCK_N", 128))
        prob_num_warps = max(1, _env_int("VGGT_SPARSE_FLASH_PROB_NUM_WARPS", num_warps_default))
        prob_num_stages = max(1, _env_int("VGGT_SPARSE_FLASH_PROB_NUM_STAGES", 1))
        _attn_prob_kernel[grid_prob](
            q,
            k,
            kv_positions,
            lse,
            prob_cache,
            softmax_scale,
            q.stride(0),
            q_stride_h,
            q_stride_m,
            k.stride(0),
            k_stride_h,
            k_stride_n,
            kv_positions.stride(0),
            kv_positions.stride(1),
            prob_cache.stride(0),
            prob_cache.stride(1),
            prob_cache.stride(2),
            prob_cache.stride(3),
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            num_kv_candidates,
            d,
            seqlen_q // 32,
            seqlen_k // 32,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_N=prob_block_n,
            num_warps=prob_num_warps,
            num_stages=prob_num_stages,
        )
    return o, attn_sum, lse, softmax_scale, prob_cache  # softmax_scale could have been updated


def _prepare_value_gate_bhtd(
    value_gate: torch.Tensor,
    *,
    batch: int,
    nheads: int,
    seqlen_q: int,
    num_kv_candidates: int,
    q_dtype: torch.dtype,
) -> Tuple[torch.Tensor, bool]:
    if value_gate.dim() == 3:
        value_gate = value_gate.unsqueeze(1)
    assert value_gate.dim() == 4
    gate_heads = int(value_gate.shape[1])
    if gate_heads not in (1, nheads):
        raise ValueError(f"value_gate head dim must be 1 or nheads ({nheads}), got {gate_heads}")
    expect_shape = (batch, gate_heads, seqlen_q, num_kv_candidates)
    if tuple(value_gate.shape) != expect_shape:
        raise ValueError(f"value_gate shape mismatch: got={tuple(value_gate.shape)}, expected={expect_shape}")
    if value_gate.dtype not in (q_dtype, torch.float32):
        value_gate = value_gate.to(q_dtype)
    if value_gate.stride(-1) != 1:
        value_gate = value_gate.contiguous()
    if not value_gate.is_cuda:
        raise ValueError("value_gate must be CUDA tensor")
    return value_gate, gate_heads == 1


def _flash_attn_forward_value_gate_bhtd(
    q,
    k,
    v,
    kv_positions,
    value_gate,
    softmax_scale=None,
    return_attn=True,
    save_prob_for_bwd: bool = False,
):
    batch, nheads, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape
    assert k.shape == (batch, nheads, seqlen_k, d)
    assert v.shape == (batch, nheads, seqlen_k, d)

    num_kv_candidates = kv_positions.shape[-1]
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert kv_positions.dtype == torch.int32
    assert kv_positions.shape == (batch, seqlen_q, num_kv_candidates)

    value_gate, shared_gate = _prepare_value_gate_bhtd(
        value_gate,
        batch=batch,
        nheads=nheads,
        seqlen_q=seqlen_q,
        num_kv_candidates=num_kv_candidates,
        q_dtype=q.dtype,
    )

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    attn_sum = None
    prob_cache = None
    if return_attn:
        attn_sum = torch.zeros((batch, seqlen_q, num_kv_candidates), device=q.device, dtype=torch.float32)
    if save_prob_for_bwd:
        prob_cache = torch.empty((batch, nheads, seqlen_q, num_kv_candidates), device=q.device, dtype=q.dtype)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    block_n = max(32, _env_int("VGGT_SPARSE_FLASH_FWD_BLOCK_N", 128))
    num_warps_default = 4 if d <= 64 else 8
    num_warps = max(1, _env_int("VGGT_SPARSE_FLASH_FWD_NUM_WARPS", num_warps_default))
    num_stages = max(1, _env_int("VGGT_SPARSE_FLASH_FWD_NUM_STAGES", 1))
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_value_gate_bhtd_kernel[grid](
        q,
        k,
        v,
        kv_positions,
        value_gate,
        o,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        value_gate.stride(0),
        value_gate.stride(1),
        value_gate.stride(2),
        value_gate.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_positions.stride(0),
        kv_positions.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        num_kv_candidates,
        d,
        seqlen_q // 32,
        seqlen_k // 32,
        shared_gate,
        BLOCK_HEADDIM,
        BLOCK_M=1,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if return_attn:
        attn_block_n = max(32, _env_int("VGGT_SPARSE_FLASH_ATTN_BLOCK_N", 128))
        attn_num_warps = max(1, _env_int("VGGT_SPARSE_FLASH_ATTN_NUM_WARPS", num_warps_default))
        attn_num_stages = max(1, _env_int("VGGT_SPARSE_FLASH_ATTN_NUM_STAGES", 1))
        has_prob_inline = prob_cache is not None
        prob_inline = prob_cache if has_prob_inline else q
        prob_inline_strides = (
            prob_cache.stride(0),
            prob_cache.stride(1),
            prob_cache.stride(2),
            prob_cache.stride(3),
        ) if has_prob_inline else (0, 0, 0, 0)
        grid_attn = lambda META: (seqlen_q, batch * nheads)
        _attn_sum_kernel[grid_attn](
            q,
            k,
            kv_positions,
            lse,
            attn_sum,
            prob_inline,
            softmax_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            kv_positions.stride(0),
            kv_positions.stride(1),
            attn_sum.stride(0),
            attn_sum.stride(1),
            attn_sum.stride(2),
            *prob_inline_strides,
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            num_kv_candidates,
            d,
            seqlen_q // 32,
            seqlen_k // 32,
            HAS_PROB=has_prob_inline,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_N=attn_block_n,
            num_warps=attn_num_warps,
            num_stages=attn_num_stages,
        )
    elif prob_cache is not None:
        prob_block_n = max(32, _env_int("VGGT_SPARSE_FLASH_PROB_BLOCK_N", 128))
        prob_num_warps = max(1, _env_int("VGGT_SPARSE_FLASH_PROB_NUM_WARPS", num_warps_default))
        prob_num_stages = max(1, _env_int("VGGT_SPARSE_FLASH_PROB_NUM_STAGES", 1))
        grid_prob = lambda META: (seqlen_q, batch * nheads)
        _attn_prob_kernel[grid_prob](
            q,
            k,
            kv_positions,
            lse,
            prob_cache,
            softmax_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            kv_positions.stride(0),
            kv_positions.stride(1),
            prob_cache.stride(0),
            prob_cache.stride(1),
            prob_cache.stride(2),
            prob_cache.stride(3),
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            num_kv_candidates,
            d,
            seqlen_q // 32,
            seqlen_k // 32,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_N=prob_block_n,
            num_warps=prob_num_warps,
            num_stages=prob_num_stages,
        )
    return o, attn_sum, lse, softmax_scale, prob_cache


def _sparse_index_flash_attn_backward(
    do,
    q,
    k,
    v,
    kv_positions,
    o,
    lse,
    dq,
    dk,
    dv,
    dattn_sum=None,
    prob_cache=None,
    bias=None,
    softmax_scale=None,
    q_layout: str = "bthd",
):
    if do.stride(-1) != 1:
        do = do.contiguous()
    if q_layout == "bthd":
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        q_stride_h, q_stride_m = q.stride(2), q.stride(1)
        o_stride_h, o_stride_m = o.stride(2), o.stride(1)
        k_stride_h, k_stride_n = k.stride(2), k.stride(1)
        v_stride_h, v_stride_n = v.stride(2), v.stride(1)
        do_stride_h, do_stride_m = do.stride(2), do.stride(1)
        dq_stride_h, dq_stride_m = dq.stride(2), dq.stride(1)
        dk_stride_h, dk_stride_n = dk.stride(2), dk.stride(1)
        dv_stride_h, dv_stride_n = dv.stride(2), dv.stride(1)
    elif q_layout == "bhtd":
        batch, nheads, seqlen_q, d = q.shape
        _, _, seqlen_k, _ = k.shape
        q_stride_h, q_stride_m = q.stride(1), q.stride(2)
        o_stride_h, o_stride_m = o.stride(1), o.stride(2)
        k_stride_h, k_stride_n = k.stride(1), k.stride(2)
        v_stride_h, v_stride_n = v.stride(1), v.stride(2)
        do_stride_h, do_stride_m = do.stride(1), do.stride(2)
        dq_stride_h, dq_stride_m = dq.stride(1), dq.stride(2)
        dk_stride_h, dk_stride_n = dk.stride(1), dk.stride(2)
        dv_stride_h, dv_stride_n = dv.stride(1), dv.stride(2)
    else:
        raise ValueError(f"Unsupported q_layout: {q_layout}")
    assert d <= 128
    num_kv_candidates = kv_positions.shape[-1]
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    delta = torch.empty_like(lse)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o_stride_h,
        o_stride_m,
        do.stride(0),
        do_stride_h,
        do_stride_m,
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, num_kv_candidates):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, num_kv_candidates):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, num_kv_candidates) or (seqlen_q, num_kv_candidates)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, num_kv_candidates)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    use_prob_cache = (
        prob_cache is not None
        and prob_cache.numel() > 0
        and (not has_bias)
    )
    if use_prob_cache:
        if prob_cache.stride(-1) != 1:
            prob_cache = prob_cache.contiguous()
        expect_shape = (batch, nheads, seqlen_q, num_kv_candidates)
        if tuple(prob_cache.shape) != expect_shape:
            raise ValueError(
                f"prob_cache shape mismatch: got={tuple(prob_cache.shape)}, expected={expect_shape}"
            )
    prob_strides = (
        prob_cache.stride(0),
        prob_cache.stride(1),
        prob_cache.stride(2),
        prob_cache.stride(3),
    ) if use_prob_cache else (0, 0, 0, 0)

    has_dattn = dattn_sum is not None
    if has_dattn and dattn_sum.stride(-1) != 1:
        dattn_sum = dattn_sum.contiguous()
    dattn_strides = (
        dattn_sum.stride(0),
        dattn_sum.stride(1),
        dattn_sum.stride(2),
    ) if has_dattn else (0, 0, 0)
    use_lowp_acc = _env_int("VGGT_SPARSE_FLASH_BWD_ACC_LOWP", 0) != 0
    use_lowp_dkdv = use_lowp_acc and _env_flag("VGGT_SPARSE_FLASH_BWD_DKDV_FP16", False)
    bwd_block_n_default = 32 if (use_lowp_acc and d <= 64) else 128
    bwd_num_warps_default = 16 if (use_lowp_acc and d <= 64) else (4 if d <= 64 else 8)
    bwd_num_stages_default = 2 if (use_lowp_acc and d <= 64) else 1
    bwd_block_n = max(32, _env_int("VGGT_SPARSE_FLASH_BWD_BLOCK_N", bwd_block_n_default))
    bwd_num_warps = max(1, _env_int("VGGT_SPARSE_FLASH_BWD_NUM_WARPS", bwd_num_warps_default))
    bwd_num_stages = max(1, _env_int("VGGT_SPARSE_FLASH_BWD_NUM_STAGES", bwd_num_stages_default))
    assume_valid_pos = use_prob_cache and _env_flag("VGGT_SPARSE_FLASH_ASSUME_VALID_POS", False)
    if assume_valid_pos and _env_flag("VGGT_SPARSE_FLASH_ASSUME_VALID_POS_CHECK", False):
        with torch.no_grad():
            pos_min = int(kv_positions.min().item())
            pos_max = int(kv_positions.max().item())
        assume_valid_pos = (pos_min >= 0) and (pos_max < int(seqlen_k))
    fast_topk = max(1, _env_int("VGGT_SPARSE_FLASH_BWD_FAST_TOPK", 512))
    use_fast_topk_prob = (
        use_prob_cache
        and assume_valid_pos
        and _env_flag("VGGT_SPARSE_FLASH_BWD_FAST_TOPK512", False)
        and int(num_kv_candidates) == int(fast_topk)
        and (int(fast_topk) % int(bwd_block_n) == 0)
    )

    use_native_grouped = (
        use_prob_cache
        and (not has_bias)
        and _env_flag("VGGT_SPARSE_FLASH_BWD_NATIVE_GROUPED", False)
        and (sparse_prob_bwd_grouped is not None)
        and q.dtype in (torch.float16, torch.bfloat16)
        and dq.dtype == torch.float32
        and dk.dtype == torch.float32
        and dv.dtype == torch.float32
    )

    def _launch_prob_cache_bwd(
        q_in: torch.Tensor,
        do_in: torch.Tensor,
        dq_in: torch.Tensor,
        kv_pos_in: torch.Tensor,
        prob_in: torch.Tensor,
        dattn_in: Optional[torch.Tensor],
        delta_in: torch.Tensor,
        seqlen_q_local: int,
        delta_stride_h: int,
    ) -> None:
        local_prob_strides = (
            prob_in.stride(0),
            prob_in.stride(1),
            prob_in.stride(2),
            prob_in.stride(3),
        )
        has_dattn_local = dattn_in is not None
        local_dattn_strides = (
            dattn_in.stride(0),
            dattn_in.stride(1),
            dattn_in.stride(2),
        ) if has_dattn_local else (0, 0, 0)
        grid_local = lambda META: (seqlen_q_local, batch * nheads)
        if use_fast_topk_prob:
            _bwd_sparse_index_cached_p_fast_topk_kernel[grid_local](
                q_in,
                k,
                v,
                kv_pos_in,
                do_in,
                dq_in,
                dk,
                dv,
                delta_in,
                prob_in,
                dattn_in if has_dattn_local else q_in,
                softmax_scale,
                q_in.stride(0),
                q_stride_h,
                q_stride_m,
                k.stride(0),
                k_stride_h,
                k_stride_n,
                v.stride(0),
                v_stride_h,
                v_stride_n,
                do_in.stride(0),
                do_stride_h,
                do_stride_m,
                dq_in.stride(0),
                dq_stride_h,
                dq_stride_m,
                dk.stride(0),
                dk_stride_h,
                dk_stride_n,
                dv.stride(0),
                dv_stride_h,
                dv_stride_n,
                kv_pos_in.stride(0),
                kv_pos_in.stride(1),
                *local_prob_strides,
                *local_dattn_strides,
                nheads,
                seqlen_q_local,
                seqlen_k,
                delta_stride_h,
                d,
                seqlen_q_local // 32,
                seqlen_k // 32,
                NUM_TOPK=512,
                HAS_DATTN=has_dattn_local,
                LOWP_DKDV=use_lowp_dkdv,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                BLOCK_N=bwd_block_n,
                num_warps=bwd_num_warps,
                num_stages=bwd_num_stages,
            )
        else:
            bwd_kernel = _bwd_sparse_index_cached_p_fast_kernel if assume_valid_pos else _bwd_sparse_index_cached_p_kernel
            bwd_kernel[grid_local](
                q_in,
                k,
                v,
                kv_pos_in,
                do_in,
                dq_in,
                dk,
                dv,
                delta_in,
                prob_in,
                dattn_in if has_dattn_local else q_in,
                softmax_scale,
                q_in.stride(0),
                q_stride_h,
                q_stride_m,
                k.stride(0),
                k_stride_h,
                k_stride_n,
                v.stride(0),
                v_stride_h,
                v_stride_n,
                do_in.stride(0),
                do_stride_h,
                do_stride_m,
                dq_in.stride(0),
                dq_stride_h,
                dq_stride_m,
                dk.stride(0),
                dk_stride_h,
                dk_stride_n,
                dv.stride(0),
                dv_stride_h,
                dv_stride_n,
                kv_pos_in.stride(0),
                kv_pos_in.stride(1),
                *local_prob_strides,
                *local_dattn_strides,
                nheads,
                seqlen_q_local,
                seqlen_k,
                delta_stride_h,
                num_kv_candidates,
                d,
                seqlen_q_local // 32,
                seqlen_k // 32,
                has_dattn_local,
                use_lowp_dkdv,
                BLOCK_HEADDIM,
                BLOCK_N=bwd_block_n,
                num_warps=bwd_num_warps,
                num_stages=bwd_num_stages,
            )

    grid = lambda META: (seqlen_q, batch * nheads)
    if use_prob_cache:
        if use_native_grouped:
            native_query_chunk = max(1, _env_int("VGGT_SPARSE_FLASH_BWD_NATIVE_QCHUNK", 128))
            try:
                sparse_prob_bwd_grouped(
                    q,
                    k,
                    v,
                    kv_positions,
                    do,
                    delta,
                    prob_cache,
                    dq,
                    dk,
                    dv,
                    softmax_scale=float(softmax_scale),
                    q_layout=q_layout,
                    query_chunk=int(native_query_chunk),
                    dattn=dattn_sum if has_dattn else None,
                )
                return
            except Exception:
                if _env_flag("VGGT_SPARSE_FLASH_BWD_NATIVE_STRICT", False):
                    raise
        query_chunk = max(0, _env_int("VGGT_SPARSE_FLASH_BWD_QUERY_CHUNK", 0))
        use_query_chunk = query_chunk > 0 and query_chunk < seqlen_q
        if use_query_chunk:
            for q_start in range(0, seqlen_q, query_chunk):
                q_end = min(q_start + query_chunk, seqlen_q)
                q_len = q_end - q_start
                if q_layout == "bthd":
                    q_slice = q[:, q_start:q_end]
                    do_slice = do[:, q_start:q_end]
                    dq_slice = dq[:, q_start:q_end]
                else:
                    q_slice = q[:, :, q_start:q_end]
                    do_slice = do[:, :, q_start:q_end]
                    dq_slice = dq[:, :, q_start:q_end]
                kv_slice = kv_positions[:, q_start:q_end]
                prob_slice = prob_cache[:, :, q_start:q_end]
                dattn_slice = dattn_sum[:, q_start:q_end] if has_dattn else None
                delta_slice = delta[:, :, q_start:q_end]
                _launch_prob_cache_bwd(
                    q_slice,
                    do_slice,
                    dq_slice,
                    kv_slice,
                    prob_slice,
                    dattn_slice,
                    delta_slice,
                    q_len,
                    int(delta_slice.stride(1)),
                )
        else:
            _launch_prob_cache_bwd(
                q,
                do,
                dq,
                kv_positions,
                prob_cache,
                dattn_sum if has_dattn else None,
                delta,
                seqlen_q,
                seqlen_q_rounded,
            )
    else:
        _bwd_sparse_index_kernel[grid](
            q,
            k,
            v,
            kv_positions,
            bias,
            do,
            dq,
            dk,
            dv,
            lse,
            delta,
            dattn_sum if has_dattn else q,
            softmax_scale,
            q.stride(0),
            q_stride_h,
            q_stride_m,
            k.stride(0),
            k_stride_h,
            k_stride_n,
            v.stride(0),
            v_stride_h,
            v_stride_n,
            *bias_strides,
            do.stride(0),
            do_stride_h,
            do_stride_m,
            dq.stride(0),
            dq_stride_h,
            dq_stride_m,
            dk.stride(0),
            dk_stride_h,
            dk_stride_n,
            dv.stride(0),
            dv_stride_h,
            dv_stride_n,
            kv_positions.stride(0),
            kv_positions.stride(1),
            *dattn_strides,
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            num_kv_candidates,
            d,
            seqlen_q // 32,
            seqlen_k // 32,
            bias_type,
            has_dattn,
            BLOCK_HEADDIM,
            BLOCK_N=bwd_block_n,
            num_warps=bwd_num_warps,
            num_stages=bwd_num_stages,
        )


def _sparse_index_flash_attn_value_gate_backward_bhtd(
    do,
    q,
    k,
    v,
    kv_positions,
    value_gate,
    o,
    lse,
    dq,
    dk,
    dv,
    dvalue_gate,
    dattn_sum=None,
    prob_cache=None,
    softmax_scale=None,
):
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, nheads, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape
    num_kv_candidates = kv_positions.shape[-1]
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    value_gate, shared_gate = _prepare_value_gate_bhtd(
        value_gate,
        batch=batch,
        nheads=nheads,
        seqlen_q=seqlen_q,
        num_kv_candidates=num_kv_candidates,
        q_dtype=q.dtype,
    )
    if dvalue_gate.stride(-1) != 1:
        raise ValueError("dvalue_gate last dim must be contiguous")

    delta = torch.empty_like(lse)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    use_prob_cache = prob_cache is not None and prob_cache.numel() > 0
    if use_prob_cache and prob_cache.stride(-1) != 1:
        prob_cache = prob_cache.contiguous()
    has_dattn = dattn_sum is not None
    if has_dattn and dattn_sum.stride(-1) != 1:
        dattn_sum = dattn_sum.contiguous()
    use_lowp_acc = _env_int("VGGT_SPARSE_FLASH_BWD_ACC_LOWP", 0) != 0
    use_lowp_dkdv = use_lowp_acc and _env_flag("VGGT_SPARSE_FLASH_BWD_DKDV_FP16", False)
    bwd_block_n_default = 32 if (use_lowp_acc and d <= 64) else 128
    bwd_num_warps_default = 16 if (use_lowp_acc and d <= 64) else (4 if d <= 64 else 8)
    bwd_num_stages_default = 2 if (use_lowp_acc and d <= 64) else 1
    bwd_block_n = max(32, _env_int("VGGT_SPARSE_FLASH_BWD_BLOCK_N", bwd_block_n_default))
    bwd_num_warps = max(1, _env_int("VGGT_SPARSE_FLASH_BWD_NUM_WARPS", bwd_num_warps_default))
    bwd_num_stages = max(1, _env_int("VGGT_SPARSE_FLASH_BWD_NUM_STAGES", bwd_num_stages_default))

    def _launch_prob_cache_bwd(
        q_in: torch.Tensor,
        do_in: torch.Tensor,
        dq_in: torch.Tensor,
        kv_pos_in: torch.Tensor,
        gate_in: torch.Tensor,
        prob_in: torch.Tensor,
        dattn_in: Optional[torch.Tensor],
        delta_in: torch.Tensor,
        seqlen_q_local: int,
        delta_stride_h: int,
    ) -> None:
        grid_local = lambda META: (seqlen_q_local, batch * nheads)
        _bwd_sparse_index_cached_p_value_gate_kernel[grid_local](
            q_in,
            k,
            v,
            kv_pos_in,
            gate_in,
            do_in,
            dq_in,
            dk,
            dv,
            dvalue_gate,
            delta_in,
            prob_in,
            dattn_in if dattn_in is not None else q_in,
            softmax_scale,
            q_in.stride(0),
            q_in.stride(1),
            q_in.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            gate_in.stride(0),
            gate_in.stride(1),
            gate_in.stride(2),
            gate_in.stride(3),
            do_in.stride(0),
            do_in.stride(1),
            do_in.stride(2),
            dq_in.stride(0),
            dq_in.stride(1),
            dq_in.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dvalue_gate.stride(0),
            dvalue_gate.stride(1),
            dvalue_gate.stride(2),
            dvalue_gate.stride(3),
            kv_pos_in.stride(0),
            kv_pos_in.stride(1),
            prob_in.stride(0),
            prob_in.stride(1),
            prob_in.stride(2),
            prob_in.stride(3),
            *(dattn_in.stride(0), dattn_in.stride(1), dattn_in.stride(2)) if dattn_in is not None else (0, 0, 0),
            nheads,
            seqlen_q_local,
            seqlen_k,
            delta_stride_h,
            num_kv_candidates,
            d,
            seqlen_q_local // 32,
            seqlen_k // 32,
            shared_gate,
            dattn_in is not None,
            use_lowp_dkdv,
            BLOCK_HEADDIM,
            BLOCK_N=bwd_block_n,
            num_warps=bwd_num_warps,
            num_stages=bwd_num_stages,
        )

    if use_prob_cache:
        query_chunk = max(0, _env_int("VGGT_SPARSE_FLASH_BWD_QUERY_CHUNK", 0))
        use_query_chunk = query_chunk > 0 and query_chunk < seqlen_q
        if use_query_chunk:
            for q_start in range(0, seqlen_q, query_chunk):
                q_end = min(q_start + query_chunk, seqlen_q)
                q_slice = q[:, :, q_start:q_end]
                do_slice = do[:, :, q_start:q_end]
                dq_slice = dq[:, :, q_start:q_end]
                kv_slice = kv_positions[:, q_start:q_end]
                gate_slice = value_gate[:, :, q_start:q_end]
                prob_slice = prob_cache[:, :, q_start:q_end]
                dattn_slice = dattn_sum[:, q_start:q_end] if has_dattn else None
                delta_slice = delta[:, :, q_start:q_end]
                _launch_prob_cache_bwd(
                    q_slice,
                    do_slice,
                    dq_slice,
                    kv_slice,
                    gate_slice,
                    prob_slice,
                    dattn_slice,
                    delta_slice,
                    q_end - q_start,
                    int(delta_slice.stride(1)),
                )
        else:
            _launch_prob_cache_bwd(
                q,
                do,
                dq,
                kv_positions,
                value_gate,
                prob_cache,
                dattn_sum if has_dattn else None,
                delta,
                seqlen_q,
                seqlen_q_rounded,
            )
    else:
        grid = lambda META: (seqlen_q, batch * nheads)
        _bwd_sparse_index_value_gate_kernel[grid](
            q,
            k,
            v,
            kv_positions,
            value_gate,
            do,
            dq,
            dk,
            dv,
            dvalue_gate,
            lse,
            delta,
            dattn_sum if has_dattn else q,
            softmax_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            value_gate.stride(0),
            value_gate.stride(1),
            value_gate.stride(2),
            value_gate.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dvalue_gate.stride(0),
            dvalue_gate.stride(1),
            dvalue_gate.stride(2),
            dvalue_gate.stride(3),
            kv_positions.stride(0),
            kv_positions.stride(1),
            *(dattn_sum.stride(0), dattn_sum.stride(1), dattn_sum.stride(2)) if has_dattn else (0, 0, 0),
            nheads,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            num_kv_candidates,
            d,
            seqlen_q // 32,
            seqlen_k // 32,
            shared_gate,
            has_dattn,
            BLOCK_HEADDIM,
            BLOCK_N=bwd_block_n,
            num_warps=bwd_num_warps,
            num_stages=bwd_num_stages,
        )


def _build_dense_kv_positions(
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create broadcasted dense key positions [B, Tq, Tk] with O(Tk) storage.
    """
    base = torch.arange(seqlen_k, device=device, dtype=torch.int32)
    return torch.as_strided(base, size=(batch, seqlen_q, seqlen_k), stride=(0, 0, 1))


def dense_index_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dense attention entry reusing sparse-index flash attention kernels.
    """
    batch, seqlen_q, _, _ = q.shape
    seqlen_k = k.shape[1]
    kv_positions = _build_dense_kv_positions(batch, seqlen_q, seqlen_k, q.device)
    return SparseIndexFlashAttnFunc.apply(q, k, v, kv_positions, bias, causal, softmax_scale)


@torch.inference_mode()
def sparse_index_flash_attn_inference_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_positions: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    return_attn: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Inference-only entry that can skip attn_sum materialization.
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
    if kv_positions.stride(-1) != 1:
        kv_positions = kv_positions.contiguous()
    o, attn_sum, _, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        kv_positions,
        bias=bias,
        causal=causal,
        softmax_scale=softmax_scale,
        return_attn=return_attn,
        q_layout="bthd",
    )
    return o, attn_sum


@torch.inference_mode()
def sparse_index_flash_attn_inference_bhtd_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_positions: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    return_attn: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Inference-only entry that keeps q/k/v in [B, H, T, D] layout.
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
    if kv_positions.stride(-1) != 1:
        kv_positions = kv_positions.contiguous()
    o, attn_sum, _, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        kv_positions,
        bias=bias,
        causal=causal,
        softmax_scale=softmax_scale,
        return_attn=return_attn,
        q_layout="bhtd",
    )
    return o, attn_sum


class SparseIndexFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kv_positions, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        if kv_positions.stride(-1) != 1:
            kv_positions = kv_positions.contiguous()
        save_prob_for_bwd = _should_save_prob_cache_for_bwd(
            batch=int(q.shape[0]),
            nheads=int(q.shape[2]),
            seqlen_q=int(q.shape[1]),
            num_kv_candidates=int(kv_positions.shape[-1]),
            has_bias=bias is not None,
        )
        o, attn_sum, lse, ctx.softmax_scale, prob_cache = _flash_attn_forward(
            q,
            k,
            v,
            kv_positions,
            bias=bias,
            causal=causal,
            softmax_scale=softmax_scale,
            return_attn=True,
            save_prob_for_bwd=save_prob_for_bwd,
            q_layout="bthd",
        )
        ctx.has_bias = bias is not None
        if bias is None:
            bias = q.new_empty(0)
        ctx.has_prob_cache = prob_cache is not None
        if prob_cache is None:
            prob_cache = q.new_empty(0)
        ctx.save_for_backward(q, k, v, kv_positions, o, lse, bias, prob_cache)
        ctx.causal = causal
        return o, attn_sum

    @staticmethod
    def backward(ctx, do, dattn_sum):
        q, k, v, kv_positions, o, lse, bias, prob_cache = ctx.saved_tensors
        if not ctx.has_bias:
            bias = None
        if not ctx.has_prob_cache:
            prob_cache = None
        if not do.is_contiguous():
            do = do.contiguous()
        if dattn_sum is not None and dattn_sum.dtype != q.dtype:
            dattn_sum = dattn_sum.to(q.dtype)
        use_lowp_acc = _env_int("VGGT_SPARSE_FLASH_BWD_ACC_LOWP", 0) != 0
        if _env_flag("VGGT_SPARSE_FLASH_BWD_NATIVE_GROUPED", False):
            use_lowp_acc = False
        dq_acc_dtype = q.dtype if use_lowp_acc else torch.float32
        dk_acc_dtype = k.dtype if use_lowp_acc else torch.float32
        dv_acc_dtype = v.dtype if use_lowp_acc else torch.float32
        # dq is fully overwritten in-kernel, while dk/dv need zero-init for atomic adds.
        dq = torch.empty(q.shape, device=q.device, dtype=dq_acc_dtype)
        dk = torch.zeros(k.shape, device=k.device, dtype=dk_acc_dtype)
        dv = torch.zeros(v.shape, device=v.device, dtype=dv_acc_dtype)
        _sparse_index_flash_attn_backward(
            do,
            q,
            k,
            v,
            kv_positions,
            o,
            lse,
            dq,
            dk,
            dv,
            dattn_sum=dattn_sum,
            prob_cache=prob_cache,
            bias=bias,
            softmax_scale=ctx.softmax_scale,
            q_layout="bthd",
        )
        if dq.dtype != q.dtype:
            dq = dq.to(q.dtype)
        if dk.dtype != k.dtype:
            dk = dk.to(k.dtype)
        if dv.dtype != v.dtype:
            dv = dv.to(v.dtype)
        return dq, dk, dv, None, None, None, None


sparse_index_flash_attn_func = SparseIndexFlashAttnFunc.apply


class SparseIndexFlashAttnBhtdFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kv_positions, bias=None, causal=False, softmax_scale=None):
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        if kv_positions.stride(-1) != 1:
            kv_positions = kv_positions.contiguous()
        save_prob_for_bwd = _should_save_prob_cache_for_bwd(
            batch=int(q.shape[0]),
            nheads=int(q.shape[1]),
            seqlen_q=int(q.shape[2]),
            num_kv_candidates=int(kv_positions.shape[-1]),
            has_bias=bias is not None,
        )
        o, attn_sum, lse, ctx.softmax_scale, prob_cache = _flash_attn_forward(
            q,
            k,
            v,
            kv_positions,
            bias=bias,
            causal=causal,
            softmax_scale=softmax_scale,
            return_attn=True,
            save_prob_for_bwd=save_prob_for_bwd,
            q_layout="bhtd",
        )
        ctx.has_bias = bias is not None
        if bias is None:
            bias = q.new_empty(0)
        ctx.has_prob_cache = prob_cache is not None
        if prob_cache is None:
            prob_cache = q.new_empty(0)
        ctx.save_for_backward(q, k, v, kv_positions, o, lse, bias, prob_cache)
        ctx.causal = causal
        return o, attn_sum

    @staticmethod
    def backward(ctx, do, dattn_sum):
        q, k, v, kv_positions, o, lse, bias, prob_cache = ctx.saved_tensors
        if not ctx.has_bias:
            bias = None
        if not ctx.has_prob_cache:
            prob_cache = None
        if not do.is_contiguous():
            do = do.contiguous()
        if dattn_sum is not None and dattn_sum.dtype != q.dtype:
            dattn_sum = dattn_sum.to(q.dtype)
        use_lowp_acc = _env_int("VGGT_SPARSE_FLASH_BWD_ACC_LOWP", 0) != 0
        if _env_flag("VGGT_SPARSE_FLASH_BWD_NATIVE_GROUPED", False):
            use_lowp_acc = False
        dq_acc_dtype = q.dtype if use_lowp_acc else torch.float32
        dk_acc_dtype = k.dtype if use_lowp_acc else torch.float32
        dv_acc_dtype = v.dtype if use_lowp_acc else torch.float32
        dq = torch.empty(q.shape, device=q.device, dtype=dq_acc_dtype)
        dk = torch.zeros(k.shape, device=k.device, dtype=dk_acc_dtype)
        dv = torch.zeros(v.shape, device=v.device, dtype=dv_acc_dtype)
        _sparse_index_flash_attn_backward(
            do,
            q,
            k,
            v,
            kv_positions,
            o,
            lse,
            dq,
            dk,
            dv,
            dattn_sum=dattn_sum,
            prob_cache=prob_cache,
            bias=bias,
            softmax_scale=ctx.softmax_scale,
            q_layout="bhtd",
        )
        if dq.dtype != q.dtype:
            dq = dq.to(q.dtype)
        if dk.dtype != k.dtype:
            dk = dk.to(k.dtype)
        if dv.dtype != v.dtype:
            dv = dv.to(v.dtype)
        return dq, dk, dv, None, None, None, None


sparse_index_flash_attn_bhtd_func = SparseIndexFlashAttnBhtdFunc.apply


class SparseIndexFlashAttnValueGateBhtdFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kv_positions, value_gate, softmax_scale=None, return_attn=True):
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        if kv_positions.stride(-1) != 1:
            kv_positions = kv_positions.contiguous()
        save_prob_for_bwd = _should_save_prob_cache_for_bwd(
            batch=int(q.shape[0]),
            nheads=int(q.shape[1]),
            seqlen_q=int(q.shape[2]),
            num_kv_candidates=int(kv_positions.shape[-1]),
            has_bias=False,
        )
        o, attn_sum, lse, ctx.softmax_scale, prob_cache = _flash_attn_forward_value_gate_bhtd(
            q,
            k,
            v,
            kv_positions,
            value_gate=value_gate,
            softmax_scale=softmax_scale,
            return_attn=bool(return_attn),
            save_prob_for_bwd=save_prob_for_bwd,
        )
        ctx.return_attn = bool(return_attn)
        if attn_sum is None:
            attn_sum = q.new_empty(0)
        ctx.has_prob_cache = prob_cache is not None
        if prob_cache is None:
            prob_cache = q.new_empty(0)
        ctx.save_for_backward(q, k, v, kv_positions, value_gate, o, lse, prob_cache)
        return o, attn_sum

    @staticmethod
    def backward(ctx, do, dattn_sum):
        q, k, v, kv_positions, value_gate, o, lse, prob_cache = ctx.saved_tensors
        if not ctx.return_attn:
            dattn_sum = None
        elif dattn_sum is not None and dattn_sum.numel() == 0:
            dattn_sum = None
        if not ctx.has_prob_cache:
            prob_cache = None
        if not do.is_contiguous():
            do = do.contiguous()
        if dattn_sum is not None and dattn_sum.dtype != q.dtype:
            dattn_sum = dattn_sum.to(q.dtype)
        use_lowp_acc = _env_int("VGGT_SPARSE_FLASH_BWD_ACC_LOWP", 0) != 0
        if _env_flag("VGGT_SPARSE_FLASH_BWD_NATIVE_GROUPED", False):
            use_lowp_acc = False
        dq_acc_dtype = q.dtype if use_lowp_acc else torch.float32
        dk_acc_dtype = k.dtype if use_lowp_acc else torch.float32
        dv_acc_dtype = v.dtype if use_lowp_acc else torch.float32
        dq = torch.empty(q.shape, device=q.device, dtype=dq_acc_dtype)
        dk = torch.zeros(k.shape, device=k.device, dtype=dk_acc_dtype)
        dv = torch.zeros(v.shape, device=v.device, dtype=dv_acc_dtype)
        dvalue_gate = torch.zeros(value_gate.shape, device=value_gate.device, dtype=torch.float32)
        _sparse_index_flash_attn_value_gate_backward_bhtd(
            do,
            q,
            k,
            v,
            kv_positions,
            value_gate,
            o,
            lse,
            dq,
            dk,
            dv,
            dvalue_gate,
            dattn_sum=dattn_sum,
            prob_cache=prob_cache,
            softmax_scale=ctx.softmax_scale,
        )
        if dq.dtype != q.dtype:
            dq = dq.to(q.dtype)
        if dk.dtype != k.dtype:
            dk = dk.to(k.dtype)
        if dv.dtype != v.dtype:
            dv = dv.to(v.dtype)
        if dvalue_gate.dtype != value_gate.dtype:
            dvalue_gate = dvalue_gate.to(value_gate.dtype)
        return dq, dk, dv, None, dvalue_gate, None, None


sparse_index_flash_attn_value_gate_bhtd_func = SparseIndexFlashAttnValueGateBhtdFunc.apply
