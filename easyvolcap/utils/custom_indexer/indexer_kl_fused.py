from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _kl_stats_kernel(
    ATTN,
    LOGITS,
    SUM_ATTN,
    MAX_LOGIT,
    SUM_EXP,
    stride_ar,
    stride_ac,
    stride_lr,
    stride_lc,
    stride_sr,
    stride_mr,
    stride_er,
    num_cols,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    sum_attn = tl.zeros((), dtype=tl.float32)
    max_logit = tl.full((), -float("inf"), dtype=tl.float32)
    for start in range(0, num_cols, BLOCK_N):
        idx = start + offs
        mask = idx < num_cols
        a = tl.load(ATTN + row * stride_ar + idx * stride_ac, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(LOGITS + row * stride_lr + idx * stride_lc, mask=mask, other=-float("inf")).to(tl.float32)
        sum_attn += tl.sum(a, axis=0)
        max_logit = tl.maximum(max_logit, tl.max(z, axis=0))

    sum_exp = tl.zeros((), dtype=tl.float32)
    for start in range(0, num_cols, BLOCK_N):
        idx = start + offs
        mask = idx < num_cols
        z = tl.load(LOGITS + row * stride_lr + idx * stride_lc, mask=mask, other=-float("inf")).to(tl.float32)
        sum_exp += tl.sum(tl.exp(z - max_logit), axis=0)

    tl.store(SUM_ATTN + row * stride_sr, sum_attn)
    tl.store(MAX_LOGIT + row * stride_mr, max_logit)
    tl.store(SUM_EXP + row * stride_er, sum_exp)


@triton.jit
def _kl_row_loss_kernel(
    ATTN,
    LOGITS,
    SUM_ATTN,
    MAX_LOGIT,
    SUM_EXP,
    ROW_LOSS,
    stride_ar,
    stride_ac,
    stride_lr,
    stride_lc,
    stride_sr,
    stride_mr,
    stride_er,
    stride_rr,
    num_cols,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    sum_attn = tl.load(SUM_ATTN + row * stride_sr).to(tl.float32)
    max_logit = tl.load(MAX_LOGIT + row * stride_mr).to(tl.float32)
    sum_exp = tl.load(SUM_EXP + row * stride_er).to(tl.float32)
    denom_attn = sum_attn + eps
    lse = max_logit + tl.log(sum_exp)

    row_loss = tl.zeros((), dtype=tl.float32)
    for start in range(0, num_cols, BLOCK_N):
        idx = start + offs
        mask = idx < num_cols
        a = tl.load(ATTN + row * stride_ar + idx * stride_ac, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(LOGITS + row * stride_lr + idx * stride_lc, mask=mask, other=0.0).to(tl.float32)
        p = a / denom_attn
        log_p = tl.log(p + eps)
        log_q = z - lse
        term = p * (log_p - log_q)
        term = tl.where(mask, term, 0.0)
        row_loss += tl.sum(term, axis=0)

    tl.store(ROW_LOSS + row * stride_rr, row_loss)


@triton.jit
def _kl_row_bwd_kernel(
    ATTN,
    LOGITS,
    SUM_ATTN,
    MAX_LOGIT,
    SUM_EXP,
    DATTN,
    DLOGITS,
    stride_ar,
    stride_ac,
    stride_lr,
    stride_lc,
    stride_sr,
    stride_mr,
    stride_er,
    stride_dar,
    stride_dac,
    stride_dlr,
    stride_dlc,
    num_cols,
    eps,
    grad_scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    sum_attn = tl.load(SUM_ATTN + row * stride_sr).to(tl.float32)
    max_logit = tl.load(MAX_LOGIT + row * stride_mr).to(tl.float32)
    sum_exp = tl.load(SUM_EXP + row * stride_er).to(tl.float32)
    denom_attn = sum_attn + eps
    inv_sum_exp = 1.0 / sum_exp
    lse = max_logit + tl.log(sum_exp)

    gbar = tl.zeros((), dtype=tl.float32)
    for start in range(0, num_cols, BLOCK_N):
        idx = start + offs
        mask = idx < num_cols
        a = tl.load(ATTN + row * stride_ar + idx * stride_ac, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(LOGITS + row * stride_lr + idx * stride_lc, mask=mask, other=0.0).to(tl.float32)
        p = a / denom_attn
        log_q = z - lse
        g = tl.log(p + eps) - log_q + p / (p + eps)
        pg = tl.where(mask, p * g, 0.0)
        gbar += tl.sum(pg, axis=0)

    inv_denom = grad_scale / denom_attn
    for start in range(0, num_cols, BLOCK_N):
        idx = start + offs
        mask = idx < num_cols
        a = tl.load(ATTN + row * stride_ar + idx * stride_ac, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(LOGITS + row * stride_lr + idx * stride_lc, mask=mask, other=0.0).to(tl.float32)
        p = a / denom_attn
        q = tl.exp(z - max_logit) * inv_sum_exp
        log_q = z - lse
        g = tl.log(p + eps) - log_q + p / (p + eps)
        d_attn = (g - gbar) * inv_denom
        d_logits = (q - p) * grad_scale
        tl.store(DATTN + row * stride_dar + idx * stride_dac, d_attn, mask=mask)
        tl.store(DLOGITS + row * stride_dlr + idx * stride_dlc, d_logits, mask=mask)


def _reference_indexer_kl_loss(attn_sum: torch.Tensor, logits: torch.Tensor, eps: float) -> torch.Tensor:
    p = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + eps)
    log_q = torch.log_softmax(logits, dim=-1)
    return (p * (torch.log(p + eps) - log_q)).sum(dim=-1).mean()


class _SparseIndexerKLLossFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_sum: torch.Tensor, logits: torch.Tensor, eps: float):
        if attn_sum.stride(-1) != 1:
            attn_sum = attn_sum.contiguous()
        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        if attn_sum.dtype != logits.dtype:
            logits = logits.to(attn_sum.dtype)

        bsz, tgt_len, topk = attn_sum.shape
        rows = int(bsz) * int(tgt_len)
        attn_2d = attn_sum.view(rows, topk)
        logits_2d = logits.view(rows, topk)

        sum_attn = torch.empty((rows,), device=attn_sum.device, dtype=torch.float32)
        max_logit = torch.empty((rows,), device=attn_sum.device, dtype=torch.float32)
        sum_exp = torch.empty((rows,), device=attn_sum.device, dtype=torch.float32)
        row_loss = torch.empty((rows,), device=attn_sum.device, dtype=torch.float32)

        block_n = max(32, min(256, int(triton.next_power_of_2(topk))))
        grid = (rows,)
        _kl_stats_kernel[grid](
            attn_2d,
            logits_2d,
            sum_attn,
            max_logit,
            sum_exp,
            attn_2d.stride(0),
            attn_2d.stride(1),
            logits_2d.stride(0),
            logits_2d.stride(1),
            sum_attn.stride(0),
            max_logit.stride(0),
            sum_exp.stride(0),
            topk,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=1,
        )
        _kl_row_loss_kernel[grid](
            attn_2d,
            logits_2d,
            sum_attn,
            max_logit,
            sum_exp,
            row_loss,
            attn_2d.stride(0),
            attn_2d.stride(1),
            logits_2d.stride(0),
            logits_2d.stride(1),
            sum_attn.stride(0),
            max_logit.stride(0),
            sum_exp.stride(0),
            row_loss.stride(0),
            topk,
            float(eps),
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=1,
        )

        ctx.save_for_backward(attn_sum, logits, sum_attn, max_logit, sum_exp)
        ctx.rows = rows
        ctx.topk = int(topk)
        ctx.eps = float(eps)
        return row_loss.mean()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        attn_sum, logits, sum_attn, max_logit, sum_exp = ctx.saved_tensors
        rows = int(ctx.rows)
        topk = int(ctx.topk)
        eps = float(ctx.eps)

        if attn_sum.stride(-1) != 1:
            attn_sum = attn_sum.contiguous()
        if logits.stride(-1) != 1:
            logits = logits.contiguous()

        attn_2d = attn_sum.view(rows, topk)
        logits_2d = logits.view(rows, topk)
        d_attn = torch.empty_like(attn_2d)
        d_logits = torch.empty_like(logits_2d)

        if grad_out is None:
            grad_scale = 0.0
        else:
            grad_scale = float(grad_out.float().item()) / float(max(1, rows))

        block_n = max(32, min(256, int(triton.next_power_of_2(topk))))
        grid = (rows,)
        _kl_row_bwd_kernel[grid](
            attn_2d,
            logits_2d,
            sum_attn,
            max_logit,
            sum_exp,
            d_attn,
            d_logits,
            attn_2d.stride(0),
            attn_2d.stride(1),
            logits_2d.stride(0),
            logits_2d.stride(1),
            sum_attn.stride(0),
            max_logit.stride(0),
            sum_exp.stride(0),
            d_attn.stride(0),
            d_attn.stride(1),
            d_logits.stride(0),
            d_logits.stride(1),
            topk,
            eps,
            grad_scale,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=1,
        )
        return d_attn.view_as(attn_sum), d_logits.view_as(logits), None


def sparse_indexer_kl_loss(attn_sum: torch.Tensor, logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if (
        (not torch.is_tensor(attn_sum))
        or (not torch.is_tensor(logits))
        or attn_sum.ndim != 3
        or logits.ndim != 3
        or tuple(attn_sum.shape) != tuple(logits.shape)
        or (not attn_sum.is_cuda)
        or (not logits.is_cuda)
    ):
        return _reference_indexer_kl_loss(attn_sum, logits, float(eps))
    if attn_sum.numel() == 0:
        return attn_sum.new_zeros(())
    return _SparseIndexerKLLossFunc.apply(attn_sum, logits, float(eps))
