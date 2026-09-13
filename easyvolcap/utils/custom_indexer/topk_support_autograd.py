from typing import Optional

import torch

from .streaming_kl_autograd import _score_chunk_backward, _score_chunk_forward


def _selected_score_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    score_head_chunk_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    bsz, tgt_len, n_heads, _ = q.shape
    support = int(indices.shape[-1])
    batch_idx = torch.arange(bsz, device=indices.device, dtype=torch.long).view(bsz, 1, 1)
    score_chunk = None
    for h_start in range(0, n_heads, score_head_chunk_size):
        h_end = min(h_start + score_head_chunk_size, n_heads)
        q_h = q[:, :, h_start:h_end]
        k_h = k[:, :, h_start:h_end]
        w_h = w[:, :, h_start:h_end]
        k_sel = k_h[batch_idx, indices]
        w_sel = w_h[batch_idx, indices].permute(0, 1, 3, 2)
        head_scores = torch.einsum("bthd,btkhd->bthk", q_h, k_sel) * scale
        head_scores = torch.relu(head_scores)
        head_scores = head_scores * w_sel
        head_scores = head_scores.sum(dim=2, dtype=out_dtype)
        score_chunk = head_scores if score_chunk is None else score_chunk + head_scores
    if score_chunk is None:
        score_chunk = q.new_zeros((bsz, tgt_len, support), dtype=out_dtype)
    return score_chunk


def _selected_score_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    indices: torch.Tensor,
    grad_score: torch.Tensor,
    scale: float,
    score_head_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, tgt_len, n_heads, head_dim = q.shape
    support = int(indices.shape[-1])
    batch_idx = torch.arange(bsz, device=indices.device, dtype=torch.long).view(bsz, 1, 1)
    dq = torch.zeros_like(q)
    dk_selected = torch.zeros((bsz, tgt_len, support, n_heads, head_dim), device=k.device, dtype=k.dtype)
    dw_selected = torch.zeros((bsz, tgt_len, support, n_heads), device=w.device, dtype=w.dtype)
    grad_score = grad_score.to(q.dtype)

    for h_start in range(0, n_heads, score_head_chunk_size):
        h_end = min(h_start + score_head_chunk_size, n_heads)
        q_h = q[:, :, h_start:h_end]
        k_h = k[:, :, h_start:h_end]
        w_h = w[:, :, h_start:h_end]
        k_sel = k_h[batch_idx, indices]
        w_sel = w_h[batch_idx, indices]

        pre_act = torch.einsum("bthd,btkhd->bthk", q_h, k_sel) * scale
        relu_scores = torch.relu(pre_act)
        grad_relu = grad_score.unsqueeze(2) * w_sel.permute(0, 1, 3, 2)
        grad_pre = torch.where(pre_act > 0, grad_relu, torch.zeros_like(grad_relu))

        dq[:, :, h_start:h_end] = dq[:, :, h_start:h_end] + torch.einsum(
            "bthk,btkhd->bthd",
            grad_pre,
            k_sel,
        ) * scale
        dk_selected[:, :, :, h_start:h_end] = torch.einsum(
            "bthk,bthd->btkhd",
            grad_pre,
            q_h,
        ) * scale
        dw_selected[:, :, :, h_start:h_end] = (
            grad_score.unsqueeze(2) * relu_scores
        ).permute(0, 1, 3, 2)

    return dq, dk_selected, dw_selected


class _TopKSupportLossAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        p: torch.Tensor,
        mask: torch.Tensor,
        support_topk: int,
        scale: float,
        score_head_chunk_size: int,
        score_key_chunk_size: int,
        support_chunk_size: int,
        query_chunk_size: int,
    ) -> torch.Tensor:
        if p.dim() != 3:
            raise ValueError(f"Expected p shape [B, T, S], got {tuple(p.shape)}")
        bsz, tgt_len, src_len = p.shape
        support_topk = max(1, min(int(support_topk), int(src_len)))
        score_key_chunk_size = max(1, min(int(score_key_chunk_size), int(src_len)))
        support_chunk_size = max(1, min(int(support_chunk_size), int(support_topk)))
        query_chunk_size = max(1, min(int(query_chunk_size), int(tgt_len)))
        accum_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        has_mask = mask.numel() > 0
        norm_mask = mask if has_mask else None

        with torch.no_grad():
            _, target_indices = torch.topk(p, k=support_topk, dim=-1, largest=True, sorted=False)
            target_indices = target_indices.to(torch.long)

        score_chunk_buf = q.new_empty((bsz, tgt_len, score_key_chunk_size), dtype=q.dtype)
        log_denom = torch.full((bsz, tgt_len), float("-inf"), device=p.device, dtype=accum_dtype)
        for s_start in range(0, src_len, score_key_chunk_size):
            s_end = min(s_start + score_key_chunk_size, src_len)
            mask_chunk = None if norm_mask is None else norm_mask[:, :, s_start:s_end]
            score_chunk = _score_chunk_forward(
                q=q,
                k_chunk=k[:, s_start:s_end],
                w_chunk=w[:, s_start:s_end],
                scale=float(scale),
                score_head_chunk_size=int(score_head_chunk_size),
                out_dtype=q.dtype,
                total_src_len=src_len,
                mask_chunk=mask_chunk,
                out=score_chunk_buf[:, :, : (s_end - s_start)],
            )
            log_denom = torch.logaddexp(log_denom, torch.logsumexp(score_chunk, dim=-1).to(accum_dtype))

        support_log_denom = torch.full((bsz, tgt_len), float("-inf"), device=p.device, dtype=accum_dtype)
        for q_start in range(0, tgt_len, query_chunk_size):
            q_end = min(q_start + query_chunk_size, tgt_len)
            q_chunk = q[:, q_start:q_end]
            q_support_log = torch.full((bsz, q_end - q_start), float("-inf"), device=p.device, dtype=accum_dtype)
            for k_start in range(0, support_topk, support_chunk_size):
                k_end = min(k_start + support_chunk_size, support_topk)
                indices = target_indices[:, q_start:q_end, k_start:k_end]
                selected_scores = _selected_score_forward(
                    q=q_chunk,
                    k=k,
                    w=w,
                    indices=indices,
                    scale=float(scale),
                    score_head_chunk_size=int(score_head_chunk_size),
                    out_dtype=q.dtype,
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

        ctx.save_for_backward(q, k, w, mask, target_indices, log_denom, support_log_denom)
        ctx.scale = float(scale)
        ctx.score_head_chunk_size = int(score_head_chunk_size)
        ctx.score_key_chunk_size = int(score_key_chunk_size)
        ctx.support_chunk_size = int(support_chunk_size)
        ctx.query_chunk_size = int(query_chunk_size)
        return (log_denom - support_log_denom).mean()

    @staticmethod
    def backward(ctx, grad_out: Optional[torch.Tensor]):
        q, k, w, mask, target_indices, log_denom, support_log_denom = ctx.saved_tensors
        if grad_out is None:
            return (torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(w), None, None, None, None, None, None, None, None)

        bsz, tgt_len, n_heads, head_dim = q.shape
        src_len = k.shape[1]
        support_topk = int(target_indices.shape[-1])
        score_key_chunk_size = int(ctx.score_key_chunk_size)
        score_head_chunk_size = int(ctx.score_head_chunk_size)
        support_chunk_size = int(ctx.support_chunk_size)
        query_chunk_size = int(ctx.query_chunk_size)
        scale = float(ctx.scale)
        has_mask = mask.numel() > 0
        norm_mask = mask if has_mask else None
        grad_scale = grad_out.detach().float() / float(max(1, bsz * tgt_len))

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dw = torch.zeros_like(w)
        score_chunk_buf = q.new_empty((bsz, tgt_len, score_key_chunk_size), dtype=q.dtype)

        for s_start in range(0, src_len, score_key_chunk_size):
            s_end = min(s_start + score_key_chunk_size, src_len)
            mask_chunk = None if norm_mask is None else norm_mask[:, :, s_start:s_end]
            score_chunk = _score_chunk_forward(
                q=q,
                k_chunk=k[:, s_start:s_end],
                w_chunk=w[:, s_start:s_end],
                scale=scale,
                score_head_chunk_size=score_head_chunk_size,
                out_dtype=q.dtype,
                total_src_len=src_len,
                mask_chunk=mask_chunk,
                out=score_chunk_buf[:, :, : (s_end - s_start)],
            )
            grad_score = torch.exp(score_chunk.to(log_denom.dtype) - log_denom.unsqueeze(-1))
            grad_score = grad_score * grad_scale.to(log_denom.dtype)
            dq_chunk, dk_chunk, dw_chunk = _score_chunk_backward(
                q=q,
                k_chunk=k[:, s_start:s_end],
                w_chunk=w[:, s_start:s_end],
                grad_score_chunk=grad_score,
                scale=scale,
                score_head_chunk_size=score_head_chunk_size,
            )
            dq = dq + dq_chunk
            dk[:, s_start:s_end] = dk[:, s_start:s_end] + dk_chunk
            dw[:, s_start:s_end] = dw[:, s_start:s_end] + dw_chunk

        for q_start in range(0, tgt_len, query_chunk_size):
            q_end = min(q_start + query_chunk_size, tgt_len)
            q_chunk = q[:, q_start:q_end]
            q_support_log = support_log_denom[:, q_start:q_end].unsqueeze(-1)
            for k_start in range(0, support_topk, support_chunk_size):
                k_end = min(k_start + support_chunk_size, support_topk)
                indices = target_indices[:, q_start:q_end, k_start:k_end]
                selected_scores = _selected_score_forward(
                    q=q_chunk,
                    k=k,
                    w=w,
                    indices=indices,
                    scale=scale,
                    score_head_chunk_size=score_head_chunk_size,
                    out_dtype=q.dtype,
                )
                if norm_mask is not None:
                    selected_scores = selected_scores + torch.gather(
                        norm_mask[:, q_start:q_end],
                        dim=-1,
                        index=indices,
                    )
                grad_score = -torch.exp(selected_scores.to(log_denom.dtype) - q_support_log)
                grad_score = grad_score * grad_scale.to(log_denom.dtype)
                dq_selected, dk_selected, dw_selected = _selected_score_backward(
                    q=q_chunk,
                    k=k,
                    w=w,
                    indices=indices,
                    grad_score=grad_score,
                    scale=scale,
                    score_head_chunk_size=score_head_chunk_size,
                )
                dq[:, q_start:q_end] = dq[:, q_start:q_end] + dq_selected

                flat_indices = indices.reshape(bsz, -1)
                dk.scatter_add_(
                    dim=1,
                    index=flat_indices[:, :, None, None].expand(-1, -1, n_heads, head_dim),
                    src=dk_selected.reshape(bsz, -1, n_heads, head_dim),
                )
                dw.scatter_add_(
                    dim=1,
                    index=flat_indices[:, :, None].expand(-1, -1, n_heads),
                    src=dw_selected.reshape(bsz, -1, n_heads),
                )

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
            None,
            None,
        )


def topk_support_autograd_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    mask: Optional[torch.Tensor],
    support_topk: int,
    scale: float,
    score_head_chunk_size: int,
    score_key_chunk_size: int,
    support_chunk_size: int,
    query_chunk_size: int,
) -> torch.Tensor:
    mask_tensor = mask if mask is not None else q.new_empty(0)
    return _TopKSupportLossAutogradFunc.apply(
        q,
        k,
        w,
        p,
        mask_tensor,
        int(support_topk),
        float(scale),
        int(score_head_chunk_size),
        int(score_key_chunk_size),
        int(support_chunk_size),
        int(query_chunk_size),
    )
