import time
import random
import math
import os

try:
    import torch
except Exception as exc:  # pragma: no cover - runtime env may miss torch
    torch = None
    TORCH_ERROR = exc
else:
    TORCH_ERROR = None
    import torch.nn.functional as F

if torch is not None:
    try:
        from easyvolcap.utils.custom_flash_attn.sparse_index_flash_attn import sparse_index_flash_attn_func
    except Exception as exc:
        sparse_index_flash_attn_func = None
        IMPORT_ERROR = exc
    else:
        IMPORT_ERROR = None
    try:
        from easyvolcap.utils.custom_indexer import sparse_topk_indexer_func
    except Exception as exc:
        sparse_topk_indexer_func = None
        INDEXER_IMPORT_ERROR = exc
    else:
        INDEXER_IMPORT_ERROR = None
else:
    sparse_index_flash_attn_func = None
    IMPORT_ERROR = None
    sparse_topk_indexer_func = None
    INDEXER_IMPORT_ERROR = None


def _make_kv_positions(batch, seqlen, topk, device):
    kv_positions = torch.full((batch, seqlen, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        for t in range(seqlen):
            length = random.randint(max(1, topk // 2), topk)
            idx = torch.randperm(seqlen, device=device)[:length]
            kv_positions[b, t, :length] = idx
    return kv_positions


def _reference_sparse_attn(q, k, v, kv_positions, bias=None, softmax_scale=None):
    # q, k, v: (B, T, H, D)
    bsz, tgt_len, nheads, head_dim = q.shape
    topk = kv_positions.shape[-1]
    scale = softmax_scale or (1.0 / (head_dim ** 0.5))

    qh = q.transpose(1, 2)  # (B, H, T, D)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)

    valid = kv_positions >= 0
    kv_safe = kv_positions.clamp(min=0).to(torch.long)
    index = kv_safe[:, None, :, :].unsqueeze(-1).expand(bsz, nheads, tgt_len, topk, head_dim)
    k_sel = torch.gather(kh.unsqueeze(3).expand(bsz, nheads, tgt_len, topk, head_dim), dim=2, index=index)
    v_sel = torch.gather(vh.unsqueeze(3).expand(bsz, nheads, tgt_len, topk, head_dim), dim=2, index=index)

    scores = (qh.unsqueeze(3) * k_sel).sum(-1) * scale
    if bias is not None:
        scores = scores + bias
    scores = scores.masked_fill(~valid[:, None, :, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(~valid[:, None, :, :], 0.0)
    out = (attn.unsqueeze(-1) * v_sel).sum(dim=3)
    attn_sum = attn.sum(dim=1)
    return out.transpose(1, 2), attn_sum


def _reference_indexer_scores(q, k, w, mask, softmax_scale):
    scores = torch.einsum("bthd,bshd->bths", q, k) * softmax_scale
    scores = torch.relu(scores)
    scores = scores * w.permute(0, 2, 1).unsqueeze(1)
    scores = scores.sum(dim=2)
    if mask is not None:
        scores = scores + mask
    return scores


def _reference_indexer_topk(q, k, w, mask, topk, softmax_scale):
    scores = _reference_indexer_scores(q, k, w, mask, softmax_scale)
    topk = min(int(topk), scores.shape[-1])
    topk_scores, topk_indices = torch.topk(scores, topk, dim=-1)
    return topk_indices, topk_scores


def _kernel_indexer_topk(q, k, w, mask, topk, softmax_scale, block_k=256):
    if sparse_topk_indexer_func is None:
        raise RuntimeError(f"sparse_topk_indexer unavailable: {INDEXER_IMPORT_ERROR}")
    mask_tensor = mask if mask is not None else q.new_empty(0)
    return sparse_topk_indexer_func(q, k, w, mask_tensor, topk, float(softmax_scale), int(block_k))


def _kernel_indexer_topk_with_sorted_mode(q, k, w, mask, topk, softmax_scale, block_k=256, sorted_mode=False):
    key = "VGGT_INDEXER_TOPK_SORTED"
    previous = os.environ.get(key)
    os.environ[key] = "1" if sorted_mode else "0"
    try:
        return _kernel_indexer_topk(q, k, w, mask, topk, softmax_scale, block_k=block_k)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _run_indexer_case(batch=1, seqlen=128, nheads=4, head_dim=64, topk=32, dtype=None):
    device = torch.device("cuda")
    if dtype is None:
        dtype = torch.bfloat16
    q = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    w = torch.randn(batch, seqlen, nheads, device=device, dtype=dtype, requires_grad=True)
    mask = None
    scale = 1.0 / math.sqrt(head_dim)

    scores_full = _reference_indexer_scores(q, k, w, mask, scale)
    idx_ref, scores_ref = _reference_indexer_topk(q, k, w, mask, topk, scale)
    idx_ker, scores_ker = _kernel_indexer_topk_with_sorted_mode(
        q, k, w, mask, topk, scale, block_k=256, sorted_mode=False
    )
    idx_ker_sorted, scores_ker_sorted = _kernel_indexer_topk_with_sorted_mode(
        q, k, w, mask, topk, scale, block_k=256, sorted_mode=True
    )

    idx_min = idx_ker.min().item()
    idx_max = idx_ker.max().item()
    print(f"[indexer fwd] idx range {idx_min}..{idx_max}")

    idx_ker_cpu = idx_ker.cpu()
    idx_ref_cpu = idx_ref.cpu()
    scores_ref_cpu = scores_full.cpu()
    scores_ker_cpu = scores_ker.cpu()

    invalid = ((idx_ker_cpu < 0) | (idx_ker_cpu >= seqlen)).sum().item()
    if invalid:
        print(f"[indexer fwd] invalid indices {invalid}")

    ref_at_ker = scores_ref_cpu.gather(-1, idx_ker_cpu.to(torch.long))
    diff = (scores_ker_cpu - ref_at_ker).abs()
    match = (idx_ref_cpu.sort(-1).values == idx_ker_cpu.sort(-1).values).float().mean().item()
    print(
        f"[indexer fwd] max diff {diff.max().item():.6f}, mean diff {diff.mean().item():.6f}, "
        f"match {match:.4f}"
    )
    ker_match = (
        idx_ker_sorted.cpu().sort(-1).values == idx_ker.cpu().sort(-1).values
    ).float().mean().item()
    score_gap = (
        scores_ker_sorted.cpu().sort(-1).values - scores_ker_cpu.sort(-1).values
    ).abs()
    print(
        f"[indexer fwd sorted-vs-unsorted] max diff {score_gap.max().item():.6f}, "
        f"mean diff {score_gap.mean().item():.6f}, match {ker_match:.4f}"
    )

    loss_ker = scores_ker.sum()
    loss_ref = scores_ref.sum()
    loss_ker.backward(retain_graph=True)
    grads_ker = (q.grad.float().clone(), k.grad.float().clone(), w.grad.float().clone())
    q.grad.zero_()
    k.grad.zero_()
    w.grad.zero_()
    loss_ref.backward()
    grads_ref = (q.grad.float(), k.grad.float(), w.grad.float())

    for name, gk, gr in zip(("dq", "dk", "dw"), grads_ker, grads_ref):
        gdiff = (gk - gr).abs()
        rel = (gdiff / gr.abs().clamp(min=1e-6)).mean().item()
        print(f"[indexer bwd] {name} max {gdiff.max().item():.6f}, mean {gdiff.mean().item():.6f}, rel {rel:.6f}")

    iters = 10

    def _ker_forward(sorted_mode):
        _kernel_indexer_topk_with_sorted_mode(q, k, w, None, topk, scale, block_k=256, sorted_mode=sorted_mode)

    def _ker_backward(sorted_mode):
        idx_k, scores_k = _kernel_indexer_topk_with_sorted_mode(
            q, k, w, None, topk, scale, block_k=256, sorted_mode=sorted_mode
        )
        loss_k = scores_k.sum()
        loss_k.backward()
        if q.grad is not None:
            q.grad.zero_()
        if k.grad is not None:
            k.grad.zero_()
        if w.grad is not None:
            w.grad.zero_()

    def _ref_forward():
        scores_full = _reference_indexer_scores(q, k, w, None, scale)
        torch.topk(scores_full, topk, dim=-1)

    def _ref_backward():
        scores_full = _reference_indexer_scores(q, k, w, None, scale)
        scores_topk = torch.topk(scores_full, topk, dim=-1).values
        loss_r = scores_topk.sum()
        loss_r.backward()
        if q.grad is not None:
            q.grad.zero_()
        if k.grad is not None:
            k.grad.zero_()
        if w.grad is not None:
            w.grad.zero_()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ker_forward(False)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(iters):
        _ker_forward(True)
    torch.cuda.synchronize()
    t_sorted = time.time()
    for _ in range(iters):
        _ref_forward()
    torch.cuda.synchronize()
    t2 = time.time()
    print(
        f"[indexer perf] fwd kernel(unsorted) {t1 - t0:.4f}s, "
        f"kernel(sorted) {t_sorted - t1:.4f}s, dense_ref {t2 - t_sorted:.4f}s"
    )

    torch.cuda.synchronize()
    t3 = time.time()
    for _ in range(iters):
        _ker_backward(False)
    torch.cuda.synchronize()
    t4 = time.time()
    for _ in range(iters):
        _ker_backward(True)
    torch.cuda.synchronize()
    t_sorted_bwd = time.time()
    for _ in range(iters):
        _ref_backward()
    torch.cuda.synchronize()
    t5 = time.time()
    print(
        f"[indexer perf] bwd kernel(unsorted) {t4 - t3:.4f}s, "
        f"kernel(sorted) {t_sorted_bwd - t4:.4f}s, dense_ref {t5 - t_sorted_bwd:.4f}s"
    )


def _run_joint_case(batch=1, seqlen=128, nheads=8, head_dim=64, topk=32, dtype=None):
    device = torch.device("cuda")
    if dtype is None:
        dtype = torch.bfloat16

    q = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    idx_q = torch.randn(batch, seqlen, 4, head_dim, device=device, dtype=dtype, requires_grad=True)
    idx_k = torch.randn_like(idx_q, requires_grad=True)
    idx_w = torch.randn(batch, seqlen, 4, device=device, dtype=dtype, requires_grad=True)

    scale_attn = 1.0 / math.sqrt(head_dim)
    scale_idx = 1.0 / math.sqrt(head_dim)

    idx_ref, scores_ref = _reference_indexer_topk(idx_q, idx_k, idx_w, None, topk, scale_idx)
    idx_ker, scores_ker = _kernel_indexer_topk(idx_q, idx_k, idx_w, None, topk, scale_idx)

    out_ref, attn_ref = _reference_sparse_attn(q, k, v, idx_ref, bias=None, softmax_scale=scale_attn)
    out_ker, attn_ker = sparse_index_flash_attn_func(q, k, v, idx_ker.to(torch.int32), None, False, scale_attn)

    diff_out = (out_ker - out_ref).abs()
    print(f"[joint] out max {diff_out.max().item():.6f}, mean {diff_out.mean().item():.6f}")

    eps = 1e-6
    p_ref = attn_ref.sum(dim=1)
    p_ref = p_ref / (p_ref.sum(dim=-1, keepdim=True) + eps)
    p_ker = attn_ker / (attn_ker.sum(dim=-1, keepdim=True) + eps)

    loss_ref = (p_ref * (torch.log(p_ref + eps) - F.log_softmax(scores_ref, dim=-1))).sum(dim=-1).mean()
    loss_ker = (p_ker * (torch.log(p_ker + eps) - F.log_softmax(scores_ker, dim=-1))).sum(dim=-1).mean()
    print(f"[joint] indexer loss diff {(loss_ker - loss_ref).abs().item():.6f}")


def _run_joint_memory_case(views=12, img=518, patch=14, num_register=4, topk=512, dtype=None):
    device = torch.device("cuda")
    if dtype is None:
        dtype = torch.bfloat16

    tokens_per_view = (img // patch) * (img // patch) + 1 + num_register
    seqlen = views * tokens_per_view
    bsz = 1
    nheads_attn = 16
    head_dim_attn = 64
    nheads_idx = 4
    head_dim_idx = 64

    print(f"[joint mem] views={views}, tokens_per_view={tokens_per_view}, T={seqlen}")

    q = torch.randn(bsz, seqlen, nheads_attn, head_dim_attn, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    idx_q = torch.randn(bsz, seqlen, nheads_idx, head_dim_idx, device=device, dtype=dtype, requires_grad=True)
    idx_k = torch.randn_like(idx_q, requires_grad=True)
    idx_w = torch.randn(bsz, seqlen, nheads_idx, device=device, dtype=dtype, requires_grad=True)

    scale_attn = 1.0 / math.sqrt(head_dim_attn)
    scale_idx = 1.0 / math.sqrt(head_dim_idx)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    idx_ker, scores_ker = _kernel_indexer_topk(idx_q, idx_k, idx_w, None, topk, scale_idx)
    out_ker, attn_ker = sparse_index_flash_attn_func(q, k, v, idx_ker.to(torch.int32), None, False, scale_attn)
    eps = 1e-6
    p_ker = attn_ker / (attn_ker.sum(dim=-1, keepdim=True) + eps)
    loss_ker = (p_ker * (torch.log(p_ker + eps) - F.log_softmax(scores_ker, dim=-1))).sum(dim=-1).mean()
    loss_ker.backward()

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    print(f"[joint mem] kernel peak {peak / (1024 ** 2):.1f} MiB")

    bytes_bf16 = 2
    bytes_f32 = 4
    n_qkv = bsz * seqlen * nheads_attn * head_dim_attn
    n_attn = bsz * nheads_attn * seqlen * topk
    n_kv = bsz * nheads_attn * seqlen * topk * head_dim_attn
    attn_lower = (n_qkv * 3 + n_qkv) * bytes_bf16 + n_kv * 2 * bytes_bf16 + n_attn * bytes_bf16 * 2

    n_scores = bsz * seqlen * seqlen
    idx_lower = n_scores * bytes_bf16
    idx_upper = n_scores * bytes_f32
    print(f"[joint mem] ref_attention_lower approx {attn_lower / (1024 ** 3):.1f} GiB")
    print(f"[joint mem] ref_indexer_scores approx {idx_lower / (1024 ** 3):.1f}-{idx_upper / (1024 ** 3):.1f} GiB")


def _run_case(batch=2, seqlen=128, nheads=4, head_dim=64, topk=64, dtype=None):
    device = torch.device("cuda")
    if dtype is None:
        dtype = torch.bfloat16
    q = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    kv_positions = _make_kv_positions(batch, seqlen, topk, device)
    bias = None

    scale = 1.0 / (head_dim ** 0.5)
    out_ref, attn_ref = _reference_sparse_attn(q, k, v, kv_positions, bias=bias, softmax_scale=scale)

    out, attn = sparse_index_flash_attn_func(q, k, v, kv_positions, bias, False, scale)

    out_ref = out_ref.to(out.dtype)
    attn_ref = attn_ref.to(attn.dtype)

    diff = (out - out_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / out_ref.abs().clamp(min=1e-6)).mean().item()
    attn_diff = (attn - attn_ref).abs()
    max_attn_diff = attn_diff.max().item()
    mean_attn_rel = (attn_diff / attn_ref.abs().clamp(min=1e-6)).mean().item()
    print(
        f"[fwd] max diff {max_diff:.6f}, mean diff {mean_diff:.6f}, rel {rel_diff:.6f}, "
        f"attn max diff {max_attn_diff:.6f}, attn rel {mean_attn_rel:.6f}"
    )

    do = torch.randn_like(out)
    dattn = torch.randn_like(attn)

    grads = torch.autograd.grad(
        outputs=(out, attn),
        inputs=(q, k, v),
        grad_outputs=(do, dattn),
        retain_graph=True,
        allow_unused=False,
    )

    grads_ref = torch.autograd.grad(
        outputs=(out_ref, attn_ref),
        inputs=(q, k, v),
        grad_outputs=(do, dattn),
        retain_graph=True,
        allow_unused=False,
    )

    for name, g, g_ref in zip(("dq", "dk", "dv"), grads, grads_ref):
        diff = (g.float() - g_ref.float()).abs()
        rel = (diff / g_ref.float().abs().clamp(min=1e-6)).mean().item()
        print(f"[bwd] {name} max {diff.max().item():.6f}, mean {diff.mean().item():.6f}, rel {rel:.6f}")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        sparse_index_flash_attn_func(q, k, v, kv_positions, bias, False, scale)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(10):
        _reference_sparse_attn(q, k, v, kv_positions, bias=bias, softmax_scale=scale)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"[perf] kernel {t1 - t0:.4f}s, ref {t2 - t1:.4f}s")

    def _measure_peak(name, fn):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        print(f"[mem] {name} peak {peak / (1024 ** 2):.1f} MiB")

    def _run_kernel():
        qk = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
        kk = torch.randn_like(qk, requires_grad=True)
        vk = torch.randn_like(qk, requires_grad=True)
        out_k, attn_k = sparse_index_flash_attn_func(qk, kk, vk, kv_positions, bias, False, scale)
        do_k = torch.randn_like(out_k)
        dattn_k = torch.randn_like(attn_k)
        loss_k = (out_k * do_k).sum() + (attn_k * dattn_k).sum()
        loss_k.backward()

    def _run_ref():
        qr = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
        kr = torch.randn_like(qr, requires_grad=True)
        vr = torch.randn_like(qr, requires_grad=True)
        out_r, attn_r = _reference_sparse_attn(qr, kr, vr, kv_positions, bias=bias, softmax_scale=scale)
        do_r = torch.randn_like(out_r)
        dattn_r = torch.randn_like(attn_r)
        loss_r = (out_r * do_r).sum() + (attn_r * dattn_r).sum()
        loss_r.backward()

    _measure_peak("kernel", _run_kernel)
    _measure_peak("ref", _run_ref)


def main():
    if torch is None:
        print(f"torch unavailable: {TORCH_ERROR}")
        return
    if sparse_index_flash_attn_func is None:
        print(f"sparse_index_flash_attn unavailable: {IMPORT_ERROR}")
        return
    if not torch.cuda.is_available():
        print("CUDA not available; skip GPU validation.")
        return

    torch.manual_seed(0)
    random.seed(0)

    _run_case(batch=2, seqlen=128, nheads=4, head_dim=64, topk=64, dtype=torch.bfloat16)
    _run_case(batch=1, seqlen=256, nheads=8, head_dim=64, topk=128, dtype=torch.bfloat16)

    if sparse_topk_indexer_func is None:
        print(f"sparse_topk_indexer unavailable: {INDEXER_IMPORT_ERROR}")
        return

    _run_indexer_case(batch=1, seqlen=128, nheads=4, head_dim=64, topk=32, dtype=torch.bfloat16)
    _run_joint_case(batch=1, seqlen=128, nheads=8, head_dim=64, topk=32, dtype=torch.bfloat16)
    _run_joint_memory_case(views=12, topk=512, dtype=torch.bfloat16)
    _run_joint_memory_case(views=24, topk=512, dtype=torch.bfloat16)


if __name__ == "__main__":
    main()
