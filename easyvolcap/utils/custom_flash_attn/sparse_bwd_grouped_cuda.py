import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_EXT = None
_EXT_LOAD_ERROR: Optional[Exception] = None
_EXT_CALL_ERROR_COUNT = 0


def _load_extension():
    global _EXT
    global _EXT_LOAD_ERROR

    if _EXT is not None:
        return _EXT
    if _EXT_LOAD_ERROR is not None:
        return None

    src_dir = Path(__file__).resolve().parent / "csrc"
    cpp = src_dir / "sparse_bwd_grouped.cpp"
    cu = src_dir / "sparse_bwd_grouped_kernel.cu"

    try:
        _EXT = load(
            name="vggt_sparse_bwd_grouped_cuda",
            sources=[str(cpp), str(cu)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.getenv("VGGT_SPARSE_FLASH_BWD_NATIVE_VERBOSE", "0") == "1",
        )
    except Exception as exc:
        _EXT_LOAD_ERROR = exc
        return None

    return _EXT


def sparse_prob_bwd_grouped(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_pos: torch.Tensor,
    do: torch.Tensor,
    delta: torch.Tensor,
    prob: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    *,
    softmax_scale: float,
    q_layout: str,
    query_chunk: int,
    dattn: Optional[torch.Tensor] = None,
) -> None:
    global _EXT_CALL_ERROR_COUNT

    ext = _load_extension()
    if ext is None:
        raise RuntimeError(f"sparse_bwd_grouped extension unavailable: {_EXT_LOAD_ERROR!r}")

    if q_layout not in ("bthd", "bhtd"):
        raise ValueError(f"unsupported q_layout={q_layout}")
    layout_code = 0 if q_layout == "bthd" else 1

    if dattn is None:
        dattn = q.new_empty(0)

    if not (q.is_cuda and k.is_cuda and v.is_cuda and kv_pos.is_cuda and do.is_cuda and delta.is_cuda and prob.is_cuda):
        raise ValueError("inputs must be CUDA tensors")

    if kv_pos.dtype != torch.int32:
        raise ValueError(f"kv_pos must be int32, got {kv_pos.dtype}")
    if delta.dtype != torch.float32:
        raise ValueError(f"delta must be float32, got {delta.dtype}")
    if dq.dtype != torch.float32 or dk.dtype != torch.float32 or dv.dtype != torch.float32:
        raise ValueError("dq/dk/dv must be float32 for grouped backward")

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not do.is_contiguous():
        do = do.contiguous()
    if not kv_pos.is_contiguous():
        kv_pos = kv_pos.contiguous()
    if not prob.is_contiguous():
        prob = prob.contiguous()
    if dattn.numel() > 0 and (not dattn.is_contiguous()):
        dattn = dattn.contiguous()

    try:
        ext.sparse_prob_bwd_grouped_cuda(
            q,
            k,
            v,
            kv_pos,
            do,
            delta,
            prob,
            dattn,
            dq,
            dk,
            dv,
            float(softmax_scale),
            int(layout_code),
            int(query_chunk),
        )
    except Exception as exc:
        _EXT_CALL_ERROR_COUNT += 1
        raise RuntimeError(f"sparse_bwd_grouped call failed ({_EXT_CALL_ERROR_COUNT}): {exc!r}") from exc


def is_sparse_prob_bwd_grouped_available() -> bool:
    return _load_extension() is not None
