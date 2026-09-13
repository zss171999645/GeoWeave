import os
import sys
from pathlib import Path
from typing import Optional, Tuple

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

    python_bin = Path(sys.executable).resolve().parent
    if (python_bin / "ninja").is_file():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(python_bin) not in path_entries:
            os.environ["PATH"] = str(python_bin) + os.pathsep + os.environ.get("PATH", "")

    src_dir = Path(__file__).resolve().parent / "csrc"
    cpp = src_dir / "score_topk_fused.cpp"
    cu = src_dir / "score_topk_fused_kernel.cu"

    try:
        _EXT = load(
            name="vggt_score_topk_fused_cuda",
            sources=[str(cpp), str(cu)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.getenv("VGGT_INDEXER_SCORE_TOPK_FUSED_VERBOSE", "0") == "1",
        )
    except Exception as exc:
        _EXT_LOAD_ERROR = exc
        return None

    return _EXT


def score_topk_fused_select(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    *,
    softmax_scale: float,
    out_scores: Optional[torch.Tensor] = None,
    out_pos: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global _EXT_CALL_ERROR_COUNT

    ext = _load_extension()
    if ext is None:
        raise RuntimeError(f"score_topk_fused extension unavailable: {_EXT_LOAD_ERROR!r}")

    if q.ndim != 4 or k.ndim != 4 or w.ndim != 3:
        raise ValueError(
            f"invalid shapes q={tuple(q.shape)} k={tuple(k.shape)} w={tuple(w.shape)}; "
            "expect q/k as [B,T(orS),H,D] and w as [B,S,H]"
        )
    if not (q.is_cuda and k.is_cuda and w.is_cuda):
        raise ValueError("q/k/w must be CUDA tensors")

    bq, tq, hq, dq = q.shape
    bk, sk, hk, dk = k.shape
    bw, sw, hw = w.shape
    if bq != bk or bq != bw or hq != hk or hq != hw or dq != dk or sk != sw:
        raise ValueError("q/k/w shape mismatch")

    if out_scores is None:
        out_scores = torch.empty((bq, tq, 512), device=q.device, dtype=q.dtype)
    if out_pos is None:
        out_pos = torch.empty((bq, tq, 512), device=q.device, dtype=torch.int32)

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()

    try:
        ext.score_topk_fused_cuda(q, k, w, out_scores, out_pos, float(softmax_scale))
    except Exception as exc:
        _EXT_CALL_ERROR_COUNT += 1
        raise RuntimeError(f"score_topk_fused call failed ({_EXT_CALL_ERROR_COUNT}): {exc!r}") from exc

    return out_scores, out_pos


def is_score_topk_fused_available() -> bool:
    return _load_extension() is not None
