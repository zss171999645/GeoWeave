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
    cpp = src_dir / "topk_exact.cpp"
    cu = src_dir / "topk_exact_kernel.cu"

    try:
        _EXT = load(
            name="vggt_topk_exact_cuda",
            sources=[str(cpp), str(cu)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.getenv("VGGT_INDEXER_TOPK_EXACT_VERBOSE", "0") == "1",
        )
    except Exception as exc:
        _EXT_LOAD_ERROR = exc
        return None

    return _EXT


def topk_exact_select(
    scores: torch.Tensor,
    out_scores: Optional[torch.Tensor] = None,
    out_pos: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global _EXT_CALL_ERROR_COUNT

    ext = _load_extension()
    if ext is None:
        raise RuntimeError(f"topk_exact extension unavailable: {_EXT_LOAD_ERROR!r}")

    if scores.ndim != 3:
        raise ValueError(f"scores must be 3D, got shape={tuple(scores.shape)}")
    if not scores.is_cuda:
        raise ValueError("scores must be CUDA tensor")
    if scores.shape[-1] < 512:
        raise ValueError(f"scores last dim must be >= 512, got {scores.shape[-1]}")

    if not scores.is_contiguous():
        scores = scores.contiguous()

    bsz, rows, _ = scores.shape
    if out_scores is None:
        out_scores = torch.empty((bsz, rows, 512), device=scores.device, dtype=scores.dtype)
    if out_pos is None:
        out_pos = torch.empty((bsz, rows, 512), device=scores.device, dtype=torch.long)

    try:
        ext.topk_exact_cuda(scores, out_scores, out_pos)
    except Exception as exc:
        _EXT_CALL_ERROR_COUNT += 1
        raise RuntimeError(f"topk_exact call failed ({_EXT_CALL_ERROR_COUNT}): {exc!r}") from exc

    return out_scores, out_pos


def is_topk_exact_available() -> bool:
    return _load_extension() is not None
