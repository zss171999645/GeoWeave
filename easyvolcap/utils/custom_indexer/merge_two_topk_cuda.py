import os
import sys
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

    python_bin = Path(sys.executable).resolve().parent
    if (python_bin / "ninja").is_file():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(python_bin) not in path_entries:
            os.environ["PATH"] = str(python_bin) + os.pathsep + os.environ.get("PATH", "")

    src_dir = Path(__file__).resolve().parent / "csrc"
    cpp = src_dir / "merge_two_topk.cpp"
    cu = src_dir / "merge_two_topk_kernel.cu"

    try:
        _EXT = load(
            name="vggt_merge_two_topk_cuda",
            sources=[str(cpp), str(cu)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.getenv("VGGT_INDEXER_MERGE_TWO_VERBOSE", "0") == "1",
        )
    except Exception as exc:
        _EXT_LOAD_ERROR = exc
        return None

    return _EXT


def merge_two_topk_inplace(
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    pending_scores: torch.Tensor,
    pending_pos: torch.Tensor,
    pending_start: int,
    out_scores: torch.Tensor,
    out_indices: torch.Tensor,
) -> bool:
    global _EXT_CALL_ERROR_COUNT
    ext = _load_extension()
    if ext is None:
        if os.getenv("VGGT_INDEXER_MERGE_TWO_DEBUG", "0") == "1":
            print("[merge_two_topk_cuda] extension unavailable:", repr(_EXT_LOAD_ERROR))
        return False

    try:
        ext.merge_two_topk_cuda(
            topk_scores,
            topk_indices,
            pending_scores,
            pending_pos,
            out_scores,
            out_indices,
            int(pending_start),
        )
        return True
    except Exception as exc:
        _EXT_CALL_ERROR_COUNT += 1
        if os.getenv("VGGT_INDEXER_MERGE_TWO_DEBUG", "0") == "1" and _EXT_CALL_ERROR_COUNT <= 8:
            print("[merge_two_topk_cuda] call failed:", repr(exc))
        return False


def is_merge_two_topk_available() -> bool:
    return _load_extension() is not None
