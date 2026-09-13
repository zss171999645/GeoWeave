from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2


def _normalize_exr_depth(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3:
        depth = depth[..., 0]
    return np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def read_exr_depth(filename: str, *, retries: int = 1, retry_sleep_s: float = 0.05) -> np.ndarray:
    path = Path(filename)
    for attempt in range(retries + 1):
        depth = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is not None and depth.size > 0:
            return _normalize_exr_depth(depth)

        try:
            encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        except Exception:
            depth = None
        if depth is not None and depth.size > 0:
            return _normalize_exr_depth(depth)

        if attempt < retries:
            time.sleep(retry_sleep_s)

    raise ValueError(
        "Failed to read EXR depth via OpenCV path decode and byte decode. "
        f"File: {filename}"
    )
