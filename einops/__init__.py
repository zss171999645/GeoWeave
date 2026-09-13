from __future__ import annotations

import sys
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path
from typing import Any


_THIS_FILE = Path(__file__).resolve()
_THIS_PACKAGE_DIR = _THIS_FILE.parent.resolve()
_THIS_REPO_ROOT = _THIS_PACKAGE_DIR.parent.resolve()


def _load_real_einops():
    def _looks_like_meshx_checkout(origin: Path) -> bool:
        package_dir = origin.parent
        repo_root = package_dir.parent
        return (repo_root / "easyvolcap").is_dir() and (repo_root / "einops").resolve() == package_dir

    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if not resolved.exists() or not resolved.is_dir():
            continue
        if resolved == _THIS_PACKAGE_DIR or resolved == _THIS_REPO_ROOT or _THIS_REPO_ROOT in resolved.parents:
            continue

        spec = PathFinder.find_spec(__name__, [str(resolved)])
        if spec is None or spec.loader is None or spec.origin is None:
            continue

        origin = Path(spec.origin).resolve()
        if origin == _THIS_FILE or _looks_like_meshx_checkout(origin):
            continue

        module = module_from_spec(spec)
        sys.modules[__name__] = module
        spec.loader.exec_module(module)
        return module

    return None


_REAL_MODULE = _load_real_einops()

if _REAL_MODULE is not None:
    globals().update(_REAL_MODULE.__dict__)
else:
    def _normalize_pattern(pattern: str) -> str:
        return " ".join(pattern.strip().split())


    def rearrange(tensor, pattern: str, **axes_lengths: Any):
        normalized = _normalize_pattern(pattern)

        if normalized == "b n c -> (b n) c":
            b, n, c = tensor.shape
            return tensor.reshape(b * n, c)

        if normalized == "b n (h w) c -> b n c h w":
            if "h" not in axes_lengths or "w" not in axes_lengths:
                raise ValueError("Missing axis lengths `h`/`w` for pattern 'b n (h w) c -> b n c h w'")
            b, n, hw, c = tensor.shape
            h = int(axes_lengths["h"])
            w = int(axes_lengths["w"])
            if hw != h * w:
                raise ValueError(f"Cannot reshape third axis {hw} into h={h} and w={w}")
            return tensor.reshape(b, n, h, w, c).transpose(0, 1, 4, 2, 3)

        if normalized == "(b n) s d -> b n s d":
            if "b" not in axes_lengths:
                raise ValueError("Missing axis length `b` for pattern '(b n) s d -> b n s d'")
            b = int(axes_lengths["b"])
            bn, s, d = tensor.shape
            if bn % b != 0:
                raise ValueError(f"Cannot reshape first axis {bn} into b={b} and integer n")
            n = bn // b
            return tensor.reshape(b, n, s, d)

        if normalized == "b n s d -> (b n) s d":
            b, n, s, d = tensor.shape
            return tensor.reshape(b * n, s, d)

        if normalized == "b s h w c -> b s (h w) c":
            b, s, h, w, c = tensor.shape
            return tensor.reshape(b, s, h * w, c)

        if normalized == "b s h w -> b s (h w) 1":
            b, s, h, w = tensor.shape
            return tensor.reshape(b, s, h * w, 1)

        if normalized == "b (h w) c -> b c h w":
            if "h" not in axes_lengths or "w" not in axes_lengths:
                raise ValueError("Missing axis lengths `h`/`w` for pattern 'b (h w) c -> b c h w'")
            b, hw, c = tensor.shape
            h = int(axes_lengths["h"])
            w = int(axes_lengths["w"])
            if hw != h * w:
                raise ValueError(f"Cannot reshape second axis {hw} into h={h} and w={w}")
            return tensor.reshape(b, h, w, c).transpose(0, 3, 1, 2)

        raise NotImplementedError(f"Unsupported minimal einops.rearrange pattern: {pattern}")


    def repeat(*args: Any, **kwargs: Any):
        raise NotImplementedError("Minimal einops compatibility layer does not implement repeat()")


    def reduce(*args: Any, **kwargs: Any):
        raise NotImplementedError("Minimal einops compatibility layer does not implement reduce()")
