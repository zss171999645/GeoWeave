from __future__ import annotations

import sys
import os
import subprocess
import tempfile
import unittest
from importlib.machinery import PathFinder
from pathlib import Path

import numpy as np

import einops
from einops import rearrange


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_EINOPS = (REPO_ROOT / "einops" / "__init__.py").resolve()


def _find_external_einops() -> Path | None:
    search_paths = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if not resolved.exists() or not resolved.is_dir():
            continue
        if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
            continue
        search_paths.append(str(resolved))

    spec = PathFinder.find_spec("einops", search_paths)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve()


class EinopsShimTests(unittest.TestCase):
    def test_vggt_sampler_rearrange_pattern_works(self):
        x = np.zeros((1, 2, 12, 3), dtype=np.float32)
        y = rearrange(x, "b n (h w) c -> b n c h w", h=3, w=4)
        self.assertEqual(tuple(y.shape), (1, 2, 3, 3, 4))

    def test_real_package_is_preferred_when_available(self):
        external = _find_external_einops()
        if external is None:
            self.skipTest("No external einops installation found")

        self.assertNotEqual(LOCAL_EINOPS, external)
        self.assertEqual(Path(einops.__file__).resolve(), external)

    def test_sibling_meshx_checkout_shim_is_not_treated_as_real_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_repo = Path(tmp) / "meshx_5090"
            fake_einops = fake_repo / "einops"
            fake_easyvolcap = fake_repo / "easyvolcap"
            fake_einops.mkdir(parents=True)
            fake_easyvolcap.mkdir(parents=True)
            fake_easyvolcap.joinpath("__init__.py").write_text("", encoding="utf-8")
            fake_einops.joinpath("__init__.py").write_text(
                "def rearrange(*args, **kwargs):\n"
                "    raise RuntimeError('loaded sibling shim')\n",
                encoding="utf-8",
            )

            env = dict(os.environ)
            env["PYTHONPATH"] = f"{REPO_ROOT}:{fake_repo}"
            code = f"""
from pathlib import Path
import numpy as np
import einops
from einops import rearrange
fake = Path({str(fake_einops / "__init__.py")!r}).resolve()
assert Path(einops.__file__).resolve() != fake, einops.__file__
x = np.zeros((1, 2, 12, 3), dtype=np.float32)
y = rearrange(x, "b n (h w) c -> b n c h w", h=3, w=4)
assert tuple(y.shape) == (1, 2, 3, 3, 4), y.shape
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
