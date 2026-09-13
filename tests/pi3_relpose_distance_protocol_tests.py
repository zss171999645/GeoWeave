from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "aidi" / "scripts" / "baselines" / "eval_pi3_relpose_distance_protocol.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("eval_pi3_relpose_distance_protocol", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pi3RelposeDistanceProtocolTests(unittest.TestCase):
    def test_list_images_for_sequence_falls_back_to_supported_extensions(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "scene0000_00" / "color_90"
            image_dir.mkdir(parents=True)
            (image_dir / "frame_0000.png").write_bytes(b"png")
            (image_dir / "frame_0001.jpeg").write_bytes(b"jpeg")
            (image_dir / "notes.txt").write_text("ignore", encoding="utf-8")

            spec = module.DATASET_SPECS["scannetv2"]
            images = module.list_images_for_sequence(spec=spec, root=root, seq="scene0000_00")

        self.assertEqual([Path(path).name for path in images], ["frame_0000.png", "frame_0001.jpeg"])


if __name__ == "__main__":
    unittest.main()
