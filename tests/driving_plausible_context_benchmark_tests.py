import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "aidi" / "scripts" / "baselines" / "build_driving_plausible_context_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_driving_plausible_context_benchmark", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DrivingPlausibleContextBenchmarkTests(unittest.TestCase):
    def test_parse_args_keeps_shared_discovery_compatibility_fields(self):
        module = load_module()
        with mock.patch.object(sys, "argv", ["prog"]):
            args = module.parse_args()

        self.assertEqual(args.waymo_camera, "00")
        self.assertEqual(args.kitti_camera, "2")
        self.assertEqual(args.vkitti_camera, "00")

    def test_retrieval_selects_visually_closest_cross_scene_candidate(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def make_image(path: Path, value: int) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((8, 12, 3), value, dtype=np.uint8)).save(path)

            target_images = {}
            for idx in range(10):
                path = root / "target" / f"{idx:06d}.jpg"
                make_image(path, 80)
                target_images[f"{idx:06d}"] = path

            close_images = {}
            far_images = {}
            for idx in range(10):
                close = root / "close" / f"{idx:06d}.jpg"
                far = root / "far" / f"{idx:06d}.jpg"
                make_image(close, 82)
                make_image(far, 220)
                close_images[f"{idx:06d}"] = close
                far_images[f"{idx:06d}"] = far

            target = {"frames": sorted(target_images), "images": target_images}
            close = {"frames": sorted(close_images), "images": close_images}
            far = {"frames": sorted(far_images), "images": far_images}
            target_tail = module.sequence_feature(
                target,
                ["000006", "000007", "000008", "000009"],
                size=(4, 3),
                cache={},
            )
            args = type(
                "Args",
                (),
                {
                    "noise_views": 4,
                    "frame_stride": 1,
                    "retrieval_candidates_per_scene": 1,
                    "retrieval_quantiles": "0.0",
                    "allow_resolution_mismatch": False,
                },
            )()
            payload, frames, start, score = module.find_plausible_distractor_selection(
                scene_payloads=[target, far, close],
                target_scene_index=0,
                target_tail_feature=target_tail,
                target_resolution=(12, 8),
                args=args,
                feature_size=(4, 3),
                feature_cache={},
            )

        self.assertIs(payload, close)
        self.assertEqual(frames, ["000000", "000001", "000002", "000003"])
        self.assertEqual(start, 0)
        self.assertLess(score, 0.01)


if __name__ == "__main__":
    unittest.main()
