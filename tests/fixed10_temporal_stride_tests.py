from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "aidi" / "scripts" / "baselines" / "build_fixed10_temporal_stride_benchmark.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_fixed10_temporal_stride_benchmark", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fixed10TemporalStrideTests(unittest.TestCase):
    def test_select_start_positions_respects_stride_window(self):
        module = load_script_module()
        starts = module.select_start_positions(
            num_frames=30,
            stride=2,
            num_views=10,
            max_anchors=3,
            anchor_quantiles=[0.0, 0.5, 1.0],
        )
        self.assertEqual(starts, [0, 6, 11])
        self.assertEqual(module.select_start_positions(num_frames=9, stride=1, num_views=10, max_anchors=3, anchor_quantiles=[0.5]), [])

    def test_build_temporal_stride_materializes_fixed10_tuples(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "7scenes"
            scene_root = dataset_root / "chess_seq-03"
            (scene_root / "images").mkdir(parents=True)
            (scene_root / "depths").mkdir(parents=True)
            output_root = Path(tmpdir) / "out"
            fake_scene_record = {
                "scene_name": "chess_seq-03",
                "frame_ids": list(range(30)),
                "color_paths": [],
                "depth_paths": [],
                "poses": np.zeros((30, 4, 4), dtype=np.float32),
                "intrinsics": [],
            }

            def fake_materialize(**kwargs):
                tuple_root = output_root / kwargs["tuple_name"]
                tuple_root.mkdir(parents=True, exist_ok=True)
                return tuple_root

            with mock.patch.object(module, "load_scene_record", return_value=fake_scene_record):
                with mock.patch.object(module, "materialize_tuple_sequence", side_effect=fake_materialize):
                    summary_path = module.build_fixed10_temporal_stride_benchmark(
                        dataset_root=dataset_root,
                        output_root=output_root,
                        scene_specs=["chess_seq-03"],
                        strides=[1, 2],
                        anchor_quantiles=[0.5],
                        max_anchors_per_scene=1,
                        image_subdir="images",
                        depth_subdir="depths",
                        camera_subdir="cameras/00",
                        num_views=10,
                        source_layout="nested_evc",
                    )

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_tuples"], 2)
            self.assertTrue(all(len(row["ordered_frame_ids"]) == 10 for row in payload["tuples"]))
            self.assertEqual(payload["tuples"][0]["eval_frame_indices"], list(range(10)))


if __name__ == "__main__":
    unittest.main()
