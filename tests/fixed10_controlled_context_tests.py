from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "aidi" / "scripts" / "baselines" / "build_fixed10_controlled_context_benchmark.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_fixed10_controlled_context_benchmark", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fixed10ControlledContextTests(unittest.TestCase):
    def test_select_controlled_context_keeps_same_core_for_all_variants(self):
        module = load_script_module()
        overlap_by_ref = {
            1: 0.95,
            2: 0.90,
            3: 0.85,
            4: 0.80,
            5: 0.75,
            6: 0.18,
            7: 0.15,
            8: 0.08,
            9: 0.06,
            10: 0.04,
            11: 0.02,
            12: 0.015,
            13: 0.009,
            14: 0.006,
            15: 0.004,
            16: 0.003,
            17: 0.002,
            18: 0.001,
            19: 0.0,
        }

        variants = module.select_fixed10_controlled_context_variants(
            ref_id=0,
            overlap_by_ref=overlap_by_ref,
            total_size=10,
            core_size=6,
            clean_overlap_threshold=0.20,
            clean_temporal_dedup=1,
        )

        self.assertEqual(set(variants.keys()), {"redundant10", "mixed10", "tail10"})
        self.assertTrue(all(ids[:6] == [0, 1, 2, 3, 4, 5] for ids in variants.values()))
        self.assertTrue(all(len(ids) == 10 for ids in variants.values()))
        self.assertEqual(variants["redundant10"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(variants["mixed10"], [0, 1, 2, 3, 4, 5, 6, 8, 11, 15])
        self.assertEqual(variants["tail10"], [0, 1, 2, 3, 4, 5, 19, 18, 17, 16])

    def test_build_benchmark_records_fixed_eval_prefix(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "scannetpp"
            scene_root = dataset_root / "scene-a"
            (scene_root / "images").mkdir(parents=True)
            (scene_root / "depths").mkdir(parents=True)
            output_root = Path(tmpdir) / "out"

            fake_scene_record = {
                "scene_name": "scene-a",
                "frame_ids": list(range(20)),
                "color_paths": [],
                "depth_paths": [],
                "poses": np.zeros((20, 4, 4), dtype=np.float32),
                "intrinsics": [],
            }
            overlap_by_ref = {
                1: 0.95,
                2: 0.90,
                3: 0.85,
                4: 0.80,
                5: 0.75,
                6: 0.18,
                7: 0.15,
                8: 0.08,
                9: 0.06,
                10: 0.04,
                11: 0.02,
                12: 0.015,
                13: 0.009,
                14: 0.006,
                15: 0.004,
                16: 0.003,
                17: 0.002,
                18: 0.001,
                19: 0.0,
            }

            def fake_materialize(**kwargs):
                tuple_root = output_root / kwargs["tuple_name"]
                tuple_root.mkdir(parents=True, exist_ok=True)
                return tuple_root

            with mock.patch.object(module, "load_scene_record", return_value=fake_scene_record):
                with mock.patch.object(module, "build_scene_overlap_table", return_value={0: overlap_by_ref}):
                    with mock.patch.object(module, "materialize_tuple_sequence", side_effect=fake_materialize):
                        summary_path = module.build_fixed10_controlled_context_benchmark(
                            dataset_root=dataset_root,
                            output_root=output_root,
                            scene_specs=["scene-a"],
                            sample_stride=16,
                            depth_rel_tol=0.01,
                            clean_overlap_threshold=0.20,
                            clean_temporal_dedup=1,
                            anchor_quantiles=[0.1],
                            max_anchors_per_scene=1,
                            image_subdir="images",
                            depth_subdir="depths",
                            camera_subdir="cameras/00",
                            target_num_frames=90,
                            source_prestride=3,
                            source_layout="nested_evc",
                        )

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_tuples"], 3)
            self.assertEqual(payload["config"]["eval_frame_indices"], [0, 1, 2, 3, 4, 5])
            self.assertTrue(all(row["ordered_frame_ids"][:6] == [0, 1, 2, 3, 4, 5] for row in payload["tuples"]))


if __name__ == "__main__":
    unittest.main()
