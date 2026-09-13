from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "baselines"
    / "eval_pi3_co3d_pose_official.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("eval_pi3_co3d_pose_official", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pi3Co3dPoseEvalTests(unittest.TestCase):
    def test_camera_poses_to_w2c_inverts_c2w(self):
        module = load_module()

        camera_poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        camera_poses[1, :3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        camera_poses[1, :3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        pred_w2c = module.camera_poses_to_w2c(camera_poses)

        expected = np.linalg.inv(camera_poses)[:, :3, :4]
        self.assertEqual(tuple(pred_w2c.shape), (2, 3, 4))
        self.assertTrue(np.allclose(pred_w2c, expected))

    def test_extract_pred_extrinsics_requires_camera_poses(self):
        module = load_module()

        with self.assertRaisesRegex(KeyError, "camera_poses"):
            module.extract_pred_extrinsics({"pose_enc": np.zeros((1,), dtype=np.float32)})

    def test_build_overall_summary_uses_category_macro_mean(self):
        module = load_module()

        per_category_results = {
            "apple": {
                "AUC_30": 0.9,
                "AUC_20": 0.8,
                "AUC_15": 0.7,
                "AUC_10": 0.6,
                "AUC_5": 0.5,
                "AUC_3": 0.4,
                "cam:pose_auc_10_mean": 0.6,
                "cam:pose_auc_20_mean": 0.8,
                "cam:pose_auc_30_mean": 0.9,
            },
            "chair": {
                "AUC_30": 0.3,
                "AUC_20": 0.2,
                "AUC_15": 0.1,
                "AUC_10": 0.05,
                "AUC_5": 0.01,
                "AUC_3": 0.001,
                "cam:pose_auc_10_mean": 0.05,
                "cam:pose_auc_20_mean": 0.2,
                "cam:pose_auc_30_mean": 0.3,
            },
        }

        summary = module.build_overall_summary(per_category_results, num_sequences=7)

        self.assertTrue(np.isclose(summary["AUC_30_mean"], 0.6))
        self.assertTrue(np.isclose(summary["cam:pose_auc_20_mean"], 0.5))
        self.assertEqual(summary["num_categories"], 2)
        self.assertEqual(summary["num_sequences"], 7)

    def test_build_scene_pools_from_manifest_groups_fixed_frames(self):
        module = load_module()

        manifest = [
            {
                "category": "apple",
                "sequence_name": "seq_a",
                "frame_index": 18,
                "filepath": "apple/seq_a/images/frame000034.jpg",
            },
            {
                "category": "apple",
                "sequence_name": "seq_a",
                "frame_index": 171,
                "filepath": "apple/seq_a/images/frame000140.jpg",
            },
            {
                "category": "chair",
                "sequence_name": "seq_b",
                "frame_index": 3,
                "filepath": "chair/seq_b/images/frame000003.jpg",
            },
        ]

        pools = module.build_scene_pools_from_manifest(manifest)

        self.assertEqual(sorted(pools.keys()), ["apple/seq_a", "chair/seq_b"])
        self.assertEqual(pools["apple/seq_a"]["frame_indices"], [18, 171])
        self.assertEqual(
            pools["apple/seq_a"]["filepaths"],
            [
                "apple/seq_a/images/frame000034.jpg",
                "apple/seq_a/images/frame000140.jpg",
            ],
        )


if __name__ == "__main__":
    unittest.main()
