from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "baselines"
    / "eval_pi3_re10k_pose_official.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing evaluator script: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("eval_pi3_re10k_pose_official", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pi3Re10kPoseEvalTests(unittest.TestCase):
    def test_module_import_bootstraps_repo_root_for_runtime_imports(self):
        module_root = str(MODULE_PATH.parents[3])
        original_path = list(sys.path)
        try:
            sys.path[:] = [entry for entry in sys.path if entry != module_root]
            module = load_module()
            self.assertIn(module_root, sys.path)
            self.assertEqual(module.repo_root(), MODULE_PATH.parents[3])
        finally:
            sys.path[:] = original_path

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

    def test_stable_scene_seed_matches_aligned_protocol(self):
        module = load_module()

        scene_key = "test/000c3ab189999a83"
        seed = 20260215
        payload = f"{seed}:{scene_key}".encode("utf-8", errors="ignore")
        expected = int.from_bytes(hashlib.sha1(payload).digest()[:8], byteorder="little", signed=False)

        actual = module.stable_scene_seed(scene_key=scene_key, seed=seed)

        self.assertEqual(actual, expected)

    def test_sample_scene_pool_indices_keeps_rng_sample_order(self):
        module = load_module()

        scene_key = "test/000c3ab189999a83"
        seed = 20260215
        n_frames = 300
        pool_size = 10
        rng = random.Random(module.stable_scene_seed(scene_key=scene_key, seed=seed))
        expected = rng.sample(range(n_frames), pool_size)

        actual = module.sample_scene_pool_indices(
            scene_key=scene_key,
            total_frames=n_frames,
            pool_size=pool_size,
            seed=seed,
        )

        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), pool_size)
        self.assertNotEqual(actual, sorted(actual))

    def test_select_scene_input_indices_can_use_staged_scene_order(self):
        module = load_module()

        ordered = module.select_scene_input_indices(
            scene_key="test/tuple_scene",
            total_frames=8,
            seed=20260215,
            pool_size=0,
            n_srcs=999,
            use_scene_order=True,
        )
        capped = module.select_scene_input_indices(
            scene_key="test/tuple_scene",
            total_frames=8,
            seed=20260215,
            pool_size=5,
            n_srcs=999,
            use_scene_order=True,
        )

        self.assertEqual(ordered, list(range(8)))
        self.assertEqual(capped, list(range(5)))

    def test_setup_args_accepts_staged_tuple_eval_flags(self):
        module = load_module()

        args = module.setup_args(
            [
                "--re10k-root",
                "/tmp/re10k",
                "--use-scene-order",
                "--pool-size",
                "0",
                "--eval-frame-indices",
                "0,1,2,3,4,5",
                "--translation-error-mode",
                "centers",
                "--auc-combine",
                "min",
            ]
        )

        self.assertTrue(args.use_scene_order)
        self.assertEqual(args.pool_size, 0)
        self.assertEqual(args.eval_frame_indices, "0,1,2,3,4,5")
        self.assertEqual(args.translation_error_mode, "centers")
        self.assertEqual(args.auc_combine, "min")

    def test_parse_and_subset_eval_frame_indices(self):
        module = load_module()

        poses = np.arange(8 * 3 * 4, dtype=np.float64).reshape(8, 3, 4)
        indices = module.parse_eval_frame_indices("0, 2,5")
        subset = module.subset_pose_array_for_eval(poses, indices)

        self.assertEqual(indices, [0, 2, 5])
        self.assertTrue(np.array_equal(subset, poses[[0, 2, 5]]))

    def test_module_provides_local_pi3_helpers(self):
        module = load_module()

        self.assertTrue(callable(getattr(module, "load_images_for_pi3", None)))
        self.assertTrue(callable(getattr(module, "load_pi3_model", None)))

    def test_build_summary_exports_re10k_auc_fields(self):
        module = load_module()

        metrics = [
            {"cam:pose_auc_10": 0.7, "cam:pose_auc_20": 0.8, "cam:pose_auc_30": 0.9},
            {"cam:pose_auc_10": 0.5, "cam:pose_auc_20": 0.6, "cam:pose_auc_30": 0.7},
        ]

        summary = module.build_summary(metrics)

        self.assertTrue(np.isclose(summary["cam:pose_auc_10_mean"], 0.6))
        self.assertTrue(np.isclose(summary["cam:pose_auc_20_mean"], 0.7))
        self.assertTrue(np.isclose(summary["cam:pose_auc_30_mean"], 0.8))

    def test_resolve_scene_roots_falls_back_when_data_roots_unreadable(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "0001").mkdir()
            (root / "0002").mkdir()
            (root / "data_roots.txt").write_text("0001\n0002\n")

            original_read_text = Path.read_text

            def _raise_for_data_roots(path_self, *args, **kwargs):
                if path_self == root / "data_roots.txt":
                    raise PermissionError("permission denied")
                return original_read_text(path_self, *args, **kwargs)

            with mock.patch("pathlib.Path.read_text", autospec=True, side_effect=_raise_for_data_roots):
                scene_roots = module.resolve_scene_roots(root)

        self.assertEqual([path.name for path in scene_roots], ["0001", "0002"])

    def test_materialize_camera_files_copies_yaml_locally(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            camera_root = tmp_root / "scene_a" / "cameras" / "00"
            camera_root.mkdir(parents=True)
            (camera_root / "intri.yml").write_text("intri")
            (camera_root / "extri.yml").write_text("extri")
            cache_root = tmp_root / "cache"

            local_root = module.materialize_camera_files(camera_root, cache_root=cache_root)

            self.assertTrue((local_root / "intri.yml").is_file())
            self.assertTrue((local_root / "extri.yml").is_file())
            self.assertEqual((local_root / "intri.yml").read_text(), "intri")
            self.assertEqual((local_root / "extri.yml").read_text(), "extri")


if __name__ == "__main__":
    unittest.main()
