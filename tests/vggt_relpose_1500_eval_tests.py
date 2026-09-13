from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "vggt"
    / "eval_relpose_1500_benchmark.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing evaluator script: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("eval_relpose_1500_benchmark", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VggtRelpose1500EvalTests(unittest.TestCase):
    def test_module_import_bootstraps_repo_root(self):
        module_root = str(MODULE_PATH.parents[3])
        original_path = list(sys.path)
        try:
            sys.path[:] = [entry for entry in sys.path if entry != module_root]
            module = load_module()
            self.assertIn(module_root, sys.path)
            self.assertEqual(module.repo_root(), MODULE_PATH.parents[3])
        finally:
            sys.path[:] = original_path

    def test_compute_pair_pose_metrics_returns_one_for_perfect_predictions(self):
        module = load_module()

        gt = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        gt[1, :3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        metrics = module.compute_pair_pose_metrics(gt_c2w=gt, pred_c2w=gt.copy())

        self.assertAlmostEqual(metrics["pose_auc_05"], 1.0)
        self.assertAlmostEqual(metrics["pose_auc_10"], 1.0)
        self.assertAlmostEqual(metrics["pose_auc_20"], 1.0)

    def test_compute_pair_pose_metrics_is_invariant_to_global_gauge_rotation(self):
        module = load_module()

        gt = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        gt[1, :3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        rot_z_90 = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pred = gt.copy()
        pred[:, :3, :3] = rot_z_90
        pred[0, :3, 3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pred[1, :3, 3] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        metrics = module.compute_pair_pose_metrics(gt_c2w=gt, pred_c2w=pred)

        self.assertAlmostEqual(metrics["rotation_error_deg"], 0.0, places=5)
        self.assertAlmostEqual(metrics["translation_error_deg"], 0.0, places=5)
        self.assertAlmostEqual(metrics["pose_auc_05"], 1.0, places=5)
        self.assertAlmostEqual(metrics["pose_auc_10"], 1.0, places=5)
        self.assertAlmostEqual(metrics["pose_auc_20"], 1.0, places=5)

    def test_parse_megadepth_pair_infos_accepts_loftr_tuple_format(self):
        module = load_module()

        raw = np.asarray([((0, 2), 0.3, None), ((1, 3), 0.5, None)], dtype=object)

        parsed = module.parse_megadepth_pair_infos(raw)

        self.assertEqual(parsed, [(0, 2), (1, 3)])

    def test_parse_megadepth_pair_infos_accepts_dense_int_matrix(self):
        module = load_module()

        raw = np.asarray([[0, 2], [1, 3]], dtype=np.int32)

        parsed = module.parse_megadepth_pair_infos(raw)

        self.assertEqual(parsed, [(0, 2), (1, 3)])

    def test_load_megadepth_pairs_reads_scene_info_manifest(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "megadepth"
            manifest_root = root / "manifest"
            dataset_root.mkdir()
            manifest_root.mkdir()

            (dataset_root / "scene" / "a").mkdir(parents=True)
            (dataset_root / "scene" / "b").mkdir(parents=True)
            (dataset_root / "scene" / "a" / "img0.jpg").write_bytes(b"jpg")
            (dataset_root / "scene" / "b" / "img1.jpg").write_bytes(b"jpg")
            (manifest_root / "megadepth_test_1500.txt").write_text("sample_scene\n", encoding="utf-8")

            poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
            poses[1, 0, 3] = 1.0
            np.savez(
                manifest_root / "sample_scene.npz",
                image_paths=np.asarray(["scene/a/img0.jpg", "scene/b/img1.jpg"], dtype=object),
                intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
                poses=poses,
                pair_infos=np.asarray([((0, 1), 0.3, None)], dtype=object),
            )

            pairs = module.load_megadepth_pairs(manifest_root=manifest_root, dataset_root=dataset_root)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(Path(pairs[0].image_paths[0]).name, "img0.jpg")
        self.assertEqual(Path(pairs[0].image_paths[1]).name, "img1.jpg")
        self.assertEqual(tuple(pairs[0].gt_c2w.shape), (2, 4, 4))

    def test_load_scannet_pairs_reads_loftr_test_manifest(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "scannet"
            manifest_root = root / "manifest"
            scene_root = dataset_root / "scene0707_00"
            (scene_root / "color").mkdir(parents=True)
            (scene_root / "pose").mkdir(parents=True)
            (scene_root / "intrinsic").mkdir(parents=True)
            manifest_root.mkdir()

            (scene_root / "color" / "1.jpg").write_bytes(b"jpg")
            (scene_root / "color" / "3.jpg").write_bytes(b"jpg")
            np.savetxt(scene_root / "pose" / "1.txt", np.eye(4, dtype=np.float32))
            pose3 = np.eye(4, dtype=np.float32)
            pose3[0, 3] = 2.0
            np.savetxt(scene_root / "pose" / "3.txt", pose3)
            np.savetxt(scene_root / "intrinsic" / "intrinsic_color.txt", np.eye(4, dtype=np.float32))

            (manifest_root / "scannet_test.txt").write_text("test.npz\n", encoding="utf-8")
            np.savez(manifest_root / "test.npz", name=np.asarray([[707, 0, 1, 3]], dtype=np.int32))

            pairs = module.load_scannet_pairs(manifest_root=manifest_root, dataset_root=dataset_root)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].scene_name, "scene0707_00")
        self.assertEqual(Path(pairs[0].image_paths[0]).name, "1.jpg")
        self.assertEqual(Path(pairs[0].image_paths[1]).name, "3.jpg")
        self.assertEqual(tuple(pairs[0].gt_c2w.shape), (2, 4, 4))

    def test_call_runtime_infer_cameras_c2w_accepts_legacy_runtime_signature(self):
        module = load_module()

        calls = {}

        def infer_cameras_c2w(*, args, image_names, model, device, load_img_size, image_load_retries, image_load_retry_sleep):
            calls.update(
                args=args,
                image_names=image_names,
                model=model,
                device=device,
                load_img_size=load_img_size,
                image_load_retries=image_load_retries,
                image_load_retry_sleep=image_load_retry_sleep,
            )
            return "ok"

        runtime = SimpleNamespace(infer_cameras_c2w=infer_cameras_c2w)
        args = SimpleNamespace(
            load_img_size=640,
            image_load_retries=3,
            image_load_retry_sleep=1.5,
            verbose=True,
        )

        result = module.call_runtime_infer_cameras_c2w(
            runtime=runtime,
            args=args,
            image_names=["a.jpg", "b.jpg"],
            model="model",
            device="cpu",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(calls["image_names"], ["a.jpg", "b.jpg"])
        self.assertEqual(calls["load_img_size"], 640)
        self.assertEqual(calls["image_load_retries"], 3)
        self.assertEqual(calls["image_load_retry_sleep"], 1.5)
        self.assertNotIn("verbose", calls)

    def test_call_runtime_infer_cameras_c2w_passes_verbose_when_supported(self):
        module = load_module()

        calls = {}

        def infer_cameras_c2w(*, args, image_names, model, device, verbose):
            calls.update(
                args=args,
                image_names=image_names,
                model=model,
                device=device,
                verbose=verbose,
            )
            return "ok"

        runtime = SimpleNamespace(infer_cameras_c2w=infer_cameras_c2w)
        args = SimpleNamespace(
            load_img_size=640,
            image_load_retries=3,
            image_load_retry_sleep=1.5,
            verbose=True,
        )

        result = module.call_runtime_infer_cameras_c2w(
            runtime=runtime,
            args=args,
            image_names=["a.jpg", "b.jpg"],
            model="model",
            device="cpu",
        )

        self.assertEqual(result, "ok")
        self.assertTrue(calls["verbose"])


if __name__ == "__main__":
    unittest.main()
