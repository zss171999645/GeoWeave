from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "aidi/scripts/vggt/eval_external_multiview_benchmark.py"


class ExternalMultiviewBenchmarkTests(unittest.TestCase):
    def test_runner_supports_public_driving_external_models(self):
        source = SCRIPT.read_text()

        self.assertIn('choices=("vggt_omega", "depthanything3", "pi3")', source)
        self.assertIn("--config", source)
        self.assertIn("image_files_from_batch", source)
        self.assertIn("GeometryEvaluator", source)
        self.assertIn("evaluator_kwargs", source)
        self.assertIn("compute_cam_metrics", source)
        self.assertIn("compute_dpt_metrics", source)
        self.assertIn("compute_xyz_metrics", source)

    def test_runner_uses_config_metrics_instead_of_badcase_defaults(self):
        source = SCRIPT.read_text()

        self.assertNotIn('compute_cam_metrics=["CAM_ACC_AUC"]', source)
        self.assertNotIn('compute_dpt_metrics=["DPT"]', source)
        self.assertNotIn('compute_xyz_metrics=["XYZ"]', source)
        self.assertIn("target_hw_from_batch", source)
        self.assertIn("metrics_file or ev_cfg.get", source)
        self.assertIn("batch.meta.iter = processed", source)

    def test_runner_resolves_camera_directory_names_from_dataset_layout(self):
        source = SCRIPT.read_text()

        self.assertIn("_camera_dir_candidates", source)
        self.assertIn("images_root.iterdir()", source)
        self.assertIn('view_index = _safe_meta_int(batch.meta, "view_index", camera)', source)

    def test_kitti_external_camera_basis_maps_opencv_forward_to_evc_forward(self):
        from aidi.scripts.vggt.eval_external_multiview_benchmark import apply_external_camera_basis_to_extrinsics

        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :3], 2, axis=0)
        extrinsics[1, :3, 3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        converted = apply_external_camera_basis_to_extrinsics(extrinsics, "kitti_odometry_evc")

        np.testing.assert_allclose(converted[1, :3, 3], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.det(converted[0, :3, :3])), 1.0, places=6)

    def test_auto_camera_basis_detects_kitti_odometry_evc_layout(self):
        from aidi.scripts.vggt.eval_external_multiview_benchmark import resolve_external_camera_basis

        args = SimpleNamespace(external_camera_basis="auto")
        batch = SimpleNamespace(
            meta=SimpleNamespace(
                data_root=[
                    "/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/tao02.xie/datasets/kitti/odometry/00"
                ]
            )
        )

        self.assertEqual(resolve_external_camera_basis(args, batch), "kitti_odometry_evc")

    def test_pi3_output_applies_external_camera_basis(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in the local lightweight test environment")
        from aidi.scripts.vggt.eval_external_multiview_benchmark import pi3_output_to_evc

        class DotDict(dict):
            def __getattr__(self, name):
                return self[name]

        def dotdict(**kwargs):
            return DotDict(kwargs)

        def capture_pose_encoding(extrinsics, intrinsics, image_size_hw):
            return extrinsics[..., :9].clone()

        batch = SimpleNamespace(
            msk=torch.ones(1, 2, 4, 1),
            dpt=torch.ones(1, 2, 4, 1),
            meta=SimpleNamespace(H=2, W=2),
        )
        camera_poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
        # Pi3 outputs c2w. This makes the second view translate +Z in camera/OpenCV world;
        # after c2w->w2c inversion the w2c translation is -Z, and KITTI EVC basis maps it to -X.
        camera_poses[0, 1, 2, 3] = 1.0
        prediction = {
            "local_points": torch.ones(1, 2, 2, 2, 3),
            "camera_poses": camera_poses,
        }

        output = pi3_output_to_evc(
            prediction=prediction,
            batch=batch,
            device=torch.device("cpu"),
            extri_intri_to_pose_encoding=capture_pose_encoding,
            dotdict=dotdict,
            use_native_xyz=False,
            camera_basis="kitti_odometry_evc",
        )

        self.assertEqual(tuple(output.cam_map.shape), (1, 2, 3, 4))
        np.testing.assert_allclose(output.cam_map[0, 1, :3, 3].numpy(), np.array([-1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
