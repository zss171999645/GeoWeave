from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "aidi/scripts/vggt/eval_vggt_omega_business_10view_badcase.py"


class VggtOmegaBusinessBadcaseBenchmarkTests(unittest.TestCase):
    def test_runner_defaults_to_frozen_10view_badcase_benchmark(self):
        source = SCRIPT.read_text()
        self.assertIn("vggt_business_10view_badcase_top100_lio.yaml", source)
        self.assertIn("metrics_vggt_omega_business_10view_badcase_top100.json", source)
        self.assertIn("vggt_omega_1b_512.pt", source)
        self.assertIn("sys.path.insert(0, str(REPO_ROOT))", source)
        self.assertIn("log_stage", source)

    def test_runner_uses_geometry_evaluator_and_depth_pose_xyz(self):
        source = SCRIPT.read_text()
        self.assertIn("GeometryEvaluator", source)
        self.assertIn('compute_cam_metrics=["CAM_ACC_AUC"]', source)
        self.assertIn('compute_dpt_metrics=["DPT"]', source)
        self.assertIn('compute_xyz_metrics=["XYZ"]', source)
        self.assertIn("force_xyz_from_depth=True", source)
        self.assertIn("normalize_device", source)
        self.assertIn("batch.meta.iter = processed", source)
        self.assertIn("target + 9", (REPO_ROOT / "aidi/benchmarks/business_10view_badcase_top100/README.md").read_text())


if __name__ == "__main__":
    unittest.main()
