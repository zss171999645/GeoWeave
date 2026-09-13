import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DEPTH_SCRIPT = REPO_ROOT / "aidi" / "scripts" / "baselines" / "eval_pi3_videodepth_protocol.py"


class VggtOmegaVideodepthStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = VIDEO_DEPTH_SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_model_family_choices_include_vggt_omega(self):
        for model_family in ('"pi3"', '"vggt"', '"vggt_omega"', '"depthanything3"'):
            self.assertIn(model_family, self.source)

    def test_vggt_omega_cli_options_are_exposed(self):
        for option in (
            "--vggt-omega-repo",
            "--vggt-omega-checkpoint",
            "--vggt-omega-resolution",
            "--vggt-omega-mode",
            "--no-vggt-omega-crop-gt-to-input-fov",
            "--eval-fov-policy",
            "--scale-shift-fit-max-pixels",
            "--vggt-omega-query-view-index",
            "--vggt-omega-query-view-sweep",
        ):
            self.assertIn(option, self.source)

    def test_vggt_omega_inference_and_summary_paths_are_wired(self):
        self.assertIn("load_vggt_omega_videodepth_model", self.source)
        self.assertIn("infer_vggt_omega_videodepth", self.source)
        self.assertIn('"vggt_omega_checkpoint"', self.source)
        self.assertIn('"vggt_omega_repo"', self.source)
        self.assertIn('"vggt_omega_crop_gt_to_input_fov"', self.source)
        self.assertIn('"eval_fov_policy"', self.source)
        self.assertIn('"eval_fov_policy_effective"', self.source)
        self.assertIn("crop_vggt_omega_gt_to_input_fov", self.source)
        self.assertIn("crop_depth_sequence_with_crop_info", self.source)
        self.assertIn("resize_predictions_for_eval_fov", self.source)
        self.assertIn("scale_shift_fit_max_pixels", self.source)

    def test_eth3d_pi3_style_layout_is_supported_for_videodepth(self):
        self.assertIn("is_eth3d_pi3_style_layout", self.source)
        self.assertIn("ground_truth_depth", self.source)
        self.assertIn("custom_undistorted", self.source)

    def test_query_view_sweep_metrics_are_recorded_for_omega(self):
        self.assertIn("query_view_image_global", self.source)
        self.assertIn("query_view_depth_metrics", self.source)
        self.assertIn("vggt_omega_query_view_sweep", self.source)


if __name__ == "__main__":
    unittest.main()
