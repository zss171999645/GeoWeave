import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DEPTH_SCRIPT = REPO_ROOT / "aidi" / "scripts" / "baselines" / "eval_pi3_videodepth_protocol.py"


class DepthAnything3VideodepthStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = VIDEO_DEPTH_SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_model_family_choices_include_depthanything3(self):
        self.assertIn('choices=["pi3", "vggt", "vggt_omega", "depthanything3"]', self.source)

    def test_depthanything3_cli_options_are_exposed(self):
        for option in ("--da3-repo", "--da3-model", "--process-res", "--process-res-method"):
            self.assertIn(option, self.source)

    def test_depthanything3_inference_and_summary_paths_are_wired(self):
        for marker in (
            "load_depthanything3_videodepth_model",
            "infer_depthanything3_videodepth",
            '"da3_repo"',
            '"da3_model"',
        ):
            self.assertIn(marker, self.source)


if __name__ == "__main__":
    unittest.main()
