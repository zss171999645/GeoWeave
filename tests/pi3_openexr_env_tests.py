import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Pi3OpenExrEnvTests(unittest.TestCase):
    def check_source_sets_openexr_env_before_cv2_import(self, relative_path: str) -> None:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        env_pos = source.find('os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")')
        cv2_pos = source.find("import cv2")
        self.assertGreaterEqual(env_pos, 0)
        self.assertGreaterEqual(cv2_pos, 0)
        self.assertLess(env_pos, cv2_pos)

    def test_monodepth_source_sets_openexr_env_before_cv2_import(self) -> None:
        self.check_source_sets_openexr_env_before_cv2_import(
            "aidi/scripts/baselines/eval_pi3_monodepth_protocol.py"
        )

    def test_videodepth_source_sets_openexr_env_before_cv2_import(self) -> None:
        self.check_source_sets_openexr_env_before_cv2_import(
            "aidi/scripts/baselines/eval_pi3_videodepth_protocol.py"
        )


if __name__ == "__main__":
    unittest.main()
