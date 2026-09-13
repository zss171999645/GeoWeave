import unittest
from pathlib import Path


RUN_SH = Path(__file__).resolve().parents[1] / "aidi" / "run.sh"


class AidiRunPythonShimTests(unittest.TestCase):
    def test_dist_shim_uses_captured_python3_after_path_changes(self):
        text = RUN_SH.read_text()

        self.assertIn('export EVC_PYTHON_BIN', text)
        self.assertIn('"${EVC_PYTHON_BIN:-python3}" - "$@"', text)

    def test_triton_checks_use_captured_python(self):
        text = RUN_SH.read_text()
        self.assertIn('if ! "${EVC_PYTHON_BIN}" - <<', text)
        self.assertIn('"${EVC_PYTHON_BIN}" -m pip install triton', text)


if __name__ == "__main__":
    unittest.main()
