import ast
import unittest
from pathlib import Path

import numpy as np


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "easyvolcap"
    / "dataloaders"
    / "datasets"
    / "volumetric_video_dataset.py"
)


def _load_normalizer():
    tree = ast.parse(DATASET_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize_camera_distortion":
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            ns = {"np": np}
            exec(compile(module, str(DATASET_PATH), "exec"), ns)
            return ns[node.name]
    raise AssertionError("_normalize_camera_distortion not found")


class VolumetricVideoDistortionShapeTests(unittest.TestCase):
    def test_normalizes_mixed_distortion_lengths_to_common_dim(self):
        normalize = _load_normalizer()

        self.assertEqual(normalize(np.arange(4), 8).shape, (8, 1))
        np.testing.assert_array_equal(normalize(np.arange(4), 8).reshape(-1)[:4], np.arange(4))
        np.testing.assert_array_equal(normalize(np.arange(4), 8).reshape(-1)[4:], np.zeros(4))

        self.assertEqual(normalize(np.arange(8), 8).shape, (8, 1))
        np.testing.assert_array_equal(normalize(np.arange(8), 8).reshape(-1), np.arange(8))

    def test_defaults_to_at_least_five_coefficients(self):
        normalize = _load_normalizer()

        self.assertEqual(normalize(None).shape, (5, 1))
        self.assertEqual(normalize(np.arange(4)).shape, (5, 1))


if __name__ == "__main__":
    unittest.main()
