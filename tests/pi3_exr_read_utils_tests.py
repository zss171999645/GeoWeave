import unittest
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "baselines"
    / "exr_read_utils.py"
)


def load_module():
    fake_cv2 = types.SimpleNamespace(
        IMREAD_ANYCOLOR=1,
        IMREAD_ANYDEPTH=2,
        IMREAD_UNCHANGED=4,
        imread=lambda *args, **kwargs: None,
        imdecode=lambda *args, **kwargs: None,
    )
    spec = importlib.util.spec_from_file_location("exr_read_utils_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"cv2": fake_cv2}):
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    return module


class ExrReadUtilsTests(unittest.TestCase):
    def test_read_exr_depth_uses_imdecode_when_imread_fails(self) -> None:
        module = load_module()

        decoded = np.arange(4, dtype=np.float32).reshape(2, 2)
        with mock.patch.object(module.cv2, "imread", return_value=None) as imread_mock, mock.patch.object(
            Path, "read_bytes", return_value=b"fake-exr"
        ) as read_bytes_mock, mock.patch.object(
            module.cv2, "imdecode", return_value=decoded
        ) as imdecode_mock:
            result = module.read_exr_depth("/tmp/fake.exr", retries=0)

        imread_mock.assert_called_once()
        read_bytes_mock.assert_called_once()
        imdecode_mock.assert_called_once()
        self.assertTrue(np.array_equal(result, decoded))

    def test_read_exr_depth_drops_extra_channels(self) -> None:
        module = load_module()

        decoded = np.ones((2, 3, 3), dtype=np.float32)
        with mock.patch.object(module.cv2, "imread", return_value=decoded):
            result = module.read_exr_depth("/tmp/fake.exr", retries=0)

        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
