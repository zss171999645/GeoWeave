import unittest

from easyvolcap.runners import volumetric_video_runner as runner_module


class VolumetricVideoRunnerEpochBoundaryTests(unittest.TestCase):
    def test_defer_prefetch_only_at_positive_yield_boundary(self):
        self.assertFalse(runner_module._should_defer_epoch_boundary_prefetch(0, -1))
        self.assertFalse(runner_module._should_defer_epoch_boundary_prefetch(0, 0))
        self.assertFalse(runner_module._should_defer_epoch_boundary_prefetch(0, 2))
        self.assertTrue(runner_module._should_defer_epoch_boundary_prefetch(1, 2))
        self.assertTrue(runner_module._should_defer_epoch_boundary_prefetch(949, 950))


if __name__ == "__main__":
    unittest.main()
