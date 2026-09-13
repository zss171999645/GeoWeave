import unittest

import torch

from easyvolcap.official_vggt.utils.epipolar_selector import augment_indexer_state_with_camera
from easyvolcap.utils.base_utils import dotdict


class EpipolarSelectorStateTests(unittest.TestCase):
    def test_augment_indexer_state_with_camera_adds_camera_tensors_and_image_hw(self):
        state = dotdict(enabled=True, sparse=True)
        official_batch = dict(
            extrinsics=torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(1, 2, 1, 1)[:, :, :3, :4],
            intrinsics=torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
        )

        merged = augment_indexer_state_with_camera(state, official_batch, image_height=6, image_width=8)

        self.assertEqual(tuple(merged.extrinsics.shape), (1, 2, 3, 4))
        self.assertEqual(tuple(merged.intrinsics.shape), (1, 2, 3, 3))
        self.assertEqual(merged.image_height, 6)
        self.assertEqual(merged.image_width, 8)
        self.assertTrue(bool(merged.enabled))


if __name__ == "__main__":
    unittest.main()
