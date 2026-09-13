import unittest

import torch

from easyvolcap.official_vggt.utils import epipolar_selector as epi


class EpipolarSelectorVisualizationTests(unittest.TestCase):
    def test_reshape_epipolar_teacher_probs_to_view_patch_maps_preserves_layout(self):
        probs = torch.arange(12, dtype=torch.float32).view(1, 1, 12)

        maps = epi.reshape_epipolar_teacher_probs_to_view_patch_maps(
            probs,
            num_views=2,
            patch_grid_height=2,
            patch_grid_width=3,
        )

        self.assertEqual(tuple(maps.shape), (1, 1, 2, 2, 3))
        expected_view0 = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        expected_view1 = torch.tensor([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
        self.assertTrue(torch.equal(maps[0, 0, 0], expected_view0))
        self.assertTrue(torch.equal(maps[0, 0, 1], expected_view1))

    def test_pixel_to_patch_token_index_maps_point_into_expected_patch_cell(self):
        token_idx = epi.pixel_to_patch_token_index(
            x=25.0,
            y=15.0,
            image_width=60,
            image_height=40,
            patch_grid_width=3,
            patch_grid_height=2,
            tokens_per_view=11,
            patch_start_idx=5,
            view_idx=0,
        )

        self.assertEqual(token_idx, 6)


if __name__ == "__main__":
    unittest.main()
