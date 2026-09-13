import unittest

import torch

from easyvolcap.official_vggt.utils.epipolar_selector import (
    build_downsampled_patch_indices,
    build_depth_reprojection_support_mask,
    build_epipolar_band_mask,
    compute_epipolar_band_loss_from_scores,
)


def _rectified_stereo_extrinsics() -> torch.Tensor:
    extrinsics = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
    extrinsics[:, :, :3, :3] = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3)
    extrinsics[:, 1, 0, 3] = -1.0
    return extrinsics


def _identity_intrinsics() -> torch.Tensor:
    return torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 2, 1, 1)


class EpipolarSelectorUtilsTests(unittest.TestCase):
    def test_epipolar_band_mask_follows_horizontal_line_and_masks_self_view(self):
        band_mask, valid_mask, source_token_indices = build_epipolar_band_mask(
            query_token_indices=torch.tensor([[5]], dtype=torch.long),
            num_views=2,
            tokens_per_view=10,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            band_px=0.75,
            exclude_self_view=True,
        )

        mask = band_mask[0, 0]
        self.assertTrue(valid_mask.dtype == torch.bool)
        self_view_mass = mask[:9].sum()
        source_view_mass = mask[9:].sum()
        same_row_mass = mask[12:15].sum()
        other_rows_mass = mask[9:12].sum() + mask[15:18].sum()

        self.assertEqual(int(self_view_mass.item()), 0)
        self.assertEqual(int(source_view_mass.item()), 3)
        self.assertEqual(int(same_row_mass.item()), 3)
        self.assertEqual(int(other_rows_mass.item()), 0)
        self.assertEqual(source_token_indices.tolist(), list(range(1, 10)) + list(range(11, 20)))

    def test_epipolar_band_mask_can_use_downsampled_source_patches(self):
        source_patch_indices = build_downsampled_patch_indices(
            patch_grid_height=4,
            patch_grid_width=4,
            factor=2,
            device=torch.device("cpu"),
        )
        band_mask, valid_mask, source_token_indices = build_epipolar_band_mask(
            query_token_indices=torch.tensor([[6]], dtype=torch.long),
            num_views=2,
            tokens_per_view=17,
            patch_start_idx=1,
            patch_grid_height=4,
            patch_grid_width=4,
            image_height=8,
            image_width=8,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            band_px=100.0,
            exclude_self_view=True,
            source_patch_indices=source_patch_indices,
        )

        self.assertEqual(source_patch_indices.tolist(), [5, 7, 13, 15])
        self.assertEqual(source_token_indices.tolist(), [6, 8, 14, 16, 23, 25, 31, 33])
        self.assertEqual(tuple(band_mask.shape), (1, 1, 8))
        self.assertEqual(tuple(valid_mask.shape), (1, 1, 8))

    def test_depth_reprojection_support_mask_selects_projected_disk(self):
        depths = torch.zeros(1, 2, 6, 6, dtype=torch.float32)
        depths[:, :, 3, 3] = 2.0
        point_masks = depths > 0

        support_mask, valid_mask, source_token_indices, query_depth_valid = build_depth_reprojection_support_mask(
            query_token_indices=torch.tensor([[5]], dtype=torch.long),
            num_views=2,
            tokens_per_view=10,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            depths=depths,
            point_masks=point_masks,
            radius_px=0.75,
            exclude_self_view=True,
        )

        support = support_mask[0, 0]
        self.assertTrue(bool(query_depth_valid[0, 0]))
        self.assertEqual(source_token_indices.tolist(), list(range(1, 10)) + list(range(11, 20)))
        self.assertEqual(int(support[:9].sum().item()), 0)
        self.assertEqual(int(support[9:].sum().item()), 1)
        self.assertEqual(int(support[13].item()), 1)
        self.assertEqual(int(valid_mask[0, 0, :9].sum().item()), 0)
        self.assertEqual(int(valid_mask[0, 0, 9:].sum().item()), 9)

    def test_epipolar_band_loss_penalizes_outside_band_but_not_in_band_shape(self):
        band_mask, valid_mask, _ = build_epipolar_band_mask(
            query_token_indices=torch.tensor([[5]], dtype=torch.long),
            num_views=2,
            tokens_per_view=10,
            patch_start_idx=1,
            patch_grid_height=3,
            patch_grid_width=3,
            image_height=6,
            image_width=6,
            extrinsics=_rectified_stereo_extrinsics(),
            intrinsics=_identity_intrinsics(),
            band_px=0.75,
            exclude_self_view=True,
        )

        spike_scores = torch.full((1, 1, 18), -6.0, dtype=torch.float32)
        spike_scores[..., 13] = 6.0

        line_scores = torch.full((1, 1, 18), -6.0, dtype=torch.float32)
        line_scores[..., 12:15] = 6.0

        outside_scores = torch.full((1, 1, 18), -6.0, dtype=torch.float32)
        outside_scores[..., 9:12] = 6.0

        spike_loss = compute_epipolar_band_loss_from_scores(spike_scores, band_mask, valid_mask)
        line_loss = compute_epipolar_band_loss_from_scores(line_scores, band_mask, valid_mask)
        outside_loss = compute_epipolar_band_loss_from_scores(outside_scores, band_mask, valid_mask)

        self.assertLess(float(spike_loss), float(outside_loss))
        self.assertLess(float(line_loss), float(outside_loss))
        self.assertAlmostEqual(float(spike_loss), float(line_loss), places=4)


if __name__ == "__main__":
    unittest.main()
