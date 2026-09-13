import unittest

import numpy as np

from easyvolcap.utils.viser_utils import (
    build_camera_labels,
    build_point_cloud_labels,
    compute_valid_depth_ratio,
    flatten_depth_samples,
    flatten_xyz_point_cloud,
    get_meta_value,
    merge_secondary_visualization_data,
    normalize_xyz_point_cloud,
)


class ViserUtilsTests(unittest.TestCase):
    def test_build_labels(self):
        self.assertEqual(
            build_point_cloud_labels("Official", "Ours")["show_primary_xyz"],
            "Official XYZ",
        )
        self.assertEqual(
            build_point_cloud_labels("Official", "Ours")["show_secondary_reprojected"],
            "Ours Reprojected",
        )
        self.assertEqual(
            build_camera_labels("Official", "Ours")["show_secondary_camera"],
            "Show Ours Camera",
        )

    def test_merge_secondary_visualization_data(self):
        primary = {"depth_preds": "primary_depth"}
        secondary = {
            "depth_preds": "secondary_depth",
            "c2ws_pred": "secondary_pose",
            "Ks_pred": "secondary_intrinsics",
            "xyzs": "secondary_xyz",
        }
        merged = merge_secondary_visualization_data(primary, secondary)
        self.assertIs(merged, primary)
        self.assertEqual(merged["depth_preds2"], "secondary_depth")
        self.assertEqual(merged["c2ws_pred2"], "secondary_pose")
        self.assertEqual(merged["Ks_pred2"], "secondary_intrinsics")
        self.assertEqual(merged["xyzs2"], "secondary_xyz")

    def test_normalize_xyz_point_cloud_accepts_hwc_layout(self):
        xyzs = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
        out = normalize_xyz_point_cloud(xyzs, 2, 3, 4)
        self.assertEqual(out.shape, (2, 3, 4, 3))
        np.testing.assert_array_equal(out, xyzs)

    def test_normalize_xyz_point_cloud_accepts_flat_layout(self):
        xyzs = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 12, 3)
        out = normalize_xyz_point_cloud(xyzs, 2, 3, 4)
        self.assertEqual(out.shape, (2, 3, 4, 3))
        np.testing.assert_array_equal(out.reshape(2, 12, 3), xyzs)

    def test_flatten_xyz_point_cloud_uses_grid_downsample(self):
        xyzs = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
        rgbs = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3) / 255.0

        points, colors = flatten_xyz_point_cloud(xyzs, rgbs, downsample=2)

        expected_xyzs = xyzs[::2, ::2].reshape(-1, 3)
        expected_rgbs = rgbs[::2, ::2].reshape(-1, 3)
        np.testing.assert_array_equal(points, expected_xyzs)
        np.testing.assert_array_equal(colors, expected_rgbs)

    def test_flatten_xyz_point_cloud_filters_non_finite_points(self):
        xyzs = np.zeros((4, 4, 3), dtype=np.float32)
        rgbs = np.ones((4, 4, 3), dtype=np.float32)
        xyzs[0, 0] = np.array([np.nan, 0.0, 0.0], dtype=np.float32)
        xyzs[2, 2] = np.array([np.inf, 1.0, 1.0], dtype=np.float32)

        points, colors = flatten_xyz_point_cloud(xyzs, rgbs, downsample=2)

        self.assertEqual(points.shape, (2, 3))
        self.assertEqual(colors.shape, (2, 3))
        self.assertTrue(np.isfinite(points).all())

    def test_flatten_depth_samples_preserves_sparse_valid_pixels(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        depth[1, 1] = 2.0
        depth[1, 3] = 3.0
        depth[3, 1] = 4.0
        rgbs = np.ones((4, 4, 3), dtype=np.float32)

        u, v, z, colors = flatten_depth_samples(depth, rgbs, downsample=2, preserve_valid=True)

        np.testing.assert_array_equal(u, np.array([1.0, 3.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(v, np.array([1.0, 1.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(z, np.array([2.0, 3.0, 4.0], dtype=np.float32))
        self.assertEqual(colors.shape, (3, 3))

    def test_compute_valid_depth_ratio_counts_valid_entries(self):
        depth = np.zeros((2, 4), dtype=np.float32)
        depth[0, 1] = 1.0
        depth[1, 2] = 2.0

        ratio = compute_valid_depth_ratio(depth)

        self.assertAlmostEqual(ratio, 0.25)

    def test_get_meta_value_falls_back_for_missing_key(self):
        meta = {"dataset_name": "eth3d"}
        self.assertEqual(get_meta_value(meta, "dataset_name", "unknown"), "eth3d")
        self.assertEqual(get_meta_value(meta, "scalar_stats", {}), {})


if __name__ == "__main__":
    unittest.main()
