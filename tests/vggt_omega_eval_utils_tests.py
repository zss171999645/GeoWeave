import unittest

import numpy as np

from aidi.scripts.vggt.vggt_omega_eval_utils import (
    as_homogeneous_w2c,
    camera_register_query_positions,
    configure_vggt_omega_attention_experiment,
    depth_to_world_points,
    query_view_image_global_query_positions,
)


class VggtOmegaEvalUtilsTests(unittest.TestCase):
    def test_camera_register_query_positions_are_interleaved_by_frame(self):
        self.assertEqual(
            camera_register_query_positions(num_frames=3, num_tokens=5, patch_token_start=2),
            [0, 1, 5, 6, 10, 11],
        )

    def test_query_view_image_global_positions_include_one_views_image_tokens(self):
        self.assertEqual(
            query_view_image_global_query_positions(
                num_frames=3,
                num_tokens=5,
                patch_token_start=2,
                query_view_index=1,
            ),
            [0, 1, 5, 6, 7, 8, 9, 10, 11],
        )

    def test_configure_camera_register_query_delegates_non_global_blocks(self):
        class FakeAggregator:
            patch_token_start = 2

            def _run_inter_frame_attention_block(
                self,
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                attention_type,
            ):
                return {
                    "tokens": tokens,
                    "batch_size": batch_size,
                    "num_frames": num_frames,
                    "num_tokens": num_tokens,
                    "embed_dim": embed_dim,
                    "block_idx": block_idx,
                    "attention_type": attention_type,
                }

        class FakeModel:
            aggregator = FakeAggregator()

        model = FakeModel()

        mode = configure_vggt_omega_attention_experiment(model, global_attention_mode="camera_register_query")
        result = model.aggregator._run_inter_frame_attention_block("x", 1, 2, 3, 4, 5, "register")

        self.assertEqual(mode, "camera_register_query")
        self.assertEqual(model.aggregator._meshx_global_attention_mode, "camera_register_query")
        self.assertEqual(result["tokens"], "x")
        self.assertEqual(result["attention_type"], "register")

    def test_as_homogeneous_w2c_adds_last_row(self):
        extrinsics = np.array(
            [[[[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 3.0], [0.0, 0.0, 1.0, 4.0]]]],
            dtype=np.float32,
        )

        hom = as_homogeneous_w2c(extrinsics)

        self.assertEqual(hom.shape, (1, 4, 4))
        np.testing.assert_allclose(hom[0, :3, :4], extrinsics[0, 0])
        np.testing.assert_allclose(hom[0, 3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    def test_depth_to_world_points_inverts_w2c_extrinsics(self):
        depth = np.ones((1, 1, 2, 2, 1), dtype=np.float32)
        extrinsics = np.array(
            [[[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]]]],
            dtype=np.float32,
        )
        intrinsics = np.array([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]], dtype=np.float32)

        points = depth_to_world_points(depth, extrinsics, intrinsics)

        expected = np.array(
            [[[[-1.0, -2.0, -2.0], [0.0, -2.0, -2.0]], [[-1.0, -1.0, -2.0], [0.0, -1.0, -2.0]]]],
            dtype=np.float32,
        )
        self.assertEqual(points.shape, (1, 2, 2, 3))
        np.testing.assert_allclose(points, expected)

    def test_loader_uses_meta_assignment_and_debug_logging(self):
        source = (
            __import__("pathlib")
            .Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("aidi/scripts/vggt/vggt_omega_eval_utils.py")
            .read_text()
        )

        self.assertIn('torch.device("meta")', source)
        self.assertIn("assign=True", source)
        self.assertIn("VGGT_OMEGA_EVAL_DEBUG", source)


if __name__ == "__main__":
    unittest.main()
