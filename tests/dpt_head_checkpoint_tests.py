import unittest
from unittest.mock import patch

import torch

from easyvolcap.official_vggt.heads import utils as head_utils
from easyvolcap.official_vggt.heads.dpt_head import DPTHead, DPTHeadWithChunkwiseBP
from easyvolcap.utils.base_utils import dotdict


class DPTHeadCheckpointTests(unittest.TestCase):
    def test_scratch_fusion_uses_checkpoint_when_enabled(self):
        head = DPTHead(
            dim_in=8,
            patch_size=14,
            output_dim=2,
            features=8,
            out_channels=[8, 8, 8, 8],
            intermediate_layer_idx=[0, 1, 2, 3],
            use_checkpoint=True,
        )
        tokens = [torch.randn(1, 1, 1, 8, requires_grad=True) for _ in range(4)]
        images = torch.randn(1, 1, 3, 14, 14)

        calls = []

        def fake_checkpoint(fn, *args, **kwargs):
            calls.append(kwargs)
            return fn(*args)

        with patch("easyvolcap.official_vggt.heads.dpt_head.checkpoint", side_effect=fake_checkpoint):
            pred, conf = head._forward_impl(tokens, images, patch_start_idx=0)

        self.assertEqual(pred.shape, (1, 1, 14, 14, 1))
        self.assertEqual(conf.shape, (1, 1, 14, 14))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["use_reentrant"], False)

    def test_scratch_fusion_can_use_reentrant_checkpoint_when_configured(self):
        head = DPTHead(
            dim_in=8,
            patch_size=14,
            output_dim=2,
            features=8,
            out_channels=[8, 8, 8, 8],
            intermediate_layer_idx=[0, 1, 2, 3],
            use_checkpoint=True,
            checkpoint_use_reentrant=True,
        )
        tokens = [torch.randn(1, 1, 1, 8, requires_grad=True) for _ in range(4)]
        images = torch.randn(1, 1, 3, 14, 14)

        calls = []

        def fake_checkpoint(fn, *args, **kwargs):
            calls.append(kwargs)
            return fn(*args)

        with patch("easyvolcap.official_vggt.heads.dpt_head.checkpoint", side_effect=fake_checkpoint):
            pred, conf = head._forward_impl(tokens, images, patch_start_idx=0)

        self.assertEqual(pred.shape, (1, 1, 14, 14, 1))
        self.assertEqual(conf.shape, (1, 1, 14, 14))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["use_reentrant"], True)

    def test_apply_pos_embed_avoids_full_embedding_materialization(self):
        head = DPTHead(
            dim_in=8,
            patch_size=14,
            output_dim=2,
            features=8,
            out_channels=[8, 8, 8, 8],
            intermediate_layer_idx=[0, 1, 2, 3],
        )
        x = torch.randn(1, 8, 14, 14)
        pos_grid = head_utils.create_uv_grid(14, 14, aspect_ratio=1.0, dtype=x.dtype, device=x.device)
        reference = x.clone().add_(
            head_utils.position_grid_to_embed(pos_grid, x.shape[1]).permute(2, 0, 1).unsqueeze(0).mul(0.1)
        )

        with patch(
            "easyvolcap.official_vggt.heads.dpt_head.position_grid_to_embed",
            side_effect=AssertionError("full positional embedding should not be materialized"),
        ):
            actual = head._apply_pos_embed(x.clone(), W=14, H=14)

        self.assertTrue(torch.allclose(actual, reference, atol=1e-6, rtol=1e-6))

    def test_chunkwise_head_can_skip_return_predictions_for_training_memory(self):
        head = DPTHeadWithChunkwiseBP(
            dim_in=8,
            patch_size=2,
            output_dim=2,
            features=8,
            out_channels=[8, 8, 8, 8],
            intermediate_layer_idx=[0, 1, 2, 3],
            chunk_cfg={"frames_chunk_size": 1, "return_predictions": False},
        )
        images = torch.randn(1, 2, 3, 4, 4)
        tokens = [torch.randn(1, 2, 4, 8, requires_grad=True) for _ in range(4)]
        official_batch = {
            "depths": torch.ones(1, 2, 4, 4),
            "point_masks": torch.ones(1, 2, 4, 4, dtype=torch.bool),
        }

        def fake_forward_impl(aggregated_tokens_list, chunk_images, *args, **kwargs):
            chunk = chunk_images.shape[1]
            return torch.ones(1, chunk, 4, 4, 1), torch.ones(1, chunk, 4, 4)

        def fake_compute_chunk_loss(pred, conf, batch_data):
            zero = pred.sum() * 0.0 + conf.sum() * 0.0
            return {
                "loss_conf_depth": zero,
                "loss_reg_depth": zero,
                "loss_grad_depth": zero,
            }

        head._forward_impl = fake_forward_impl
        head._compute_chunk_loss = fake_compute_chunk_loss

        pred, conf = head(tokens, images, patch_start_idx=0, batch=dotdict(), official_batch=official_batch)

        self.assertIsNone(pred)
        self.assertIsNone(conf)
        self.assertIsNotNone(head.last_loss_dict)

    def test_chunkwise_head_can_backward_token_auxiliary_immediately(self):
        head = DPTHeadWithChunkwiseBP(
            dim_in=8,
            patch_size=2,
            output_dim=2,
            features=8,
            out_channels=[8, 8, 8, 8],
            intermediate_layer_idx=[0, 1, 2, 3],
            chunk_cfg={
                "frames_chunk_size": 1,
                "return_predictions": False,
                "backward_auxiliary_immediately": True,
            },
        )
        images = torch.randn(1, 1, 3, 4, 4)
        tokens = [torch.randn(1, 1, 4, 8, requires_grad=True) for _ in range(4)]
        official_batch = {
            "depths": torch.ones(1, 1, 4, 4),
            "point_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        }
        batch = dotdict(loss_scaler=2.0)

        def fake_forward_impl(aggregated_tokens_list, *args, **kwargs):
            base = aggregated_tokens_list[0][:, :, :1, :1].reshape(1, 1, 1, 1, 1)
            return base.expand(1, 1, 4, 4, 1), base.reshape(1, 1, 1, 1).expand(1, 1, 4, 4)

        def fake_compute_chunk_loss(pred, conf, batch_data):
            zero = conf.sum() * 0.0
            return {
                "loss_conf_depth": pred.sum(),
                "loss_reg_depth": zero,
                "loss_grad_depth": zero,
            }

        head._forward_impl = fake_forward_impl
        head._compute_chunk_loss = fake_compute_chunk_loss

        pred, conf = head(tokens, images, patch_start_idx=0, batch=batch, official_batch=official_batch)

        self.assertIsNone(pred)
        self.assertIsNone(conf)
        self.assertIsNotNone(tokens[0].grad)
        grad_after_forward = tokens[0].grad.clone()

        batch.chunkwise_bp_loss.backward()

        self.assertTrue(torch.equal(tokens[0].grad, grad_after_forward))

    def test_chunkwise_head_clears_detached_token_grad_between_frame_chunks(self):
        head = DPTHeadWithChunkwiseBP(
            dim_in=8,
            patch_size=2,
            output_dim=2,
            features=8,
            out_channels=[8, 8, 8, 8],
            intermediate_layer_idx=[0, 1, 2, 3],
            chunk_cfg={"frames_chunk_size": 1, "return_predictions": False},
        )
        images = torch.randn(1, 2, 3, 4, 4)
        tokens = [torch.randn(1, 2, 4, 8, requires_grad=True) for _ in range(4)]
        official_batch = {
            "depths": torch.ones(1, 2, 4, 4),
            "point_masks": torch.ones(1, 2, 4, 4, dtype=torch.bool),
        }
        grad_is_none_at_forward = []

        def fake_forward_impl(aggregated_tokens_list, *args, **kwargs):
            grad_is_none_at_forward.append(aggregated_tokens_list[0].grad is None)
            chunk = aggregated_tokens_list[0].shape[1]
            base = aggregated_tokens_list[0][:, :, :1, :1].reshape(1, chunk, 1, 1, 1)
            return base.expand(1, chunk, 4, 4, 1), base.reshape(1, chunk, 1, 1).expand(1, chunk, 4, 4)

        def fake_compute_chunk_loss(pred, conf, batch_data):
            zero = conf.sum() * 0.0
            return {
                "loss_conf_depth": pred.sum(),
                "loss_reg_depth": zero,
                "loss_grad_depth": zero,
            }

        head._forward_impl = fake_forward_impl
        head._compute_chunk_loss = fake_compute_chunk_loss

        head(tokens, images, patch_start_idx=0, batch=dotdict(), official_batch=official_batch)

        self.assertEqual(grad_is_none_at_forward, [True, True])

    def test_chunkwise_head_can_empty_cuda_cache_before_tight_backward(self):
        head = DPTHeadWithChunkwiseBP(
            dim_in=8,
            patch_size=2,
            output_dim=2,
            features=8,
            out_channels=[8, 8, 8, 8],
            intermediate_layer_idx=[0, 1, 2, 3],
            chunk_cfg={"empty_cache_before_backward_mb": 512},
        )

        class FakeTensor:
            is_cuda = True
            device = torch.device("cuda", 0)

        with patch("torch.cuda.mem_get_info", return_value=(128 * 1024 * 1024, 32 * 1024 * 1024 * 1024)):
            with patch("torch.cuda.empty_cache") as empty_cache:
                did_empty = head._maybe_empty_cache_before_backward(FakeTensor())

        self.assertTrue(did_empty)
        empty_cache.assert_called_once()

    def test_chunkwise_head_can_checkpoint_entire_chunk_forward(self):
        head = DPTHeadWithChunkwiseBP(
            dim_in=8,
            patch_size=2,
            output_dim=2,
            features=8,
            out_channels=[8, 8, 8, 8],
            intermediate_layer_idx=[0, 1, 2, 3],
            chunk_cfg={
                "frames_chunk_size": 1,
                "return_predictions": False,
                "forward_checkpoint": True,
            },
        )
        images = torch.randn(1, 1, 3, 4, 4)
        tokens = [torch.randn(1, 1, 4, 8, requires_grad=True) for _ in range(4)]
        official_batch = {
            "depths": torch.ones(1, 1, 4, 4),
            "point_masks": torch.ones(1, 1, 4, 4, dtype=torch.bool),
        }

        def fake_forward_impl(aggregated_tokens_list, *args, **kwargs):
            base = aggregated_tokens_list[0][:, :, :1, :1].reshape(1, 1, 1, 1, 1)
            return base.expand(1, 1, 4, 4, 1), base.reshape(1, 1, 1, 1).expand(1, 1, 4, 4)

        def fake_compute_chunk_loss(pred, conf, batch_data):
            zero = conf.sum() * 0.0
            return {
                "loss_conf_depth": pred.sum(),
                "loss_reg_depth": zero,
                "loss_grad_depth": zero,
            }

        calls = []

        def fake_checkpoint(fn, *args, **kwargs):
            calls.append(kwargs)
            return fn(*args)

        head._forward_impl = fake_forward_impl
        head._compute_chunk_loss = fake_compute_chunk_loss

        with patch("easyvolcap.official_vggt.heads.dpt_head.checkpoint", side_effect=fake_checkpoint):
            pred, conf = head(tokens, images, patch_start_idx=0, batch=dotdict(), official_batch=official_batch)

        self.assertIsNone(pred)
        self.assertIsNone(conf)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["use_reentrant"], False)


if __name__ == "__main__":
    unittest.main()
