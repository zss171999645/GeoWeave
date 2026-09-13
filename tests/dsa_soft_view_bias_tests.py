import unittest

import torch
from torch import nn

from easyvolcap.official_vggt.layers import indexer as indexer_mod
from easyvolcap.official_vggt.layers.dsa_attention import DSAAttention
from easyvolcap.official_vggt.layers.indexer import LightningIndexer
from easyvolcap.utils.base_utils import dotdict


class DSASoftViewBiasTests(unittest.TestCase):
    def _build_attention(self, descriptor: str, init_scale: float = 0.7) -> DSAAttention:
        return DSAAttention(
            dim=4,
            num_heads=1,
            indexer_cfg=dotdict(
                enabled=False,
                soft_view_bias_enabled=True,
                soft_view_bias_descriptor=descriptor,
                soft_view_bias_init_scale=init_scale,
            ),
        )

    def test_camera_soft_view_bias_scale_is_fixed_point_one(self):
        attention = self._build_attention("camera", init_scale=0.7)

        self.assertNotIn("soft_view_bias_scale", dict(attention.named_parameters()))
        self.assertIn("soft_view_bias_scale", dict(attention.named_buffers()))
        self.assertTrue(torch.allclose(attention.soft_view_bias_scale, torch.tensor(0.1)))

    def test_image_pool_soft_view_bias_scale_remains_learned(self):
        attention = self._build_attention("image_pool", init_scale=0.7)

        self.assertIsInstance(attention.soft_view_bias_scale, nn.Parameter)
        self.assertIn("soft_view_bias_scale", dict(attention.named_parameters()))
        self.assertAlmostEqual(float(attention.soft_view_bias_scale.detach()), 0.7)

    def test_no_grad_soft_view_bias_topk_uses_blockwise_kernel(self):
        class FakeCudaTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data, data.requires_grad)

            @property
            def is_cuda(self):
                return True

        calls = {"count": 0}
        original_func = indexer_mod.sparse_topk_indexer_func

        def fake_sparse_topk(q, k, w, mask, topk, softmax_scale, block_k, merge_blocks, view_bias_data=None):
            calls["count"] += 1
            self.assertIsNotNone(view_bias_data)
            bsz, tgt_len = q.shape[:2]
            return (
                torch.zeros((bsz, tgt_len, int(topk)), dtype=torch.long),
                torch.zeros((bsz, tgt_len, int(topk)), dtype=q.dtype),
            )

        indexer_mod.sparse_topk_indexer_func = fake_sparse_topk
        try:
            layer = LightningIndexer(
                dim=2,
                n_heads=1,
                head_dim=2,
                use_topk_kernel=True,
                soft_view_bias_training_kernel_enabled=True,
            )
            q = FakeCudaTensor(torch.randn(1, 4, 1, 2))
            k = FakeCudaTensor(torch.randn(1, 4, 1, 2))
            w = FakeCudaTensor(torch.randn(1, 4, 1))
            view_bias_data = {
                "q_view_ids": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
                "s_view_ids": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
                "view_bias": torch.zeros(1, 2, 2),
            }

            with torch.no_grad():
                layer.select_topk_projected(q, k, w, topk=2, return_scores=True, view_bias_data=view_bias_data)
        finally:
            indexer_mod.sparse_topk_indexer_func = original_func

        self.assertEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
