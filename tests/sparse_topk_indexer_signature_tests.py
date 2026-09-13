import unittest
from unittest import mock

import torch

import easyvolcap.utils.custom_indexer.sparse_topk_indexer as mod


class SparseTopkIndexerSignatureTests(unittest.TestCase):
    def test_autograd_path_explicitly_passes_none_view_bias_data(self):
        q = torch.randn(1, 1, 1, 2)
        k = torch.randn(1, 1, 1, 2)
        w = torch.randn(1, 1, 1)
        captured = {}

        def fake_blockwise_topk_scores(**kwargs):
            captured.update(kwargs)
            return (
                torch.zeros((1, 1, 1), dtype=torch.int32),
                torch.zeros((1, 1, 1), dtype=q.dtype),
            )

        with mock.patch.object(mod, '_blockwise_topk_scores', side_effect=fake_blockwise_topk_scores):
            topk_indices, topk_scores = mod.SparseTopkIndexerFunc.apply(q, k, w, None, 1, None, 16, 0)

        self.assertIn('view_bias_data', captured)
        self.assertIsNone(captured['view_bias_data'])
        self.assertEqual(tuple(topk_indices.shape), (1, 1, 1))
        self.assertEqual(tuple(topk_scores.shape), (1, 1, 1))


if __name__ == '__main__':
    unittest.main()
