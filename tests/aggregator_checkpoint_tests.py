import unittest
from unittest.mock import patch

import torch
from torch import nn

from easyvolcap.official_vggt.models.aggregator import Aggregator


class _TinyBlock(nn.Module):
    def forward(self, x, pos=None):
        return x + 1.0


class AggregatorCheckpointTests(unittest.TestCase):
    def test_dense_global_checkpoint_can_use_reentrant_memory_mode(self):
        agg = Aggregator.__new__(Aggregator)
        nn.Module.__init__(agg)
        agg.aa_block_size = 1
        agg.global_blocks = nn.ModuleList([_TinyBlock()])
        agg.global_frame_attention_layers = set()
        agg.use_checkpoint = True
        agg.use_reentrant = False
        agg.global_dense_checkpoint_reentrant = True
        agg.reuse_topk_source_by_layer = {}
        agg.reuse_topk_source_layer = None

        calls = []

        def fake_checkpoint(fn, *args, **kwargs):
            calls.append(kwargs)
            return fn(*args)

        tokens = torch.zeros(1, 4, 8, requires_grad=True)
        pos = torch.zeros(1, 4, 2)

        with patch("easyvolcap.official_vggt.models.aggregator.checkpoint", side_effect=fake_checkpoint):
            agg._process_global_attention(tokens, B=1, S=2, P=2, C=8, global_idx=0, pos=pos)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["use_reentrant"], True)


if __name__ == "__main__":
    unittest.main()
