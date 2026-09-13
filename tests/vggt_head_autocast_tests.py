import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from easyvolcap.official_vggt.models.vggt import VGGT


class _FakeAggregator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, images):
        token = torch.zeros(images.shape[0], images.shape[1], 1, 8, device=images.device)
        return [token], 0


class _FakeDepthHead(nn.Module):
    def forward(self, *args, **kwargs):
        images = kwargs["images"]
        B, S, _, H, W = images.shape
        return images.new_zeros(B, S, H, W, 1), images.new_zeros(B, S, H, W)


class VGGTHeadAutocastTests(unittest.TestCase):
    def test_head_autocast_can_be_enabled_from_config(self):
        calls = []

        class FakeAutocast:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch("easyvolcap.official_vggt.models.vggt.Aggregator", _FakeAggregator):
            model = VGGT(
                embed_dim=4,
                enable_camera=False,
                enable_point=False,
                enable_depth=False,
                enable_track=False,
                head_autocast_enabled=True,
                head_autocast_dtype="float16",
            )
        model.depth_head = _FakeDepthHead()

        with patch("easyvolcap.official_vggt.models.vggt.torch.cuda.amp.autocast", FakeAutocast):
            model(torch.zeros(1, 1, 3, 14, 14))

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["enabled"], True)
        self.assertIs(calls[0]["dtype"], torch.float16)


if __name__ == "__main__":
    unittest.main()
