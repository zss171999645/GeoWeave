from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PI3_BLOCK_PATH = (
    REPO_ROOT
    / "aidi"
    / "third_party"
    / "pi3_training"
    / "pi3"
    / "models"
    / "layers"
    / "block.py"
)


class Pi3IndexerWarmupLossGuardTests(unittest.TestCase):
    @contextmanager
    def _patched_env(self, **updates):
        import os

        previous = {key: os.environ.get(key) for key in updates}
        os.environ.update({key: str(value) for key, value in updates.items()})
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_lightning_indexer_scratch_gate_starts_positive(self):
        try:
            import torch

            from easyvolcap.official_vggt.layers.indexer import LightningIndexer
        except Exception as exc:
            self.skipTest(f"torch/indexer runtime unavailable: {exc}")

        indexer = LightningIndexer(dim=16, n_heads=2, head_dim=4)

        self.assertTrue(torch.equal(indexer.w_proj.weight, torch.zeros_like(indexer.w_proj.weight)))
        self.assertTrue(torch.equal(indexer.w_proj.bias, torch.ones_like(indexer.w_proj.bias)))

    def test_warmup_zero_fallback_is_forbidden(self):
        source = PI3_BLOCK_PATH.read_text()

        error_idx = source.find("did not produce indexer_loss")
        zero_idx = source.find("indexer_loss = torch.zeros((), device=x.device, dtype=x.dtype)")

        self.assertGreaterEqual(error_idx, 0)
        self.assertGreaterEqual(zero_idx, 0)
        self.assertLess(error_idx, zero_idx)

    def test_warmup_requires_real_indexer_loss(self):
        try:
            import torch
            from torch import nn

            from aidi.third_party.pi3_training.pi3.models.layers.block import DSABlockRope
            from easyvolcap.utils.base_utils import dotdict
        except Exception as exc:
            self.skipTest(f"torch/Pi3 runtime unavailable: {exc}")

        class _NoLossAttention(nn.Module):
            def forward(self, x, pos=None):
                return torch.zeros_like(x), None

        block = DSABlockRope(
            dim=8,
            num_heads=2,
            indexer_cfg={"enabled": False},
        )
        block.attn = _NoLossAttention()
        block.indexer_state = dotdict(
            enabled=True,
            warmup=True,
            compute_loss=True,
            warmup_only_indexer_loss=True,
        )
        block.return_indexer_loss = True

        x = torch.randn(1, 3, 8)

        with self.assertRaisesRegex(RuntimeError, "indexer_loss"):
            block(x)

    def test_dense_warmup_kl_loss_backpropagates_to_indexer(self):
        try:
            import torch

            from easyvolcap.official_vggt.layers.dsa_attention import DSAAttention
            from easyvolcap.utils.base_utils import dotdict
        except Exception as exc:
            self.skipTest(f"torch/DSA runtime unavailable: {exc}")

        attn = DSAAttention(
            dim=8,
            num_heads=2,
            qkv_bias=True,
            proj_bias=True,
            qk_norm=False,
            fused_attn=False,
            indexer_cfg={
                "enabled": True,
                "n_heads": 2,
                "head_dim": 4,
                "score_head_chunk_size": 1,
                "score_key_chunk_size": 4,
            },
        )
        attn.train()
        attn.indexer_state = dotdict(
            enabled=True,
            warmup=True,
            sparse=False,
            compute_loss=True,
            detach_input=True,
            warmup_only_indexer_loss=True,
            warmup_no_grad_attn=True,
            head_chunk_size=1,
            streaming_kl_loss=False,
            warmup_indexer_loss_mode="kl",
            loss_weight=1.0,
            eps=1e-6,
            layer_idx=0,
        )

        x = torch.randn(1, 4, 8)
        _, loss = attn(x)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.detach().abs().item()), 0.0)
        self.assertTrue(loss.requires_grad)

        loss.backward()
        grad_norm = sum(
            float(param.grad.detach().abs().sum().item())
            for param in attn.indexer.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_scratch_dense_warmup_kl_has_useful_signal(self):
        try:
            import torch

            from easyvolcap.official_vggt.layers.dsa_attention import DSAAttention
            from easyvolcap.utils.base_utils import dotdict
        except Exception as exc:
            self.skipTest(f"torch/DSA runtime unavailable: {exc}")

        torch.manual_seed(11)
        attn = DSAAttention(
            dim=16,
            num_heads=4,
            qkv_bias=True,
            proj_bias=True,
            qk_norm=False,
            fused_attn=False,
            indexer_cfg={
                "enabled": True,
                "n_heads": 2,
                "head_dim": 4,
                "score_head_chunk_size": 1,
                "score_key_chunk_size": 4,
            },
        )
        attn.train()
        attn.indexer_state = dotdict(
            enabled=True,
            warmup=True,
            sparse=False,
            compute_loss=True,
            detach_input=True,
            warmup_only_indexer_loss=True,
            warmup_no_grad_attn=True,
            head_chunk_size=1,
            streaming_kl_loss=False,
            warmup_indexer_loss_mode="kl",
            loss_weight=1.0,
            eps=1e-6,
            layer_idx=0,
        )

        x = torch.randn(1, 12, 16)
        _, loss = attn(x)

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.detach().item()), 1e-5)

        loss.backward()
        grad_norm = sum(
            float(param.grad.detach().abs().sum().item())
            for param in attn.indexer.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 1e-5)

    def test_streaming_kl_autograd_handles_mixed_precision_score_inputs(self):
        try:
            import torch

            from easyvolcap.utils.custom_indexer.streaming_kl_autograd import streaming_kl_autograd_loss
        except Exception as exc:
            self.skipTest(f"torch/streaming KL runtime unavailable: {exc}")

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the mixed precision streaming KL regression")

        torch.manual_seed(7)
        device = torch.device("cuda")
        q = torch.randn(1, 8, 2, 4, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(1, 8, 2, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(1, 8, 2, device=device, dtype=torch.bfloat16, requires_grad=True)
        p = torch.rand(1, 8, 8, device=device, dtype=torch.float32)
        p = p / p.sum(dim=-1, keepdim=True)

        with self._patched_env(
            VGGT_STREAMING_KL_SCORE_MODE="legacy",
            VGGT_STREAMING_KL_FWD_MODE="score",
            VGGT_STREAMING_KL_BWD_MODE="score",
        ):
            loss = streaming_kl_autograd_loss(
                q=q,
                k=k,
                w=w,
                p=p,
                mask=None,
                scale=0.5,
                score_head_chunk_size=1,
                score_key_chunk_size=4,
                eps=1e-6,
            )
            self.assertTrue(torch.isfinite(loss).item())

            loss.backward()

        for tensor in (q, k, w):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad.float()).all().item())
            self.assertGreater(float(tensor.grad.float().abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
