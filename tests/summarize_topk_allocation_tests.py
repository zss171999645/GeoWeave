import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "aidi" / "scripts" / "vggt" / "summarize_topk_allocation.py"
MODULE_NAME = "codex_summarize_topk_allocation"


def _load_module():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class SummarizeTopkAllocationTests(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("torch is required for top-k allocation bundle tests")

    def test_query_patch_tokens_skip_special_tokens_and_stride(self):
        module = _load_module()

        tokens = module._query_patch_tokens(
            clean_views=2,
            query_views=2,
            tokens_per_view=5,
            patch_start_idx=2,
            token_stride=2,
        )

        self.assertEqual(tokens.tolist(), [2, 4, 7, 9])

    def test_summarize_bundle_reports_dense_and_topk_noise_share(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Path(tmpdir) / "sample__pt"
            (bundle / "layers").mkdir(parents=True)
            summary = {
                "num_views": 3,
                "tokens_per_view": 4,
                "patch_start_idx": 1,
                "captured_layers": [0],
                "layer_files": {"00": "layers/layer_00.pt"},
                "bundle_meta": {"bundle_id": "sample__pt"},
            }
            (bundle / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            q = torch.zeros((1, 12, 2), dtype=torch.float32)
            k = torch.zeros((1, 12, 2), dtype=torch.float32)
            topk = torch.zeros((1, 12, 2), dtype=torch.int32)
            # Query patch tokens are views 0 and 1, local token 1..3.
            # Four selections go to clean views and two selections go to noise view 2.
            for token in [1, 2, 3, 5, 6, 7]:
                topk[0, token] = torch.tensor([0, 8], dtype=torch.int32)
            torch.save(
                {
                    "mode": "sparse_qk_topk",
                    "q": q,
                    "k": k,
                    "scale": 1.0,
                    "topk_indices": topk,
                    "num_heads": 1,
                },
                bundle / "layers" / "layer_00.pt",
            )

            result = module.summarize_bundle(
                bundle,
                clean_views=2,
                query_views=2,
                query_token_stride=1,
                query_chunk=2,
                layers="all",
            )

        layer = result["layers"][0]
        self.assertAlmostEqual(layer["dense_attention_noise_share"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(layer["topk_count_noise_share"], 0.5, places=6)
        self.assertAlmostEqual(layer["topk_attention_noise_share"], 0.5, places=6)
        self.assertEqual(result["sampled_query_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
