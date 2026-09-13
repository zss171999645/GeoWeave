import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - local machines may not have torch.
    torch = None


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "aidi/scripts/vggt/summarize_view_distance_allocation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("summarize_view_distance_allocation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch is required for allocation summary tests")
class SummarizeViewDistanceAllocationTests(unittest.TestCase):
    def test_distance_metrics_detect_near_allocation(self):
        module = _load_module()
        metrics = module._distance_metrics(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.0, 0.5, 0.5],
            ]
        )
        self.assertAlmostEqual(metrics["near1_share"], 1.0)
        self.assertAlmostEqual(metrics["far_gt2_share"], 0.0)
        self.assertLess(metrics["mean_abs_view_distance"], 0.4)

    def test_summarize_bundle_includes_topk_sources(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp)
            layer_dir = bundle_dir / "layers"
            layer_dir.mkdir()
            summary = {
                "num_views": 3,
                "tokens_per_view": 3,
                "patch_start_idx": 0,
                "captured_layers": [0],
                "layer_files": {"00": "layers/layer_00.pt"},
                "bundle_meta": {"bundle_id": "toy_seq"},
            }
            (bundle_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            total_tokens = 9
            topk_indices = torch.zeros((1, total_tokens, 2), dtype=torch.long)
            for token_idx in range(total_tokens):
                view = token_idx // 3
                topk_indices[0, token_idx, 0] = view * 3
                topk_indices[0, token_idx, 1] = min(view + 1, 2) * 3
            torch.save(
                {
                    "q": torch.randn(1, total_tokens, 4),
                    "k": torch.randn(1, total_tokens, 4),
                    "scale": 1.0,
                    "mode": "sparse_qk_topk",
                    "topk_indices": topk_indices,
                },
                layer_dir / "layer_00.pt",
            )
            result = module.summarize_bundle(
                bundle_dir,
                query_views_text="all",
                query_token_stride=3,
                query_chunk=2,
                layers="0",
            )
        sources = result["layers"][0]["sources"]
        self.assertEqual([row["source"] for row in sources], ["dense_attention", "topk_count", "topk_attention"])
        self.assertGreaterEqual(sources[1]["metrics"]["near1_share"], 0.8)


if __name__ == "__main__":
    unittest.main()
