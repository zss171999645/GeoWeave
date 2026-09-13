import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "aidi" / "scripts" / "baselines" / "build_waymo_plausible_context_diagnostic.py"
SIMPLE_BUILDER_PATH = ROOT / "aidi" / "scripts" / "baselines" / "build_waymo_simple_plausible_context_benchmark.py"
PREPARE_PATH = ROOT / "aidi" / "scripts" / "vggt" / "prepare_pi3_attention_seq_bundle.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WaymoPlausibleContextDiagnosticTests(unittest.TestCase):
    def test_interleaved_midpoint_clean_context_is_inside_eval_span(self):
        module = _load_module(SIMPLE_BUILDER_PATH, "codex_waymo_simple_plausible_builder")
        frames = [f"{idx:06d}" for idx in range(40)]

        prefix, context = module.select_prefix_and_clean_context(
            frames,
            start=4,
            total_views=10,
            eval_views=6,
            context_views=4,
            frame_stride=4,
            clean_context_mode="interleaved_midpoint",
        )

        self.assertEqual(prefix, ["000004", "000008", "000012", "000016", "000020", "000024"])
        self.assertEqual(context, ["000006", "000010", "000014", "000018"])

    def test_chronological_clean_context_keeps_legacy_tail(self):
        module = _load_module(SIMPLE_BUILDER_PATH, "codex_waymo_simple_plausible_builder_legacy")
        frames = [f"{idx:06d}" for idx in range(50)]

        prefix, context = module.select_prefix_and_clean_context(
            frames,
            start=4,
            total_views=10,
            eval_views=6,
            context_views=4,
            frame_stride=4,
            clean_context_mode="chronological_tail",
        )

        self.assertEqual(prefix, ["000004", "000008", "000012", "000016", "000020", "000024"])
        self.assertEqual(context, ["000028", "000032", "000036", "000040"])

    def test_rank_plausible_candidates_prefers_official_minus_sparse_gap(self):
        module = _load_module(BUILDER_PATH, "codex_waymo_plausible_builder")
        candidates = [
            {
                "sample_id": "a",
                "selection_metrics": {
                    "official_ATE_delta": 2.0,
                    "ATE_delta_gap_official_minus_sparse": 0.5,
                },
            },
            {
                "sample_id": "b",
                "selection_metrics": {
                    "official_ATE_delta": 1.0,
                    "ATE_delta_gap_official_minus_sparse": 3.0,
                },
            },
            {
                "sample_id": "c",
                "selection_metrics": {
                    "official_ATE_delta": -0.1,
                    "ATE_delta_gap_official_minus_sparse": 9.0,
                },
            },
        ]

        ranked = module.rank_plausible_candidates(candidates, min_official_ate_delta=0.0)

        self.assertEqual([item["sample_id"] for item in ranked], ["b", "a"])

    def test_read_metric_pairs_and_candidate_delta(self):
        module = _load_module(BUILDER_PATH, "codex_waymo_plausible_builder_pairs")
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "seq_metrics.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["seq", "ATE", "RPE trans", "RPE rot"])
                writer.writeheader()
                writer.writerow({"seq": "sample__clean", "ATE": "1.0", "RPE trans": "2.0", "RPE rot": "3.0"})
                writer.writerow({"seq": "sample__noise", "ATE": "4.0", "RPE trans": "6.0", "RPE rot": "8.0"})

            pairs = module.read_metric_pairs(csv_path)
            candidate = module.candidate_from_metric_pair(
                source="old",
                base="sample",
                clean_dir=Path("/clean"),
                noise_dir=Path("/noise"),
                official=pairs["sample"],
                sparse={
                    "clean_ATE": 1.0,
                    "noise_ATE": 2.0,
                    "clean_RPE trans": 1.0,
                    "noise_RPE trans": 1.5,
                    "clean_RPE rot": 1.0,
                    "noise_RPE rot": 1.1,
                },
                note="test",
            )

        metrics = candidate["selection_metrics"]
        self.assertAlmostEqual(metrics["official_ATE_delta"], 3.0)
        self.assertAlmostEqual(metrics["sparse_ATE_delta"], 1.0)
        self.assertAlmostEqual(metrics["ATE_delta_gap_official_minus_sparse"], 2.0)


class PreparePi3AttentionSeqBundleTests(unittest.TestCase):
    def test_prepare_bundle_resizes_to_pi3_eval_width_and_multiple_of_14_height(self):
        module = _load_module(PREPARE_PATH, "codex_prepare_pi3_attention_seq_bundle")
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_root = Path(tmpdir) / "scene__anchor__noise"
            color_dir = seq_root / "color_90"
            color_dir.mkdir(parents=True)
            for idx in range(2):
                image = Image.fromarray(np.full((30, 60, 3), 40 + idx, dtype=np.uint8))
                image.save(color_dir / f"frame_{idx:04d}.jpg")

            paths = module.discover_image_paths(seq_root, "color_90")
            images = module.load_resized_images(paths, target_width=56)
            pi3_images = module.load_resized_images(paths, target_width=512)
            meta = module.build_bundle_meta(
                type("Args", (), {
                    "sample_id": "",
                    "sample_name": "",
                    "bundle_id": "",
                    "bundle_label": "",
                    "sample_scene": "",
                    "sample_stride": 0,
                    "sample_views": 0,
                    "sample_note": "",
                })(),
                seq_root,
                images,
            )

        self.assertEqual(tuple(images.shape), (2, 28, 56, 3))
        self.assertEqual(tuple(pi3_images.shape), (2, 252, 504, 3))
        self.assertEqual(meta["sample_id"], "scene__anchor__noise")
        self.assertEqual(meta["sample_views"], 2)


if __name__ == "__main__":
    unittest.main()
