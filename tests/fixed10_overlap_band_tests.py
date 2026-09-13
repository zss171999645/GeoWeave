from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "aidi" / "scripts" / "baselines" / "build_fixed10_overlap_band_benchmark.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_fixed10_overlap_band_benchmark", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fixed10OverlapBandTests(unittest.TestCase):
    def test_select_fixed10_overlap_variants_keeps_total_count_and_roles(self):
        module = load_script_module()
        overlap_by_ref = {
            1: 0.95,
            2: 0.90,
            3: 0.85,
            4: 0.80,
            5: 0.75,
            6: 0.70,
            7: 0.65,
            8: 0.60,
            9: 0.55,
            10: 0.18,
            11: 0.08,
            12: 0.02,
            13: 0.003,
            14: 0.001,
        }

        variants = module.select_fixed10_overlap_variants(
            ref_id=0,
            overlap_by_ref=overlap_by_ref,
            total_size=10,
            core_size=6,
            clean_overlap_threshold=0.20,
            clean_temporal_dedup=1,
        )

        self.assertEqual(set(variants.keys()), {"near10", "mixed10", "tail10"})
        self.assertTrue(all(len(ids) == 10 for ids in variants.values()))
        self.assertEqual(variants["near10"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(variants["mixed10"], [0, 1, 2, 3, 4, 5, 10, 11, 12, 13])
        self.assertEqual(variants["tail10"], [0, 1, 2, 3, 4, 5, 14, 13, 12, 11])

    def test_frame_rows_label_core_and_overlap_bands(self):
        module = load_script_module()
        rows = module.frame_rows_for_variant(
            ordered_ids=[0, 1, 2, 10, 11, 12],
            ref_id=0,
            core_ids=[0, 1, 2],
            overlap_by_ref={1: 0.9, 2: 0.8, 10: 0.18, 11: 0.08, 12: 0.002},
        )

        self.assertEqual([row["role"] for row in rows], ["ref", "core", "core", "candidate", "candidate", "candidate"])
        self.assertEqual([row["band"] for row in rows], ["ref", "core", "core", "near", "mid", "tail"])


if __name__ == "__main__":
    unittest.main()
