from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_MODULE_PATH = (
    REPO_ROOT
    / "aidi"
    / "scripts"
    / "baselines"
    / "build_dtu_diagnostic_seq_maps.py"
)
SEQ_MAP_MODULE_PATH = (
    REPO_ROOT
    / "aidi"
    / "scripts"
    / "baselines"
    / "mv_recon_seq_map_utils.py"
)
TAIL_RANKING_MODULE_PATH = (
    REPO_ROOT
    / "aidi"
    / "scripts"
    / "baselines"
    / "build_dtu_geometry_tail_rankings.py"
)


def load_module(module_path: Path, module_name: str):
    if not module_path.is_file():
        raise AssertionError(f"Missing module: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DtuDiagnosticSeqMapTests(unittest.TestCase):
    def test_select_rank_band_views_uses_tail_bands(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_dtu_diagnostic_seq_maps")

        source_views = list(range(101, 121))
        picked = module.select_rank_band_views(source_views, exclude_top_k=10, num_select=4)

        self.assertEqual(picked, [111, 114, 116, 119])

    def test_build_setting_seq_maps_keeps_scene_binding_and_tuple_order(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_dtu_diagnostic_seq_maps")

        pair_by_ref = {
            0: list(range(101, 121)),
            1: list(range(201, 221)),
        }
        scan_to_refs = {"scan24": [0, 1]}

        clean_map, mix_map = module.build_setting_seq_maps(
            pair_by_ref=pair_by_ref,
            scan_to_refs=scan_to_refs,
            clean_src_count=9,
            mix_clean_src_count=5,
            mix_out_count=4,
            exclude_top_k=10,
        )

        self.assertEqual(
            clean_map["scan24/ref_00000000"]["ids"],
            [0, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        )
        self.assertEqual(clean_map["scan24/ref_00000000"]["scene"], "scan24")
        self.assertEqual(
            mix_map["scan24/ref_00000000"]["ids"],
            [0, 101, 102, 103, 104, 105, 111, 114, 116, 119],
        )
        self.assertEqual(
            mix_map["scan24/ref_00000001"]["ids"],
            [1, 201, 202, 203, 204, 205, 211, 214, 216, 219],
        )

    def test_parse_pair_file_reads_ranked_view_ids(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_dtu_diagnostic_seq_maps")
        with tempfile.TemporaryDirectory() as tmpdir:
            pair_path = Path(tmpdir) / "pair.txt"
            pair_path.write_text(
                "\n".join(
                    [
                        "2",
                        "0",
                        "4 11 0.9 12 0.8 13 0.7 14 0.6",
                        "1",
                        "4 21 0.95 22 0.85 23 0.75 24 0.65",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            pair_by_ref = module.parse_pair_file(pair_path)

        self.assertEqual(pair_by_ref[0], [11, 12, 13, 14])
        self.assertEqual(pair_by_ref[1], [21, 22, 23, 24])

    def test_merge_ranked_sources_appends_tail_after_official_topk(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_dtu_diagnostic_seq_maps")

        merged = module.merge_ranked_sources(
            scan_name="scan24",
            ref_id=0,
            official_sources=list(range(101, 111)),
            tail_rankings={"scan24": {"0": list(range(111, 121))}},
        )

        self.assertEqual(merged, list(range(101, 121)))

    def test_build_setting_seq_maps_uses_tail_rankings_when_pair_txt_has_only_top10(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_dtu_diagnostic_seq_maps")

        pair_by_ref = {
            0: list(range(101, 111)),
        }
        scan_to_refs = {"scan24": [0]}

        clean_map, mix_map = module.build_setting_seq_maps(
            pair_by_ref=pair_by_ref,
            scan_to_refs=scan_to_refs,
            clean_src_count=9,
            mix_clean_src_count=5,
            mix_out_count=4,
            exclude_top_k=10,
            tail_rankings={"scan24": {"0": list(range(111, 121))}},
        )

        self.assertEqual(
            clean_map["scan24/ref_00000000"]["ids"],
            [0, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        )
        self.assertEqual(
            mix_map["scan24/ref_00000000"]["ids"],
            [0, 101, 102, 103, 104, 105, 111, 114, 116, 119],
        )

    def test_load_scan_pair_maps_falls_back_to_scan_local_pair_files(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_dtu_diagnostic_seq_maps")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for scan_name, src_a, src_b in (("scan24", 11, 12), ("scan25", 21, 22)):
                scan_dir = root / scan_name
                scan_dir.mkdir(parents=True, exist_ok=True)
                (scan_dir / "pair.txt").write_text(
                    "\n".join(
                        [
                            "1",
                            "0",
                            f"2 {src_a} 0.9 {src_b} 0.8",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

            pair_maps = module.load_scan_pair_maps(
                dataset_root=root,
                scan_names=["scan24", "scan25"],
                pair_txt="",
            )

        self.assertEqual(pair_maps["scan24"][0], [11, 12])
        self.assertEqual(pair_maps["scan25"][0], [21, 22])

    def test_write_seq_maps_emits_summary_counts(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_dtu_diagnostic_seq_maps")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            clean_map = {"scan24/ref_00000000": {"scene": "scan24", "ids": list(range(10))}}
            mix_map = {"scan24/ref_00000000": {"scene": "scan24", "ids": list(range(10, 20))}}

            written = module.write_seq_maps(
                output_dir=out_dir,
                clean_map=clean_map,
                mix_map=mix_map,
                config_summary={"pair_txt": "/tmp/pair.txt"},
            )

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(written["clean"], out_dir / "dtu_diagnostic_clean10.json")
        self.assertEqual(written["mix"], out_dir / "dtu_diagnostic_mix10_rankband4.json")
        self.assertEqual(summary["counts"]["clean10"], 1)
        self.assertEqual(summary["counts"]["mix10_rankband4"], 1)
        self.assertEqual(summary["config"]["pair_txt"], "/tmp/pair.txt")

    def test_load_tail_rankings_reads_scan_and_ref_keys(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_dtu_diagnostic_seq_maps")
        with tempfile.TemporaryDirectory() as tmpdir:
            tail_path = Path(tmpdir) / "tail.json"
            tail_path.write_text(
                json.dumps(
                    {
                        "scan24": {
                            "0": [11, 12, 13],
                        }
                    }
                ),
                encoding="utf-8",
            )

            tail_rankings = module.load_tail_rankings(tail_path)

        self.assertEqual(tail_rankings, {"scan24": {"0": [11, 12, 13]}})


class DtuGeometryTailRankingTests(unittest.TestCase):
    def test_load_mask_resizes_to_target_shape(self):
        module = load_module(TAIL_RANKING_MODULE_PATH, "build_dtu_geometry_tail_rankings")
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "mask.png"
            from PIL import Image
            import numpy as np

            Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8)).save(mask_path)

            mask = module.load_mask(mask_path, target_hw=(4, 6))

        self.assertEqual(mask.shape, (4, 6))
        self.assertTrue(mask.dtype == bool)

    def test_rank_candidates_by_score_prefers_higher_overlap_then_baseline(self):
        module = load_module(TAIL_RANKING_MODULE_PATH, "build_dtu_geometry_tail_rankings")

        ordered = module.rank_candidates_by_score(
            {
                12: {"overlap": 0.4, "baseline": 0.9},
                13: {"overlap": 0.7, "baseline": 0.1},
                14: {"overlap": 0.7, "baseline": 0.5},
            }
        )

        self.assertEqual(ordered, [14, 13, 12])


class MvReconSeqMapUtilsTests(unittest.TestCase):
    def test_legacy_seq_map_entry_uses_key_as_scene_name(self):
        module = load_module(SEQ_MAP_MODULE_PATH, "mv_recon_seq_map_utils")

        entries = list(module.iter_seq_map_entries({"scan24": [0, 1, 2]}))

        self.assertEqual(entries, [("scan24", "scan24", [0, 1, 2])])

    def test_structured_seq_map_entry_can_override_scene_name(self):
        module = load_module(SEQ_MAP_MODULE_PATH, "mv_recon_seq_map_utils")

        entries = list(
            module.iter_seq_map_entries(
                {
                    "scan24/ref_00000000": {
                        "scene": "scan24",
                        "ids": [0, 11, 12, 13],
                        "reference_id": 0,
                    }
                }
            )
        )

        self.assertEqual(entries, [("scan24/ref_00000000", "scan24", [0, 11, 12, 13])])


if __name__ == "__main__":
    unittest.main()
