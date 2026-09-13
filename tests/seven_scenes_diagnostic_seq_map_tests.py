from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_MODULE_PATH = (
    REPO_ROOT
    / "aidi"
    / "scripts"
    / "baselines"
    / "build_7scenes_diagnostic_seq_maps.py"
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


class SevenScenesDiagnosticSeqMapTests(unittest.TestCase):
    def test_build_dense_ids_uses_fixed_stride(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_7scenes_diagnostic_seq_maps")

        valid_ids = list(range(200))
        dense_ids = module.build_dense_ids(valid_ids, start_index=3, stride=5, count=10)

        self.assertEqual(dense_ids, [3, 8, 13, 18, 23, 28, 33, 38, 43, 48])

    def test_build_twochunk_ids_uses_gap_after_first_chunk(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_7scenes_diagnostic_seq_maps")

        valid_ids = list(range(300))
        twochunk_ids = module.build_twochunk_ids(valid_ids, start_index=4, stride=5, chunk_size=5, gap=80)

        self.assertEqual(twochunk_ids, [4, 9, 14, 19, 24, 84, 89, 94, 99, 104])

    def test_select_anchor_indices_requires_both_settings_complete(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_7scenes_diagnostic_seq_maps")

        anchor_indices = module.select_anchor_indices(
            num_valid_frames=150,
            anchor_stride=40,
            dense_stride=5,
            dense_count=10,
            twochunk_stride=5,
            chunk_size=5,
            gap=80,
        )

        self.assertEqual(anchor_indices, [0, 40])

    def test_build_setting_seq_maps_keeps_common_anchor_keys(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_7scenes_diagnostic_seq_maps")

        scene_to_valid_ids = {
            "chess/seq-03": list(range(150)),
        }

        dense_map, twochunk_map = module.build_setting_seq_maps(
            scene_to_valid_ids=scene_to_valid_ids,
            anchor_stride=40,
            dense_stride=5,
            dense_count=10,
            twochunk_stride=5,
            chunk_size=5,
            gap=80,
        )

        self.assertEqual(set(dense_map.keys()), set(twochunk_map.keys()))
        self.assertIn("chess/seq-03/ref_000000", dense_map)
        self.assertEqual(dense_map["chess/seq-03/ref_000000"]["scene"], "chess/seq-03")
        self.assertEqual(
            dense_map["chess/seq-03/ref_000000"]["ids"],
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        )
        self.assertEqual(
            twochunk_map["chess/seq-03/ref_000000"]["ids"],
            [0, 5, 10, 15, 20, 80, 85, 90, 95, 100],
        )
        self.assertEqual(
            twochunk_map["chess/seq-03/ref_000040"]["ids"],
            [40, 45, 50, 55, 60, 120, 125, 130, 135, 140],
        )

    def test_filter_valid_frame_ids_intersects_available_dirs_and_camera_keys(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_7scenes_diagnostic_seq_maps")

        available_frame_ids = [0, 1, 2, 4, 6, 8]
        camera_ids = ["000000", "000001", "000003", "000004", "000006"]

        valid_ids = module.filter_valid_frame_ids(
            available_frame_ids=available_frame_ids,
            camera_ids=camera_ids,
        )

        self.assertEqual(valid_ids, [0, 1, 4, 6])

    def test_list_available_frame_ids_uses_digit_named_frame_dirs(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_7scenes_diagnostic_seq_maps")
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_root = Path(tmpdir) / "images"
            (frame_root / "000000").mkdir(parents=True)
            (frame_root / "000001").mkdir(parents=True)
            (frame_root / "misc").mkdir(parents=True)
            (frame_root / "README.txt").write_text("ignore", encoding="utf-8")

            frame_ids = module.list_available_frame_ids(frame_root, suffixes=[".png"])

        self.assertEqual(frame_ids, [0, 1])

    def test_collect_valid_frame_ids_uses_camera_ids_when_roots_exist(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_7scenes_diagnostic_seq_maps")
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_dir = Path(tmpdir) / "chess_seq-03"
            (seq_dir / "images").mkdir(parents=True)
            (seq_dir / "depths").mkdir(parents=True)
            (seq_dir / "intri.yml").write_text("stub", encoding="utf-8")
            (seq_dir / "extri.yml").write_text("stub", encoding="utf-8")

            with mock.patch.object(module, "load_camera_ids", return_value=["000000", "000005", "000010"]):
                valid_ids = module.collect_valid_frame_ids(seq_dir)

        self.assertEqual(valid_ids, [0, 5, 10])

    def test_write_seq_maps_emits_summary_counts(self):
        module = load_module(GENERATOR_MODULE_PATH, "build_7scenes_diagnostic_seq_maps")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            dense_map = {"chess/seq-03/ref_000000": {"scene": "chess/seq-03", "ids": list(range(10))}}
            twochunk_map = {"chess/seq-03/ref_000000": {"scene": "chess/seq-03", "ids": list(range(10, 20))}}

            written = module.write_seq_maps(
                output_dir=out_dir,
                dense_map=dense_map,
                twochunk_map=twochunk_map,
                config_summary={"dataset_root": "/tmp/7scenes"},
            )

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(written["dense"], out_dir / "7scenes_diagnostic_dense10.json")
        self.assertEqual(written["twochunk"], out_dir / "7scenes_diagnostic_twochunk10.json")
        self.assertEqual(summary["counts"]["dense10"], 1)
        self.assertEqual(summary["counts"]["twochunk10"], 1)
        self.assertEqual(summary["config"]["dataset_root"], "/tmp/7scenes")


if __name__ == "__main__":
    unittest.main()
