from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "aidi" / "scripts" / "baselines" / "build_scannetpp_paired_overlap_sweep.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("build_scannetpp_paired_overlap_sweep", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PairedOverlapSweepSelectionTests(unittest.TestCase):
    def test_classify_overlap_band_uses_half_open_boundaries(self):
        module = load_script_module()

        cases = {
            0.0: "near_zero",
            0.004999: "near_zero",
            0.005: "low",
            0.029999: "low",
            0.03: "medium",
            0.099999: "medium",
            0.10: "high",
            0.199999: "high",
            0.20: None,
            -1e-6: None,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(module.classify_overlap_band(value), expected)

    def test_select_group_b_by_band_uses_midpoint_internal_overlap_and_frame_id_tiebreaks(self):
        module = load_script_module()
        group_a = [10, 11, 12, 13, 14]
        candidates = [
            {"anchor_b": 200, "group_b_ids": [200, 201, 202, 203, 204], "cross_overlap_mean": 0.151, "group_b_internal_overlap_mean": 0.90},
            {"anchor_b": 190, "group_b_ids": [190, 191, 192, 193, 194], "cross_overlap_mean": 0.149, "group_b_internal_overlap_mean": 0.80},
            {"anchor_b": 180, "group_b_ids": [180, 181, 182, 183, 184], "cross_overlap_mean": 0.149, "group_b_internal_overlap_mean": 0.90},
            {"anchor_b": 170, "group_b_ids": [170, 171, 172, 173, 174], "cross_overlap_mean": 0.149, "group_b_internal_overlap_mean": 0.90},
            {"anchor_b": 300, "group_b_ids": [300, 301, 302, 303, 304], "cross_overlap_mean": 0.065, "group_b_internal_overlap_mean": 0.70},
            {"anchor_b": 400, "group_b_ids": [400, 401, 402, 403, 404], "cross_overlap_mean": 0.0175, "group_b_internal_overlap_mean": 0.70},
            {"anchor_b": 500, "group_b_ids": [500, 501, 502, 503, 504], "cross_overlap_mean": 0.0025, "group_b_internal_overlap_mean": 0.70},
        ]

        selected = module.select_group_b_by_band(group_a, candidates)

        self.assertIsNotNone(selected)
        self.assertEqual(list(selected), ["high", "medium", "low", "near_zero"])
        self.assertEqual(selected["high"]["anchor_b"], 170)
        self.assertEqual(selected["medium"]["anchor_b"], 300)
        self.assertEqual(selected["low"]["anchor_b"], 400)
        self.assertEqual(selected["near_zero"]["anchor_b"], 500)

    def test_select_group_b_by_band_rejects_incomplete_anchor_and_keeps_group_a_fixed(self):
        module = load_script_module()
        group_a = [10, 11, 12, 13, 14]
        incomplete = [
            {"anchor_b": 200, "group_b_ids": [200, 201, 202, 203, 204], "cross_overlap_mean": 0.15, "group_b_internal_overlap_mean": 0.8},
            {"anchor_b": 300, "group_b_ids": [300, 301, 302, 303, 304], "cross_overlap_mean": 0.06, "group_b_internal_overlap_mean": 0.8},
            {"anchor_b": 400, "group_b_ids": [400, 401, 402, 403, 404], "cross_overlap_mean": 0.02, "group_b_internal_overlap_mean": 0.8},
        ]
        self.assertIsNone(module.select_group_b_by_band(group_a, incomplete))

        complete = [
            *incomplete,
            {"anchor_b": 500, "group_b_ids": [500, 501, 502, 503, 504], "cross_overlap_mean": 0.002, "group_b_internal_overlap_mean": 0.8},
        ]
        rows = module.selection_rows_for_anchor(
            scene_name="scene_a",
            anchor_a=10,
            group_a_ids=group_a,
            selected_by_band=module.select_group_b_by_band(group_a, complete),
        )

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["group_a_ids"] == group_a for row in rows))
        self.assertEqual([row["overlap_level"] for row in rows], ["high", "medium", "low", "near_zero"])

    def test_select_group_b_by_quantiles_returns_ordered_distinct_levels(self):
        module = load_script_module()
        candidates = [
            {
                "anchor_b": index * 10,
                "group_b_ids": list(range(index * 10, index * 10 + 5)),
                "cross_overlap_mean": overlap,
                "group_b_internal_overlap_mean": 0.8,
            }
            for index, overlap in enumerate((0.01, 0.03, 0.07, 0.12, 0.20, 0.35, 0.55, 0.75), start=1)
        ]

        selected = module.select_group_b_by_quantiles(candidates)

        self.assertEqual(list(selected), ["highest", "medium", "low", "lowest"])
        means = [selected[level]["cross_overlap_mean"] for level in selected]
        self.assertTrue(all(lhs > rhs for lhs, rhs in zip(means, means[1:])))
        self.assertEqual(len({selected[level]["anchor_b"] for level in selected}), 4)

        zero_heavy = [
            {
                "anchor_b": index * 10,
                "group_b_ids": list(range(index * 10, index * 10 + 5)),
                "cross_overlap_mean": overlap,
                "group_b_internal_overlap_mean": 0.8,
            }
            for index, overlap in enumerate((0.0, 0.0, 0.0, 0.01, 0.03, 0.05, 0.10), start=1)
        ]
        selected = module.select_group_b_by_quantiles(zero_heavy)
        means = [selected[level]["cross_overlap_mean"] for level in selected]
        self.assertTrue(all(lhs > rhs for lhs, rhs in zip(means, means[1:])))

    def test_select_balanced_anchors_is_round_robin_and_respects_scene_cap(self):
        module = load_script_module()
        rows = []
        for scene_name, anchors in (("scene_a", [10, 20, 30]), ("scene_b", [40]), ("scene_c", [50, 60])):
            for anchor_a in anchors:
                for level, mean in (("high", 0.15), ("medium", 0.06), ("low", 0.02), ("near_zero", 0.002)):
                    rows.append(
                        {
                            "scene_name": scene_name,
                            "anchor_a": anchor_a,
                            "overlap_level": level,
                            "group_a_ids": [anchor_a, anchor_a + 1, anchor_a + 2, anchor_a + 3, anchor_a + 4],
                            "group_b_ids": [anchor_a + 100, anchor_a + 101, anchor_a + 102, anchor_a + 103, anchor_a + 104],
                            "cross_overlap_mean": mean,
                            "group_a_anchor_support_min": 0.2,
                            "group_b_anchor_support_min": 0.2,
                        }
                    )

        selected = module.select_balanced_anchor_rows(rows, target_anchors=5, max_anchors_per_scene=2)

        selected_keys = []
        for row in selected:
            key = (row["scene_name"], row["anchor_a"])
            if key not in selected_keys:
                selected_keys.append(key)
        self.assertEqual(selected_keys, [("scene_a", 10), ("scene_b", 40), ("scene_c", 50), ("scene_a", 30), ("scene_c", 60)])
        self.assertEqual(len(selected), 20)

        one_per_scene = module.select_balanced_anchor_rows(rows, target_anchors=3, max_anchors_per_scene=2)
        one_per_scene_keys = []
        for row in one_per_scene:
            key = (row["scene_name"], row["anchor_a"])
            if key not in one_per_scene_keys:
                one_per_scene_keys.append(key)
        self.assertEqual(one_per_scene_keys, [("scene_a", 20), ("scene_b", 40), ("scene_c", 50)])

    def test_validate_selection_rows_enforces_balance_uniqueness_thresholds_and_bands(self):
        module = load_script_module()
        rows = []
        group_a = [10, 11, 12, 13, 14]
        for level, mean, start in (
            ("high", 0.15, 100),
            ("medium", 0.06, 200),
            ("low", 0.02, 300),
            ("near_zero", 0.002, 400),
        ):
            rows.append(
                {
                    "scene_name": "scene_a",
                    "anchor_a": 10,
                    "overlap_level": level,
                    "group_a_ids": group_a,
                    "group_b_ids": list(range(start, start + 5)),
                    "cross_overlap_mean": mean,
                    "group_a_anchor_support_min": 0.10,
                    "group_b_anchor_support_min": 0.09,
                }
            )

        report = module.validate_selection_rows(rows, min_anchors=1, local_overlap_threshold=0.08)
        self.assertEqual(report["num_anchors"], 1)
        self.assertEqual(report["num_rows"], 4)

        invalid = [dict(row) for row in rows]
        invalid[1]["group_b_ids"] = [10, 201, 202, 203, 204]
        with self.assertRaisesRegex(ValueError, "overlap between Group A and Group B"):
            module.validate_selection_rows(invalid, min_anchors=1, local_overlap_threshold=0.08)

        invalid = [dict(row) for row in rows]
        invalid[2]["group_b_anchor_support_min"] = 0.079
        with self.assertRaisesRegex(ValueError, "local-overlap threshold"):
            module.validate_selection_rows(invalid, min_anchors=1, local_overlap_threshold=0.08)

        invalid = [dict(row) for row in rows]
        invalid[0]["cross_overlap_mean"] = 0.099
        with self.assertRaisesRegex(ValueError, "outside declared band"):
            module.validate_selection_rows(invalid, min_anchors=1, local_overlap_threshold=0.08)

        with self.assertRaisesRegex(ValueError, "at least 2 balanced anchors"):
            module.validate_selection_rows(rows, min_anchors=2, local_overlap_threshold=0.08)

    def test_enumerate_scene_candidates_emits_four_levels_for_one_fixed_group_a(self):
        module = load_script_module()
        local_groups = [list(range(start, start + 5)) for start in (0, 10, 20, 30, 40)]
        ids = [frame_id for group in local_groups for frame_id in group]
        table = {frame_id: {} for frame_id in ids}

        for group in local_groups:
            for lhs in group:
                for rhs in group:
                    if lhs != rhs:
                        table[lhs][rhs] = 0.9
        for group, overlap in zip(local_groups[1:], (0.15, 0.06, 0.02, 0.002)):
            for lhs in local_groups[0]:
                for rhs in group:
                    table[lhs][rhs] = overlap
                    table[rhs][lhs] = overlap

        candidate_rows, complete_rows = module.enumerate_scene_candidates(
            scene_name="scene_a",
            scene_record={"frame_ids": ids},
            overlap_table=table,
            local_overlap_threshold=0.08,
            anchor_stride=100,
        )

        self.assertGreaterEqual(len(candidate_rows), 20)
        self.assertEqual(len(complete_rows), 4)
        self.assertTrue(all(row["group_a_ids"] == [0, 1, 2, 3, 4] for row in complete_rows))
        self.assertEqual([row["overlap_level"] for row in complete_rows], ["high", "medium", "low", "near_zero"])

    def test_uniform_frame_indices_cover_full_sequence_without_duplicates(self):
        module = load_script_module()
        self.assertEqual(module.uniform_frame_indices(total=5, target=0), [0, 1, 2, 3, 4])
        self.assertEqual(module.uniform_frame_indices(total=5, target=5), [0, 1, 2, 3, 4])
        self.assertEqual(module.uniform_frame_indices(total=11, target=4), [0, 3, 7, 10])
        self.assertEqual(len(set(module.uniform_frame_indices(total=1418, target=256))), 256)

    def test_merge_uniform_with_required_frames_preserves_legacy_anchors(self):
        module = load_script_module()
        self.assertEqual(
            module.merge_uniform_with_required_frames(total=11, target=4, required=[2, 9]),
            [0, 2, 3, 7, 9, 10],
        )

    def test_nerfstudio_opengl_pose_is_converted_to_opencv_camera_axes(self):
        module = load_script_module()
        c2w_gl = np.eye(4, dtype=np.float64)
        c2w_gl[:3, 3] = [1.0, 2.0, 3.0]

        c2w_cv = module.nerfstudio_c2w_to_opencv(c2w_gl)

        np.testing.assert_allclose(c2w_cv[:3, :3], np.diag([1.0, -1.0, -1.0]))
        np.testing.assert_allclose(c2w_cv[:3, 3], [1.0, 2.0, 3.0])

    def test_decode_complete_anchor_csv_row_restores_numeric_and_list_types(self):
        module = load_script_module()
        row = module.decode_complete_anchor_csv_row(
            {
                "scene_name": "scene_a",
                "anchor_a": "10",
                "anchor_b": "20",
                "overlap_level": "low",
                "group_a_ids": "[10, 11, 12, 13, 14]",
                "group_b_ids": "[20, 21, 22, 23, 24]",
                "cross_overlap_mean": "0.02",
                "group_a_anchor_support_min": "0.1",
                "group_b_anchor_support_min": "0.09",
            }
        )

        self.assertEqual(row["anchor_a"], 10)
        self.assertEqual(row["group_a_ids"], [10, 11, 12, 13, 14])
        self.assertIsInstance(row["cross_overlap_mean"], float)

    def test_materialize_from_frozen_manifest_keeps_group_a_bytes_identical(self):
        module = load_script_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "data"
            scene_root = dataset_root / "scene_a"
            image_root = scene_root / "dslr" / "resized_images"
            transforms_root = scene_root / "dslr" / "nerfstudio"
            image_root.mkdir(parents=True)
            transforms_root.mkdir(parents=True)
            frames = []
            for frame_id in range(25):
                name = f"frame_{frame_id:04d}.JPG"
                Image.new("RGB", (12, 8), (frame_id, frame_id, frame_id)).save(image_root / name)
                pose = np.eye(4, dtype=float)
                pose[0, 3] = frame_id
                frames.append({"file_path": name, "transform_matrix": pose.tolist()})
            (transforms_root / "transforms.json").write_text(
                json.dumps({"frames": frames, "w": 12, "h": 8, "fl_x": 8, "fl_y": 8, "cx": 6, "cy": 4}),
                encoding="utf-8",
            )

            group_a = [0, 1, 2, 3, 4]
            rows = []
            for level, start, mean in (
                ("highest", 5, 0.15),
                ("medium", 10, 0.06),
                ("low", 15, 0.02),
                ("lowest", 20, 0.002),
            ):
                rows.append(
                    {
                        "scene_name": "scene_a",
                        "anchor_a": 0,
                        "anchor_b": start,
                        "overlap_level": level,
                        "group_a_ids": group_a,
                        "group_b_ids": list(range(start, start + 5)),
                        "cross_overlap_mean": mean,
                    }
                )
            manifest_path = root / "selection_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset_root),
                        "levels": ["highest", "medium", "low", "lowest"],
                        "rows": rows,
                    }
                ),
                encoding="utf-8",
            )

            report = module.materialize_from_manifest(
                manifest_path=manifest_path,
                output_root=root / "tuples",
                dataset_name="scannetpp_overlap_sweep",
                overwrite=False,
            )

            self.assertEqual(report["num_tuples"], 4)
            self.assertEqual(
                [record["tuple_name"] for record in report["tuples"]],
                ["scene_a__a000000__highest", "scene_a__a000000__medium", "scene_a__a000000__low", "scene_a__a000000__lowest"],
            )
            hashes_by_level = [record["group_a_sha256"] for record in report["tuples"]]
            self.assertTrue(all(hashes == hashes_by_level[0] for hashes in hashes_by_level))
            for record in report["tuples"]:
                pose_rows = np.loadtxt(Path(record["tuple_root"]) / "pose_90.txt")
                self.assertEqual(pose_rows.shape, (10, 16))

            protocol = json.loads((root / "tuples" / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["variants"], ["highest", "medium", "low", "lowest"])
            self.assertEqual(protocol["num_samples"], 1)
            self.assertEqual(set(protocol["samples"][0]["variants"]), set(protocol["variants"]))
            self.assertTrue(protocol["samples"][0]["variants"]["highest"]["input_dir"].endswith("/color_90"))

            with self.assertRaises(FileExistsError):
                module.materialize_from_manifest(
                    manifest_path=manifest_path,
                    output_root=root / "tuples",
                    dataset_name="scannetpp_overlap_sweep",
                    overwrite=False,
                )

    def test_enumerate_fixed_group_a_candidates_keeps_supplied_five_views(self):
        module = load_script_module()
        local_groups = [list(range(start, start + 5)) for start in (0, 10, 20, 30, 40)]
        ids = [frame_id for group in local_groups for frame_id in group]
        table = {frame_id: {} for frame_id in ids}
        for group in local_groups:
            for lhs in group:
                for rhs in group:
                    if lhs != rhs:
                        table[lhs][rhs] = 0.9
        for group, overlap in zip(local_groups[1:], (0.15, 0.06, 0.02, 0.002)):
            for lhs in local_groups[0]:
                for rhs in group:
                    table[lhs][rhs] = overlap
                    table[rhs][lhs] = overlap

        candidates, rows = module.enumerate_fixed_group_a_candidates(
            scene_name="scene_a",
            fixed_anchor={
                "anchor_name": "scene_a__legacy_a000001",
                "legacy_tuple_name": "legacy_tuple",
                "legacy_anchor_a": 1,
                "group_a_ids": [0, 1, 2, 3, 4],
            },
            scene_record={"frame_ids": ids},
            overlap_table=table,
            local_overlap_threshold=0.08,
        )

        self.assertGreaterEqual(len(candidates), 20)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["group_a_ids"] == [0, 1, 2, 3, 4] for row in rows))
        self.assertTrue(all(row["anchor_name"] == "scene_a__legacy_a000001" for row in rows))


if __name__ == "__main__":
    unittest.main()
