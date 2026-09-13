from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "vggt"
    / "build_re10k_same_scene_candidate_pool_benchmark.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing RE10K candidate-pool builder: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("build_re10k_same_scene_candidate_pool_benchmark", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Re10kSameSceneCandidatePoolTests(unittest.TestCase):
    def test_pose_baseline_selection_keeps_clean_core_fixed_and_round_robins_bands(self):
        module = load_module()

        frame_names = [f"{idx:06d}" for idx in range(80)]
        camera_centers = np.stack(
            [
                np.asarray(
                    [
                        float(idx),
                        0.0,
                        0.0,
                    ],
                    dtype=np.float64,
                )
                for idx in range(len(frame_names))
            ],
            axis=0,
        )

        selection = module.select_same_scene_candidate_pool(
            frame_names=frame_names,
            camera_centers=camera_centers,
            ref_index=40,
            scene_key="scene-test",
            core_size=6,
            pool_sizes=[6, 10, 14],
            clean_core_policy="temporal",
        )

        self.assertEqual(selection.core_indices, [40, 35, 45, 30, 50, 25])
        self.assertEqual(selection.pool_indices_by_size[6], selection.core_indices)
        self.assertEqual(selection.pool_indices_by_size[10][:6], selection.core_indices)
        self.assertEqual(selection.pool_indices_by_size[14][:6], selection.core_indices)
        self.assertEqual(
            [row.band for row in selection.candidate_rows[:8]],
            ["near", "mid", "far", "tail", "near", "mid", "far", "tail"],
        )

    def test_paper10_style_clean_core_uses_scene_wide_stable_supports(self):
        module = load_module()

        frame_names = [f"{idx:06d}" for idx in range(120)]
        camera_centers = np.stack(
            [np.asarray([float(idx), 0.0, 0.0], dtype=np.float64) for idx in range(len(frame_names))],
            axis=0,
        )

        selection_a = module.select_same_scene_candidate_pool(
            frame_names=frame_names,
            camera_centers=camera_centers,
            ref_index=40,
            scene_key="scene-test",
            core_size=6,
            pool_sizes=[6, 10],
            clean_core_policy="paper10",
            seed=20260215,
        )
        selection_b = module.select_same_scene_candidate_pool(
            frame_names=frame_names,
            camera_centers=camera_centers,
            ref_index=40,
            scene_key="scene-test",
            core_size=6,
            pool_sizes=[6, 10],
            clean_core_policy="paper10",
            seed=20260215,
        )

        self.assertEqual(selection_a.core_indices, selection_b.core_indices)
        self.assertEqual(selection_a.core_indices[0], 40)
        self.assertEqual(len(set(selection_a.core_indices)), 6)
        self.assertNotEqual(selection_a.core_indices, [40, 35, 45, 30, 50, 25])
        self.assertEqual(selection_a.pool_indices_by_size[6], selection_a.core_indices)
        self.assertEqual(selection_a.pool_indices_by_size[10][:6], selection_a.core_indices)

    def test_anchor_indices_stay_away_from_sequence_edges(self):
        module = load_module()

        anchors = module.select_anchor_indices(total_frames=100, anchors_per_scene=5, min_edge_margin=20)

        self.assertEqual(anchors, [20, 35, 50, 65, 79])

    def test_stage_image_file_symlink_mode_links_without_copying(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src = root / "source.jpg"
            dst = root / "nested" / "staged.jpg"
            src.write_bytes(b"image")

            status = module.stage_image_file(src=src, dst=dst, skip_existing=False, image_stage_mode="symlink")
            second = module.stage_image_file(src=src, dst=dst, skip_existing=True, image_stage_mode="symlink")

            self.assertEqual(status, "linked")
            self.assertEqual(second, "skipped")
            self.assertTrue(dst.is_symlink())
            self.assertEqual(dst.readlink(), src)

    def test_clean_core_baseline_filter_rejects_tiny_motion_core(self):
        module = load_module()

        camera_centers = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.05, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        stats = module.compute_clean_core_baseline_stats(
            camera_centers=camera_centers,
            ref_index=0,
            core_indices=[0, 1, 2, 3, 4, 5],
        )

        self.assertAlmostEqual(stats["median"], 0.03)
        self.assertFalse(
            module.clean_core_passes_baseline_filter(
                baseline_stats=stats,
                min_median_baseline=0.15,
                min_max_baseline=0.3,
            )
        )

    def test_clean_core_baseline_filter_accepts_reasonable_motion_core(self):
        module = load_module()

        camera_centers = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        stats = module.compute_clean_core_baseline_stats(
            camera_centers=camera_centers,
            ref_index=0,
            core_indices=[0, 1, 2, 3, 4, 5],
        )

        self.assertAlmostEqual(stats["median"], 0.6)
        self.assertTrue(
            module.clean_core_passes_baseline_filter(
                baseline_stats=stats,
                min_median_baseline=0.15,
                min_max_baseline=0.3,
            )
        )


if __name__ == "__main__":
    unittest.main()
