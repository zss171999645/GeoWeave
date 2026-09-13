import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from aidi.scripts.baselines import eval_pi3_videodepth_protocol as videodepth
from easyvolcap.utils.pi3 import kitti_videodepth as kitti_videodepth_utils


class PI3VideodepthAlignmentTests(unittest.TestCase):
    def test_parse_eval_frame_indices_supports_empty_and_csv(self):
        self.assertEqual(videodepth.parse_eval_frame_indices(""), [])
        self.assertEqual(videodepth.parse_eval_frame_indices("0, 2,5"), [0, 2, 5])

    def test_resolve_eval_frame_indices_reads_tuple_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            seq_root = data_root / "demo"
            seq_root.mkdir(parents=True)
            (seq_root / "tuple_meta.json").write_text(
                '{"eval_frame_indices": [4, 5, 6, 7, 8, 9]}\n',
                encoding="utf-8",
            )

            indices = videodepth.resolve_eval_frame_indices_for_sequence(
                eval_frame_indices_spec="tuple_meta",
                data_root=data_root,
                seq="demo",
            )

            self.assertEqual(indices, [4, 5, 6, 7, 8, 9])

    def test_subset_depth_sequence_for_eval_keeps_only_requested_frames(self):
        sequence = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)

        subset = videodepth.subset_depth_sequence_for_eval(sequence, [0, 2], "demo", "prediction")

        self.assertEqual(subset.shape, (2, 2, 3))
        np.testing.assert_array_equal(subset[0], sequence[0])
        np.testing.assert_array_equal(subset[1], sequence[2])

    def test_target_only_depth_metrics_ignore_non_eval_candidate_frames(self):
        gt = np.ones((3, 2, 2), dtype=np.float32)
        pred = gt.copy()
        pred[2] = 100.0

        full = videodepth.evaluate_depth_sequence(
            predicted_depth_original=pred,
            ground_truth_depth_original=gt,
            alignment="metric",
            max_depth=None,
            post_clip_max=None,
            device=torch.device("cpu"),
        )
        target_only = videodepth.evaluate_depth_sequence(
            predicted_depth_original=videodepth.subset_depth_sequence_for_eval(pred, [0, 1], "demo", "prediction"),
            ground_truth_depth_original=videodepth.subset_depth_sequence_for_eval(gt, [0, 1], "demo", "ground truth"),
            alignment="metric",
            max_depth=None,
            post_clip_max=None,
            device=torch.device("cpu"),
        )

        self.assertGreater(full["Abs Rel"], 0.0)
        self.assertEqual(target_only["Abs Rel"], 0.0)

    def test_scale_shift_alignment_uses_official_lad2_learning_rate(self):
        called = {}

        def fake_absolute_value_scaling2(
            predicted_depth: torch.Tensor,
            ground_truth_depth: torch.Tensor,
            s_init: float,
            t_init: float = 0.0,
            lr: float = 1e-4,
            max_iters: int = 1000,
            tol: float = 1e-6,
        ):
            called["lr"] = lr
            called["s_init"] = s_init
            return 1.0, 0.0

        predicted = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        ground_truth = np.array([[[1.1, 1.9], [3.2, 3.8]]], dtype=np.float32)

        with patch.object(videodepth, "absolute_value_scaling2", fake_absolute_value_scaling2):
            videodepth.evaluate_depth_sequence(
                predicted_depth_original=predicted,
                ground_truth_depth_original=ground_truth,
                alignment="scale&shift",
                max_depth=None,
                post_clip_max=None,
                device=torch.device("cpu"),
            )

        self.assertEqual(called["lr"], 1e-4)

    def test_kitti_videodepth_accepts_flat_official_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            image_dir = data_root / "image"
            gt_dir = data_root / "groundtruth_depth"
            image_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)
            self._write_rgb_png(image_dir / "2011_09_26_drive_0002_sync_image_0000000000_image_02.png")
            self._write_depth_png(gt_dir / "2011_09_26_drive_0002_sync_groundtruth_depth_0000000000_image_02.png")

            layout = videodepth.detect_layout(videodepth.DATASET_SPECS["kitti"], data_root, "02")
            self.assertEqual(layout, "official")
            names = videodepth.get_sequence_names_for_layout(videodepth.DATASET_SPECS["kitti"], data_root, layout, "02")
            self.assertEqual(names, ["2011_09_26_drive_0002_sync_02"])
            image_files, gt_files = videodepth.list_sequence_files_for_layout(
                videodepth.DATASET_SPECS["kitti"],
                data_root,
                layout,
                "2011_09_26_drive_0002_sync_02",
                "02",
            )
            self.assertEqual([path.name for path in image_files], ["2011_09_26_drive_0002_sync_image_0000000000_image_02.png"])
            self.assertEqual([path.name for path in gt_files], ["2011_09_26_drive_0002_sync_groundtruth_depth_0000000000_image_02.png"])

    def test_prepare_kitti_monst3r_gathered_layout_uses_image02_and_caps_frames(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            raw_root = tmp_root / "raw"
            annotated_root = tmp_root / "annotated"
            output_root = tmp_root / "prepared"

            raw_image_dir = raw_root / "2011_09_26" / "2011_09_26_drive_0002_sync" / "image_02" / "data"
            raw_image_dir.mkdir(parents=True, exist_ok=True)
            gt_dir = annotated_root / "val" / "2011_09_26_drive_0002_sync" / "proj_depth" / "groundtruth" / "image_02"
            gt_dir.mkdir(parents=True, exist_ok=True)

            for frame_idx in range(3):
                file_name = f"{frame_idx:010d}.png"
                self._write_rgb_png(raw_image_dir / file_name, color=frame_idx + 1)
                self._write_depth_png(gt_dir / file_name, depth_value=frame_idx + 1)

            summary = kitti_videodepth_utils.prepare_kitti_monst3r_gathered_layout(
                raw_root=raw_root,
                annotated_root=annotated_root,
                output_root=output_root,
                camera="02",
                max_frames_per_seq=2,
            )

            self.assertEqual(summary["num_sequences"], 1)
            self.assertEqual(summary["num_frames"], 2)

            seq_name = "2011_09_26_drive_0002_sync_02"
            prepared_image_dir = output_root / "image_gathered" / seq_name
            prepared_gt_dir = output_root / "groundtruth_depth_gathered" / seq_name
            self.assertEqual(
                [path.name for path in sorted(prepared_image_dir.glob("*.png"))],
                ["0000000000.png", "0000000001.png"],
            )
            self.assertEqual(
                [path.name for path in sorted(prepared_gt_dir.glob("*.png"))],
                ["0000000000.png", "0000000001.png"],
            )

    @staticmethod
    def _write_rgb_png(path: Path, color: int = 1) -> None:
        Image.fromarray(np.full((2, 3, 3), fill_value=color, dtype=np.uint8), mode="RGB").save(path)

    @staticmethod
    def _write_depth_png(path: Path, depth_value: int = 1) -> None:
        Image.fromarray(np.full((2, 3), fill_value=depth_value, dtype=np.uint16), mode="I;16").save(path)


if __name__ == "__main__":
    unittest.main()
