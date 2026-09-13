from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PI3_ROOT = REPO_ROOT / "aidi" / "third_party" / "pi3_training"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PI3_ROOT))

from aidi.utils.vggt_core4_main_val import _stable_int_seed  # noqa: E402
from datasets.co3dv2_core4_loop_dataset import CO3DV2Core4LoopDataset  # noqa: E402


def _write_jgz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(json.dumps(payload).encode("utf-8"))


def _write_depth(path: Path, value: float = 2.0, size: tuple[int, int] = (64, 64)) -> None:
    depth = np.full((size[1], size[0]), value, dtype=np.float16).view(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth).save(path)


def _build_fake_co3d_root(root: Path, frame_count: int = 12) -> None:
    category = "apple"
    seq = "seq001"
    seq_root = root / category / seq
    (root / category / "set_lists").mkdir(parents=True, exist_ok=True)
    (seq_root / "images").mkdir(parents=True, exist_ok=True)

    set_list_rows = []
    frame_annotations = []
    for frame_idx in range(frame_count):
        filename = f"frame{frame_idx:06d}.jpg"
        filepath = f"{category}/{seq}/images/{filename}"
        set_list_rows.append([seq, frame_idx, filepath])

        image = np.full((64, 64, 3), 128 + frame_idx, dtype=np.uint8)
        Image.fromarray(image).save(seq_root / "images" / filename)
        _write_depth(seq_root / "depths" / f"{filename}.geometric.png")
        mask = np.full((64, 64), 255, dtype=np.uint8)
        (seq_root / "masks").mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(seq_root / "masks" / filename.replace(".jpg", ".png"))

        frame_annotations.append(
            {
                "sequence_name": seq,
                "frame_number": frame_idx,
                "image": {"path": filepath, "size": [64, 64]},
                "viewpoint": {
                    "R": np.eye(3, dtype=np.float32).tolist(),
                    "T": [0.0, 0.0, 2.0 + 0.01 * frame_idx],
                    "focal_length": [2.0, 2.0],
                    "principal_point": [0.0, 0.0],
                },
            }
        )

    (root / category / "set_lists" / "set_lists_fewview_dev.json").write_text(
        json.dumps({"train": [], "val": [], "test": set_list_rows}),
        encoding="utf-8",
    )
    _write_jgz(root / category / "frame_annotations.jgz", frame_annotations)
    _write_jgz(
        root / category / "sequence_annotations.jgz",
        [{"sequence_name": seq, "viewpoint_quality_score": 1.0}],
    )


class CO3DV2Core4LoopDatasetTests(unittest.TestCase):
    def test_dataset_uses_core4_seeded_frame_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _build_fake_co3d_root(root)

            dataset = CO3DV2Core4LoopDataset(
                data_root=str(root),
                categories="apple",
                resolution=[[64, 64]],
                frame_num=10,
                num_frames=10,
                min_num_images=10,
                sample_stride=1,
                max_sequences=1,
                core4_seed=0,
                mask_bg=False,
                aug_crop=False,
                aug_focal=False,
                shuffle=False,
            )

            self.assertEqual(len(dataset), 1)
            sample = dataset.samples[0]
            self.assertEqual(sample["path"], "apple/seq001")
            expected_seed = _stable_int_seed(0, "apple", "seq001")
            self.assertEqual(sample["selection_seed"], expected_seed)

            expected_ids = np.random.RandomState(expected_seed).choice(12, 10, replace=False)
            expected_frame_numbers = [int(idx) for idx in expected_ids]
            actual_frame_numbers = [frame["frame_number"] for frame in sample["frames"]]
            self.assertEqual(actual_frame_numbers, expected_frame_numbers)

            views = dataset[0]
            self.assertEqual(len(views), 10)
            self.assertTrue(all(view["img"].shape[-2:] == (64, 64) for view in views))
            self.assertTrue(all(view["valid_mask"].any() for view in views))
            self.assertTrue(all(view["dataset"] == "CO3Dv2" for view in views))


if __name__ == "__main__":
    unittest.main()
