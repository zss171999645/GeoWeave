from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "baselines"
    / "build_variable_distractor_from_pairs.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("build_variable_distractor_from_pairs", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_tuple(root: Path, pose_prefix: str, meta: dict) -> None:
    color = root / "color_90"
    color.mkdir(parents=True)
    rows = []
    for index in range(10):
        (color / f"frame_{index:04d}.jpg").write_bytes(f"{pose_prefix}-img-{index}".encode("utf-8"))
        rows.append(" ".join([f"{pose_prefix}{index}"] * 16))
    (root / "pose_90.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "tuple_meta.json").write_text(json.dumps(meta), encoding="utf-8")


class VariableDistractorFromPairsTests(unittest.TestCase):
    def test_live_suffix_pair_preserves_clean_metadata(self):
        module = load_module()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "input" / "waymo"
            clean = input_root / "scene_a__clean_tail"
            noise = input_root / "scene_a__plausible_noise_tail"
            write_tuple(
                clean,
                "c",
                {
                    "prefix_scene": "scene-a",
                    "prefix_camera": "03",
                    "prefix_frame_names": [0, 1, 2, 3, 4, 5],
                },
            )
            write_tuple(noise, "n", {"distractor_scene": "scene-b"})
            output_root = tmp / "output"

            pairs = module.paired_clean_noise(input_root)
            summary = module.build_dataset(
                input_root=tmp / "input",
                output_root=output_root,
                dataset="waymo",
                variants=[2],
                source_total_views=10,
                eval_views=6,
                output_total_mode="fixed",
                clean_fill_mode="tail_clean",
                copy_images=True,
            )
            result_meta = json.loads(
                (output_root / "waymo/noise2/scene_a__noise2/tuple_meta.json").read_text(encoding="utf-8")
            )

        self.assertEqual(pairs, [(clean, noise)])
        self.assertEqual(summary["counts"], {"noise2": 1})
        self.assertEqual(result_meta["prefix_scene"], "scene-a")
        self.assertEqual(result_meta["prefix_camera"], "03")
        self.assertEqual(result_meta["source_labels"], ["clean"] * 8 + ["distractor"] * 2)

    def test_materialize_variant_replaces_only_tail_frames(self):
        module = load_module()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            clean = tmp / "sample__clean"
            noise = tmp / "sample__noise"
            write_tuple(clean, "c", {"tuple_kind": "clean"})
            write_tuple(noise, "n", {"tuple_kind": "noise"})

            out = tmp / "out" / "sample__noise2"
            module.materialize_variant(
                output_tuple=out,
                clean_tuple=clean,
                noise_tuple=noise,
                distractor_count=2,
                source_total_views=10,
                eval_views=6,
                output_total_mode="fixed",
                clean_fill_mode="tail_clean",
                copy_images=True,
                metadata={"dataset": "toy"},
            )
            poses = (out / "pose_90.txt").read_text(encoding="utf-8").splitlines()
            meta = json.loads((out / "tuple_meta.json").read_text(encoding="utf-8"))

        self.assertTrue(poses[0].startswith("c0 "))
        self.assertTrue(poses[7].startswith("c7 "))
        self.assertTrue(poses[8].startswith("n8 "))
        self.assertTrue(poses[9].startswith("n9 "))
        self.assertEqual(meta["eval_frame_indices"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(meta["source_labels"], ["clean"] * 8 + ["distractor"] * 2)

    def test_repeat_last_eval_fill_mode_keeps_tail_neutral_until_distractor_slots(self):
        module = load_module()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            clean = tmp / "sample__clean"
            noise = tmp / "sample__noise"
            write_tuple(clean, "c", {"tuple_kind": "clean"})
            write_tuple(noise, "n", {"tuple_kind": "noise"})

            out = tmp / "out" / "sample__noise2"
            module.materialize_variant(
                output_tuple=out,
                clean_tuple=clean,
                noise_tuple=noise,
                distractor_count=2,
                source_total_views=10,
                eval_views=6,
                output_total_mode="fixed",
                clean_fill_mode="repeat_last_eval",
                copy_images=True,
                metadata={"dataset": "toy"},
            )
            poses = (out / "pose_90.txt").read_text(encoding="utf-8").splitlines()
            meta = json.loads((out / "tuple_meta.json").read_text(encoding="utf-8"))

            self.assertTrue(poses[0].startswith("c0 "))
            self.assertTrue(poses[6].startswith("c5 "))
            self.assertTrue(poses[7].startswith("c5 "))
            self.assertTrue(poses[8].startswith("n8 "))
            self.assertTrue(poses[9].startswith("n9 "))
            self.assertEqual(
                meta["source_labels"],
                ["clean"] * 6 + ["clean_repeat_last_eval"] * 2 + ["distractor"] * 2,
            )
            self.assertEqual(meta["clean_fill_mode"], "repeat_last_eval")

    def test_eval_plus_distractors_outputs_variable_length_without_fillers(self):
        module = load_module()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            clean = tmp / "sample__clean"
            noise = tmp / "sample__noise"
            write_tuple(clean, "c", {"tuple_kind": "clean"})
            write_tuple(noise, "n", {"tuple_kind": "noise"})

            out = tmp / "out" / "sample__noise2"
            module.materialize_variant(
                output_tuple=out,
                clean_tuple=clean,
                noise_tuple=noise,
                distractor_count=2,
                source_total_views=10,
                eval_views=6,
                output_total_mode="eval_plus_distractors",
                clean_fill_mode="tail_clean",
                copy_images=True,
                metadata={"dataset": "toy"},
            )
            poses = (out / "pose_90.txt").read_text(encoding="utf-8").splitlines()
            meta = json.loads((out / "tuple_meta.json").read_text(encoding="utf-8"))

            self.assertEqual(len(poses), 8)
            self.assertTrue(poses[0].startswith("c0 "))
            self.assertTrue(poses[5].startswith("c5 "))
            self.assertTrue(poses[6].startswith("n6 "))
            self.assertTrue(poses[7].startswith("n7 "))
            self.assertEqual(meta["total_views"], 8)
            self.assertEqual(meta["output_total_mode"], "eval_plus_distractors")
            self.assertEqual(meta["source_labels"], ["clean"] * 6 + ["distractor"] * 2)

    def test_build_dataset_creates_variant_roots_from_paired_tuples(self):
        module = load_module()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "input" / "waymo"
            write_tuple(input_root / "scene_a__clean", "c", {"target_scene_name": "a"})
            write_tuple(input_root / "scene_a__noise", "n", {"distractor_scene_name": "b"})
            output_root = tmp / "output"

            summary = module.build_dataset(
                input_root=tmp / "input",
                output_root=output_root,
                dataset="waymo",
                variants=[0, 2, 4],
                source_total_views=10,
                eval_views=6,
                output_total_mode="fixed",
                clean_fill_mode="tail_clean",
                copy_images=True,
            )

            self.assertEqual(summary["counts"], {"noise0": 1, "noise2": 1, "noise4": 1})
            self.assertTrue((output_root / "waymo" / "noise0" / "scene_a__noise0" / "pose_90.txt").is_file())
            self.assertTrue((output_root / "waymo" / "noise2" / "scene_a__noise2" / "tuple_meta.json").is_file())
            self.assertTrue((output_root / "waymo" / "noise4" / "scene_a__noise4" / "color_90" / "frame_0009.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
