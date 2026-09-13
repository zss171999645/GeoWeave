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
    / "reorder_re10k_staged_tuples.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("reorder_re10k_staged_tuples", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_camera_yaml(path: Path, names: list[str]) -> None:
    lines = ["%YAML:1.0", "---", "names:"]
    lines.extend([f'  - "{name}"' for name in names])
    for name in names:
        lines.extend(
            [
                f"K_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 3",
                "  dt: d",
                "  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]",
                f"RT_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 4",
                "  dt: d",
                "  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tuple(root: Path, total_views: int = 10, core_size: int = 6) -> None:
    names = [f"{index:06d}" for index in range(total_views)]
    write_camera_yaml(root / "cameras" / "00" / "intri.yml", names)
    write_camera_yaml(root / "cameras" / "00" / "extri.yml", names)
    image_dir = root / "images" / "00"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (image_dir / f"{name}.jpg").write_bytes(f"image-{name}".encode("utf-8"))
    meta = {
        "tuple_scene_name": root.name,
        "pool_size": total_views,
        "core_size": core_size,
        "staged_names": names,
        "eval_frame_indices": list(range(core_size)),
        "frames": [
            {"output_index": index, "output_name": name, "role": "clean_core" if index < core_size else "candidate"}
            for index, name in enumerate(names)
        ],
    }
    (root / "tuple_meta.json").write_text(json.dumps(meta), encoding="utf-8")


class ReorderRe10KStagedTuplesTests(unittest.TestCase):
    def test_candidate_first_moves_eval_indices_to_tail(self):
        module = load_module()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "input" / "scene__pool10"
            write_tuple(source, total_views=10, core_size=6)
            summary = module.build_dataset(
                input_root=tmp / "input",
                output_root=tmp / "output",
                source_pool_size=10,
                order_mode="candidate_first",
                image_stage_mode="copy",
            )
            out = Path(summary["tuples"][0]["tuple_root"])
            meta = json.loads((out / "tuple_meta.json").read_text(encoding="utf-8"))

            self.assertEqual(meta["source_order_indices"], [6, 7, 8, 9, 0, 1, 2, 3, 4, 5])
            self.assertEqual(meta["eval_frame_indices"], [4, 5, 6, 7, 8, 9])
            self.assertEqual(meta["frames"][0]["old_output_index"], 6)
            self.assertEqual((out / "images" / "00" / "000000.jpg").read_bytes(), b"image-000006")

    def test_interleave_records_all_core_positions(self):
        module = load_module()
        order = module.compute_order(total_views=10, core_size=6, order_mode="interleave", rng=__import__("random").Random(0))
        eval_indices = module.eval_indices_for_order(order, core_size=6)

        self.assertEqual(sorted(order), list(range(10)))
        self.assertEqual(len(eval_indices), 6)
        self.assertTrue(all(order[index] < 6 for index in eval_indices))


if __name__ == "__main__":
    unittest.main()
