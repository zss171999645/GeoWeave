from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "baselines"
    / "stage_re10k_aligned_subset.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing staging script: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("stage_re10k_aligned_subset", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_camera_names_yaml(path: Path, names: list[str]) -> None:
    lines = ["%YAML:1.0", "---", "names:"]
    lines.extend([f'- "{name}"' for name in names])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_camera_files(camera_root: Path, names: list[str]) -> None:
    camera_root.mkdir(parents=True, exist_ok=True)

    intri_lines = ["%YAML:1.0", "---", "names:"]
    extri_lines = ["%YAML:1.0", "---", "names:"]
    intri_lines.extend([f'  - "{name}"' for name in names])
    extri_lines.extend([f'  - "{name}"' for name in names])

    for idx, name in enumerate(names):
        intri_lines.extend(
            [
                f"K_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 3",
                "  dt: d",
                "  data: [1., 0., 0., 0., 1., 0., 0., 0., 1.]",
                f"H_{name}: 480.",
                f"W_{name}: 640.",
                f"D_{name}: !!opencv-matrix",
                "  rows: 5",
                "  cols: 1",
                "  dt: d",
                "  data: [0., 0., 0., 0., 0.]",
            ]
        )
        extri_lines.extend(
            [
                f"R_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 1",
                "  dt: d",
                "  data: [0., 0., 0.]",
                f"Rot_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 3",
                "  dt: d",
                "  data: [1., 0., 0., 0., 1., 0., 0., 0., 1.]",
                f"T_{name}: !!opencv-matrix",
                "  rows: 3",
                "  cols: 1",
                "  dt: d",
                f"  data: [0., 0., {float(idx):.1f}]",
                f"t_{name}: {float(idx):.1f}",
                f"n_{name}: 0.1",
                f"f_{name}: 1000.0",
            ]
        )

    (camera_root / "intri.yml").write_text("\n".join(intri_lines) + "\n", encoding="utf-8")
    (camera_root / "extri.yml").write_text("\n".join(extri_lines) + "\n", encoding="utf-8")


class Pi3Re10kStageTests(unittest.TestCase):
    def test_parse_camera_names_reads_names_block(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            extri = Path(tmpdir) / "extri.yml"
            write_camera_names_yaml(extri, ["000000", "000005", "000010"])

            names = module.parse_camera_names(extri)

        self.assertEqual(names, ["000000", "000005", "000010"])

    def test_stage_dataset_copies_only_aligned_scene_subset(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "test"
            stage_root = root / "stage"
            scene_dir = source_root / "scene_a"
            image_root = scene_dir / "images" / "00"
            camera_root = scene_dir / "cameras" / "00"
            image_root.mkdir(parents=True)

            names = [f"{idx:06d}" for idx in range(12)]
            write_camera_files(camera_root, names)
            for name in names:
                (image_root / f"{name}.jpg").write_text(f"image {name}\n", encoding="utf-8")
            (source_root / "data_roots.txt").write_text("scene_a\n", encoding="utf-8")

            report = module.stage_dataset(
                source_root=source_root,
                stage_root=stage_root,
                seed=20260215,
                pool_size=10,
                n_srcs=9,
                scene_filter="",
                limit_scenes=0,
                copy_workers=1,
                skip_existing=False,
            )

            expected_indices = module.sample_scene_pool_indices(
                scene_key="test/scene_a",
                total_frames=len(names),
                pool_size=10,
                seed=20260215,
            )
            expected_source_names = [names[idx] for idx in expected_indices[:10]]
            expected_staged_names = [f"{idx:06d}" for idx in range(len(expected_source_names))]
            staged_images = sorted(path.stem for path in (stage_root / "scene_a" / "images" / "00").glob("*.jpg"))
            manifest = report["manifest"]
            staged_extri_names = module.parse_camera_names(stage_root / "scene_a" / "cameras" / "00" / "extri.yml")
            staged_intri_names = module.parse_camera_names(stage_root / "scene_a" / "cameras" / "00" / "intri.yml")

            self.assertEqual(report["scene_count"], 1)
            self.assertEqual((stage_root / "data_roots.txt").read_text(encoding="utf-8").strip(), "scene_a")
            self.assertTrue((stage_root / "scene_a" / "cameras" / "00" / "intri.yml").is_file())
            self.assertTrue((stage_root / "scene_a" / "cameras" / "00" / "extri.yml").is_file())
            self.assertTrue((stage_root / "scene_a" / "intri.yml").is_file())
            self.assertTrue((stage_root / "scene_a" / "extri.yml").is_file())
            self.assertEqual(
                (stage_root / "scene_a" / "intri.yml").read_text(encoding="utf-8"),
                (stage_root / "scene_a" / "cameras" / "00" / "intri.yml").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (stage_root / "scene_a" / "extri.yml").read_text(encoding="utf-8"),
                (stage_root / "scene_a" / "cameras" / "00" / "extri.yml").read_text(encoding="utf-8"),
            )
            self.assertEqual(staged_images, expected_staged_names)
            self.assertEqual(staged_extri_names, expected_staged_names)
            self.assertEqual(staged_intri_names, expected_staged_names)
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["scene_key"], "test/scene_a")
            self.assertEqual(manifest[0]["sampled_source_names"], expected_source_names)
            self.assertEqual(manifest[0]["sampled_names"], expected_staged_names)


if __name__ == "__main__":
    unittest.main()
