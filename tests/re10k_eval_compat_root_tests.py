from __future__ import annotations

import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "vggt"
    / "build_re10k_eval_compat_root.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing compat-root script: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("build_re10k_eval_compat_root", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class Re10kEvalCompatRootTests(unittest.TestCase):
    def test_stage_compat_root_keeps_monocular_layout_and_writes_meta(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "test"
            stage_root = root / "stage"
            scene_dir = source_root / "scene_a"
            image_root = scene_dir / "images" / "00"
            camera_root = scene_dir / "cameras" / "00"
            image_root.mkdir(parents=True)

            names = [f"{idx:06d}" for idx in range(4)]
            write_camera_files(camera_root, names)
            for name in names:
                (image_root / f"{name}.jpg").write_text(f"image {name}\n", encoding="utf-8")
            (source_root / "data_roots.txt").write_text("scene_a\n", encoding="utf-8")

            report = module.build_compat_root(
                source_root=source_root,
                stage_root=stage_root,
                scene_filter="",
                limit_scenes=0,
                copy_images=True,
                copy_workers=2,
                skip_existing=False,
                clear_stage_root=False,
            )

            staged_scene = stage_root / "scene_a"
            staged_images = sorted(path.name for path in (staged_scene / "images" / "00").glob("*.jpg"))

            self.assertEqual(report["scene_count"], 1)
            self.assertEqual((stage_root / "data_roots.txt").read_text(encoding="utf-8").strip(), "scene_a")
            self.assertEqual(staged_images, [f"{name}.jpg" for name in names])
            self.assertTrue((staged_scene / "cameras" / "00" / "intri.yml").is_file())
            self.assertTrue((staged_scene / "cameras" / "00" / "extri.yml").is_file())
            self.assertFalse((staged_scene / "intri.yml").exists())
            self.assertFalse((staged_scene / "extri.yml").exists())
            self.assertTrue((staged_scene / "__meta.pkl").is_file())

            with (staged_scene / "__meta.pkl").open("rb") as handle:
                meta = pickle.load(handle)

            self.assertEqual(meta["n_view_total"], 1)
            self.assertEqual(meta["n_frames_total"], 4)
            self.assertIn("00/intri", meta)
            self.assertIn("00/extri", meta)
            self.assertEqual(list(meta["00/intri"]["names"]), names)
            self.assertEqual(list(meta["00/extri"]["names"]), names)


if __name__ == "__main__":
    unittest.main()
