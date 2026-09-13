from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aidi"
    / "scripts"
    / "vggt"
    / "run_standard_pose_benchmarks.py"
)


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing suite runner: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("run_standard_pose_benchmarks", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VggtStandardPoseBenchmarksTests(unittest.TestCase):
    def test_split_benchmark_groups(self):
        module = load_module()

        da3, relpose = module.split_benchmark_groups(["eth3d", "7scenes", "megadepth1500", "scannet1500"])

        self.assertEqual(da3, ["eth3d", "7scenes"])
        self.assertEqual(relpose, ["megadepth1500", "scannet1500"])

    def test_build_da3_command_forwards_roots(self):
        module = load_module()
        args, passthrough = module.parse_args(
            [
                "--eth3d-root",
                "/tmp/eth3d",
                "--7scenes-root",
                "/tmp/7s",
                "--dtu64-camera-root",
                "/tmp/cams",
                "--model-family",
                "pi3",
            ]
        )

        cmd = module.build_da3_command(args=args, passthrough_args=passthrough, output_root=Path("/tmp/out"), datasets=["7scenes"])

        self.assertIn("--eth3d-root", cmd)
        self.assertIn("/tmp/eth3d", cmd)
        self.assertIn("--7scenes-root", cmd)
        self.assertIn("/tmp/7s", cmd)
        self.assertIn("--dtu64-camera-root", cmd)
        self.assertIn("/tmp/cams", cmd)
        self.assertIn("--model-family", cmd)
        self.assertIn("pi3", cmd)

    def test_build_relpose_command_uses_dataset_specific_roots(self):
        module = load_module()
        args, passthrough = module.parse_args(
            [
                "--megadepth1500-root",
                "/tmp/megadepth",
                "--megadepth1500-manifest-root",
                "/tmp/manifest",
                "--device",
                "cuda:0",
            ]
        )

        cmd = module.build_relpose_command(
            args=args,
            passthrough_args=passthrough,
            output_root=Path("/tmp/out"),
            dataset_name="megadepth1500",
        )

        self.assertIn("--dataset-root", cmd)
        self.assertIn("/tmp/megadepth", cmd)
        self.assertIn("--manifest-root", cmd)
        self.assertIn("/tmp/manifest", cmd)
        self.assertIn("--device", cmd)
        self.assertIn("cuda:0", cmd)

    def test_load_and_render_summary_artifacts(self):
        module = load_module()

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            da3_root = root / "da3_pose" / "metric_results"
            relpose_root = root / "relpose1500" / "metric_results"
            da3_root.mkdir(parents=True)
            relpose_root.mkdir(parents=True)

            (da3_root / "summary.json").write_text(
                json.dumps(
                    {
                        "datasets": {
                            "eth3d": {"Auc3": 0.5, "Auc30": 0.9},
                            "7scenes": {"Auc3": 0.4, "Auc30": 0.8},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (relpose_root / "megadepth1500_pose.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "AUC@5": 0.1,
                            "AUC@10": 0.2,
                            "AUC@20": 0.3,
                            "rotation_error_median": 1.5,
                            "translation_error_median": 2.5,
                        }
                    }
                ),
                encoding="utf-8",
            )

            requested = ["eth3d", "7scenes", "megadepth1500"]
            summaries = module.load_benchmark_summaries(output_root=root, benchmarks=requested)
            markdown = module.render_summary_markdown(summaries=summaries, requested=requested)
            module.write_summary_artifacts(output_root=root, requested=requested, summaries=summaries)

            payload = json.loads((root / "metric_results" / "summary.json").read_text(encoding="utf-8"))

        self.assertIn("## DA3 Pose", markdown)
        self.assertIn("## Relative Pose 1500", markdown)
        self.assertEqual(payload["requested_benchmarks"], requested)
        self.assertIn("eth3d", payload["benchmarks"])
        self.assertIn("7scenes", payload["benchmarks"])
        self.assertIn("megadepth1500", payload["benchmarks"])


if __name__ == "__main__":
    unittest.main()
