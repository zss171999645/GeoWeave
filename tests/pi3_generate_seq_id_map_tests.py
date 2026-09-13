import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "aidi/scripts/baselines/generate_seq_id_map.py"


class GenerateSeqIdMapTests(unittest.TestCase):
    def test_generates_7scenes_seq_map_from_raw_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "7Scenes"
            seq = root / "chess" / "seq-01"
            seq.mkdir(parents=True)
            for frame_id in (0, 40, 80):
                (seq / f"frame-{frame_id:06d}.color.png").write_bytes(b"")
            output = Path(tmpdir) / "7scenes_map.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dataset-root",
                    str(root),
                    "--dataset",
                    "7scenes",
                    "--kf-step",
                    "2",
                    "--output",
                    str(output),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            payload = json.loads(output.read_text())
            self.assertEqual(payload, {"chess_seq-01": [0, 80]})

    def test_generates_nrgbd_seq_map_from_flat_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "NRGBD"
            seq = root / "scene_01"
            images = seq / "images"
            images.mkdir(parents=True)
            for frame_id in (0, 100, 200):
                (images / f"img{frame_id}.png").write_bytes(b"")
            output = Path(tmpdir) / "nrgbd_map.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dataset-root",
                    str(root),
                    "--dataset",
                    "nrgbd",
                    "--kf-step",
                    "2",
                    "--output",
                    str(output),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            payload = json.loads(output.read_text())
            self.assertEqual(payload, {"scene_01": [0, 200]})


if __name__ == "__main__":
    unittest.main()
