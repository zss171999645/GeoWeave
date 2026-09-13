import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from aidi.scripts.baselines.eval_config_utils import load_resolved_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/exps/vggt/vggt_official_finetune_5090.yaml"
SCRIPT_PATH = REPO_ROOT / "aidi/scripts/vggt/submit_official_vggt_5090_sparse.sh"


class VggtOfficialFinetune5090ValidationTests(unittest.TestCase):
    def test_official_finetune_5090_validation_matches_origin_hypersim_test(self):
        cfg = load_resolved_config(CONFIG_PATH)

        val_cfg = cfg["val_dataloader_cfg"]
        dataset_cfg = val_cfg["dataset_cfg"]
        sampler_cfg = val_cfg["sampler_cfg"]

        self.assertEqual(dataset_cfg["type"], "GeneralizableDataset")
        self.assertEqual(dataset_cfg["split"], "VAL")
        self.assertEqual(
            dataset_cfg["meta_roots"],
            ["/horizon-bucket/saturn_v_dev/users/tao02.xie/datasets/hypersim/test/"],
        )
        self.assertEqual(dataset_cfg["metaset_cfgs"][0]["dataset_name"], "hypersim/test")
        self.assertEqual(dataset_cfg["metaset_cfgs"][0]["max_depth_quantile"], 0.98)

        point_cfg = dataset_cfg["metaset_cfgs"][-1]
        self.assertEqual(point_cfg["type"], "MultiviewPointDataset")
        self.assertEqual(point_cfg["split"], "VAL")
        self.assertEqual(point_cfg["extra_src_pool"], 0)

        self.assertEqual(sampler_cfg["frame_sample"], [0, None, 10])
        self.assertEqual(sampler_cfg["view_sample"], [0, None, 1])

    def test_sparse_submit_enables_epoch_validation_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fakebin = tmpdir / "fakebin"
            fakebin.mkdir()
            capture_env = tmpdir / "captured.env"
            capture_args = tmpdir / "captured.args"
            fake_bash = fakebin / "bash"
            fake_bash.write_text(
                "\n".join(
                    [
                        "#!/bin/sh",
                        f"env | sort > {capture_env}",
                        f"printf '%s\\n' \"$@\" > {capture_args}",
                        "exit 0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_bash.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fakebin}:{env['PATH']}",
                    "MODE": "remote",
                    "USER": "feng01.zhou",
                }
            )
            subprocess.run(
                ["/bin/bash", str(SCRIPT_PATH)],
                cwd=REPO_ROOT,
                check=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            captured_env = capture_env.read_text(encoding="utf-8")
            captured_args = capture_args.read_text(encoding="utf-8")

            self.assertIn("aidi/scripts/vggt/train_official_vggt.sh", captured_args)
            self.assertIn("EXTRA_OVERRIDES=", captured_env)
            self.assertIn("runner_cfg.eval_ep=1", captured_env)


if __name__ == "__main__":
    unittest.main()
