import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPARSE_SCRIPT = REPO_ROOT / "aidi/scripts/vggt/submit_official_vggt_5090_sparse.sh"
COMPARE_V3_SCRIPT = (
    REPO_ROOT
    / "aidi/scripts/vggt/submit_official_vggt_5090_sparse_topk1024_full17_dataset_compare_v3.sh"
)
GLOBALFRAME_HYPERSIMVAL_SCRIPT = (
    REPO_ROOT
    / "aidi/scripts/vggt/submit_official_vggt_5090_sparse_topk1024_globalframe0_8_hypersimval.sh"
)
GLOBALFRAME_CORE4VAL_SCRIPT = (
    REPO_ROOT
    / "aidi/scripts/vggt/submit_official_vggt_5090_sparse_topk1024_globalframe0_8_core4val.sh"
)


def run_script_with_fake_bash(script_path: Path, extra_env: dict[str, str]) -> tuple[str, str]:
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
        env.update(extra_env)
        subprocess.run(
            ["/bin/bash", str(script_path)],
            cwd=REPO_ROOT,
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return (
            capture_env.read_text(encoding="utf-8"),
            capture_args.read_text(encoding="utf-8"),
        )


class VggtSparseLrWarmupSubmitTests(unittest.TestCase):
    def test_sparse_submit_supports_configurable_lr_warmup(self):
        captured_env, captured_args = run_script_with_fake_bash(
            SPARSE_SCRIPT,
            {
                "SPARSE_LR": "2.0e-5",
                "SPARSE_LR_WARMUP_START": "2.0e-6",
            },
        )

        self.assertIn("aidi/scripts/vggt/train_official_vggt.sh", captured_args)
        self.assertIn("runner_cfg.optimizer_cfg.optimizer_cfg.lr=2.0e-5", captured_env)
        self.assertIn(
            "runner_cfg.scheduler_cfg.option_schedulers.lr.0.scheduler.schedulers.0.start_value=2.0e-6",
            captured_env,
        )
        self.assertIn(
            "runner_cfg.scheduler_cfg.option_schedulers.lr.0.scheduler.schedulers.0.end_value=2.0e-5",
            captured_env,
        )
        self.assertIn(
            "runner_cfg.scheduler_cfg.option_schedulers.lr.0.scheduler.schedulers.1.start_value=2.0e-5",
            captured_env,
        )

    def test_compare_v3_submit_sets_full17_queue_and_lr_defaults(self):
        captured_env, captured_args = run_script_with_fake_bash(COMPARE_V3_SCRIPT, {})

        self.assertIn("aidi/scripts/vggt/submit_official_vggt_5090_sparse_fp16_layers9_19_topk2048_pointhead.sh", captured_args)
        self.assertIn("CLUSTER=project-5090-4dlabel-perception-v2-acloud-langfang", captured_env)
        self.assertIn("NUM_NODES=2", captured_env)
        self.assertIn("GPUS_PER_NODE=8", captured_env)
        self.assertIn("TOPK=1024", captured_env)
        self.assertIn("DISABLE_WARMUP_MATCH_DATASET_OVERRIDES=1", captured_env)
        self.assertIn("SPARSE_LR=2.0e-5", captured_env)
        self.assertIn("SPARSE_LR_WARMUP_START=2.0e-6", captured_env)
        self.assertIn("JOB_NAME=5090x16_vggt_official_5090_sparse_20260417_topk1024_full17_dataset_compare_v3_lr2e5_warmup_v1", captured_env)
        self.assertIn(
            "EXP_NAME=vggt/official/finetune_5090_sparse_20260417_topk1024_full17_dataset_compare_v3_lr2e5_warmup_v1",
            captured_env,
        )
        self.assertIn(
            "EXTRA_OVERRIDES=runner_cfg.epochs=80;model_cfg.vggt_cfg.depth_head_cfg.chunk_cfg.frames_chunk_size=2;model_cfg.vggt_cfg.point_head_cfg.chunk_cfg.frames_chunk_size=2",
            captured_env,
        )

    def test_globalframe_hypersimval_submit_sets_attention_scope_and_val(self):
        captured_env, captured_args = run_script_with_fake_bash(GLOBALFRAME_HYPERSIMVAL_SCRIPT, {})

        self.assertIn("aidi/scripts/vggt/submit_official_vggt_5090_sparse_fp16_layers9_19_topk2048_pointhead.sh", captured_args)
        self.assertIn("CLUSTER=project-5090-4dlabel-perception-v2-acloud-langfang", captured_env)
        self.assertIn("NUM_NODES=2", captured_env)
        self.assertIn("GPUS_PER_NODE=8", captured_env)
        self.assertIn("TOPK=1024", captured_env)
        self.assertIn("INDEXER_LAYERS=9-19", captured_env)
        self.assertIn("EVAL_EP=1", captured_env)
        self.assertIn("DISABLE_WARMUP_MATCH_DATASET_OVERRIDES=1", captured_env)
        self.assertIn(
            "JOB_NAME=vggt_official_5090_sparse_20260423_resumept30_oldlogic_keepdecay_ep80_2x8_chunk2_topk1024_globalframe0_8_hypersimval_v1",
            captured_env,
        )
        self.assertIn(
            "EXP_NAME=vggt/official/finetune_5090_sparse_20260423_resumept30_oldlogic_keepdecay_ep80_2x8_chunk2_topk1024_globalframe0_8_hypersimval_v1",
            captured_env,
        )
        self.assertIn("model_cfg.vggt_cfg.aggregator_cfg.global_frame_attention_layers=0-8", captured_env)
        self.assertIn("runner_cfg.main_val_core4_cfg=[]", captured_env)
        self.assertIn(
            "runner_cfg.main_val_cfgs=[configs/exps/vggt/evaluation/hypersim_train.yaml,configs/exps/vggt/evaluation/hypersim.yaml]",
            captured_env,
        )
        self.assertIn("runner_cfg.test_before_first_epoch=True", captured_env)
        self.assertIn("runner_cfg.epochs=80", captured_env)
        self.assertIn("model_cfg.vggt_cfg.depth_head_cfg.chunk_cfg.frames_chunk_size=2", captured_env)
        self.assertIn("model_cfg.vggt_cfg.point_head_cfg.chunk_cfg.frames_chunk_size=2", captured_env)
        self.assertIn("dataloader_cfg.dataset_cfg.metaset_cfgs.5.prob=0.0", captured_env)
        self.assertIn("dataloader_cfg.dataset_cfg.metaset_cfgs.10.prob=0.0", captured_env)
        self.assertIn("dataloader_cfg.dataset_cfg.metaset_cfgs.11.prob=0.0", captured_env)
        self.assertIn("dataloader_cfg.dataset_cfg.metaset_cfgs.13.prob=0.0", captured_env)
        self.assertNotIn("val_dataloader_cfg.dataset_cfg.metaset_cfgs.5.prob=0.0", captured_env)
        self.assertIn(
            "EXTRA_OVERRIDES=runner_cfg.epochs=80;model_cfg.vggt_cfg.depth_head_cfg.chunk_cfg.frames_chunk_size=2;model_cfg.vggt_cfg.point_head_cfg.chunk_cfg.frames_chunk_size=2;model_cfg.vggt_cfg.aggregator_cfg.global_frame_attention_layers=0-8",
            captured_env,
        )

    def test_globalframe_core4val_submit_uses_base_core4_val_logic(self):
        captured_env, captured_args = run_script_with_fake_bash(GLOBALFRAME_CORE4VAL_SCRIPT, {})

        self.assertIn("aidi/scripts/vggt/submit_official_vggt_5090_sparse_fp16_layers9_19_topk2048_pointhead.sh", captured_args)
        self.assertIn("CLUSTER=project-5090-4dlabel-perception-v2-acloud-langfang", captured_env)
        self.assertIn("NUM_NODES=2", captured_env)
        self.assertIn("GPUS_PER_NODE=8", captured_env)
        self.assertIn("TOPK=1024", captured_env)
        self.assertIn("INDEXER_LAYERS=9-19", captured_env)
        self.assertIn("EVAL_EP=1", captured_env)
        self.assertIn("DISABLE_WARMUP_MATCH_DATASET_OVERRIDES=1", captured_env)
        self.assertIn(
            "JOB_NAME=vggt_official_5090_sparse_20260423_resumept30_oldlogic_keepdecay_ep80_2x8_chunk2_topk1024_globalframe0_8_core4val_v1",
            captured_env,
        )
        self.assertIn(
            "EXP_NAME=vggt/official/finetune_5090_sparse_20260423_resumept30_oldlogic_keepdecay_ep80_2x8_chunk2_topk1024_globalframe0_8_core4val_v1",
            captured_env,
        )
        self.assertIn("model_cfg.vggt_cfg.aggregator_cfg.global_frame_attention_layers=0-8", captured_env)
        self.assertIn("runner_cfg.test_before_first_epoch=True", captured_env)
        self.assertIn("dataloader_cfg.dataset_cfg.metaset_cfgs.5.prob=0.0", captured_env)
        self.assertNotIn("runner_cfg.main_val_core4_cfg=[]", captured_env)
        self.assertNotIn("runner_cfg.main_val_cfgs=[configs/exps/vggt/evaluation/hypersim_train.yaml", captured_env)
        self.assertNotIn("val_dataloader_cfg.dataset_cfg.metaset_cfgs.5.prob=0.0", captured_env)


if __name__ == "__main__":
    unittest.main()
