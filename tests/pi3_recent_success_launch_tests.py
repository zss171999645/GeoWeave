from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DOC = ROOT / "aidi/docs/pi3_native_framework_boundary.md"


def read_repo_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


class Pi3RecentSuccessLaunchTests(unittest.TestCase):
    def assertContainsAll(self, source: str, needles: list[str]) -> None:
        missing = [needle for needle in needles if needle not in source]
        self.assertFalse(missing, f"missing expected launch invariants: {missing}")

    def test_full17_highres_launcher_keeps_recent_success_invariants(self):
        wrapper = read_repo_file("aidi/scripts/pi3/submit_pi3_5090_training.sh")
        launcher = read_repo_file("aidi/scripts/pi3/train_pi3_official.sh")

        self.assertContainsAll(
            wrapper,
            [
                "STAGE=${STAGE:-highres}",
                "DATA_CFG=${DATA_CFG:-meshx_pi3_vggt17}",
                "LOAD_VGGT=${LOAD_VGGT:-0}",
                'MODEL_CKPT=${MODEL_CKPT:-"${SAVE_ROOT}/pretrained/pi3/yyfz233_Pi3_model.safetensors"}',
                "SAVE_TO_AIDI=${SAVE_TO_AIDI:-1}",
                "PI3_USE_INDEX_CACHE=${PI3_USE_INDEX_CACHE:-1}",
                "PI3_INDEX_CACHE_WAIT_SEC=${PI3_INDEX_CACHE_WAIT_SEC:-0}",
                "PI3_LAZY_SEQUENCE_INDEX=${PI3_LAZY_SEQUENCE_INDEX:-1}",
                "PI3_DECODER_ATTN_BACKEND=${PI3_DECODER_ATTN_BACKEND:-flash}",
                "PI3_QK_NORM_CHUNK_SIZE=${PI3_QK_NORM_CHUNK_SIZE:-2048}",
                "PI3_HEAD_USE_CHECKPOINT=${PI3_HEAD_USE_CHECKPOINT:-1}",
                "PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT=${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT:-0}",
                "PI3_HEAD_VIEW_CHUNK_SIZE=${PI3_HEAD_VIEW_CHUNK_SIZE:-4}",
                "PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE=${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE:-1}",
            ],
        )
        self.assertContainsAll(
            launcher,
            [
                "append_override_from_env TRAIN_ITERS_PER_EPOCH train.iters_per_epoch",
                "append_override_from_env TRAIN_IMAGE_NUM_RANGE train.image_num_range",
                "append_override_from_env TEST_IMAGE_NUM_RANGE test.image_num_range",
                "append_override_from_env TEST_EVAL_INTERVAL test.eval_interval",
                "append_bool_override_from_env TEST_BEFORE_FIRST_EPOCH test.before_first_epoch",
                "append_override_from_env TRAIN_GRAD_ACCUM_STEPS train.gradient_accumulation_steps",
                "append_override_from_env TRAIN_SCHEDULER_PCT_START train.lr_scheduler.pct_start",
                "append_override_from_env PI3_DECODER_ATTN_BACKEND model.decoder_attn_backend",
                "append_override_from_env PI3_QK_NORM_CHUNK_SIZE model.qk_norm_chunk_size",
                'append_bool_override_from_env PI3_USE_POSE_PRIOR "++model.use_pose_prior"',
                'append_bool_override_from_env PI3_POSE_PRIOR_REQUIRED "++model.pose_prior_required"',
                'append_override_from_env PI3_POSE_PRIOR_DROPOUT "++model.pose_prior_dropout"',
                'append_override_from_env PI3_POSE_PRIOR_FORMAT "++model.pose_prior_format"',
                'append_bool_override_from_env PI3_HEAD_USE_CHECKPOINT "++model.head_use_checkpoint"',
                'append_override_from_env PI3_HEAD_VIEW_CHUNK_SIZE "++model.head_view_chunk_size"',
                "extras=core4_main_val",
                "++main_val_core4_cfg.run_first_eval=True",
                "++log.tensorboard_dir=${PI3_TB_RECORD_DIR}",
                "++log.direct_tensorboard=True",
            ],
        )

    def test_pi3_native_validation_can_match_vggt_business_cadence(self):
        launcher = read_repo_file("aidi/scripts/pi3/train_pi3_official.sh")
        trainer = read_repo_file("aidi/third_party/pi3_training/trainers/base_trainer_accelerate.py")
        model_cfg = read_repo_file("aidi/third_party/pi3_training/configs/model/pi3.yaml")

        self.assertContainsAll(
            launcher,
            [
                "append_override_from_env TEST_IMAGE_NUM_RANGE test.image_num_range",
                "append_override_from_env TEST_EVAL_INTERVAL test.eval_interval",
                "append_bool_override_from_env TEST_BEFORE_FIRST_EPOCH test.before_first_epoch",
            ],
        )
        self.assertContainsAll(
            trainer,
            [
                "def native_validation_first_eval_enabled(self):",
                "def native_validation_epoch_enabled(self, epoch):",
                "self.run_native_validation_before_first_epoch()",
                "self.before_epoch(0)",
                "val_stats = self.validate(-1)",
                "if self.native_validation_epoch_enabled(epoch):",
                "self.log_all(val_stats, step=self.global_step, prefix='val')",
                "forward_outputs = self.forward_batch(batch, mode='test')",
                "outputs = self.calculate_loss(forward_outputs, batch, mode='test')",
                "log_scaler[prefix.upper()+'/'+k] = v",
                "if start_steps % self.cfg.train.print_freq == 0",
                '"TRAIN/lr": max_lr',
            ],
        )
        self.assertContainsAll(
            model_cfg,
            [
                "image_num_range: [8, 8]",
                "eval_interval: 1",
                "before_first_epoch: false",
            ],
        )

    def test_pi3_business_validation_can_emit_vggt_style_tb_metrics(self):
        launcher = read_repo_file("aidi/scripts/pi3/train_pi3_official.sh")
        wrapper = read_repo_file("aidi/scripts/pi3/submit_pi3_5090_training.sh")
        trainer = read_repo_file("aidi/third_party/pi3_training/trainers/base_trainer_accelerate.py")
        model_cfg = read_repo_file("aidi/third_party/pi3_training/configs/model/pi3.yaml")
        metrics = read_repo_file("aidi/third_party/pi3_training/utils/vggt_validation.py")

        self.assertContainsAll(
            launcher,
            [
                "append_bool_override_from_env PI3_VGGT_VAL_METRICS test.vggt_style_metrics.enabled",
                "append_override_from_env PI3_VGGT_VAL_MAX_POINTS test.vggt_style_metrics.max_points_per_view",
            ],
        )
        self.assertContainsAll(
            wrapper,
            [
                "PI3_VGGT_VAL_METRICS=${PI3_VGGT_VAL_METRICS:-0}",
                "PI3_VGGT_VAL_MAX_POINTS=${PI3_VGGT_VAL_MAX_POINTS:-65536}",
                "PI3_NORMAL_LOSS_WEIGHT=${PI3_NORMAL_LOSS_WEIGHT:-1.0}",
            ],
        )
        self.assertContainsAll(
            trainer,
            [
                "from utils.vggt_validation import VggtStylePi3MetricAccumulator",
                "self.vggt_style_metrics_cfg()",
                "vggt_metric_accumulator.update(",
                "val_stats.update(vggt_metric_accumulator.summarize())",
            ],
        )
        self.assertContainsAll(
            model_cfg,
            [
                "vggt_style_metrics:",
                "enabled: false",
                "max_points_per_view: 65536",
            ],
        )
        self.assertContainsAll(
            metrics,
            [
                "class VggtStylePi3MetricAccumulator",
                "def compute_batch_metrics",
                "cam:pose_auc_30",
                "dpt:abs_rel",
                "xyz:rmse",
                "training:loss",
            ],
        )

    def test_pi3_launcher_supports_eval_only_checkpoint_validation(self):
        launcher = read_repo_file("aidi/scripts/pi3/train_pi3_official.sh")
        trainer = read_repo_file("aidi/third_party/pi3_training/trainers/base_trainer_accelerate.py")

        self.assertContainsAll(
            launcher,
            [
                'append_bool_override_from_env PI3_EVAL_ONLY "++eval_only"',
            ],
        )
        self.assertContainsAll(
            trainer,
            [
                "self.eval_only = self._truthy_config_value(cfg.get(\"eval_only\", False))",
                "if self.eval_only:",
                "Eval-only mode finished after native validation.",
            ],
        )

    def test_pi3_normal_loss_is_explicitly_weighted_for_business_alignment(self):
        launcher = read_repo_file("aidi/scripts/pi3/train_pi3_official.sh")
        loss_py = read_repo_file("aidi/third_party/pi3_training/pi3/models/loss.py")
        model_cfg = read_repo_file("aidi/third_party/pi3_training/configs/model/pi3.yaml")

        self.assertContainsAll(
            launcher,
            [
                "append_override_from_env PI3_NORMAL_LOSS_WEIGHT loss.train_loss.normal_loss_weight",
                "append_override_from_env PI3_TEST_NORMAL_LOSS_WEIGHT loss.test_loss.normal_loss_weight",
            ],
        )
        self.assertContainsAll(
            model_cfg,
            [
                "normal_loss_weight: 1.0",
            ],
        )
        self.assertContainsAll(
            loss_py,
            [
                "normal_loss_weight=1.0",
                "self.normal_loss_weight = float(normal_loss_weight)",
                "final_loss += self.normal_loss_weight * normal_loss.mean()",
                "details['normal_loss_weighted']",
                "details['normal_loss_active_ratio']",
            ],
        )

    def test_indexer_warmup_launcher_keeps_recent_success_knobs(self):
        warmup = read_repo_file("aidi/scripts/pi3/submit_pi3_5090_warmup.sh")
        launcher = read_repo_file("aidi/scripts/pi3/train_pi3_official.sh")

        self.assertContainsAll(
            warmup,
            [
                'TRAIN_CFG=${TRAIN_CFG:-"train_pi3_lowres_indexer_warmup"}',
                'INDEXER_LAYERS=${INDEXER_LAYERS:-"all"}',
                "INDEXER_INIT_FROM_ATTN=${INDEXER_INIT_FROM_ATTN:-1}",
                '"model.indexer_cfg.init_from_attn=${INDEXER_INIT_FROM_ATTN}"',
                'echo "[INFO] INDEXER_INIT_FROM_ATTN=${INDEXER_INIT_FROM_ATTN}"',
                "TOPK=${TOPK:-512}",
                "WARMUP_ITERS_PER_EPOCH=${WARMUP_ITERS_PER_EPOCH:-${WARMUP_STEPS}}",
                "WARMUP_ONLY_INDEXER_TRAIN=${WARMUP_ONLY_INDEXER_TRAIN:-True}",
                "TRAIN_DYNAMIC_RES=${TRAIN_DYNAMIC_RES:-0}",
                'if is_truthy "${TRAIN_DYNAMIC_RES}"; then',
                '"++train.random_reslution=True"',
                '"++train.pixel_count_range=${TRAIN_DYNAMIC_PIXEL_COUNT_RANGE}"',
                "PI3_FIND_UNUSED_PARAMETERS=${PI3_FIND_UNUSED_PARAMETERS:-0}",
                "PI3_STATIC_GRAPH=${PI3_STATIC_GRAPH:-1}",
                "INDEXER_SCORE_KEY_CHUNK_SIZE=${INDEXER_SCORE_KEY_CHUNK_SIZE:-1024}",
                'SEQ_NUM_OVERRIDES=("")',
                "STREAMING_KL_LOSS=${STREAMING_KL_LOSS:-${DEFAULT_STREAMING_KL_LOSS}}",
                "STREAMING_KL_AUTOGRAD=${STREAMING_KL_AUTOGRAD:-${STREAMING_KL_LOSS}}",
                'STREAMING_KL_SCORE_MODE=${STREAMING_KL_SCORE_MODE:-"legacy"}',
                "ALLOW_LOW_WARMUP_CLIP_LOSS=${ALLOW_LOW_WARMUP_CLIP_LOSS:-0}",
                "TRAIN_CLIP_LOSS=${TRAIN_CLIP_LOSS} is too low for all-layer KL indexer warm-up",
                "SAVE_TO_AIDI=${SAVE_TO_AIDI:-1}",
            ],
        )
        self.assertContainsAll(
            launcher,
            [
                'INDEXER_INIT_CKPT="${INDEXER_INIT_CKPT:-}"',
                'OVERRIDES+=("model.indexer_init_ckpt=${INDEXER_INIT_CKPT}")',
            ],
        )

    def test_indexer_warmup_launcher_rejects_low_all_layer_kl_clip_loss(self):
        launcher = ROOT / "aidi/scripts/pi3/submit_pi3_5090_warmup.sh"
        with tempfile.TemporaryDirectory() as pi3_root:
            env = os.environ.copy()
            env.update(
                {
                    "PI3_ROOT": pi3_root,
                    "INDEXER_LAYERS": "all",
                    "WARMUP_LOSS_MODE": "kl",
                    "WARMUP_ONLY_INDEXER_TRAIN": "True",
                    "TRAIN_CLIP_LOSS": "10",
                }
            )
            result = subprocess.run(
                ["bash", str(launcher)],
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TRAIN_CLIP_LOSS=10 is too low", result.stdout)
        self.assertIn("zeros the whole loss", result.stdout)

    def test_co3dv2_train_only_data_config_keeps_single_dataset_semantics(self):
        cfg = read_repo_file("aidi/third_party/pi3_training/configs/data/meshx_pi3_co3dv2_train_only.yaml")

        self.assertContainsAll(
            cfg,
            [
                "train only on the MeshX/EVC CO3Dv2 train split",
                "weights:\n        CO3Dv2: 1000",
                "data_root: ${oc.env:CO3DV2_ROOT,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/co3dv2/train}",
                "dataset_label: CO3Dv2",
                "use_masks: true",
                "masks_dir: masks",
                "use_index_cache: ${oc.env:PI3_USE_INDEX_CACHE,1}",
                "index_cache_dir: ${oc.env:PI3_DATASET_CACHE_DIR,}",
                "test_dataset:",
                "CO3Dv2: 1",
            ],
        )

    def test_vggt15_data_config_removes_taskonomy_and_mapillary(self):
        cfg = read_repo_file(
            "aidi/third_party/pi3_training/configs/data/meshx_pi3_vggt15_no_taskonomy_mapillary.yaml"
        )

        self.assertContainsAll(
            cfg,
            [
                "without Taskonomy and Mapillary",
                "BlendedMVS: 1000",
                "CO3Dv2: 1000",
                "DL3DV: 1000",
                "use_index_cache: ${oc.env:PI3_USE_INDEX_CACHE,1}",
                "index_cache_dir: ${oc.env:PI3_DATASET_CACHE_DIR,}",
                "test_dataset:",
            ],
        )
        self.assertNotIn("Taskonomy:", cfg)
        self.assertNotIn("Mapillary:", cfg)

    def test_business_stage0_data_config_uses_vggt_business_roots(self):
        cfg = read_repo_file("aidi/third_party/pi3_training/configs/data/meshx_pi3_business_stage0.yaml")

        self.assertContainsAll(
            cfg,
            [
                "Step0 business-data audit config",
                "BusinessDriving: 1000",
                "BusinessParking: 300",
                "BusinessParkingMechanical: 1000",
                "BusinessParkingMechanicalFisheye: 1000",
                "business_data/driving_data/20250613_train",
                "business_data/parking_data/20250613_train",
                "business_data/parking_data/20250728_mechanical_at128_train",
                "business_data/driving_data/20250521/test",
                "business_data/parking_data/20250630_mechanical_at128_eval",
                "camera_ids: &standard_camera_ids",
                '- "06"',
                "max_depth: 60",
                "use_index_cache: ${oc.env:PI3_USE_INDEX_CACHE,1}",
            ],
        )

    def test_business_multiview_stage1_config_keeps_pi3_native_sampler_contract(self):
        cfg = read_repo_file(
            "aidi/third_party/pi3_training/configs/data/meshx_pi3_business_multiview_stage1.yaml"
        )

        self.assertContainsAll(
            cfg,
            [
                "VGGT-aligned scene-level sampling",
                "native Pi3 sampler and view-count contract",
                "_target_: datasets.meshx_evc_dataset.MeshXBusinessMultiviewDataset",
                "BusinessDriving: 1000",
                "BusinessParking: 300",
                "BusinessParkingMechanical: 1000",
                "BusinessParkingMechanicalFisheye: 1000",
                "business_data/driving_data/20250613_train",
                "business_data/parking_data/20250613_train",
                "business_data/parking_data/20250728_mechanical_at128_train",
                "business_data/driving_data/20250521/test",
                "business_data/parking_data/20250630_mechanical_at128_eval",
                "min_cameras: 2",
                "camera_ids: &standard_camera_ids",
                '- "06"',
                "max_depth: 60",
                "use_index_cache: ${oc.env:PI3_USE_INDEX_CACHE,1}",
            ],
        )

    def test_business_2000hv2_val_config_uses_vggt_protocol(self):
        cfg = read_repo_file(
            "aidi/third_party/pi3_training/configs/data/meshx_pi3_business_2000h_v2_stage1.yaml"
        )

        self.assertContainsAll(
            cfg,
            [
                "vggt_val_protocol: true",
                "vggt_val_extra_src_pool: 5",
                "vggt_val_frame_sample: [0, null, 1500]",
                "distort_2000h_v1/eval1000clip_202603051529",
                "vggt_val_target_list_file: ${oc.env:BUSINESS_DRIVING_STANDARD_160K_VAL_TARGET_LIST_FILE,",
                "aidi/benchmarks/business_std160k74/target_img_list.txt",
                "vggt_val_target_metrics_file: ${oc.env:BUSINESS_DRIVING_STANDARD_160K_VAL_TARGET_METRICS_FILE,\"\"}",
                "image_num_range: [36, 36]",
            ],
        )

    def test_pi3_eval_weights_do_not_resize_validation_dataset_to_one_sample(self):
        dataloader_py = read_repo_file("aidi/third_party/pi3_training/datasets/__init__.py")

        self.assertContainsAll(
            dataloader_py,
            [
                "resize_weighted_dataset = mode == 'train' or 'length' in cfg_dataset",
                "datasets_all.append((weight @ dataset_i) if resize_weighted_dataset else dataset_i)",
            ],
        )

    def test_pi3_dynamic_batch_sampler_len_counts_eval_batches(self):
        sampler_py = read_repo_file("aidi/third_party/pi3_training/datasets/base/batched_sampler.py")

        self.assertContainsAll(
            sampler_py,
            [
                "if min_image_num == max_image_num:",
                "images_per_sample_for_len = max_image_num",
                "batch_size = max(1, int(np.floor(self.max_img_per_gpu / images_per_sample_for_len)))",
                "return int(np.ceil(sampler_len / batch_size))",
            ],
        )
        self.assertNotIn("return len(self.sampler) // self.image_num_range[0]", sampler_py)

    def test_recent_pi3_success_records_remain_traceable(self):
        records = read_repo_file("aidi/docs/records/experiment.md")

        self.assertContainsAll(
            records,
            [
                "Pi3 native full17/core4 TB下载与epoch日志修复重提",
                "TRAIN_ITERS_PER_EPOCH=800",
                "PI3_DECODER_ATTN_BACKEND=flash",
                "PI3_QK_NORM_CHUNK_SIZE=2048",
                "Pi3 native indexer warm-up init-old TB下载与epoch日志修复重提",
                "INDEXER_INIT_CKPT=${SAVE_ROOT}/trained_model/pi3/official/pi3_warmup_evc_5090_fp16_20260416_192959_layersall_h4_vggt17_lidar/latest.pt",
                "Pi3 native indexer warm-up scratch all-layer lossfix 重提",
                "workspace: `/home/users/feng01.zhou/workspace/meshx_submit_pi3_warmup_alllayer_cf6a8df0`",
                "INDEXER_LAYERS=all; no INDEXER_INIT_CKPT",
            ],
        )

    def test_framework_boundary_doc_names_recent_pi3_success_chains(self):
        self.assertTrue(BOUNDARY_DOC.is_file(), f"missing boundary doc: {BOUNDARY_DOC}")
        doc = BOUNDARY_DOC.read_text()

        self.assertContainsAll(
            doc,
            [
                "Pi3 Native 成功启动链路",
                "aidi/scripts/pi3/submit_pi3_5090_training.sh",
                "aidi/scripts/pi3/submit_pi3_5090_warmup.sh",
                "aidi/scripts/pi3/train_pi3_official.sh",
                "meshx_pi3_vggt17",
                "TRAIN_ITERS_PER_EPOCH=800",
                "WARMUP_ITERS_PER_EPOCH=800",
                "INDEXER_LAYERS=all",
                "不移动 Pi3 native launcher",
                "aidi/utils/core4_main_val.py",
                "aidi/utils/core4_model_adapter.py",
                "aidi/utils/vggt_core4_main_val.py",
                "easyvolcap/runners/volumetric_video_runner.py",
            ],
        )


if __name__ == "__main__":
    unittest.main()
