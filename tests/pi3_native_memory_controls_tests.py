from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pi3_native_launcher_keeps_highres_memory_controls_active():
    launcher = ROOT / "aidi/scripts/pi3/train_pi3_official.sh"
    source = launcher.read_text()

    assert "append_override_from_env PI3_DECODER_ATTN_BACKEND model.decoder_attn_backend" in source
    assert "append_override_from_env PI3_QK_NORM_CHUNK_SIZE model.qk_norm_chunk_size" in source
    assert 'append_bool_override_from_env PI3_HEAD_USE_CHECKPOINT "++model.head_use_checkpoint"' in source
    assert 'append_override_from_env PI3_HEAD_VIEW_CHUNK_SIZE "++model.head_view_chunk_size"' in source
    assert "warn_ignored_env PI3_DECODER_ATTN_BACKEND" not in source
    assert "warn_ignored_env PI3_QK_NORM_CHUNK_SIZE" not in source
    assert "warn_ignored_env PI3_HEAD_USE_CHECKPOINT" not in source
    assert "warn_ignored_env PI3_HEAD_VIEW_CHUNK_SIZE" not in source


def test_pi3_native_launcher_supports_shell_safe_fixed_resolution_controls():
    launcher = ROOT / "aidi/scripts/pi3/train_pi3_official.sh"
    source = launcher.read_text()

    assert 'if [ -n "${PI3_TRAIN_FIXED_RES:-}" ]; then' in source
    assert '"++train.random_reslution=False"' in source
    assert '"++train.num_resolution=1"' in source
    assert '"++train.resolution=[[${PI3_TRAIN_FIXED_RES},${PI3_TRAIN_FIXED_RES}]]"' in source
    assert 'if [ -n "${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE:-}" ]; then' in source
    assert '"++loss.train_loss.normal_loss_view_chunk_size=${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE}"' in source
    assert '"++loss.test_loss.normal_loss_view_chunk_size=${PI3_TEST_NORMAL_LOSS_VIEW_CHUNK_SIZE:-${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE}}"' in source


def test_pi3_5090_wrapper_defaults_to_validated_memory_safe_controls():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_training.sh"
    source = wrapper.read_text()

    assert "PI3_DECODER_ATTN_BACKEND=${PI3_DECODER_ATTN_BACKEND:-flash}" in source
    assert "PI3_QK_NORM_CHUNK_SIZE=${PI3_QK_NORM_CHUNK_SIZE:-2048}" in source
    assert "PI3_USE_POSE_PRIOR=${PI3_USE_POSE_PRIOR:-0}" in source
    assert "PI3_POSE_PRIOR_REQUIRED=${PI3_POSE_PRIOR_REQUIRED:-0}" in source
    assert "PI3_POSE_PRIOR_DROPOUT=${PI3_POSE_PRIOR_DROPOUT:-0.0}" in source
    assert "PI3_HEAD_USE_CHECKPOINT=${PI3_HEAD_USE_CHECKPOINT:-1}" in source
    assert "PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT=${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT:-0}" in source
    assert "PI3_HEAD_VIEW_CHUNK_SIZE=${PI3_HEAD_VIEW_CHUNK_SIZE:-4}" in source
    assert "PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE=${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE:-1}" in source
    assert "SAVE_TO_AIDI=${SAVE_TO_AIDI:-1}" in source
    assert 'REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"' in source
    assert 'BUSINESS_DRIVING_STANDARD_160K_VAL_TARGET_LIST_FILE=${BUSINESS_DRIVING_STANDARD_160K_VAL_TARGET_LIST_FILE:-"${REPO_ROOT}/aidi/benchmarks/business_std160k74/target_img_list.txt"}' in source
    assert "BUSINESS_DRIVING_STANDARD_160K_VAL_TARGET_METRICS_FILE=${BUSINESS_DRIVING_STANDARD_160K_VAL_TARGET_METRICS_FILE:-}" in source
    assert "export RUN_NAME EXP_NAME WORK_DIR PI3_TB_MIRROR_DIR SAVE_TO_AIDI" in source
    assert "BUSINESS_DRIVING_STANDARD_160K_VAL_TARGET_LIST_FILE BUSINESS_DRIVING_STANDARD_160K_VAL_TARGET_METRICS_FILE" in source
    assert "export PI3_DECODER_ATTN_BACKEND PI3_QK_NORM_CHUNK_SIZE PI3_USE_POSE_PRIOR PI3_POSE_PRIOR_REQUIRED PI3_POSE_PRIOR_DROPOUT" in source
    assert "export PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT" in source
    assert 'echo "[INFO] PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT=${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT}"' in source


def test_pi3_5090_wrapper_keeps_cpu_oom_guard_opt_in_and_preserves_throughput_defaults():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_training.sh"
    source = wrapper.read_text()

    assert "PI3_CPU_OOM_GUARD=${PI3_CPU_OOM_GUARD:-0}" in source
    assert "PI3_TRAIN_NUM_WORKERS=${PI3_TRAIN_NUM_WORKERS:-${TRAIN_NUM_WORKERS:-8}}" in source
    assert "PI3_TEST_NUM_WORKERS=${PI3_TEST_NUM_WORKERS:-${TEST_NUM_WORKERS:-8}}" in source
    assert "TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-2}" in source
    assert "TEST_PREFETCH_FACTOR=${TEST_PREFETCH_FACTOR:-2}" in source
    assert "TRAIN_PERSISTENT_WORKERS=${TRAIN_PERSISTENT_WORKERS:-1}" in source
    assert "TEST_PERSISTENT_WORKERS=${TEST_PERSISTENT_WORKERS:-1}" in source
    assert "TRAIN_PIN_MEMORY=${TRAIN_PIN_MEMORY:-1}" in source
    assert "TEST_PIN_MEMORY=${TEST_PIN_MEMORY:-1}" in source
    assert "LOG_CKPT_INTERVAL=${LOG_CKPT_INTERVAL:-1}" in source
    assert "LOG_BEST_MODEL_SAVE_MODE=${LOG_BEST_MODEL_SAVE_MODE:-full_state}" in source
    assert "PI3_BUSINESS_CAMERA_CACHE_SIZE=${PI3_BUSINESS_CAMERA_CACHE_SIZE:-0}" in source
    assert "PI3_LAZY_GT_POINTS=${PI3_LAZY_GT_POINTS:-1}" in source
    assert "export PI3_USE_INDEX_CACHE PI3_REBUILD_INDEX_CACHE PI3_DATASET_CACHE_DIR PI3_INDEX_CACHE_WAIT_SEC PI3_LAZY_SEQUENCE_INDEX PI3_BUSINESS_CAMERA_CACHE_SIZE PI3_LAZY_GT_POINTS" in source
    assert 'echo "[INFO] PI3_LAZY_GT_POINTS=${PI3_LAZY_GT_POINTS}"' in source
    assert "PI3_MAX_TRAIN_NUM_WORKERS=${PI3_MAX_TRAIN_NUM_WORKERS:-2}" in source
    assert "PI3_MAX_TEST_NUM_WORKERS=${PI3_MAX_TEST_NUM_WORKERS:-1}" in source
    assert "PI3_MAX_PREFETCH_FACTOR=${PI3_MAX_PREFETCH_FACTOR:-1}" in source
    assert "PI3_MIN_LOG_CKPT_INTERVAL=${PI3_MIN_LOG_CKPT_INTERVAL:-5}" in source
    assert "clamp_int_var PI3_TRAIN_NUM_WORKERS" in source
    assert "clamp_int_var PI3_TEST_NUM_WORKERS" in source
    assert "force_zero_var TRAIN_PIN_MEMORY" in source
    assert "force_zero_var TRAIN_PERSISTENT_WORKERS" in source
    assert "ensure_min_int_var LOG_CKPT_INTERVAL" in source
    assert "PI3_MEMORY_REPORT=${PI3_MEMORY_REPORT:-1}" in source
    assert "PI3_MEMORY_ABORT_FRACTION=${PI3_MEMORY_ABORT_FRACTION:-0.90}" in source
    assert "PI3_MEMORY_ABORT_METRIC=${PI3_MEMORY_ABORT_METRIC:-working_set}" in source
    assert "PI3_MEMORY_ABORT_METRIC" in source


def test_pi3_business_config_defaults_to_lazy_gt_points_to_reduce_worker_memory():
    cfg = ROOT / "aidi/third_party/pi3_training/configs/data/meshx_pi3_business_2000h_v2_stage1.yaml"
    source = cfg.read_text()

    assert "lazy_gt_points: ${oc.env:PI3_LAZY_GT_POINTS,1}" in source


def test_pi3_business_config_exposes_3ddr_pose_prior_file():
    cfg = ROOT / "aidi/third_party/pi3_training/configs/data/meshx_pi3_business_2000h_v2_stage1.yaml"
    source = cfg.read_text()

    assert "pose_prior_extri_file: ${oc.env:BUSINESS_DRIVING_2000H_V2_POSE_PRIOR_EXTRI_FILE,extri_3ddr.yml}" in source
    assert "pose_prior_required: ${oc.env:PI3_POSE_PRIOR_REQUIRED,0}" in source


def test_pi3_default_epoch_length_is_1k_iterations():
    cfg = ROOT / "aidi/third_party/pi3_training/configs/model/pi3.yaml"
    source = cfg.read_text()

    assert "iters_per_epoch: 1000" in source


def test_pi3_loss_and_vggt_validation_support_lazy_gt_points():
    loss = ROOT / "aidi/third_party/pi3_training/pi3/models/loss.py"
    validation = ROOT / "aidi/third_party/pi3_training/utils/vggt_validation.py"
    gt_points = ROOT / "aidi/third_party/pi3_training/pi3/utils/gt_points.py"

    assert "def stack_gt_points_and_masks" in gt_points.read_text()
    assert "from ..utils.gt_points import stack_gt_points_and_masks" in loss.read_text()
    assert "from pi3.utils.gt_points import stack_gt_points_and_masks" in validation.read_text()


def test_pi3_native_launcher_passes_checkpoint_and_memory_guard_overrides():
    launcher = ROOT / "aidi/scripts/pi3/train_pi3_official.sh"
    source = launcher.read_text()

    assert "append_override_from_env LOG_BEST_MODEL_SAVE_MODE log.best_model_save_mode" in source
    assert "append_override_from_env LOG_CHECKPOINT_SAVE_MODE log.checkpoint_save_mode" in source
    assert "append_bool_override_from_env PI3_MEMORY_REPORT log.memory_report" in source
    assert "append_override_from_env PI3_MEMORY_ABORT_FRACTION log.memory_abort_fraction" in source
    assert "append_override_from_env PI3_MEMORY_ABORT_METRIC log.memory_abort_metric" in source


def test_pi3_trainer_has_cgroup_memory_guard_and_configurable_checkpoint_modes():
    trainer = ROOT / "aidi/third_party/pi3_training/trainers/base_trainer_accelerate.py"
    source = trainer.read_text()

    assert "def _log_memory_snapshot" in source
    assert "memory.current" in source
    assert "def _read_process_tree_rss_bytes" in source
    assert "def _read_cgroup_memory_stat_bytes" in source
    assert "memory.stat" in source
    assert "max_tree_rss_rank" in source
    assert "working_set_ratio" in source
    assert "memory_abort_metric" in source
    assert "local_working_set_ratio=" in source
    assert "memory_abort_fraction" in source
    assert "best_model_save_mode" in source
    assert "checkpoint_save_mode" in source
    assert "def _save_checkpoint_path" in source
    assert "model_only" in source
    assert "none" in source


def test_pi3_meshx_business_loader_avoids_full_size_rgb_numpy_copy():
    dataset = ROOT / "aidi/third_party/pi3_training/datasets/meshx_evc_dataset.py"
    source = dataset.read_text()

    assert "np.array(Image.open(image_path))" not in source
    assert "with Image.open(image_path) as image:" in source
    assert 'rgb_image = image.convert("RGB")' in source


def test_pi3_default_log_config_exposes_memory_and_checkpoint_safety_controls():
    cfg = ROOT / "aidi/third_party/pi3_training/configs/general/default.yaml"
    source = cfg.read_text()

    assert "best_model_save_mode: full_state" in source
    assert "checkpoint_save_mode: full_state" in source
    assert "memory_report: false" in source
    assert "memory_abort_fraction: 0.0" in source
    assert "memory_abort_metric: raw" in source


def test_aidi_submit_can_request_larger_cpu_memory_ratio_for_v2_jobs():
    submit = ROOT / "aidi/submit.py"
    source = submit.read_text()

    assert "--cpu_mem_ratio" in source
    assert "AIDI_CPU_MEM_RATIO" in source
    assert '"cpu_mem_ratio": cpu_mem_ratio' in source
    assert '"CPU_MEM_RATIO": config["cpu_mem_ratio"]' in source
    assert "cpu_mem_ratio=config[\"cpu_mem_ratio\"]" in source


def test_pi3_5090_wrapper_uses_safe_env_knobs_for_fixed_res_and_loss_chunk():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_training.sh"
    source = wrapper.read_text()

    assert "PI3_TRAIN_FIXED_RES=${PI3_TRAIN_FIXED_RES:-}" in source
    assert "PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE=${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE:-1}" in source
    assert "PI3_TEST_NORMAL_LOSS_VIEW_CHUNK_SIZE=${PI3_TEST_NORMAL_LOSS_VIEW_CHUNK_SIZE:-${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE}}" in source
    assert "export PI3_TRAIN_FIXED_RES PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE PI3_TEST_NORMAL_LOSS_VIEW_CHUNK_SIZE" in source
    assert "PI3_TRAIN_FIXED_RES=${PI3_TRAIN_FIXED_RES:-<dynamic>}" in source
    assert "PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE=${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE:-<disabled>}" in source
    assert "USER_EXTRA_OVERRIDES" not in source
    assert "export EXTRA_OVERRIDES" not in source
    assert "EXTRA_OVERRIDES=${" not in source


def test_pi3_5090_warmup_wrapper_enables_aidi_tensorboard_by_default():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_warmup.sh"
    source = wrapper.read_text()

    assert "SAVE_TO_AIDI=${SAVE_TO_AIDI:-1}" in source
    assert "export SAVE_TO_AIDI" in source
    assert 'echo "[INFO] SAVE_TO_AIDI=${SAVE_TO_AIDI}"' in source


def test_pi3_5090_warmup_wrapper_can_use_native_dynamic_resolution():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_warmup.sh"
    source = wrapper.read_text()

    assert "TRAIN_DYNAMIC_RES=${TRAIN_DYNAMIC_RES:-0}" in source
    assert 'if is_truthy "${TRAIN_DYNAMIC_RES}"; then' in source
    assert '"++train.random_reslution=True"' in source
    assert '"++train.aspect_ratio_range=${TRAIN_DYNAMIC_ASPECT_RATIO_RANGE}"' in source
    assert '"++train.pixel_count_range=${TRAIN_DYNAMIC_PIXEL_COUNT_RANGE}"' in source
    assert '"++train.patch_size=${TRAIN_DYNAMIC_PATCH_SIZE}"' in source
    assert '"++train.num_resolution=${TRAIN_DYNAMIC_NUM_RESOLUTION}"' in source
    assert 'WARMUP_OVERRIDES+=("train.resolution=[[${TRAIN_RES},${TRAIN_RES}]]")' in source


def test_pi3_5090_sparse_wrapper_can_use_native_dynamic_resolution():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh"
    source = wrapper.read_text()

    assert "TRAIN_MODEL_DTYPE=${TRAIN_MODEL_DTYPE:-fp16}" in source
    assert "SPARSE_LR=${SPARSE_LR:-}" in source
    assert "SPARSE_LOSS_WEIGHT=${SPARSE_LOSS_WEIGHT:-}" in source
    assert "TRAIN_DYNAMIC_RES=${TRAIN_DYNAMIC_RES:-0}" in source
    assert "PI3_FIND_UNUSED_PARAMETERS=${PI3_FIND_UNUSED_PARAMETERS:-0}" in source
    assert "PI3_STATIC_GRAPH=${PI3_STATIC_GRAPH:-0}" in source
    assert 'if is_truthy "${TRAIN_DYNAMIC_RES}"; then' in source
    assert '"++train.random_reslution=True"' in source
    assert '"++train.aspect_ratio_range=${TRAIN_DYNAMIC_ASPECT_RATIO_RANGE}"' in source
    assert '"++train.pixel_count_range=${TRAIN_DYNAMIC_PIXEL_COUNT_RANGE}"' in source
    assert '"++train.patch_size=${TRAIN_DYNAMIC_PATCH_SIZE}"' in source
    assert '"++train.num_resolution=${TRAIN_DYNAMIC_NUM_RESOLUTION}"' in source
    assert '"++train.resolution=[[${TRAIN_RES},${TRAIN_RES}]]"' in source
    assert '"model.indexer_cfg.sparse_lr=${SPARSE_LR}"' in source
    assert '"model.indexer_cfg.sparse_loss_weight=${SPARSE_LOSS_WEIGHT}"' in source
    assert '"train.find_unused_parameters=${PI3_FIND_UNUSED_PARAMETERS}"' in source
    assert '"train.static_graph=${PI3_STATIC_GRAPH}"' in source
    assert '"++train_dataset.${dataset_key}.use_index_cache=${PI3_USE_INDEX_CACHE}"' in source
    assert '"++train_dataset.${dataset_key}.index_cache_dir=${PI3_DATASET_CACHE_DIR}"' in source


def test_pi3_5090_sparse_wrapper_defaults_to_decoder_checkpointing():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh"
    source = wrapper.read_text()

    assert "PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT=${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT:-0}" in source
    assert "PI3_DECODER_ATTN_BACKEND=${PI3_DECODER_ATTN_BACKEND:-flash}" in source
    assert "PI3_QK_NORM_CHUNK_SIZE=${PI3_QK_NORM_CHUNK_SIZE:-2048}" in source
    assert "PI3_HEAD_USE_CHECKPOINT=${PI3_HEAD_USE_CHECKPOINT:-1}" in source
    assert "PI3_HEAD_VIEW_CHUNK_SIZE=${PI3_HEAD_VIEW_CHUNK_SIZE:-${PI3_SPARSE_VIEW_CHUNK_SIZE:-4}}" in source
    assert "PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE=${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE:-1}" in source
    assert "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" in source
    assert "export PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT" in source
    assert "export PI3_DECODER_ATTN_BACKEND" in source
    assert "export PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE" in source


def test_pi3_5090_sparse_wrapper_maps_legacy_sparse_chunk_envs():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh"
    source = wrapper.read_text()

    assert "PI3_SPARSE_HEAD_CHUNK_SIZE=${PI3_SPARSE_HEAD_CHUNK_SIZE:-}" in source
    assert "PI3_SPARSE_VIEW_CHUNK_SIZE=${PI3_SPARSE_VIEW_CHUNK_SIZE:-}" in source
    assert "PI3_SPARSE_QUERY_CHUNK_SIZE=${PI3_SPARSE_QUERY_CHUNK_SIZE:-}" in source
    assert "INDEXER_HEAD_CHUNK_SIZE=${INDEXER_HEAD_CHUNK_SIZE:-${PI3_SPARSE_HEAD_CHUNK_SIZE:-4}}" in source
    assert "INDEXER_SCORE_HEAD_CHUNK_SIZE=${INDEXER_SCORE_HEAD_CHUNK_SIZE:-${PI3_SPARSE_HEAD_CHUNK_SIZE:-4}}" in source
    assert "INDEXER_SCORE_KEY_CHUNK_SIZE=${INDEXER_SCORE_KEY_CHUNK_SIZE:-${PI3_SPARSE_QUERY_CHUNK_SIZE:-4096}}" in source


def test_pi3_5090_sparse_wrapper_exposes_objective_value_gate_envs():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh"
    source = wrapper.read_text()

    assert "SPARSE_OBJECTIVE_VALUE_GATE=${SPARSE_OBJECTIVE_VALUE_GATE:-0}" in source
    assert "SPARSE_OBJECTIVE_VALUE_GATE_SCALE=${SPARSE_OBJECTIVE_VALUE_GATE_SCALE:-}" in source
    assert "SPARSE_OBJECTIVE_VALUE_GATE_TAU=${SPARSE_OBJECTIVE_VALUE_GATE_TAU:-}" in source
    assert "SPARSE_OBJECTIVE_VALUE_GATE_QUERY_CHUNK_SIZE=${SPARSE_OBJECTIVE_VALUE_GATE_QUERY_CHUNK_SIZE:-}" in source
    assert '"model.indexer_cfg.objective_value_gate_enabled=${SPARSE_OBJECTIVE_VALUE_GATE}"' in source
    assert '"model.indexer_cfg.objective_value_gate_scale=${SPARSE_OBJECTIVE_VALUE_GATE_SCALE}"' in source
    assert '"model.indexer_cfg.objective_value_gate_tau=${SPARSE_OBJECTIVE_VALUE_GATE_TAU}"' in source
    assert '"model.indexer_cfg.objective_value_gate_query_chunk_size=${SPARSE_OBJECTIVE_VALUE_GATE_QUERY_CHUNK_SIZE}"' in source


def test_pi3_klwarm_staged_wrapper_propagates_phase2_memory_controls():
    wrapper = ROOT / "aidi/scripts/pi3/submit_pi3_5090_sparse_klwarm_then_task.sh"
    source = wrapper.read_text()

    assert "TRAIN_MODEL_DTYPE=${TRAIN_MODEL_DTYPE:-bf16}" in source
    assert "TRAIN_MAX_IMG_PER_GPU=${TRAIN_MAX_IMG_PER_GPU:-24}" in source
    assert "PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT=${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT:-0}" in source
    assert "PI3_DECODER_ATTN_BACKEND=${PI3_DECODER_ATTN_BACKEND:-flash}" in source
    assert "PI3_QK_NORM_CHUNK_SIZE=${PI3_QK_NORM_CHUNK_SIZE:-2048}" in source
    assert "PI3_HEAD_VIEW_CHUNK_SIZE=${PI3_HEAD_VIEW_CHUNK_SIZE:-4}" in source
    assert "PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE=${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE:-1}" in source
    assert "INDEXER_SCORE_KEY_CHUNK_SIZE=${INDEXER_SCORE_KEY_CHUNK_SIZE:-4096}" in source
    assert "PHASE2_ENV=(" in source
    assert "PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT=${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT}" in source
    assert "PI3_DECODER_ATTN_BACKEND=${PI3_DECODER_ATTN_BACKEND}" in source
    assert "PI3_HEAD_VIEW_CHUNK_SIZE=${PI3_HEAD_VIEW_CHUNK_SIZE}" in source
    assert "PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE=${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE}" in source
    assert "INDEXER_SCORE_KEY_CHUNK_SIZE=${INDEXER_SCORE_KEY_CHUNK_SIZE}" in source


def test_pi3_native_tensorboard_bridge_keeps_aidi_download_copy():
    launcher = ROOT / "aidi/scripts/pi3/train_pi3_official.sh"
    source = launcher.read_text()

    assert 'SAVE_TO_AIDI="${SAVE_TO_AIDI:-0}"' in source
    assert 'PI3_TB_RECORD_DIR="/job_tboard/record/${EXP_NAME}"' in source
    assert 'PI3_TB_AIDI_MIRROR_DIR="/job_data/record/${EXP_NAME}"' in source
    assert "_sync_pi3_tb_to_dir" in source
    assert '_sync_pi3_tb_to_dir "${PI3_TB_MIRROR_DIR}"' in source
    assert '_sync_pi3_tb_to_dir "${PI3_TB_AIDI_MIRROR_DIR}"' in source
    assert "[INFO] PI3_TB_AIDI_MIRROR_DIR=" in source


def test_pi3_native_train_progress_uses_configured_epoch_length():
    logger = ROOT / "aidi/third_party/pi3_training/utils/dist.py"
    logger_source = logger.read_text()
    trainer = ROOT / "aidi/third_party/pi3_training/trainers/base_trainer_accelerate.py"
    trainer_source = trainer.read_text()

    assert "def log_every(self, iterable, print_freq, header=None, max_iters=None)" in logger_source
    assert "display_total = min(int(max_iters), iterable_len)" in logger_source
    assert "processed >= int(max_iters)" in logger_source
    assert "total_time / processed" in logger_source
    assert "max_iters=self.iters_per_epoch" in trainer_source
    assert "val_log_max_iters = max_test_iters if max_test_iters > 0 else None" in trainer_source
    assert "self.cfg.test.print_freq" in trainer_source


def test_pi3_native_dataloader_respects_runtime_worker_flags():
    dataset_init = ROOT / "aidi/third_party/pi3_training/datasets/__init__.py"
    source = dataset_init.read_text()

    assert "pin_memory = cfg_dataloader.pin_memory" in source
    assert "persistent_workers = cfg_dataloader.persistent_workers" in source
    assert "prefetch_factor = cfg_dataloader.prefetch_factor" in source
    assert 'loader_kwargs["persistent_workers"] = bool(persistent_workers)' in source
    assert 'loader_kwargs["prefetch_factor"] = int(prefetch_factor)' in source
    assert "pin_memory=True" not in source
    assert "persistent_workers=True" not in source


def test_pi3_native_code_does_not_cross_into_evc_vggt_training_entrypoints():
    pi3_roots = [
        ROOT / "aidi/scripts/pi3",
        ROOT / "aidi/third_party/pi3_training",
    ]
    forbidden = [
        "easyvolcap.runners",
        "volumetric_video_runner",
        "OfficialVGGTModel",
        "vggt_core4_main_val",
        "evc-train",
        "evc-test",
    ]

    offenders = []
    for root in pi3_roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".sh", ".yaml", ".yml"}:
                continue
            source = path.read_text(errors="ignore")
            for needle in forbidden:
                if needle in source:
                    offenders.append(f"{path.relative_to(ROOT)} contains {needle}")

    assert not offenders, "\n".join(offenders)


def test_pi3_model_exposes_memory_control_knobs():
    cfg = ROOT / "aidi/third_party/pi3_training/configs/model/pi3.yaml"
    cfg_source = cfg.read_text()

    assert "decoder_attn_backend: auto" in cfg_source
    assert "qk_norm_chunk_size: 0" in cfg_source
    assert "head_use_checkpoint: false" in cfg_source
    assert "head_view_chunk_size: 0" in cfg_source

    model = ROOT / "aidi/third_party/pi3_training/pi3/models/pi3_training.py"
    model_source = model.read_text()

    assert "decoder_attn_backend=\"auto\"" in model_source
    assert "qk_norm_chunk_size=0" in model_source
    assert "head_use_checkpoint=False" in model_source
    assert "head_view_chunk_size=0" in model_source
    assert "resolve_attention_backend(decoder_attn_backend, rope=True)" in model_source
    assert "self.head_use_checkpoint = bool(head_use_checkpoint)" in model_source
    assert "self.head_view_chunk_size = int(head_view_chunk_size)" in model_source


def test_pi3_warmup_only_forward_skips_point_and_camera_heads():
    model = ROOT / "aidi/third_party/pi3_training/pi3/models/pi3_training.py"
    model_source = model.read_text()

    assert "def decode(self, hidden, N, H, W, collect_head_input=True" in model_source
    assert "final_output = [] if collect_head_input else None" in model_source
    assert "if not collect_head_input:" in model_source
    assert "def forward(self, imgs, warmup_only_indexer_loss=False" in model_source
    assert "collect_head_input=not warmup_only_indexer_loss" in model_source
    assert "return dict(indexer_warmup_only=True)" in model_source


def test_pi3_trainer_passes_warmup_only_flag_and_handles_layerwise_loss():
    trainer = ROOT / "aidi/third_party/pi3_training/trainers/pi3_trainer.py"
    trainer_source = trainer.read_text()
    base_trainer = ROOT / "aidi/third_party/pi3_training/trainers/base_trainer_accelerate.py"
    base_source = base_trainer.read_text()

    assert "def _use_warmup_only_indexer_loss" in trainer_source
    assert "warmup_only_indexer_loss = self._use_warmup_only_indexer_loss(state, mode=mode)" in trainer_source
    assert "warmup_only_indexer_loss=warmup_only_indexer_loss" in trainer_source
    assert "pose_prior=pose_prior" in trainer_source
    assert "pose_prior_intrinsics=pose_prior_intrinsics" in trainer_source
    assert "return [pred, batch, warmup_only_indexer_loss]" in trainer_source
    assert "output, batch, warmup_only_indexer_loss = output" in trainer_source
    assert "isinstance(indexer_loss, (list, tuple))" in trainer_source
    assert "loss = list(indexer_loss)" in trainer_source
    assert "def _loss_items(loss)" in base_source
    assert "def _backward_loss_items(self, loss_items)" in base_source
    assert "loss_items = self._loss_items(batch_output.loss)" in base_source


def test_pi3_official_ckpt_initializes_missing_indexer_from_attention():
    model = ROOT / "aidi/third_party/pi3_training/pi3/models/pi3_training.py"
    model_source = model.read_text()

    assert "missing_indexer_keys" in model_source
    assert "loaded_indexer_keys" in model_source
    assert "if missing_indexer_keys and not loaded_indexer_keys:" in model_source
    assert "self._init_indexer_from_attention()" in model_source


def test_pi3_sparse_stage_can_disable_indexer_loss_and_freeze_indexer():
    script = ROOT / "aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh"
    script_source = script.read_text()
    model = ROOT / "aidi/third_party/pi3_training/pi3/models/pi3_training.py"
    model_source = model.read_text()
    model_cfg = ROOT / "aidi/third_party/pi3_training/configs/model/pi3.yaml"
    model_cfg_source = model_cfg.read_text()
    sparse_cfg = ROOT / "aidi/third_party/pi3_training/configs/train/train_pi3_lowres_indexer_sparse.yaml"
    sparse_cfg_source = sparse_cfg.read_text()

    assert "SPARSE_COMPUTE_LOSS=${SPARSE_COMPUTE_LOSS:-1}" in script_source
    assert "SPARSE_FREEZE_INDEXER=${SPARSE_FREEZE_INDEXER:-0}" in script_source
    assert '"model.indexer_cfg.compute_loss=${SPARSE_COMPUTE_LOSS}"' in script_source
    assert '"model.indexer_cfg.freeze_indexer=${SPARSE_FREEZE_INDEXER}"' in script_source
    assert "compute_loss=True" in model_source
    assert "freeze_indexer=False" in model_source
    assert 'compute_loss = bool(training) and self._indexer_cfg_bool("compute_loss", True)' in model_source
    assert "def _set_indexer_frozen(self, freeze_indexer: bool) -> None:" in model_source
    assert 'blk.return_indexer_loss = bool(layer_indexer_state.get("compute_loss", False))' in model_source
    assert 'self._indexer_cfg_bool("freeze_indexer", False)' in model_source
    assert "compute_loss: true" in model_cfg_source
    assert "freeze_indexer: false" in model_cfg_source
    assert "compute_loss: true" in sparse_cfg_source
    assert "freeze_indexer: false" in sparse_cfg_source


def test_pi3_sparse_stage_supports_kl_only_warmup_then_task_loss_transition():
    script = ROOT / "aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh"
    script_source = script.read_text()
    model = ROOT / "aidi/third_party/pi3_training/pi3/models/pi3_training.py"
    model_source = model.read_text()
    model_cfg = ROOT / "aidi/third_party/pi3_training/configs/model/pi3.yaml"
    model_cfg_source = model_cfg.read_text()
    sparse_cfg = ROOT / "aidi/third_party/pi3_training/configs/train/train_pi3_lowres_indexer_sparse.yaml"
    sparse_cfg_source = sparse_cfg.read_text()

    assert "SPARSE_WARMUP_STEPS=${SPARSE_WARMUP_STEPS:-0}" in script_source
    assert "SPARSE_START_STEP=${SPARSE_START_STEP:-${SPARSE_WARMUP_STEPS}}" in script_source
    assert "SPARSE_WARMUP_ONLY_INDEXER_LOSS=${SPARSE_WARMUP_ONLY_INDEXER_LOSS:-0}" in script_source
    assert "SPARSE_WARMUP_ONLY_INDEXER_TRAIN=${SPARSE_WARMUP_ONLY_INDEXER_TRAIN:-0}" in script_source
    assert "SPARSE_WARMUP_OPTIMIZER_FILTER=${SPARSE_WARMUP_OPTIMIZER_FILTER:-1}" in script_source
    assert "SPARSE_WARMUP_INDEXER_LOSS_MODE=${SPARSE_WARMUP_INDEXER_LOSS_MODE:-}" in script_source
    assert '"model.indexer_cfg.warmup_steps=${SPARSE_WARMUP_STEPS}"' in script_source
    assert '"model.indexer_cfg.sparse_start_step=${SPARSE_START_STEP}"' in script_source
    assert '"model.indexer_cfg.warmup_only_indexer_loss=${SPARSE_WARMUP_ONLY_INDEXER_LOSS}"' in script_source
    assert '"model.indexer_cfg.warmup_only_indexer_train=${SPARSE_WARMUP_ONLY_INDEXER_TRAIN}"' in script_source
    assert '"model.indexer_cfg.warmup_optimizer_filter=${SPARSE_WARMUP_OPTIMIZER_FILTER}"' in script_source
    assert '"model.indexer_cfg.warmup_indexer_loss_mode=${SPARSE_WARMUP_INDEXER_LOSS_MODE}"' in script_source
    assert '"model.indexer_cfg.warmup_loss_weight=${SPARSE_WARMUP_LOSS_WEIGHT}"' in script_source

    assert "warmup_optimizer_filter=True" in model_source
    assert 'self._indexer_cfg_bool("warmup_optimizer_filter", True)' in model_source
    assert "warmup_optimizer_filter: true" in model_cfg_source
    assert "warmup_optimizer_filter: true" in sparse_cfg_source


def test_pi3_staged_klwarm_wrapper_runs_warmup_before_sparse_task_phase():
    script = ROOT / "aidi/scripts/pi3/submit_pi3_5090_sparse_klwarm_then_task.sh"
    source = script.read_text()

    assert "KLWARM_STEPS=${KLWARM_STEPS:-3000}" in source
    assert "PHASE1_CKPT=${PHASE1_CKPT:-${PHASE1_OUTPUT_DIR}/ckpts/checkpoint_0/pytorch_model.bin}" in source
    assert "bash aidi/scripts/pi3/submit_pi3_5090_warmup.sh" in source
    assert "bash aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh" in source
    assert "RUN_NAME=${PHASE1_RUN_NAME}" in source
    assert "WARMUP_STEPS=${KLWARM_STEPS}" in source
    assert "WARMUP_NUM_EPOCH=1" in source
    assert "WARMUP_ITERS_PER_EPOCH=${KLWARM_STEPS}" in source
    assert "WARMUP_ONLY_INDEXER_TRAIN=True" in source
    assert "TEST_ITERS_PER_TEST=0" in source
    assert "[ ! -f \"${PHASE1_CKPT}\" ]" in source
    assert "RUN_NAME=${PHASE2_RUN_NAME}" in source
    assert "MODEL_CKPT=${PHASE1_CKPT}" in source
    assert "SPARSE_WARMUP_STEPS=0" in source
    assert "SPARSE_WARMUP_ONLY_INDEXER_LOSS=0" in source
    assert "SPARSE_WARMUP_ONLY_INDEXER_TRAIN=0" in source
    assert "LOAD_VGGT=0" in source
