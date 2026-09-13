import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pi3_trainer_uses_direct_tensorboard_writer():
    trainer = ROOT / "aidi/third_party/pi3_training/trainers/base_trainer_accelerate.py"
    source = trainer.read_text()

    assert "SummaryWriter" in source
    assert "direct_tensorboard" in source
    assert "_log_scalars" in source


def test_pi3_grad_norm_logs_pre_clip_value_for_vggt_alignment():
    trainer = ROOT / "aidi/third_party/pi3_training/trainers/base_trainer_accelerate.py"
    source = trainer.read_text()

    pre_clip_idx = source.index("grad_norm_pre_clip = self._grad_norm_to_float(")
    clip_idx = source.index("self.accelerator.clip_grad_norm_")
    post_clip_idx = source.index("grad_norm_post_clip = get_gradient_norm(self.model.parameters())")
    alias_idx = source.index('"TRAIN/grad_norm": grad_norm_pre_clip')

    assert pre_clip_idx < clip_idx < post_clip_idx < alias_idx
    assert "def _grad_norm_to_float(grad_norm):" in source
    assert "def _combine_grad_norms(*grad_norms):" in source
    assert "grad_norm_pre_clip = self._combine_grad_norms(" in source
    assert '"grad_norm": grad_norm_pre_clip' in source
    assert '"grad_norm_post_clip": grad_norm_post_clip' in source
    assert '"TRAIN/grad_norm_post_clip": grad_norm_post_clip' in source


def test_pi3_launcher_mirrors_job_tboard_to_record_root():
    launcher = ROOT / "aidi/scripts/pi3/train_pi3_official.sh"
    source = launcher.read_text()

    assert "PI3_TB_RECORD_DIR" in source
    assert "PI3_TB_MIRROR_DIR" in source
    assert "++log.tensorboard_dir=${PI3_TB_RECORD_DIR}" in source
    assert "PI3_TB_MIRROR_DIR=${PI3_TB_MIRROR_DIR:-${WORK_DIR:-}}" in source
    assert "PI3_TB_MIRROR_DIR=${PI3_TB_MIRROR_DIR:-${WORK_DIR:-}/tensorboard}" not in source


def test_pi3_launcher_keeps_core4_val_controls():
    launcher = ROOT / "aidi/scripts/pi3/train_pi3_official.sh"
    source = launcher.read_text()

    assert "CORE4_VAL" in source
    assert "extras=core4_main_val" in source
    assert "++main_val_core4_cfg.run_first_eval=True" in source
    assert "test.iters_per_test=0" in source


def test_pi3_core4_val_config_is_not_filtered_from_submit_package():
    submit = ROOT / "aidi/submit.py"
    tree = ast.parse(submit.read_text())
    exclude_assign = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "EXCLUDE_PATH" for target in node.targets)
    )
    exclude_path = ast.literal_eval(exclude_assign.value)

    assert (ROOT / "aidi/third_party/pi3_training/configs/extras/core4_main_val.yaml").is_file()
    assert "core*" not in exclude_path
    assert "/core*" in exclude_path


def test_pi3_trainer_runs_core4_first_eval():
    trainer = ROOT / "aidi/third_party/pi3_training/trainers/base_trainer_accelerate.py"
    source = trainer.read_text()

    assert "core4_main_val_first_eval_enabled" in source
    assert "run_core4_main_val(-1)" in source
    assert "extract_core4_tb_scalars" in source
    assert "from aidi.utils.core4_main_val import extract_core4_tb_scalars, run_core4_main_val" in source
    assert "from aidi.utils.vggt_core4_main_val import extract_core4_tb_scalars, run_core4_main_val" not in source


def test_pi3_warmup_defaults_to_memory_safe_streaming_loss():
    launcher = ROOT / "aidi/scripts/pi3/submit_pi3_5090_warmup.sh"
    source = launcher.read_text()

    assert "DEFAULT_STREAMING_KL_LOSS=1" in source
    assert "STREAMING_KL_FWD_MODE=${STREAMING_KL_FWD_MODE:-\"score\"}" in source
    assert "STREAMING_KL_BWD_MODE=${STREAMING_KL_BWD_MODE:-\"score\"}" in source
    assert "export VGGT_STREAMING_KL_FWD_MODE=${VGGT_STREAMING_KL_FWD_MODE:-${STREAMING_KL_FWD_MODE}}" in source
    assert "DEFAULT_STREAMING_KL_LOSS=0" not in source
