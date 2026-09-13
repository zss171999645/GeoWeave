#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STRICT_IMPORT="${STRICT_IMPORT:-0}"
if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_CMD="${PYTHON_BIN}"
elif [ -x /opt/miniconda3/envs/easyvolcap/bin/python ]; then
    PYTHON_CMD="/opt/miniconda3/envs/easyvolcap/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "[ERROR] No usable Python found. Set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi

require_path() {
    local path="$1"
    if [ ! -e "${path}" ]; then
        echo "[ERROR] Missing path: ${path}" >&2
        exit 1
    fi
}

require_path "${HANDOFF_ROOT}/anonymous_model/geoweave_model"
require_path "${SCRIPT_DIR}/robustness_pi3.sh"
require_path "${REPO_ROOT}/aidi/scripts/vggt/train_official_vggt.sh"
require_path "${REPO_ROOT}/aidi/scripts/vggt/submit_official_vggt_5090_warmup.sh"
require_path "${REPO_ROOT}/aidi/scripts/pi3/train_pi3_official.sh"
require_path "${REPO_ROOT}/aidi/scripts/pi3/submit_pi3_5090_warmup.sh"
require_path "${REPO_ROOT}/aidi/scripts/vggt/run_unified_eval.py"
require_path "${REPO_ROOT}/aidi/scripts/vggt/eval_co3d_pose_official_upstream.py"
require_path "${REPO_ROOT}/aidi/scripts/vggt/eval_da3_pose_benchmark.py"
require_path "${REPO_ROOT}/aidi/scripts/vggt/eval_relpose_1500_benchmark.py"
require_path "${REPO_ROOT}/aidi/scripts/vggt/eval_vggt_re10k_pose_lightweight.py"
require_path "${REPO_ROOT}/aidi/scripts/vggt/run_co3dv2_official_upstream.sh"
require_path "${REPO_ROOT}/aidi/scripts/vggt/run_eval_dtu_mv_recon_pi3_style.sh"
require_path "${REPO_ROOT}/aidi/scripts/vggt/run_eval_eth3d_mv_recon_pi3_style.sh"
require_path "${REPO_ROOT}/aidi/scripts/vggt/run_eval_7scenes_mv_recon_pi3_style.sh"
require_path "${REPO_ROOT}/aidi/scripts/vggt/run_eval_nrgbd_mv_recon_pi3_style.sh"
require_path "${REPO_ROOT}/aidi/scripts/vggt/vggt_omega_eval_utils.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/eval_config_utils.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/eval_pi3_co3d_pose_official.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/eval_pi3_depth_protocol.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/eval_pi3_monodepth_protocol.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/eval_pi3_re10k_pose_official.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/eval_pi3_relpose_distance_protocol.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/eval_pi3_videodepth_protocol.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/eval_pi3_mv_recon_core.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/exr_read_utils.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/generate_seq_id_map.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/pi3_checkpoint_loader.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/pi3_metric_utils.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/prepare_eth3d_pi3_style.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/overlap_noise_seq_map_utils.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/build_evc_same_scene_candidate_pool_benchmark.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/build_fixed10_overlap_band_benchmark.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/build_same_scene_low_overlap_5plus5_benchmark.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/build_three_dataset_stride4_distractor_benchmark.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/build_driving_multicamera_all_eval_benchmark.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/build_waymo_plausible_context_diagnostic.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/build_waymo_plausible_wrong_trigger_protocol.py"
require_path "${REPO_ROOT}/aidi/scripts/baselines/summarize_waymo_plausible_wrong_trigger_protocol.py"
require_path "${REPO_ROOT}/easyvolcap/official_vggt/layers/dsa_attention.py"
require_path "${REPO_ROOT}/easyvolcap/official_vggt/layers/indexer.py"
require_path "${REPO_ROOT}/easyvolcap/utils/custom_indexer"
require_path "${REPO_ROOT}/easyvolcap/utils/custom_flash_attn"
require_path "${REPO_ROOT}/aidi/third_party/pi3_training/pi3/models/pi3_training.py"
require_path "${HANDOFF_ROOT}/configs/eval/vggt_geoweave_final_trusted.yaml"
require_path "${HANDOFF_ROOT}/configs/eval/vggt_geoweave_smoke_eth3d_pose.yaml"
require_path "${HANDOFF_ROOT}/configs/eval/pi3_geoweave_final_all_tasks.yaml"
require_path "${HANDOFF_ROOT}/configs/eval/pi3_geoweave_smoke_eth3d_pose.yaml"

bash -n "${SCRIPT_DIR}/train_vggt.sh"
bash -n "${SCRIPT_DIR}/train_pi3.sh"
bash -n "${SCRIPT_DIR}/eval_unified.sh"
bash -n "${SCRIPT_DIR}/eval_vggt.sh"
bash -n "${SCRIPT_DIR}/eval_pi3.sh"
bash -n "${SCRIPT_DIR}/robustness_pi3.sh"
bash -n "${REPO_ROOT}/aidi/scripts/vggt/train_official_vggt.sh"
bash -n "${REPO_ROOT}/aidi/scripts/vggt/submit_official_vggt_5090_warmup.sh"
bash -n "${REPO_ROOT}/aidi/scripts/pi3/train_pi3_official.sh"
bash -n "${REPO_ROOT}/aidi/scripts/pi3/submit_pi3_5090_warmup.sh"
bash -n "${REPO_ROOT}/aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh"
bash -n "${REPO_ROOT}/aidi/scripts/vggt/run_co3dv2_official_upstream.sh"
bash -n "${REPO_ROOT}/aidi/scripts/vggt/run_eval_dtu_mv_recon_pi3_style.sh"
bash -n "${REPO_ROOT}/aidi/scripts/vggt/run_eval_eth3d_mv_recon_pi3_style.sh"
bash -n "${REPO_ROOT}/aidi/scripts/vggt/run_eval_7scenes_mv_recon_pi3_style.sh"
bash -n "${REPO_ROOT}/aidi/scripts/vggt/run_eval_nrgbd_mv_recon_pi3_style.sh"

"${PYTHON_CMD}" - <<'PY' "${REPO_ROOT}" "${HANDOFF_ROOT}"
import ast
import pathlib
import sys
import yaml

repo = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(repo))
for rel in [
    "aidi/scripts/vggt/run_unified_eval.py",
    "aidi/scripts/vggt/eval_co3d_pose_official_upstream.py",
    "aidi/scripts/vggt/eval_da3_pose_benchmark.py",
    "aidi/scripts/vggt/eval_relpose_1500_benchmark.py",
    "aidi/scripts/vggt/eval_vggt_re10k_pose_lightweight.py",
    "aidi/scripts/vggt/resolve_co3dv2_official_scene_roots.py",
    "aidi/scripts/vggt/vggt_omega_eval_utils.py",
    "aidi/scripts/baselines/eval_config_utils.py",
    "aidi/scripts/baselines/eval_pi3_co3d_pose_official.py",
    "aidi/scripts/baselines/eval_pi3_depth_protocol.py",
    "aidi/scripts/baselines/eval_pi3_monodepth_protocol.py",
    "aidi/scripts/baselines/eval_pi3_relpose_distance_protocol.py",
    "aidi/scripts/baselines/eval_pi3_mv_recon_core.py",
    "aidi/scripts/baselines/eval_pi3_re10k_pose_official.py",
    "aidi/scripts/baselines/eval_pi3_videodepth_protocol.py",
    "aidi/scripts/baselines/exr_read_utils.py",
    "aidi/scripts/baselines/generate_seq_id_map.py",
    "aidi/scripts/baselines/pi3_checkpoint_loader.py",
    "aidi/scripts/baselines/pi3_metric_utils.py",
    "aidi/scripts/baselines/prepare_eth3d_pi3_style.py",
    "aidi/scripts/baselines/overlap_noise_seq_map_utils.py",
    "aidi/scripts/baselines/build_evc_same_scene_candidate_pool_benchmark.py",
    "aidi/scripts/baselines/build_fixed10_overlap_band_benchmark.py",
    "aidi/scripts/baselines/build_same_scene_low_overlap_5plus5_benchmark.py",
    "aidi/scripts/baselines/build_three_dataset_stride4_distractor_benchmark.py",
    "aidi/scripts/baselines/build_driving_multicamera_all_eval_benchmark.py",
    "aidi/scripts/baselines/build_waymo_plausible_context_diagnostic.py",
    "aidi/scripts/baselines/build_waymo_plausible_wrong_trigger_protocol.py",
    "aidi/scripts/baselines/summarize_waymo_plausible_wrong_trigger_protocol.py",
]:
    path = repo / rel
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in (root / "anonymous_model" / "geoweave_model").glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for rel in [
    "configs/eval/vggt_geoweave_final_trusted.yaml",
    "configs/eval/vggt_geoweave_smoke_eth3d_pose.yaml",
    "configs/eval/pi3_geoweave_final_all_tasks.yaml",
    "configs/eval/pi3_geoweave_smoke_eth3d_pose.yaml",
]:
    path = root / rel
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "model" not in data:
        raise SystemExit(f"missing model block in {path}")

from aidi.scripts.vggt.run_unified_eval import load_yaml_config, normalize_config, select_task_ids, task_value

WEIGHT_ROOT = "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/trained_model/geoweave_paper_handoff_20260602_final"

def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)

def load(rel: str):
    return normalize_config(load_yaml_config(root / rel), repo_root=repo)

vggt_paper = load("configs/eval/vggt_geoweave_final_trusted.yaml")
vggt_paper_tasks = select_task_ids(vggt_paper)
expect(vggt_paper["model"]["kind"] == "custom", "VGGT paper config must use custom GeoWeave checkpoint.")
expect(str(vggt_paper["model"]["checkpoint"]) == f"{WEIGHT_ROOT}/vggt_geoweave_20260503_pt79/79.pt", "VGGT paper checkpoint path drifted.")
expect(str(vggt_paper["model"]["config"]).endswith("aidi/configs/vggt/finetune_5090_sparse_topk1024_l9_19_p34_record.yaml"), "VGGT paper model config drifted.")
expect(len(vggt_paper_tasks) == 24, f"VGGT paper task count drifted: {len(vggt_paper_tasks)}")
expect("eth3d_pointcloud" in vggt_paper_tasks, "VGGT paper pointcloud task is not enabled.")
expect(int(task_value(vggt_paper, "eth3d_pose", "max_frames", 100)) == 100, "VGGT paper ETH3D pose must use max_frames=100.")

vggt_smoke = load("configs/eval/vggt_geoweave_smoke_eth3d_pose.yaml")
expect(select_task_ids(vggt_smoke) == ["eth3d_pose"], "VGGT smoke config must only enable eth3d_pose.")
expect(int(task_value(vggt_smoke, "eth3d_pose", "max_frames", 0)) == 12, "VGGT smoke ETH3D pose must use max_frames=12.")

pi3_paper = load("configs/eval/pi3_geoweave_final_all_tasks.yaml")
pi3_paper_tasks = select_task_ids(pi3_paper)
expect(pi3_paper["model"]["kind"] == "pi3", "Pi3 paper config must use native Pi3 GeoWeave.")
expect(str(pi3_paper["model"]["checkpoint"]) == f"{WEIGHT_ROOT}/pi3_geoweave_native_sparse_20260505_checkpoint_79/checkpoint_79/pytorch_model.bin", "Pi3 paper checkpoint path drifted.")
expect(str(pi3_paper["model"]["pi3_config"]) == f"{WEIGHT_ROOT}/pi3_geoweave_native_sparse_20260505_checkpoint_79/config.yaml", "Pi3 paper config path drifted.")
expect(str(pi3_paper["model"]["pi3_model_impl"]) == "native_sparse", "Pi3 paper model implementation drifted.")
expect(len(pi3_paper_tasks) == 30, f"Pi3 paper task count drifted: {len(pi3_paper_tasks)}")
expect("eth3d_pointcloud" in pi3_paper_tasks, "Pi3 paper pointcloud task is not enabled.")
expect(int(task_value(pi3_paper, "eth3d_pose", "max_frames", 100)) == 100, "Pi3 paper ETH3D pose must use max_frames=100 by default.")

pi3_smoke = load("configs/eval/pi3_geoweave_smoke_eth3d_pose.yaml")
expect(select_task_ids(pi3_smoke) == ["eth3d_pose"], "Pi3 smoke config must only enable eth3d_pose.")
expect(int(task_value(pi3_smoke, "eth3d_pose", "max_frames", 0)) == 12, "Pi3 smoke ETH3D pose must use max_frames=12.")
PY

bash "${SCRIPT_DIR}/train_vggt.sh" --smoke --dry-run >/dev/null
bash "${SCRIPT_DIR}/train_vggt.sh" --warmup-smoke --dry-run >/dev/null
bash "${SCRIPT_DIR}/train_pi3.sh" --smoke --dry-run >/dev/null
bash "${SCRIPT_DIR}/train_pi3.sh" --warmup-smoke --dry-run >/dev/null
bash "${SCRIPT_DIR}/eval_vggt.sh" --smoke --dry-run >/dev/null
bash "${SCRIPT_DIR}/eval_pi3.sh" --smoke --dry-run >/dev/null
bash "${SCRIPT_DIR}/robustness_pi3.sh" --list >/dev/null
bash "${SCRIPT_DIR}/robustness_pi3.sh" --eval-weak-scannetpp --model geoweave --dry-run >/dev/null
bash "${SCRIPT_DIR}/robustness_pi3.sh" --eval-weak-waymo --model geoweave --dry-run >/dev/null
bash "${SCRIPT_DIR}/robustness_pi3.sh" --eval-distractor-waymo --model geoweave --dry-run >/dev/null
bash "${SCRIPT_DIR}/robustness_pi3.sh" --summarize-distractor-waymo --dry-run >/dev/null

if PYTHONPATH="${HANDOFF_ROOT}/anonymous_model:${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_CMD}" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
    PYTHONPATH="${HANDOFF_ROOT}/anonymous_model:${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_CMD}" - <<'PY'
import geoweave_model

required = {
    "GeoWeaveConfig",
    "GeoWeavePi3",
    "GeoWeaveVGGT",
    "build_geoweave_pi3",
    "build_geoweave_vggt",
}
missing = sorted(name for name in required if not hasattr(geoweave_model, name))
if missing:
    raise SystemExit(f"missing anonymous model exports: {missing}")
PY
else
    if [ "${STRICT_IMPORT}" = "1" ]; then
        echo "[ERROR] torch is not installed; cannot runtime-import anonymous GeoWeave model." >&2
        exit 1
    fi
    echo "[WARN] torch is not installed; skipped anonymous model runtime import. Re-run with STRICT_IMPORT=1 in the training environment."
fi

echo "[OK] GeoWeave handoff entry points are present and syntactically valid."
