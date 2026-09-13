#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_PARENT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

pick_auto_device() {
  python3 - <<'PY'
import subprocess

try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception:
    print("cpu")
    raise SystemExit(0)

rows = []
for line in out.strip().splitlines():
    idx, mem_used, mem_total, util = [x.strip() for x in line.split(",")]
    rows.append((int(idx), int(mem_used), int(mem_total), int(util)))

preferred = [r for r in rows if r[1] <= 1024 and r[3] == 0]
pool = preferred if preferred else sorted(rows, key=lambda r: (r[1], r[3], -r[2]))
if not pool:
    print("cpu")
else:
    print(f"cuda:{pool[0][0]}")
PY
}

first_existing_dir() {
  local p=""
  for p in "$@"; do
    [ -n "${p}" ] || continue
    if [ -d "${p}" ]; then
      printf '%s\n' "${p}"
      return 0
    fi
  done
  return 1
}

first_existing_file() {
  local p=""
  for p in "$@"; do
    [ -n "${p}" ] || continue
    if [ -f "${p}" ]; then
      printf '%s\n' "${p}"
      return 0
    fi
  done
  return 1
}

is_valid_eth3d_root() {
  local root="$1"
  local seq_map="$2"
  [ -d "${root}" ] || return 1
  [ -f "${seq_map}" ] || return 1
  python3 - "${root}" "${seq_map}" <<'PY' >/dev/null 2>&1
import json
import os
import sys

root, seq_map = sys.argv[1:3]
with open(seq_map, "r") as f:
    required_scenes = sorted(json.load(f).keys())

def has_valid_files(path, suffix=None):
    for name in os.listdir(path):
        if name.startswith("."):
            continue
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        if suffix is not None and not name.endswith(suffix):
            continue
        return True
    return False

for scene in required_scenes:
    scene_dir = os.path.join(root, scene)
    image_dir = os.path.join(scene_dir, "images", "custom_undistorted")
    depth_dir = os.path.join(scene_dir, "ground_truth_depth", "custom_undistorted")
    cam_dir = os.path.join(scene_dir, "custom_undistorted_cam")
    if not (os.path.isdir(scene_dir) and os.path.isdir(image_dir) and os.path.isdir(depth_dir) and os.path.isdir(cam_dir)):
        raise SystemExit(1)
    if not has_valid_files(image_dir, ".JPG"):
        raise SystemExit(1)
    if not has_valid_files(depth_dir, ".JPG"):
        raise SystemExit(1)
    if not has_valid_files(cam_dir, ".npz"):
        raise SystemExit(1)
PY
}

pick_valid_eth3d_root_and_seqmap() {
  local args=("$@")
  local half=$((${#args[@]} / 2))
  local i=""
  for ((i = 0; i < half; ++i)); do
    local root="${args[$i]}"
    local seq="${args[$((i + half))]}"
    [ -n "${root}" ] || continue
    [ -n "${seq}" ] || continue
    if is_valid_eth3d_root "${root}" "${seq}"; then
      printf '%s\n%s\n' "${root}" "${seq}"
      return 0
    fi
  done
  return 1
}

MODEL_FAMILY="${MODEL_FAMILY:-vggt}"
MODEL_TAG="${MODEL_TAG:-official}"
POINT_SOURCE="${POINT_SOURCE:-native}"
LEGACY_VGGT_PT34_CKPT="${VGGT_PT34_CKPT:-/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/trained_model/vggt/official/finetune_5090_sparse_20260226_0139_resume_finetune_5090_sparse_topk2048-layers9_19-resumept30-fp16-pointhead/34.pt}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-}"
if [ "${MODEL_FAMILY}" = "vggt_omega" ]; then
  MODEL_TAG_EFFECTIVE="omega"
  MODEL_TAG_ARG="official"
elif [ -n "${VGGT_CHECKPOINT}" ]; then
  MODEL_TAG_EFFECTIVE="custom"
  MODEL_TAG_ARG="custom"
elif [ "${MODEL_TAG}" = "pt34" ]; then
  MODEL_TAG_EFFECTIVE="custom"
  MODEL_TAG_ARG="pt34"
  VGGT_CHECKPOINT="${LEGACY_VGGT_PT34_CKPT}"
else
  MODEL_TAG_EFFECTIVE="official"
  MODEL_TAG_ARG="official"
fi
DEVICE="${DEVICE:-auto}"
LOAD_IMG_SIZE="${LOAD_IMG_SIZE:-518}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"

ETH3D_RAW_ROOT="${ETH3D_RAW_ROOT:-/horizon-bucket/saturn_v_4dlabel/008_Simulation/001_users/junyuan.deng/evaluation_datasets/metricdepth/eth3d_full}"
ETH3D_SHARED_ROOT="${ETH3D_SHARED_ROOT:-/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/eth3d_pi3_style_root}"
ETH3D_SHARED_SEQMAP="${ETH3D_SHARED_SEQMAP:-/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/eth3d_pi3_style_seqmap.json}"

DATASET_ROOT="${DATASET_ROOT:-}"
SEQ_MAP="${SEQ_MAP:-}"
if [ -z "${DATASET_ROOT}" ] && [ -z "${SEQ_MAP}" ]; then
  mapfile -t ETH3D_SELECTED < <(
    pick_valid_eth3d_root_and_seqmap \
      "${ETH3D_SHARED_ROOT}" \
      "${REPO_ROOT}/tmp/eth3d_pi3_style_root" \
      "${REPO_PARENT}/eth3d_pi3_style_root" \
      "/home/feng01.zhou/tmp/eth3d_pi3_style_root" \
      "${ETH3D_SHARED_SEQMAP}" \
      "${REPO_ROOT}/tmp/eth3d_pi3_style_seqmap.json" \
      "${REPO_PARENT}/eth3d_pi3_style_seqmap.json" \
      "/home/feng01.zhou/tmp/eth3d_pi3_style_seqmap.json" || true
  )
  if [ "${#ETH3D_SELECTED[@]}" -ge 2 ]; then
    DATASET_ROOT="${ETH3D_SELECTED[0]}"
    SEQ_MAP="${ETH3D_SELECTED[1]}"
  fi
else
  if [ -z "${DATASET_ROOT}" ]; then
    DATASET_ROOT="$(
      first_existing_dir \
        "${ETH3D_SHARED_ROOT}" \
        "${REPO_ROOT}/tmp/eth3d_pi3_style_root" \
        "/home/feng01.zhou/tmp/eth3d_pi3_style_root" || true
    )"
  fi
  if [ -z "${SEQ_MAP}" ]; then
    SEQ_MAP="$(
      first_existing_file \
        "${ETH3D_SHARED_SEQMAP}" \
        "${REPO_ROOT}/tmp/eth3d_pi3_style_seqmap.json" \
        "/home/feng01.zhou/tmp/eth3d_pi3_style_seqmap.json" || true
    )"
  fi
fi
PI3_ROOT="${PI3_ROOT:-${REPO_ROOT}/tmp/external_refs/pi3-official2}"
VGGT_OFFICIAL_CKPT_ROOT="${VGGT_OFFICIAL_CKPT_ROOT:-/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B}"
VGGT_CONFIG="${VGGT_CONFIG:-}"
VGGT_OMEGA_REPO="${VGGT_OMEGA_REPO:-}"
VGGT_OMEGA_CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-}"
VGGT_OMEGA_RESOLUTION="${VGGT_OMEGA_RESOLUTION:-512}"
VGGT_OMEGA_MODE="${VGGT_OMEGA_MODE:-balanced}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/tmp/eth3d_mv_recon_pi3_style_${MODEL_FAMILY}_${MODEL_TAG_EFFECTIVE}_$(date +%Y%m%d_%H%M%S)}"

if [ "${DEVICE}" = "auto" ]; then
  DEVICE="$(pick_auto_device)"
fi

echo "[eth3d-mv-recon-pi3-style] repo=${REPO_ROOT}"
echo "[eth3d-mv-recon-pi3-style] model_family=${MODEL_FAMILY} model_tag=${MODEL_TAG_EFFECTIVE} point_source=${POINT_SOURCE}"
if [ "${MODEL_FAMILY}" = "vggt_omega" ]; then
  echo "[eth3d-mv-recon-pi3-style] vggt_omega_repo=${VGGT_OMEGA_REPO}"
  echo "[eth3d-mv-recon-pi3-style] vggt_omega_checkpoint=${VGGT_OMEGA_CHECKPOINT}"
fi
echo "[eth3d-mv-recon-pi3-style] device=${DEVICE}"
echo "[eth3d-mv-recon-pi3-style] dataset_root=${DATASET_ROOT}"
echo "[eth3d-mv-recon-pi3-style] seq_map=${SEQ_MAP}"
echo "[eth3d-mv-recon-pi3-style] pi3_root=${PI3_ROOT}"
echo "[eth3d-mv-recon-pi3-style] output_dir=${OUTPUT_DIR}"
echo "[eth3d-mv-recon-pi3-style] max_sequences=${MAX_SEQUENCES}"

if [ -z "${DATASET_ROOT}" ] || [ -z "${SEQ_MAP}" ] || ! is_valid_eth3d_root "${DATASET_ROOT}" "${SEQ_MAP}"; then
  mkdir -p "${REPO_ROOT}/tmp"
  DATASET_ROOT="${REPO_ROOT}/tmp/eth3d_pi3_style_root"
  SEQ_MAP="${REPO_ROOT}/tmp/eth3d_pi3_style_seqmap.json"
  echo "[eth3d-mv-recon-pi3-style] prepared root or seq_map missing/incomplete, auto-preparing from raw root=${ETH3D_RAW_ROOT}"
  python3 aidi/scripts/baselines/prepare_eth3d_pi3_style.py \
    --raw-root "${ETH3D_RAW_ROOT}" \
    --output-root "${DATASET_ROOT}" \
    --seq-map "${REPO_ROOT}/aidi/assets/pi3_seq_id_maps/ETH3D_mv-recon_seq-id-map-kf5.json" \
    --write-remapped-seq-map "${SEQ_MAP}" \
    --skip-existing
fi

EXTRA_ARGS=()
if [ "${MODEL_FAMILY}" = "vggt_omega" ]; then
  EXTRA_ARGS+=(
    --vggt-omega-repo "${VGGT_OMEGA_REPO}"
    --vggt-omega-checkpoint "${VGGT_OMEGA_CHECKPOINT}"
    --vggt-omega-resolution "${VGGT_OMEGA_RESOLUTION}"
    --vggt-omega-mode "${VGGT_OMEGA_MODE}"
  )
fi

exec python3 aidi/scripts/baselines/eval_pi3_mv_recon_core.py \
  --dataset eth3d \
  --protocol auto \
  --model-family "${MODEL_FAMILY}" \
  --dataset-root "${DATASET_ROOT}" \
  --seq-map "${SEQ_MAP}" \
  --pi3-root "${PI3_ROOT}" \
  --vggt-model-tag "${MODEL_TAG_ARG}" \
  --vggt-config "${VGGT_CONFIG}" \
  --vggt-official-ckpt-root "${VGGT_OFFICIAL_CKPT_ROOT}" \
  --vggt-pt34-ckpt "${LEGACY_VGGT_PT34_CKPT}" \
  --vggt-checkpoint "${VGGT_CHECKPOINT}" \
  --point-source "${POINT_SOURCE}" \
  --load-img-size "${LOAD_IMG_SIZE}" \
  --device "${DEVICE}" \
  --max-sequences "${MAX_SEQUENCES}" \
  --output-dir "${OUTPUT_DIR}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  "$@"
