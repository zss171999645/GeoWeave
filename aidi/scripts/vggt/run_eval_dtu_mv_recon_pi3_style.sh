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

is_valid_dtu_root() {
  local root="$1"
  local seq_map="${DTU_SEQ_MAP_ASSET:-${REPO_ROOT}/aidi/assets/pi3_seq_id_maps/DTU_mv-recon_seq-id-map-kf5.json}"
  [ -d "${root}" ] || return 1
  [ -f "${seq_map}" ] || return 1
  python3 - "${root}" "${seq_map}" <<'PY' >/dev/null 2>&1
import json
import os
import sys

root, seq_map = sys.argv[1:3]
with open(seq_map, "r") as f:
    required_scans = sorted(json.load(f).keys())

def valid_seq_dir(seq_dir: str) -> bool:
    return (
        os.path.isdir(os.path.join(seq_dir, "cams"))
        and os.path.isdir(os.path.join(seq_dir, "binary_masks"))
    ) or os.path.isdir(os.path.join(seq_dir, "cameras", "00"))

for scan in required_scans:
    seq_dir = os.path.join(root, scan)
    if not os.path.isdir(seq_dir) or not valid_seq_dir(seq_dir):
        raise SystemExit(1)
PY
}

first_valid_dtu_root() {
  local p=""
  for p in "$@"; do
    [ -n "${p}" ] || continue
    if is_valid_dtu_root "${p}"; then
      printf '%s\n' "${p}"
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
if [ -n "${VGGT_CHECKPOINT}" ]; then
  MODEL_TAG_EFFECTIVE="custom"
elif [ "${MODEL_TAG}" = "pt34" ]; then
  MODEL_TAG_EFFECTIVE="custom"
  VGGT_CHECKPOINT="${LEGACY_VGGT_PT34_CKPT}"
else
  MODEL_TAG_EFFECTIVE="official"
fi
if [ -z "${VGGT_INPUT_STYLE:-}" ]; then
  if [ "${MODEL_TAG_EFFECTIVE}" = "official" ]; then
    VGGT_INPUT_STYLE="pi3_resize"
  else
    VGGT_INPUT_STYLE="official_crop"
  fi
fi
DEVICE="${DEVICE:-auto}"
LOAD_IMG_SIZE="${LOAD_IMG_SIZE:-518}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
DTU_UNIT_SCALE="${DTU_UNIT_SCALE:-1.0}"
DTU_CENTER_CROP_HEIGHT="${DTU_CENTER_CROP_HEIGHT:-0}"
DTU_DATA_FORMAT="${DTU_DATA_FORMAT:-raw}"

DATASET_ROOT="${DATASET_ROOT:-}"
DTU_SHARED_FULL_ROOT="${DTU_SHARED_FULL_ROOT:-/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/dtu_test_mvsnet_release_full}"
if [ -z "${DATASET_ROOT}" ]; then
  DATASET_ROOT="$(
    first_valid_dtu_root \
      "${DTU_SHARED_FULL_ROOT}" \
      "${REPO_ROOT}/tmp/dtu_test_mvsnet_release" \
      "${REPO_PARENT}/dtu_test_mvsnet_release" \
      "/home/feng01.zhou/tmp/dtu_test_mvsnet_release" \
      "${HOME}/tmp/dtu_test_mvsnet_release" \
      "/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/dtu_test_mvsnet_release" || true
  )"
fi
PI3_ROOT="${PI3_ROOT:-${REPO_ROOT}/tmp/external_refs/pi3-official2}"
VGGT_OFFICIAL_CKPT_ROOT="${VGGT_OFFICIAL_CKPT_ROOT:-/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B}"
VGGT_CONFIG="${VGGT_CONFIG:-}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/tmp/dtu_mv_recon_pi3_style_${MODEL_FAMILY}_${MODEL_TAG_EFFECTIVE}_$(date +%Y%m%d_%H%M%S)}"

if [ "${DEVICE}" = "auto" ]; then
  DEVICE="$(pick_auto_device)"
fi

echo "[dtu-mv-recon-pi3-style] repo=${REPO_ROOT}"
echo "[dtu-mv-recon-pi3-style] model_family=${MODEL_FAMILY} model_tag=${MODEL_TAG_EFFECTIVE} point_source=${POINT_SOURCE}"
echo "[dtu-mv-recon-pi3-style] vggt_input_style=${VGGT_INPUT_STYLE}"
echo "[dtu-mv-recon-pi3-style] device=${DEVICE}"
echo "[dtu-mv-recon-pi3-style] dataset_root=${DATASET_ROOT}"
echo "[dtu-mv-recon-pi3-style] pi3_root=${PI3_ROOT}"
echo "[dtu-mv-recon-pi3-style] output_dir=${OUTPUT_DIR}"
echo "[dtu-mv-recon-pi3-style] max_sequences=${MAX_SEQUENCES}"
echo "[dtu-mv-recon-pi3-style] dtu_unit_scale=${DTU_UNIT_SCALE}"
echo "[dtu-mv-recon-pi3-style] dtu_center_crop_height=${DTU_CENTER_CROP_HEIGHT}"
echo "[dtu-mv-recon-pi3-style] dtu_data_format=${DTU_DATA_FORMAT}"

if [ -z "${DATASET_ROOT}" ] || ! is_valid_dtu_root "${DATASET_ROOT}"; then
  echo "DTU dataset root not found. Set DATASET_ROOT to the official raw DTU full root, or prepare tmp/dtu_test_mvsnet_release"
  exit 1
fi

exec python3 aidi/scripts/baselines/eval_pi3_mv_recon_core.py \
  --dataset dtu \
  --protocol auto \
  --model-family "${MODEL_FAMILY}" \
  --dataset-root "${DATASET_ROOT}" \
  --pi3-root "${PI3_ROOT}" \
  --vggt-model-tag "${MODEL_TAG_EFFECTIVE}" \
  --vggt-config "${VGGT_CONFIG}" \
  --vggt-official-ckpt-root "${VGGT_OFFICIAL_CKPT_ROOT}" \
  --vggt-pt34-ckpt "${LEGACY_VGGT_PT34_CKPT}" \
  --vggt-checkpoint "${VGGT_CHECKPOINT}" \
  --point-source "${POINT_SOURCE}" \
  --vggt-input-style "${VGGT_INPUT_STYLE}" \
  --load-img-size "${LOAD_IMG_SIZE}" \
  --device "${DEVICE}" \
  --dtu-unit-scale "${DTU_UNIT_SCALE}" \
  --dtu-center-crop-height "${DTU_CENTER_CROP_HEIGHT}" \
  --dtu-data-format "${DTU_DATA_FORMAT}" \
  --max-sequences "${MAX_SEQUENCES}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
