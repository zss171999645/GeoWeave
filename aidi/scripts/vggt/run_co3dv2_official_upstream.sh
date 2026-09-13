#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "python3/python not found"
  exit 1
fi

CHECKPOINT=${CHECKPOINT:-""}

CO3D_RAW_ROOT_DEFAULT="/horizon-bucket/saturn_v_4dlabel/009_geo/002_data/dust3r_datasets/dust3r_extracted/co3dv2"
CO3D_IMAGE_ROOT=${CO3D_IMAGE_ROOT:-"${CO3D_RAW_ROOT_DEFAULT}"}
CO3D_SETLIST_ROOT=${CO3D_SETLIST_ROOT:-"${CO3D_RAW_ROOT_DEFAULT}"}
SELECTION_SOURCE=${SELECTION_SOURCE:-"co3d_setlists"} # co3d_setlists | hf_jgz
HF_ANNO_DIR=${HF_ANNO_DIR:-""}
SUBSET=${SUBSET:-"seen41"}
SPLIT=${SPLIT:-"test"}
MIN_NUM_IMAGES=${MIN_NUM_IMAGES:-"50"}
NUM_FRAMES=${NUM_FRAMES:-"10"}
SEED=${SEED:-"0"}
MIN_QUALITY=${MIN_QUALITY:-"0.5"}
SET_LIST_TAG=${SET_LIST_TAG:-"fewview_dev"}
DEFAULT_ANNO_DIR="/tmp/co3d_official_annos_${SELECTION_SOURCE}_${SUBSET}_${SPLIT}"
CO3D_SHARED_ANNO_DIR=${CO3D_SHARED_ANNO_DIR:-"/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/eval_datasets/co3d_official_annos_co3d_setlists_seen41_test"}
CO3D_SHARED_IMAGE_ROOT=${CO3D_SHARED_IMAGE_ROOT:-""}
CO3D_SHARED_IMAGE_ANNO_DIR="${CO3D_SHARED_IMAGE_ROOT}/_meta/annos"
CO3D_SHARED_IMAGE_STATS_JSON="${CO3D_SHARED_IMAGE_ROOT}/_meta/stats.json"
ANNO_DIR=${ANNO_DIR:-"${DEFAULT_ANNO_DIR}"}
REBUILD_ANNO=${REBUILD_ANNO:-"0"}

DEVICE=${DEVICE:-"auto"}
IMAGE_MODE=${IMAGE_MODE:-"crop"}
LOAD_IMG_SIZE=${LOAD_IMG_SIZE:-"518"}
CONFIG=${CONFIG:-""}
OFFICIAL_CKPT_ROOT=${OFFICIAL_CKPT_ROOT:-"/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B"}
FAST_EVAL=${FAST_EVAL:-"0"}
CATEGORIES=${CATEGORIES:-""}
DEBUG_CATEGORY=${DEBUG_CATEGORY:-""}
OUTPUT=${OUTPUT:-""}

default_categories=(
  apple backpack banana baseballbat baseballglove bench bicycle bottle bowl broccoli
  cake car carrot cellphone chair cup donut hairdryer handbag hydrant keyboard laptop
  microwave motorcycle mouse orange parkingmeter pizza plant stopsign teddybear toaster
  toilet toybus toyplane toytrain toytruck tv umbrella vase wineglass
)

parse_target_categories() {
  local raw=""
  if [ -n "${DEBUG_CATEGORY}" ]; then
    printf '%s\n' "${DEBUG_CATEGORY}"
    return 0
  fi
  if [ -n "${CATEGORIES}" ]; then
    raw="${CATEGORIES//,/ }"
    printf '%s\n' ${raw}
    return 0
  fi
  printf '%s\n' "${default_categories[@]}"
}

using_default_official_setlists() {
  [ "${SELECTION_SOURCE}" = "co3d_setlists" ] || return 1
  [ "${SUBSET}" = "seen41" ] || return 1
  [ "${SPLIT}" = "test" ] || return 1
  [ "${MIN_NUM_IMAGES}" = "50" ] || return 1
  [ "${NUM_FRAMES}" = "10" ] || return 1
  [ "${SEED}" = "0" ] || return 1
  [ "${SET_LIST_TAG}" = "fewview_dev" ] || return 1
  [ "${MIN_QUALITY}" = "0.5" ] || return 1
  return 0
}

anno_dir_complete() {
  local dir_path="${1:-${ANNO_DIR}}"
  local cat=""
  [ -d "${dir_path}" ] || return 1
  while IFS= read -r cat; do
    [ -n "${cat}" ] || continue
    if [ ! -f "${dir_path}/${cat}_${SPLIT}.jgz" ]; then
      return 1
    fi
  done < <(parse_target_categories)
  return 0
}

shared_image_root_ready() {
  local image_root="${1:-${CO3D_SHARED_IMAGE_ROOT}}"
  [ -n "${image_root}" ] || return 1
  local stats_json="${image_root}/_meta/stats.json"
  local anno_dir="${image_root}/_meta/annos"
  [ -d "${image_root}" ] || return 1
  anno_dir_complete "${anno_dir}" || return 1
  [ -f "${stats_json}" ] || return 1
  "${PYTHON_BIN}" - "${stats_json}" <<'PY'
import json, sys
stats_path = sys.argv[1]
with open(stats_path, "r", encoding="utf-8") as f:
    stats = json.load(f)
if int(stats.get("failed_count", 1)) != 0:
    raise SystemExit(1)
expected = {
    "category_count": 41,
    "sequence_count": 2080,
    "selected_frame_count": 20800,
    "unique_file_count": 20800,
}
for key, value in expected.items():
    if int(stats.get(key, -1)) != value:
        raise SystemExit(1)
if stats.get("selection_source") not in {"hf_jgz", "co3d_setlists"}:
    raise SystemExit(1)
if stats.get("subset") != "seen41" or stats.get("split") != "test":
    raise SystemExit(1)
PY
}

if [ "${REBUILD_ANNO}" != "1" ] && using_default_official_setlists; then
  if shared_image_root_ready "${CO3D_SHARED_IMAGE_ROOT}"; then
    CO3D_IMAGE_ROOT="${CO3D_SHARED_IMAGE_ROOT}"
    ANNO_DIR="${CO3D_SHARED_IMAGE_ANNO_DIR}"
  elif [ "${ANNO_DIR}" = "${DEFAULT_ANNO_DIR}" ] && \
       ! anno_dir_complete "${ANNO_DIR}" && \
       anno_dir_complete "${CO3D_SHARED_ANNO_DIR}"; then
    ANNO_DIR="${CO3D_SHARED_ANNO_DIR}"
  fi
fi

if [ "${REBUILD_ANNO}" = "1" ] || ! anno_dir_complete "${ANNO_DIR}"; then
  mkdir -p "${ANNO_DIR}"
  resolver_args=(
    aidi/scripts/vggt/resolve_co3dv2_official_scene_roots.py
    --dataset-root "${CO3D_IMAGE_ROOT}"
    --selection-source "${SELECTION_SOURCE}"
    --split "${SPLIT}"
    --subset "${SUBSET}"
    --min-num-images "${MIN_NUM_IMAGES}"
    --num-frames "${NUM_FRAMES}"
    --seed "${SEED}"
    --output-anno-dir "${ANNO_DIR}"
    --print-format none
  )
  if [ "${SELECTION_SOURCE}" = "co3d_setlists" ]; then
    resolver_args+=(
      --co3d-v2-dir "${CO3D_SETLIST_ROOT}"
      --set-list-tag "${SET_LIST_TAG}"
      --min-quality "${MIN_QUALITY}"
    )
  else
    if [ -z "${HF_ANNO_DIR}" ]; then
      echo "HF_ANNO_DIR is required when SELECTION_SOURCE=hf_jgz"
      exit 1
    fi
    resolver_args+=(--anno-dir "${HF_ANNO_DIR}")
  fi
  "${PYTHON_BIN}" "${resolver_args[@]}"
fi

if [ "${DEVICE}" = "auto" ]; then
  DEVICE="$(pick_auto_device)"
fi

echo "[co3dv2_official_upstream] image_root=${CO3D_IMAGE_ROOT}"
echo "[co3dv2_official_upstream] anno_dir=${ANNO_DIR}"
echo "[co3dv2_official_upstream] device=${DEVICE}"
echo "[co3dv2_official_upstream] load_img_size=${LOAD_IMG_SIZE}"
echo "[co3dv2_official_upstream] selection_source=${SELECTION_SOURCE} subset=${SUBSET} split=${SPLIT} seed=${SEED} num_frames=${NUM_FRAMES}"

eval_args=(
  aidi/scripts/vggt/eval_co3d_pose_official_upstream.py
  --co3d-image-root "${CO3D_IMAGE_ROOT}"
  --co3d-anno-dir "${ANNO_DIR}"
  --seed "${SEED}"
  --num-frames "${NUM_FRAMES}"
  --min-num-images "${MIN_NUM_IMAGES}"
  --model-tag official
  --official-ckpt-root "${OFFICIAL_CKPT_ROOT}"
  --image-mode "${IMAGE_MODE}"
  --load-img-size "${LOAD_IMG_SIZE}"
  --anno-camera-convention pt3d_to_opencv
)

if [ -n "${CHECKPOINT}" ]; then
  eval_args+=(--model-path "${CHECKPOINT}")
fi

if [ -n "${DEVICE}" ]; then
  eval_args+=(--device "${DEVICE}")
fi
if [ -n "${CONFIG}" ]; then
  eval_args+=(--config "${CONFIG}")
fi
if [ -n "${CATEGORIES}" ]; then
  eval_args+=(--categories "${CATEGORIES}")
fi
if [ -n "${DEBUG_CATEGORY}" ]; then
  eval_args+=(--debug-category "${DEBUG_CATEGORY}")
fi
if [ "${FAST_EVAL}" = "1" ]; then
  eval_args+=(--fast-eval)
fi
if [ -n "${OUTPUT}" ]; then
  eval_args+=(--output "${OUTPUT}")
fi

"${PYTHON_BIN}" "${eval_args[@]}"
