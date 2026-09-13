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
import os
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

cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if cvd is not None and cvd.strip():
    visible_phys = [int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()]
    rows = [r for r in rows if r[0] in visible_phys]
    phys_to_logical = {phys: logical for logical, phys in enumerate(visible_phys)}
else:
    phys_to_logical = {r[0]: r[0] for r in rows}

preferred = [r for r in rows if r[1] <= 1024 and r[3] == 0]
pool = preferred if preferred else sorted(rows, key=lambda r: (r[1], r[3], -r[2]))
if not pool:
    print("cpu")
else:
    print(f"cuda:{phys_to_logical[pool[0][0]]}")
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

json_has_keys() {
  local path="$1"
  [ -f "${path}" ] || return 1
  python3 - "${path}" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1], "r") as f:
    obj = json.load(f)
if isinstance(obj, dict) and len(obj) > 0:
    raise SystemExit(0)
raise SystemExit(1)
PY
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
LOAD_IMG_SIZE="${LOAD_IMG_SIZE:-518}"
VGGT_INPUT_STYLE="${VGGT_INPUT_STYLE:-official_crop}"
DEVICE="${DEVICE:-auto}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"
PROTOCOL="${PROTOCOL:-auto}"

DATASET_ROOT="${DATASET_ROOT:-}"
NRGBD_ROOT_PRIMARY="/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/zinan.lv/dataset_point/NRGBD"

if [ -z "${DATASET_ROOT}" ]; then
  DATASET_ROOT="$(
    first_existing_dir \
      "${NRGBD_ROOT_PRIMARY}" \
      "${REPO_ROOT}/tmp/nrgbd" \
      "${REPO_PARENT}/nrgbd" || true
  )"
fi
PI3_ROOT="${PI3_ROOT:-${REPO_ROOT}/tmp/external_refs/pi3-official2}"
SEQ_MAP="${SEQ_MAP:-}"
VGGT_OFFICIAL_CKPT_ROOT="${VGGT_OFFICIAL_CKPT_ROOT:-/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B}"
VGGT_CONFIG="${VGGT_CONFIG:-}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/tmp/nrgbd_mv_recon_pi3_style_${MODEL_FAMILY}_${MODEL_TAG_EFFECTIVE}_$(date +%Y%m%d_%H%M%S)}"

if [ "${DEVICE}" = "auto" ]; then
  DEVICE="$(pick_auto_device)"
fi

if [ -z "${DATASET_ROOT}" ] || [ ! -d "${DATASET_ROOT}" ]; then
  echo "[nrgbd-mv-recon-pi3-style] ERROR: dataset root not found. Set DATASET_ROOT."
  echo "  Tried: ${NRGBD_ROOT_PRIMARY}"
  exit 1
fi

LOCAL_SEQ_MAP_DIR="${REPO_ROOT}/aidi/assets/pi3_seq_id_maps"
PI3_SEQ_MAP_DIR="${PI3_ROOT}/datasets/seq-id-maps"
if [ -z "${SEQ_MAP}" ]; then
  need_gen=false
  if [ "${PROTOCOL}" = "sparse" ] || [ "${PROTOCOL}" = "auto" ] || [ "${PROTOCOL}" = "both" ]; then
    sparse_local="${LOCAL_SEQ_MAP_DIR}/NRGBD_mv-recon_seq-id-map-kf500.json"
    sparse_pi3="${PI3_SEQ_MAP_DIR}/NRGBD_mv-recon_seq-id-map-kf500.json"
    if ! json_has_keys "${sparse_local}" && ! json_has_keys "${sparse_pi3}"; then
      need_gen=true
    fi
  fi
  if [ "${PROTOCOL}" = "dense" ] || [ "${PROTOCOL}" = "auto" ] || [ "${PROTOCOL}" = "both" ]; then
    dense_local="${LOCAL_SEQ_MAP_DIR}/NRGBD_mv-recon_seq-id-map-kf100.json"
    dense_pi3="${PI3_SEQ_MAP_DIR}/NRGBD_mv-recon_seq-id-map-kf100.json"
    if ! json_has_keys "${dense_local}" && ! json_has_keys "${dense_pi3}"; then
      need_gen=true
    fi
  fi

  if [ "${need_gen}" = true ]; then
    mkdir -p "${LOCAL_SEQ_MAP_DIR}"
    python3 aidi/scripts/baselines/generate_seq_id_map.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset nrgbd \
      --kf-step 500 \
      --max-views 10 \
      --output "${LOCAL_SEQ_MAP_DIR}/NRGBD_mv-recon_seq-id-map-kf500.json"
    python3 aidi/scripts/baselines/generate_seq_id_map.py \
      --dataset-root "${DATASET_ROOT}" \
      --dataset nrgbd \
      --kf-step 100 \
      --max-views 10 \
      --output "${LOCAL_SEQ_MAP_DIR}/NRGBD_mv-recon_seq-id-map-kf100.json"
  fi
fi

CMD_ARGS=(
  --dataset nrgbd
  --protocol "${PROTOCOL}"
  --model-family "${MODEL_FAMILY}"
  --dataset-root "${DATASET_ROOT}"
  --pi3-root "${PI3_ROOT}"
  --vggt-model-tag "${MODEL_TAG_EFFECTIVE}"
  --vggt-config "${VGGT_CONFIG}"
  --vggt-official-ckpt-root "${VGGT_OFFICIAL_CKPT_ROOT}"
  --vggt-pt34-ckpt "${LEGACY_VGGT_PT34_CKPT}"
  --vggt-checkpoint "${VGGT_CHECKPOINT}"
  --point-source "${POINT_SOURCE}"
  --vggt-input-style "${VGGT_INPUT_STYLE}"
  --load-img-size "${LOAD_IMG_SIZE}"
  --device "${DEVICE}"
  --max-sequences "${MAX_SEQUENCES}"
  --output-dir "${OUTPUT_DIR}"
)

if [ -n "${SEQ_MAP}" ]; then
  CMD_ARGS+=(--seq-map "${SEQ_MAP}")
fi

exec python3 aidi/scripts/baselines/eval_pi3_mv_recon_core.py "${CMD_ARGS[@]}" "$@"
