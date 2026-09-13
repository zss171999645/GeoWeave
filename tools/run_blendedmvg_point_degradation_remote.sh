#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/cfs/zhoufeng/workspace/geoweave-rebuttal-repro-20260629}"
PYTHON="${PYTHON:-/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/cfs/datasets/BlendedMVG}"
TAG="${TAG:-rebuttal_blendedmvg_point_degradation_20260629}"
INPUT_ROOT="${INPUT_ROOT:-/mnt/cfs/zhoufeng/geoweave_repro_inputs/${TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cfs/zhoufeng/geoweave_repro_outputs/${TAG}}"
LOG_ROOT="${LOG_ROOT:-/mnt/cfs/zhoufeng/geoweave_repro_logs/${TAG}}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SCRIPT="${SCRIPT:-${REPO_ROOT}/tools/blendedmvg_point_degradation_protocol.py}"

mkdir -p "$INPUT_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"

"$PYTHON" "$SCRIPT" prepare \
  --dataset-root "$DATASET_ROOT" \
  --input-root "$INPUT_ROOT" \
  --num-samples "$NUM_SAMPLES" \
  --overwrite

PROTOCOL="${INPUT_ROOT}/protocol.json"
JOBS_TSV="${LOG_ROOT}/jobs.tsv"
"$PYTHON" "$SCRIPT" emit-jobs \
  --protocol "$PROTOCOL" \
  --output-root "$OUTPUT_ROOT" > "$JOBS_TSV"

total_jobs="$(wc -l < "$JOBS_TSV" | tr -d ' ')"
echo "[launcher] jobs=${total_jobs} max_parallel=${MAX_PARALLEL}"

running=0
job_index=0
failed=0

while IFS=$'\t' read -r model sample_id variant input_dir output_dir; do
  mkdir -p "$output_dir"
  gpu="$((job_index % 8))"
  log_file="${LOG_ROOT}/${job_index}_${model}_${sample_id}_${variant}.log"
  echo "[launcher] gpu=${gpu} model=${model} sample=${sample_id} variant=${variant}"
  (
    cd "$REPO_ROOT"
    if [[ "$model" == "geoweave_final" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" tools/run_pi3_geoweave_smoke_infer.py \
        --input-dir "$input_dir" \
        --output-dir "$output_dir" \
        --max-images 10 \
        --load-img-size 224 \
        --point-source native \
        --point-stride 16
    elif [[ "$model" == "pi3_base" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" tools/run_pi3_base_smoke_infer.py \
        --input-dir "$input_dir" \
        --output-dir "$output_dir" \
        --max-images 10 \
        --load-img-size 518 \
        --point-source native \
        --point-stride 16
    else
      echo "Unknown model: $model" >&2
      exit 2
    fi
  ) > "$log_file" 2>&1 &

  running="$((running + 1))"
  job_index="$((job_index + 1))"
  if (( running >= MAX_PARALLEL )); then
    if ! wait -n; then
      failed=1
    fi
    running="$((running - 1))"
  fi
done < "$JOBS_TSV"

while (( running > 0 )); do
  if ! wait -n; then
    failed=1
  fi
  running="$((running - 1))"
done

if (( failed != 0 )); then
  echo "[launcher] at least one inference job failed; inspect ${LOG_ROOT}" >&2
  exit 1
fi

"$PYTHON" "$SCRIPT" summarize \
  --protocol "$PROTOCOL" \
  --output-root "$OUTPUT_ROOT"

echo "[launcher] complete"
echo "[launcher] protocol=${PROTOCOL}"
echo "[launcher] outputs=${OUTPUT_ROOT}"
echo "[launcher] logs=${LOG_ROOT}"
