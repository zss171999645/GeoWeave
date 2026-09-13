#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SAVE_ROOT="${SAVE_ROOT:-/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline}"
WEIGHT_ROOT="${WEIGHT_ROOT:-${SAVE_ROOT}/trained_model/geoweave_paper_handoff_20260602_final}"
RESULT_ROOT="${RESULT_ROOT:-${SAVE_ROOT}/result/pi3_noise/geoweave_handoff_repro_$(date +%Y%m%d_%H%M%S)}"
REPRO_DATA_ROOT="${REPRO_DATA_ROOT:-${SAVE_ROOT}/eval_datasets/geoweave_handoff_repro_$(date +%Y%m%d_%H%M%S)}"

PI3_OFFICIAL_CKPT="${PI3_OFFICIAL_CKPT:-${WEIGHT_ROOT}/pi3_base_yyfz233/Pi3_model.safetensors}"
PI3_GEOWEAVE_CKPT="${PI3_GEOWEAVE_CKPT:-${WEIGHT_ROOT}/pi3_geoweave_native_sparse_20260505_checkpoint_79/checkpoint_79/pytorch_model.bin}"
PI3_GEOWEAVE_CONFIG="${PI3_GEOWEAVE_CONFIG:-${WEIGHT_ROOT}/pi3_geoweave_native_sparse_20260505_checkpoint_79/config.yaml}"
PI3_NATIVE_ROOT="${PI3_NATIVE_ROOT:-aidi/third_party/pi3_training}"

SCANNETPP_WEAK_ROOT="${SCANNETPP_WEAK_ROOT:-${SAVE_ROOT}/eval_datasets/scannetpp_slight_overlap5x2_smoke_538c3f3a_20260513_0441/scannetpp_slight5x2_smoke}"
WAYMO_WEAK_ROOT="${WAYMO_WEAK_ROOT:-${SAVE_ROOT}/eval_datasets/driving_multicamera_all_eval_v1_6a6dfa66_20260512_0127_resolved_4090/waymo}"
WAYMO_DISTRACTOR_ROOT="${WAYMO_DISTRACTOR_ROOT:-${SAVE_ROOT}/eval_datasets/waymo_plausible_wrong_trigger_v1_frozen_top10_20260513_0339}"
WAYMO_DISTRACTOR_DATASET_ROOT="${WAYMO_DISTRACTOR_DATASET_ROOT:-${WAYMO_DISTRACTOR_ROOT}/waymo}"
WAYMO_DISTRACTOR_SAMPLE_SPEC="${WAYMO_DISTRACTOR_SAMPLE_SPEC:-${WAYMO_DISTRACTOR_ROOT}/frozen_top10_sample_spec.json}"

SCANNETPP_SCENE_SPECS="${SCANNETPP_SCENE_SPECS:-/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/bd7375297e,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/281ba69af1,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/0a184cf634,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/8be0cd3817,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/ab046f8faf,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/5d152fab1b,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/b20a261fdf,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/7f4d173c9c,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/1c4b893630,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/b08a908f0f,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/4ba22fa7e4,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/9f79564dbf,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/709ab5bffe,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/dc263dfbf0,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/ad2d07fd11,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/bf6e439e38,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/c0c863b72d,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/251443268c,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/dfac5b38df,/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/yingfeng.cai/evc_data/scannetpp/b26e64c4b0}"

MODE=""
MODEL="both"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash release/geoweave_paper_handoff/scripts/robustness_pi3.sh <mode> [options]

Modes:
  --list                         Print fixed paper robustness protocol roots.
  --build-weak-scannetpp          Rebuild ScanNet++ weak-overlap 5+5 tuples.
  --build-weak-waymo              Rebuild Waymo same-window multi-camera 5+5 tuples.
  --build-distractor-waymo        Rebuild Waymo frozen plausible-wrong context protocol.
  --eval-weak-scannetpp           Evaluate ScanNet++ weak-overlap protocol.
  --eval-weak-waymo               Evaluate Waymo weak-overlap protocol.
  --eval-distractor-waymo         Evaluate Waymo distractor-context protocol.
  --summarize-weak-scannetpp      Summarize paired official/GeoWeave weak-overlap output.
  --summarize-weak-waymo          Summarize paired official/GeoWeave weak-overlap output.
  --summarize-distractor-waymo    Summarize paired official/GeoWeave distractor output.

Options:
  --model official|geoweave|both  Model(s) for eval modes. Default: both.
  --dry-run                       Print resolved command(s) without executing.

Useful environment overrides:
  RESULT_ROOT, REPRO_DATA_ROOT, DEVICE, LIMIT_SEQS, PYTHON_BIN,
  SCANNETPP_WEAK_ROOT, WAYMO_WEAK_ROOT, WAYMO_DISTRACTOR_ROOT,
  PI3_OFFICIAL_CKPT, PI3_GEOWEAVE_CKPT, PI3_GEOWEAVE_CONFIG.
EOF
}

set_mode() {
    if [ -n "${MODE}" ]; then
        echo "[ERROR] Only one mode can be specified." >&2
        usage >&2
        exit 2
    fi
    MODE="$1"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --list) set_mode list; shift ;;
        --build-weak-scannetpp) set_mode build_weak_scannetpp; shift ;;
        --build-weak-waymo) set_mode build_weak_waymo; shift ;;
        --build-distractor-waymo) set_mode build_distractor_waymo; shift ;;
        --eval-weak-scannetpp) set_mode eval_weak_scannetpp; shift ;;
        --eval-weak-waymo) set_mode eval_weak_waymo; shift ;;
        --eval-distractor-waymo) set_mode eval_distractor_waymo; shift ;;
        --summarize-weak-scannetpp) set_mode summarize_weak_scannetpp; shift ;;
        --summarize-weak-waymo) set_mode summarize_weak_waymo; shift ;;
        --summarize-distractor-waymo) set_mode summarize_distractor_waymo; shift ;;
        --model)
            MODEL="${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "${MODE}" ]; then
    usage >&2
    exit 2
fi

cd "${REPO_ROOT}"

quote_cmd() {
    local arg
    for arg in "$@"; do
        printf '%q ' "${arg}"
    done
    printf '\n'
}

run_cmd() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY-RUN]"
        quote_cmd "$@"
    else
        "$@"
    fi
}

require_file() {
    local path="$1"
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing file: ${path}" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    if [ ! -d "${path}" ]; then
        echo "[ERROR] Missing directory: ${path}" >&2
        exit 1
    fi
}

model_list() {
    case "${MODEL}" in
        official) echo "official" ;;
        geoweave) echo "geoweave" ;;
        both) printf '%s\n%s\n' "official" "geoweave" ;;
        *)
            echo "[ERROR] --model must be official, geoweave, or both: ${MODEL}" >&2
            exit 2
            ;;
    esac
}

model_eval_args() {
    local model="$1"
    if [ "${model}" = "official" ]; then
        printf '%s\n' "--model-tag" "official" "--model-path" "${PI3_OFFICIAL_CKPT}"
    elif [ "${model}" = "geoweave" ]; then
        printf '%s\n' "--model-tag" "sparse79" "--model-path" "${PI3_GEOWEAVE_CKPT}" "--pi3-config" "${PI3_GEOWEAVE_CONFIG}" "--pi3-model-impl" "native_sparse" "--pi3-native-root" "${PI3_NATIVE_ROOT}"
    else
        echo "[ERROR] Unknown model: ${model}" >&2
        exit 2
    fi
}

eval_relpose() {
    local protocol="$1"
    local data_root="$2"
    local model="$3"
    local out_root="$4"
    shift 4
    local tag
    if [ "${model}" = "official" ]; then
        tag="official"
        if [ "${DRY_RUN}" != "1" ]; then
            require_file "${PI3_OFFICIAL_CKPT}"
        fi
    else
        tag="sparse79"
        if [ "${DRY_RUN}" != "1" ]; then
            require_file "${PI3_GEOWEAVE_CKPT}"
            require_file "${PI3_GEOWEAVE_CONFIG}"
        fi
    fi
    if [ "${DRY_RUN}" != "1" ]; then
        require_dir "${data_root}"
    fi
    local cmd=(
        "${PYTHON_BIN}" "aidi/scripts/baselines/eval_pi3_relpose_distance_protocol.py"
        "--datasets" "vkitti2"
        "--vkitti-root" "${data_root}"
        "--model-family" "pi3"
        "--load-img-size" "512"
        "--pose-eval-stride" "1"
        "--output-dir" "${out_root}/${protocol}/${tag}"
        "--skip-plot"
        "--require-official-layout"
    )
    if [ -n "${DEVICE:-}" ]; then
        cmd+=("--device" "${DEVICE}")
    fi
    if [ -n "${LIMIT_SEQS:-}" ]; then
        cmd+=("--limit-seqs" "${LIMIT_SEQS}")
    fi
    while IFS= read -r item; do
        cmd+=("${item}")
    done < <(model_eval_args "${model}")
    if [ "$#" -gt 0 ]; then
        cmd+=("$@")
    fi
    run_cmd "${cmd[@]}"
}

summarize_pair() {
    local result_root="$1"
    local output_name="$2"
    run_cmd "${PYTHON_BIN}" - "${result_root}" "${output_name}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_root = Path(sys.argv[1])
output_name = sys.argv[2]
models = {"official": result_root / "official" / "vkitti2", "sparse79": result_root / "sparse79" / "vkitti2"}

def read_summary(path: Path):
    data = json.loads((path / "summary.json").read_text())
    summary = data.get("summary", data)
    return {
        "ATE": float(summary["ATE"]),
        "RPE trans": float(summary["RPE trans"]),
        "RPE rot": float(summary["RPE rot"]),
        "num_sequences": int(summary["num_sequences"]),
    }

def read_seq(path: Path):
    rows = {}
    with (path / "seq_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["seq"]] = {key: float(row[key]) for key in ("ATE", "RPE trans", "RPE rot")}
    return rows

summary = {tag: read_summary(path) for tag, path in models.items()}
official = read_seq(models["official"])
sparse = read_seq(models["sparse79"])
pairs = [(seq, official[seq], sparse[seq]) for seq in sorted(official) if seq in sparse]

def mean(values):
    return sum(values) / len(values) if values else float("nan")

paired = {
    "num_pairs": len(pairs),
    "ATE_gap_official_minus_sparse": mean([off["ATE"] - sp["ATE"] for _, off, sp in pairs]),
    "RPE_trans_gap_official_minus_sparse": mean([off["RPE trans"] - sp["RPE trans"] for _, off, sp in pairs]),
    "RPE_rot_gap_official_minus_sparse": mean([off["RPE rot"] - sp["RPE rot"] for _, off, sp in pairs]),
    "sparse79_better_ATE_count": sum(sp["ATE"] < off["ATE"] for _, off, sp in pairs),
}
out = {"result_root": str(result_root), "summary": summary, "paired": paired}
output = result_root / output_name
output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False))
PY
}

case "${MODE}" in
    list)
        cat <<EOF
GeoWeave Pi3 paper robustness protocols:
- weak_scannetpp: ${SCANNETPP_WEAK_ROOT}
- weak_waymo: ${WAYMO_WEAK_ROOT}
- distractor_waymo: ${WAYMO_DISTRACTOR_DATASET_ROOT}

Fixed weights:
- Pi3 official: ${PI3_OFFICIAL_CKPT}
- Pi3 GeoWeave: ${PI3_GEOWEAVE_CKPT}
- Pi3 GeoWeave config: ${PI3_GEOWEAVE_CONFIG}

Default output root:
- ${RESULT_ROOT}
EOF
        ;;
    build_weak_scannetpp)
        run_cmd "${PYTHON_BIN}" "aidi/scripts/baselines/build_same_scene_low_overlap_5plus5_benchmark.py" \
            --dataset-root "/" \
            --dataset-name "scannetpp_slight5x2_smoke" \
            --source-layout "nested_evc" \
            --scene-specs "${SCANNETPP_SCENE_SPECS}" \
            --limit-scenes 6 \
            --output-root "${REPRO_DATA_ROOT}/scannetpp_slight_overlap5x2_repro" \
            --samples-per-scene 2 \
            --overlap-sample-stride 128 \
            --depth-rel-tol 0.08 \
            --local-overlap-threshold 0.05 \
            --anchor-low-overlap-threshold 0.18 \
            --cross-mean-overlap-min 0.015 \
            --cross-mean-overlap-threshold 0.08 \
            --cross-max-overlap-min 0.05 \
            --cross-max-overlap-threshold 0.35 \
            --min-center-distance-ratio 0.1 \
            --write-preview
        ;;
    build_weak_waymo)
        run_cmd "${PYTHON_BIN}" "aidi/scripts/baselines/build_driving_multicamera_all_eval_benchmark.py" \
            --datasets "waymo" \
            --output-root "${REPRO_DATA_ROOT}/driving_multicamera_all_eval_repro" \
            --samples-per-scene 5 \
            --views-per-camera 5 \
            --frame-stride 4 \
            --waymo-camera-pairs "03:04,04:03"
        ;;
    build_distractor_waymo)
        if [ "${DRY_RUN}" != "1" ]; then
            require_file "${WAYMO_DISTRACTOR_SAMPLE_SPEC}"
        fi
        run_cmd "${PYTHON_BIN}" "aidi/scripts/baselines/build_waymo_plausible_wrong_trigger_protocol.py" \
            --sample-spec-json "${WAYMO_DISTRACTOR_SAMPLE_SPEC}" \
            --output-root "${REPRO_DATA_ROOT}/waymo_plausible_wrong_trigger_v1_frozen_top10_repro" \
            --top-k 10 \
            --context-start 6 \
            --total-views 10 \
            --frame-stride 4 \
            --require-resolution-match
        ;;
    eval_weak_scannetpp)
        while IFS= read -r model; do
            eval_relpose "weak_scannetpp" "${SCANNETPP_WEAK_ROOT}" "${model}" "${RESULT_ROOT}"
        done < <(model_list)
        ;;
    eval_weak_waymo)
        while IFS= read -r model; do
            eval_relpose "weak_waymo" "${WAYMO_WEAK_ROOT}" "${model}" "${RESULT_ROOT}"
        done < <(model_list)
        ;;
    eval_distractor_waymo)
        while IFS= read -r model; do
            eval_relpose "distractor_waymo" "${WAYMO_DISTRACTOR_DATASET_ROOT}" "${model}" "${RESULT_ROOT}" \
                --eval-frame-indices "0,1,2,3,4,5"
        done < <(model_list)
        ;;
    summarize_weak_scannetpp)
        summarize_pair "${RESULT_ROOT}/weak_scannetpp" "weak_scannetpp_pair_summary.json"
        ;;
    summarize_weak_waymo)
        summarize_pair "${RESULT_ROOT}/weak_waymo" "weak_waymo_pair_summary.json"
        ;;
    summarize_distractor_waymo)
        run_cmd "${PYTHON_BIN}" "aidi/scripts/baselines/summarize_waymo_plausible_wrong_trigger_protocol.py" \
            --protocol-root "${WAYMO_DISTRACTOR_ROOT}" \
            --result-root "${RESULT_ROOT}/distractor_waymo" \
            --model-tags "official,sparse79" \
            --dataset-name "vkitti2"
        ;;
    *)
        echo "[ERROR] Internal unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac
