#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Official VGGT finetune entry with local/remote switch.
# Edit variables below instead of passing CLI args.

bool_to_cfg() {
    if [ "$1" -eq 1 ]; then
        echo True
    else
        echo False
    fi
}

strip_spaces() {
    echo "${1//[[:space:]]/}"
}

append_overrides_from_string() {
    local overrides_str="$1"
    local item
    IFS=';' read -r -a extra_list <<< "${overrides_str}"
    for item in "${extra_list[@]}"; do
        if [ -n "${item}" ]; then
            OVERRIDES+=("${item}")
        fi
    done
}

override_has_prefix() {
    local prefix="$1"
    local item
    for item in "${OVERRIDES[@]}"; do
        if [[ "${item}" == "${prefix}"* ]]; then
            return 0
        fi
    done
    return 1
}

append_override_if_missing() {
    local override="$1"
    local prefix="${override%%=*}="
    if ! override_has_prefix "${prefix}"; then
        OVERRIDES+=("${override}")
    fi
}

strip_override_prefixes() {
    local prefixes=("$@")
    local filtered=()
    local item
    local prefix
    local strip_item
    for item in "${OVERRIDES[@]}"; do
        strip_item=0
        for prefix in "${prefixes[@]}"; do
            if [[ "${item}" == "${prefix}"* ]]; then
                strip_item=1
                break
            fi
        done
        if [ "${strip_item}" -eq 0 ]; then
            filtered+=("${item}")
        fi
    done
    OVERRIDES=("${filtered[@]}")
}

join_by() {
    local IFS="$1"
    shift
    echo "$*"
}

build_run_env_args() {
    RUN_ENV_ARGS=("SAVE_TO_AIDI=${SAVE_TO_AIDI}")
    if [ -n "${RUN_ENV_VARS}" ]; then
        IFS=';' read -r -a RUN_ENV_LIST <<< "${RUN_ENV_VARS}"
        for env_item in "${RUN_ENV_LIST[@]}"; do
            if [ -n "${env_item}" ]; then
                RUN_ENV_ARGS+=("${env_item}")
            fi
        done
    fi
}

print_cmd() {
    printf "%q " "$@"
    echo
}

cleanup_overrides_argfile() {
    if [ -n "${OVERRIDES_ARGFILE_PATH:-}" ] && [ -f "${OVERRIDES_ARGFILE_PATH}" ]; then
        rm -f "${OVERRIDES_ARGFILE_PATH}"
    fi
}

MODE=${MODE:-"local"}  # local or remote
CONFIG=${CONFIG:-"vggt/vggt_official_finetune"}
EXP_NAME=${EXP_NAME:-"vggt/official/finetune"}
# Match origin/feat-vggt-sparse-attn submit style by default:
# keep inline --train_args in one run_cmd string unless explicitly enabled.
OVERRIDES_ARGFILE_ENABLED=${OVERRIDES_ARGFILE_ENABLED:-"0"}  # 0/1
OVERRIDES_ARGFILE_DIR=${OVERRIDES_ARGFILE_DIR:-"aidi/scripts/vggt/.generated"}
RUN_ENV_VARS=${RUN_ENV_VARS:-""}  # semicolon separated KEY=VALUE, forwarded in remote mode
INLINE_TRAIN_ARGS_MAX_LEN=${INLINE_TRAIN_ARGS_MAX_LEN:-120000}

GPU_IDS=${GPU_IDS:-"0,1,2,3"}
NUM_NODES=${NUM_NODES:-"1"}
GPUS_PER_NODE=${GPUS_PER_NODE:-""}
FORCE_DIST=${FORCE_DIST:-"0"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-""}
RDZV_ENDPOINT=${RDZV_ENDPOINT:-""}
RDZV_BACKEND=${RDZV_BACKEND:-""}
RDZV_ID=${RDZV_ID:-""}

CLUSTER=${CLUSTER:-"a800-2"}
JOB_NAME=${JOB_NAME:-"vggt_official_finetune"}
PRIORITY=${PRIORITY:-"5"}
TOTAL_GPU=${TOTAL_GPU:-""}
SUBMIT_TRIAL=${SUBMIT_TRIAL:-"5"}

DEBUG=${DEBUG:-"0"}
RESUME=${RESUME:-"0"}
RUN_TEST=${RUN_TEST:-"0"}
SUBMIT_SLEEP=${SUBMIT_SLEEP:-"0"}

SAVE_TO_AIDI=${SAVE_TO_AIDI:-"0"}

META_ROOTS=${META_ROOTS:-""}  # comma separated, overrides dataset meta_roots
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-""}  # semicolon separated overrides

AGG_CKPT=${AGG_CKPT:-"/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B/aggregator.pt"}
CAM_CKPT=${CAM_CKPT:-"/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B/camera.pt"}
XYZ_CKPT=${XYZ_CKPT:-"/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B/point.pt"}
DPT_CKPT=${DPT_CKPT:-"/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B/depth.pt"}
TRA_CKPT=${TRA_CKPT:-"/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data/trained_model/vggt/official/VGGT-1B/track.pt"}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-""}

CONFIG_PATH="configs/exps/${CONFIG}.yaml"

OVERRIDES=(
    "runner_cfg.resume=$(bool_to_cfg "${RESUME}")"
)

if [ -n "${PRETRAINED_MODEL}" ]; then
    OVERRIDES+=("model_cfg.pretrained_path=${PRETRAINED_MODEL}")
else
    OVERRIDES+=(
        "model_cfg.agg_ckpt=${AGG_CKPT}"
        "model_cfg.cam_ckpt=${CAM_CKPT}"
        "model_cfg.xyz_ckpt=${XYZ_CKPT}"
        "model_cfg.dpt_ckpt=${DPT_CKPT}"
        "model_cfg.tra_ckpt=${TRA_CKPT}"
    )
fi

if [ -n "${META_ROOTS}" ]; then
    META_ROOTS_LIST="$(strip_spaces "${META_ROOTS}")"
    OVERRIDES+=("dataloader_cfg.dataset_cfg.meta_roots=${META_ROOTS_LIST}")
    OVERRIDES+=("val_dataloader_cfg.dataset_cfg.meta_roots=${META_ROOTS_LIST}")
fi

if [ "${DEBUG}" -eq 1 ]; then
    DEBUG_OVERRIDES=(
        "runner_cfg.ep_iter=50"
        "runner_cfg.eval_ep=1"
        "runner_cfg.log_interval=1"
        "dataloader_cfg.num_workers=0"
        "val_dataloader_cfg.num_workers=0"
    )
    OVERRIDES+=("${DEBUG_OVERRIDES[@]}")
fi

if [ -n "${EXTRA_OVERRIDES}" ]; then
    append_overrides_from_string "${EXTRA_OVERRIDES}"
fi

if [ "${MODE}" = "remote" ]; then
    # Keep remote tensorboard/result paths aligned with origin/feat-vggt:
    # aidi/run.sh should own /job_tboard and /job_data wiring.
    strip_override_prefixes \
        "runner_cfg.recorder_cfg.record_dir=" \
        "runner_cfg.visualizer_cfg.result_dir="
elif [ "${MODE}" = "local" ] && [ -n "${SAVE_ROOT:-}" ]; then
    # Local mode bypasses aidi/run.sh, so inject the same save roots here.
    append_override_if_missing "runner_cfg.trained_model=${SAVE_ROOT}/trained_model/${EXP_NAME}"
    append_override_if_missing "runner_cfg.recorder_cfg.record_dir=${SAVE_ROOT}/record/${EXP_NAME}"
    append_override_if_missing "runner_cfg.visualizer_cfg.result_dir=${SAVE_ROOT}/result"
fi

if [ "${MODE}" = "local" ]; then
    build_run_env_args
    for env_item in "${RUN_ENV_ARGS[@]}"; do
        export "${env_item}"
    done

    IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
    NUM_GPUS="${#GPU_ID_LIST[@]}"
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

    TORCHRUN_ARGS=()
    if [ -z "${RDZV_ENDPOINT}" ] && [ -n "${MASTER_PORT}" ]; then
        RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
    fi
    if [ -n "${RDZV_ENDPOINT}" ]; then
        TORCHRUN_ARGS+=(--rdzv_endpoint "${RDZV_ENDPOINT}")
    fi
    if [ -n "${RDZV_BACKEND}" ]; then
        TORCHRUN_ARGS+=(--rdzv_backend "${RDZV_BACKEND}")
    fi
    if [ -n "${RDZV_ID}" ]; then
        TORCHRUN_ARGS+=(--rdzv_id "${RDZV_ID}")
    fi

    USE_DIST_LAUNCHER=0
    if [ "${NUM_GPUS}" -gt 1 ] || [ "${FORCE_DIST}" -eq 1 ]; then
        USE_DIST_LAUNCHER=1
    fi
    if [ "${USE_DIST_LAUNCHER}" -eq 1 ]; then
        if [ -n "${TORCHRUN_ARGS+x}" ] && [ "${#TORCHRUN_ARGS[@]}" -gt 0 ]; then
            LAUNCHER=(evc-dist --nproc_per_node="${NUM_GPUS}" "${TORCHRUN_ARGS[@]}" --)
        else
            LAUNCHER=(evc-dist --nproc_per_node="${NUM_GPUS}" --)
        fi
        if [ "${RUN_TEST}" -eq 1 ]; then
            CMD=("${LAUNCHER[@]}" -t test -c "${CONFIG_PATH}" "exp_name=${EXP_NAME}")
        else
            CMD=("${LAUNCHER[@]}" -c "${CONFIG_PATH}" "exp_name=${EXP_NAME}")
        fi
    else
        if [ "${RUN_TEST}" -eq 1 ]; then
            LAUNCHER=(evc-test)
        else
            LAUNCHER=(evc-train)
        fi
        CMD=("${LAUNCHER[@]}" -c "${CONFIG_PATH}" "exp_name=${EXP_NAME}")
    fi

    CMD+=("${OVERRIDES[@]}")

    echo "Running: ${CMD[*]}"
    "${CMD[@]}"
elif [ "${MODE}" = "remote" ]; then
    if [ -z "${USER:-}" ]; then
        echo "USER is not set, please export USER before remote submission"
        exit 1
    fi

    SUBMIT_ARGS=(--job_name "${JOB_NAME}" --cluster_name "${CLUSTER}")
    if [ "${SUBMIT_SLEEP}" = "1" ]; then
        SUBMIT_ARGS+=(--sleep)
    fi
    if [ -n "${TOTAL_GPU}" ]; then
        SUBMIT_ARGS+=(-n "${TOTAL_GPU}")
    else
        if [ -z "${GPUS_PER_NODE}" ]; then
            GPUS_PER_NODE=8
        fi
        SUBMIT_ARGS+=(--gpu "${GPUS_PER_NODE}" --node "${NUM_NODES}")
    fi

    if [ -n "${PRIORITY}" ]; then
        SUBMIT_ARGS+=(--priority "${PRIORITY}")
    fi
    if [ -n "${SUBMIT_TRIAL}" ]; then
        SUBMIT_ARGS+=(--trial "${SUBMIT_TRIAL}")
    fi

    RUN_ARGS_STR="${OVERRIDES[*]}"
    OVERRIDES_ARGFILE_PATH=""
    RUN_SH_HAS_AT_EXPAND="0"
    if grep -q "expand_at_file_args" aidi/run.sh; then
        RUN_SH_HAS_AT_EXPAND="1"
    fi

    if [ "${OVERRIDES_ARGFILE_ENABLED}" = "1" ] && [ "${RUN_SH_HAS_AT_EXPAND}" != "1" ]; then
        echo "[ERROR] Refuse to submit with @args: aidi/run.sh does not support expand_at_file_args."
        echo "[ERROR] Set OVERRIDES_ARGFILE_ENABLED=0 or sync newer aidi/run.sh first."
        exit 1
    fi

    if [ "${OVERRIDES_ARGFILE_ENABLED}" = "1" ]; then
        JOB_NAME_SAFE="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
        OVERRIDES_ARGFILE_PATH="${OVERRIDES_ARGFILE_DIR}/${JOB_NAME_SAFE}_$(date +%Y%m%d_%H%M%S)_$$.args"
        mkdir -p "$(dirname "${OVERRIDES_ARGFILE_PATH}")"
        printf '%s\n' "${OVERRIDES[@]}" > "${OVERRIDES_ARGFILE_PATH}"
        RUN_ARGS_STR="@${OVERRIDES_ARGFILE_PATH}"
        trap cleanup_overrides_argfile EXIT
    else
        if [ "${#RUN_ARGS_STR}" -gt "${INLINE_TRAIN_ARGS_MAX_LEN}" ]; then
            echo "[WARN] train_args is long in inline mode: len=${#RUN_ARGS_STR}, limit=${INLINE_TRAIN_ARGS_MAX_LEN}"
            echo "[WARN] Consider OVERRIDES_ARGFILE_ENABLED=1 if your cluster has command-length limits."
        fi
    fi
    echo "Submit arg strategy: OVERRIDES_ARGFILE_ENABLED=${OVERRIDES_ARGFILE_ENABLED}, run_sh_has_expand=${RUN_SH_HAS_AT_EXPAND}"
    build_run_env_args
    if [ "${RUN_TEST}" -eq 1 ]; then
        RUN_ARGS_STR_ESCAPED="${RUN_ARGS_STR//\"/\\\"}"
        RUN_ENV_PREFIX="$(join_by ' ' "${RUN_ENV_ARGS[@]}")"
        run_cmd="${RUN_ENV_PREFIX} ./aidi/run.sh --config ${CONFIG} --exp_name ${EXP_NAME} --test --test_args \"${RUN_ARGS_STR_ESCAPED}\""
    else
        RUN_ARGS_STR_ESCAPED="${RUN_ARGS_STR//\"/\\\"}"
        RUN_ENV_PREFIX="$(join_by ' ' "${RUN_ENV_ARGS[@]}")"
        run_cmd="${RUN_ENV_PREFIX} ./aidi/run.sh --config ${CONFIG} --exp_name ${EXP_NAME} --train --dist --train_args \"${RUN_ARGS_STR_ESCAPED}\""
    fi

    echo "Submitting (origin-style run_cmd):"
    echo "${run_cmd}"
    print_cmd python3 aidi/submit.py "${SUBMIT_ARGS[@]}" "${run_cmd}"
    python3 aidi/submit.py "${SUBMIT_ARGS[@]}" "${run_cmd}"

else
    echo "Unknown MODE: ${MODE} (expected local or remote)"
    exit 1
fi
