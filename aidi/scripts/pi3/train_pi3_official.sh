#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PI3_ROOT="${PI3_ROOT:-${REPO_ROOT}/aidi/third_party/pi3_training}"

if [ ! -d "${PI3_ROOT}" ]; then
    echo "[ERROR] PI3_ROOT not found: ${PI3_ROOT}"
    exit 1
fi

stage_to_train_cfg() {
    case "$1" in
        lowres) echo "train_pi3_lowres" ;;
        highres) echo "train_pi3_highres" ;;
        conf) echo "train_pi3_conf" ;;
        warmup) echo "train_pi3_lowres_indexer_warmup" ;;
        sparse) echo "train_pi3_lowres_indexer_sparse" ;;
        *)
            echo "[ERROR] Unknown STAGE=$1 (expected: lowres|highres|conf|warmup|sparse)" >&2
            exit 1
            ;;
    esac
}

bool_to_py() {
    if [ "$1" = "1" ] || [ "$1" = "true" ] || [ "$1" = "True" ]; then
        echo "True"
    else
        echo "False"
    fi
}

is_truthy() {
    [ "$1" = "1" ] || [ "$1" = "true" ] || [ "$1" = "True" ] || [ "$1" = "yes" ] || [ "$1" = "on" ]
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

append_override_from_env() {
    local env_name="$1"
    local hydra_key="$2"
    local env_val="${!env_name:-}"
    if [ -n "${env_val}" ]; then
        OVERRIDES+=("${hydra_key}=${env_val}")
    fi
}

append_bool_override_from_env() {
    local env_name="$1"
    local hydra_key="$2"
    local env_val="${!env_name:-}"
    if [ -n "${env_val}" ]; then
        OVERRIDES+=("${hydra_key}=$(bool_to_py "${env_val}")")
    fi
}

warn_ignored_env() {
    local env_name="$1"
    local reason="$2"
    local env_val="${!env_name:-}"
    if [ -n "${env_val}" ]; then
        echo "[WARN] Ignoring ${env_name}=${env_val}: ${reason}"
    fi
}

append_meshx_vggt17_dataset_override_from_env() {
    local env_name="$1"
    local key_name="$2"
    local env_val="${!env_name:-}"
    local dataset_name
    if [ -z "${env_val}" ] || [ "${DATA_CFG}" != "meshx_pi3_vggt17" ]; then
        return
    fi
    for dataset_name in BlendedMVS HypersimTrain HypersimVal GTAV Replica TarTanAir VKITTI2 ASE ADT MegaDepth Scannetpp Taskonomy WildRGBD MapFree CO3Dv2 Mapillary DL3DV; do
        OVERRIDES+=("++train_dataset.${dataset_name}.${key_name}=${env_val}")
        OVERRIDES+=("++test_dataset.${dataset_name}.${key_name}=${env_val}")
    done
}

STAGE="${STAGE:-lowres}"                  # lowres | highres | conf | warmup | sparse
TRAIN_CFG="${TRAIN_CFG:-$(stage_to_train_cfg "${STAGE}")}"
DATA_CFG="${DATA_CFG:-meshx_tartanair}"
RUN_NAME="${RUN_NAME:-pi3_${STAGE}_$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-${RUN_NAME}}"
MODEL_CKPT="${MODEL_CKPT:-}"              # required for stage2/3
INDEXER_INIT_CKPT="${INDEXER_INIT_CKPT:-}" # optional indexer-only init for warmup/sparse

NUM_PROCESSES="${NUM_PROCESSES:-${WORLD_SIZE:-1}}"
NUM_MACHINES="${NUM_MACHINES:-${NUM_NODES:-1}}"
MACHINE_RANK="${MACHINE_RANK:-${NODE_RANK:-0}}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-${HOST_NODE_ADDR:-127.0.0.1}}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-29500}}"
ACC_CONFIG="${ACC_CONFIG:-configs/accelerate/ddp.yaml}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
PIP_INDEX_URL_INTERNAL="${PIP_INDEX_URL_INTERNAL:-http://pypi.hobot.cc/simple}"
PIP_EXTRA_INDEX_URL_INTERNAL="${PIP_EXTRA_INDEX_URL_INTERNAL:-http://pypi.hobot.cc/hobot-local/simple}"
PIP_TRUSTED_HOST_INTERNAL="${PIP_TRUSTED_HOST_INTERNAL:-pypi.hobot.cc}"
RUNTIME_PIP_PKGS="${RUNTIME_PIP_PKGS:-hydra-core accelerate prettytable easydict transformers rich omegaconf pyyaml tqdm}"
CORE4_RUNTIME_PIP_PKGS="${CORE4_RUNTIME_PIP_PKGS:-open3d plyfile openpyxl}"
CORE4_FIRST_EVAL="${CORE4_FIRST_EVAL:-0}"

SMOKE="${SMOKE:-0}"                       # 1 for tiny resolution smoke run
SMOKE_RES_MODE="${SMOKE_RES_MODE:-fixed}" # fixed | dynamic
SMOKE_RES="${SMOKE_RES:-112}"
SMOKE_ITERS="${SMOKE_ITERS:-2}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_MAX_IMG_PER_GPU="${SMOKE_MAX_IMG_PER_GPU:-4}"
SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-1}"
SMOKE_SEQ_NUM="${SMOKE_SEQ_NUM:-1}"
SMOKE_MAX_FRAMES_PER_SEQUENCE="${SMOKE_MAX_FRAMES_PER_SEQUENCE:-16}"

LOAD_VGGT="${LOAD_VGGT:-1}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"    # semicolon separated hydra overrides
CORE4_VAL="${CORE4_VAL:-0}"               # 1 to run native core4 main validation after each epoch
CORE4_ONLY_CO3D="${CORE4_ONLY_CO3D:-0}"   # 1 to disable other core4 datasets for closed-loop CO3D diagnostics
PI3_SKIP_CORE4_RUNTIME_DEPS_CHECK="${PI3_SKIP_CORE4_RUNTIME_DEPS_CHECK:-0}"
PI3_NATIVE_MAX_REFETCH="${PI3_NATIVE_MAX_REFETCH:-128}"
SAVE_TO_AIDI="${SAVE_TO_AIDI:-0}"
TEST_ITERS_PER_TEST_IS_SET=0
if [ -n "${TEST_ITERS_PER_TEST:-}" ]; then
    TEST_ITERS_PER_TEST_IS_SET=1
fi

PI3_TB_MIRROR_DIR=${PI3_TB_MIRROR_DIR:-${WORK_DIR:-}}
if [ -z "${PI3_TB_MIRROR_DIR}" ] && [ -n "${SAVE_ROOT:-}" ]; then
    PI3_TB_MIRROR_DIR="${SAVE_ROOT}/record/${EXP_NAME}"
fi
PI3_TB_RECORD_DIR="${PI3_TB_RECORD_DIR:-}"
if [ -z "${PI3_TB_RECORD_DIR}" ]; then
    if [ -d "/job_tboard" ] && [ "${SAVE_TO_AIDI}" != "0" ]; then
        PI3_TB_RECORD_DIR="/job_tboard/record/${EXP_NAME}"
    elif [ -d "/job_data" ] && [ "${SAVE_TO_AIDI}" != "0" ]; then
        PI3_TB_RECORD_DIR="/job_data/record/${EXP_NAME}"
    elif [ -n "${PI3_TB_MIRROR_DIR}" ]; then
        PI3_TB_RECORD_DIR="${PI3_TB_MIRROR_DIR}"
    else
        PI3_TB_RECORD_DIR="${PI3_ROOT}/outputs/${RUN_NAME}"
    fi
fi
if [ -z "${PI3_TB_MIRROR_DIR}" ]; then
    PI3_TB_MIRROR_DIR="${PI3_TB_RECORD_DIR}"
fi
if ! mkdir -p "${PI3_TB_RECORD_DIR}" >/dev/null 2>&1; then
    echo "[WARN] Cannot create PI3_TB_RECORD_DIR=${PI3_TB_RECORD_DIR}, falling back to local Pi3 output dir"
    PI3_TB_RECORD_DIR="${PI3_ROOT}/outputs/${RUN_NAME}"
    mkdir -p "${PI3_TB_RECORD_DIR}"
fi
PI3_TB_AIDI_MIRROR_DIR="${PI3_TB_AIDI_MIRROR_DIR:-}"
if [ -z "${PI3_TB_AIDI_MIRROR_DIR}" ] && [ -d "/job_data" ] && [ "${SAVE_TO_AIDI}" != "0" ]; then
    PI3_TB_AIDI_MIRROR_DIR="/job_data/record/${EXP_NAME}"
fi
PI3_TB_MIRROR_INTERVAL="${PI3_TB_MIRROR_INTERVAL:-60}"
PI3_TB_MIRROR_BOOTSTRAP_INTERVAL="${PI3_TB_MIRROR_BOOTSTRAP_INTERVAL:-10}"
PI3_TB_MIRROR_PID=""

_sync_pi3_tb_to_dir() {
    local dst_dir="$1"
    if [ -z "${PI3_TB_RECORD_DIR}" ] || [ -z "${dst_dir}" ]; then
        return 0
    fi
    if [ "${PI3_TB_RECORD_DIR}" = "${dst_dir}" ]; then
        return 0
    fi
    if [ ! -d "${PI3_TB_RECORD_DIR}" ]; then
        return 0
    fi
    mkdir -p "${dst_dir}" >/dev/null 2>&1 || return 0
    if command -v rsync >/dev/null 2>&1; then
        rsync -az --update "${PI3_TB_RECORD_DIR}/" "${dst_dir}/" >/dev/null 2>&1 || true
    else
        cp -a "${PI3_TB_RECORD_DIR}/." "${dst_dir}/" >/dev/null 2>&1 || true
    fi
}

sync_pi3_tb_once() {
    _sync_pi3_tb_to_dir "${PI3_TB_MIRROR_DIR}"
    _sync_pi3_tb_to_dir "${PI3_TB_AIDI_MIRROR_DIR}"
}

start_pi3_tb_mirror() {
    if [ "${MACHINE_RANK}" != "0" ]; then
        echo "[INFO] PI3 TensorBoard mirror skipped on worker node (MACHINE_RANK=${MACHINE_RANK})"
        return 0
    fi
    if [ -z "${PI3_TB_RECORD_DIR}" ]; then
        return 0
    fi
    mkdir -p "${PI3_TB_RECORD_DIR}" >/dev/null 2>&1 || return 0
    local has_mirror_target=0
    if [ -n "${PI3_TB_MIRROR_DIR}" ] && [ "${PI3_TB_RECORD_DIR}" != "${PI3_TB_MIRROR_DIR}" ]; then
        has_mirror_target=1
    fi
    if [ -n "${PI3_TB_AIDI_MIRROR_DIR}" ] && [ "${PI3_TB_RECORD_DIR}" != "${PI3_TB_AIDI_MIRROR_DIR}" ]; then
        has_mirror_target=1
    fi
    if [ "${has_mirror_target}" = "0" ]; then
        echo "[INFO] PI3 TensorBoard mirror skipped because source and targets are identical or empty"
        return 0
    fi
    (
        while true; do
            sync_pi3_tb_once
            sleep "${PI3_TB_MIRROR_INTERVAL}"
        done
    ) &
    PI3_TB_MIRROR_PID=$!
    echo "[INFO] PI3 TensorBoard mirror daemon started on main node (PID: ${PI3_TB_MIRROR_PID})"
}

finish_pi3_tb_mirror() {
    if [ -n "${PI3_TB_MIRROR_PID}" ]; then
        kill "${PI3_TB_MIRROR_PID}" 2>/dev/null || true
        wait "${PI3_TB_MIRROR_PID}" 2>/dev/null || true
    fi
    sync_pi3_tb_once
}

if [ "${STAGE}" != "lowres" ] && [ "${STAGE}" != "warmup" ] && [ -z "${MODEL_CKPT}" ]; then
    echo "[ERROR] STAGE=${STAGE} requires MODEL_CKPT"
    exit 1
fi

OVERRIDES=(
    "train=${TRAIN_CFG}"
    "data=${DATA_CFG}"
    "name=${RUN_NAME}"
    "model.load_vggt=$(bool_to_py "${LOAD_VGGT}")"
    "hydra/hydra_logging=default"
    "hydra/job_logging=custom"
)

if [ -n "${MODEL_CKPT}" ]; then
    OVERRIDES+=("model.ckpt=${MODEL_CKPT}")
fi
if [ -n "${INDEXER_INIT_CKPT}" ]; then
    OVERRIDES+=("model.indexer_init_ckpt=${INDEXER_INIT_CKPT}")
fi

append_override_from_env WORK_DIR work_dir
append_override_from_env TRAIN_NUM_EPOCH train.num_epoch
append_override_from_env TRAIN_ITERS_PER_EPOCH train.iters_per_epoch
append_override_from_env TRAIN_MAX_IMG_PER_GPU train.max_img_per_gpu
append_override_from_env TRAIN_IMAGE_NUM_RANGE train.image_num_range
append_override_from_env TEST_IMAGE_NUM_RANGE test.image_num_range
append_override_from_env TRAIN_NUM_WORKERS train.num_workers
append_override_from_env TEST_NUM_WORKERS test.num_workers
append_override_from_env PI3_TRAIN_NUM_WORKERS train.num_workers
append_override_from_env PI3_TEST_NUM_WORKERS test.num_workers
append_override_from_env TRAIN_MODEL_DTYPE train.model_dtype
append_override_from_env TRAIN_GRAD_ACCUM_STEPS train.gradient_accumulation_steps
append_override_from_env TRAIN_CLIP_GRAD train.clip_grad
append_bool_override_from_env TRAIN_CLIP_GRAD_SEPARATE_INDEXER train.clip_grad_separate_indexer
append_override_from_env TRAIN_CLIP_INDEXER_GRAD train.clip_indexer_grad
append_override_from_env TRAIN_CLIP_LOSS train.clip_loss
append_override_from_env TRAIN_RESUME train.resume
append_override_from_env TRAIN_LR train.optimizer.lr
append_override_from_env TRAIN_ENCODER_LR train.optimizer.encoder_lr
append_override_from_env TRAIN_WEIGHT_DECAY train.optimizer.weight_decay
append_override_from_env TRAIN_SCHEDULER_TYPE train.lr_scheduler.type
append_override_from_env TRAIN_SCHEDULER_DECAY_ITER "++train.lr_scheduler.decay_iter"
append_override_from_env TRAIN_SCHEDULER_WARMUP_ITERS "++train.lr_scheduler.warmup_iters"
append_override_from_env TRAIN_SCHEDULER_WARMUP_FACTOR "++train.lr_scheduler.warmup_factor"
append_override_from_env TRAIN_SCHEDULER_WARMUP_METHOD "++train.lr_scheduler.warmup_method"
append_override_from_env TRAIN_SCHEDULER_WARMUP_START_LR "++train.lr_scheduler.warmup_start_lr"
append_override_from_env TRAIN_SCHEDULER_MIN_LR "++train.lr_scheduler.min_lr"
append_override_from_env TRAIN_SCHEDULER_MAX_LR train.lr_scheduler.max_lr
append_override_from_env TRAIN_SCHEDULER_PCT_START train.lr_scheduler.pct_start
append_override_from_env TRAIN_SCHEDULER_DIV_FACTOR train.lr_scheduler.div_factor
append_override_from_env TRAIN_SCHEDULER_FINAL_DIV_FACTOR train.lr_scheduler.final_div_factor
append_override_from_env TRAIN_PREFETCH_FACTOR "++train_dataloader.prefetch_factor"
append_override_from_env TEST_PREFETCH_FACTOR "++test_dataloader.prefetch_factor"
append_bool_override_from_env TRAIN_PERSISTENT_WORKERS "++train_dataloader.persistent_workers"
append_bool_override_from_env TEST_PERSISTENT_WORKERS "++test_dataloader.persistent_workers"
append_bool_override_from_env TRAIN_PIN_MEMORY "++train_dataloader.pin_memory"
append_bool_override_from_env TEST_PIN_MEMORY "++test_dataloader.pin_memory"
append_override_from_env TRAIN_PRINT_FREQ train.print_freq
append_override_from_env TEST_PRINT_FREQ test.print_freq
append_override_from_env TEST_ITERS_PER_TEST test.iters_per_test
append_override_from_env TEST_EVAL_INTERVAL test.eval_interval
append_bool_override_from_env TEST_BEFORE_FIRST_EPOCH test.before_first_epoch
append_bool_override_from_env PI3_EVAL_ONLY "++eval_only"
append_bool_override_from_env PI3_VGGT_VAL_METRICS test.vggt_style_metrics.enabled
append_override_from_env PI3_VGGT_VAL_MAX_POINTS test.vggt_style_metrics.max_points_per_view
append_override_from_env LOG_CKPT_INTERVAL log.ckpt_interval
append_override_from_env LOG_MAX_CHECKPOINTS log.max_checkpoints
append_override_from_env LOG_BEST_MODEL_SAVE_MODE log.best_model_save_mode
append_override_from_env LOG_CHECKPOINT_SAVE_MODE log.checkpoint_save_mode
append_bool_override_from_env PI3_MEMORY_REPORT log.memory_report
append_override_from_env PI3_MEMORY_ABORT_FRACTION log.memory_abort_fraction
append_override_from_env PI3_MEMORY_ABORT_METRIC log.memory_abort_metric
append_override_from_env PI3_ENCODER_ATTN_BACKEND model.encoder_attn_backend
append_override_from_env PI3_DECODER_ATTN_BACKEND model.decoder_attn_backend
append_override_from_env PI3_QK_NORM_CHUNK_SIZE model.qk_norm_chunk_size
append_bool_override_from_env PI3_USE_POSE_PRIOR "++model.use_pose_prior"
append_bool_override_from_env PI3_POSE_PRIOR_REQUIRED "++model.pose_prior_required"
append_override_from_env PI3_POSE_PRIOR_DROPOUT "++model.pose_prior_dropout"
append_override_from_env PI3_POSE_PRIOR_FORMAT "++model.pose_prior_format"
append_bool_override_from_env PI3_HEAD_USE_CHECKPOINT "++model.head_use_checkpoint"
append_override_from_env PI3_HEAD_VIEW_CHUNK_SIZE "++model.head_view_chunk_size"
append_override_from_env PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT model.num_dec_blk_not_to_checkpoint
if [ -n "${PI3_TRAIN_FIXED_RES:-}" ]; then
    OVERRIDES+=(
        "++train.random_reslution=False"
        "++train.num_resolution=1"
        "++train.resolution=[[${PI3_TRAIN_FIXED_RES},${PI3_TRAIN_FIXED_RES}]]"
    )
fi
if [ -n "${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE:-}" ]; then
    OVERRIDES+=(
        "++loss.train_loss.normal_loss_view_chunk_size=${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE}"
        "++loss.test_loss.normal_loss_view_chunk_size=${PI3_TEST_NORMAL_LOSS_VIEW_CHUNK_SIZE:-${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE}}"
    )
fi
append_override_from_env PI3_NORMAL_LOSS_WEIGHT loss.train_loss.normal_loss_weight
append_override_from_env PI3_TEST_NORMAL_LOSS_WEIGHT loss.test_loss.normal_loss_weight
append_meshx_vggt17_dataset_override_from_env PI3_USE_INDEX_CACHE use_index_cache
append_meshx_vggt17_dataset_override_from_env PI3_REBUILD_INDEX_CACHE rebuild_index_cache
append_meshx_vggt17_dataset_override_from_env PI3_DATASET_CACHE_DIR index_cache_dir
append_meshx_vggt17_dataset_override_from_env PI3_INDEX_CACHE_WAIT_SEC index_cache_wait_sec

if [ "${STAGE}" = "warmup" ] || [ "${STAGE}" = "sparse" ]; then
    OVERRIDES+=("test.iters_per_test=0")
fi

if is_truthy "${CORE4_VAL}" && [ "${TEST_ITERS_PER_TEST_IS_SET}" = "0" ]; then
    OVERRIDES+=("test.iters_per_test=0")
fi

if is_truthy "${CORE4_VAL}"; then
    if [ ! -f "${PI3_ROOT}/configs/extras/core4_main_val.yaml" ]; then
        echo "[ERROR] CORE4_VAL=1 but missing Hydra config: ${PI3_ROOT}/configs/extras/core4_main_val.yaml"
        echo "        Check submit packaging excludes; nested configs/extras/core4_main_val.yaml must not be filtered."
        exit 1
    fi
    OVERRIDES+=("extras=core4_main_val")
    if is_truthy "${CORE4_ONLY_CO3D}"; then
        OVERRIDES+=(
            "++main_val_core4_cfg.datasets.re10k.enabled=False"
            "++main_val_core4_cfg.datasets.co3dv2.enabled=True"
            "++main_val_core4_cfg.datasets.dtu.enabled=False"
            "++main_val_core4_cfg.datasets.eth3d.enabled=False"
        )
    fi
    if is_truthy "${CORE4_FIRST_EVAL}"; then
        OVERRIDES+=("++main_val_core4_cfg.run_first_eval=True")
    fi
fi

if [ "${SMOKE}" = "1" ]; then
    if [ "${SMOKE_SEQ_NUM}" -gt 0 ]; then
        export PI3_NATIVE_SEQ_NUM="${SMOKE_SEQ_NUM}"
    fi
    if [ "${SMOKE_MAX_FRAMES_PER_SEQUENCE}" -gt 0 ]; then
        export PI3_NATIVE_MAX_FRAMES_PER_SEQUENCE="${SMOKE_MAX_FRAMES_PER_SEQUENCE}"
    fi
    OVERRIDES+=(
        "train.num_epoch=${SMOKE_EPOCHS}"
        "train.iters_per_epoch=${SMOKE_ITERS}"
        "train.max_img_per_gpu=${SMOKE_MAX_IMG_PER_GPU}"
        "train.image_num_range=[2,2]"
        "train.num_workers=${SMOKE_NUM_WORKERS}"
        "test.num_workers=${SMOKE_NUM_WORKERS}"
        "train.print_freq=1"
        "test.print_freq=1"
    )
    if [ "${SMOKE_SEQ_NUM}" -gt 0 ] && [ "${DATA_CFG}" = "meshx_tartanair" ]; then
        OVERRIDES+=(
            "++train_dataset.TarTanAir.seq_num=${SMOKE_SEQ_NUM}"
            "++test_dataset.TarTanAir.seq_num=${SMOKE_SEQ_NUM}"
        )
    fi
    case "${SMOKE_RES_MODE}" in
        fixed)
            OVERRIDES+=(
                "++train.random_reslution=False"
                "++train.num_resolution=1"
                "++train.resolution=[[${SMOKE_RES},${SMOKE_RES}]]"
                "++train_dataset.TarTanAir.resolution=[[${SMOKE_RES},${SMOKE_RES}]]"
                "++test_dataset.TarTanAir.resolution=[[${SMOKE_RES},${SMOKE_RES}]]"
            )
            ;;
        dynamic)
            ;;
        *)
            echo "[ERROR] Unknown SMOKE_RES_MODE=${SMOKE_RES_MODE} (expected: fixed|dynamic)" >&2
            exit 1
            ;;
    esac
    if [ "${STAGE}" != "warmup" ] && [ "${STAGE}" != "sparse" ] && ! is_truthy "${CORE4_VAL}" && [ "${TEST_ITERS_PER_TEST_IS_SET}" = "0" ]; then
        OVERRIDES+=("test.iters_per_test=1")
    fi
fi

if [ -n "${PI3_TB_RECORD_DIR}" ]; then
    OVERRIDES+=(
        "++log.tensorboard_dir=${PI3_TB_RECORD_DIR}"
        "++log.direct_tensorboard=True"
    )
fi

if [ -n "${EXTRA_OVERRIDES}" ]; then
    append_overrides_from_string "${EXTRA_OVERRIDES}"
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] PI3_ROOT=${PI3_ROOT}"
echo "[INFO] STAGE=${STAGE} TRAIN_CFG=${TRAIN_CFG} DATA_CFG=${DATA_CFG} RUN_NAME=${RUN_NAME}"
echo "[INFO] EXP_NAME=${EXP_NAME}"
echo "[INFO] SAVE_TO_AIDI=${SAVE_TO_AIDI}"
echo "[INFO] PI3_TB_RECORD_DIR=${PI3_TB_RECORD_DIR}"
echo "[INFO] PI3_TB_MIRROR_DIR=${PI3_TB_MIRROR_DIR}"
echo "[INFO] PI3_TB_AIDI_MIRROR_DIR=${PI3_TB_AIDI_MIRROR_DIR}"
echo "[INFO] NUM_PROCESSES=${NUM_PROCESSES} NUM_MACHINES=${NUM_MACHINES} MACHINE_RANK=${MACHINE_RANK} SMOKE=${SMOKE} CORE4_VAL=${CORE4_VAL}"
echo "[INFO] TRAIN_IMAGE_NUM_RANGE=${TRAIN_IMAGE_NUM_RANGE:-<default>} TEST_IMAGE_NUM_RANGE=${TEST_IMAGE_NUM_RANGE:-<default>} TEST_EVAL_INTERVAL=${TEST_EVAL_INTERVAL:-<default>} TEST_BEFORE_FIRST_EPOCH=${TEST_BEFORE_FIRST_EPOCH:-<default>}"
echo "[INFO] MAIN_PROCESS_IP=${MAIN_PROCESS_IP} MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
echo "[INFO] PI3_NATIVE_MAX_REFETCH=${PI3_NATIVE_MAX_REFETCH}"
echo "[INFO] OVERRIDES=${OVERRIDES[*]}"

cd "${PI3_ROOT}"
export PYTHONPATH="${PI3_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export PI3_NATIVE_MAX_REFETCH
mkdir -p "${PI3_ROOT}/outputs/${RUN_NAME}"
start_pi3_tb_mirror
trap finish_pi3_tb_mirror EXIT

resolve_python_with_accelerate() {
    local candidates=()
    if [ -n "${PYTHON_BIN:-}" ]; then
        candidates+=("${PYTHON_BIN}")
    fi
    candidates+=("/opt/miniconda3/envs/easyvolcap/bin/python" "python3" "python")

    local dep_check_code='import hydra, accelerate, prettytable, easydict, transformers, rich, omegaconf, yaml, tqdm'
    local py
    for py in "${candidates[@]}"; do
        if command -v "${py}" >/dev/null 2>&1 || [ -x "${py}" ]; then
            if "${py}" -c "${dep_check_code}" >/dev/null 2>&1; then
                echo "${py}"
                return 0
            fi
            if [ "${AUTO_INSTALL_DEPS}" = "1" ]; then
                echo "[WARN] Missing runtime deps under ${py}, trying runtime install from internal mirror..." >&2
                if "${py}" -m pip install --user -U ${RUNTIME_PIP_PKGS} \
                    -i "${PIP_INDEX_URL_INTERNAL}" \
                    --extra-index-url "${PIP_EXTRA_INDEX_URL_INTERNAL}" \
                    --trusted-host "${PIP_TRUSTED_HOST_INTERNAL}" >/dev/null 2>&1; then
                    if "${py}" -c "${dep_check_code}" >/dev/null 2>&1; then
                        echo "[INFO] Runtime dependency install succeeded for ${py}." >&2
                        echo "${py}"
                        return 0
                    fi
                fi
                echo "[WARN] Runtime install via internal mirror failed for ${py}." >&2
            fi
        fi
    done
    return 1
}

if ! PYTHON_WITH_ACCELERATE="$(resolve_python_with_accelerate)"; then
    echo "[ERROR] Cannot find a python executable with required runtime deps installed."
    echo "        Tried: PYTHON_BIN, /opt/miniconda3/envs/easyvolcap/bin/python, python3, python"
    echo "        If your env blocks public internet, set internal mirror and rerun:"
    echo "        PIP_INDEX_URL_INTERNAL=${PIP_INDEX_URL_INTERNAL}"
    echo "        PIP_EXTRA_INDEX_URL_INTERNAL=${PIP_EXTRA_INDEX_URL_INTERNAL}"
    echo "        PIP_TRUSTED_HOST_INTERNAL=${PIP_TRUSTED_HOST_INTERNAL}"
    echo "        RUNTIME_PIP_PKGS=${RUNTIME_PIP_PKGS}"
    exit 1
fi

maybe_install_core4_runtime_deps() {
    if ! is_truthy "${CORE4_VAL}"; then
        return 0
    fi
    if is_truthy "${PI3_SKIP_CORE4_RUNTIME_DEPS_CHECK}"; then
        echo "[WARN] Skipping core4 runtime dependency check." >&2
        return 0
    fi

    local dep_check_code='import open3d, plyfile, openpyxl'
    if "${PYTHON_WITH_ACCELERATE}" -c "${dep_check_code}" >/dev/null 2>&1; then
        return 0
    fi
    if [ "${AUTO_INSTALL_DEPS}" != "1" ]; then
        echo "[ERROR] CORE4_VAL=1 requires runtime deps: ${CORE4_RUNTIME_PIP_PKGS}"
        echo "        Set AUTO_INSTALL_DEPS=1 or install them in the runtime image."
        exit 1
    fi

    echo "[WARN] Missing core4 runtime deps, installing: ${CORE4_RUNTIME_PIP_PKGS}" >&2
    "${PYTHON_WITH_ACCELERATE}" -m pip install --user -U ${CORE4_RUNTIME_PIP_PKGS} \
        -i "${PIP_INDEX_URL_INTERNAL}" \
        --extra-index-url "${PIP_EXTRA_INDEX_URL_INTERNAL}" \
        --trusted-host "${PIP_TRUSTED_HOST_INTERNAL}"
    if ! "${PYTHON_WITH_ACCELERATE}" -c "${dep_check_code}" >/dev/null 2>&1; then
        echo "[ERROR] Failed to import core4 runtime deps after installation: ${CORE4_RUNTIME_PIP_PKGS}"
        exit 1
    fi
}

maybe_install_core4_runtime_deps

LAUNCHER=("${PYTHON_WITH_ACCELERATE}" -m accelerate.commands.launch)

"${LAUNCHER[@]}" \
    --config_file "${ACC_CONFIG}" \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MAIN_PROCESS_IP}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    scripts/train_pi3.py \
    "${OVERRIDES[@]}"
