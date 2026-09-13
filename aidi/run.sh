#!/usr/bin/env bash

# Default Values
if [ -z "$USER" ]; then
    echo "[ERROR] USER is not set"
    exit 1
fi
echo "[INFO] USER: $USER"

# Normalize to meshx root. Never select third_party repos by pyproject/setup.py only.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${FORCE_REPO_ROOT:-}"

is_meshx_root() {
    local cand="$1"
    [ -n "${cand}" ] && [ -d "${cand}/easyvolcap" ] && [ -f "${cand}/aidi/run.sh" ]
}

resolve_meshx_root() {
    local cand=""
    local base=""
    local found=""
    for cand in "${SCRIPT_DIR}/.." "${SCRIPT_DIR}/../.." "$(pwd)" "$(pwd)/.." "$(pwd)/../.."; do
        if is_meshx_root "${cand}"; then
            (cd "${cand}" && pwd)
            return 0
        fi
    done
    for base in "${WORKING_PATH:-}" "${PWD:-}"; do
        [ -n "${base}" ] || continue
        found=$(find "${base}" -maxdepth 6 -type d -name "easyvolcap" -print -quit 2>/dev/null || true)
        if [ -n "${found}" ]; then
            cand="$(dirname "${found}")"
            if is_meshx_root "${cand}"; then
                (cd "${cand}" && pwd)
                return 0
            fi
        fi
    done
    return 1
}

if [ -n "${REPO_ROOT}" ]; then
    if ! is_meshx_root "${REPO_ROOT}"; then
        echo "[ERROR] FORCE_REPO_ROOT is not a valid meshx root: ${REPO_ROOT}"
        exit 1
    fi
else
    if ! REPO_ROOT="$(resolve_meshx_root)"; then
        echo "[ERROR] Failed to locate meshx repo root."
        echo "[ERROR] SCRIPT_DIR=${SCRIPT_DIR}"
        echo "[ERROR] PWD=$(pwd)"
        echo "[ERROR] WORKING_PATH=${WORKING_PATH:-}"
        exit 1
    fi
fi

if [ "$(pwd)" != "${REPO_ROOT}" ]; then
    echo "[INFO] Switching to repo root: ${REPO_ROOT}"
    cd "${REPO_ROOT}"
fi
EVC_REPO_ROOT="${REPO_ROOT}"
EVC_PYTHONPATH_ROOT="${EVC_REPO_ROOT}"
if [ -d "/running_package/code_package/meshx/easyvolcap" ]; then
    EVC_PYTHONPATH_ROOT="/running_package/code_package/meshx"
elif [ -d "/running_package/code_package" ]; then
    EVC_PYTHONPATH_ROOT="/running_package/code_package"
fi
export EVC_REPO_ROOT
export EVC_PYTHONPATH_ROOT

EXP_SERIES=${EXP_SERIES:-"baseline"}
DATA_ROOT="/horizon-bucket/saturn_v_dev/01_users/tao02.xie/projects/GFM/gfm/data"
SAVE_ROOT=${SAVE_ROOT:-"/horizon-bucket/saturn_v_4dlabel/004_vision/01_users/$USER/projects/GFM/gfm/$EXP_SERIES"}
SAVE_TO_AIDI=${SAVE_TO_AIDI:-"0"}
TB_ROOT=$SAVE_ROOT
VIS_ROOT=$SAVE_ROOT
TB_MIRROR_INTERVAL=${TB_MIRROR_INTERVAL:-60}
TB_MIRROR_BOOTSTRAP_INTERVAL=${TB_MIRROR_BOOTSTRAP_INTERVAL:-10}
TB_MIRROR_SRC=""
TB_MIRROR_DST=""
TB_MIRROR_PID=""
WATCH_EPOCH_PT_FROM_NPZ=${WATCH_EPOCH_PT_FROM_NPZ:-0}
WATCH_EPOCH_PT_POLL_SECONDS=${WATCH_EPOCH_PT_POLL_SECONDS:-60}
WATCH_EPOCH_PT_PID=""

DEFAULT_CONFIG="vggt/vggt"  # ./configs/exps/vggt/vggt.yaml
DEFAULT_EXP_NAME="vggt/all_data_b24"


# added by xiaoyang: to make this script easier to use for different users
echo "[INFO] USER: $USER"
echo "[INFO] SAVE_TO_AIDI: $SAVE_TO_AIDI"
# if not USER, then exit the script
if [ -z "$USER" ]; then
    echo "[ERROR] USER is not set"
    exit 1
fi

expand_at_file_args() {
    local raw_args="$1"
    local expanded=()
    local token
    local token_unquoted
    local ref
    local line

    if [ -z "$raw_args" ]; then
        echo ""
        return 0
    fi

    for token in $raw_args; do
        token_unquoted="${token}"
        if [ "${token_unquoted#\"}" != "${token_unquoted}" ] && [ "${token_unquoted%\"}" != "${token_unquoted}" ]; then
            token_unquoted="${token_unquoted#\"}"
            token_unquoted="${token_unquoted%\"}"
        fi
        if [ "${token_unquoted#\'}" != "${token_unquoted}" ] && [ "${token_unquoted%\'}" != "${token_unquoted}" ]; then
            token_unquoted="${token_unquoted#\'}"
            token_unquoted="${token_unquoted%\'}"
        fi

        if [[ "$token_unquoted" == @* ]]; then
            ref="${token_unquoted#@}"
            if [ ! -f "$ref" ]; then
                echo "[ERROR] args file not found: $ref"
                exit 1
            fi
            while IFS= read -r line || [ -n "$line" ]; do
                # trim leading/trailing spaces
                line="${line#"${line%%[![:space:]]*}"}"
                line="${line%"${line##*[![:space:]]}"}"
                if [ -z "$line" ]; then
                    continue
                fi
                if [[ "$line" == \#* ]]; then
                    continue
                fi
                expanded+=("$line")
            done < "$ref"
        else
            expanded+=("$token_unquoted")
        fi
    done

    local IFS=' '
    echo "${expanded[*]}"
}

################################################################################
# Parse Command-Line Arguments
#   --config <string>
#   --exp_name <string>
#   --data_path <string>
#
#   --train
#   --train_args "<additional train arguments>"
#
#   --dist
#   --dist_args "<additional dist arguments>"
#
#   --accelerate
#   --accelerate_args "<additional accelerate arguments>"
#
#   --test
#   --test_args "<additional test arguments>"
#
# If no flags ( --train, --test, --accelerate ) are provided,
# then output warning and exit.
################################################################################

# Initialize parameters
CONFIG="$DEFAULT_CONFIG"
EXP_NAME="$DEFAULT_EXP_NAME"
MAX_RETRY=1

# Flags for each step
RUN_CUSTOM=false
RUN_TRAIN=false
RUN_TEST=false
RUN_ACC=false
RUN_DIST=false
MULTI_MACHINE=false

# Extra arguments for each step
CUSTOM_ARGS=""
TRAIN_ARGS=""
TEST_ARGS=""
ACC_ARGS=""
DIST_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --max_retry)
            MAX_RETRY="$2"
            shift 2
            ;;

        # Step flags
        --custom)
            RUN_CUSTOM=true
            shift
            ;;
        --train)
            RUN_TRAIN=true
            shift
            ;;
        --test)
            RUN_TEST=true
            shift
            ;;
        --dist)
            RUN_DIST=true
            RUN_TRAIN=true
            shift
            ;;
        --accelerate)
            RUN_ACC=true
            RUN_TRAIN=true
            shift
            ;;

        # Multi-machine flag
        --multi_machine)
            MULTI_MACHINE=true
            shift
            ;;

        # Additional arguments for each step
        --custom_args)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                if [ -n "$CUSTOM_ARGS" ]; then
                    CUSTOM_ARGS="${CUSTOM_ARGS} $1"
                else
                    CUSTOM_ARGS="$1"
                fi
                shift
            done
            ;;
        --train_args)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                if [ -n "$TRAIN_ARGS" ]; then
                    TRAIN_ARGS="${TRAIN_ARGS} $1"
                else
                    TRAIN_ARGS="$1"
                fi
                shift
            done
            ;;
        --test_args)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                if [ -n "$TEST_ARGS" ]; then
                    TEST_ARGS="${TEST_ARGS} $1"
                else
                    TEST_ARGS="$1"
                fi
                shift
            done
            ;;
        --dist_args)
            shift
            if [[ $# -gt 0 ]]; then
                if [ -n "$DIST_ARGS" ]; then
                    DIST_ARGS="${DIST_ARGS} $1"
                else
                    DIST_ARGS="$1"
                fi
                shift
            fi
            while [[ $# -gt 0 && "$1" != --* ]]; do
                if [ -n "$DIST_ARGS" ]; then
                    DIST_ARGS="${DIST_ARGS} $1"
                else
                    DIST_ARGS="$1"
                fi
                shift
            done
            ;;
        --accelerate_args)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                if [ -n "$ACC_ARGS" ]; then
                    ACC_ARGS="${ACC_ARGS} $1"
                else
                    ACC_ARGS="$1"
                fi
                shift
            done
            ;;

        # Unrecognized argument
        *)
            echo "[WARN] Unrecognized argument: $1"
            shift
            ;;
    esac
done

TRAIN_ARGS="$(expand_at_file_args "${TRAIN_ARGS}")"
TEST_ARGS="$(expand_at_file_args "${TEST_ARGS}")"

# if /job_data exists and SAVE_TO_AIDI!="0", then set SAVE_ROOT to /job_data
if [ -d "/job_data" ] && [ "$SAVE_TO_AIDI" != "0" ]; then
    TB_ROOT="/job_tboard"
    VIS_ROOT="/job_data"
    if ! mkdir -p "$TB_ROOT/record/$EXP_NAME" >/dev/null 2>&1; then
        echo "[WARN] TB_ROOT=$TB_ROOT is unavailable, fallback to /job_data for TensorBoard"
        TB_ROOT="/job_data"
        mkdir -p "$TB_ROOT/record/$EXP_NAME"
    fi
    # start transfer script in the background (sync every 10 min), only on main node
    if [ "${NODE_RANK:-0}" = "0" ]; then
        nohup python3 aidi/transfer_results_to_bucket.py \
            --exp_name "$EXP_NAME" \
            --src_tb_root "$TB_ROOT" \
            --src_vis_root "$VIS_ROOT" \
            --dst_root "$SAVE_ROOT" \
            > /tmp/transfer_daemon.log 2>&1 &
        echo "[INFO] Transfer daemon started on main node (PID: $!)"
    else
        echo "[INFO] Transfer daemon skipped on worker node (NODE_RANK=$NODE_RANK)"
    fi
fi

if [ -d "/job_data" ] && [ "$SAVE_TO_AIDI" != "0" ] && [ "${NODE_RANK:-0}" = "0" ]; then
    if [ "$TB_ROOT" != "/job_data" ]; then
        TB_MIRROR_SRC="$TB_ROOT/record/$EXP_NAME"
        TB_MIRROR_DST="/job_data/record/$EXP_NAME"
        mkdir -p "$TB_MIRROR_SRC"
        mkdir -p "$TB_MIRROR_DST"
        (
            while true; do
                if [ -d "$TB_MIRROR_SRC" ]; then
                    rsync -az --update "$TB_MIRROR_SRC/" "$TB_MIRROR_DST/" >/dev/null 2>&1 || true
                    sleep "$TB_MIRROR_INTERVAL"
                else
                    sleep "$TB_MIRROR_BOOTSTRAP_INTERVAL"
                fi
            done
        ) &
        TB_MIRROR_PID=$!
        echo "[INFO] TB mirror daemon started on main node (PID: $TB_MIRROR_PID)"
    else
        echo "[INFO] TB mirror skipped because TensorBoard writes directly to /job_data"
    fi
fi

if [ "${WATCH_EPOCH_PT_FROM_NPZ}" = "1" ] && [ "${NODE_RANK:-0}" = "0" ]; then
    MODEL_DIR="${SAVE_ROOT}/trained_model/${EXP_NAME}"
    mkdir -p "$MODEL_DIR"
    nohup python3 aidi/scripts/vggt/watch_epoch_npz_snapshots.py \
        --model-dir "$MODEL_DIR" \
        --poll-seconds "$WATCH_EPOCH_PT_POLL_SECONDS" \
        --copy-pt \
        --ensure-latest-pt \
        > /tmp/watch_epoch_pt.log 2>&1 &
    WATCH_EPOCH_PT_PID=$!
    echo "[INFO] PT watcher started on main node (PID: $WATCH_EPOCH_PT_PID)"
fi

trap 'kill "$TB_MIRROR_PID" 2>/dev/null || true; kill "$WATCH_EPOCH_PT_PID" 2>/dev/null || true' EXIT

# If no steps are selected, then output warning and exit.
if [ "$RUN_TRAIN" = false ] && [ "$RUN_TEST" = false ] && [ "$RUN_CUSTOM" = false ]; then
    echo "[ERROR] No execution step selected. Please provide at least one flag: --train, --test, --custom."
    exit 1
fi

# # If either --dist or --accelerate is not provided, then set RUN_DIST to true.
# if [ "$RUN_TRAIN" = true ] && [ "$RUN_ACC" = false ] && [ "$RUN_DIST" = false ]; then
#     RUN_DIST=true
# fi

# xiaoyang: if /job_data exists and RUN_DIST is true, set MULTI_MACHINE to true by default
if [ -d "/job_data" ] && [ "$RUN_DIST" = true ]; then
    MULTI_MACHINE=true
fi

# Display Current Configuration
echo "[INFO] Using CONFIG:    ${CONFIG}"
echo "[INFO] Using EXP_NAME:  ${EXP_NAME}"
echo "[INFO] TRAIN: $RUN_TRAIN, DIST: $RUN_DIST, ACCELERATE: $RUN_ACC, TEST: $RUN_TEST"
echo "[INFO] TRAIN_ARGS: \"$TRAIN_ARGS\""
echo "[INFO] DIST_ARGS: \"$DIST_ARGS\""
echo "[INFO] ACCELERATE_ARGS: \"$ACC_ARGS\""
echo "[INFO] TEST_ARGS: \"$TEST_ARGS\""
echo "[INFO] MULTI_MACHINE: $MULTI_MACHINE"
echo "[INFO] CUSTOM_ARGS: \"$CUSTOM_ARGS\""
echo "[INFO] MAX_RETRY: $MAX_RETRY"

# Activate the conda environment first
if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate easyvolcap
else
    echo "[WARN] Conda not found at /opt/miniconda3, skipping activation"
fi
export PATH="$HOME/.local/bin:$PATH"
EVC_PYTHON_BIN="$(command -v python3 || command -v python)"
export EVC_PYTHON_BIN
echo "[INFO] EVC_PYTHON_BIN=${EVC_PYTHON_BIN}"
"${EVC_PYTHON_BIN}" -m pip install ninja numexpr==2.11.0 pillow==11.2.1 evo==1.31.1 sympy -i http://pypi.hobot.cc/simple --trusted-host pypi.hobot.cc
"${EVC_PYTHON_BIN}" -m pip install open3d plyfile openpyxl -i http://pypi.hobot.cc/simple --trusted-host pypi.hobot.cc

# 1. Environment setup
# export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=1  # may slow down training
# 1.1 Set the NCCL_IB_DISABLE to force the use of the IB network.
# export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=600
export NCCL_IB_TIMEOUT=31
# NOTE: the following 3 settings may slow down training
# export NCCL_P2P_DISABLE=1
# export NCCL_P2P_LEVEL=NVL
# export TORCH_NCCL_BLOCKING_WAIT=1
export FI_VRDMA_FORK_SAFE=1
# NOTE: H20 in bcloud seems to have a bug in the IB network, so we need to set this to 0
export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_NTHREADS="8"
export NCCL_NSOCKS_PERTHREAD="2"
export NCCL_IB_QPS_PER_CONNECTION=8
export IB_NET_GDR_LEVEL=2
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GDR_LEVEL=2
# Enable CUDNN v8
export TORCH_CUDNN_V8_API_ENABLED=1
# export UCX_IB_GPU_DIRECT_RDMA=yes
# export UCX_TLS=rc,cuda_copy,gdr_copy
# export NCCL_IB_GDR_LEVEL=2
# export ENV_VRDMA_PROVIDER_RNR_REFILL_SQE=1
# Enable OpenEXR
export OPENCV_IO_ENABLE_OPENEXR=1

# 1.2 Conda-related stuffs
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID # OPTIONAL: defaults to capability order, might be different for GL and CUDA

# 1.3 Install easyvolcap and add it to PATH
export PATH=~/.local/bin:$PATH  # the default easyvolcap installation path is not in PATH
if [ "$(pwd)" != "${EVC_REPO_ROOT}" ]; then
    cd "${EVC_REPO_ROOT}"
fi
if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
    if [ -d "/job_data" ]; then
        pip install --upgrade pip -i http://pypi.hobot.cc/simple --trusted-host pypi.hobot.cc
        pip install . --no-deps -i http://pypi.hobot.cc/simple --trusted-host pypi.hobot.cc  # remember to install the package
    else
        pip install -e . --no-build-isolation --no-deps -i http://pypi.hobot.cc/simple --trusted-host pypi.hobot.cc  # remember to install the package
    fi
else
    echo "[WARN] pyproject.toml/setup.py not found under $(pwd), skip pip install"
fi
export PYTHONPATH="${EVC_REPO_ROOT}:${EVC_PYTHONPATH_ROOT}:${PYTHONPATH:-}"
if ! "${EVC_PYTHON_BIN}" - <<'PY'
import sys
import easyvolcap
print("[INFO] easyvolcap import ok:", easyvolcap.__file__)
print("[INFO] sys.path[0:3]:", sys.path[:3])
PY
then
    echo "[ERROR] Failed to import easyvolcap after environment setup."
    echo "[ERROR] PWD=$(pwd)"
    echo "[ERROR] EVC_REPO_ROOT=${EVC_REPO_ROOT}"
    echo "[ERROR] EVC_PYTHONPATH_ROOT=${EVC_PYTHONPATH_ROOT}"
    exit 1
fi

# Triton cache/setup (avoid FileNotFound on some cluster envs).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TRITON_CACHE_DIR}}"
mkdir -p "${TRITON_CACHE_DIR}"
if [ ! -w "${TRITON_CACHE_DIR}" ]; then
    # Fallback when /tmp cache dir exists but is not writable by current user.
    # Some shared containers expose a mismatched HOME (e.g. another user's path),
    # so resolve the login-home from passwd first.
    RESOLVED_USER_HOME=""
    if [ -n "${USER:-}" ] && command -v getent >/dev/null 2>&1; then
        RESOLVED_USER_HOME="$(getent passwd "${USER}" | cut -d: -f6 || true)"
    fi
    if [ -z "${RESOLVED_USER_HOME}" ]; then
        RESOLVED_USER_HOME="${HOME:-/tmp}"
    fi
    TRITON_CACHE_FALLBACK="${RESOLVED_USER_HOME}/.cache/triton_cache_${USER}"
    mkdir -p "${TRITON_CACHE_FALLBACK}" 2>/dev/null || true
    if [ -w "${TRITON_CACHE_FALLBACK}" ]; then
        export TRITON_CACHE_DIR="${TRITON_CACHE_FALLBACK}"
        export XDG_CACHE_HOME="${TRITON_CACHE_DIR}"
    else
        # Last-resort fallback under repo tmp for strict permission environments.
        TRITON_CACHE_FALLBACK="${EVC_REPO_ROOT:-$(pwd)}/tmp/triton_cache_${USER}"
        mkdir -p "${TRITON_CACHE_FALLBACK}"
        export TRITON_CACHE_DIR="${TRITON_CACHE_FALLBACK}"
        export XDG_CACHE_HOME="${TRITON_CACHE_DIR}"
    fi
fi
if command -v gcc >/dev/null 2>&1; then
    GCC_BIN="$(command -v gcc)"
else
    GCC_BIN="/usr/local/gcc-11.4/bin/gcc"
fi
export TRITON_CC="${GCC_BIN}"
export CC="${GCC_BIN}"
GCC_DIR="$(dirname "${GCC_BIN}")"
if [ -x "${GCC_DIR}/g++" ]; then
    export CXX="${GCC_DIR}/g++"
elif command -v g++ >/dev/null 2>&1; then
    export CXX="$(command -v g++)"
fi
export PATH="${GCC_DIR}:${PATH}"
echo "[INFO] TRITON_CC=${TRITON_CC} CC=${CC} CXX=${CXX:-} PATH=${GCC_DIR}:..."

# Ensure Triton is available for sparse flash attention.
if ! "${EVC_PYTHON_BIN}" - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("triton") is not None else 1)
PY
then
    echo "[WARN] triton not found, installing..."
    "${EVC_PYTHON_BIN}" -m pip install triton -i http://pypi.hobot.cc/simple --trusted-host pypi.hobot.cc
fi
"${EVC_PYTHON_BIN}" - <<'PY'
import os
try:
    os.environ.setdefault("TRITON_CACHE_DIR", os.environ.get("TRITON_CACHE_DIR", "/tmp/triton_cache"))
    os.environ.setdefault("XDG_CACHE_HOME", os.environ.get("XDG_CACHE_HOME", os.environ["TRITON_CACHE_DIR"]))
    from easyvolcap.utils.custom_flash_attn.sparse_index_flash_attn import sparse_index_flash_attn_func
    print("[INFO] sparse_index_flash_attn_func:", sparse_index_flash_attn_func)
except Exception as e:
    import traceback
    print("[WARN] sparse_index_flash_attn import failed:", repr(e))
    traceback.print_exc()
PY
# Always ensure evc-dist shim matches the debug environment.
echo "[INFO] Ensuring evc-dist shim under ~/.local/bin"
mkdir -p ~/.local/bin
cat > ~/.local/bin/evc-dist <<'EOF'
#!/usr/bin/env bash
export PYTHONPATH="${EVC_REPO_ROOT}:${EVC_PYTHONPATH_ROOT}:${PYTHONPATH:-}"
"${EVC_PYTHON_BIN:-python3}" - "$@" <<'PY'
from easyvolcap.scripts.wrap import dist_entrypoint
dist_entrypoint()
PY
EOF
chmod +x ~/.local/bin/evc-dist
which evc-acc


# 1.4 To avoid CUDA OOM
# https://blog.csdn.net/MirageTanker/article/details/127998036
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1 # DO NOT USE THIS, extremely slow
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# ,garbage_collection_threshold:0.8
# ,backend:cudaMallocAsync
# export CUDA_MODULE_LOADING=LAZY

# pip install --upgrade deepspeed -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
# # IP and PORT
# hostname -I | awk '{print $1}'
# python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
# echo $NODE_RANK
# echo $HOST_NODE_ADDR
# # Triton
# mkdir -p /home/users/tao02.xie/.triton/autotune
# chmod 700 /home/users/tao02.xie/.triton
# chmod 700 /home/users/tao02.xie/.triton/autotune


#  Wrap all training-related invocations in a loop up to $MAX_RETRY times.,
#  incrementing MASTER_PORT by 1 on each retry if it's a multi-machine job.
if { [ "$RUN_CUSTOM" = true ] || [ "$RUN_ACC" = true ] || [ "$RUN_DIST" = true ] || [ "$RUN_TRAIN" = true ]; } && [ "$RUN_TEST" = false ]; then
    COUNT=1
    while [ $COUNT -le $MAX_RETRY ]; do
        echo "[INFO] Attempt #$COUNT of $MAX_RETRY"

        # Custom training
        if [ "$RUN_CUSTOM" = true ]; then
            echo "# Custom Training"
            $CUSTOM_ARGS

        # Distributed training using accelerate
        elif [ "$RUN_ACC" = true ]; then
            echo "# Distributed Training Using Accelerate"
            if [ "$MULTI_MACHINE" = true ]; then
                if [ -n "$ACC_ARGS" ]; then
                    evc-acc $ACC_ARGS \
                        --machine_rank=$NODE_RANK \
                        --main_process_ip=$HOST_NODE_ADDR \
                        --num_machines=$NUM_NODES \
                        --num_processes=$WORLD_SIZE -- \
                        -c configs/exps/"${CONFIG}".yaml \
                        exp_name="${EXP_NAME}" \
                        runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                        runner_cfg.recorder_cfg.record_dir="${SAVE_ROOT}/record/${EXP_NAME}" \
                        runner_cfg.visualizer_cfg.result_dir="${SAVE_ROOT}/result" \
                        $TRAIN_ARGS
                else
                    evc-acc \
                        --machine_rank=$NODE_RANK \
                        --main_process_ip=$HOST_NODE_ADDR \
                        --num_machines=$NUM_NODES \
                        --num_processes=$WORLD_SIZE -- \
                        -c configs/exps/"${CONFIG}".yaml \
                        exp_name="${EXP_NAME}" \
                        runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                        runner_cfg.recorder_cfg.record_dir="${SAVE_ROOT}/record/${EXP_NAME}" \
                        runner_cfg.visualizer_cfg.result_dir="${SAVE_ROOT}/result" \
                        $TRAIN_ARGS
                fi
            else
                if [ -n "$ACC_ARGS" ]; then
                    evc-acc $ACC_ARGS -- \
                        -c configs/exps/"${CONFIG}".yaml \
                        exp_name="${EXP_NAME}" \
                        runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                        runner_cfg.recorder_cfg.record_dir="${SAVE_ROOT}/record/${EXP_NAME}" \
                        runner_cfg.visualizer_cfg.result_dir="${SAVE_ROOT}/result" \
                        $TRAIN_ARGS
                else
                    evc-acc \
                        -c configs/exps/"${CONFIG}".yaml \
                        exp_name="${EXP_NAME}" \
                        runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                        runner_cfg.recorder_cfg.record_dir="${SAVE_ROOT}/record/${EXP_NAME}" \
                        runner_cfg.visualizer_cfg.result_dir="${SAVE_ROOT}/result" \
                        $TRAIN_ARGS
                fi
            fi

        # Distributed training using native PyTorch DDP
        elif [ "$RUN_DIST" = true ]; then
            echo "# Distributed Training"
            if [ "$MULTI_MACHINE" = true ]; then
                if [ -n "$DIST_ARGS" ]; then
                    evc-dist \
                        --nnodes=$NUM_NODES \
                        --nproc_per_node=$GPU_PER_NODE \
                        --node_rank=$NODE_RANK \
                        --rdzv_backend=${RDZV_BACKEND:-c10d} \
                        --rdzv_endpoint=$HOST_NODE_ADDR:$MASTER_PORT \
                        $DIST_ARGS -- \
                        -c configs/exps/"${CONFIG}".yaml \
                        exp_name="${EXP_NAME}" \
                        runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                        runner_cfg.recorder_cfg.record_dir="${TB_ROOT}/record/${EXP_NAME}" \
                        runner_cfg.visualizer_cfg.result_dir="${VIS_ROOT}/result" \
                        $TRAIN_ARGS
                else
                    evc-dist \
                        --nnodes=$NUM_NODES \
                        --nproc_per_node=$GPU_PER_NODE \
                        --node_rank=$NODE_RANK \
                        --rdzv_backend=${RDZV_BACKEND:-c10d} \
                        --rdzv_endpoint=$HOST_NODE_ADDR:$MASTER_PORT -- \
                        -c configs/exps/"${CONFIG}".yaml \
                        exp_name="${EXP_NAME}" \
                        runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                        runner_cfg.recorder_cfg.record_dir="${TB_ROOT}/record/${EXP_NAME}" \
                        runner_cfg.visualizer_cfg.result_dir="${VIS_ROOT}/result" \
                        $TRAIN_ARGS
                fi
            else
                if [ -n "$DIST_ARGS" ]; then
                    evc-dist $DIST_ARGS -- \
                        -c configs/exps/"${CONFIG}".yaml \
                        exp_name="${EXP_NAME}" \
                        runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                        runner_cfg.recorder_cfg.record_dir="${TB_ROOT}/record/${EXP_NAME}" \
                        runner_cfg.visualizer_cfg.result_dir="${VIS_ROOT}/result" \
                        $TRAIN_ARGS
                else
                    evc-dist \
                        -c configs/exps/"${CONFIG}".yaml \
                        exp_name="${EXP_NAME}" \
                        runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                        runner_cfg.recorder_cfg.record_dir="${TB_ROOT}/record/${EXP_NAME}" \
                        runner_cfg.visualizer_cfg.result_dir="${VIS_ROOT}/result" \
                        $TRAIN_ARGS
                fi
            fi

        # Local training
        elif [ "$RUN_TRAIN" = true ]; then
            echo "# Normal Training"
            if [ -n "$TRAIN_ARGS" ]; then
                evc-train \
                    -c configs/exps/"${CONFIG}".yaml \
                    exp_name="${EXP_NAME}" \
                    runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                    runner_cfg.recorder_cfg.record_dir="${SAVE_ROOT}/record/${EXP_NAME}" \
                    runner_cfg.visualizer_cfg.result_dir="${SAVE_ROOT}/result" \
                    $TRAIN_ARGS
            else
                evc-train \
                    -c configs/exps/"${CONFIG}".yaml \
                    exp_name="${EXP_NAME}" \
                    runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
                    runner_cfg.recorder_cfg.record_dir="${SAVE_ROOT}/record/${EXP_NAME}" \
                    runner_cfg.visualizer_cfg.result_dir="${SAVE_ROOT}/result"
            fi
        fi

        RET_VAL=$?

        # If this is a multi-machine job, bump MASTER_PORT by 1 before next attempt.
        if [ "$MULTI_MACHINE" = true ]; then
            MASTER_PORT=$((MASTER_PORT + 1))
        fi

        # Increment the loop counter
        COUNT=$((COUNT + 1))

        # Short sleep to avoid immediate rapid restarts
        sleep 5
    done
fi


# 2. Test the model using the specified configuration.
if [ "$RUN_TEST" = true ]; then
    echo "# Testing"
    evc-dist $DIST_ARGS -- \
        -t test \
        -c configs/exps/"${CONFIG}".yaml \
        exp_name="${EXP_NAME}" \
        runner_cfg.trained_model="${SAVE_ROOT}/trained_model/${EXP_NAME}" \
        runner_cfg.recorder_cfg.record_dir="${SAVE_ROOT}/record/${EXP_NAME}" \
        runner_cfg.visualizer_cfg.result_dir="${SAVE_ROOT}/result" \
        $TEST_ARGS
    RET_VAL=$?
fi

# Perform final sync if transfer daemon was started, only on main node
if [ -d "/job_data" ] && [ "$SAVE_TO_AIDI" != "0" ] && [ "${NODE_RANK:-0}" = "0" ]; then
    echo "[INFO] Final sync on main node..."
    if [ -n "$TB_MIRROR_PID" ]; then
        kill "$TB_MIRROR_PID" 2>/dev/null || true
        wait "$TB_MIRROR_PID" 2>/dev/null || true
    fi
    if [ -n "$TB_MIRROR_SRC" ] && [ -n "$TB_MIRROR_DST" ] && [ -d "$TB_MIRROR_SRC" ]; then
        rsync -az --update "$TB_MIRROR_SRC/" "$TB_MIRROR_DST/" >/dev/null 2>&1 || true
    fi
    [ -d "$TB_ROOT/record/$EXP_NAME" ] && rsync -avz --update "$TB_ROOT/record/$EXP_NAME/" "$SAVE_ROOT/record/$EXP_NAME/"
    [ -d "$VIS_ROOT/result/$EXP_NAME" ] && rsync -avz --update "$VIS_ROOT/result/$EXP_NAME/" "$SAVE_ROOT/result/$EXP_NAME/"
fi


if [ $RET_VAL -ne 0 ]; then
    echo "[ERROR] Training failed"
    exit $RET_VAL
fi
