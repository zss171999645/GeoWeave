#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash release/geoweave_paper_handoff/scripts/train_pi3.sh --warmup-smoke [--dry-run]
  bash release/geoweave_paper_handoff/scripts/train_pi3.sh --warmup-paper [--dry-run]
  bash release/geoweave_paper_handoff/scripts/train_pi3.sh --smoke [--dry-run]
  bash release/geoweave_paper_handoff/scripts/train_pi3.sh --paper [--dry-run]
  bash release/geoweave_paper_handoff/scripts/train_pi3.sh --resume-final [--dry-run]

Modes:
  --warmup-smoke 2-GPU native Pi3 GeoWeave indexer warm-up smoke from the fixed
                 Pi3 base checkpoint. Runs tiny low-resolution warm-up through
                 forward/backward/optimizer and exits.
  --warmup-paper Paper warm-up recipe that produced the checkpoint_49 family
                 used by sparse stage2.
  --smoke        2-GPU sparse/indexer stage2 smoke from the warm-up checkpoint.
                 Runs tiny low-resolution training through forward/backward/
                 optimizer and exits.
  --paper        Paper sparse/indexer stage2 recipe used for the 20260505 Pi3
                 GeoWeave run.
  --resume-final Continue training from the fixed handoff checkpoint_79 full
                 accelerator state. Keep NUM_PROCESSES aligned with the target
                 run for formal continuation.

Environment overrides:
  GPU_IDS, NUM_PROCESSES, SAVE_ROOT, RUN_NAME, EXP_NAME, WORK_DIR, MODEL_CKPT,
  TRAIN_RESUME, PI3_PRETRAIN_CKPT, and EXTRA_OVERRIDES may be overridden by
  the caller.
  DRY_RUN=1 prints the resolved command instead of executing it.
EOF
}

count_gpu_ids() {
    local ids="$1"
    awk -F',' '{print NF}' <<<"${ids}"
}

MODE_NAME=""
DRY_RUN="${DRY_RUN:-0}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --smoke)
            MODE_NAME="smoke"
            shift
            ;;
        --warmup-smoke)
            MODE_NAME="warmup-smoke"
            shift
            ;;
        --warmup-paper)
            MODE_NAME="warmup-paper"
            shift
            ;;
        --paper)
            MODE_NAME="paper"
            shift
            ;;
        --resume-final)
            MODE_NAME="resume-final"
            shift
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
            exit 1
            ;;
    esac
done

if [ -z "${MODE_NAME}" ]; then
    usage >&2
    exit 2
fi

SAVE_ROOT="${SAVE_ROOT:-/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline}"
HANDOFF_WEIGHT_ROOT="${HANDOFF_WEIGHT_ROOT:-${SAVE_ROOT}/trained_model/geoweave_paper_handoff_20260602_final}"
PI3_FINAL_HANDOFF_ROOT="${PI3_FINAL_HANDOFF_ROOT:-${HANDOFF_WEIGHT_ROOT}/pi3_geoweave_native_sparse_20260505_checkpoint_79}"
PI3_FINAL_CKPT_DIR="${PI3_FINAL_CKPT_DIR:-${PI3_FINAL_HANDOFF_ROOT}/checkpoint_79}"
PI3_FINAL_MODEL_BIN="${PI3_FINAL_MODEL_BIN:-${PI3_FINAL_CKPT_DIR}/pytorch_model.bin}"
PI3_STAGE2_INIT_CKPT="${PI3_STAGE2_INIT_CKPT:-${HANDOFF_WEIGHT_ROOT}/pi3_geoweave_stage2_init_checkpoint_49/pytorch_model.bin}"
PI3_BASE_CKPT="${PI3_BASE_CKPT:-${HANDOFF_WEIGHT_ROOT}/pi3_base_yyfz233/Pi3_model.safetensors}"

export SAVE_ROOT
export HANDOFF_WEIGHT_ROOT
export PI3_BASE_CKPT
export LOAD_VGGT="${LOAD_VGGT:-0}"
export ACC_CONFIG="${ACC_CONFIG:-configs/accelerate/ddp.yaml}"
export AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
export AIDI_SKIP_BASE_PIP_INSTALL="${AIDI_SKIP_BASE_PIP_INSTALL:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export SPARSE_LR="${SPARSE_LR:-1e-5}"
export SPARSE_LR_DECAY="${SPARSE_LR_DECAY:-scheduler}"
export SPARSE_LOSS_WEIGHT="${SPARSE_LOSS_WEIGHT:-10}"
export SPARSE_SCORE_DTYPE="${SPARSE_SCORE_DTYPE:-float16}"
export PI3_SPARSE_HEAD_CHUNK_SIZE="${PI3_SPARSE_HEAD_CHUNK_SIZE:-4}"
export PI3_SPARSE_VIEW_CHUNK_SIZE="${PI3_SPARSE_VIEW_CHUNK_SIZE:-4}"
export PI3_SPARSE_QUERY_CHUNK_SIZE="${PI3_SPARSE_QUERY_CHUNK_SIZE:-4096}"
export PI3_FIND_UNUSED_PARAMETERS="${PI3_FIND_UNUSED_PARAMETERS:-0}"
export PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT="${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT:-0}"
export PI3_DECODER_ATTN_BACKEND="${PI3_DECODER_ATTN_BACKEND:-flash}"
export PI3_QK_NORM_CHUNK_SIZE="${PI3_QK_NORM_CHUNK_SIZE:-2048}"
export PI3_HEAD_USE_CHECKPOINT="${PI3_HEAD_USE_CHECKPOINT:-1}"
export PI3_HEAD_VIEW_CHUNK_SIZE="${PI3_HEAD_VIEW_CHUNK_SIZE:-4}"
export PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE="${PI3_NORMAL_LOSS_VIEW_CHUNK_SIZE:-1}"
export TRAIN_MODEL_DTYPE="${TRAIN_MODEL_DTYPE:-bf16}"
export TRAIN_GRAD_ACCUM_STEPS="${TRAIN_GRAD_ACCUM_STEPS:-2}"
export TRAIN_CLIP_GRAD="${TRAIN_CLIP_GRAD:-1.0}"
export TRAIN_CLIP_GRAD_SEPARATE_INDEXER="${TRAIN_CLIP_GRAD_SEPARATE_INDEXER:-1}"
export TRAIN_CLIP_INDEXER_GRAD="${TRAIN_CLIP_INDEXER_GRAD:-1.0}"
export TRAIN_LR="${TRAIN_LR:-1e-5}"
export TRAIN_ENCODER_LR="${TRAIN_ENCODER_LR:-1e-6}"
export TRAIN_SCHEDULER_MAX_LR="${TRAIN_SCHEDULER_MAX_LR:-1e-5}"
export TRAIN_SCHEDULER_PCT_START="${TRAIN_SCHEDULER_PCT_START:-0.03}"
export TRAIN_SCHEDULER_DIV_FACTOR="${TRAIN_SCHEDULER_DIV_FACTOR:-10}"
export TRAIN_SCHEDULER_FINAL_DIV_FACTOR="${TRAIN_SCHEDULER_FINAL_DIV_FACTOR:-10}"
export TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-2}"
export TEST_PREFETCH_FACTOR="${TEST_PREFETCH_FACTOR:-2}"
export TRAIN_PERSISTENT_WORKERS="${TRAIN_PERSISTENT_WORKERS:-0}"
export TEST_PERSISTENT_WORKERS="${TEST_PERSISTENT_WORKERS:-0}"
export TRAIN_PIN_MEMORY="${TRAIN_PIN_MEMORY:-0}"
export TEST_PIN_MEMORY="${TEST_PIN_MEMORY:-0}"
export PI3_LAZY_SEQUENCE_INDEX="${PI3_LAZY_SEQUENCE_INDEX:-1}"
export PI3_USE_INDEX_CACHE="${PI3_USE_INDEX_CACHE:-1}"
export PI3_REBUILD_INDEX_CACHE="${PI3_REBUILD_INDEX_CACHE:-0}"
export PI3_INDEX_CACHE_WAIT_SEC="${PI3_INDEX_CACHE_WAIT_SEC:-0}"

if [ "${MODE_NAME}" = "warmup-smoke" ]; then
    export STAGE="${STAGE:-warmup}"
    export TRAIN_CFG="${TRAIN_CFG:-train_pi3_lowres_indexer_warmup}"
    export DATA_CFG="${DATA_CFG:-meshx_tartanair}"
    export GPU_IDS="${GPU_IDS:-0,1}"
    export NUM_PROCESSES="${NUM_PROCESSES:-$(count_gpu_ids "${GPU_IDS}")}"
    export NUM_MACHINES="${NUM_MACHINES:-1}"
    export PI3_PRETRAIN_CKPT="${PI3_PRETRAIN_CKPT:-${PI3_BASE_CKPT}}"
    export MODEL_CKPT="${MODEL_CKPT:-${PI3_PRETRAIN_CKPT}}"
    export RUN_NAME="${RUN_NAME:-pi3_geoweave_warmup_smoke_$(date +%Y%m%d_%H%M%S)}"
    export EXP_NAME="${EXP_NAME:-release/geoweave_paper_handoff/${RUN_NAME}}"
    export WORK_DIR="${WORK_DIR:-${SAVE_ROOT}/record/${EXP_NAME}}"
    export PI3_WORK_DIR="${PI3_WORK_DIR:-${WORK_DIR}}"
    export PI3_OUTPUT_DIR="${PI3_OUTPUT_DIR:-${PI3_WORK_DIR}/outputs/${RUN_NAME}}"
    export PI3_TB_MIRROR_DIR="${PI3_TB_MIRROR_DIR:-${WORK_DIR}}"
    export SAVE_TO_AIDI="${SAVE_TO_AIDI:-0}"
    export SMOKE="${SMOKE:-1}"
    export SMOKE_ITERS="${SMOKE_ITERS:-2}"
    export SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
    export SMOKE_RES="${SMOKE_RES:-112}"
    export SMOKE_MAX_IMG_PER_GPU="${SMOKE_MAX_IMG_PER_GPU:-2}"
    export SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-1}"
    export SMOKE_SEQ_NUM="${SMOKE_SEQ_NUM:-1}"
    export SMOKE_MAX_FRAMES_PER_SEQUENCE="${SMOKE_MAX_FRAMES_PER_SEQUENCE:-12}"
    export INDEXER_LAYERS="${INDEXER_LAYERS:-all}"
    export INDEXER_LAYERS_TAG="${INDEXER_LAYERS_TAG:-all}"
    export TOPK="${TOPK:-512}"
    export WARMUP_LOSS_MODE="${WARMUP_LOSS_MODE:-kl}"
    export WARMUP_LR="${WARMUP_LR:-1e-4}"
    export WARMUP_STEPS="${WARMUP_STEPS:-40000}"
    export WARMUP_NUM_EPOCH="${WARMUP_NUM_EPOCH:-1}"
    export WARMUP_ITERS_PER_EPOCH="${WARMUP_ITERS_PER_EPOCH:-2}"
    export TRAIN_IMAGE_NUM_RANGE="${TRAIN_IMAGE_NUM_RANGE:-[2,2]}"
    export TRAIN_DYNAMIC_RES="${TRAIN_DYNAMIC_RES:-0}"
    export TRAIN_RES="${TRAIN_RES:-112}"
    export TEST_RES="${TEST_RES:-112}"
    export TRAIN_MAX_IMG_PER_GPU="${TRAIN_MAX_IMG_PER_GPU:-2}"
    export TRAIN_MODEL_DTYPE="${TRAIN_MODEL_DTYPE:-bf16}"
    export TRAIN_CLIP_LOSS="${TRAIN_CLIP_LOSS:-1000000}"
    export PI3_TRAIN_NUM_WORKERS="${PI3_TRAIN_NUM_WORKERS:-1}"
    export PI3_TEST_NUM_WORKERS="${PI3_TEST_NUM_WORKERS:-1}"
    export TEST_ITERS_PER_TEST="${TEST_ITERS_PER_TEST:-0}"
    export WARMUP_ONLY_INDEXER_TRAIN="${WARMUP_ONLY_INDEXER_TRAIN:-True}"
    export STREAMING_KL_LOSS="${STREAMING_KL_LOSS:-1}"
    export STREAMING_KL_AUTOGRAD="${STREAMING_KL_AUTOGRAD:-1}"
    export STREAMING_KL_SCORE_MODE="${STREAMING_KL_SCORE_MODE:-legacy}"
    export STREAMING_KL_FWD_MODE="${STREAMING_KL_FWD_MODE:-score}"
    export STREAMING_KL_BWD_MODE="${STREAMING_KL_BWD_MODE:-score}"
    export PI3_FIND_UNUSED_PARAMETERS="${PI3_FIND_UNUSED_PARAMETERS:-0}"
    export PI3_STATIC_GRAPH="${PI3_STATIC_GRAPH:-1}"
    export LOG_CKPT_INTERVAL="${LOG_CKPT_INTERVAL:-999}"
    export LOG_MAX_CHECKPOINTS="${LOG_MAX_CHECKPOINTS:-1}"
    export PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT="${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT:-0}"
    export INDEXER_HEAD_CHUNK_SIZE="${INDEXER_HEAD_CHUNK_SIZE:-4}"
    export INDEXER_SCORE_HEAD_CHUNK_SIZE="${INDEXER_SCORE_HEAD_CHUNK_SIZE:-4}"
    export INDEXER_SCORE_KEY_CHUNK_SIZE="${INDEXER_SCORE_KEY_CHUNK_SIZE:-4096}"
    export PI3_INDEXING_WORKERS="${PI3_INDEXING_WORKERS:-4}"
    export PI3_DATASET_CACHE_DIR="${PI3_DATASET_CACHE_DIR:-${SAVE_ROOT}/cache/pi3_dataset_index/geoweave_handoff_warmup_smoke}"
elif [ "${MODE_NAME}" = "warmup-paper" ]; then
    export STAGE="${STAGE:-warmup}"
    export TRAIN_CFG="${TRAIN_CFG:-train_pi3_lowres_indexer_warmup}"
    export DATA_CFG="${DATA_CFG:-meshx_pi3_vggt17}"
    export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
    export NUM_MACHINES="${NUM_MACHINES:-2}"
    export NUM_PROCESSES="${NUM_PROCESSES:-16}"
    export PI3_PRETRAIN_CKPT="${PI3_PRETRAIN_CKPT:-${PI3_BASE_CKPT}}"
    export MODEL_CKPT="${MODEL_CKPT:-${PI3_PRETRAIN_CKPT}}"
    export RUN_NAME="${RUN_NAME:-pi3_native_indexerwarmup_highres_scratch_streamkl_lowlr1e4_freezepi3_dynres_clipfix_alllayer_handoff}"
    export EXP_NAME="${EXP_NAME:-pi3/native_indexer_warmup/${RUN_NAME}}"
    export WORK_DIR="${WORK_DIR:-${SAVE_ROOT}/record/${EXP_NAME}}"
    export PI3_WORK_DIR="${PI3_WORK_DIR:-${SAVE_ROOT}/trained_model/pi3/official}"
    export PI3_OUTPUT_DIR="${PI3_OUTPUT_DIR:-${PI3_WORK_DIR}/outputs/${RUN_NAME}}"
    export PI3_TB_MIRROR_DIR="${PI3_TB_MIRROR_DIR:-${WORK_DIR}}"
    export SAVE_TO_AIDI="${SAVE_TO_AIDI:-1}"
    export CLUSTER="${CLUSTER:-project-5090-4dlabel-depthgt-bcloud}"
    export INDEXER_LAYERS="${INDEXER_LAYERS:-all}"
    export INDEXER_LAYERS_TAG="${INDEXER_LAYERS_TAG:-all}"
    export TOPK="${TOPK:-512}"
    export WARMUP_LOSS_MODE="${WARMUP_LOSS_MODE:-kl}"
    export WARMUP_LR="${WARMUP_LR:-1e-4}"
    export WARMUP_STEPS="${WARMUP_STEPS:-40000}"
    export WARMUP_NUM_EPOCH="${WARMUP_NUM_EPOCH:-80}"
    export WARMUP_ITERS_PER_EPOCH="${WARMUP_ITERS_PER_EPOCH:-800}"
    export TRAIN_IMAGE_NUM_RANGE="${TRAIN_IMAGE_NUM_RANGE:-[2,24]}"
    export TRAIN_DYNAMIC_RES="${TRAIN_DYNAMIC_RES:-1}"
    export TRAIN_DYNAMIC_ASPECT_RATIO_RANGE="${TRAIN_DYNAMIC_ASPECT_RATIO_RANGE:-[0.5,2.0]}"
    export TRAIN_DYNAMIC_PIXEL_COUNT_RANGE="${TRAIN_DYNAMIC_PIXEL_COUNT_RANGE:-[100000,255000]}"
    export TRAIN_DYNAMIC_PATCH_SIZE="${TRAIN_DYNAMIC_PATCH_SIZE:-14}"
    export TRAIN_DYNAMIC_NUM_RESOLUTION="${TRAIN_DYNAMIC_NUM_RESOLUTION:--1}"
    export TEST_RES="${TEST_RES:-518}"
    export TRAIN_MODEL_DTYPE="${TRAIN_MODEL_DTYPE:-bf16}"
    export TRAIN_MAX_IMG_PER_GPU="${TRAIN_MAX_IMG_PER_GPU:-24}"
    export TRAIN_CLIP_LOSS="${TRAIN_CLIP_LOSS:-1000000}"
    export PI3_TRAIN_NUM_WORKERS="${PI3_TRAIN_NUM_WORKERS:-4}"
    export PI3_TEST_NUM_WORKERS="${PI3_TEST_NUM_WORKERS:-4}"
    export TEST_ITERS_PER_TEST="${TEST_ITERS_PER_TEST:-0}"
    export WARMUP_ONLY_INDEXER_TRAIN="${WARMUP_ONLY_INDEXER_TRAIN:-True}"
    export STREAMING_KL_LOSS="${STREAMING_KL_LOSS:-1}"
    export STREAMING_KL_AUTOGRAD="${STREAMING_KL_AUTOGRAD:-1}"
    export STREAMING_KL_SCORE_MODE="${STREAMING_KL_SCORE_MODE:-legacy}"
    export STREAMING_KL_FWD_MODE="${STREAMING_KL_FWD_MODE:-score}"
    export STREAMING_KL_BWD_MODE="${STREAMING_KL_BWD_MODE:-score}"
    export PI3_FIND_UNUSED_PARAMETERS="${PI3_FIND_UNUSED_PARAMETERS:-0}"
    export PI3_STATIC_GRAPH="${PI3_STATIC_GRAPH:-1}"
    export LOG_CKPT_INTERVAL="${LOG_CKPT_INTERVAL:-1}"
    export LOG_MAX_CHECKPOINTS="${LOG_MAX_CHECKPOINTS:-10}"
    export PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT="${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT:-0}"
    export INDEXER_HEAD_CHUNK_SIZE="${INDEXER_HEAD_CHUNK_SIZE:-4}"
    export INDEXER_SCORE_HEAD_CHUNK_SIZE="${INDEXER_SCORE_HEAD_CHUNK_SIZE:-4}"
    export INDEXER_SCORE_KEY_CHUNK_SIZE="${INDEXER_SCORE_KEY_CHUNK_SIZE:-4096}"
    export PI3_INDEXING_WORKERS="${PI3_INDEXING_WORKERS:-16}"
    export PI3_DATASET_CACHE_DIR="${PI3_DATASET_CACHE_DIR:-${SAVE_ROOT}/cache/pi3_dataset_index/native_indexerwarmup_highres_scratch_streamkl_lowlr1e4_freezepi3_dynres_clipfix_alllayer}"
elif [ "${MODE_NAME}" = "smoke" ]; then
    export STAGE="${STAGE:-sparse}"
    export TRAIN_CFG="${TRAIN_CFG:-train_pi3_lowres_indexer_sparse}"
    export DATA_CFG="${DATA_CFG:-meshx_tartanair}"
    export INDEXER_LAYERS="${INDEXER_LAYERS:-9-17}"
    export INDEXER_LAYERS_TAG="${INDEXER_LAYERS_TAG:-9_17}"
    export TOPK="${TOPK:-1024}"
    export PI3_DATASET_CACHE_DIR="${PI3_DATASET_CACHE_DIR:-${SAVE_ROOT}/cache/pi3_dataset_index/vggt15_no_taskonomy_mapillary}"
    export PI3_DATA_ROOTS_CACHE_DIR="${PI3_DATA_ROOTS_CACHE_DIR:-${SAVE_ROOT}/cache/evc_runtime_scene_cache/pi3_sparse}"
    export PI3_STATIC_GRAPH="${PI3_STATIC_GRAPH:-0}"
    export PI3_TRAIN_NUM_WORKERS="${PI3_TRAIN_NUM_WORKERS:-4}"
    export PI3_TEST_NUM_WORKERS="${PI3_TEST_NUM_WORKERS:-4}"
    export GPU_IDS="${GPU_IDS:-0,1}"
    export NUM_PROCESSES="${NUM_PROCESSES:-$(count_gpu_ids "${GPU_IDS}")}"
    export NUM_MACHINES="${NUM_MACHINES:-1}"
    export MODEL_CKPT="${MODEL_CKPT:-${PI3_STAGE2_INIT_CKPT}}"
    export RUN_NAME="${RUN_NAME:-pi3_geoweave_sparse_smoke_$(date +%Y%m%d_%H%M%S)}"
    export EXP_NAME="${EXP_NAME:-release/geoweave_paper_handoff/${RUN_NAME}}"
    export WORK_DIR="${WORK_DIR:-${SAVE_ROOT}/record/${EXP_NAME}}"
    export PI3_TB_MIRROR_DIR="${PI3_TB_MIRROR_DIR:-${WORK_DIR}}"
    export SAVE_TO_AIDI="${SAVE_TO_AIDI:-0}"
    export SMOKE="${SMOKE:-1}"
    export SMOKE_ITERS="${SMOKE_ITERS:-2}"
    export SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
    export SMOKE_RES="${SMOKE_RES:-112}"
    export SMOKE_MAX_IMG_PER_GPU="${SMOKE_MAX_IMG_PER_GPU:-2}"
    export SMOKE_NUM_WORKERS="${SMOKE_NUM_WORKERS:-1}"
    export SMOKE_SEQ_NUM="${SMOKE_SEQ_NUM:-1}"
    export SMOKE_MAX_FRAMES_PER_SEQUENCE="${SMOKE_MAX_FRAMES_PER_SEQUENCE:-12}"
    export TRAIN_DYNAMIC_RES="${TRAIN_DYNAMIC_RES:-0}"
    export TRAIN_RES="${TRAIN_RES:-112}"
    export TEST_RES="${TEST_RES:-112}"
    export CORE4_VAL="${CORE4_VAL:-0}"
    export CORE4_FIRST_EVAL="${CORE4_FIRST_EVAL:-0}"
    export TEST_ITERS_PER_TEST="${TEST_ITERS_PER_TEST:-0}"
    export LOG_CKPT_INTERVAL="${LOG_CKPT_INTERVAL:-999}"
    export LOG_MAX_CHECKPOINTS="${LOG_MAX_CHECKPOINTS:-1}"
elif [ "${MODE_NAME}" = "paper" ]; then
    export STAGE="${STAGE:-sparse}"
    export TRAIN_CFG="${TRAIN_CFG:-train_pi3_lowres_indexer_sparse}"
    export DATA_CFG="${DATA_CFG:-meshx_pi3_vggt15_no_taskonomy_mapillary}"
    export INDEXER_LAYERS="${INDEXER_LAYERS:-9-17}"
    export INDEXER_LAYERS_TAG="${INDEXER_LAYERS_TAG:-9_17}"
    export TOPK="${TOPK:-1024}"
    export PI3_DATASET_CACHE_DIR="${PI3_DATASET_CACHE_DIR:-${SAVE_ROOT}/cache/pi3_dataset_index/vggt15_no_taskonomy_mapillary}"
    export PI3_DATA_ROOTS_CACHE_DIR="${PI3_DATA_ROOTS_CACHE_DIR:-${SAVE_ROOT}/cache/evc_runtime_scene_cache/pi3_sparse}"
    export PI3_STATIC_GRAPH="${PI3_STATIC_GRAPH:-0}"
    export PI3_TRAIN_NUM_WORKERS="${PI3_TRAIN_NUM_WORKERS:-4}"
    export PI3_TEST_NUM_WORKERS="${PI3_TEST_NUM_WORKERS:-4}"
    export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
    export NUM_MACHINES="${NUM_MACHINES:-2}"
    export NUM_PROCESSES="${NUM_PROCESSES:-16}"
    export MODEL_CKPT="${MODEL_CKPT:-${PI3_STAGE2_INIT_CKPT}}"
    export RUN_NAME="${RUN_NAME:-pi3_native_sparse_vggt15_warmup49_dynres_topk1024_l9_17_lw10_splitclip_static0_depthgt_handoff}"
    export EXP_NAME="${EXP_NAME:-pi3/native_sparse/${RUN_NAME}}"
    export WORK_DIR="${WORK_DIR:-${SAVE_ROOT}/record/${EXP_NAME}}"
    export PI3_TB_MIRROR_DIR="${PI3_TB_MIRROR_DIR:-${WORK_DIR}}"
    export SAVE_TO_AIDI="${SAVE_TO_AIDI:-1}"
    export CORE4_VAL="${CORE4_VAL:-1}"
    export CORE4_FIRST_EVAL="${CORE4_FIRST_EVAL:-1}"
    export TEST_ITERS_PER_TEST="${TEST_ITERS_PER_TEST:-0}"
    export TRAIN_NUM_EPOCH="${TRAIN_NUM_EPOCH:-80}"
    export TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS_PER_EPOCH:-800}"
    export LOG_CKPT_INTERVAL="${LOG_CKPT_INTERVAL:-1}"
    export LOG_MAX_CHECKPOINTS="${LOG_MAX_CHECKPOINTS:-10}"
    export TRAIN_IMAGE_NUM_RANGE="${TRAIN_IMAGE_NUM_RANGE:-[2,24]}"
    export TRAIN_MAX_IMG_PER_GPU="${TRAIN_MAX_IMG_PER_GPU:-24}"
    export TRAIN_DYNAMIC_RES="${TRAIN_DYNAMIC_RES:-1}"
    export TRAIN_DYNAMIC_ASPECT_RATIO_RANGE="${TRAIN_DYNAMIC_ASPECT_RATIO_RANGE:-[0.5,2.0]}"
    export TRAIN_DYNAMIC_PIXEL_COUNT_RANGE="${TRAIN_DYNAMIC_PIXEL_COUNT_RANGE:-[100000,255000]}"
    export TRAIN_DYNAMIC_PATCH_SIZE="${TRAIN_DYNAMIC_PATCH_SIZE:-14}"
    export TRAIN_DYNAMIC_NUM_RESOLUTION="${TRAIN_DYNAMIC_NUM_RESOLUTION:--1}"
    export TEST_RES="${TEST_RES:-518}"
elif [ "${MODE_NAME}" = "resume-final" ]; then
    export STAGE="${STAGE:-sparse}"
    export TRAIN_CFG="${TRAIN_CFG:-train_pi3_lowres_indexer_sparse}"
    export DATA_CFG="${DATA_CFG:-meshx_pi3_vggt15_no_taskonomy_mapillary}"
    export INDEXER_LAYERS="${INDEXER_LAYERS:-9-17}"
    export INDEXER_LAYERS_TAG="${INDEXER_LAYERS_TAG:-9_17}"
    export TOPK="${TOPK:-1024}"
    export PI3_DATASET_CACHE_DIR="${PI3_DATASET_CACHE_DIR:-${SAVE_ROOT}/cache/pi3_dataset_index/vggt15_no_taskonomy_mapillary}"
    export PI3_DATA_ROOTS_CACHE_DIR="${PI3_DATA_ROOTS_CACHE_DIR:-${SAVE_ROOT}/cache/evc_runtime_scene_cache/pi3_sparse}"
    export PI3_STATIC_GRAPH="${PI3_STATIC_GRAPH:-0}"
    export PI3_TRAIN_NUM_WORKERS="${PI3_TRAIN_NUM_WORKERS:-4}"
    export PI3_TEST_NUM_WORKERS="${PI3_TEST_NUM_WORKERS:-4}"
    export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
    export NUM_MACHINES="${NUM_MACHINES:-2}"
    export NUM_PROCESSES="${NUM_PROCESSES:-16}"
    export MODEL_CKPT="${MODEL_CKPT:-${PI3_FINAL_MODEL_BIN}}"
    export TRAIN_RESUME="${TRAIN_RESUME:-${PI3_FINAL_CKPT_DIR}}"
    export RUN_NAME="${RUN_NAME:-pi3_geoweave_resume_checkpoint79_handoff}"
    export EXP_NAME="${EXP_NAME:-pi3/native_sparse/${RUN_NAME}}"
    export WORK_DIR="${WORK_DIR:-${SAVE_ROOT}/record/${EXP_NAME}}"
    export PI3_TB_MIRROR_DIR="${PI3_TB_MIRROR_DIR:-${WORK_DIR}}"
    export SAVE_TO_AIDI="${SAVE_TO_AIDI:-1}"
    export CORE4_VAL="${CORE4_VAL:-1}"
    export CORE4_FIRST_EVAL="${CORE4_FIRST_EVAL:-1}"
    export TRAIN_NUM_EPOCH="${TRAIN_NUM_EPOCH:-160}"
    export TRAIN_ITERS_PER_EPOCH="${TRAIN_ITERS_PER_EPOCH:-800}"
    export LOG_CKPT_INTERVAL="${LOG_CKPT_INTERVAL:-1}"
    export LOG_MAX_CHECKPOINTS="${LOG_MAX_CHECKPOINTS:-1000}"
    export TRAIN_IMAGE_NUM_RANGE="${TRAIN_IMAGE_NUM_RANGE:-[2,24]}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ "${MODE_NAME}" = "warmup-smoke" ] || [ "${MODE_NAME}" = "warmup-paper" ]; then
    CMD=(bash "${REPO_ROOT}/aidi/scripts/pi3/submit_pi3_5090_warmup.sh")
else
    CMD=(bash "${REPO_ROOT}/aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh")
fi
if [ "${DRY_RUN}" = "1" ]; then
    printf 'cd %q\n' "${REPO_ROOT}"
    env | grep -E '^(STAGE|TRAIN_CFG|DATA_CFG|LOAD_VGGT|MODEL_CKPT|PI3_PRETRAIN_CKPT|PI3_BASE_CKPT|TRAIN_RESUME|RUN_NAME|EXP_NAME|WORK_DIR|PI3_WORK_DIR|PI3_OUTPUT_DIR|GPU_IDS|NUM_PROCESSES|NUM_MACHINES|SAVE_ROOT|SAVE_TO_AIDI|SMOKE|SMOKE_ITERS|SMOKE_RES|WARMUP_STEPS|WARMUP_NUM_EPOCH|WARMUP_ITERS_PER_EPOCH|WARMUP_LR|WARMUP_LOSS_MODE|WARMUP_ONLY_INDEXER_TRAIN|TRAIN_NUM_EPOCH|TRAIN_ITERS_PER_EPOCH|TRAIN_IMAGE_NUM_RANGE|TRAIN_MAX_IMG_PER_GPU|TRAIN_DYNAMIC_RES|TRAIN_MODEL_DTYPE|INDEXER_LAYERS|TOPK|SPARSE_LR|SPARSE_LOSS_WEIGHT|STREAMING_KL_LOSS|STREAMING_KL_AUTOGRAD|STREAMING_KL_SCORE_MODE|STREAMING_KL_FWD_MODE|STREAMING_KL_BWD_MODE|PI3_STATIC_GRAPH|CORE4_VAL|CORE4_FIRST_EVAL|TEST_ITERS_PER_TEST)=' | sort
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
