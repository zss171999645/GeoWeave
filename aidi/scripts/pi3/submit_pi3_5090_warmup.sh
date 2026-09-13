#!/usr/bin/env bash
set -euo pipefail

USER=${USER:-"feng01.zhou"}

# 5090 resources (single machine by default)
CLUSTER=${CLUSTER:-"5090"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_MACHINES=${NUM_MACHINES:-${NUM_NODES:-1}}
NUM_PROCESSES=${NUM_PROCESSES:-${WORLD_SIZE:-$(echo "${GPU_IDS}" | awk -F',' '{print NF}')}}
ACC_CONFIG=${ACC_CONFIG:-"configs/accelerate/ddp.yaml"}

# Warm-up experiment settings
# Align with PI3 paper training data by keeping overlap with our VGGT 17-set pool.
DATA_CFG=${DATA_CFG:-"meshx_pi3_paper_overlap"}
TRAIN_CFG=${TRAIN_CFG:-"train_pi3_lowres_indexer_warmup"}
INDEXER_LAYERS=${INDEXER_LAYERS:-"all"}
INDEXER_LAYERS_TAG=${INDEXER_LAYERS_TAG:-${INDEXER_LAYERS//[^0-9A-Za-z]/_}}
INDEXER_HEADS=${INDEXER_HEADS:-4}
INDEXER_INIT_FROM_ATTN=${INDEXER_INIT_FROM_ATTN:-1}
TOPK=${TOPK:-512}
WARMUP_STEPS=${WARMUP_STEPS:-40000}
WARMUP_LR=${WARMUP_LR:-}
WARMUP_LOSS_MODE=${WARMUP_LOSS_MODE:-"kl"}
TOPK_COVERAGE_K=${TOPK_COVERAGE_K:-${TOPK}}
TOPK_COVERAGE_CHUNK_SIZE=${TOPK_COVERAGE_CHUNK_SIZE:-128}
TOPK_COVERAGE_QUERY_CHUNK_SIZE=${TOPK_COVERAGE_QUERY_CHUNK_SIZE:-256}
TOPK_COVERAGE_QUERY_SAMPLE_SIZE=${TOPK_COVERAGE_QUERY_SAMPLE_SIZE:-0}
# 518px / 24-view warm-up cannot materialize dense [T,T] indexer KL scores
# on 32GB cards. Keep the chunked/streaming loss path enabled by default.
DEFAULT_STREAMING_KL_LOSS=1
STREAMING_KL_LOSS=${STREAMING_KL_LOSS:-${DEFAULT_STREAMING_KL_LOSS}}
STREAMING_KL_AUTOGRAD=${STREAMING_KL_AUTOGRAD:-${STREAMING_KL_LOSS}}
TOPK_SUPPORT_AUTOGRAD=${TOPK_SUPPORT_AUTOGRAD:-${STREAMING_KL_LOSS}}
# Pi3 all-layer dynamic warm-up has hit asynchronous CUDA illegal-memory
# failures in the Triton score helper during streaming-KL backward. Keep the
# mathematically equivalent PyTorch score path as the safe default; callers can
# still opt into Triton explicitly after a dedicated stress test.
STREAMING_KL_SCORE_MODE=${STREAMING_KL_SCORE_MODE:-"legacy"}
STREAMING_KL_FWD_MODE=${STREAMING_KL_FWD_MODE:-"score"}
STREAMING_KL_BWD_MODE=${STREAMING_KL_BWD_MODE:-"score"}
STREAMING_KL_FLASH_USE_TC=${STREAMING_KL_FLASH_USE_TC:-1}
STREAMING_KL_BWD_BLOCK_M=${STREAMING_KL_BWD_BLOCK_M:-64}
STREAMING_KL_BWD_BLOCK_N=${STREAMING_KL_BWD_BLOCK_N:-64}
STREAMING_KL_BWD_NUM_WARPS=${STREAMING_KL_BWD_NUM_WARPS:-4}
STREAMING_KL_BWD_NUM_STAGES=${STREAMING_KL_BWD_NUM_STAGES:-1}
WARMUP_NUM_EPOCH=${WARMUP_NUM_EPOCH:-1}
WARMUP_ITERS_PER_EPOCH=${WARMUP_ITERS_PER_EPOCH:-${WARMUP_STEPS}}
TRAIN_RES=${TRAIN_RES:-518}
TEST_RES=${TEST_RES:-${TRAIN_RES}}
TRAIN_DYNAMIC_RES=${TRAIN_DYNAMIC_RES:-0}
TRAIN_DYNAMIC_ASPECT_RATIO_RANGE=${TRAIN_DYNAMIC_ASPECT_RATIO_RANGE:-"[0.5,2.0]"}
TRAIN_DYNAMIC_PIXEL_COUNT_RANGE=${TRAIN_DYNAMIC_PIXEL_COUNT_RANGE:-"[100000,255000]"}
TRAIN_DYNAMIC_PATCH_SIZE=${TRAIN_DYNAMIC_PATCH_SIZE:-14}
TRAIN_DYNAMIC_NUM_RESOLUTION=${TRAIN_DYNAMIC_NUM_RESOLUTION:--1}
TEST_IMAGE_NUM_RANGE=${TEST_IMAGE_NUM_RANGE:-"[2,2]"}
TEST_ITERS_PER_TEST=${TEST_ITERS_PER_TEST:-0}
TRAIN_MODEL_DTYPE=${TRAIN_MODEL_DTYPE:-fp16}
TRAIN_MAX_IMG_PER_GPU=${TRAIN_MAX_IMG_PER_GPU:-2}
TRAIN_CLIP_LOSS=${TRAIN_CLIP_LOSS:-1000000}
PI3_TRAIN_NUM_WORKERS=${PI3_TRAIN_NUM_WORKERS:-1}
PI3_TEST_NUM_WORKERS=${PI3_TEST_NUM_WORKERS:-1}
PI3_FIND_UNUSED_PARAMETERS=${PI3_FIND_UNUSED_PARAMETERS:-0}
# Warm-up trains the same indexer modules every iteration. Keep DDP static graph
# enabled by default so decoder activation checkpointing does not mark the same
# indexer parameter ready twice during streaming-KL backward.
PI3_STATIC_GRAPH=${PI3_STATIC_GRAPH:-1}
PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT=${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT:-0}
WARMUP_ONLY_INDEXER_TRAIN=${WARMUP_ONLY_INDEXER_TRAIN:-True}
INDEXER_HEAD_CHUNK_SIZE=${INDEXER_HEAD_CHUNK_SIZE:-1}
INDEXER_SCORE_HEAD_CHUNK_SIZE=${INDEXER_SCORE_HEAD_CHUNK_SIZE:-1}
INDEXER_SCORE_KEY_CHUNK_SIZE=${INDEXER_SCORE_KEY_CHUNK_SIZE:-1024}
SEQ_NUM=${SEQ_NUM:--1}
RESUME=${RESUME:-0}
PI3_PRETRAIN_CKPT=${PI3_PRETRAIN_CKPT:-"/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/pretrained/pi3/yyfz233_Pi3_model.safetensors"}
MODEL_CKPT=${MODEL_CKPT:-"${PI3_PRETRAIN_CKPT}"}
LOAD_VGGT=${LOAD_VGGT:-0}

if [ "${RESUME}" = "1" ] && [ -z "${MODEL_CKPT}" ]; then
  echo "[ERROR] RESUME=1 requires MODEL_CKPT."
  exit 1
fi

is_truthy() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_numeric_lt() {
  awk -v lhs="$1" -v rhs="$2" 'BEGIN { exit !(lhs + 0 < rhs + 0) }'
}

ALLOW_LOW_WARMUP_CLIP_LOSS=${ALLOW_LOW_WARMUP_CLIP_LOSS:-0}
if is_truthy "${WARMUP_ONLY_INDEXER_TRAIN}" \
  && [ "$(echo "${WARMUP_LOSS_MODE}" | tr '[:upper:]' '[:lower:]')" = "kl" ] \
  && { [ "$(echo "${INDEXER_LAYERS}" | tr '[:upper:]' '[:lower:]')" = "all" ] || [ "${INDEXER_LAYERS}" = "*" ]; } \
  && is_numeric_lt "${TRAIN_CLIP_LOSS}" 100000 \
  && ! is_truthy "${ALLOW_LOW_WARMUP_CLIP_LOSS}"; then
  echo "[ERROR] TRAIN_CLIP_LOSS=${TRAIN_CLIP_LOSS} is too low for all-layer KL indexer warm-up."
  echo "[ERROR] Pi3 trainer zeros the whole loss when loss > train.clip_loss; this makes indexer_loss and grad_norm log as 0."
  echo "[ERROR] Use TRAIN_CLIP_LOSS=1000000, or set ALLOW_LOW_WARMUP_CLIP_LOSS=1 only for a deliberate clipping test."
  exit 1
fi

if is_truthy "${STREAMING_KL_AUTOGRAD}" \
  && [ "${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT}" -le 0 ] \
  && ! is_truthy "${PI3_STATIC_GRAPH}"; then
  echo "[ERROR] streaming KL autograd with decoder activation checkpointing requires PI3_STATIC_GRAPH=1."
  echo "[ERROR] Otherwise DDP can mark indexer parameters ready twice. Set PI3_STATIC_GRAPH=1 or disable decoder checkpointing."
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M)
RUN_NAME=${RUN_NAME:-"pi3_warmup_5090_${TIMESTAMP}_w${WARMUP_STEPS}_layers${INDEXER_LAYERS_TAG}_h${INDEXER_HEADS}"}
SAVE_ROOT=${SAVE_ROOT:-"/horizon-bucket/saturn_v_dev/01_users/${USER}/projects/meshx/baseline"}
LOG_DIR=${LOG_DIR:-"${SAVE_ROOT}/logs"}
LOG_PATH=${LOG_PATH:-"${LOG_DIR}/pi3_warmup_5090_${TIMESTAMP}.log"}
PI3_WORK_DIR=${PI3_WORK_DIR:-"${SAVE_ROOT}/trained_model/pi3/official"}
PI3_OUTPUT_DIR=${PI3_OUTPUT_DIR:-"${PI3_WORK_DIR}/outputs/${RUN_NAME}"}
SAVE_TO_AIDI=${SAVE_TO_AIDI:-1}

# VGGT-like data loading knobs:
# - disable __tmp_{intri,extri}.pkl probing by default
# - enable dataset index cache
# - keep wait disabled by default so lazy indexing can stream instead of
#   synchronizing on a full cache build before training
# - store cache under a persistent path (instead of volatile /tmp)
PI3_DATASET_CACHE_DIR=${PI3_DATASET_CACHE_DIR:-"${SAVE_ROOT}/cache/pi3_dataset_index"}
PI3_USE_INDEX_CACHE=${PI3_USE_INDEX_CACHE:-1}
PI3_REBUILD_INDEX_CACHE=${PI3_REBUILD_INDEX_CACHE:-0}
PI3_USE_CAMERA_PKL_CACHE=${PI3_USE_CAMERA_PKL_CACHE:-0}
PI3_INDEX_CACHE_WAIT_SEC=${PI3_INDEX_CACHE_WAIT_SEC:-0}
PI3_INDEXING_WORKERS=${PI3_INDEXING_WORKERS:-64}
PI3_LAZY_SEQUENCE_INDEX=${PI3_LAZY_SEQUENCE_INDEX:-1}

USER_EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-""}
WARMUP_OVERRIDES=(
  "hydra/hydra_logging=default"
  "hydra/job_logging=custom"
  "model.indexer_cfg.enabled=True"
  "model.indexer_cfg.indexer_layers=${INDEXER_LAYERS}"
  "model.indexer_cfg.n_heads=${INDEXER_HEADS}"
  "model.indexer_cfg.init_from_attn=${INDEXER_INIT_FROM_ATTN}"
  "model.indexer_cfg.topk=${TOPK}"
  "model.indexer_cfg.enable_sparse=False"
  "model.indexer_cfg.warmup_steps=${WARMUP_STEPS}"
  "model.indexer_cfg.sparse_start_step=${WARMUP_STEPS}"
  "model.indexer_cfg.warmup_only_indexer_loss=True"
  "model.indexer_cfg.warmup_only_indexer_train=${WARMUP_ONLY_INDEXER_TRAIN}"
  "model.indexer_cfg.score_dtype=float16"
  "model.indexer_cfg.head_chunk_size=${INDEXER_HEAD_CHUNK_SIZE}"
  "model.indexer_cfg.score_head_chunk_size=${INDEXER_SCORE_HEAD_CHUNK_SIZE}"
  "model.indexer_cfg.score_key_chunk_size=${INDEXER_SCORE_KEY_CHUNK_SIZE}"
  "model.indexer_cfg.warmup_no_grad_attn=True"
  "model.indexer_cfg.detach_input=True"
  "model.indexer_cfg.use_dense_flash_attn_warmup_kernel=True"
  "model.indexer_cfg.streaming_kl_loss=${STREAMING_KL_LOSS}"
  "model.indexer_cfg.streaming_kl_autograd=${STREAMING_KL_AUTOGRAD}"
  "model.indexer_cfg.warmup_indexer_loss_mode=${WARMUP_LOSS_MODE}"
  "model.indexer_cfg.warmup_topk_coverage_k=${TOPK_COVERAGE_K}"
  "model.indexer_cfg.warmup_topk_coverage_chunk_size=${TOPK_COVERAGE_CHUNK_SIZE}"
  "model.indexer_cfg.warmup_topk_coverage_query_chunk_size=${TOPK_COVERAGE_QUERY_CHUNK_SIZE}"
  "model.indexer_cfg.warmup_topk_coverage_query_sample_size=${TOPK_COVERAGE_QUERY_SAMPLE_SIZE}"
  "model.num_dec_blk_not_to_checkpoint=${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT}"
  "work_dir=${PI3_WORK_DIR}"
  "train.model_dtype=${TRAIN_MODEL_DTYPE}"
  "train.num_workers=${PI3_TRAIN_NUM_WORKERS}"
  "test.num_workers=${PI3_TEST_NUM_WORKERS}"
  "train.find_unused_parameters=${PI3_FIND_UNUSED_PARAMETERS}"
  "train.static_graph=${PI3_STATIC_GRAPH}"
  "train.max_img_per_gpu=${TRAIN_MAX_IMG_PER_GPU}"
  "train.clip_loss=${TRAIN_CLIP_LOSS}"
  "train.num_epoch=${WARMUP_NUM_EPOCH}"
  "train.iters_per_epoch=${WARMUP_ITERS_PER_EPOCH}"
  "test.iters_per_test=${TEST_ITERS_PER_TEST}"
  "log.output_dir=${PI3_OUTPUT_DIR}"
  "log.ckpt_dir=${PI3_OUTPUT_DIR}/ckpts"
  "hydra.run.dir=${PI3_OUTPUT_DIR}"
  "+test.image_num_range=${TEST_IMAGE_NUM_RANGE}"
)

if is_truthy "${TRAIN_DYNAMIC_RES}"; then
  WARMUP_OVERRIDES+=(
    "++train.random_reslution=True"
    "++train.aspect_ratio_range=${TRAIN_DYNAMIC_ASPECT_RATIO_RANGE}"
    "++train.pixel_count_range=${TRAIN_DYNAMIC_PIXEL_COUNT_RANGE}"
    "++train.patch_size=${TRAIN_DYNAMIC_PATCH_SIZE}"
    "++train.num_resolution=${TRAIN_DYNAMIC_NUM_RESOLUTION}"
  )
else
  WARMUP_OVERRIDES+=("train.resolution=[[${TRAIN_RES},${TRAIN_RES}]]")
fi

if [ -n "${WARMUP_LR}" ]; then
  WARMUP_OVERRIDES+=("model.indexer_cfg.warmup_lr=${WARMUP_LR}")
fi

if [ "${DATA_CFG}" = "meshx_pi3_paper_overlap" ]; then
  WARMUP_DATASET_KEYS=(
    "BlendedMVS"
    "HyperSimTrain"
    "TarTanAir"
    "MegaDepth"
    "ScanNetPP"
    "Taskonomy"
    "WildRGBD"
    "CO3Dv2"
  )
else
  WARMUP_DATASET_KEYS=("TarTanAir")
fi

TEST_RES_OVERRIDES=()
dataset_key=
for dataset_key in "${WARMUP_DATASET_KEYS[@]}"; do
  TEST_RES_OVERRIDES+=("test_dataset.${dataset_key}.resolution=[[${TEST_RES},${TEST_RES}]]")
done

DATA_LOADING_OVERRIDES=()
for dataset_key in "${WARMUP_DATASET_KEYS[@]}"; do
  DATA_LOADING_OVERRIDES+=("++train_dataset.${dataset_key}.use_camera_pkl_cache=${PI3_USE_CAMERA_PKL_CACHE}")
  DATA_LOADING_OVERRIDES+=("++test_dataset.${dataset_key}.use_camera_pkl_cache=${PI3_USE_CAMERA_PKL_CACHE}")
  DATA_LOADING_OVERRIDES+=("++train_dataset.${dataset_key}.use_index_cache=${PI3_USE_INDEX_CACHE}")
  DATA_LOADING_OVERRIDES+=("++test_dataset.${dataset_key}.use_index_cache=${PI3_USE_INDEX_CACHE}")
  DATA_LOADING_OVERRIDES+=("++train_dataset.${dataset_key}.rebuild_index_cache=${PI3_REBUILD_INDEX_CACHE}")
  DATA_LOADING_OVERRIDES+=("++test_dataset.${dataset_key}.rebuild_index_cache=${PI3_REBUILD_INDEX_CACHE}")
  DATA_LOADING_OVERRIDES+=("++train_dataset.${dataset_key}.index_cache_dir=${PI3_DATASET_CACHE_DIR}")
  DATA_LOADING_OVERRIDES+=("++test_dataset.${dataset_key}.index_cache_dir=${PI3_DATASET_CACHE_DIR}")
  DATA_LOADING_OVERRIDES+=("++train_dataset.${dataset_key}.index_cache_wait_sec=${PI3_INDEX_CACHE_WAIT_SEC}")
  DATA_LOADING_OVERRIDES+=("++test_dataset.${dataset_key}.index_cache_wait_sec=${PI3_INDEX_CACHE_WAIT_SEC}")
done

# Bash 4.2 treats expanding an empty array under set -u as unbound.
# Keep a filtered empty placeholder so the default SEQ_NUM=-1 path is safe.
SEQ_NUM_OVERRIDES=("")
if [ "${SEQ_NUM}" -gt 0 ]; then
  for dataset_key in "${WARMUP_DATASET_KEYS[@]}"; do
    SEQ_NUM_OVERRIDES+=("+train_dataset.${dataset_key}.seq_num=${SEQ_NUM}")
    SEQ_NUM_OVERRIDES+=("+test_dataset.${dataset_key}.seq_num=${SEQ_NUM}")
  done
fi

join_by() {
  local IFS="$1"
  shift
  echo "$*"
}

build_overrides() {
  local items=("$@")
  local filtered=()
  local item
  for item in "${items[@]}"; do
    if [ -n "${item}" ]; then
      filtered+=("${item}")
    fi
  done
  join_by ';' "${filtered[@]}"
}

EXTRA_OVERRIDES_ALL=$(build_overrides "${WARMUP_OVERRIDES[@]}" "${TEST_RES_OVERRIDES[@]}" "${DATA_LOADING_OVERRIDES[@]}" "${SEQ_NUM_OVERRIDES[@]}" "${USER_EXTRA_OVERRIDES}")

mkdir -p "${LOG_DIR}" "${PI3_OUTPUT_DIR}" "${PI3_WORK_DIR}/outputs/${RUN_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export VGGT_DENSE_FLASH_ATTN_WARMUP=${VGGT_DENSE_FLASH_ATTN_WARMUP:-1}
export VGGT_TOPK_SUPPORT_AUTOGRAD=${VGGT_TOPK_SUPPORT_AUTOGRAD:-${TOPK_SUPPORT_AUTOGRAD}}
export VGGT_STREAMING_KL_SCORE_MODE=${VGGT_STREAMING_KL_SCORE_MODE:-${STREAMING_KL_SCORE_MODE}}
export VGGT_STREAMING_KL_FWD_MODE=${VGGT_STREAMING_KL_FWD_MODE:-${STREAMING_KL_FWD_MODE}}
export VGGT_STREAMING_KL_BWD_MODE=${VGGT_STREAMING_KL_BWD_MODE:-${STREAMING_KL_BWD_MODE}}
export VGGT_STREAMING_KL_FLASH_USE_TC=${VGGT_STREAMING_KL_FLASH_USE_TC:-${STREAMING_KL_FLASH_USE_TC}}
export VGGT_STREAMING_KL_BWD_BLOCK_M=${VGGT_STREAMING_KL_BWD_BLOCK_M:-${STREAMING_KL_BWD_BLOCK_M}}
export VGGT_STREAMING_KL_BWD_BLOCK_N=${VGGT_STREAMING_KL_BWD_BLOCK_N:-${STREAMING_KL_BWD_BLOCK_N}}
export VGGT_STREAMING_KL_BWD_NUM_WARPS=${VGGT_STREAMING_KL_BWD_NUM_WARPS:-${STREAMING_KL_BWD_NUM_WARPS}}
export VGGT_STREAMING_KL_BWD_NUM_STAGES=${VGGT_STREAMING_KL_BWD_NUM_STAGES:-${STREAMING_KL_BWD_NUM_STAGES}}
export PI3_DATASET_CACHE_DIR
export PI3_INDEX_CACHE_WAIT_SEC
export PI3_INDEXING_WORKERS
export PI3_LAZY_SEQUENCE_INDEX

export STAGE=${STAGE:-warmup}
export TRAIN_CFG
export DATA_CFG
export RUN_NAME
export SAVE_TO_AIDI
export INDEXER_LAYERS
export ACC_CONFIG
export NUM_MACHINES
export NUM_PROCESSES
export LOAD_VGGT
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES_ALL}"

if [ -n "${MODEL_CKPT}" ]; then
  export MODEL_CKPT
fi

echo "[INFO] CLUSTER=${CLUSTER} GPU_IDS=${GPU_IDS} NUM_PROCESSES=${NUM_PROCESSES}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] TOPK=${TOPK} WARMUP_LOSS_MODE=${WARMUP_LOSS_MODE}"
echo "[INFO] INDEXER_INIT_FROM_ATTN=${INDEXER_INIT_FROM_ATTN}"
echo "[INFO] TRAIN_DYNAMIC_RES=${TRAIN_DYNAMIC_RES} TRAIN_RES=${TRAIN_RES}"
echo "[INFO] STREAMING_KL_LOSS=${STREAMING_KL_LOSS} STREAMING_KL_AUTOGRAD=${STREAMING_KL_AUTOGRAD} VGGT_TOPK_SUPPORT_AUTOGRAD=${VGGT_TOPK_SUPPORT_AUTOGRAD}"
echo "[INFO] STREAMING_KL_MODES score:${VGGT_STREAMING_KL_SCORE_MODE} fwd:${VGGT_STREAMING_KL_FWD_MODE} bwd:${VGGT_STREAMING_KL_BWD_MODE}"
echo "[INFO] TOPK_COVERAGE_K=${TOPK_COVERAGE_K} TOPK_COVERAGE_QUERY_SAMPLE_SIZE=${TOPK_COVERAGE_QUERY_SAMPLE_SIZE}"
echo "[INFO] WARMUP_STEPS=${WARMUP_STEPS} WARMUP_NUM_EPOCH=${WARMUP_NUM_EPOCH} WARMUP_ITERS_PER_EPOCH=${WARMUP_ITERS_PER_EPOCH}"
echo "[INFO] WARMUP_LR=${WARMUP_LR:-<config-default>}"
echo "[INFO] TRAIN_RES=${TRAIN_RES} TEST_RES=${TEST_RES}"
echo "[INFO] TRAIN_MODEL_DTYPE=${TRAIN_MODEL_DTYPE} TRAIN_MAX_IMG_PER_GPU=${TRAIN_MAX_IMG_PER_GPU}"
echo "[INFO] TRAIN_CLIP_LOSS=${TRAIN_CLIP_LOSS}"
echo "[INFO] PI3_TRAIN_NUM_WORKERS=${PI3_TRAIN_NUM_WORKERS} PI3_TEST_NUM_WORKERS=${PI3_TEST_NUM_WORKERS}"
echo "[INFO] PI3_FIND_UNUSED_PARAMETERS=${PI3_FIND_UNUSED_PARAMETERS} PI3_STATIC_GRAPH=${PI3_STATIC_GRAPH}"
echo "[INFO] PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT=${PI3_NUM_DEC_BLK_NOT_TO_CHECKPOINT}"
echo "[INFO] INDEXER_CHUNKS=head:${INDEXER_HEAD_CHUNK_SIZE} score_head:${INDEXER_SCORE_HEAD_CHUNK_SIZE} score_key:${INDEXER_SCORE_KEY_CHUNK_SIZE}"
echo "[INFO] TEST_IMAGE_NUM_RANGE=${TEST_IMAGE_NUM_RANGE}"
echo "[INFO] TEST_ITERS_PER_TEST=${TEST_ITERS_PER_TEST}"
echo "[INFO] SEQ_NUM=${SEQ_NUM}"
echo "[INFO] PI3_DATASET_CACHE_DIR=${PI3_DATASET_CACHE_DIR}"
echo "[INFO] PI3_USE_INDEX_CACHE=${PI3_USE_INDEX_CACHE} PI3_REBUILD_INDEX_CACHE=${PI3_REBUILD_INDEX_CACHE}"
echo "[INFO] PI3_USE_CAMERA_PKL_CACHE=${PI3_USE_CAMERA_PKL_CACHE} PI3_INDEX_CACHE_WAIT_SEC=${PI3_INDEX_CACHE_WAIT_SEC} PI3_INDEXING_WORKERS=${PI3_INDEXING_WORKERS}"
echo "[INFO] PI3_LAZY_SEQUENCE_INDEX=${PI3_LAZY_SEQUENCE_INDEX}"
echo "[INFO] WARMUP_DATASETS=${#WARMUP_DATASET_KEYS[@]} (${WARMUP_DATASET_KEYS[*]})"
echo "[INFO] LOG_PATH=${LOG_PATH}"
echo "[INFO] PI3_WORK_DIR=${PI3_WORK_DIR}"
echo "[INFO] PI3_OUTPUT_DIR=${PI3_OUTPUT_DIR}"
echo "[INFO] SAVE_TO_AIDI=${SAVE_TO_AIDI}"
echo "[INFO] EXTRA_OVERRIDES=${EXTRA_OVERRIDES}"

bash aidi/scripts/pi3/train_pi3_official.sh 2>&1 | tee "${LOG_PATH}"
