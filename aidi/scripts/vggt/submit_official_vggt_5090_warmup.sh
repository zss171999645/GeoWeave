#!/usr/bin/env bash
set -euo pipefail

USER=${USER:-"feng01.zhou"}
MODE=${MODE:-"remote"}

# Cluster and config
# Align with previous A800 warmup configuration.
# Only queue and resource scale differ on 5090.
CONFIG=${CONFIG:-"vggt/vggt_official_finetune_a800_paper"}
CLUSTER=${CLUSTER:-"project-5090-4dlabel-perception-acloud-langfang"}
NUM_NODES=${NUM_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
SAVE_ROOT=${SAVE_ROOT:-"/horizon-bucket/saturn_v_dev/01_users/${USER}/projects/meshx/baseline"}
SAVE_TO_AIDI=${SAVE_TO_AIDI:-1}
RUN_TEST=${RUN_TEST:-0}
RESUME=${RESUME:-0}
RESUME_EXP_NAME=${RESUME_EXP_NAME:-""}
SUBMIT_SLEEP=${SUBMIT_SLEEP:-0}
AIDI_SLEEP_HOURS=${AIDI_SLEEP_HOURS:-300}
OVERRIDES_ARGFILE_ENABLED=${OVERRIDES_ARGFILE_ENABLED:-1}
SUBMIT_TRIAL=${SUBMIT_TRIAL:-1}


# Warmup schedule (keeps indexer in warmup stage; other training params from YAML)
WARMUP_STEPS=${WARMUP_STEPS:-40000}
INDEXER_HEADS=${INDEXER_HEADS:-4}
INDEXER_LAYERS=${INDEXER_LAYERS:-"9-19"}
HEAD_CHUNK_SIZE=${HEAD_CHUNK_SIZE:-1}
SCORE_HEAD_CHUNK_SIZE=${SCORE_HEAD_CHUNK_SIZE:-1}
SCORE_KEY_CHUNK_SIZE=${SCORE_KEY_CHUNK_SIZE:-1024}
LAYERWISE_BACKWARD=${LAYERWISE_BACKWARD:-True}
# 5090 warmup memory-saving / kernel toggles.
# Dev014 DSAAttention warmup benchmark (16384 tokens, 10 iters, same GPU) showed:
# - legacy:  forward=358.46ms, backward=72.67ms, total=431.13ms
# - score-only triton: forward=344.60ms, backward=55.98ms, total=400.58ms
# - full flash fwd+bwd: forward=331.58ms, backward=21.50ms, total=353.08ms
# Recent full warm-up single-step checks found noticeable gradient drift under flash
# backward; therefore default to score-backward for stability while keeping flash
# forward + triton score path. You can still override with STREAMING_KL_BWD_MODE=flash.
STREAMING_KL_SCORE_MODE=${STREAMING_KL_SCORE_MODE:-"triton"}
INDEXER_SCORE_USE_TC=${INDEXER_SCORE_USE_TC:-1}
INDEXER_SCORE_BLOCK_M=${INDEXER_SCORE_BLOCK_M:-64}
INDEXER_SCORE_BLOCK_N=${INDEXER_SCORE_BLOCK_N:-256}
INDEXER_SCORE_NUM_WARPS=${INDEXER_SCORE_NUM_WARPS:-8}
INDEXER_SCORE_NUM_STAGES=${INDEXER_SCORE_NUM_STAGES:-2}
STREAMING_KL_FWD_MODE=${STREAMING_KL_FWD_MODE:-"flash"}
STREAMING_KL_BWD_MODE=${STREAMING_KL_BWD_MODE:-"score"}
STREAMING_KL_FLASH_USE_TC=${STREAMING_KL_FLASH_USE_TC:-${INDEXER_SCORE_USE_TC}}
STREAMING_KL_BWD_BLOCK_M=${STREAMING_KL_BWD_BLOCK_M:-64}
STREAMING_KL_BWD_BLOCK_N=${STREAMING_KL_BWD_BLOCK_N:-64}
STREAMING_KL_BWD_NUM_WARPS=${STREAMING_KL_BWD_NUM_WARPS:-4}
STREAMING_KL_BWD_NUM_STAGES=${STREAMING_KL_BWD_NUM_STAGES:-1}
# Optional runtime launch-overhead reduction. Dev014 synthetic warmup benchmark
# shows `torch.compile(mode=reduce-overhead)` can further cut exact-KL total step
# time by ~2.4% on top of the current flash baseline, while keeping math fixed.
RUNNER_USE_TORCH_COMPILE=${RUNNER_USE_TORCH_COMPILE:-0}
RUNNER_TORCH_COMPILE_MODE=${RUNNER_TORCH_COMPILE_MODE:-"reduce-overhead"}
WARMUP_MASK_SPARSE_METASETS=${WARMUP_MASK_SPARSE_METASETS:-auto}
WARMUP_KERNEL_RUN_ENVS=${WARMUP_KERNEL_RUN_ENVS:-"VGGT_DENSE_FLASH_ATTN_WARMUP=1;VGGT_STREAMING_KL_SCORE_MODE=${STREAMING_KL_SCORE_MODE};VGGT_STREAMING_KL_FWD_MODE=${STREAMING_KL_FWD_MODE};VGGT_STREAMING_KL_BWD_MODE=${STREAMING_KL_BWD_MODE};VGGT_STREAMING_KL_FLASH_USE_TC=${STREAMING_KL_FLASH_USE_TC};VGGT_INDEXER_SCORE_USE_TC=${INDEXER_SCORE_USE_TC};VGGT_INDEXER_SCORE_BLOCK_M=${INDEXER_SCORE_BLOCK_M};VGGT_INDEXER_SCORE_BLOCK_N=${INDEXER_SCORE_BLOCK_N};VGGT_INDEXER_SCORE_NUM_WARPS=${INDEXER_SCORE_NUM_WARPS};VGGT_INDEXER_SCORE_NUM_STAGES=${INDEXER_SCORE_NUM_STAGES};VGGT_STREAMING_KL_BWD_BLOCK_M=${STREAMING_KL_BWD_BLOCK_M};VGGT_STREAMING_KL_BWD_BLOCK_N=${STREAMING_KL_BWD_BLOCK_N};VGGT_STREAMING_KL_BWD_NUM_WARPS=${STREAMING_KL_BWD_NUM_WARPS};VGGT_STREAMING_KL_BWD_NUM_STAGES=${STREAMING_KL_BWD_NUM_STAGES}"}
if [ -n "${RUN_ENV_VARS:-}" ]; then
    RUN_ENV_VARS="${WARMUP_KERNEL_RUN_ENVS};${RUN_ENV_VARS}"
else
    RUN_ENV_VARS="${WARMUP_KERNEL_RUN_ENVS}"
fi

# Naming
TIMESTAMP=$(date +%Y%m%d_%H%M)
JOB_NAME=${JOB_NAME:-"vggt_official_5090_warmup_${TIMESTAMP}_w${WARMUP_STEPS}_indexerhead${INDEXER_HEADS}"}
EXP_NAME=${EXP_NAME:-"vggt/official/finetune_5090_warmup_${TIMESTAMP}_w${WARMUP_STEPS}"}
if [ "${RESUME}" = "1" ] && [ -n "${RESUME_EXP_NAME}" ]; then
    EXP_NAME="${RESUME_EXP_NAME}"
    RESUME_TAG="${EXP_NAME##*/}"
    JOB_NAME="vggt_official_5090_warmup_resume_${RESUME_TAG}"
fi
LOG_DIR=${LOG_DIR:-"${SAVE_ROOT}/logs"}
LOG_PATH="${LOG_DIR}/warmup_5090_${TIMESTAMP}.log"

# Warmup-only overrides (everything else from YAML)
WARMUP_STAGE_OVERRIDES=(
    "runner_cfg.eval_ep=0"
    "runner_cfg.extra_eval_every=0"
    "runner_cfg.save_ep=1"
    "runner_cfg.save_latest_ep=1"
    "runner_cfg.cuda_mem_trace_cfg.enabled=False"
    "runner_cfg.cuda_mem_trace_cfg.record_history=False"
    "runner_cfg.cuda_mem_trace_cfg.log_json=False"
    "runner_cfg.use_torch_compile=$( [ \"${RUNNER_USE_TORCH_COMPILE}\" = \"1\" ] && echo True || echo False )"
    "runner_cfg.torch_compile_mode=${RUNNER_TORCH_COMPILE_MODE}"
    "model_cfg.vggt_cfg.enable_camera=False"
    "model_cfg.vggt_cfg.enable_depth=False"
    "model_cfg.vggt_cfg.enable_point=False"
    "model_cfg.vggt_cfg.memory_cfg.patch_embed_chunk_size=8"
    "model_cfg.vggt_cfg.memory_cfg.output_list_cfg.keep_layers=[]"
    "model_cfg.vggt_cfg.memory_cfg.output_list_cfg.keep_last=False"
    "model_cfg.vggt_cfg.indexer_cfg.enable_sparse=False"
    # Align warm-up with the sparse stage target layers to avoid paying dense KL on all 24 global blocks.
    "model_cfg.vggt_cfg.indexer_cfg.indexer_layers=${INDEXER_LAYERS}"
    "model_cfg.vggt_cfg.indexer_cfg.n_heads=${INDEXER_HEADS}"
    "model_cfg.vggt_cfg.indexer_cfg.head_chunk_size=${HEAD_CHUNK_SIZE}"
    "model_cfg.vggt_cfg.indexer_cfg.score_dtype=float16"
    "model_cfg.vggt_cfg.indexer_cfg.score_head_chunk_size=${SCORE_HEAD_CHUNK_SIZE}"
    "model_cfg.vggt_cfg.indexer_cfg.score_key_chunk_size=${SCORE_KEY_CHUNK_SIZE}"
    "model_cfg.vggt_cfg.indexer_cfg.streaming_kl_loss=True"
    "model_cfg.vggt_cfg.indexer_cfg.streaming_kl_autograd=True"
    "model_cfg.vggt_cfg.indexer_cfg.warmup_no_grad_attn=True"
    "model_cfg.vggt_cfg.indexer_cfg.detach_input=True"
    "model_cfg.vggt_cfg.indexer_cfg.layerwise_backward=${LAYERWISE_BACKWARD}"
    "model_cfg.vggt_cfg.indexer_cfg.use_dense_flash_attn_warmup_kernel=True"
    "model_cfg.vggt_cfg.indexer_cfg.warmup_only_indexer_loss=True"
    # Keep warmup resolution aligned to prior A800 setup.
    "dataloader_cfg.dataset_cfg.proc_max_size=518"
    "val_dataloader_cfg.dataset_cfg.proc_max_size=518"
)

mask_sparse_metasets=false
if [ "${WARMUP_MASK_SPARSE_METASETS}" = "1" ]; then
    mask_sparse_metasets=true
elif [ "${WARMUP_MASK_SPARSE_METASETS}" = "auto" ]; then
    case "${CONFIG}" in
        *paper*|*a800*)
            mask_sparse_metasets=true
            ;;
    esac
fi

if [ "${mask_sparse_metasets}" = "true" ]; then
    WARMUP_STAGE_OVERRIDES+=(
    # Remove sparse-only datasets from warmup sampling.
    "dataloader_cfg.dataset_cfg.metaset_cfgs.5.prob=0.0"
    "dataloader_cfg.dataset_cfg.metaset_cfgs.10.prob=0.0"
    "dataloader_cfg.dataset_cfg.metaset_cfgs.11.prob=0.0"
    "dataloader_cfg.dataset_cfg.metaset_cfgs.13.prob=0.0"
    "val_dataloader_cfg.dataset_cfg.metaset_cfgs.5.prob=0.0"
    "val_dataloader_cfg.dataset_cfg.metaset_cfgs.10.prob=0.0"
    "val_dataloader_cfg.dataset_cfg.metaset_cfgs.11.prob=0.0"
    "val_dataloader_cfg.dataset_cfg.metaset_cfgs.13.prob=0.0"
    )
fi

# Extra overrides appended at the end (keep empty unless needed)
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-""}

SAVE_OVERRIDES=()
if [ -n "${SAVE_ROOT}" ]; then
    SAVE_OVERRIDES+=(
        "runner_cfg.trained_model=${SAVE_ROOT}/trained_model/${EXP_NAME}"
        "runner_cfg.recorder_cfg.record_dir=${SAVE_ROOT}/record/${EXP_NAME}"
        "runner_cfg.visualizer_cfg.result_dir=${SAVE_ROOT}/result"
    )
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

OVERRIDES=$(build_overrides \
    "${WARMUP_STAGE_OVERRIDES[@]}" \
    "${SAVE_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES}")

mkdir -p "${LOG_DIR}"
SAVE_ROOT="${SAVE_ROOT}" AIDI_SLEEP_HOURS="${AIDI_SLEEP_HOURS}" OVERRIDES_ARGFILE_ENABLED="${OVERRIDES_ARGFILE_ENABLED}" \
SUBMIT_TRIAL="${SUBMIT_TRIAL}" \
MODE="${MODE}" CLUSTER="${CLUSTER}" NUM_NODES="${NUM_NODES}" GPUS_PER_NODE="${GPUS_PER_NODE}" GPU_IDS="${GPU_IDS}" \
RUN_TEST="${RUN_TEST}" CONFIG="${CONFIG}" \
JOB_NAME="${JOB_NAME}" EXP_NAME="${EXP_NAME}" \
SAVE_TO_AIDI="${SAVE_TO_AIDI}" \
RESUME="${RESUME}" SUBMIT_SLEEP="${SUBMIT_SLEEP}" PRETRAINED_MODEL="" \
RUN_ENV_VARS="${RUN_ENV_VARS}" \
EXTRA_OVERRIDES="${OVERRIDES}" \
bash aidi/scripts/vggt/train_official_vggt.sh 2>&1 | tee "${LOG_PATH}"
