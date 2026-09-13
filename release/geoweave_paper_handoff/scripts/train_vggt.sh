#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash release/geoweave_paper_handoff/scripts/train_vggt.sh --smoke [--dry-run]
  bash release/geoweave_paper_handoff/scripts/train_vggt.sh --paper [--dry-run]
  bash release/geoweave_paper_handoff/scripts/train_vggt.sh --warmup-smoke [--dry-run]
  bash release/geoweave_paper_handoff/scripts/train_vggt.sh --warmup-paper [--dry-run]

Modes:
  --smoke   2-GPU dev smoke. Uses the GeoWeave/VGGT sparse path with tiny
            iteration counts and a verified Hypersim train root.
  --paper   Paper recipe. Uses the 5090 paper training config family and the
            20260503 topk1024/l9-19 GeoWeave settings.
  --warmup-smoke
            2-GPU dev smoke for the dense-attention KL indexer warm-up path.
  --warmup-paper
            Paper warm-up recipe shape, using the 5090 warm-up launcher.

Environment overrides:
  GPU_IDS       Comma-separated GPUs. Smoke defaults to 0,1.
  SAVE_ROOT     Output root. Defaults to the feng01.zhou meshx bucket.
  VGGT_OFFICIAL_ROOT
                Fixed VGGT-1B split checkpoint directory. Defaults to the
                handoff bucket copy.
  EXP_NAME      Experiment name override.
  EXTRA_OVERRIDES
                Semicolon-separated EasyVolcap config overrides appended last.
  DRY_RUN=1     Print the command instead of executing it.
EOF
}

append_extra_override() {
    local item="$1"
    if [ -z "${item}" ]; then
        return 0
    fi
    if [ -n "${EXTRA_OVERRIDES:-}" ]; then
        EXTRA_OVERRIDES="${EXTRA_OVERRIDES};${item}"
    else
        EXTRA_OVERRIDES="${item}"
    fi
}

MODE_NAME=""
DRY_RUN="${DRY_RUN:-0}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --smoke)
            MODE_NAME="smoke"
            shift
            ;;
        --paper)
            MODE_NAME="paper"
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
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --override)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] --override requires KEY=VALUE" >&2
                exit 1
            fi
            append_extra_override "$2"
            shift 2
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
VGGT_OFFICIAL_ROOT="${VGGT_OFFICIAL_ROOT:-${HANDOFF_WEIGHT_ROOT}/vggt_official_VGGT-1B}"
export SAVE_ROOT
export HANDOFF_WEIGHT_ROOT
export VGGT_OFFICIAL_ROOT
export AGG_CKPT="${AGG_CKPT:-${VGGT_OFFICIAL_ROOT}/aggregator.pt}"
export CAM_CKPT="${CAM_CKPT:-${VGGT_OFFICIAL_ROOT}/camera.pt}"
export XYZ_CKPT="${XYZ_CKPT:-${VGGT_OFFICIAL_ROOT}/point.pt}"
export DPT_CKPT="${DPT_CKPT:-${VGGT_OFFICIAL_ROOT}/depth.pt}"
export TRA_CKPT="${TRA_CKPT:-${VGGT_OFFICIAL_ROOT}/track.pt}"

if [ "${MODE_NAME}" = "smoke" ]; then
    export MODE="${MODE:-local}"
    export CONFIG="${CONFIG:-vggt/vggt_official_finetune_5090_dev_hypersim_train}"
    export EXP_NAME="${EXP_NAME:-release/geoweave_paper_handoff/vggt_geoweave_smoke_$(date +%Y%m%d_%H%M%S)}"
    export GPU_IDS="${GPU_IDS:-0,1}"
    export FORCE_DIST="${FORCE_DIST:-1}"
    export DEBUG="${DEBUG:-0}"
    export RESUME="${RESUME:-0}"
    export SAVE_TO_AIDI="${SAVE_TO_AIDI:-0}"
    append_extra_override "runner_cfg.epochs=1"
    append_extra_override "runner_cfg.ep_iter=2"
    append_extra_override "runner_cfg.eval_ep=999"
    append_extra_override "runner_cfg.save_ep=999"
    append_extra_override "runner_cfg.save_latest_ep=999"
    append_extra_override "runner_cfg.log_interval=1"
    append_extra_override "runner_cfg.test_before_first_epoch=False"
    append_extra_override "dataloader_cfg.num_workers=0"
    append_extra_override "val_dataloader_cfg.num_workers=0"
    append_extra_override "dataloader_cfg.batch_sampler_cfg.n_srcs_list=[1]"
    append_extra_override "dataloader_cfg.batch_sampler_cfg.n_srcs_prob=[1.0]"
    append_extra_override "val_dataloader_cfg.max_iter=1"
    append_extra_override "val_dataloader_cfg.batch_sampler_cfg.n_srcs_list=[1]"
    append_extra_override "val_dataloader_cfg.batch_sampler_cfg.n_srcs_prob=[1.0]"
    append_extra_override "model_cfg.vggt_cfg.indexer_cfg.topk=1024"
elif [ "${MODE_NAME}" = "paper" ]; then
    export MODE="${MODE:-local}"
    export CONFIG="${CONFIG:-vggt/vggt_official_finetune_5090_paper}"
    export EXP_NAME="${EXP_NAME:-vggt/official/geoweave_5090x64_20260503_topk1024_l9_19}"
    export JOB_NAME="${JOB_NAME:-vggt_geoweave_5090x64_20260503_topk1024_l9_19}"
    export CLUSTER="${CLUSTER:-5090-release}"
    export NUM_NODES="${NUM_NODES:-8}"
    export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
    export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
    export FORCE_DIST="${FORCE_DIST:-1}"
    export SAVE_TO_AIDI="${SAVE_TO_AIDI:-1}"
    append_extra_override "model_cfg.vggt_cfg.indexer_cfg.topk=1024"
    append_extra_override "model_cfg.vggt_cfg.indexer_cfg.indexer_layers=9-19"
elif [ "${MODE_NAME}" = "warmup-smoke" ]; then
    export MODE="${MODE:-local}"
    export CONFIG="${CONFIG:-vggt/vggt_official_finetune_5090_dev_hypersim_train}"
    export EXP_NAME="${EXP_NAME:-release/geoweave_paper_handoff/vggt_geoweave_warmup_smoke_$(date +%Y%m%d_%H%M%S)}"
    export GPU_IDS="${GPU_IDS:-0,1}"
    export FORCE_DIST="${FORCE_DIST:-1}"
    export DEBUG="${DEBUG:-0}"
    export RESUME="${RESUME:-0}"
    export SAVE_TO_AIDI="${SAVE_TO_AIDI:-0}"
    export WARMUP_STEPS="${WARMUP_STEPS:-40000}"
    export INDEXER_HEADS="${INDEXER_HEADS:-4}"
    export INDEXER_LAYERS="${INDEXER_LAYERS:-9-19}"
    export WARMUP_MASK_SPARSE_METASETS="${WARMUP_MASK_SPARSE_METASETS:-0}"
    export HEAD_CHUNK_SIZE="${HEAD_CHUNK_SIZE:-1}"
    export SCORE_HEAD_CHUNK_SIZE="${SCORE_HEAD_CHUNK_SIZE:-1}"
    export SCORE_KEY_CHUNK_SIZE="${SCORE_KEY_CHUNK_SIZE:-1024}"
    export LAYERWISE_BACKWARD="${LAYERWISE_BACKWARD:-True}"
    export RUNNER_USE_TORCH_COMPILE="${RUNNER_USE_TORCH_COMPILE:-0}"
    append_extra_override "runner_cfg.epochs=1"
    append_extra_override "runner_cfg.ep_iter=2"
    append_extra_override "runner_cfg.eval_ep=999"
    append_extra_override "runner_cfg.extra_eval_every=0"
    append_extra_override "runner_cfg.save_ep=999"
    append_extra_override "runner_cfg.save_latest_ep=999"
    append_extra_override "runner_cfg.log_interval=1"
    append_extra_override "runner_cfg.test_before_first_epoch=False"
    append_extra_override "dataloader_cfg.num_workers=0"
    append_extra_override "val_dataloader_cfg.num_workers=0"
    append_extra_override "dataloader_cfg.dataset_cfg.proc_max_size=112"
    append_extra_override "val_dataloader_cfg.dataset_cfg.proc_max_size=112"
    append_extra_override "dataloader_cfg.batch_sampler_cfg.n_srcs_list=[1]"
    append_extra_override "dataloader_cfg.batch_sampler_cfg.n_srcs_prob=[1.0]"
    append_extra_override "val_dataloader_cfg.max_iter=1"
    append_extra_override "val_dataloader_cfg.batch_sampler_cfg.n_srcs_list=[1]"
    append_extra_override "val_dataloader_cfg.batch_sampler_cfg.n_srcs_prob=[1.0]"
    append_extra_override "model_cfg.vggt_cfg.indexer_cfg.topk=1024"
elif [ "${MODE_NAME}" = "warmup-paper" ]; then
    export MODE="${MODE:-remote}"
    export CONFIG="${CONFIG:-vggt/vggt_official_finetune_a800_paper}"
    export EXP_NAME="${EXP_NAME:-vggt/official/finetune_5090_warmup_currentbest_handoff}"
    export JOB_NAME="${JOB_NAME:-vggt_geoweave_warmup_5090x32_currentbest_handoff}"
    export CLUSTER="${CLUSTER:-project-5090-4dlabel-perception-acloud-langfang}"
    export NUM_NODES="${NUM_NODES:-4}"
    export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
    export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
    export SAVE_TO_AIDI="${SAVE_TO_AIDI:-1}"
    export WARMUP_STEPS="${WARMUP_STEPS:-40000}"
    export INDEXER_HEADS="${INDEXER_HEADS:-4}"
    export INDEXER_LAYERS="${INDEXER_LAYERS:-9-19}"
    export WARMUP_MASK_SPARSE_METASETS="${WARMUP_MASK_SPARSE_METASETS:-1}"
    export HEAD_CHUNK_SIZE="${HEAD_CHUNK_SIZE:-1}"
    export SCORE_HEAD_CHUNK_SIZE="${SCORE_HEAD_CHUNK_SIZE:-1}"
    export SCORE_KEY_CHUNK_SIZE="${SCORE_KEY_CHUNK_SIZE:-1024}"
    export LAYERWISE_BACKWARD="${LAYERWISE_BACKWARD:-True}"
    append_extra_override "model_cfg.vggt_cfg.indexer_cfg.topk=1024"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

if [ "${MODE_NAME}" = "warmup-smoke" ] || [ "${MODE_NAME}" = "warmup-paper" ]; then
    CMD=(bash "${REPO_ROOT}/aidi/scripts/vggt/submit_official_vggt_5090_warmup.sh")
else
    CMD=(bash "${REPO_ROOT}/aidi/scripts/vggt/train_official_vggt.sh")
fi
if [ "${DRY_RUN}" = "1" ]; then
    printf 'cd %q\n' "${REPO_ROOT}"
    env | grep -E '^(MODE|CONFIG|EXP_NAME|JOB_NAME|CLUSTER|NUM_NODES|GPUS_PER_NODE|GPU_IDS|FORCE_DIST|SAVE_ROOT|SAVE_TO_AIDI|HANDOFF_WEIGHT_ROOT|VGGT_OFFICIAL_ROOT|AGG_CKPT|CAM_CKPT|XYZ_CKPT|DPT_CKPT|TRA_CKPT|WARMUP_STEPS|INDEXER_HEADS|INDEXER_LAYERS|WARMUP_MASK_SPARSE_METASETS|HEAD_CHUNK_SIZE|SCORE_HEAD_CHUNK_SIZE|SCORE_KEY_CHUNK_SIZE|LAYERWISE_BACKWARD|EXTRA_OVERRIDES)=' | sort
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
