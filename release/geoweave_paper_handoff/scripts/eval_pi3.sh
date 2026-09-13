#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG="${REPO_ROOT}/release/geoweave_paper_handoff/configs/eval/pi3_geoweave_final_all_tasks.yaml"
if [ "${1:-}" = "--smoke" ]; then
    CONFIG="${REPO_ROOT}/release/geoweave_paper_handoff/configs/eval/pi3_geoweave_smoke_eth3d_pose.yaml"
    shift
elif [ "${1:-}" = "--paper" ]; then
    shift
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec bash "${SCRIPT_DIR}/eval_unified.sh" --config "${CONFIG}" "$@"
