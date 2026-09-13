#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_CMD="${PYTHON_BIN}"
elif [ -x /opt/miniconda3/envs/easyvolcap/bin/python ]; then
    PYTHON_CMD="/opt/miniconda3/envs/easyvolcap/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "[ERROR] No usable Python found. Set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi

exec "${PYTHON_CMD}" "${REPO_ROOT}/aidi/scripts/vggt/run_unified_eval.py" "$@"
