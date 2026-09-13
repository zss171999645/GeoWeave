#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash release/geoweave_paper_handoff/scripts/build_clean_meshx_package.sh <dest_dir>

Creates a meshx-shaped clean GeoWeave handoff code package from tracked files.
The package keeps the familiar meshx architecture while excluding local caches,
temporary experiment files, personal agent notes, and historical one-off wrappers.

The destination directory must not already exist.
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

DEST="${1:-}"
if [ -z "${DEST}" ]; then
    usage >&2
    exit 2
fi

if [ -e "${DEST}" ]; then
    echo "[ERROR] Destination already exists: ${DEST}" >&2
    echo "[ERROR] Pick a new path. This script never deletes or overwrites an existing package." >&2
    exit 1
fi

PYTHON_CMD="${PYTHON_CMD:-python3}"
if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
    echo "[ERROR] Python is required to build the filtered file list." >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

FILE_LIST="${TMP_DIR}/files.txt"
MANIFEST="${TMP_DIR}/manifest.tsv"

"${PYTHON_CMD}" - "${REPO_ROOT}" "${FILE_LIST}" "${MANIFEST}" <<'PY'
from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
file_list = Path(sys.argv[2])
manifest = Path(sys.argv[3])

exclude_globs = [
    ".autonomous/**",
    ".codex/**",
    ".superpowers/**",
    ".worktrees/**",
    ".research/**",
    ".idea/**",
    ".vscode/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    ".DS_Store",
    "**/.DS_Store",
    "tmp/**",
    "tensorboard_logs/**",
    "expanded_configs/**",
    "experiment_records/**",
    "Running/**",
    "build/**",
    "dist/**",
    "*.egg-info/**",
    "job.tar.gz",
    "job.tar.gz.enc",
    "job.tar.gz.enc.*",
    "AGENTS.md",
    "task_plan.md",
    "findings.md",
    "progress.md",
    "docs/superpowers/**",
    "aidi/docs/ops_audit.md",
    "aidi/docs/submit_records.md",
    "aidi/docs/records/**",
    "release/anonymous_geoweave_model/**",
    "scripts/**",
    "aidi/scripts/ops/**",
    "aidi/scripts/baselines/**",
    "aidi/scripts/codex/**",
    "aidi/scripts/moge/**",
    "aidi/scripts/vggt/**",
    "aidi/scripts/pi3/**",
    "aidi/scripts/vggt/submit_official_vggt_5090_sparse_topk*.sh",
    "aidi/scripts/vggt/submit_official_vggt_a800_*.sh",
    "aidi/scripts/vggt/submit_official_vggt_l20_*.sh",
    "aidi/scripts/vggt/submit_eval_*",
    "aidi/scripts/vggt/eval_official_vggt_sparse_datasets_*.sh",
    "aidi/scripts/vggt/bench_*.py",
    "aidi/scripts/vggt/check_kernel_forward_backward_diff.py",
    "aidi/scripts/vggt/render_*.py",
    "aidi/scripts/vggt/export_*_visual*.py",
    "aidi/scripts/baselines/build_*.py",
    "aidi/scripts/baselines/eval_pi3_*.py",
]

allow_exact = {
    "aidi/scripts/ops/sync_host_groups_to_container.sh",
    "aidi/scripts/ops/check_group_membership.sh",
    "aidi/scripts/vggt/run_unified_eval.py",
    "aidi/scripts/vggt/train_official_vggt.sh",
    "aidi/scripts/vggt/submit_official_vggt_5090_warmup.sh",
    "aidi/scripts/vggt/eval_co3d_pose_official_upstream.py",
    "aidi/scripts/vggt/eval_da3_pose_benchmark.py",
    "aidi/scripts/vggt/eval_relpose_1500_benchmark.py",
    "aidi/scripts/vggt/eval_vggt_re10k_pose_lightweight.py",
    "aidi/scripts/vggt/resolve_co3dv2_official_scene_roots.py",
    "aidi/scripts/vggt/run_co3dv2_official_upstream.sh",
    "aidi/scripts/vggt/run_eval_7scenes_mv_recon_pi3_style.sh",
    "aidi/scripts/vggt/run_eval_dtu_mv_recon_pi3_style.sh",
    "aidi/scripts/vggt/run_eval_eth3d_mv_recon_pi3_style.sh",
    "aidi/scripts/vggt/run_eval_nrgbd_mv_recon_pi3_style.sh",
    "aidi/scripts/vggt/vggt_omega_eval_utils.py",
    "aidi/scripts/baselines/eval_config_utils.py",
    "aidi/scripts/baselines/eval_pi3_co3d_pose_official.py",
    "aidi/scripts/baselines/eval_pi3_depth_protocol.py",
    "aidi/scripts/baselines/eval_pi3_monodepth_protocol.py",
    "aidi/scripts/baselines/eval_pi3_mv_recon_core.py",
    "aidi/scripts/baselines/eval_pi3_re10k_pose_official.py",
    "aidi/scripts/baselines/eval_pi3_relpose_distance_protocol.py",
    "aidi/scripts/baselines/eval_pi3_videodepth_protocol.py",
    "aidi/scripts/baselines/exr_read_utils.py",
    "aidi/scripts/baselines/generate_seq_id_map.py",
    "aidi/scripts/baselines/pi3_checkpoint_loader.py",
    "aidi/scripts/baselines/pi3_metric_utils.py",
    "aidi/scripts/baselines/prepare_eth3d_pi3_style.py",
    "aidi/scripts/baselines/overlap_noise_seq_map_utils.py",
    "aidi/scripts/baselines/build_evc_same_scene_candidate_pool_benchmark.py",
    "aidi/scripts/baselines/build_fixed10_overlap_band_benchmark.py",
    "aidi/scripts/baselines/build_same_scene_low_overlap_5plus5_benchmark.py",
    "aidi/scripts/baselines/build_three_dataset_stride4_distractor_benchmark.py",
    "aidi/scripts/baselines/build_driving_multicamera_all_eval_benchmark.py",
    "aidi/scripts/baselines/build_waymo_plausible_context_diagnostic.py",
    "aidi/scripts/baselines/build_waymo_plausible_wrong_trigger_protocol.py",
    "aidi/scripts/baselines/summarize_waymo_plausible_wrong_trigger_protocol.py",
    "aidi/scripts/pi3/train_pi3_official.sh",
    "aidi/scripts/pi3/submit_pi3_5090_warmup.sh",
    "aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh",
}

allow_globs = set()

def allowed(path: str) -> bool:
    if path in allow_exact:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in allow_globs)

def excluded(path: str) -> bool:
    if allowed(path):
        return False
    return any(fnmatch.fnmatch(path, pattern) for pattern in exclude_globs)

tracked = subprocess.check_output(
    ["git", "-C", str(repo), "ls-files"],
    text=True,
).splitlines()

kept: list[str] = []
excluded_paths: list[tuple[str, str]] = []
for path in tracked:
    if allowed(path):
        kept.append(path)
        continue
    match = next((pattern for pattern in exclude_globs if fnmatch.fnmatch(path, pattern)), None)
    if match:
        excluded_paths.append((path, match))
    else:
        kept.append(path)

file_list.write_text("\n".join(kept) + "\n", encoding="utf-8")
manifest.write_text(
    "path\tmatched_exclude\n"
    + "\n".join(f"{path}\t{pattern}" for path, pattern in excluded_paths)
    + ("\n" if excluded_paths else ""),
    encoding="utf-8",
)

required = [
    "easyvolcap/official_vggt/layers/dsa_attention.py",
    "easyvolcap/official_vggt/layers/indexer.py",
    "easyvolcap/official_vggt/models/aggregator.py",
    "easyvolcap/models/official_vggt_model.py",
    "easyvolcap/utils/custom_indexer",
    "easyvolcap/utils/custom_flash_attn",
    "aidi/third_party/pi3_training/pi3/models/pi3_training.py",
    "aidi/scripts/vggt/train_official_vggt.sh",
    "aidi/scripts/vggt/submit_official_vggt_5090_warmup.sh",
    "aidi/scripts/pi3/train_pi3_official.sh",
    "aidi/scripts/pi3/submit_pi3_5090_warmup.sh",
    "aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh",
    "aidi/scripts/vggt/run_unified_eval.py",
    "aidi/scripts/vggt/eval_co3d_pose_official_upstream.py",
    "aidi/scripts/vggt/eval_da3_pose_benchmark.py",
    "aidi/scripts/vggt/eval_relpose_1500_benchmark.py",
    "aidi/scripts/vggt/eval_vggt_re10k_pose_lightweight.py",
    "aidi/scripts/vggt/run_co3dv2_official_upstream.sh",
    "aidi/scripts/vggt/run_eval_dtu_mv_recon_pi3_style.sh",
    "aidi/scripts/vggt/run_eval_eth3d_mv_recon_pi3_style.sh",
    "aidi/scripts/vggt/run_eval_7scenes_mv_recon_pi3_style.sh",
    "aidi/scripts/vggt/run_eval_nrgbd_mv_recon_pi3_style.sh",
    "aidi/scripts/baselines/eval_pi3_re10k_pose_official.py",
    "aidi/scripts/baselines/eval_pi3_co3d_pose_official.py",
    "aidi/scripts/baselines/eval_pi3_depth_protocol.py",
    "aidi/scripts/baselines/eval_pi3_monodepth_protocol.py",
    "aidi/scripts/baselines/eval_pi3_videodepth_protocol.py",
    "aidi/scripts/baselines/eval_pi3_relpose_distance_protocol.py",
    "aidi/scripts/baselines/eval_pi3_mv_recon_core.py",
    "aidi/scripts/baselines/overlap_noise_seq_map_utils.py",
    "aidi/scripts/baselines/build_evc_same_scene_candidate_pool_benchmark.py",
    "aidi/scripts/baselines/build_fixed10_overlap_band_benchmark.py",
    "aidi/scripts/baselines/build_same_scene_low_overlap_5plus5_benchmark.py",
    "aidi/scripts/baselines/build_three_dataset_stride4_distractor_benchmark.py",
    "aidi/scripts/baselines/build_driving_multicamera_all_eval_benchmark.py",
    "aidi/scripts/baselines/build_waymo_plausible_context_diagnostic.py",
    "aidi/scripts/baselines/build_waymo_plausible_wrong_trigger_protocol.py",
    "aidi/scripts/baselines/summarize_waymo_plausible_wrong_trigger_protocol.py",
    "release/geoweave_paper_handoff/scripts/check_handoff.sh",
    "release/geoweave_paper_handoff/scripts/robustness_pi3.sh",
]

missing = []
kept_set = set(kept)
for item in required:
    item_path = repo / item
    if item_path.is_dir():
        if not any(path == item or path.startswith(item + "/") for path in kept_set):
            missing.append(item)
    elif item not in kept_set:
        missing.append(item)

if missing:
    print("[ERROR] Required handoff paths were filtered out:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(1)

print(f"[INFO] kept_files={len(kept)} excluded_files={len(excluded_paths)}")
PY

mkdir -p "${DEST}"
rsync -a --files-from="${FILE_LIST}" "${REPO_ROOT}/" "${DEST}/"
mkdir -p "${DEST}/release/geoweave_paper_handoff/manifests"
cp "${MANIFEST}" "${DEST}/release/geoweave_paper_handoff/manifests/CLEAN_PACKAGE_EXCLUDED.tsv"

cat > "${DEST}/GEOWEAVE_CLEAN_CODE_README.md" <<'EOF'
# GeoWeave Clean Code Package

This directory is a clean meshx-shaped code package for GeoWeave handoff.
It keeps the familiar meshx core layout and removes local caches, temporary
experiments, personal agent notes, generic root-level utility scripts, and
historical one-off wrappers. Under `aidi/scripts`, only GeoWeave paper training,
unified evaluation, and their direct runtime dependencies are retained.

Primary entry points:

- `release/geoweave_paper_handoff/scripts/train_vggt.sh`
- `release/geoweave_paper_handoff/scripts/train_pi3.sh`
- `release/geoweave_paper_handoff/scripts/eval_vggt.sh`
- `release/geoweave_paper_handoff/scripts/eval_pi3.sh`
- `release/geoweave_paper_handoff/scripts/robustness_pi3.sh`
- `release/geoweave_paper_handoff/scripts/check_handoff.sh`

Run the preflight from this directory:

```bash
bash release/geoweave_paper_handoff/scripts/check_handoff.sh
```

Smoke configs are only startup checks. Use `--paper --dry-run` to inspect the
fixed standard benchmark evaluation configs, and use `robustness_pi3.sh` for
the paper weak-overlap and distractor-context protocols.
EOF

echo "[OK] Clean GeoWeave meshx package written to ${DEST}"
echo "[OK] Exclusion manifest: ${DEST}/release/geoweave_paper_handoff/manifests/CLEAN_PACKAGE_EXCLUDED.tsv"
