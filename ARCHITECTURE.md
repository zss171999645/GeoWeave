# Meshx Architecture Notes

This document is the current map of the repository after the paper-submission
phase. It describes the practical module boundaries used for maintenance, not a
claim that the repository is already a clean product package.

## Current Posture

`meshx` is a research repository composed of several layers:

- EasyVolcap framework code.
- Official VGGT fork and paper-era sparse/indexer modifications.
- Custom sparse indexing and attention kernels.
- Pi3/native baseline training and evaluation code.
- AIDI submission, smoke, benchmark, and experiment-record tooling.
- Paper records, archive manifests, and result summaries.

The paper-era VGGT sparse/indexer implementation is now treated as frozen
research code. Keep it available and reproducible, but do not refactor it for
style unless a concrete bug, reproduction blocker, or new experiment requires
the change.

## Top-Level Boundaries

### `easyvolcap/`

Framework and importable runtime code.

- `easyvolcap/engine/`: registries, config loading, execution helpers.
- `easyvolcap/dataloaders/`: dataset and dataloader implementations.
- `easyvolcap/models/`: EasyVolcap model adapters. The main VGGT adapter is
  `easyvolcap/models/official_vggt_model.py`.
- `easyvolcap/runners/`: training, validation, distributed execution, and
  visualization runners.
- `easyvolcap/scripts/`: installed CLI entry points such as `evc-train`,
  `evc-test`, `evc-dist`, and `evc-pi3`.
- `easyvolcap/utils/`: shared helpers and model-family utilities.

### `easyvolcap/official_vggt/`

Vendored VGGT implementation plus sparse/indexer changes used by the paper-era
large-scene work.

Review anchors:

- `easyvolcap/official_vggt/models/vggt.py`: model assembly.
- `easyvolcap/official_vggt/models/aggregator.py`: global/frame attention flow
  and indexer stage scheduling.
- `easyvolcap/official_vggt/layers/indexer.py`: indexer scoring and top-k
  selection.
- `easyvolcap/official_vggt/layers/dsa_attention.py`: dense/sparse attention
  switching, source downsample paths, selector losses, and debug fast paths.
- `easyvolcap/official_vggt/README.md`: focused review guide for this subtree.

Maintenance rule: preserve behavior first. This subtree is not the first target
for architecture cleanup.

### `easyvolcap/utils/custom_indexer/` and `easyvolcap/utils/custom_flash_attn/`

Custom sparse top-k and sparse attention implementations. These are coupled to
the VGGT sparse/indexer path and should be changed only with targeted tests or
explicit performance/debug goals.

### `easyvolcap/utils/vggt/`

Older VGGT/EVC utility implementation. Keep this distinct from
`easyvolcap/official_vggt/`; new work should state explicitly which VGGT family
it targets.

### `easyvolcap/utils/pi3/` and `aidi/third_party/pi3_training/`

Pi3-related runtime code exists in two forms:

- `easyvolcap/utils/pi3/`: importable utility/model code used from this repo.
- `aidi/third_party/pi3_training/`: native Pi3 training/evaluation fork.

See `aidi/docs/pi3_native_framework_boundary.md` before moving code across this
boundary.

General third-party and in-house ownership labels are tracked in
`aidi/docs/third_party_boundaries.md`.

## Config Boundaries

### `configs/`

Primary EasyVolcap config tree.

- `configs/models/`: reusable model definitions.
- `configs/exps/vggt/`: VGGT experiment and evaluation configs.
- `configs/exps/vggt/evaluation/`: protocol-specific evaluation configs.
- `configs/specs/`: reusable runtime specs and overrides.

### `aidi/configs/`

AIDI and unified-evaluation configs. These often include absolute bucket paths,
checkpoint paths, and run-specific output roots. Treat them as reproducibility
records unless they are explicitly named as reusable templates.

## AIDI And Runtime Entry Points

Submission and execution are intentionally separated:

- `aidi/submit.py`: host-side AIDI packaging and job submission.
- `aidi/run.sh`: cluster/container runtime launcher.
- `aidi/scripts/vggt/`: VGGT training, evaluation, smoke, and benchmark
  wrappers.
- `aidi/scripts/pi3/`: Pi3 native training/submission wrappers.
- `aidi/scripts/baselines/`: baseline and protocol-specific evaluation tools.
- `aidi/scripts/ops/`: operational helpers for container/user/group setup.

For formal training or evaluation, follow `AGENTS.md` rather than inventing a
new ad hoc path. In particular, formal submissions should use the same entry,
image, config, and override path as the eventual job.

Current runtime entry-point index:

- `aidi/docs/runtime_entrypoints.md`

## Documentation And Records

Important current sources of truth:

- `AGENTS.md`: operational rules, branch constraints, remote-machine rules, and
  experiment-record requirements.
- `aidi/docs/vggt_train_test_protocol_overview.md`: training and evaluation
  protocol overview.
- `aidi/docs/vggt_official_finetune_memory.md`: VGGT official vs local
  implementation notes.
- `aidi/docs/runtime_entrypoints.md`: canonical and specialized runtime entry
  points.
- `aidi/docs/third_party_boundaries.md`: ownership labels for vendored forks,
  adapters, kernels, and baseline tooling.
- `aidi/docs/vggt_unified_eval_usage.md`: unified evaluation usage.
- `aidi/docs/submit_records.md`: submitted job records.
- `aidi/docs/records/experiment.md`: experiment-level records and commands.
- `aidi/docs/ops_audit.md`: remote/development-machine operation audit.
- `aidi/docs/archive/paper_submit_20260518/`: paper-submission archive indexes.

## Packaging Boundary

The importable package boundary is defined in `pyproject.toml`. Because this
repo contains large data, temporary results, and historical artifacts, package
discovery is intentionally not automatic. When a new importable
`easyvolcap/**/__init__.py` package is added, also add it to
`pyproject.toml`.

Regression guard:

```bash
python3 - <<'PY'
import importlib.util
from pathlib import Path
path = Path('tests/packaging_config_tests.py')
spec = importlib.util.spec_from_file_location('packaging_config_tests', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.test_pyproject_lists_all_easyvolcap_init_packages()
PY
```

If a full test environment is available, prefer:

```bash
python3 -m pytest -q tests/packaging_config_tests.py
```

## Cleanup Policy

Do not delete non-temporary files or directories without explicit confirmation.

Safe cleanup candidates are limited to clearly temporary outputs, caches, upload
packages, build products, logs, and generated snapshots. Paper records, configs,
checkpoints, result summaries, and protocol docs should be indexed before any
move or deletion.

## Near-Term Organization Priorities

1. Keep packaging metadata and importable package lists correct.
2. Keep root-level architecture and protocol entry documentation current.
3. Add lightweight indexes for scripts/configs that remain active.
4. Split new reusable evaluation logic into importable helpers before adding new
   large shell wrappers.
5. Leave paper-frozen VGGT sparse/indexer internals alone unless there is a
   concrete bug or reproduction need.

## Test Taxonomy Target

Tests should gradually be labelled or grouped by cost and dependency:

- Fast unit tests: no GPU, no bucket, no remote machine.
- Packaging/config tests: validate metadata and config expansion.
- GPU smoke tests: single machine, short runtime, no formal submission.
- Submission dry-runs: verify AIDI package contents and run commands.
- Formal/canary jobs: same entry and image as long-running jobs.

This taxonomy is a target for future cleanup; existing tests are not yet fully
organized this way.
