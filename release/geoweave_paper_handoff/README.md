# GeoWeave Paper Handoff

This directory is the paper-facing control and documentation area inside the
clean GeoWeave handoff code package. The handoff package itself keeps the
standard meshx core layout (`easyvolcap/`, `aidi/`, `configs/`) so
company engineers can read the code through the same architecture they already
know, while excluding research-time scripts, temporary configs, local caches,
and historical wrappers.

## Source State

| Field | Value |
| --- | --- |
| Main repo | `meshx` |
| Clean handoff package | `/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/code/geoweave_paper_clean_code_20260603_strict_002.tar.gz` |
| Paper baseline commit | `e631d7c50c5bc1c6fb2c0515c9d87da80981b12f` |
| Branch | `feat-vggt-eval-largescene` |
| Archive tag | `paper-submit-20260518` |

`paper-submit-20260518` records the post-submission archive index. The
method code baseline for handoff is the commit above.
Extract the clean handoff package to get a meshx-shaped directory named
`geoweave_paper_clean_code_20260603_strict_002`.

## Directory Layout

| Path | Purpose |
| --- | --- |
| `anonymous_model/` | Model-only anonymous GeoWeave package extracted from the paper baseline. |
| `scripts/` | Clean executable wrappers for training and inference/evaluation. |
| `configs/` | Pointers to the final config families; avoids copying configs that would drift. |
| `docs/` | Code scope and cleanup policy for handoff. |
| `manifests/` | Keep/archive/delete candidate lists for pruning research-time clutter. |

Final paper-era weights are fixed in the account bucket and indexed in
`manifests/WEIGHTS.tsv`.

To regenerate a complete meshx-shaped clean code package:

```bash
bash release/geoweave_paper_handoff/scripts/build_clean_meshx_package.sh /path/to/geoweave_paper_clean_code
```

## First Commands

Run the preflight before handing the directory to another engineer:

```bash
bash release/geoweave_paper_handoff/scripts/check_handoff.sh
```

On a real training/inference environment with `torch` installed, run the
stricter runtime import check:

```bash
STRICT_IMPORT=1 bash release/geoweave_paper_handoff/scripts/check_handoff.sh
```

Training and inference/evaluation entry points:

```bash
# VGGT / GeoWeave 2-GPU training smoke
bash release/geoweave_paper_handoff/scripts/train_vggt.sh --smoke

# VGGT / GeoWeave 2-GPU warm-up training smoke
bash release/geoweave_paper_handoff/scripts/train_vggt.sh --warmup-smoke

# VGGT / GeoWeave paper training recipe expansion
bash release/geoweave_paper_handoff/scripts/train_vggt.sh --paper --dry-run

# VGGT / GeoWeave paper warm-up recipe expansion
bash release/geoweave_paper_handoff/scripts/train_vggt.sh --warmup-paper --dry-run

# Pi3 / GeoWeave 2-GPU sparse/indexer training smoke
bash release/geoweave_paper_handoff/scripts/train_pi3.sh --smoke

# Pi3 / GeoWeave 2-GPU warm-up training smoke
bash release/geoweave_paper_handoff/scripts/train_pi3.sh --warmup-smoke

# Pi3 / GeoWeave paper sparse/indexer recipe expansion
bash release/geoweave_paper_handoff/scripts/train_pi3.sh --paper --dry-run

# Pi3 / GeoWeave paper warm-up recipe expansion
bash release/geoweave_paper_handoff/scripts/train_pi3.sh --warmup-paper --dry-run

# Continue Pi3 training from the fixed handoff checkpoint_79
bash release/geoweave_paper_handoff/scripts/train_pi3.sh --resume-final --dry-run

# Unified inference/evaluation entry
bash release/geoweave_paper_handoff/scripts/eval_unified.sh --help

# VGGT / Pi3 final paper evaluation configs, using fixed handoff weights
bash release/geoweave_paper_handoff/scripts/eval_vggt.sh --paper --dry-run
bash release/geoweave_paper_handoff/scripts/eval_pi3.sh --paper --dry-run

# Minimal ETH3D pose evaluation smoke configs
bash release/geoweave_paper_handoff/scripts/eval_vggt.sh --smoke --dry-run
bash release/geoweave_paper_handoff/scripts/eval_pi3.sh --smoke --dry-run

# Paper robustness protocols: weak-overlap and distractor context
bash release/geoweave_paper_handoff/scripts/robustness_pi3.sh --list
bash release/geoweave_paper_handoff/scripts/robustness_pi3.sh --eval-weak-scannetpp --model both --dry-run
bash release/geoweave_paper_handoff/scripts/robustness_pi3.sh --eval-weak-waymo --model both --dry-run
bash release/geoweave_paper_handoff/scripts/robustness_pi3.sh --eval-distractor-waymo --model both --dry-run

# Paper-era eval command expansion without running GPU jobs
bash release/geoweave_paper_handoff/scripts/eval_unified.sh \
  --config aidi/configs/vggt/unified_eval_vggt_indexer_20260503_pt79_trusted.yaml \
  --dry-run
```

The wrappers intentionally delegate to the original repository entry points
instead of copying large scripts into this directory. That preserves executable
relative paths and avoids creating a second copy of training logic.

`aidi/configs/vggt/unified_eval_local.yaml` is useful as a template, but it may
not enable any dataset-centric tasks. Use the trusted paper-era config above
when checking final paper command expansion.

## Smoke Vs Paper Configs

The `--smoke` and `--warmup-smoke` modes are startup checks only. They reduce
data, iterations, resolution, checkpoint writing, or enabled tasks so that a
new environment can verify imports, launch, forward/backward, optimizer step,
and logging. They must not be compared with paper tables.

For paper-result command expansion, use `--paper --dry-run`. The fixed final
eval configs are `configs/eval/vggt_geoweave_final_trusted.yaml` and
`configs/eval/pi3_geoweave_final_all_tasks.yaml`; they point to the archived
handoff weights under the account bucket. The ETH3D pose paper config uses the
full `max_frames=100` protocol. The ETH3D pose smoke configs intentionally use
`max_frames=12`.

The clean package intentionally excludes historical one-off wrappers and
generic root-level utility scripts. Do not recover old wrappers unless a
handoff task explicitly needs them and the dependency is verified by
`scripts/check_handoff.sh`.

For handoff, prefer the dedicated wrappers:

| Wrapper | Default config/recipe | Purpose |
| --- | --- | --- |
| `scripts/train_vggt.sh --warmup-smoke` | `aidi/scripts/vggt/submit_official_vggt_5090_warmup.sh` plus Hypersim dev data and tiny iteration overrides, without checkpoint save | Prove VGGT GeoWeave dense-KL indexer warm-up starts on 2 GPUs. |
| `scripts/train_vggt.sh --smoke` | `configs/exps/vggt/vggt_official_finetune_5090_dev_hypersim_train.yaml` plus tiny iteration overrides, without checkpoint save | Prove VGGT GeoWeave sparse training starts on 2 GPUs. |
| `scripts/train_vggt.sh --warmup-paper` | `aidi/scripts/vggt/submit_official_vggt_5090_warmup.sh` current-best 5090 warm-up settings | Reproduce the VGGT warm-up recipe shape. |
| `scripts/train_vggt.sh --paper` | `configs/exps/vggt/vggt_official_finetune_5090_paper.yaml` | Reproduce the paper training recipe shape. |
| `scripts/train_pi3.sh --warmup-smoke` | Pi3 warm-up from fixed base safetensors, TarTanAir low-res 2-GPU smoke | Prove Pi3 GeoWeave indexer warm-up starts on 2 GPUs. |
| `scripts/train_pi3.sh --smoke` | Pi3 sparse stage2 from fixed warm-up checkpoint_49, stable TarTanAir low-res 2-GPU smoke | Prove Pi3 GeoWeave sparse training starts on 2 GPUs. |
| `scripts/train_pi3.sh --warmup-paper` | 20260430 all-layer Pi3 native indexer warm-up recipe that produced the checkpoint_49 family | Reproduce the Pi3 warm-up recipe shape. |
| `scripts/train_pi3.sh --paper` | Final 20260505 Pi3 sparse/indexer stage2 recipe | Reproduce the paper Pi3 recipe shape. |
| `scripts/train_pi3.sh --resume-final` | Fixed handoff checkpoint_79 full accelerator directory | Continue training from the archived final Pi3 state. |
| `scripts/eval_vggt.sh --paper` | `configs/eval/vggt_geoweave_final_trusted.yaml` | Final VGGT trusted unified eval using fixed handoff `79.pt`. |
| `scripts/eval_pi3.sh --paper` | `configs/eval/pi3_geoweave_final_all_tasks.yaml` | Final Pi3 unified eval using fixed handoff `checkpoint_79`. |
| `scripts/robustness_pi3.sh --eval-weak-scannetpp` | Fixed ScanNet++ low-overlap 5+5 protocol, 12 tuples | Reproduce the paper weak-overlap ScanNet++ Pi3 comparison. |
| `scripts/robustness_pi3.sh --eval-weak-waymo` | Fixed Waymo same-window camera 03/04 multi-camera 5+5 protocol, 90 tuples | Reproduce the paper weak-overlap Waymo Pi3 comparison. |
| `scripts/robustness_pi3.sh --eval-distractor-waymo` | Fixed Waymo frozen plausible-wrong context protocol, eval frames 0..5 | Reproduce the paper distractor-context Pi3 comparison. |

For the robustness table, run both models with `--model both`, then use
`--summarize-weak-scannetpp`, `--summarize-weak-waymo`, or
`--summarize-distractor-waymo` against the same `RESULT_ROOT`.
