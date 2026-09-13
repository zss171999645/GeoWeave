# Code Scope

This handoff keeps the paper code in three layers.

## Layer 1: Public / Anonymous Model Code

`release/geoweave_paper_handoff/anonymous_model` is extracted from the paper
baseline and mirrors the submitted model-only package. It contains the
GeoWeave dependency scorer, selected-token attention, compact configs, and
VGGT/Pi3-style model entries. It intentionally excludes training launchers,
dataset adapters, private paths, checkpoints, logs, and experiment records.

## Layer 2: Internal Executable Mainline

The executable training and inference/evaluation path remains in the main
repository:

| Role | Path |
| --- | --- |
| VGGT selected-context attention | `easyvolcap/official_vggt/layers/dsa_attention.py` |
| Dependency scorer | `easyvolcap/official_vggt/layers/indexer.py` |
| VGGT block integration | `easyvolcap/official_vggt/layers/block.py` |
| VGGT aggregator integration | `easyvolcap/official_vggt/models/aggregator.py` |
| EasyVolcap wrapper | `easyvolcap/models/official_vggt_model.py` |
| Top-k/indexer kernels | `easyvolcap/utils/custom_indexer/` |
| Sparse flash attention kernels | `easyvolcap/utils/custom_flash_attn/` |
| Pi3 training fork | `aidi/third_party/pi3_training/` |
| Unified eval runner | `aidi/scripts/vggt/run_unified_eval.py` |

## Layer 3: Historical Research Material

Historical wrapper scripts, temporary configs, speed benchmarks, debug helpers,
and one-off ablations should not be presented as mainline handoff code. They
are kept only when they preserve experiment traceability.
