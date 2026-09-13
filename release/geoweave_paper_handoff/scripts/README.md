# Scripts

These scripts are the main handoff entry points inside a clean meshx-shaped
code package. They keep the implementation in the familiar meshx locations
instead of duplicating model code under `release/`.

| Script | Delegates to | Use |
| --- | --- | --- |
| `build_clean_meshx_package.sh` | `git ls-files` + `rsync` | Build a complete clean meshx-shaped code package for handoff. |
| `train_vggt.sh` | `aidi/scripts/vggt/train_official_vggt.sh` and `aidi/scripts/vggt/submit_official_vggt_5090_warmup.sh` | VGGT / GeoWeave warm-up and sparse training. |
| `train_pi3.sh` | `aidi/scripts/pi3/submit_pi3_5090_warmup.sh`, `aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh`, and `aidi/scripts/pi3/train_pi3_official.sh` | Pi3 / GeoWeave warm-up and sparse/indexer training. |
| `eval_unified.sh` | `aidi/scripts/vggt/run_unified_eval.py` | Inference/evaluation through unified benchmark runner. |
| `eval_vggt.sh` | `eval_unified.sh` | VGGT final/smoke eval configs with fixed handoff weights. |
| `eval_pi3.sh` | `eval_unified.sh` | Pi3 final/smoke eval configs with fixed handoff weights. |
| `robustness_pi3.sh` | Pi3 relpose evaluator plus protocol builders/summarizer | Paper weak-overlap and distractor-context reproducibility entry. |
| `check_handoff.sh` | Repository paths and anonymous package | Non-GPU preflight. |

Training scripts should be executed only after the normal development-machine
sync and validation procedure. The preflight script is safe to run locally.

Both training wrappers require an explicit mode. Use `--warmup-smoke` and
`--smoke` for the 2-GPU startup checks, `--warmup-paper` and `--paper` for the
full recipe expansions, and `--dry-run` to print resolved environment variables
without launching training.

Paper robustness experiments are separate from the standard benchmark configs.
Use `robustness_pi3.sh --list` to inspect the fixed protocol roots, then run
`--eval-weak-scannetpp`, `--eval-weak-waymo`, or `--eval-distractor-waymo`.
The matching `--build-*` modes rebuild the protocol tuples, and the
`--summarize-*` modes aggregate official-vs-GeoWeave outputs.
