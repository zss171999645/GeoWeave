# Training And Evaluation Handoff

## VGGT / GeoWeave

The executable VGGT path is the internal EasyVolcap model:

- `easyvolcap/official_vggt/layers/indexer.py`
- `easyvolcap/official_vggt/layers/dsa_attention.py`
- `easyvolcap/official_vggt/layers/block.py`
- `easyvolcap/official_vggt/models/aggregator.py`
- `easyvolcap/models/official_vggt_model.py`

Use `scripts/train_vggt.sh --warmup-smoke` to prove the dense-attention KL
indexer warm-up path on 2 GPUs. It delegates to
`aidi/scripts/vggt/submit_official_vggt_5090_warmup.sh`, but uses the verified
Hypersim dev root, tiny iteration counts, reduced resolution, and no checkpoint
save. Use `scripts/train_vggt.sh --smoke` to prove the sparse stage2 path on 2
GPUs. Use `scripts/train_vggt.sh --warmup-paper --dry-run` and
`scripts/train_vggt.sh --paper --dry-run` to inspect the full warm-up and
sparse recipes before launching long runs.

Final evaluation uses `scripts/eval_vggt.sh --paper`, which points to the fixed
handoff `79.pt` and the surrogate eval architecture config used for the paper.
The VGGT ETH3D pose paper path resolves to the full `max_frames=100` protocol;
the smoke path is deliberately reduced to ETH3D pose with `max_frames=12`.

## Pi3 / GeoWeave

The executable Pi3 path is the native Pi3 fork:

- `aidi/third_party/pi3_training/pi3/models/pi3_training.py`
- `aidi/third_party/pi3_training/pi3/layers/`
- `aidi/third_party/pi3_training/trainers/pi3_trainer.py`
- `aidi/scripts/pi3/submit_pi3_5090_sparse_fp16_layers9_21.sh`
- `aidi/scripts/pi3/train_pi3_official.sh`

Use `scripts/train_pi3.sh --warmup-smoke` to prove the native Pi3 GeoWeave
indexer warm-up path on 2 GPUs from the fixed Pi3 base safetensors. The paper
warm-up recipe is exposed by `scripts/train_pi3.sh --warmup-paper --dry-run`;
it matches the 20260430 all-layer streaming-KL warm-up family that produced
checkpoint_49. Use `scripts/train_pi3.sh --smoke` to run the sparse/indexer
stage2 smoke from the fixed warm-up checkpoint_49. This is the same GeoWeave
sparse path as the paper run, reduced to 2 GPUs and low resolution. The
original 20260505 paper stage2 record used checkpoint_24, but that file is no
longer visible in the bucket; checkpoint_49 is the available warm-up state
retained for executable handoff. Use `scripts/train_pi3.sh --paper --dry-run`
to inspect the handoff stage2 recipe, and
`scripts/train_pi3.sh --resume-final --dry-run` to inspect continuation from
the fixed checkpoint_79 full accelerator state.

Final evaluation uses `scripts/eval_pi3.sh --paper`, which points to the fixed
handoff checkpoint_79 model file and copied Hydra config.
The Pi3 paper config enables the full unified-eval task set retained for
handoff, while the smoke config only enables ETH3D pose with `max_frames=12`.

The paper Section 4.3 robustness experiments are exposed separately through
`scripts/robustness_pi3.sh`. `--eval-weak-scannetpp` uses the fixed ScanNet++
same-scene low-overlap 5+5 protocol with 12 tuples. `--eval-weak-waymo` uses
the fixed Waymo same-window camera 03/04 multi-camera 5+5 protocol with 90
tuples. `--eval-distractor-waymo` uses the frozen Waymo plausible-wrong context
protocol and scores only eval frames `0..5`, matching the clean-prefix
definition. The paired summarizers in the same wrapper aggregate Pi3 official
and GeoWeave outputs from a shared `RESULT_ROOT`.

## Wrapper Scope

The clean package keeps only the wrappers needed by the handoff surfaces:
VGGT training, Pi3 training, VGGT/Pi3 unified evaluation, and Pi3 paper
robustness reproduction. Historical badcase wrappers, unrelated baseline
reproduction helpers, generic root-level utility scripts, and intermediate
experiment launchers are excluded from the generated package. If a future task
needs one of those files, add it back through `build_clean_meshx_package.sh`
with a direct dependency reason and rerun `scripts/check_handoff.sh`.

## Fixed Weight Root

All handoff eval configs use:

`/horizon-bucket/saturn_v_dev/01_users/feng01.zhou/projects/meshx/baseline/trained_model/geoweave_paper_handoff_20260602_final`

The local index is `release/geoweave_paper_handoff/manifests/WEIGHTS.tsv`.
