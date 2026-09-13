# ScanNet++ Paired Overlap Sweep Execution Record

Date: 2026-08-25

## Frozen configuration

- Branch: `revision-required-metrics-20260824`
- Worktree: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/worktrees/revision-required-metrics-20260824`
- Design commit: `f93d0ad`
- Dataset root on shared CFS/inside AIHC: `/mnt/cfs/datasets/scannetpp/scannetpplus/Scannetpp/data`
- Source protocol: official DSLR images + Nerfstudio poses; virtual 64-pixel-high pinhole depths ray-cast from `mesh_aligned_0.05.ply` for overlap measurement.
- Output root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/scannetpp_overlap_sweep_20260825`
- Log root: `/mnt/cfs/train_logs/zhoufeng`
- AIHC queue: `aihcq-em3tafa9yg0u`
- AIHC resource pool: `aihc-b0dc2g8oshzx`
- Pi3 checkpoint: `/mnt/cfs/zhoufeng/geoweave_rebuttal_code_weights_20260604/weights/geoweave_paper_handoff_20260602_final/pi3_base_yyfz233/Pi3_model.safetensors`
- GeoWeave-Pi3 checkpoint: `/mnt/cfs/zhoufeng/geoweave_rebuttal_code_weights_20260604/weights/geoweave_paper_handoff_20260602_final/pi3_geoweave_native_sparse_20260505_checkpoint_79/checkpoint_79/pytorch_model.bin`
- Formal sample: 20 paired Group A anchors x 4 overlap levels = 80 tuples per model.
- Overlap bands: high `[0.10,0.20)`, medium `[0.03,0.10)`, low `[0.005,0.03)`, near-zero `[0,0.005)`.

## Job ledger

| Purpose | Job ID | Commit | Start | End | Status | Notes |
|---|---|---|---|---|---|---|
| CFS mount probe | local CPU smoke | uncommitted | 2026-08-25 | 2026-08-26 | passed | Official ScanNet++ root, DSLR images, transforms, and mesh verified |
| One-scene preflight | local CPU smoke | uncommitted | 2026-08-26 | 2026-08-26 | passed | `bd7375297e`, 256 views, 8,455 candidates, 125 complete anchors |
| Formal preflight | `job-567h27gznw6n` | `d5ee226` | 2026-08-26 00:14 | 2026-08-26 00:20 | succeeded | 20 scenes; 306,194 candidates; 3,178 complete anchors |
| Materialization (initial) | `job-ruars73cmw24` | `c3c4cbc` | 2026-08-26 00:22 | 2026-08-26 00:22 | succeeded | Superseded after temporal-anchor audit |
| Materialization (quantile manifest) | `job-a6zugv0hz7t5` | `5174da3` | 2026-08-26 00:32 | 2026-08-26 00:32 | succeeded | 20 anchors, 80 tuples, prior directory preserved |
| Pi3 inference (stopped audit run) | `job-7ytui7dtiivx` | `c3c4cbc` | 2026-08-26 00:23 | 2026-08-26 00:29 | stopped | Stopped before formal completion after anchor-selection audit |
| GeoWeave inference (stopped audit run) | `job-gud9zsey8o7x` | `c3c4cbc` | 2026-08-26 00:23 | 2026-08-26 00:29 | stopped | Stopped before formal completion after anchor-selection audit |
| Pi3 inference (dense-cache partial) | `job-ruvjjo5ltydz` | `5174da3` | 2026-08-26 00:33 | 2026-08-26 00:43 | stopped | 38 valid outputs preserved before switching to camera-only caches |
| GeoWeave inference (dense-cache partial) | `job-kk6uqk9t7mch` | `5174da3` | 2026-08-26 00:33 | 2026-08-26 00:43 | stopped | 38 valid outputs preserved before switching to camera-only caches |
| Pi3 inference completion | `job-ff2zm1ohcj90` | `517ec42` | 2026-08-26 00:44 | 2026-08-26 00:45 | succeeded | File-count gate passed; later integrity scan found two interrupted caches from earlier stopped runs |
| GeoWeave inference completion | `job-tyh4f2fvtjj1` | `517ec42` | 2026-08-26 00:44 | 2026-08-26 00:45 | succeeded | File-count gate passed; later integrity scan found two interrupted caches from earlier stopped runs |
| Corrupt-cache repair | common dev GPUs 0/1 | `af815ba` | 2026-08-26 00:50 | 2026-08-26 00:52 | succeeded | Direct common-dev execution; repaired two invalid NPZ files per model |
| Pose evaluation and aggregation | common dev | `f46a747` | 2026-08-26 00:53 | 2026-08-26 00:54 | succeeded | Direct common-dev execution with `evo==1.31.1`; no AIHC job |

## Verification ledger

- Unit tests: 10 paired-sweep tests pass; evaluator and aggregator tests pass.
- `py_compile`: paired builder, generic pose evaluator, and overlap aggregator pass.
- Dataset mount: verified official ScanNet++ CFS root, DSLR images, Nerfstudio transforms, and aligned mesh.
- Balanced-anchor preflight: passed with 20 anchors, 80 rows, 19 scenes, and 20 rows per level.
- Group A identity hashes: passed across all four levels per anchor.
- Tuple completeness: 80/80 directories, 800/800 linked images, and 800/800 finite GT poses.
- Inference completeness: 80/80 valid Pi3 and 80/80 valid GeoWeave outputs; every cache contains ten finite 4x4 camera poses and ten image paths.
- Evaluation completeness: 80/80 finite ATE/RPE-t/RPE-r rows per model.
- Bootstrap reproducibility: 10,000 paired-anchor repeats with seed `20260826`; deterministic rerun passed.
- Paper integration: tracked/length main and appendix variants compile; main remains 11 pages and the appendix is 10 pages after adding both controlled sweeps.

## Execution notes

All thresholds, band boundaries, and sample-size gates are frozen by the design specification. Failures are recorded here and in persistent logs; partial bands are not averaged or substituted.

- A 64-view smoke retained no complete four-band anchor because the sparse uniform sample lacked locally supported far groups. The diagnostic was preserved at `preflight_smoke_bd64`.
- Raising the same scene to the formal 256-view sampling produced candidate counts `high=2419`, `medium=1316`, `low=924`, and `near_zero=3796`, with 125 anchors containing all four levels. The selected two-anchor smoke passed the strict balance and threshold validator.
- Protocol audit: the first balanced selector took the smallest eligible anchor in each scene, which violated the design's temporal-quantile requirement. Both first inference jobs were stopped before completion. The selection was replaced from the frozen 3,178-anchor candidate pool using per-scene temporal quantiles followed by scene round-robin allocation; no overlap band, candidate metric, or model result was used in reselection. The earlier tuple directory was preserved before rematerialization.
- Cache-integrity audit: interrupting inference left two malformed `points.npz` files per model. A file-existence-only skip allowed them through the later count gate. Commit `af815ba` adds structural/finite cache validation; direct common-dev repair recomputed only those four invalid files. A fresh full scan then verified all 160 caches.
- Controlled-sweep result: mean ATE is similar in the high/medium/low bands. At near-zero overlap, Pi3 mean ATE is 0.2769 and GeoWeave mean ATE is 0.1788, but the paired mean-difference interval crosses zero and the paired median difference is -0.0043. The mean separation is driven by one severe Pi3 failure; maximum ATE is 2.629 for Pi3 versus 0.563 for GeoWeave. The paper therefore describes failure-tail robustness rather than a consistent per-tuple improvement.
