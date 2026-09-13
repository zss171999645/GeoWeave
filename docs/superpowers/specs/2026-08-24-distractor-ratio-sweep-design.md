# Distractor-Ratio Sweep Design

## Goal

Measure how Pi3 and GeoWeave-Pi3 change as the fraction of distractor views increases while keeping the reconstruction target, total number of input views, image positions, model checkpoints, and evaluation protocol fixed.

## Input corpus

Use the existing 20 paired Waymo tuples from the paper's distractor experiment. Every pair contains:

- one `clean` tuple with six fixed target views and four geometrically compatible context views;
- one `noise` tuple with the same first six target views and four appearance-similar cross-scene distractor views.

No new scene search or pair selection is performed.

## Controlled variants

Each generated input contains exactly ten views. Views 0--5 are the identical clean evaluation prefix at every level. Views 6--9 form the context pool. The last `k` context positions are replaced by the corresponding distractor views, producing nested distractor sets:

| Variant | Clean context views | Distractor context views | Context distractor ratio | Total-input distractor ratio |
|---|---:|---:|---:|---:|
| `noise0` | 4 | 0 | 0% | 0% |
| `noise1` | 3 | 1 | 25% | 10% |
| `noise2` | 2 | 2 | 50% | 20% |
| `noise3` | 1 | 3 | 75% | 30% |
| `noise4` | 0 | 4 | 100% | 40% |

The order is deterministic: `noise1` replaces view 9, `noise2` replaces views 8--9, and so on. Images are materialized as symlinks on CFS; source data are not modified.

## Implementation

Port the existing reference `build_variable_distractor_from_pairs.py` from the archived `meshx_bundle_inspect_20260701` repository into the active revision worktree. The active repository already contains its real-file unit tests, so implementation follows their expected API.

Create a protocol JSON compatible with `reprodata_pi3_protocol_infer.py`. Each sample exposes variants `noise0` through `noise4`, and every variant points to its generated fixed-ten-view directory. Validate that:

- every variant has ten images and ten pose rows;
- views 0--5 have identical file content across all five variants;
- the number and positions of `distractor` labels equal `k`;
- every tuple records evaluation indices `[0,1,2,3,4,5]`.

## Model execution

Run the paper checkpoints for:

- `pi3_base`;
- `geoweave_pi3`.

Use the same image loading width (`518`), native global point output, and ten-view inference path as the existing distractor experiment. Run one model per GPU on common-dev, using at most two A800 GPUs. Expected workload: `20 samples x 5 levels x 2 models = 200` bounded inference jobs.

## Evaluation

Always evaluate only views 0--5.

### Pose metrics

Report ATE, translation RPE, and rotation RPE after the existing tuple-level alignment/evaluation procedure. For each model and level, report the absolute metric and change relative to `noise0`.

### Structure metrics

Reuse the verified required-metrics evaluator and the same GT subset to report:

- point accuracy and completeness after one tuple-level Sim(3);
- normal consistency;
- depth AbsRel, normalized RMSE, and delta1 after applying the same Sim(3) and transforming points into GT camera coordinates.

Waymo depth remains sparse LiDAR-projected GT, so the report must include its valid-pixel ratio and treat depth/normal evidence as complementary to pose metrics.

## Result presentation

Produce:

- per-job CSV/JSON records;
- aggregate tables by model and distractor count;
- clean-relative degradation tables;
- a compact line plot for ATE, point Acc/Comp, NC, and depth metrics versus context distractor ratio;
- a concise interpretation that explicitly reports mixed trends rather than selecting only favorable metrics.

## Failure handling

- Use a new output root and refuse accidental duplicate writers.
- Run a one-pair, five-level construction and inference smoke before the full run.
- Any missing image, broken symlink, non-finite output, failed metric row, or inconsistent prefix blocks the full result from being called complete.
- If a model fails at one level, retain the failure record; do not silently drop that level.

## Acceptance criteria

- Construction validation passes for all `20 x 5 = 100` input tuples.
- Pi3 and GeoWeave each produce 100 finite outputs.
- Pose and structure evaluations contain all expected model/level rows with zero missing jobs.
- Result summaries and commands are stored under `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_distractor_ratio_20260824`.
- The implementation, tests, design, experiment record, and report are committed on `revision-required-metrics-20260824`.
