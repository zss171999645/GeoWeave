# ScanNet++ Paired Cross-Group Overlap Sweep

Date: 2026-08-25

## Goal

Measure how Pi3 and GeoWeave-Pi3 behave as the geometric overlap between two local five-view groups decreases, while keeping the first group fixed for every paired comparison. The primary paper-facing outputs are ATE-versus-overlap and paired improvement curves.

## Existing Assets

- Full official ScanNet++ root available on shared CFS and inside AIHC jobs: `/mnt/cfs/datasets/scannetpp/scannetpplus/Scannetpp/data`.
- The sweep uses official DSLR images and Nerfstudio poses. Low-resolution virtual pinhole depth maps are ray-cast from each scene's aligned mesh solely for overlap measurement; the calibrated OpenGL poses are converted to OpenCV camera axes before rendering.
- Existing tuple builder: `aidi/scripts/baselines/build_same_scene_low_overlap_5plus5_benchmark.py`.
- Existing overlap-band utilities and tests: `aidi/scripts/baselines/build_fixed10_overlap_band_benchmark.py` and `tests/fixed10_overlap_band_tests.py`.
- Existing 12-tuple protocol is retained only as a reproduction anchor. Its overlap range is too narrow for the requested sweep.
- Pi3 and GeoWeave-Pi3 checkpoints, inference, and pose evaluators are the same verified versions used for the revision experiments.

## Experimental Unit

Each experimental unit is one ScanNet++ scene anchor with:

- a fixed Group A containing five locally overlapping views;
- four Group B variants from the same scene;
- five locally overlapping views in every Group B;
- no repeated frame between Group A and Group B;
- ten total input views for every variant;
- all ten views evaluated after one tuple-level Sim(3) alignment.

The Group A frame IDs and bytes must be identical across all four variants of an anchor. Only Group B may change.

## Overlap Levels

Cross-group overlap is the mean of the 25 bidirectional GT-depth visibility values between the five Group A and five Group B views. The initial fixed bands reuse the repository's existing overlap convention:

| Level | Cross-group mean overlap |
|---|---:|
| high | `[0.10, 0.20)` |
| medium | `[0.03, 0.10)` |
| low | `[0.005, 0.03)` |
| near-zero | `[0.00, 0.005)` |

The preflight must report the full candidate distribution before any inference. Band boundaries remain fixed if at least 20 anchors have one valid Group B in every band. If this criterion is not met, no bands are silently changed: the preflight stops and reports candidate counts so the design can be revised explicitly.

## Tuple Selection

For each scene:

1. Build the GT-depth visibility overlap table with the existing bidirectional overlap implementation.
2. Select candidate Group A anchors at deterministic temporal quantiles.
3. Form Group A from the anchor and its four highest-overlap local supports, requiring anchor-to-support overlap of at least `0.08`.
4. Enumerate non-overlapping Group B anchors and construct each five-view local group with the same `0.08` within-group threshold.
5. Compute cross-group mean, maximum, minimum, and anchor-pair overlap.
6. Assign Group B to exactly one fixed overlap band.
7. For every Group A, retain at most one Group B per band. Within a band, prefer the candidate closest to the band midpoint; break ties by higher internal overlap, then frame ID.
8. Retain the Group A only if all four levels are present.

Selection uses no Pi3 or GeoWeave predictions.

## Sample Size and Balance

- Target: 20 paired Group A anchors, yielding 80 ten-view tuples.
- Minimum: 20 anchors with all four levels; otherwise stop after preflight.
- Use multiple scenes and cap anchors per scene so that one scene cannot dominate the curve.
- Every reported band has exactly the same anchors and sample count.

## Data and Execution Flow

1. The CPU development machine prepares code, job configuration, persistent output paths, and logs.
2. An AIHC CPU preflight job mounts shared CFS and verifies the official ScanNet++ images, transforms, and aligned meshes.
3. The preflight scans candidate scenes, computes overlap distributions, and writes JSON/CSV summaries without materializing images or using GPUs.
4. After the preflight passes, a materialization job creates the 80 tuple directories under `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/` and validates Group A identity plus band membership.
5. A bounded common-dev smoke runs one anchor across all four levels for both models.
6. Full inference uses at most two GPUs, one model per GPU.
7. Pose evaluation reuses the existing tuple-level Sim(3) ATE/RPE evaluator.

All formal jobs write persistent logs under `/mnt/cfs/train_logs/zhoufeng/` and record the code commit, checkpoint paths, job IDs, start/end times, counts, and failures.

## Metrics and Figures

Primary metrics:

- ATE for Pi3 and GeoWeave-Pi3 at each overlap level;
- paired improvement `Delta ATE = ATE(Pi3) - ATE(GeoWeave-Pi3)` for each anchor and level.

Secondary metric:

- RPE-t, reported in the appendix table rather than the main curve.

Figures:

1. Mean ATE versus measured cross-group overlap for both models, with paired-bootstrap 95% confidence intervals.
2. Mean paired Delta ATE versus measured cross-group overlap, with paired-bootstrap 95% confidence intervals and a zero reference line.

The x-coordinate for each band is the mean measured cross-group overlap of the retained tuples, not an arbitrary categorical index. Per-anchor rows and band summaries are exported as CSV and JSON.

## Validation Gates

Preflight must establish:

- at least 20 anchors with all four bands;
- identical Group A frame IDs and file hashes across levels;
- five unique Group B frames per level and no Group A/Group B duplication;
- within-group overlap threshold satisfied for both groups;
- cross-group mean overlap inside the declared band;
- exactly ten images and ten GT poses per tuple;
- multiple scenes represented with the configured per-scene cap.

Inference/evaluation must establish:

- one-anchor, eight-job smoke passes with finite points and camera poses;
- 80/80 outputs per model in the full run;
- 160/160 pose rows complete, with zero missing or failed rows;
- four ordered levels per anchor and model;
- finite aggregate metrics and confidence intervals;
- reproduction of the existing slight-overlap operating point within expected checkpoint/runtime tolerance.

## Failure Handling

- Missing CFS mount or official scene source: fail the AIHC preflight with the unresolved path; do not fall back to the 12 exported tuples.
- Insufficient balanced anchors: stop before materialization and report counts by scene and band.
- Empty or invalid GT depth: exclude the candidate before band assignment and record the reason.
- Missing inference output or non-finite metric: fail the result-completeness gate; do not average partial bands.
- Existing output directories are never overwritten without an explicit `--overwrite` run and a preserved manifest.

## Paper-Facing Scope

The experiment is a controlled analysis of overlap level, not a new benchmark. The main paper will summarize the ATE and Delta ATE trend and point to complete RPE-t and protocol details in the supplementary appendix. No manuscript claim is changed until the verified curve is available.
