# Required Normal, Depth, and Training-Time Metrics

Date: 2026-08-24

Git branch: `revision-required-metrics-20260824`

Implementation commits:

- `f3e4919 feat: add required revision metrics`
- `0c84180 fix: align training benchmark model setup`

## Evaluation protocol

Point and normal metrics use cached Pi3/GeoWeave global point maps and GT point maps. Each tuple is aligned once with a single Sim(3). Normal consistency is estimated on the aligned sampled clouds using KNN normals. Depth metrics apply the same tuple-level Sim(3), transform the aligned predicted world points into the corresponding GT camera frame, and report AbsRel, normalized RMSE, and delta1 on valid GT-depth pixels.

Waymo GT depth is sparse LiDAR-projected depth (about 1.4% valid pixels); ScanNet++ depth is dense. These metrics complement pose ATE/RPE and must not be conflated with them.

## Weak-overlap structure metrics

| Setting | Model | Acc ↓ | Comp ↓ | NC ↑ | Depth AbsRel ↓ | Depth RMSE norm ↓ | Depth delta1 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| ScanNet++ weak, 12 tuples | Pi3 | 0.078830 | 0.085188 | 0.704217 | 0.118414 | 0.148751 | 0.866428 |
| ScanNet++ weak, 12 tuples | GeoWeave-Pi3 | 0.080231 | 0.087886 | 0.706442 | 0.114658 | 0.139922 | 0.886732 |
| Waymo weak, 90 tuples | Pi3 | 0.119668 | 0.182409 | 0.534481 | 0.465735 | 0.519079 | 0.266263 |
| Waymo weak, 90 tuples | GeoWeave-Pi3 | 0.093662 | 0.113711 | 0.579546 | 0.364003 | 0.412151 | 0.427705 |

Interpretation:

- ScanNet++ is mixed: GeoWeave is slightly worse in Acc/Comp but slightly better in NC and all reported depth metrics.
- Waymo weak is consistently positive for GeoWeave across point, normal, and depth metrics.
- The evidence does not support a universal structure-accuracy claim.

## Waymo distractor structure metrics

| Model | Variant | Acc ↓ | Comp ↓ | NC ↑ | Depth AbsRel ↓ | Depth RMSE norm ↓ | Depth delta1 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| Pi3 | Clean | 0.030547 | 0.069366 | 0.678188 | 0.241666 | 0.212626 | 0.776648 |
| Pi3 | Distractor | 0.030877 | 0.070024 | 0.676698 | 0.241658 | 0.213159 | 0.769527 |
| GeoWeave-Pi3 | Clean | 0.031418 | 0.076775 | 0.677296 | 0.245047 | 0.211095 | 0.764433 |
| GeoWeave-Pi3 | Distractor | 0.031516 | 0.076522 | 0.674312 | 0.247003 | 0.212330 | 0.758745 |

Clean-to-distractor changes:

| Model | Delta Acc ↓ | Delta Comp ↓ | Delta NC ↑ | Delta AbsRel ↓ | Delta RMSE norm ↓ | Delta delta1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Pi3 | +0.000331 | +0.000657 | -0.001490 | -0.000007 | +0.000533 | -0.007121 |
| GeoWeave-Pi3 | +0.000098 | -0.000253 | -0.002984 | +0.001956 | +0.001235 | -0.005687 |

Interpretation: both models change only slightly. GeoWeave has a smaller Acc change, improves Comp slightly, and has a smaller delta1 drop; Pi3 has smaller NC, AbsRel, and normalized-RMSE degradation. This result is mixed and should be reported as a limitation rather than a universal robustness win.

## Controlled forward/backward step-time benchmark

Both models use the same native Pi3 training class, frozen encoder, activation-checkpointing policy, output proxy loss, and bfloat16 precision on one A800. Pi3 disables the indexer; GeoWeave uses sparse Stage-2 state with `compute_loss=true` and includes `indexer_loss`. Timing covers `zero_grad + forward + loss + backward`; it excludes data loading and optimizer update. Each result uses one warmup and three measured repeats.

| Shape | Model | Mean step time | Peak allocated GPU memory | Trainable parameters |
|---|---|---:|---:|---:|
| 8 views, 518x518 | Pi3 | 1.225 s | 6.809 GiB | 587,995,224 |
| 8 views, 518x518 | GeoWeave-Pi3 | 7.981 s | 7.408 GiB | 592,756,476 |
| 10 views, 392x518 | Pi3 | 1.238 s | 6.689 GiB | 587,995,224 |
| 10 views, 392x518 | GeoWeave-Pi3 | 7.561 s | 7.255 GiB | 592,756,476 |

Overhead:

- 8-view 518x518 step latency: `6.52x`; peak allocated memory: `1.088x`.
- 10-view 392x518 step latency: `6.11x`; peak allocated memory: `1.085x`.

This controlled step benchmark is not an epoch-wall-clock measurement. It conflicts with the `1.8x per epoch` wording in the submitted rebuttal, so that wording should not be reused unless the original epoch logs are recovered and verified.

## Completeness

- ScanNet++: 24/24 rows ok.
- Waymo weak: 180/180 rows ok.
- Waymo distractor: 80/80 rows ok.
- Total: 284/284 rows ok, 0 missing, 0 failed.
- Training benchmark: both models completed all three repeats for both representative shapes.

## Result locations

- `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_required_metrics_20260824/full_scannetpp`
- `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_required_metrics_20260824/full_waymo_weak`
- `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_required_metrics_20260824/full_waymo_plausible`
- `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_required_metrics_20260824/training_fair_8f_518x518`
- `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_required_metrics_20260824/training_fair_10f_392x518`
