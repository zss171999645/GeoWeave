# Required Normal, Depth, and Training-Time Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce verified normal-consistency, depth, and training-step overhead results for the three SIGGRAPH Asia 2026 robustness settings without rerunning cached inference.

**Architecture:** Extend the existing GT-backed evaluator so one Sim(3) alignment per tuple drives point, normal, and camera-depth metrics from cached global point maps. Add a separate synthetic forward/backward benchmark for Pi3 and GeoWeave-Pi3 using identical tensors and a differentiable output proxy loss. Keep all experiment outputs and logs on CFS.

**Tech Stack:** Python 3.10, NumPy, SciPy/Open3D, PyTorch CUDA, pytest-style function tests, Baidu common-dev A800.

---

### Task 1: Preserve the baseline and isolated workspace

**Files:**
- Create: `docs/superpowers/plans/2026-08-24-required-normal-depth-training-metrics.md`
- Create: `experiments/revision_required_metrics_20260824.md`

- [x] **Step 1: Create branch and worktree**

Run:
```bash
git worktree add /mnt/cfs/zhoufeng/siggraph26_rebuttal/worktrees/revision-required-metrics-20260824 -b revision-required-metrics-20260824
```

- [x] **Step 2: Verify relevant baseline tests**

Run:
```bash
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_reprodata_gt_point_accuracy.py
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_reprodata_runtime_probe.py
```
Expected: both print `ok`.

### Task 2: Add normal metrics to GT-backed summaries

**Files:**
- Modify: `tools/reprodata_gt_point_accuracy.py`
- Modify: `tests/test_reprodata_gt_point_accuracy.py`

- [x] **Step 1: Write failing aggregate test**

Add a test whose successful row contains `nc_mean=0.9` and assert that `summarize_rows` returns the same aggregate value.

- [x] **Step 2: Verify RED**

Run:
```bash
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_reprodata_gt_point_accuracy.py
```
Expected: failure because `nc_mean` is absent from the aggregate.

- [x] **Step 3: Add normal fields to JSON and Markdown summaries**

Aggregate `nc_acc_mean`, `nc_comp_mean`, and `nc_mean`, and add `nc_mean` to the Markdown table. Preserve `nan` when normals cannot be computed.

- [x] **Step 4: Verify GREEN**

Run the same test and expect `ok`.

### Task 3: Add depth metrics after the tuple-level Sim(3)

**Files:**
- Modify: `tools/reprodata_gt_point_accuracy.py`
- Modify: `tests/test_reprodata_gt_point_accuracy.py`

- [x] **Step 1: Write failing geometry tests**

Add tests for:
```python
depth = world_points_to_camera_depth(points_world, c2w)
metrics = compute_aligned_depth_metrics(pred, gt, gt_c2w, valid)
```
The first test checks a translated camera. The second constructs non-degenerate corresponding point maps related by a known Sim(3) and expects `abs_rel < 1e-6`, `rmse < 1e-6`, and `delta1 == 1.0`.

- [x] **Step 2: Verify RED**

Run the evaluator tests and expect an attribute error for the new helper.

- [x] **Step 3: Implement camera-depth evaluation**

Return the GT `c2w` stack from both dataset builders. Estimate one Sim(3) from sampled valid paired points, align the full predicted point maps, transform aligned points into each GT camera frame, and report:

```text
depth_abs_rel
depth_rmse
depth_rmse_norm
depth_delta1
depth_valid_pixels
```

Expose the behavior under `--depth-metrics`, aggregate the fields, and render them in the summary.

- [x] **Step 4: Verify GREEN and regression suite**

Run:
```bash
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_reprodata_gt_point_accuracy.py
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_reprodata_runtime_probe.py
```

### Task 4: Add a real forward/backward timing benchmark

**Files:**
- Create: `tools/synthetic_pi3_training_time_benchmark.py`
- Create: `tests/test_synthetic_pi3_training_time_benchmark.py`

- [x] **Step 1: Write failing proxy-loss tests**

Test that `proxy_training_loss` selects floating tensors from Pi3 output dictionaries, returns a scalar requiring gradients, and produces nonzero gradients after `backward()`.

- [x] **Step 2: Verify RED**

Run:
```bash
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_synthetic_pi3_training_time_benchmark.py
```
Expected: import failure because the benchmark module does not exist.

- [x] **Step 3: Implement benchmark**

Load the same checkpoints as `synthetic_pi3_runtime_benchmark.py`, use `model.train()`, identical synthetic tensors, bfloat16 autocast, and measure `zero_grad + forward + proxy loss + backward` with CUDA events. Record each repeat, mean/median latency, peak allocated/reserved memory, model/checkpoint, tensor shape, and benchmark scope. Do not perform optimizer steps.

- [x] **Step 4: Verify GREEN**

Run the unit test and `py_compile` for both modified tools.

### Task 5: Commit and execute real measurements

**Files:**
- Update: `experiments/revision_required_metrics_20260824.md`
- Create outputs under: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_required_metrics_20260824`
- Create log: `/mnt/cfs/train_logs/zhoufeng/geoweave_revision_required_metrics_2026-08-24.log`

- [x] **Step 1: Commit tested code**

Run:
```bash
git add docs/superpowers/plans/2026-08-24-required-normal-depth-training-metrics.md experiments/revision_required_metrics_20260824.md tools/reprodata_gt_point_accuracy.py tests/test_reprodata_gt_point_accuracy.py tools/synthetic_pi3_training_time_benchmark.py tests/test_synthetic_pi3_training_time_benchmark.py
git commit -m "feat: add required revision metrics"
```

- [x] **Step 2: Run one-sample normal/depth smoke per setting**

Use the archived protocols, cached outputs, and GT root with `--sample-limit 1 --normal-metrics --depth-metrics`. Require all Pi3/GeoWeave rows to be `ok` and all requested fields finite.

- [x] **Step 3: Run full cached normal/depth evaluation**

Run ScanNet++ weak (12 tuples), Waymo weak (90 tuples), and Waymo plausible prefix (20 tuples x 2 variants) for Pi3 and GeoWeave. Dense-teacher is optional for the distractor appendix but is not required for the primary Pi3/GeoWeave table.

- [x] **Step 4: Run bounded A800 training-time benchmark**

Run a 2-frame 224x224 smoke, then a representative 10-frame 392x518 benchmark if both models fit. Use one GPU, one warmup, and at least three measured repeats.

- [x] **Step 5: Verify results and update record**

Check result row counts, zero failures, finite metrics, git SHA, GPU model, commands, and terminal log lines. Record any OOM or metric limitation explicitly rather than omitting it.
