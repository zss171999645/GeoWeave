# Distractor-Ratio Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fixed-ten-view Waymo distractor-ratio sweep for Pi3 and GeoWeave-Pi3 at 0/25/50/75/100% context distractor ratios.

**Architecture:** Materialize nested fixed-length input variants from the existing clean/noise pairs, convert them into the repository's protocol JSON, and run one combined Pi3 forward per model/sample/level that stores both point maps and camera poses. Evaluate fixed-prefix pose and verified point/normal/depth metrics from those cached outputs, then aggregate clean-relative trends and a paper-ready plot.

**Tech Stack:** Python 3.10, pathlib/NumPy, PyTorch CUDA, evo, Open3D, Matplotlib, two A800 GPUs, CFS storage.

**Execution status:** Complete on 2026-08-24. All implementation, smoke, full inference, evaluation, report-generation, and artifact-copy steps were completed; see `experiments/distractor_ratio_sweep_20260824.md`.

---

### Task 1: Port and validate fixed-ten-view distractor construction

**Files:**
- Create: `aidi/scripts/baselines/build_variable_distractor_from_pairs.py`
- Modify: `tests/variable_distractor_from_pairs_tests.py`

- [x] **Step 1: Run the existing test to verify RED**

Run:
```bash
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/variable_distractor_from_pairs_tests.py
```
Expected: import failure because `build_variable_distractor_from_pairs.py` is absent.

- [x] **Step 2: Add tests for the live clean/noise suffixes and metadata preservation**

Add a paired fixture named `sample__clean_tail` / `sample__plausible_noise_tail` and assert:

```python
pairs = paired_clean_noise(root)
assert pairs == [(clean, noise)]
assert result_meta["prefix_scene"] == "scene-a"
assert result_meta["source_labels"] == ["clean"] * 8 + ["distractor"] * 2
```

- [x] **Step 3: Port the reference builder and support both suffix conventions**

Implement deterministic suffix parsing for:

```python
PAIR_SUFFIXES = (
    ("__clean", "__noise"),
    ("__clean_tail", "__plausible_noise_tail"),
)
```

For every variant, start metadata from `clean_meta`, then append protocol, source-pair, distractor-count, source-label, and eval-index fields. Preserve the reference fixed-length `tail_clean` behavior.

- [x] **Step 4: Verify GREEN**

Run the full test and expect five passing unittest cases.

- [x] **Step 5: Commit**

```bash
git add aidi/scripts/baselines/build_variable_distractor_from_pairs.py tests/variable_distractor_from_pairs_tests.py
git commit -m "feat: add fixed-view distractor sweep builder"
```

### Task 2: Build a five-variant protocol and validate the fixed prefix

**Files:**
- Create: `tools/reprodata_variable_distractor_protocol.py`
- Create: `tests/test_reprodata_variable_distractor_protocol.py`

- [x] **Step 1: Write failing protocol tests**

Construct five temporary variant roots and assert that `build_protocol` returns one sample with ordered variants:

```python
assert protocol["variants"] == ["noise0", "noise1", "noise2", "noise3", "noise4"]
assert protocol["samples"][0]["prefix_size"] == 6
assert set(protocol["samples"][0]["variants"]) == set(protocol["variants"])
```

Also alter one prefix file and assert `validate_fixed_prefix` raises `RuntimeError`.

- [x] **Step 2: Verify RED**

Expected: module import failure.

- [x] **Step 3: Implement protocol discovery and byte-level prefix validation**

Group directories by removing `__noiseK`; read `tuple_meta.json`; validate ten images, ten pose rows, expected source labels, and SHA-256 equality for views 0--5 across all variants. Emit `input_dir`, `sequence_dir`, `tuple_meta_path`, and per-frame roles.

- [x] **Step 4: Verify GREEN and commit**

```bash
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_reprodata_variable_distractor_protocol.py
git add tools/reprodata_variable_distractor_protocol.py tests/test_reprodata_variable_distractor_protocol.py
git commit -m "feat: add distractor sweep protocol"
```

### Task 3: Cache point maps and camera poses in one inference pass

**Files:**
- Modify: `aidi/scripts/baselines/eval_pi3_mv_recon_core.py`
- Modify: `tools/reprodata_pi3_protocol_infer.py`
- Modify: `tools/test_reprodata_pi3_protocol_infer.py`

- [x] **Step 1: Write failing extraction test**

Use real torch tensors in a prediction dictionary and assert:

```python
result = extract_pi3_points_and_poses(pred, data_size=(4, 5), point_source="native")
assert result["points"].shape == (2, 4, 5, 3)
assert result["camera_poses"].shape == (2, 4, 4)
```

- [x] **Step 2: Verify RED**

Expected: missing extraction helper.

- [x] **Step 3: Refactor inference without changing existing callers**

Add `infer_pi3_mv_predictions(...) -> dict[str, np.ndarray]`. Keep `infer_pi3_mv_pointclouds(...)` as a compatibility wrapper returning only `result["points"]`. Save `camera_poses` beside `points`, `colors`, and `image_paths` in every protocol `points.npz`.

- [x] **Step 4: Verify GREEN and regression tests**

```bash
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tools/test_reprodata_pi3_protocol_infer.py
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_reprodata_runtime_probe.py
```

- [x] **Step 5: Commit**

```bash
git add aidi/scripts/baselines/eval_pi3_mv_recon_core.py tools/reprodata_pi3_protocol_infer.py tools/test_reprodata_pi3_protocol_infer.py
git commit -m "feat: cache Pi3 poses with point outputs"
```

### Task 4: Evaluate cached camera poses across all ratios

**Files:**
- Create: `tools/eval_reprodata_precomputed_pi3_relpose.py`
- Create: `tests/test_eval_reprodata_precomputed_pi3_relpose.py`

- [x] **Step 1: Write failing tests**

Write real NPZ/JSON fixtures and assert that the evaluator loads `camera_poses`, respects `[0,1,2,3,4,5]`, groups `noise0..noise4`, and computes per-sample deltas relative to `noise0`.

- [x] **Step 2: Verify RED**

Expected: module import failure.

- [x] **Step 3: Implement precomputed evaluation**

Reuse the existing evo runtime helpers and GT pose loading. Write per-job CSV/JSON, aggregate metrics for ATE/RPE-t/RPE-r, and clean-relative deltas for every model and level.

- [x] **Step 4: Verify GREEN and commit**

```bash
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_eval_reprodata_precomputed_pi3_relpose.py
git add tools/eval_reprodata_precomputed_pi3_relpose.py tests/test_eval_reprodata_precomputed_pi3_relpose.py
git commit -m "feat: evaluate cached Pi3 pose sweeps"
```

### Task 5: Aggregate structure/pose trends and render the plot

**Files:**
- Create: `tools/summarize_distractor_ratio_sweep.py`
- Create: `tests/test_summarize_distractor_ratio_sweep.py`

- [x] **Step 1: Write failing summary tests**

Provide fixture aggregates for `noise0`, `noise2`, and `noise4`, then assert ratio parsing and clean-relative degradation signs for lower-is-better and higher-is-better metrics.

- [x] **Step 2: Verify RED**

Expected: module import failure.

- [x] **Step 3: Implement report generation**

Join pose and structure aggregates by model/variant. Emit CSV, JSON, Markdown tables, and a Matplotlib PNG with x-axis `[0,25,50,75,100]`. Include absolute values and changes relative to `noise0`; never invert or suppress mixed trends.

- [x] **Step 4: Verify GREEN and commit**

```bash
/mnt/cfs/zhoufeng/env/geoweave-rebuttal/bin/python tests/test_summarize_distractor_ratio_sweep.py
git add tools/summarize_distractor_ratio_sweep.py tests/test_summarize_distractor_ratio_sweep.py
git commit -m "feat: summarize distractor ratio trends"
```

### Task 6: Run smoke, full inference, evaluation, and verification

**Files:**
- Create: `experiments/distractor_ratio_sweep_20260824.md`
- Create output root: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/revision_distractor_ratio_20260824`
- Create log: `/mnt/cfs/train_logs/zhoufeng/geoweave_distractor_ratio_2026-08-24.log`

- [x] **Step 1: Register and commit the experiment before GPU execution**

Record Git SHA, one-GPU-per-model allocation, 20 pairs, 100 jobs per model, commands, inputs, output root, and persistent logs. Commit with:

```bash
git add experiments/distractor_ratio_sweep_20260824.md docs/superpowers/plans/2026-08-24-distractor-ratio-sweep.md
git commit -m "docs: register distractor ratio sweep"
```

- [x] **Step 2: Construct and validate all inputs**

Run fixed-ten-view `tail_clean` construction for variants `0,1,2,3,4`; build the protocol; require 20 samples, 100 variants, zero skipped inputs, and successful prefix hashes.

- [x] **Step 3: Run one-sample GPU smoke**

Run all five levels for both models. Require ten finite `points.npz` files containing `points`, `camera_poses`, `colors`, and `image_paths`.

- [x] **Step 4: Run full two-GPU inference**

Assign Pi3 to GPU 0 and GeoWeave to GPU 1. Each process writes only its own model subtree. Require 100/100 finite outputs per model.

- [x] **Step 5: Run pose and structure evaluation**

Evaluate all 200 cached outputs on views 0--5. Structure evaluation uses `--normal-metrics --depth-metrics`, a Waymo metric stride of 1, and 60,000 points.

- [x] **Step 6: Generate final report and verify**

Require zero missing/failed jobs, five ordered levels per model, finite aggregate metrics, a terminal log entry, clean git status, passing relevant tests, and copied summaries under the local anonymous-revision directory.
