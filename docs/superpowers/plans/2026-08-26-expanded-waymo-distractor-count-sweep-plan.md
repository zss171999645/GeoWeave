# Expanded Waymo Distractor-Count Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-independent 20-example Waymo testing expansion, run gated zero/four distractor endpoints, and continue to the full five-level 40- or 100-example sweep only when the predeclared robustness gates pass.

**Architecture:** A focused dataset builder scans the 80 Waymo2 testing camera sequences, creates 12 evenly spaced anchors per sequence, requires a resolution-matched cross-segment distractor tail to beat the clean context by the historical minimum margin, ranks eligible pairs with the historical composite score, and materializes nested `noise0`--`noise4` tuples. A separate gate summarizer joins per-example Pi3 and GeoWeave pose rows, computes clean-relative degradation, and emits a machine-readable proceed/stop decision before additional inference is launched.

**Tech Stack:** Python 3.10, NumPy, Pillow, pathlib, existing Pi3/GeoWeave inference and cached-pose evaluation tools, unittest-style repository tests, Git.

---

### Task 1: Waymo2 Testing Pair Selection

**Files:**
- Create: `tools/build_waymo_testing_distractor_protocol.py`
- Create: `tests/test_build_waymo_testing_distractor_protocol.py`

- [ ] **Step 1: Write failing tests for sequence parsing and centered anchors**

Create fixtures with two driving segments and one camera each. Assert that `parse_sequence_name()` returns segment/camera fields, `common_frame_indices()` intersects color and pose files, and `centered_stride_window()` returns exactly ten indices separated by four.

```python
def test_centered_window_uses_common_color_pose_indices(tmp_path: Path) -> None:
    seq = write_sequence(tmp_path, "segment_a_cam1", count=64)
    payload = builder.build_sequence_payload(seq, frame_stride=4)
    assert len(payload.frame_indices) == 10
    assert all(b - a == 4 for a, b in zip(payload.frame_indices, payload.frame_indices[1:]))
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python tests/test_build_waymo_testing_distractor_protocol.py`

Expected: import failure because `build_waymo_testing_distractor_protocol.py` does not exist.

- [ ] **Step 3: Implement deterministic sequence discovery and centered windows**

Implement immutable `SequencePayload` records containing sequence path, driving segment, camera, common frame indices, ten selected frame paths, ten pose rows, six prefix paths, and four clean-tail paths. Reject malformed names, missing modalities, or sequences without a stride-four ten-frame window.

```python
@dataclass(frozen=True)
class SequencePayload:
    path: Path
    segment: str
    camera: str
    frame_indices: tuple[int, ...]
    image_paths: tuple[Path, ...]
    pose_rows: tuple[str, ...]
```

- [ ] **Step 4: Run the focused tests**

Run: `python tests/test_build_waymo_testing_distractor_protocol.py`

Expected: centered-window tests pass; later selection tests still fail until Task 2.

- [ ] **Step 5: Commit sequence discovery**

```bash
git add tools/build_waymo_testing_distractor_protocol.py tests/test_build_waymo_testing_distractor_protocol.py
git commit -m "feat: discover Waymo testing anchors"
```

### Task 2: Model-Independent Distractor Selection and Materialization

**Files:**
- Modify: `tools/build_waymo_testing_distractor_protocol.py`
- Modify: `tests/test_build_waymo_testing_distractor_protocol.py`

- [ ] **Step 1: Add failing tests for cross-segment selection**

Assert that selection never uses the same segment, allows resolution-matched cross-camera candidates, rejects non-positive similarity margins, applies the historical composite score, and produces at most one selected pair per clean sequence.

```python
def test_select_tail_is_cross_segment_same_camera(tmp_path: Path) -> None:
    clean = make_payload(tmp_path, "segment_a", "1", value=32)
    wrong = make_payload(tmp_path, "segment_b", "1", value=34)
    other_camera = make_payload(tmp_path, "segment_c", "2", value=32)
    selected = builder.select_distractor_tail(clean, [clean, wrong, other_camera])
    assert selected.segment == "segment_b"
    assert selected.camera == clean.camera
```

- [ ] **Step 2: Verify the new test fails**

Run: `python tests/test_build_waymo_testing_distractor_protocol.py`

Expected: missing selection/materialization functions.

- [ ] **Step 3: Implement frozen feature selection**

Reuse the historical `32x18` RGB thumbnail feature, mean-squared prefix/tail distance, entropy threshold `4.0`, and edge-density threshold `0.02`. Record all candidates and the selected pair in `selection_manifest.json`. Rank pairs by ascending prefix-to-distractor feature distance; no model output may be imported or read.

- [ ] **Step 4: Implement five nested count levels**

For `noiseK`, preserve the six prefix images, retain the first `4-K` clean context images, and use the final `K` corresponding distractor-tail images. Link images into `color_90`, write ten pose rows to `pose_90.txt`, and include `eval_frame_indices=[0,1,2,3,4,5]` and source labels in `tuple_meta.json`.

```python
images = list(clean.image_paths[: 10 - k]) + list(wrong.image_paths[10 - k : 10])
labels = ["clean"] * (10 - k) + ["distractor"] * k
```

- [ ] **Step 5: Validate hashes and protocol shape**

Build the final protocol with `tools/reprodata_variable_distractor_protocol.py`. Tests must assert five variants, ten images, ten poses, correct label counts, and identical six-prefix SHA-256 hashes.

- [ ] **Step 6: Run tests and commit**

Run: `python tests/test_build_waymo_testing_distractor_protocol.py && python tests/test_reprodata_variable_distractor_protocol.py`

Expected: all tests print `ok` or exit zero.

```bash
git add tools/build_waymo_testing_distractor_protocol.py tests/test_build_waymo_testing_distractor_protocol.py
git commit -m "feat: build expanded Waymo distractor tuples"
```

### Task 3: Gated Pose Summary

**Files:**
- Create: `tools/evaluate_expanded_distractor_gate.py`
- Create: `tests/test_evaluate_expanded_distractor_gate.py`

- [ ] **Step 1: Write failing endpoint-gate tests**

Cover positive new/combined advantages, new-set double failure, and combined single-metric failure. The function returns a JSON-serializable payload with `decision`, `reasons`, and per-set ATE/RPE-t degradation advantages.

```python
def test_gate_stops_when_combined_rpe_advantage_is_nonpositive() -> None:
    result = gate.endpoint_gate(new_rows(), old_rows_with_combined_rpe_failure())
    assert result["decision"] == "stop"
    assert "combined_rpe" in result["reasons"]
```

- [ ] **Step 2: Verify the test fails**

Run: `python tests/test_evaluate_expanded_distractor_gate.py`

Expected: import failure because the gate module does not exist.

- [ ] **Step 3: Implement per-example aggregation and endpoint gate**

Read Pi3 and GeoWeave `pose_rows.csv`, require complete paired sample IDs at `noise0` and `noise4`, compute per-model means, clean-relative degradation, and robustness advantages. Join the verified old-20 endpoint rows to the new rows for Combined-40.

- [ ] **Step 4: Implement full-sweep area gate**

For completed five-level rows, compute the mean of degradation at `noise1` through `noise4` separately for ATE and RPE-t. Proceed to 100 only when endpoint absolute ordering, endpoint degradation advantage, and degradation-area advantage all favor GeoWeave for both metrics.

- [ ] **Step 5: Run tests and commit**

Run: `python tests/test_evaluate_expanded_distractor_gate.py`

Expected: all gate tests pass.

```bash
git add tools/evaluate_expanded_distractor_gate.py tests/test_evaluate_expanded_distractor_gate.py
git commit -m "feat: gate expanded distractor sweep"
```

### Task 4: Construct and Validate the New 20

**Files:**
- Create outputs under: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_sweep_20260826`
- Create experiment record: `experiments/expanded_waymo_distractor_sweep_20260826.md`

- [ ] **Step 1: Run builder over all 80 testing sequences**

```bash
python tools/build_waymo_testing_distractor_protocol.py \
  --waymo-root /mnt/cfs/datasets/Waymo2/testing \
  --output-root /mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_sweep_20260826/tuples \
  --selected-count 20 \
  --frame-stride 4
```

Expected: 80 candidate pairs, top 20 selected pairs, and 100 materialized variant directories.

- [ ] **Step 2: Build protocol and validate**

```bash
python tools/reprodata_variable_distractor_protocol.py \
  --root /mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_sweep_20260826/tuples/selected20 \
  --output /mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_sweep_20260826/protocol_selected20.json
```

Expected: `samples=20 variants=100`, with prefix-hash validation true.

- [ ] **Step 3: Run one-sample construction smoke and record validation**

Inspect all ten images and pose rows for one sample across `noise0` and `noise4`; verify no broken links and finite 4x4 poses.

- [ ] **Step 4: Commit experiment construction record**

```bash
git add experiments/expanded_waymo_distractor_sweep_20260826.md
git commit -m "docs: record expanded Waymo construction"
```

### Task 5: Run New-20 Endpoint Gate

**Files:**
- Predictions: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_sweep_20260826/predictions_selected20`
- Pose rows: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_sweep_20260826/pose_selected20_endpoints`
- Gate JSON: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/expanded_waymo_distractor_sweep_20260826/gate40_endpoints.json`

- [ ] **Step 1: Run Pi3 and GeoWeave endpoint inference on separate GPUs**

```bash
CUDA_VISIBLE_DEVICES=0 python tools/reprodata_pi3_protocol_infer.py --protocol "$PROTOCOL" --output-root "$PRED" --mode base --variants noise0 noise4 --load-img-size 512 --camera-only
CUDA_VISIBLE_DEVICES=1 python tools/reprodata_pi3_protocol_infer.py --protocol "$PROTOCOL" --output-root "$PRED" --mode geoweave --variants noise0 noise4 --load-img-size 512 --camera-only
```

Expected: 40 valid camera-pose caches per model.

- [ ] **Step 2: Evaluate cached poses**

```bash
python tools/eval_reprodata_precomputed_pi3_relpose.py --protocol "$PROTOCOL" --prediction-root "$PRED" --model pi3_base --output-dir "$POSE/pi3_base" --skip-plot
python tools/eval_reprodata_precomputed_pi3_relpose.py --protocol "$PROTOCOL" --prediction-root "$PRED" --model geoweave_pi3 --output-dir "$POSE/geoweave_pi3" --skip-plot
```

Expected: 40 finite rows per model, all evaluated on indices 0--5 from tuple metadata.

- [ ] **Step 3: Evaluate endpoint gate before any intermediate run**

Run `tools/evaluate_expanded_distractor_gate.py` with new-20 pose rows and the verified historical old-20 pose rows. If decision is `stop`, terminate execution and report the complete numbers to the user.

- [ ] **Step 4: Record and commit endpoint result**

```bash
git add experiments/expanded_waymo_distractor_sweep_20260826.md
git commit -m "docs: record expanded Waymo endpoint gate"
```

### Task 6: Conditional Full 40 and Optional 100

**Files:**
- Reuse Task 5 output root with separate `selected40`/`all80` protocols and summaries.

- [ ] **Step 1: If the endpoint gate passes, infer `noise1 noise2 noise3` for the new 20**

Run the same two inference commands with `--variants noise1 noise2 noise3`, then rerun cached-pose evaluation. Expected: 100 finite pose rows per model for New-20.

- [ ] **Step 2: Compute Original-20, New-20, and Combined-40 summaries**

Run the full-sweep gate. If its decision is `stop`, do not construct or infer the remaining 60 examples; report the 40-example results.

- [ ] **Step 3: If the full-sweep gate passes, materialize remaining 60 pairs**

Build an `all80` protocol using the frozen 80-pair manifest, then first run only `noise0 noise4` for the remaining 60 pairs.

- [ ] **Step 4: Apply the Combined-100 endpoint gate**

If the endpoint gate stops, do not run intermediate levels. Otherwise run `noise1 noise2 noise3`, evaluate all rows, and create final Combined-100 summaries.

- [ ] **Step 5: Verify completeness and commit final record**

Require exact sample/model/level counts, finite camera caches, finite pose rows, clean-prefix hash equality, and a clean Git worktree.

```bash
git add experiments/expanded_waymo_distractor_sweep_20260826.md
git commit -m "docs: record expanded Waymo distractor sweep"
```
