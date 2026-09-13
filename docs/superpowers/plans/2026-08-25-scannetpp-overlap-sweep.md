# ScanNet++ Paired Overlap Sweep Implementation Plan

> **Execution:** Use the executing-plans workflow and implement each behavior test-first. The existing isolated branch and worktree are reused.

**Goal:** Build and evaluate a balanced ScanNet++ 5+5 protocol with fixed Group A and four controlled cross-group overlap levels, then produce verified ATE/RPE-t curves and paper-ready artifacts for the SIGGRAPH Asia revision.

**Architecture:** Extend the existing same-scene 5+5 tuple builder with a separate paired-sweep selector. A CPU preflight computes GT-depth overlap tables and emits a selection manifest without copying images. Materialization consumes only that frozen manifest, verifies byte-identical Group A inputs across levels, and writes 80 Pi3-style tuples. Existing Pi3/GeoWeave inference and tuple-level Sim(3) pose evaluation are reused unchanged. A deterministic aggregator computes paired bootstrap intervals and exports CSV/JSON/PDF/PNG artifacts.

**Tech stack:** Python 3.10, NumPy, Pillow, existing EVC/ScanNet++ camera and overlap utilities, existing Pi3/GeoWeave CUDA inference, AIHC jobs, unittest-style tests, Matplotlib.

---

### Task 1: Freeze the workspace and execution record

**Files:**
- Create: `docs/superpowers/plans/2026-08-25-scannetpp-overlap-sweep.md`
- Create: `experiments/scannetpp_overlap_sweep_20260825.md`

- [ ] Verify the active branch is `revision-required-metrics-20260824`, the worktree is clean, and the design commit is present.
- [ ] Record dataset path, model checkpoints, intended output roots, queue/resource pool, and every submitted job ID in the experiment record.
- [ ] Commit the plan and empty execution record with `docs: plan ScanNet++ overlap sweep`.

### Task 2: Implement paired four-band selection test-first

**Files:**
- Create: `aidi/scripts/baselines/build_scannetpp_paired_overlap_sweep.py`
- Create: `tests/scannetpp_paired_overlap_sweep_tests.py`

- [ ] Write a failing test for half-open band classification using `high=[0.10,0.20)`, `medium=[0.03,0.10)`, `low=[0.005,0.03)`, and `near_zero=[0,0.005)`.
- [ ] Write a failing test showing that one fixed Group A selects exactly one Group B per band by midpoint distance, then internal overlap, then frame ID.
- [ ] Write a failing test that rejects an anchor missing any band and preserves identical Group A IDs across all retained levels.
- [ ] Run the test and confirm it fails because the module does not exist.
- [ ] Implement only the pure selection functions needed by the tests; rerun until green.

### Task 3: Add preflight manifests and strict validation

**Files:**
- Modify: `aidi/scripts/baselines/build_scannetpp_paired_overlap_sweep.py`
- Modify: `tests/scannetpp_paired_overlap_sweep_tests.py`

- [ ] Add failing tests for deterministic per-scene caps, balanced anchor counts, ten unique views, no A/B duplication, local-overlap threshold, and in-band cross-group means.
- [ ] Implement `--mode preflight` to scan configured scenes, build GT-depth overlap tables, select balanced anchors, and write `candidate_rows.csv`, `selection_manifest.json`, `preflight_summary.json`, and `skipped.csv` without materializing images.
- [ ] Enforce the hard minimum of 20 fully paired anchors. Return a nonzero exit code and retain diagnostics when the minimum is not met.
- [ ] Add `--allow-insufficient-for-smoke` only for one-scene development checks; record it explicitly and never permit it in the formal manifest.
- [ ] Run unit tests and `py_compile`.

### Task 4: Prove the AIHC dataset mount and run preflight

**Files:**
- Create: `jobs/scannetpp_overlap_preflight_20260825.json`
- Update: `experiments/scannetpp_overlap_sweep_20260825.md`
- Output: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/scannetpp_overlap_sweep_20260825/preflight/`
- Log: `/mnt/cfs/train_logs/zhoufeng/scannetpp_overlap_preflight_20260825.log`

- [ ] Inspect accessible AIHC job specifications and local mount records to identify the BOS datasource that exposes `/horizon-bucket/saturn_v_4dlabel`.
- [ ] Submit a CPU-only read-only mount probe; require the configured ScanNet++ root and representative image/depth/camera files to exist.
- [ ] Run a one-scene smoke preflight and inspect band counts and overlap distributions.
- [ ] Run the formal multi-scene preflight with fixed bands and scene cap. Require at least 20 anchors with all four bands; otherwise stop before inference and preserve per-band diagnostics.
- [ ] Copy preflight summaries into the revision artifact directory and commit code, tests, job spec, and experiment record with `feat: add paired overlap sweep builder`.

### Task 5: Materialize and validate the frozen 80-tuple benchmark

**Files:**
- Modify: `aidi/scripts/baselines/build_scannetpp_paired_overlap_sweep.py`
- Modify: `tests/scannetpp_paired_overlap_sweep_tests.py`
- Create: `jobs/scannetpp_overlap_materialize_20260825.json`
- Output: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/scannetpp_overlap_sweep_20260825/tuples/`

- [ ] Add a failing test for materialization from a frozen manifest, including stable tuple names and Group A file hashes.
- [ ] Implement `--mode materialize` without reselection. Refuse existing output unless `--overwrite` is given; when overwriting, preserve the prior manifest.
- [ ] Materialize exactly four levels per retained anchor and verify 10 images, 10 GT poses, tuple metadata, and matching Group A SHA-256 hashes across levels.
- [ ] Write `materialization_validation.json`; require 80/80 valid tuples for the target sample.

### Task 6: Run bounded inference and pose evaluation

**Files:**
- Create: `jobs/scannetpp_overlap_inference_pi3_20260825.json`
- Create: `jobs/scannetpp_overlap_inference_geoweave_20260825.json`
- Update: `experiments/scannetpp_overlap_sweep_20260825.md`
- Output: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/scannetpp_overlap_sweep_20260825/predictions/`

- [ ] Run one paired anchor across all four levels for Pi3 and GeoWeave. Require finite camera poses/points and eight successful outputs.
- [ ] Run full inference with at most two GPUs, one model per GPU, using the same input resolution and checkpoints as the accepted-paper/rebuttal protocol.
- [ ] Reuse the existing tuple-level Sim(3) evaluator for all ten views and export per-tuple ATE, RPE-t, and RPE-r.
- [ ] Require 80/80 inference outputs per model and 160/160 finite evaluation rows; do not aggregate partial results.
- [ ] Compare the low-overlap operating point against the archived 12-tuple result and document expected sampling differences.

### Task 7: Aggregate paired curves and confidence intervals test-first

**Files:**
- Create: `tools/summarize_scannetpp_overlap_sweep.py`
- Create: `tests/test_summarize_scannetpp_overlap_sweep.py`
- Output: `/mnt/cfs/zhoufeng/siggraph26_rebuttal/outputs/scannetpp_overlap_sweep_20260825/analysis/`

- [ ] Write failing tests for paired row completeness, measured-overlap x coordinates, `Delta ATE = ATE(Pi3) - ATE(GeoWeave)`, deterministic anchor bootstrap, and zero-line handling.
- [ ] Implement the aggregator and produce `per_anchor_metrics.csv`, `band_summary.csv`, `summary.json`, `ate_vs_overlap.{png,pdf}`, and `delta_ate_vs_overlap.{png,pdf}`.
- [ ] Bootstrap anchors, not individual model rows, with a fixed seed and 95% confidence intervals.
- [ ] Verify four ordered levels, equal anchor counts, finite aggregates, and deterministic reruns.
- [ ] Commit with `feat: summarize paired overlap sweep`.

### Task 8: Integrate verified results into both manuscript variants

**Files:**
- Modify: local tracked main/appendix TeX variants
- Modify: local length main/appendix TeX variants
- Create: local experiment artifact copy under `anonymous-revision-20260824/experiment_results/scannetpp_overlap_sweep_20260825/`

- [ ] Copy frozen CSV/JSON/figures/log summaries from CFS into the revision directory.
- [ ] Draft one concise main-text trend sentence and place the full protocol, ATE/RPE-t table, and curve in the appendix. Do not claim monotonic improvement unless the verified confidence intervals support it.
- [ ] Apply identical added content in blue to tracked and length variants; tracked deletions remain struck through.
- [ ] Compile all four PDFs, inspect affected pages visually, check references, and record page counts/overfull boxes.
- [ ] Run the experiment completeness checklist, repository tests, git diff/status, and PDF build verification before reporting completion.
