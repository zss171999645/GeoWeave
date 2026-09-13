# Official VGGT Review Guide

This folder contains the official VGGT implementation and the indexer-based sparse attention used in
large-scene training. Use this guide to focus review on the most important logic and verify correctness.

## Scope
- Primary focus: indexer + sparse attention logic and the training flow around it.
- Non-core edits outside `easyvolcap/official_vggt/` can be ignored unless explicitly referenced.

## Key Entry Points
- `easyvolcap/official_vggt/layers/indexer.py`
  - Lightning indexer implementation and score dtype handling.
- `easyvolcap/official_vggt/layers/dsa_attention.py`
  - Dense vs sparse attention switching; top-k selection; indexer loss path.
- `easyvolcap/official_vggt/models/aggregator.py`
  - Indexer state scheduling (warmup / sparse / loss settings).
- `easyvolcap/official_vggt/models/vggt.py`
  - Model assembly and head wiring.

## Configuration Anchors
These configs commonly drive the indexer behavior:
- `configs/exps/vggt/vggt_official_finetune*.yaml`
- `configs/specs/vggt/official/*.yaml`

Common knobs (naming may vary per config):
- `indexer_cfg.enabled`, `indexer_cfg.sparse`, `indexer_cfg.topk`
- `indexer_cfg.warmup_steps`, `indexer_cfg.enable_sparse`
- `indexer_cfg.compute_loss`, `indexer_cfg.loss_weight`
- `indexer_cfg.force_keep_special_tokens`, `indexer_cfg.force_keep_self_view_tokens`

## Review Checklist (Suggested)
1. Sparse attention mask construction is correct for top-k selection.
2. Indexer loss is computed on the intended branch and respects dtype conversions.
3. Warmup-to-sparse transition is aligned with configured step counts.
4. Shape and device consistency across `q/k/v` and mask paths.
5. No silent no-op paths for loss when `sparse_use_mask` is enabled.

## Notes
- This document is intentionally short and review-focused.
- If you need a runnable minimal example, ask and I will add one.
