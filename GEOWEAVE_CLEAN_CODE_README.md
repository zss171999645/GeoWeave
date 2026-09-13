# GeoWeave Clean Code Package

This directory is a clean meshx-shaped code package for GeoWeave handoff.
It keeps the familiar meshx core layout and removes local caches, temporary
experiments, personal agent notes, generic root-level utility scripts, and
historical one-off wrappers. Under `aidi/scripts`, only GeoWeave paper training,
unified evaluation, paper robustness protocols, and their direct runtime
dependencies are retained.

Primary entry points:

- `release/geoweave_paper_handoff/scripts/train_vggt.sh`
- `release/geoweave_paper_handoff/scripts/train_pi3.sh`
- `release/geoweave_paper_handoff/scripts/eval_vggt.sh`
- `release/geoweave_paper_handoff/scripts/eval_pi3.sh`
- `release/geoweave_paper_handoff/scripts/robustness_pi3.sh`
- `release/geoweave_paper_handoff/scripts/check_handoff.sh`

Run the preflight from this directory:

```bash
bash release/geoweave_paper_handoff/scripts/check_handoff.sh
```

Smoke configs are only startup checks. Use `--paper --dry-run` to inspect the
fixed standard benchmark evaluation configs, and use `robustness_pi3.sh` for
the paper weak-overlap and distractor-context protocols.
