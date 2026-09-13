# Verification

Completed on local machine:

| Check | Command | Result |
| --- | --- | --- |
| Handoff preflight | `bash release/geoweave_paper_handoff/scripts/check_handoff.sh` | Passed path, bash syntax, Python syntax, anonymous package compile checks. Runtime import skipped because local Python has no `torch`. |
| Eval config invariants | `bash release/geoweave_paper_handoff/scripts/check_handoff.sh` | Checks paper vs smoke config separation, fixed checkpoint paths, task counts, and ETH3D pose `max_frames` protocol. |
| Eval CLI starts | `bash release/geoweave_paper_handoff/scripts/eval_unified.sh --help` | Passed. |
| Eval task registry | `bash release/geoweave_paper_handoff/scripts/eval_unified.sh --list-tasks` | Passed. |
| Paper-era eval dry-run | `bash release/geoweave_paper_handoff/scripts/eval_unified.sh --config aidi/configs/vggt/unified_eval_vggt_indexer_20260503_pt79_trusted.yaml --dry-run` | Passed and expanded planned commands. |
| VGGT handoff warm-up smoke dry-run | `bash release/geoweave_paper_handoff/scripts/train_vggt.sh --warmup-smoke --dry-run` | Checks resolved 2-GPU warm-up smoke command without launching GPU work. |
| VGGT handoff train smoke dry-run | `bash release/geoweave_paper_handoff/scripts/train_vggt.sh --smoke --dry-run` | Checks resolved 2-GPU smoke command without launching GPU work. |
| Pi3 handoff warm-up smoke dry-run | `bash release/geoweave_paper_handoff/scripts/train_pi3.sh --warmup-smoke --dry-run` | Checks resolved 2-GPU warm-up smoke command without launching GPU work. |
| Pi3 handoff train smoke dry-run | `bash release/geoweave_paper_handoff/scripts/train_pi3.sh --smoke --dry-run` | Checks resolved 2-GPU sparse/indexer smoke command without launching GPU work. |
| VGGT/Pi3 handoff eval smoke dry-run | `bash release/geoweave_paper_handoff/scripts/eval_vggt.sh --smoke --dry-run` and `bash release/geoweave_paper_handoff/scripts/eval_pi3.sh --smoke --dry-run` | Checks handoff eval YAML parsing and command expansion. |

Completed in an owned GPU runtime container:

| Check | Result |
| --- | --- |
| Runtime import check | `STRICT_IMPORT=1 bash release/geoweave_paper_handoff/scripts/check_handoff.sh` passed. |
| VGGT 2-GPU warm-up smoke | Passed in the clean package during final handoff validation. |
| VGGT 2-GPU sparse smoke | Passed in the clean package during final handoff validation. |
| Pi3 2-GPU warm-up smoke | Passed in the clean package during final handoff validation. |
| Pi3 2-GPU sparse smoke | Passed in the clean package during final handoff validation. |
| VGGT paper eval entry | Ran for 15 minutes with no error-pattern logs. |
| Pi3 paper eval entry | Ran for 15 minutes with no error-pattern logs. |

Not run:

| Check | Reason |
| --- | --- |
| Formal long training | Too expensive for handoff cleanup; run only after resource confirmation. |
| Full paper benchmark | Too expensive for handoff cleanup; current validation only proves entry/config/weight/runtime startup. |

## Required Runtime Acceptance

If the environment changes, rerun these inside the owned container on a synced
workspace:

```bash
bash release/geoweave_paper_handoff/scripts/train_vggt.sh --warmup-smoke
bash release/geoweave_paper_handoff/scripts/train_vggt.sh --smoke
bash release/geoweave_paper_handoff/scripts/train_pi3.sh --warmup-smoke
bash release/geoweave_paper_handoff/scripts/train_pi3.sh --smoke
```

Acceptance evidence should include: 2 ranks launched, model/checkpoint loaded,
at least one forward/backward/optimizer step completed, and a log or record
directory written under `SAVE_ROOT`. The VGGT smoke modes intentionally disable
checkpoint save to avoid writing large model files during startup checks; use
the paper recipes for checkpointing.

Smoke eval configs are not paper-result configs. In particular, ETH3D pose
smoke uses `max_frames=12`, while the paper config uses the full
`max_frames=100` protocol.
