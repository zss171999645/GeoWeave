# Cleanup Policy

The goal is to hand off a clean, executable paper codebase without destroying
experiment provenance.

## Rules

1. Do not delete non-temporary repository files before explicit confirmation.
2. Keep the paper baseline commit `e631d7c5` and tag `paper-submit-20260518`
   as immutable references.
3. Keep original training/evaluation entry paths stable because experiment
   records reference them.
4. Move or hide research-time clutter from the handoff surface by using this
   directory as the clean entry point.
5. Only promote scripts/configs into the clean handoff if they are needed for
   training, inference/evaluation, rebuttal experiments, or final result
   reproduction.

## Practical Cleanup Order

1. Use `manifests/KEEP_MAINLINE.tsv` as the protected set.
2. Review `manifests/ARCHIVE_CANDIDATES.tsv` and move accepted items into an
   archive directory or leave them outside the handoff surface.
3. Review `manifests/DELETE_CANDIDATES.tsv` only after confirming no active
   training/evaluation job and no paper table depends on the path.
4. Re-run `scripts/check_handoff.sh` after each cleanup batch.
