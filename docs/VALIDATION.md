# Release validation

Checked on 2026-09-14 in an isolated source copy using Python 3.10.20 and
PyTorch 2.4.1+cu121. The host had no CUDA device; inference used the CPU PyTorch
reference backend. The existing inference environment was reused, with additional
training dependencies installed separately for configuration checks.

## Executed

- Six public-entry contract tests: image ordering/empty inputs and checkpoint
  prefix/special-token normalization, including rejection of mixed prefixes and
  colliding keys.
- Both real final GeoWeave checkpoints loaded with `strict=True`: native Pi3
  `checkpoint_79/pytorch_model.bin` and VGGT `79.pt` (release filenames:
  `geoweave_pi3.pth` and `geoweave_vggt.pth`, respectively; contents unchanged).
- Both models ran on two real DL3DV frames, first at width 112, then at the default
  width 518 (actual tensor shape `[1, 2, 3, 294, 518]`). All saved predictions were
  finite. The 518 run has more than 1024 context tokens and exercises sparse
  selection in the reference backend.
- Pi3 saved world/local point maps and camera-to-world poses. VGGT saved point
  maps, depths/confidences, pose encodings, world-to-camera extrinsics, and
  intrinsics. Each run wrote a colored binary PLY and metadata alongside the NPZ.
- Native Pi3 warm-up and sparse-stage Hydra configurations fully resolved,
  including the public dataset root override. Both public local training launcher
  commands expanded with existing checkpoint paths.
- The VGGT launcher command was parsed by the native config engine: multiworker
  DDP, checkpoint initialization, and fresh-schedule options resolved correctly.
- New/changed Python sources parsed successfully; scanned new/changed files had
  no matches for the checked credential patterns.

## Not established by these checks

A GPU training step, CUDA/Triton compilation and numerical execution, full
training, paper-table reproduction, cross-backend bitwise equality, and GPU
performance were not run in this release check. Use the kernel verification
commands on the target GPU before making acceleration claims. The CPU forward
elapsed times in local metadata are not paper benchmark results.
