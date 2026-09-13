# Original-Group-A Anchored ScanNet++ Overlap Sweep

Date: 2026-08-26

## Goal

Replace the temporary 20-anchor controlled overlap sweep with a four-level paired sweep anchored at the twelve Group A viewpoints used by the original ScanNet++ slight-overlap experiment.

## Scope

- Preserve the original twelve Group A viewpoint sets, not byte-identical legacy image files.
- Map each of the sixty legacy Group A images to its corresponding official ScanNet++ DSLR frame.
- Materialize all four levels from official DSLR images so Group A and Group B share one image domain and preprocessing path.
- Keep the original four fixed cross-group mean-overlap bands: high `[0.10,0.20)`, medium `[0.03,0.10)`, low `[0.005,0.03)`, and near-zero `[0,0.005)`.
- Use the historical 512-pixel inference setting for continuity with the original robustness tables.

## Mapping

For each of the six original scenes:

1. Read the two legacy tuple metadata files and collect output indices 0--4.
2. Rank all official DSLR frames in the same scene by normalized spatial-thumbnail similarity.
3. Require one-to-one top matches, report top-1 similarity and top-1/top-2 margin, and retain the top five candidates for audit.
4. Validate the scene mapping geometrically: official and legacy camera-center distance matrices must agree after one scalar fit, and pairwise relative-rotation angles must agree within tolerance.
5. Stop before tuple construction if any image is ambiguous or the scene-level pose residual is inconsistent.

## Controlled Tuple Construction

- Each mapped five-frame Group A is fixed across all four levels.
- Group B contains five distinct same-scene official DSLR frames, has no frame overlap with Group A, and requires anchor-to-support visibility overlap of at least `0.08`.
- Cross-group overlap is the mean of all 25 bidirectional visibility values using virtual pinhole depth ray-cast from the aligned official mesh.
- Select one Group B per band by distance to the band midpoint, then higher internal overlap, then frame ID.
- All twelve Group A anchors must have all four levels. Otherwise preserve diagnostics and explicitly decide whether to enlarge the candidate pool; do not alter band boundaries silently.

## Execution and Evaluation

- Target: 12 anchors x 4 levels = 48 ten-view tuples per model.
- Group A official image hashes must be identical across the four levels of each anchor.
- Run Pi3 and GeoWeave-Pi3 directly on common-dev GPUs, one model per GPU, at input size 512.
- Evaluate all ten poses after one tuple-level Sim(3), reporting ATE and RPE-t.
- Aggregate paired means, medians, tail statistics, and paired-bootstrap 95% confidence intervals over the twelve original Group A anchors.
- Compare the new low band to the original slight-overlap operating point while clearly distinguishing official-image rematerialization from byte-identical legacy tuples.
