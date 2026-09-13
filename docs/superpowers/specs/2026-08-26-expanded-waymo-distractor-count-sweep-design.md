# Expanded Waymo Distractor-Count Sweep Design

## Goal

Extend the existing 20-example Waymo distractor-context stress test with a simple, model-independent construction that directly evaluates zero through four distractor context views. Run the expansion in stages and stop before further computation if the new evidence contradicts the paper's robustness claim. Do not modify the manuscript during this experiment.

## Fixed protocol

Each input contains ten views:

- views 0--5 are the fixed clean evaluation prefix;
- views 6--9 are context views;
- the number of distractor context views is varied from zero to four by replacing the last `k` clean context views with the corresponding frames from one cross-scene distractor tail.

Pi3 and GeoWeave-Pi3 receive identical inputs. Only the six fixed clean prefix views are evaluated. The input width, checkpoints, cached-pose evaluator, and tuple-level alignment match the verified historical 512-pixel run.

## Existing 20 examples

Retain the original 20 feature-selected clean/distractor pairs and their verified five-level results. These examples remain the continuity set for the paper and rebuttal protocol.

## New candidate pool

Use `/mnt/cfs/datasets/Waymo2/testing`, which contains 80 per-camera sequences from 16 driving segments and five cameras.

Construct 12 evenly spaced clean anchors per testing sequence and freeze the historical minimum similarity margin:

1. intersect the available color and pose frame indices;
2. choose 12 evenly spaced ten-view windows with frame stride four;
3. use its first six views as the clean prefix and its last four views as the clean context;
4. search sequences from a different driving segment, allowing cross-camera candidates when the image resolution matches;
5. require the distractor tail to be closer to the clean prefix than the sample's own clean context by at least `0.00022410058591049165`, the minimum margin observed in the historical 20 examples;
6. rank all eligible pairs with the historical composite feature/structure score, including the positive wrong-more-like-prefix margin and the existing entropy/edge-density terms.

The selection uses image statistics only and never uses Pi3, GeoWeave, pose error, depth error, or point error. Rank the resulting 80 pairs by the same plausible-context score used by the rebuttal protocol.

## Stage 1: expand from 20 to 40

Add the top 20 new testing pairs to the original 20 examples.

### Endpoint gate

First run only the zero- and four-distractor endpoints for the new 20 examples. For metric `m`, define the robustness advantage

`A_m = (m_4 - m_0)_Pi3 - (m_4 - m_0)_GeoWeave`,

where positive values favor GeoWeave.

Stop and report without running intermediate levels if either condition holds:

- both `A_ATE` and `A_RPE-t` are non-positive on the new 20 examples; or
- either `A_ATE` or `A_RPE-t` is non-positive on the combined 40 examples.

If the endpoint gate passes, run the one-, two-, and three-distractor levels for the new 20 examples and summarize Original-20, New-20, and Combined-40 separately.

## Stage 2: optional expansion to 100

Expansion to 100 is allowed only if the Combined-40 result satisfies all of the following:

- GeoWeave has lower ATE and RPE-t at four distractors;
- GeoWeave has smaller clean-relative degradation for both ATE and RPE-t at four distractors;
- GeoWeave has lower mean degradation area across the four nonzero distractor counts for both ATE and RPE-t.

If these conditions pass, add the remaining 60 testing pairs so that the final set contains the original 20 plus all 80 testing-sequence anchors. Again run the zero/four endpoints first and apply the same combined-result endpoint gate before running the intermediate levels.

## Statistics and outputs

Report, for every set and distractor count:

- mean and median ATE;
- mean and median translation RPE;
- clean-relative degradation;
- P90 and maximum error;
- paired bootstrap confidence intervals over examples.

Store complete per-example rows, aggregate CSV/JSON summaries, construction manifests, validation results, and run logs under a new output root. Generated tuples must preserve identical prefix image hashes across all five levels, contain ten finite pose rows, and record evaluation indices `[0,1,2,3,4,5]`.

## Acceptance criteria

- New sample selection is reproducible and independent of model outputs.
- Every constructed input has ten images and a byte-identical six-view prefix across levels.
- Every completed model/level cell contains all expected finite pose rows; no failures are silently dropped.
- Stage gates are evaluated before launching additional inference.
- The manuscript and its PDFs remain unchanged until the user reviews the experiment results.
