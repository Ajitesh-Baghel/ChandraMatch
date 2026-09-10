# ChandraMatch Project Status

## Completed

- Chandrayaan-2 TMC-2 product ingestion
- GeoTIFF metadata inspection
- geographic bounding-box intersection
- valid observation footprint verification
- automatic high-validity lunar crop discovery
- canonical 2048 × 2048 Pair 001 generation
- TMC-2 intensity normalization
- SIFT correspondence baseline
- SuperPoint + LightGlue integration
- LoFTR integration
- affine RANSAC geometric verification
- correspondence CSV output
- spatial coverage metric
- registered source generation
- overlay and difference visualization
- automated benchmark summary generation

## Pair 001

Source:
TMC-2 — 2019-12-12

Reference:
TMC-2 — 2020-01-08

Patch:
2048 × 2048

Common valid ratio:
~99.03%

## Current Matching Results

### SIFT
- 1024 candidate matches
- 980 RANSAC inliers
- 95.70% inlier ratio
- 0.611 px RANSAC reprojection RMSE
- 100% spatial coverage
- ~0.86 s runtime

### SuperPoint + LightGlue
- 1714 candidate matches
- 1329 RANSAC inliers
- 77.54% inlier ratio
- 1.483 px RANSAC reprojection RMSE
- 100% spatial coverage
- ~4.48 s runtime

### LoFTR
- 9696 candidate matches
- 9388 RANSAC inliers
- 96.82% inlier ratio
- 0.975 px RANSAC reprojection RMSE
- 100% spatial coverage
- ~1.11 s runtime

## Anchor-confidence ML component (§5b)

A calibrated logistic-regression gate over interpretable Stage 0/1 evidence
(bbox fill ratio, NCC improvement, local-cell agreement, etc.), designed to
replace the existing hand-set threshold cascade for anchor acceptance. It
deliberately does **not** use `verification_tier` or any other
downstream/outcome-dependent signal as a live inference feature, since that
information only exists after the anchor has already been used to run a
perturbation test — it is used only to weight training-row confidence when a
model is eventually fit. Current status: `insufficient_training_data` on
both the dense (2 rows) and fallback (7 rows) feature spaces, below the
20-row floor this project set for fitting safely — no model is fit, by
design, rather than overfitting to a handful of points. The existing
hand-set cascade remains the sole live decision path for all 5 validated
pairs' verdicts. The fallback feature space could reasonably activate once
more keypoint-fallback pairs are processed; the dense feature space needs
many more dense-correlation-anchored pairs. Full detail:
`docs/UNIFIED_PIPELINE_CHECKPOINT_LOG.md`.

## Next Technical Milestones

1. controlled geometric benchmark
2. ground-truth transformation evaluation
3. sub-pixel refinement
4. coverage-aware final correspondence selection
5. illumination robustness benchmark
6. OHRC ↔ TMC-2
7. IIRS ↔ TMC-2
8. unified ChandraMatch pipeline
9. judge-facing application