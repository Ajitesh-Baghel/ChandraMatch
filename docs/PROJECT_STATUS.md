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