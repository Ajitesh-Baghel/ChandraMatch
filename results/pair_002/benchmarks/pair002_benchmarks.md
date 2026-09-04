# Pair 002 Benchmark

Real Chandrayaan-2 OHRC ↔ TMC-2 cross-sensor correspondence benchmark.

**Important:** `ransac_reprojection_rmse_px` measures self-consistency of the fitted affine model. It is not independent ground-truth registration accuracy.

| matcher   | preprocessing   | status   |   mask_valid_candidate_matches |   inliers |   inlier_ratio |   ransac_reprojection_rmse_px |   ransac_median_residual_px |   valid_region_coverage |   uniform_correspondences |   total_runtime_sec |
|:----------|:----------------|:---------|-------------------------------:|----------:|---------------:|------------------------------:|----------------------------:|------------------------:|--------------------------:|--------------------:|
| sift      | baseline        | success  |                             35 |         5 |      0.142857  |                      0.812976 |                    0.41036  |               0.0697674 |                         5 |            0.1622   |
| sift      | gradient        | success  |                             42 |         4 |      0.0952381 |                      0.850666 |                    0.666122 |               0.0697674 |                         4 |            0.184371 |
| lightglue | baseline        | success  |                             35 |        14 |      0.4       |                      1.50857  |                    1.39254  |               0.0930233 |                         9 |            0.880174 |
| lightglue | gradient        | success  |                            104 |        36 |      0.346154  |                      1.67505  |                    1.50116  |               0.232558  |                        32 |            0.256858 |
| loftr     | baseline        | success  |                            206 |        22 |      0.106796  |                      0.694536 |                    0.171329 |               0.116279  |                        16 |            0.47586  |
| loftr     | gradient        | success  |                            238 |        22 |      0.092437  |                      0.475713 |                    0.15059  |               0.116279  |                        14 |            0.382916 |
