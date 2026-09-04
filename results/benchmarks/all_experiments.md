# ChandraMatch Benchmark Summary

| experiment   | pair     | matcher                |   candidate_matches |   inlier_matches |   inlier_ratio |   reprojection_rmse_px |   reprojection_median_px |   spatial_coverage |   runtime_seconds |
|:-------------|:---------|:-----------------------|--------------------:|-----------------:|---------------:|-----------------------:|-------------------------:|-------------------:|------------------:|
| sift         | pair_001 | SIFT                   |                1024 |              980 |       0.957031 |               0.61087  |                 0.263793 |                  1 |          0.860424 |
| lightglue    | pair_001 | SuperPoint + LightGlue |                1714 |             1329 |       0.775379 |               1.48259  |                 1.14645  |                  1 |          4.47773  |
| loftr        | pair_001 | LoFTR                  |                9696 |             9388 |       0.968234 |               0.975198 |                 0.781769 |                  1 |          1.11357  |
