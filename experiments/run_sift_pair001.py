import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from prometheus_client import metrics

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing.tmc2 import (
    read_tmc2,
    preprocess_tmc2
)

from src.matching.sift_matcher import (
    match_sift
)

from src.geometry.ransac_filter import (
    filter_matches_ransac
)

from src.evaluation.metrics import (
    calculate_metrics
)

from src.utils.visualize_matches import (
    draw_matches
)

from src.registration.affine_registration import (
    register_affine,
    create_overlay,
    create_difference
)
from src.geometry.coverage_filter import (
    calculate_spatial_coverage
)
PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_001"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "pair_001"
    / "sift"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SOURCE_PATH = PAIR_DIR / "source.tif"
REFERENCE_PATH = PAIR_DIR / "reference.tif"


print("=" * 60)
print("CHANDRAMATCH - SIFT BASELINE")
print("=" * 60)


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

source_raw, _ = read_tmc2(
    SOURCE_PATH
)

reference_raw, _ = read_tmc2(
    REFERENCE_PATH
)


# ---------------------------------------------------------
# Valid mask
# ---------------------------------------------------------

valid_mask = (
    (source_raw > 0)
    &
    (reference_raw > 0)
)


print(
    "Common valid ratio:",
    valid_mask.mean()
)


# ---------------------------------------------------------
# Preprocess
# ---------------------------------------------------------

source = preprocess_tmc2(
    source_raw,
    use_clahe=False
)

reference = preprocess_tmc2(
    reference_raw,
    use_clahe=False
)


cv2.imwrite(
    str(RESULT_DIR / "source_preprocessed.png"),
    source
)

cv2.imwrite(
    str(RESULT_DIR / "reference_preprocessed.png"),
    reference
)


# ---------------------------------------------------------
# SIFT
# ---------------------------------------------------------

start_time = time.perf_counter()

(
    source_points,
    reference_points,
    good_matches,
    source_keypoints,
    reference_keypoints

) = match_sift(
    source,
    reference,
    mask=valid_mask,
    ratio_threshold=0.75
)


candidate_count = len(
    good_matches
)

print(
    "Candidate matches:",
    candidate_count
)


# ---------------------------------------------------------
# RANSAC
# ---------------------------------------------------------

(
    transformation,
    inlier_mask,
    inlier_source,
    inlier_reference

) = filter_matches_ransac(
    source_points,
    reference_points,
    reprojection_threshold=3.0
)

# ---------------------------------------------------------
# Save correspondence points
# ---------------------------------------------------------

correspondence_df = pd.DataFrame({
    "source_x": inlier_source[:, 0],
    "source_y": inlier_source[:, 1],
    "reference_x": inlier_reference[:, 0],
    "reference_y": inlier_reference[:, 1]
})

correspondence_df.to_csv(
    RESULT_DIR / "correspondences.csv",
    index=False
)
runtime = (
    time.perf_counter()
    - start_time
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

metrics = calculate_metrics(
    candidate_count,
    inlier_source,
    inlier_reference,
    transformation
)

metrics["runtime_seconds"] = float(
    runtime
)
coverage = calculate_spatial_coverage(
    inlier_source,
    source.shape,
    grid_rows=8,
    grid_cols=8
)

metrics["spatial_coverage"] = (
    coverage["coverage_ratio"]
)

metrics["occupied_grid_cells"] = (
    coverage["occupied_cells"]
)

metrics["total_grid_cells"] = (
    coverage["total_cells"]
)
metrics["matcher"] = "SIFT"
metrics["pair"] = "pair_001"
metrics["transformation"] = (
    transformation.tolist()
)
# ---------------------------------------------------------
# Registration
# ---------------------------------------------------------

registered = register_affine(
    source,
    transformation,
    reference.shape
)

overlay = create_overlay(
    registered,
    reference
)

difference = create_difference(
    registered,
    reference
)


cv2.imwrite(
    str(
        RESULT_DIR
        / "registered_source.png"
    ),
    registered
)

cv2.imwrite(
    str(
        RESULT_DIR
        / "overlay.png"
    ),
    overlay
)

cv2.imwrite(
    str(
        RESULT_DIR
        / "difference.png"
    ),
    difference
)
# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\nTransformation:")
print(transformation)

print("\nMetrics:")

for key, value in metrics.items():
    print(
        f"{key}: {value}"
    )


# ---------------------------------------------------------
# Save metrics
# ---------------------------------------------------------

with open(
    RESULT_DIR / "metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ---------------------------------------------------------
# Candidate matches
# ---------------------------------------------------------

draw_matches(
    source,
    reference,
    source_points,
    reference_points,
    RESULT_DIR / "candidate_matches.png",
    max_matches=150
)


# ---------------------------------------------------------
# RANSAC inliers
# ---------------------------------------------------------

draw_matches(
    source,
    reference,
    inlier_source,
    inlier_reference,
    RESULT_DIR / "ransac_inliers.png",
    max_matches=150
)


print("\nResults saved to:")
print(RESULT_DIR)