import sys
import json
import time
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocessing.tmc2 import (
    read_tmc2,
    preprocess_tmc2
)

from src.matching.lightglue_matcher import (
    SuperPointLightGlueMatcher
)

from src.geometry.ransac_filter import (
    filter_matches_ransac
)

from src.geometry.coverage_filter import (
    calculate_spatial_coverage
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
    / "lightglue"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_PATH = (
    PAIR_DIR / "source.tif"
)

REFERENCE_PATH = (
    PAIR_DIR / "reference.tif"
)


print("=" * 60)
print(
    "CHANDRAMATCH - SUPERPOINT + LIGHTGLUE"
)
print("=" * 60)


# ---------------------------------------------------------
# Load original TMC-2 images
# ---------------------------------------------------------

source_raw, _ = read_tmc2(
    SOURCE_PATH
)

reference_raw, _ = read_tmc2(
    REFERENCE_PATH
)


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
# TMC-2 preprocessing
# ---------------------------------------------------------

source = preprocess_tmc2(
    source_raw,
    use_clahe=False
)

reference = preprocess_tmc2(
    reference_raw,
    use_clahe=False
)


SOURCE_PREPROCESSED = (
    RESULT_DIR
    / "source_preprocessed.png"
)

REFERENCE_PREPROCESSED = (
    RESULT_DIR
    / "reference_preprocessed.png"
)


cv2.imwrite(
    str(SOURCE_PREPROCESSED),
    source
)

cv2.imwrite(
    str(REFERENCE_PREPROCESSED),
    reference
)


# ---------------------------------------------------------
# Initialise model
#
# Do this BEFORE timing so model loading does not distort
# inference/runtime comparison.
# ---------------------------------------------------------

lightglue = SuperPointLightGlueMatcher(
    max_keypoints=4096
)


# ---------------------------------------------------------
# SuperPoint + LightGlue
# ---------------------------------------------------------

start_time = time.perf_counter()

(
    source_points,
    reference_points,
    confidence

) = lightglue.match(
    SOURCE_PREPROCESSED,
    REFERENCE_PREPROCESSED
)


candidate_count = len(
    source_points
)

print(
    "\nCandidate matches:",
    candidate_count
)

if candidate_count < 3:
    raise RuntimeError(
        "Not enough LightGlue matches for RANSAC."
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


runtime = (
    time.perf_counter()
    - start_time
)


inlier_confidence = confidence[
    inlier_mask
]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

metrics = calculate_metrics(
    candidate_count,
    inlier_source,
    inlier_reference,
    transformation
)


# ---------------------------------------------------------
# Spatial coverage
# ---------------------------------------------------------

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


metrics["runtime_seconds"] = float(
    runtime
)

metrics["matcher"] = (
    "SuperPoint + LightGlue"
)

metrics["pair"] = "pair_001"

metrics["transformation"] = (
    transformation.tolist()
)


# ---------------------------------------------------------
# Correspondence export
# ---------------------------------------------------------

correspondence_df = pd.DataFrame({
    "source_x": inlier_source[:, 0],
    "source_y": inlier_source[:, 1],

    "reference_x":
        inlier_reference[:, 0],

    "reference_y":
        inlier_reference[:, 1],

    "confidence":
        inlier_confidence
})


correspondence_df.to_csv(
    RESULT_DIR / "correspondences.csv",
    index=False
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
# Match visualisations
# ---------------------------------------------------------

draw_matches(
    source,
    reference,
    source_points,
    reference_points,
    RESULT_DIR / "candidate_matches.png",
    max_matches=150
)

draw_matches(
    source,
    reference,
    inlier_source,
    inlier_reference,
    RESULT_DIR / "ransac_inliers.png",
    max_matches=150
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
# Terminal output
# ---------------------------------------------------------

print("\nTransformation:")
print(transformation)

print("\nMetrics:")

for key, value in metrics.items():
    print(
        f"{key}: {value}"
    )


print("\nResults saved to:")
print(RESULT_DIR)