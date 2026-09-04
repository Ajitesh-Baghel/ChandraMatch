import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.preprocessing.tmc2 import (
    read_tmc2,
    preprocess_tmc2
)

from src.preprocessing.illumination import (
    get_illumination_variant
)

from src.matching.sift_matcher import (
    match_sift
)

from src.matching.lightglue_matcher import (
    SuperPointLightGlueMatcher
)

from src.matching.loftr_matcher import (
    LoFTRMatcher
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

from src.registration.affine_registration import (
    register_affine,
    create_overlay,
    create_difference
)

from src.utils.visualize_matches import (
    draw_matches
)


# ============================================================
# PATHS
# ============================================================

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_001"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "illumination"
    / "pair_001"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

BENCHMARK_DIR = (
    ROOT
    / "results"
    / "benchmarks"
)

BENCHMARK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SOURCE_PATH = (
    PAIR_DIR
    / "source.tif"
)

REFERENCE_PATH = (
    PAIR_DIR
    / "reference.tif"
)


# ============================================================
# KNOWN SUN METADATA
# ============================================================

SOURCE_SUN_AZIMUTH = 233.270861
SOURCE_SUN_ELEVATION = 67.441269

REFERENCE_SUN_AZIMUTH = 57.237516
REFERENCE_SUN_ELEVATION = 72.758283


azimuth_difference = abs(
    SOURCE_SUN_AZIMUTH
    - REFERENCE_SUN_AZIMUTH
)

azimuth_difference = min(
    azimuth_difference,
    360.0 - azimuth_difference
)

elevation_difference = abs(
    SOURCE_SUN_ELEVATION
    - REFERENCE_SUN_ELEVATION
)


# ============================================================
# LOAD IMAGES
# ============================================================

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


source_base = preprocess_tmc2(
    source_raw,
    use_clahe=False
)

reference_base = preprocess_tmc2(
    reference_raw,
    use_clahe=False
)


print("=" * 70)

print(
    "CHANDRAMATCH - REAL SUN-ANGLE BENCHMARK"
)

print("=" * 70)

print(
    "Source Sun azimuth:",
    SOURCE_SUN_AZIMUTH
)

print(
    "Reference Sun azimuth:",
    REFERENCE_SUN_AZIMUTH
)

print(
    "Azimuth difference:",
    azimuth_difference
)

print(
    "Elevation difference:",
    elevation_difference
)

print(
    "Common valid ratio:",
    valid_mask.mean()
)


# ============================================================
# LOAD DEEP MODELS ONCE
# ============================================================

print(
    "\nLoading SuperPoint + LightGlue..."
)

lightglue = (
    SuperPointLightGlueMatcher(
        max_keypoints=4096
    )
)


print(
    "\nLoading LoFTR..."
)

loftr = LoFTRMatcher(
    max_dimension=840,
    confidence_threshold=0.2
)


# ============================================================
# SETTINGS
# ============================================================

variants = [
    "baseline",
    "clahe",
    "local_contrast",
    "gradient"
]

matchers = [
    "SIFT",
    "LightGlue",
    "LoFTR"
]


all_results = []


# ============================================================
# RUN
# ============================================================

for variant in variants:

    print("\n")
    print("=" * 70)
    print(
        "PREPROCESSING:",
        variant
    )
    print("=" * 70)


    source = get_illumination_variant(
        source_base,
        variant
    )

    reference = get_illumination_variant(
        reference_base,
        variant
    )


    variant_dir = (
        RESULT_ROOT
        / variant
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    source_png = (
        variant_dir
        / "source.png"
    )

    reference_png = (
        variant_dir
        / "reference.png"
    )


    cv2.imwrite(
        str(source_png),
        source
    )

    cv2.imwrite(
        str(reference_png),
        reference
    )


    for matcher_name in matchers:

        print("\n")
        print("-" * 60)

        print(
            variant,
            "-",
            matcher_name
        )

        print("-" * 60)


        matcher_dir = (
            variant_dir
            / matcher_name.lower()
        )

        matcher_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()


        # ====================================================
        # MATCH
        # ====================================================

        if matcher_name == "SIFT":

            (
                source_points,
                reference_points,
                good_matches,
                _,
                _

            ) = match_sift(
                source,
                reference,
                mask=valid_mask,
                ratio_threshold=0.75,
                max_features=10000
            )

            confidence = np.ones(
                len(source_points),
                dtype=np.float32
            )


        elif matcher_name == "LightGlue":

            (
                source_points,
                reference_points,
                confidence

            ) = lightglue.match(
                source_png,
                reference_png
            )


        elif matcher_name == "LoFTR":

            (
                source_points,
                reference_points,
                confidence

            ) = loftr.match(
                source,
                reference
            )


        else:

            raise ValueError(
                matcher_name
            )


        if torch.cuda.is_available():
            torch.cuda.synchronize()


        runtime = (
            time.perf_counter()
            - start
        )


        # ====================================================
        # VALID-AREA FILTER
        # ====================================================

        if len(source_points) < 3:

            print(
                "Too few matches."
            )

            continue


        sx = np.round(
            source_points[:, 0]
        ).astype(int)

        sy = np.round(
            source_points[:, 1]
        ).astype(int)

        rx = np.round(
            reference_points[:, 0]
        ).astype(int)

        ry = np.round(
            reference_points[:, 1]
        ).astype(int)


        height, width = (
            valid_mask.shape
        )


        inside = (
            (sx >= 0)
            & (sx < width)
            & (sy >= 0)
            & (sy < height)
            & (rx >= 0)
            & (rx < width)
            & (ry >= 0)
            & (ry < height)
        )


        source_points = (
            source_points[inside]
        )

        reference_points = (
            reference_points[inside]
        )

        confidence = (
            confidence[inside]
        )


        sx = sx[inside]
        sy = sy[inside]

        rx = rx[inside]
        ry = ry[inside]


        valid = (
            valid_mask[
                sy,
                sx
            ]
            &
            valid_mask[
                ry,
                rx
            ]
        )


        source_points = (
            source_points[valid]
        )

        reference_points = (
            reference_points[valid]
        )

        confidence = (
            confidence[valid]
        )


        candidate_count = len(
            source_points
        )


        print(
            "Candidate matches:",
            candidate_count
        )


        if candidate_count < 3:
            continue


        # ====================================================
        # RANSAC
        # ====================================================

        try:

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

        except RuntimeError as error:

            print(
                "RANSAC failed:",
                error
            )

            continue


        inlier_confidence = (
            confidence[
                inlier_mask
            ]
        )


        # ====================================================
        # METRICS
        # ====================================================

        metrics = calculate_metrics(
            candidate_count,
            inlier_source,
            inlier_reference,
            transformation
        )


        coverage = (
            calculate_spatial_coverage(
                inlier_source,
                source.shape,
                grid_rows=8,
                grid_cols=8
            )
        )


        metrics[
            "spatial_coverage"
        ] = coverage[
            "coverage_ratio"
        ]

        metrics[
            "runtime_seconds"
        ] = float(
            runtime
        )

        metrics[
            "matcher"
        ] = matcher_name

        metrics[
            "preprocessing"
        ] = variant

        metrics[
            "sun_azimuth_difference_deg"
        ] = float(
            azimuth_difference
        )

        metrics[
            "sun_elevation_difference_deg"
        ] = float(
            elevation_difference
        )

        metrics[
            "transformation"
        ] = transformation.tolist()


        # ====================================================
        # SAVE CORRESPONDENCES
        # ====================================================

        df = pd.DataFrame({

            "source_x":
                inlier_source[:, 0],

            "source_y":
                inlier_source[:, 1],

            "reference_x":
                inlier_reference[:, 0],

            "reference_y":
                inlier_reference[:, 1],

            "confidence":
                inlier_confidence
        })


        df.to_csv(
            matcher_dir
            / "correspondences.csv",
            index=False
        )


        # ====================================================
        # REGISTRATION
        # ====================================================

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
                matcher_dir
                / "registered.png"
            ),
            registered
        )

        cv2.imwrite(
            str(
                matcher_dir
                / "overlay.png"
            ),
            overlay
        )

        cv2.imwrite(
            str(
                matcher_dir
                / "difference.png"
            ),
            difference
        )


        draw_matches(
            source,
            reference,
            inlier_source,
            inlier_reference,
            matcher_dir
            / "inlier_matches.png",
            max_matches=150
        )


        with open(
            matcher_dir
            / "metrics.json",
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )


        all_results.append(
            metrics
        )


        print(
            "Inliers:",
            metrics[
                "inlier_matches"
            ]
        )

        print(
            "Inlier ratio:",
            metrics[
                "inlier_ratio"
            ]
        )

        print(
            "Reprojection RMSE:",
            metrics[
                "ransac_reprojection_rmse_px"
            ]
        )

        print(
            "Coverage:",
            metrics[
                "spatial_coverage"
            ]
        )

        print(
            "Runtime:",
            runtime
        )


# ============================================================
# SAVE MASTER TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)


output_csv = (
    BENCHMARK_DIR
    / "illumination_pair001.csv"
)


results_df.to_csv(
    output_csv,
    index=False
)


print("\n")
print("=" * 70)

print(
    "REAL SUN-ANGLE BENCHMARK COMPLETE"
)

print("=" * 70)


show_columns = [
    "preprocessing",
    "matcher",
    "candidate_matches",
    "inlier_matches",
    "inlier_ratio",
    "ransac_reprojection_rmse_px",
    "spatial_coverage",
    "runtime_seconds"
]


print(
    results_df[
        show_columns
    ].to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    output_csv
)