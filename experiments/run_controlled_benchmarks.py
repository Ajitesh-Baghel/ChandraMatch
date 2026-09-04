import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


# ============================================================
# CHANDRAMATCH MODULES
# ============================================================

from src.preprocessing.tmc2 import (
    read_tmc2,
    preprocess_tmc2
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

from src.registration.affine_registration import (
    register_affine,
    create_overlay,
    create_difference
)

from src.utils.visualize_matches import (
    draw_matches
)

from src.evaluation.controlled_transform import (
    create_affine_transform,
    apply_affine
)

from src.evaluation.ground_truth_metrics import (
    correspondence_ground_truth_errors,
    calculate_error_summary,
    transformation_ground_truth_error
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

CONFIG_PATH = (
    ROOT
    / "configs"
    / "controlled_benchmarks.json"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "controlled"
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


REFERENCE_TIF = (
    PAIR_DIR
    / "reference.tif"
)


# ============================================================
# LOAD REFERENCE IMAGE
# ============================================================

reference_raw, _ = read_tmc2(
    REFERENCE_TIF
)

reference = preprocess_tmc2(
    reference_raw,
    use_clahe=False
)


# Use original valid mask rather than brightness of the
# normalized PNG.

reference_mask = (
    reference_raw > 0
)


print("=" * 70)
print("CHANDRAMATCH - CONTROLLED GROUND TRUTH BENCHMARK")
print("=" * 70)

print(
    "Reference shape:",
    reference.shape
)


# ============================================================
# LOAD BENCHMARK CONFIGURATION
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    configurations = json.load(
        file
    )


print(
    "Controlled cases:",
    len(configurations)
)


# ============================================================
# LOAD DEEP MODELS ONCE
# ============================================================

print(
    "\nLoading SuperPoint + LightGlue..."
)

lightglue = SuperPointLightGlueMatcher(
    max_keypoints=4096
)


print(
    "\nLoading LoFTR..."
)

loftr = LoFTRMatcher(
    max_dimension=840,
    confidence_threshold=0.2
)


# ============================================================
# RESULT RECORDS
# ============================================================

all_results = []


# ============================================================
# HELPER — VALID MATCH FILTER
# ============================================================

def filter_valid_matches(
    source_points,
    reference_points,
    confidence,
    source_mask,
    reference_mask
):

    if len(source_points) == 0:

        return (
            source_points,
            reference_points,
            confidence
        )


    height, width = source_mask.shape


    source_x = np.round(
        source_points[:, 0]
    ).astype(int)

    source_y = np.round(
        source_points[:, 1]
    ).astype(int)

    reference_x = np.round(
        reference_points[:, 0]
    ).astype(int)

    reference_y = np.round(
        reference_points[:, 1]
    ).astype(int)


    inside = (

        (source_x >= 0)
        &
        (source_x < width)
        &
        (source_y >= 0)
        &
        (source_y < height)
        &

        (reference_x >= 0)
        &
        (reference_x < width)
        &
        (reference_y >= 0)
        &
        (reference_y < height)
    )


    source_points = source_points[
        inside
    ]

    reference_points = reference_points[
        inside
    ]

    confidence = confidence[
        inside
    ]


    source_x = source_x[
        inside
    ]

    source_y = source_y[
        inside
    ]

    reference_x = reference_x[
        inside
    ]

    reference_y = reference_y[
        inside
    ]


    valid = (

        source_mask[
            source_y,
            source_x
        ]

        &

        reference_mask[
            reference_y,
            reference_x
        ]
    )


    return (

        source_points[
            valid
        ],

        reference_points[
            valid
        ],

        confidence[
            valid
        ]
    )


# ============================================================
# RUN ONE MATCHER
# ============================================================

def run_matcher(
    matcher_name,
    source,
    reference,
    source_path,
    reference_path,
    source_mask,
    reference_mask,
    ground_truth_transform,
    output_dir
):

    print(
        "\n",
        "-" * 60,
        sep=""
    )

    print(
        "Matcher:",
        matcher_name
    )

    print(
        "-" * 60
    )


    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()


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

            mask=None,

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
            source_path,
            reference_path
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
            f"Unknown matcher: {matcher_name}"
        )


    if torch.cuda.is_available():
        torch.cuda.synchronize()


    runtime = (
        time.perf_counter()
        - start
    )


    raw_candidate_count = len(
        source_points
    )


    # --------------------------------------------------------
    # VALID PIXEL FILTER
    # --------------------------------------------------------

    (
        source_points,
        reference_points,
        confidence

    ) = filter_valid_matches(

        source_points,
        reference_points,
        confidence,

        source_mask,
        reference_mask
    )


    candidate_count = len(
        source_points
    )


    print(
        "Raw candidate matches:",
        raw_candidate_count
    )

    print(
        "Valid candidate matches:",
        candidate_count
    )


    if candidate_count < 3:

        print(
            "Not enough matches. Skipping."
        )

        return None


    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    try:

        (
            estimated_transform,
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

        return None


    inlier_confidence = (
        confidence[
            inlier_mask
        ]
    )


    inlier_count = len(
        inlier_source
    )


    inlier_ratio = (
        inlier_count
        / candidate_count
    )


    print(
        "RANSAC inliers:",
        inlier_count
    )

    print(
        "Inlier ratio:",
        inlier_ratio
    )


    # --------------------------------------------------------
    # TRUE CORRESPONDENCE ERROR
    # --------------------------------------------------------

    gt_errors = (
        correspondence_ground_truth_errors(

            inlier_source,

            inlier_reference,

            ground_truth_transform
        )
    )


    correspondence_gt = (
        calculate_error_summary(
            gt_errors
        )
    )


    # --------------------------------------------------------
    # TRUE TRANSFORMATION ERROR
    # --------------------------------------------------------

    transform_gt = (
        transformation_ground_truth_error(

            estimated_transform,

            ground_truth_transform,

            source_mask,

            reference_mask,

            grid_step=64
        )
    )


    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    coverage = (
        calculate_spatial_coverage(

            inlier_source,

            source.shape,

            grid_rows=8,

            grid_cols=8
        )
    )


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # CORRESPONDENCE CSV
    # --------------------------------------------------------

    correspondence_df = pd.DataFrame({

        "source_x":
            inlier_source[:, 0],

        "source_y":
            inlier_source[:, 1],

        "reference_x":
            inlier_reference[:, 0],

        "reference_y":
            inlier_reference[:, 1],

        "confidence":
            inlier_confidence,

        "ground_truth_error_px":
            gt_errors
    })


    correspondence_df.to_csv(

        output_dir
        / "correspondences.csv",

        index=False
    )


    # --------------------------------------------------------
    # MATCH VISUALISATION
    # --------------------------------------------------------

    draw_matches(

        source,
        reference,

        source_points,
        reference_points,

        output_dir
        / "candidate_matches.png",

        max_matches=150
    )


    draw_matches(

        source,
        reference,

        inlier_source,
        inlier_reference,

        output_dir
        / "ransac_inliers.png",

        max_matches=150
    )


    # --------------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------------

    registered = register_affine(

        source,

        estimated_transform,

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
            output_dir
            / "registered_source.png"
        ),

        registered
    )


    cv2.imwrite(

        str(
            output_dir
            / "overlay.png"
        ),

        overlay
    )


    cv2.imwrite(

        str(
            output_dir
            / "difference.png"
        ),

        difference
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = {

        "matcher":
            matcher_name,

        "raw_candidate_matches":
            raw_candidate_count,

        "candidate_matches":
            candidate_count,

        "inlier_matches":
            inlier_count,

        "inlier_ratio":
            float(inlier_ratio),

        "ground_truth_correspondence_rmse_px":
            correspondence_gt["rmse"],

        "ground_truth_correspondence_mean_px":
            correspondence_gt["mean"],

        "ground_truth_correspondence_median_px":
            correspondence_gt["median"],

        "ground_truth_correspondence_max_px":
            correspondence_gt["max"],

        "ground_truth_registration_rmse_px":
            transform_gt["rmse"],

        "ground_truth_registration_mean_px":
            transform_gt["mean"],

        "ground_truth_registration_median_px":
            transform_gt["median"],

        "ground_truth_registration_max_px":
            transform_gt["max"],

        "ground_truth_evaluated_points":
            transform_gt[
                "evaluated_points"
            ],

        "spatial_coverage":
            coverage[
                "coverage_ratio"
            ],

        "occupied_grid_cells":
            coverage[
                "occupied_cells"
            ],

        "runtime_seconds":
            float(runtime),

        "estimated_transformation":
            estimated_transform.tolist(),

        "ground_truth_transformation":
            ground_truth_transform.tolist()
    }


    with open(

        output_dir
        / "metrics.json",

        "w"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4
        )


    print(
        "GT correspondence RMSE:",
        correspondence_gt["rmse"]
    )

    print(
        "GT registration RMSE:",
        transform_gt["rmse"]
    )

    print(
        "Spatial coverage:",
        coverage["coverage_ratio"]
    )

    print(
        "Runtime:",
        runtime
    )


    return metrics


# ============================================================
# RUN CONTROLLED CASES
# ============================================================

for config in configurations:

    benchmark_id = config["id"]

    print("\n")
    print("=" * 70)
    print("BENCHMARK:", benchmark_id)
    print("=" * 70)

    print(
        "Rotation:",
        config["rotation_deg"]
    )

    print(
        "Scale:",
        config["scale"]
    )

    print(
        "Translation:",
        config["translation_x"],
        config["translation_y"]
    )


    benchmark_dir = (
        RESULT_ROOT
        / benchmark_id
    )

    benchmark_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # CREATE KNOWN TRANSFORMATION
    # --------------------------------------------------------

    (
        reference_to_source,
        source_to_reference_gt

    ) = create_affine_transform(

        reference.shape,

        rotation_deg=
            config["rotation_deg"],

        scale=
            config["scale"],

        translation_x=
            config["translation_x"],

        translation_y=
            config["translation_y"]
    )


    # --------------------------------------------------------
    # CREATE SOURCE IMAGE
    # --------------------------------------------------------

    source = apply_affine(

        reference,

        reference_to_source
    )


    # Transform actual valid-data mask separately.

    source_mask_uint8 = (
        reference_mask.astype(
            np.uint8
        )
        * 255
    )


    source_mask_uint8 = cv2.warpAffine(

        source_mask_uint8,

        reference_to_source,

        (
            reference.shape[1],
            reference.shape[0]
        ),

        flags=cv2.INTER_NEAREST,

        borderMode=cv2.BORDER_CONSTANT,

        borderValue=0
    )


    source_mask = (
        source_mask_uint8 > 0
    )


    # --------------------------------------------------------
    # SAVE INPUTS
    # --------------------------------------------------------

    source_path = (
        benchmark_dir
        / "source.png"
    )

    reference_path = (
        benchmark_dir
        / "reference.png"
    )


    cv2.imwrite(
        str(source_path),
        source
    )

    cv2.imwrite(
        str(reference_path),
        reference
    )


    cv2.imwrite(

        str(
            benchmark_dir
            / "source_mask.png"
        ),

        source_mask_uint8
    )


    # --------------------------------------------------------
    # SAVE GROUND TRUTH
    # --------------------------------------------------------

    gt_metadata = {

        "benchmark_id":
            benchmark_id,

        "rotation_deg":
            config["rotation_deg"],

        "scale":
            config["scale"],

        "translation_x":
            config["translation_x"],

        "translation_y":
            config["translation_y"],

        "reference_to_source":
            reference_to_source.tolist(),

        "source_to_reference_ground_truth":
            source_to_reference_gt.tolist()
    }


    with open(

        benchmark_dir
        / "ground_truth.json",

        "w"

    ) as file:

        json.dump(

            gt_metadata,

            file,

            indent=4
        )


    # --------------------------------------------------------
    # RUN ALL THREE MATCHERS
    # --------------------------------------------------------

    for matcher_name in [

        "SIFT",
        "LightGlue",
        "LoFTR"

    ]:

        matcher_dir = (

            benchmark_dir
            / matcher_name.lower()
        )


        metrics = run_matcher(

            matcher_name,

            source,
            reference,

            source_path,
            reference_path,

            source_mask,
            reference_mask,

            source_to_reference_gt,

            matcher_dir
        )


        if metrics is None:
            continue


        metrics[
            "benchmark_id"
        ] = benchmark_id


        metrics[
            "rotation_deg"
        ] = config[
            "rotation_deg"
        ]


        metrics[
            "scale"
        ] = config[
            "scale"
        ]


        metrics[
            "translation_x"
        ] = config[
            "translation_x"
        ]


        metrics[
            "translation_y"
        ] = config[
            "translation_y"
        ]


        all_results.append(
            metrics
        )


# ============================================================
# SAVE MASTER CONTROLLED BENCHMARK TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)


output_csv = (

    BENCHMARK_DIR
    / "controlled_benchmarks.csv"
)


results_df.to_csv(
    output_csv,
    index=False
)


print("\n")
print("=" * 70)
print("CONTROLLED BENCHMARK COMPLETE")
print("=" * 70)


columns_to_show = [

    "benchmark_id",
    "matcher",
    "rotation_deg",
    "scale",
    "candidate_matches",
    "inlier_matches",
    "inlier_ratio",
    "ground_truth_correspondence_rmse_px",
    "ground_truth_registration_rmse_px",
    "spatial_coverage",
    "runtime_seconds"
]


if len(results_df) > 0:

    print(
        results_df[
            columns_to_show
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