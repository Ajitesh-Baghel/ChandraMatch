import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


# ============================================================
# MODULES
# ============================================================

from src.preprocessing.tmc2 import (
    read_tmc2
)

from src.geometry.uniform_selector import (
    select_uniform_correspondences
)

from src.geometry.ransac_filter import (
    filter_matches_ransac
)

from src.geometry.coverage_filter import (
    calculate_spatial_coverage
)

from src.refinement.subpixel import (
    refine_correspondences_phase
)

from src.evaluation.ground_truth_metrics import (
    correspondence_ground_truth_errors,
    calculate_error_summary,
    transformation_ground_truth_error
)

from src.utils.visualize_matches import (
    draw_matches
)

from src.registration.affine_registration import (
    register_affine,
    create_overlay,
    create_difference
)


# ============================================================
# PATHS
# ============================================================

CONTROLLED_ROOT = (
    ROOT
    / "results"
    / "controlled"
)

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_001"
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


# ============================================================
# ORIGINAL REFERENCE VALID MASK
# ============================================================

reference_raw, _ = read_tmc2(

    PAIR_DIR
    / "reference.tif"
)

reference_valid_mask = (
    reference_raw > 0
)


# ============================================================
# SETTINGS
# ============================================================

GRID_ROWS = 8
GRID_COLS = 8

MAX_PER_CELL = 5

PATCH_SIZE = 64

MAX_LOCAL_SHIFT = 3.0

MIN_PHASE_RESPONSE = 0.05


matcher_folders = {

    "sift":
        "SIFT",

    "lightglue":
        "LightGlue",

    "loftr":
        "LoFTR"
}


all_results = []


print("=" * 70)

print(
    "CHANDRAMATCH - SUBPIXEL REFINEMENT BENCHMARK"
)

print("=" * 70)


# ============================================================
# LOOP THROUGH CONTROLLED BENCHMARKS
# ============================================================

for benchmark_dir in sorted(
    CONTROLLED_ROOT.glob("CTRL_*")
):

    if not benchmark_dir.is_dir():
        continue


    benchmark_id = (
        benchmark_dir.name
    )


    print("\n")
    print("=" * 70)
    print("BENCHMARK:", benchmark_id)
    print("=" * 70)


    source = cv2.imread(

        str(
            benchmark_dir
            / "source.png"
        ),

        cv2.IMREAD_GRAYSCALE
    )


    reference = cv2.imread(

        str(
            benchmark_dir
            / "reference.png"
        ),

        cv2.IMREAD_GRAYSCALE
    )


    source_mask_image = cv2.imread(

        str(
            benchmark_dir
            / "source_mask.png"
        ),

        cv2.IMREAD_GRAYSCALE
    )


    if (
        source is None
        or
        reference is None
        or
        source_mask_image is None
    ):

        print(
            "Missing benchmark images. Skipping."
        )

        continue


    source_valid_mask = (
        source_mask_image > 0
    )


    # ========================================================
    # EACH MATCHER
    # ========================================================

    for folder_name, matcher_name in (
        matcher_folders.items()
    ):

        matcher_dir = (
            benchmark_dir
            / folder_name
        )


        metrics_path = (
            matcher_dir
            / "metrics.json"
        )


        correspondence_path = (
            matcher_dir
            / "correspondences.csv"
        )


        if (
            not metrics_path.exists()
            or
            not correspondence_path.exists()
        ):

            print(
                matcher_name,
                "results missing. Skipping."
            )

            continue


        print("\n")
        print("-" * 60)
        print(
            benchmark_id,
            "-",
            matcher_name
        )
        print("-" * 60)


        # ----------------------------------------------------
        # LOAD OLD RESULTS
        # ----------------------------------------------------

        with open(
            metrics_path,
            "r"
        ) as file:

            old_metrics = json.load(
                file
            )


        correspondence_df = pd.read_csv(
            correspondence_path
        )


        source_points = (
            correspondence_df[
                [
                    "source_x",
                    "source_y"
                ]
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        reference_points = (
            correspondence_df[
                [
                    "reference_x",
                    "reference_y"
                ]
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        if (
            "confidence"
            in correspondence_df.columns
        ):

            confidence = (
                correspondence_df[
                    "confidence"
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

        else:

            confidence = np.ones(

                len(source_points),

                dtype=np.float32
            )


        estimated_transform = np.asarray(

            old_metrics[
                "estimated_transformation"
            ],

            dtype=np.float32
        )


        ground_truth_transform = np.asarray(

            old_metrics[
                "ground_truth_transformation"
            ],

            dtype=np.float32
        )


        # ----------------------------------------------------
        # UNIFORM SELECTION
        # ----------------------------------------------------

        selected = (
            select_uniform_correspondences(

                source_points,

                reference_points,

                confidence,

                estimated_transform,

                source.shape,

                grid_rows=
                    GRID_ROWS,

                grid_cols=
                    GRID_COLS,

                max_per_cell=
                    MAX_PER_CELL
            )
        )


        selected_source = (
            selected[
                "source_points"
            ]
        )


        selected_reference = (
            selected[
                "reference_points"
            ]
        )


        selected_confidence = (
            selected[
                "confidence"
            ]
        )


        print(
            "Original inliers:",
            len(source_points)
        )

        print(
            "Uniformly selected:",
            len(selected_source)
        )


        # ----------------------------------------------------
        # RAW SELECTED MATCH ERROR
        # ----------------------------------------------------

        raw_errors = (
            correspondence_ground_truth_errors(

                selected_source,

                selected_reference,

                ground_truth_transform
            )
        )


        raw_summary = (
            calculate_error_summary(
                raw_errors
            )
        )


        # ----------------------------------------------------
        # SUBPIXEL REFINEMENT
        # ----------------------------------------------------

        refinement = (
            refine_correspondences_phase(

                source,

                reference,

                selected_source,

                estimated_transform,

                patch_size=
                    PATCH_SIZE,

                max_shift=
                    MAX_LOCAL_SHIFT,

                min_response=
                    MIN_PHASE_RESPONSE,

                use_gradient=True
            )
        )


        refined_source = (
            refinement[
                "source_points"
            ]
        )


        refined_reference = (
            refinement[
                "reference_points"
            ]
        )


        geometric_reference = (
            refinement[
                "geometric_reference_points"
            ]
        )


        responses = (
            refinement[
                "responses"
            ]
        )


        local_shifts = (
            refinement[
                "local_shifts"
            ]
        )


        kept_indices = (
            refinement[
                "kept_indices"
            ]
        )


        print(
            "Successfully refined:",
            len(refined_source)
        )


        if len(
            refined_source
        ) < 3:

            print(
                "Too few refined points."
            )

            continue


        # ----------------------------------------------------
        # RAW ERROR ON EXACT SAME REFINED SUBSET
        # ----------------------------------------------------

        raw_subset_reference = (
            selected_reference[
                kept_indices
            ]
        )


        raw_subset_errors = (
            correspondence_ground_truth_errors(

                refined_source,

                raw_subset_reference,

                ground_truth_transform
            )
        )


        raw_subset_summary = (
            calculate_error_summary(
                raw_subset_errors
            )
        )


        # ----------------------------------------------------
        # GLOBAL GEOMETRIC PREDICTION ERROR
        # ----------------------------------------------------

        geometric_errors = (
            correspondence_ground_truth_errors(

                refined_source,

                geometric_reference,

                ground_truth_transform
            )
        )


        geometric_summary = (
            calculate_error_summary(
                geometric_errors
            )
        )


        # ----------------------------------------------------
        # REFINED CORRESPONDENCE ERROR
        # ----------------------------------------------------

        refined_errors = (
            correspondence_ground_truth_errors(

                refined_source,

                refined_reference,

                ground_truth_transform
            )
        )


        refined_summary = (
            calculate_error_summary(
                refined_errors
            )
        )


        # ----------------------------------------------------
        # RE-ESTIMATE TRANSFORMATION FROM REFINED MATCHES
        # ----------------------------------------------------

        try:

            (
                refined_transform,
                refined_inlier_mask,
                refined_inlier_source,
                refined_inlier_reference

            ) = filter_matches_ransac(

                refined_source,

                refined_reference,

                reprojection_threshold=
                    1.5
            )

        except RuntimeError as error:

            print(
                "Refined RANSAC failed:",
                error
            )

            continue


        refined_registration = (
            transformation_ground_truth_error(

                refined_transform,

                ground_truth_transform,

                source_valid_mask,

                reference_valid_mask,

                grid_step=64
            )
        )


        # ----------------------------------------------------
        # COVERAGE
        # ----------------------------------------------------

        coverage = (
            calculate_spatial_coverage(

                refined_inlier_source,

                source.shape,

                grid_rows=8,

                grid_cols=8
            )
        )


        # ----------------------------------------------------
        # OUTPUT DIRECTORY
        # ----------------------------------------------------

        output_dir = (
            matcher_dir
            / "refinement"
        )


        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # SAVE CORRESPONDENCE TABLE
        # ----------------------------------------------------

        refined_df = pd.DataFrame({

            "source_x":
                refined_source[:, 0],

            "source_y":
                refined_source[:, 1],

            "raw_reference_x":
                raw_subset_reference[:, 0],

            "raw_reference_y":
                raw_subset_reference[:, 1],

            "geometric_reference_x":
                geometric_reference[:, 0],

            "geometric_reference_y":
                geometric_reference[:, 1],

            "refined_reference_x":
                refined_reference[:, 0],

            "refined_reference_y":
                refined_reference[:, 1],

            "phase_response":
                responses,

            "shift_x":
                local_shifts[:, 0],

            "shift_y":
                local_shifts[:, 1],

            "raw_gt_error_px":
                raw_subset_errors,

            "geometric_gt_error_px":
                geometric_errors,

            "refined_gt_error_px":
                refined_errors
        })


        refined_df.to_csv(

            output_dir
            / "refined_correspondences.csv",

            index=False
        )


        # ----------------------------------------------------
        # VISUALISE FINAL REFINED MATCHES
        # ----------------------------------------------------

        draw_matches(

            source,

            reference,

            refined_inlier_source,

            refined_inlier_reference,

            output_dir
            / "refined_matches.png",

            max_matches=150
        )


        # ----------------------------------------------------
        # REFINED REGISTRATION
        # ----------------------------------------------------

        registered = register_affine(

            source,

            refined_transform,

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
                / "registered_refined.png"
            ),

            registered
        )


        cv2.imwrite(

            str(
                output_dir
                / "overlay_refined.png"
            ),

            overlay
        )


        cv2.imwrite(

            str(
                output_dir
                / "difference_refined.png"
            ),

            difference
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        result = {

            "benchmark_id":
                benchmark_id,

            "matcher":
                matcher_name,

            "original_inlier_count":
                int(
                    len(source_points)
                ),

            "uniform_selected_count":
                int(
                    len(selected_source)
                ),

            "refined_count":
                int(
                    len(refined_source)
                ),

            "refined_ransac_inliers":
                int(
                    len(
                        refined_inlier_source
                    )
                ),

            "raw_selected_gt_rmse_px":
                raw_summary[
                    "rmse"
                ],

            "raw_same_subset_gt_rmse_px":
                raw_subset_summary[
                    "rmse"
                ],

            "geometric_prediction_gt_rmse_px":
                geometric_summary[
                    "rmse"
                ],

            "refined_correspondence_gt_rmse_px":
                refined_summary[
                    "rmse"
                ],

            "refined_registration_gt_rmse_px":
                refined_registration[
                    "rmse"
                ],

            "refined_registration_median_px":
                refined_registration[
                    "median"
                ],

            "spatial_coverage":
                coverage[
                    "coverage_ratio"
                ],

            "patch_size":
                PATCH_SIZE,

            "max_local_shift":
                MAX_LOCAL_SHIFT,

            "min_phase_response":
                MIN_PHASE_RESPONSE,

            "refined_transformation":
                refined_transform.tolist()
        }


        with open(

            output_dir
            / "refinement_metrics.json",

            "w"

        ) as file:

            json.dump(

                result,

                file,

                indent=4
            )


        all_results.append(
            result
        )


        print(
            "Raw correspondence RMSE:",
            raw_subset_summary["rmse"]
        )

        print(
            "Geometric prediction RMSE:",
            geometric_summary["rmse"]
        )

        print(
            "Refined correspondence RMSE:",
            refined_summary["rmse"]
        )

        print(
            "Refined registration RMSE:",
            refined_registration["rmse"]
        )

        print(
            "Coverage:",
            coverage["coverage_ratio"]
        )


# ============================================================
# MASTER TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)


output_path = (

    BENCHMARK_DIR
    / "refinement_benchmarks.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


print("\n")
print("=" * 70)

print(
    "SUBPIXEL REFINEMENT BENCHMARK COMPLETE"
)

print("=" * 70)


if len(results_df) > 0:

    show_columns = [

        "benchmark_id",

        "matcher",

        "uniform_selected_count",

        "refined_count",

        "raw_same_subset_gt_rmse_px",

        "geometric_prediction_gt_rmse_px",

        "refined_correspondence_gt_rmse_px",

        "refined_registration_gt_rmse_px",

        "spatial_coverage"
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
    output_path
)