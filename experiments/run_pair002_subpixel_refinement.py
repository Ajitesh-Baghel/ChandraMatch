import sys
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(ROOT)
)


# ============================================================
# INPUT
# ============================================================

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_002"
)


SOURCE_PATH = (
    PAIR_DIR
    / "source.png"
)


REFERENCE_PATH = (
    PAIR_DIR
    / "reference.png"
)


VALID_MASK_PATH = (
    PAIR_DIR
    / "valid_mask.png"
)


PRIOR_DIR = (
    ROOT
    / "results"
    / "pair_002"
    / "prior_guided_loftr"
)


INLIER_INPUT = (
    PRIOR_DIR
    / "inlier_correspondences.csv"
)


PRIOR_SUMMARY_INPUT = (
    PRIOR_DIR
    / "summary.json"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_002"
    / "subpixel_refinement"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


REFINED_CSV_OUTPUT = (
    OUTPUT_DIR
    / "refined_correspondences.csv"
)


UNIFORM_CSV_OUTPUT = (
    OUTPUT_DIR
    / "uniform_refined_correspondences.csv"
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "summary.json"
)


MATCH_VIS_OUTPUT = (
    OUTPUT_DIR
    / "refined_matches.png"
)


UNIFORM_VIS_OUTPUT = (
    OUTPUT_DIR
    / "uniform_refined_matches.png"
)


REGISTERED_OUTPUT = (
    OUTPUT_DIR
    / "registered_source.png"
)


OVERLAY_OUTPUT = (
    OUTPUT_DIR
    / "registered_overlay.png"
)


# ============================================================
# SETTINGS
# ============================================================

PATCH_RADIUS = 24

PATCH_SIZE = (
    PATCH_RADIUS * 2
    + 1
)


MAX_LOCAL_CORRECTION_PX = 3.0


MIN_PHASE_RESPONSE = 0.04


RANSAC_THRESHOLD_PX = 2.5

RANSAC_MAX_ITERS = 30000

RANSAC_CONFIDENCE = 0.999

RANSAC_REFINE_ITERS = 30


GRID_ROWS = 8

GRID_COLS = 8

MAX_UNIFORM_PER_CELL = 8


MASK_EROSION_RADIUS = 4


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_uint8(
    image,
    mask
):

    output = np.zeros(
        image.shape,
        dtype=np.uint8
    )


    values = image[
        mask
    ]


    if values.size == 0:

        return output


    low, high = np.percentile(
        values,
        [1, 99]
    )


    if high <= low:

        return output


    work = np.clip(
        image.astype(
            np.float32
        ),
        low,
        high
    )


    work = (
        (
            work - low
        )
        /
        (
            high - low
        )
        * 255.0
    )


    output[
        mask
    ] = work[
        mask
    ].astype(
        np.uint8
    )


    return output


# ============================================================
# GRADIENT REPRESENTATION
# ============================================================

def gradient_image(
    image,
    mask
):

    work = image.astype(
        np.float32
    )


    work = cv2.GaussianBlur(
        work,
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0
    )


    gx = cv2.Sobel(
        work,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )


    gy = cv2.Sobel(
        work,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )


    magnitude = cv2.magnitude(
        gx,
        gy
    )


    result = normalize_uint8(
        magnitude,
        mask
    )


    result[
        ~mask
    ] = 0


    return result.astype(
        np.float32
    )


# ============================================================
# MATCHING MASK
# ============================================================

def build_matching_mask(
    mask_image
):

    binary = (
        mask_image > 0
    ).astype(
        np.uint8
    )


    diameter = (
        MASK_EROSION_RADIUS
        * 2
        + 1
    )


    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            diameter,
            diameter
        )
    )


    eroded = cv2.erode(
        binary,
        kernel,
        iterations=1
    )


    return (
        eroded > 0
    )


# ============================================================
# AFFINE HELPERS
# ============================================================

def transform_points(
    points,
    transform
):

    homogeneous = np.column_stack(
        (
            points.astype(
                np.float64
            ),
            np.ones(
                len(points),
                dtype=np.float64
            )
        )
    )


    return (
        homogeneous
        @ transform.T
    )


def residuals(
    source_points,
    reference_points,
    transform
):

    predicted = transform_points(
        source_points,
        transform
    )


    return np.linalg.norm(
        predicted
        -
        reference_points,
        axis=1
    )


# ============================================================
# PATCH EXTRACTION
# ============================================================

def extract_patch(
    image,
    x,
    y
):

    x = float(x)
    y = float(y)


    height, width = (
        image.shape
    )


    if (
        x - PATCH_RADIUS < 0
        or
        x + PATCH_RADIUS >= width
        or
        y - PATCH_RADIUS < 0
        or
        y + PATCH_RADIUS >= height
    ):

        return None


    patch = cv2.getRectSubPix(
        image,
        (
            PATCH_SIZE,
            PATCH_SIZE
        ),
        (
            x,
            y
        )
    )


    return patch.astype(
        np.float32
    )


# ============================================================
# PHASE-CORRELATION REFINEMENT
# ============================================================

def refine_one(
    source_gradient,
    reference_gradient,
    source_point,
    reference_point
):

    source_patch = extract_patch(
        source_gradient,
        source_point[0],
        source_point[1]
    )


    reference_patch = extract_patch(
        reference_gradient,
        reference_point[0],
        reference_point[1]
    )


    if (
        source_patch is None
        or
        reference_patch is None
    ):

        return (
            reference_point.copy(),
            0.0,
            0.0,
            0.0,
            False
        )


    # Remove local DC component.

    source_patch = (
        source_patch
        -
        np.mean(
            source_patch
        )
    )


    reference_patch = (
        reference_patch
        -
        np.mean(
            reference_patch
        )
    )


    # Hann window improves phase-correlation stability.

    window = cv2.createHanningWindow(
        (
            PATCH_SIZE,
            PATCH_SIZE
        ),
        cv2.CV_32F
    )


    source_patch = (
        source_patch
        * window
    )


    reference_patch = (
        reference_patch
        * window
    )


    (
        shift,
        response

    ) = cv2.phaseCorrelate(
        source_patch,
        reference_patch
    )


    dx = float(
        shift[0]
    )


    dy = float(
        shift[1]
    )


    response = float(
        response
    )


    correction = math.sqrt(
        dx * dx
        +
        dy * dy
    )


    accepted = bool(

        np.isfinite(dx)

        and
        np.isfinite(dy)

        and
        np.isfinite(response)

        and
        correction
        <= MAX_LOCAL_CORRECTION_PX

        and
        response
        >= MIN_PHASE_RESPONSE
    )


    if not accepted:

        return (
            reference_point.copy(),
            dx,
            dy,
            response,
            False
        )


    refined_reference = np.array(
        [
            reference_point[0]
            +
            dx,

            reference_point[1]
            +
            dy
        ],
        dtype=np.float64
    )


    return (
        refined_reference,
        dx,
        dy,
        response,
        True
    )


# ============================================================
# GRID COVERAGE
# ============================================================

def point_cell(
    point,
    width,
    height
):

    col = int(
        np.floor(
            point[0]
            /
            width
            *
            GRID_COLS
        )
    )


    row = int(
        np.floor(
            point[1]
            /
            height
            *
            GRID_ROWS
        )
    )


    col = int(
        np.clip(
            col,
            0,
            GRID_COLS - 1
        )
    )


    row = int(
        np.clip(
            row,
            0,
            GRID_ROWS - 1
        )
    )


    return (
        row,
        col
    )


def valid_grid(
    mask
):

    height, width = (
        mask.shape
    )


    result = np.zeros(
        (
            GRID_ROWS,
            GRID_COLS
        ),
        dtype=bool
    )


    for row in range(
        GRID_ROWS
    ):

        for col in range(
            GRID_COLS
        ):

            y0 = int(
                round(
                    row
                    *
                    height
                    /
                    GRID_ROWS
                )
            )


            y1 = int(
                round(
                    (row + 1)
                    *
                    height
                    /
                    GRID_ROWS
                )
            )


            x0 = int(
                round(
                    col
                    *
                    width
                    /
                    GRID_COLS
                )
            )


            x1 = int(
                round(
                    (col + 1)
                    *
                    width
                    /
                    GRID_COLS
                )
            )


            result[
                row,
                col
            ] = bool(
                np.any(
                    mask[
                        y0:y1,
                        x0:x1
                    ]
                )
            )


    return result


def coverage(
    points,
    mask
):

    height, width = (
        mask.shape
    )


    valid = valid_grid(
        mask
    )


    occupied = np.zeros_like(
        valid
    )


    for point in points:

        row, col = point_cell(
            point,
            width,
            height
        )


        occupied[
            row,
            col
        ] = True


    valid_count = int(
        np.count_nonzero(
            valid
        )
    )


    occupied_valid = int(
        np.count_nonzero(
            valid
            &
            occupied
        )
    )


    return {

        "valid_cells":
            valid_count,

        "occupied_valid_cells":
            occupied_valid,

        "valid_region_coverage":
            float(
                occupied_valid
                /
                valid_count
            )
            if valid_count > 0
            else 0.0
    }


# ============================================================
# UNIFORM SELECTION
# ============================================================

def uniform_select(
    source_points,
    reference_points,
    confidence,
    errors,
    width,
    height
):

    groups = {}


    for index, point in enumerate(
        source_points
    ):

        cell = point_cell(
            point,
            width,
            height
        )


        groups.setdefault(
            cell,
            []
        ).append(
            index
        )


    selected = []


    for indices in groups.values():

        indices = np.asarray(
            indices,
            dtype=np.int64
        )


        order = np.lexsort(
            (
                -confidence[
                    indices
                ],
                errors[
                    indices
                ]
            )
        )


        selected.extend(
            indices[
                order[
                    :MAX_UNIFORM_PER_CELL
                ]
            ].tolist()
        )


    return np.asarray(
        selected,
        dtype=np.int64
    )


# ============================================================
# DRAW MATCHES
# ============================================================

def draw_matches(
    source,
    reference,
    source_points,
    reference_points,
    path,
    max_matches=500
):

    source_bgr = cv2.cvtColor(
        source,
        cv2.COLOR_GRAY2BGR
    )


    reference_bgr = cv2.cvtColor(
        reference,
        cv2.COLOR_GRAY2BGR
    )


    source_width = (
        source.shape[1]
    )


    height = max(
        source.shape[0],
        reference.shape[0]
    )


    canvas = np.zeros(
        (
            height,
            source.shape[1]
            +
            reference.shape[1],
            3
        ),
        dtype=np.uint8
    )


    canvas[
        :source.shape[0],
        :source.shape[1]
    ] = source_bgr


    canvas[
        :reference.shape[0],
        source_width:
    ] = reference_bgr


    count = min(
        len(source_points),
        max_matches
    )


    if count > 0:

        indices = np.linspace(
            0,
            len(source_points) - 1,
            count
        ).astype(
            np.int64
        )


        for index in indices:

            p0 = source_points[
                index
            ]


            p1 = reference_points[
                index
            ]


            pt0 = (
                int(round(p0[0])),
                int(round(p0[1]))
            )


            pt1 = (
                int(round(p1[0]))
                +
                source_width,

                int(round(p1[1]))
            )


            cv2.line(
                canvas,
                pt0,
                pt1,
                (
                    0,
                    255,
                    0
                ),
                1,
                cv2.LINE_AA
            )


            cv2.circle(
                canvas,
                pt0,
                2,
                (
                    255,
                    255,
                    255
                ),
                -1
            )


            cv2.circle(
                canvas,
                pt1,
                2,
                (
                    255,
                    255,
                    255
                ),
                -1
            )


    cv2.imwrite(
        str(path),
        canvas
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 002 SUBPIXEL REFINEMENT"
    )

    print("=" * 80)


    # ========================================================
    # LOAD IMAGES
    # ========================================================

    source = cv2.imread(
        str(SOURCE_PATH),
        cv2.IMREAD_GRAYSCALE
    )


    reference = cv2.imread(
        str(REFERENCE_PATH),
        cv2.IMREAD_GRAYSCALE
    )


    mask_image = cv2.imread(
        str(VALID_MASK_PATH),
        cv2.IMREAD_GRAYSCALE
    )


    if (
        source is None
        or
        reference is None
        or
        mask_image is None
    ):

        raise RuntimeError(
            "Could not load Pair 002."
        )


    if not INLIER_INPUT.exists():

        raise RuntimeError(
            "Prior-guided inlier table missing:\n"
            f"{INLIER_INPUT}"
        )


    if not PRIOR_SUMMARY_INPUT.exists():

        raise RuntimeError(
            "Prior-guided summary missing:\n"
            f"{PRIOR_SUMMARY_INPUT}"
        )


    height, width = (
        source.shape
    )


    matching_mask = build_matching_mask(
        mask_image
    )


    # ========================================================
    # LOAD PRIOR-GUIDED MATCHES
    # ========================================================

    dataframe = pd.read_csv(
        INLIER_INPUT
    )


    source_points = dataframe[
        [
            "source_x",
            "source_y"
        ]
    ].to_numpy(
        dtype=np.float64
    )


    reference_points = dataframe[
        [
            "reference_x",
            "reference_y"
        ]
    ].to_numpy(
        dtype=np.float64
    )


    if "confidence" in dataframe.columns:

        confidence = dataframe[
            "confidence"
        ].to_numpy(
            dtype=np.float64
        )

    else:

        confidence = np.ones(
            len(source_points),
            dtype=np.float64
        )


    with open(
        PRIOR_SUMMARY_INPUT,
        "r",
        encoding="utf-8"
    ) as file:

        prior_summary = json.load(
            file
        )


    prior_transform = np.asarray(
        prior_summary[
            "affine_transform"
        ],
        dtype=np.float64
    )


    print(
        "\nInput correspondences:",
        len(source_points)
    )


    print(
        "Prior affine:"
    )

    print(
        prior_transform
    )


    # ========================================================
    # GRADIENT IMAGES
    # ========================================================

    source_gradient = gradient_image(
        source,
        matching_mask
    )


    reference_gradient = gradient_image(
        reference,
        matching_mask
    )


    # ========================================================
    # LOCAL REFINEMENT
    # ========================================================

    refined_reference = []

    accepted_flags = []

    corrections_x = []

    corrections_y = []

    responses = []


    for (
        source_point,
        reference_point

    ) in zip(
        source_points,
        reference_points
    ):

        (
            refined,
            dx,
            dy,
            response,
            accepted

        ) = refine_one(
            source_gradient,
            reference_gradient,
            source_point,
            reference_point
        )


        refined_reference.append(
            refined
        )


        corrections_x.append(
            dx
        )


        corrections_y.append(
            dy
        )


        responses.append(
            response
        )


        accepted_flags.append(
            accepted
        )


    refined_reference = np.asarray(
        refined_reference,
        dtype=np.float64
    )


    accepted_flags = np.asarray(
        accepted_flags,
        dtype=bool
    )


    corrections_x = np.asarray(
        corrections_x,
        dtype=np.float64
    )


    corrections_y = np.asarray(
        corrections_y,
        dtype=np.float64
    )


    responses = np.asarray(
        responses,
        dtype=np.float64
    )


    print(
        "\nPhase-refinement accepted:",
        int(
            np.count_nonzero(
                accepted_flags
            )
        ),
        "/",
        len(
            accepted_flags
        )
    )


    # ========================================================
    # CONSERVATIVE GATING
    #
    # If local phase refinement fails, KEEP the original
    # correspondence rather than replacing it with a bad shift.
    # ========================================================

    final_reference = (
        reference_points.copy()
    )


    final_reference[
        accepted_flags
    ] = refined_reference[
        accepted_flags
    ]


    # ========================================================
    # RE-ESTIMATE AFFINE
    # ========================================================

    (
        refined_transform,
        ransac_mask

    ) = cv2.estimateAffine2D(

        source_points.astype(
            np.float32
        ),

        final_reference.astype(
            np.float32
        ),

        method=
            cv2.RANSAC,

        ransacReprojThreshold=
            RANSAC_THRESHOLD_PX,

        maxIters=
            RANSAC_MAX_ITERS,

        confidence=
            RANSAC_CONFIDENCE,

        refineIters=
            RANSAC_REFINE_ITERS
    )


    if (
        refined_transform is None
        or
        ransac_mask is None
    ):

        raise RuntimeError(
            "Refined affine estimation failed."
        )


    refined_transform = (
        refined_transform.astype(
            np.float64
        )
    )


    ransac_mask = (
        ransac_mask.reshape(
            -1
        ).astype(
            bool
        )
    )


    source_inliers = (
        source_points[
            ransac_mask
        ]
    )


    reference_inliers = (
        final_reference[
            ransac_mask
        ]
    )


    confidence_inliers = (
        confidence[
            ransac_mask
        ]
    )


    accepted_inliers = (
        accepted_flags[
            ransac_mask
        ]
    )


    correction_x_inliers = (
        corrections_x[
            ransac_mask
        ]
    )


    correction_y_inliers = (
        corrections_y[
            ransac_mask
        ]
    )


    response_inliers = (
        responses[
            ransac_mask
        ]
    )


    final_errors = residuals(
        source_inliers,
        reference_inliers,
        refined_transform
    )


    rmse = float(
        np.sqrt(
            np.mean(
                final_errors ** 2
            )
        )
    )


    median_error = float(
        np.median(
            final_errors
        )
    )


    mean_error = float(
        np.mean(
            final_errors
        )
    )


    print("\n")
    print("=" * 80)
    print("REFINED RESULT")
    print("=" * 80)


    print(
        "Final RANSAC inliers:",
        len(source_inliers)
    )


    print(
        "RANSAC reprojection RMSE:",
        rmse,
        "px"
    )


    print(
        "Median residual:",
        median_error,
        "px"
    )


    print(
        "Mean residual:",
        mean_error,
        "px"
    )


    print(
        "\nRefined affine:"
    )

    print(
        refined_transform
    )


    # ========================================================
    # AFFINE DIAGNOSTICS
    # ========================================================

    linear = refined_transform[
        :,
        :2
    ]


    translation = refined_transform[
        :,
        2
    ]


    scale_x = float(
        np.linalg.norm(
            linear[:, 0]
        )
    )


    scale_y = float(
        np.linalg.norm(
            linear[:, 1]
        )
    )


    rotation_deg = float(
        math.degrees(
            math.atan2(
                linear[1, 0],
                linear[0, 0]
            )
        )
    )


    translation_magnitude = float(
        np.linalg.norm(
            translation
        )
    )


    print(
        "Translation magnitude:",
        translation_magnitude,
        "px"
    )


    print(
        "Rotation:",
        rotation_deg,
        "deg"
    )


    print(
        "Scale X:",
        scale_x
    )


    print(
        "Scale Y:",
        scale_y
    )


    # ========================================================
    # COVERAGE
    # ========================================================

    coverage_result = coverage(
        source_inliers,
        matching_mask
    )


    print(
        "\nValid-region coverage:",
        coverage_result[
            "valid_region_coverage"
        ]
    )


    # ========================================================
    # UNIFORM SELECTION
    # ========================================================

    uniform_indices = uniform_select(

        source_inliers,

        reference_inliers,

        confidence_inliers,

        final_errors,

        width,

        height
    )


    source_uniform = (
        source_inliers[
            uniform_indices
        ]
    )


    reference_uniform = (
        reference_inliers[
            uniform_indices
        ]
    )


    confidence_uniform = (
        confidence_inliers[
            uniform_indices
        ]
    )


    error_uniform = (
        final_errors[
            uniform_indices
        ]
    )


    uniform_coverage_result = coverage(
        source_uniform,
        matching_mask
    )


    print(
        "Uniform correspondences:",
        len(source_uniform)
    )


    print(
        "Uniform valid-region coverage:",
        uniform_coverage_result[
            "valid_region_coverage"
        ]
    )


    # ========================================================
    # SAVE ALL REFINED INLIERS
    # ========================================================

    output_df = pd.DataFrame(
        {
            "source_x":
                source_inliers[:, 0],

            "source_y":
                source_inliers[:, 1],

            "reference_x":
                reference_inliers[:, 0],

            "reference_y":
                reference_inliers[:, 1],

            "confidence":
                confidence_inliers,

            "phase_refined":
                accepted_inliers,

            "phase_dx":
                correction_x_inliers,

            "phase_dy":
                correction_y_inliers,

            "phase_response":
                response_inliers,

            "final_residual_px":
                final_errors
        }
    )


    output_df.to_csv(
        REFINED_CSV_OUTPUT,
        index=False
    )


    # ========================================================
    # SAVE UNIFORM
    # ========================================================

    uniform_df = pd.DataFrame(
        {
            "source_x":
                source_uniform[:, 0],

            "source_y":
                source_uniform[:, 1],

            "reference_x":
                reference_uniform[:, 0],

            "reference_y":
                reference_uniform[:, 1],

            "confidence":
                confidence_uniform,

            "final_residual_px":
                error_uniform
        }
    )


    uniform_df.to_csv(
        UNIFORM_CSV_OUTPUT,
        index=False
    )


    # ========================================================
    # VISUALS
    # ========================================================

    draw_matches(
        source,
        reference,
        source_inliers,
        reference_inliers,
        MATCH_VIS_OUTPUT
    )


    draw_matches(
        source,
        reference,
        source_uniform,
        reference_uniform,
        UNIFORM_VIS_OUTPUT
    )


    # ========================================================
    # REGISTERED OUTPUT
    # ========================================================

    registered = cv2.warpAffine(

        source,

        refined_transform.astype(
            np.float32
        ),

        (
            width,
            height
        ),

        flags=
            cv2.INTER_LINEAR,

        borderMode=
            cv2.BORDER_CONSTANT,

        borderValue=0
    )


    registered[
        ~matching_mask
    ] = 0


    cv2.imwrite(
        str(
            REGISTERED_OUTPUT
        ),
        registered
    )


    overlay = cv2.addWeighted(

        registered,
        0.5,

        reference,
        0.5,

        0
    )


    overlay[
        ~matching_mask
    ] = 0


    cv2.imwrite(
        str(
            OVERLAY_OUTPUT
        ),
        overlay
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "pair_id":
            "pair_002",

        "method":
            (
                "prior_guided_loftr_"
                "with_gated_phase_subpixel_refinement"
            ),

        "input_correspondences":
            int(
                len(source_points)
            ),

        "phase_refinement_accepted":
            int(
                np.count_nonzero(
                    accepted_flags
                )
            ),

        "phase_refinement_accept_ratio":
            float(
                np.mean(
                    accepted_flags
                )
            ),

        "final_ransac_inliers":
            int(
                len(source_inliers)
            ),

        # Model residual, not independent GT.
        "ransac_reprojection_rmse_px":
            rmse,

        "ransac_median_residual_px":
            median_error,

        "ransac_mean_residual_px":
            mean_error,

        "coverage":
            coverage_result,

        "uniform_correspondences":
            int(
                len(source_uniform)
            ),

        "uniform_coverage":
            uniform_coverage_result,

        "affine_transform":
            refined_transform.tolist(),

        "affine_diagnostics": {

            "translation_magnitude_px":
                translation_magnitude,

            "rotation_deg":
                rotation_deg,

            "scale_x":
                scale_x,

            "scale_y":
                scale_y
        },

        "settings": {

            "patch_radius":
                PATCH_RADIUS,

            "max_local_correction_px":
                MAX_LOCAL_CORRECTION_PX,

            "min_phase_response":
                MIN_PHASE_RESPONSE
        }
    }


    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )


    print("\n")
    print("=" * 80)
    print(
        "PAIR 002 SUBPIXEL REFINEMENT COMPLETE"
    )
    print("=" * 80)


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_OUTPUT
    )


    print(
        "\nRefined correspondences:"
    )

    print(
        REFINED_CSV_OUTPUT
    )


    print(
        "\nUniform correspondences:"
    )

    print(
        UNIFORM_CSV_OUTPUT
    )


    print(
        "\nRegistered overlay:"
    )

    print(
        OVERLAY_OUTPUT
    )


if __name__ == "__main__":

    main()