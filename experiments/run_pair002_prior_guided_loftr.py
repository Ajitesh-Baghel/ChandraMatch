import sys
import json
import math
import time
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
# CHANDRAMATCH
# ============================================================

from src.matching.loftr_matcher import (
    LoFTRMatcher
)


# ============================================================
# INPUT PATHS
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


# Credible global solution obtained earlier.
SEED_METRICS_PATH = (
    ROOT
    / "results"
    / "pair_002"
    / "benchmarks"
    / "loftr_gradient"
    / "metrics.json"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_002"
    / "prior_guided_loftr"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "summary.json"
)


ALL_MATCHES_OUTPUT = (
    OUTPUT_DIR
    / "all_prior_gated_matches.csv"
)


INLIER_OUTPUT = (
    OUTPUT_DIR
    / "inlier_correspondences.csv"
)


UNIFORM_OUTPUT = (
    OUTPUT_DIR
    / "uniform_correspondences.csv"
)


CANDIDATE_VIS_OUTPUT = (
    OUTPUT_DIR
    / "candidate_matches.png"
)


INLIER_VIS_OUTPUT = (
    OUTPUT_DIR
    / "inlier_matches.png"
)


UNIFORM_VIS_OUTPUT = (
    OUTPUT_DIR
    / "uniform_matches.png"
)


REGISTERED_OUTPUT = (
    OUTPUT_DIR
    / "registered_source.png"
)


OVERLAY_OUTPUT = (
    OUTPUT_DIR
    / "registered_overlay.png"
)


CELL_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "cell_coverage.png"
)


MATCHING_MASK_OUTPUT = (
    OUTPUT_DIR
    / "matching_mask.png"
)


# ============================================================
# SETTINGS
# ============================================================

GRID_ROWS = 8
GRID_COLS = 8


# Local LoFTR patch centered on each valid 8×8 cell.
PATCH_SIZE = 288


# Require at least this much valid lunar terrain in a patch.
MIN_PATCH_VALID_RATIO = 0.08


# LoFTR
LOFTR_MAX_DIMENSION = 320

# We can lower confidence because the strong geometric prior
# will reject inconsistent correspondences afterward.
LOFTR_CONFIDENCE_THRESHOLD = 0.05


# ============================================================
# COARSE-TO-FINE PRIOR
#
# Candidate local matches must agree with the already credible
# global LoFTR affine to within this many pixels.
# ============================================================

MAX_PRIOR_RESIDUAL_PX = 5.0


# Final affine RANSAC
RANSAC_THRESHOLD_PX = 2.5
RANSAC_MAX_ITERS = 30000
RANSAC_CONFIDENCE = 0.999
RANSAC_REFINE_ITERS = 30


# Valid-mask boundary suppression
MASK_EROSION_RADIUS = 4


# Deduplication
DEDUPLICATION_RADIUS_PX = 1.5


# Uniform final correspondence selection
MAX_UNIFORM_PER_CELL = 8


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
            work
            - low
        )
        /
        (
            high
            - low
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
# GRADIENT PREPROCESSING
# ============================================================

def gradient_preprocess(
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


    output = normalize_uint8(
        magnitude,
        mask
    )


    output[
        ~mask
    ] = 0


    return output


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
# AFFINE TRANSFORM
# ============================================================

def transform_points(
    points,
    transform
):

    if len(points) == 0:

        return np.empty(
            (0, 2),
            dtype=np.float64
        )


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


# ============================================================
# GRID
# ============================================================

def cell_bounds(
    row,
    col,
    width,
    height
):

    x0 = int(
        round(
            col
            * width
            / GRID_COLS
        )
    )


    x1 = int(
        round(
            (col + 1)
            * width
            / GRID_COLS
        )
    )


    y0 = int(
        round(
            row
            * height
            / GRID_ROWS
        )
    )


    y1 = int(
        round(
            (row + 1)
            * height
            / GRID_ROWS
        )
    )


    return (
        x0,
        y0,
        x1,
        y1
    )


def point_cell(
    point,
    width,
    height
):

    x, y = point


    col = int(
        np.floor(
            x
            / width
            * GRID_COLS
        )
    )


    row = int(
        np.floor(
            y
            / height
            * GRID_ROWS
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


def valid_grid_cells(
    mask
):

    height, width = (
        mask.shape
    )


    valid = np.zeros(
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

            (
                x0,
                y0,
                x1,
                y1

            ) = cell_bounds(
                row,
                col,
                width,
                height
            )


            valid[
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


    return valid


# ============================================================
# PATCH AROUND GRID CELL
# ============================================================

def patch_for_cell(
    row,
    col,
    width,
    height
):

    (
        cell_x0,
        cell_y0,
        cell_x1,
        cell_y1

    ) = cell_bounds(
        row,
        col,
        width,
        height
    )


    centre_x = int(
        round(
            (
                cell_x0
                +
                cell_x1
            )
            / 2.0
        )
    )


    centre_y = int(
        round(
            (
                cell_y0
                +
                cell_y1
            )
            / 2.0
        )
    )


    half = (
        PATCH_SIZE // 2
    )


    x0 = (
        centre_x
        - half
    )


    y0 = (
        centre_y
        - half
    )


    x0 = int(
        np.clip(
            x0,
            0,
            max(
                0,
                width - PATCH_SIZE
            )
        )
    )


    y0 = int(
        np.clip(
            y0,
            0,
            max(
                0,
                height - PATCH_SIZE
            )
        )
    )


    x1 = min(
        width,
        x0 + PATCH_SIZE
    )


    y1 = min(
        height,
        y0 + PATCH_SIZE
    )


    return (
        x0,
        y0,
        x1,
        y1
    )


# ============================================================
# MASK FILTER
# ============================================================

def filter_patch_mask(
    source_points,
    reference_points,
    confidence,
    patch_mask
):

    source_points = np.asarray(
        source_points,
        dtype=np.float32
    ).reshape(
        -1,
        2
    )


    reference_points = np.asarray(
        reference_points,
        dtype=np.float32
    ).reshape(
        -1,
        2
    )


    confidence = np.asarray(
        confidence,
        dtype=np.float32
    ).reshape(
        -1
    )


    count = min(
        len(source_points),
        len(reference_points),
        len(confidence)
    )


    source_points = (
        source_points[
            :count
        ]
    )


    reference_points = (
        reference_points[
            :count
        ]
    )


    confidence = (
        confidence[
            :count
        ]
    )


    if count == 0:

        return (
            source_points,
            reference_points,
            confidence
        )


    height, width = (
        patch_mask.shape
    )


    sx = np.rint(
        source_points[:, 0]
    ).astype(
        np.int64
    )


    sy = np.rint(
        source_points[:, 1]
    ).astype(
        np.int64
    )


    rx = np.rint(
        reference_points[:, 0]
    ).astype(
        np.int64
    )


    ry = np.rint(
        reference_points[:, 1]
    ).astype(
        np.int64
    )


    inside = (

        (sx >= 0)
        &
        (sx < width)

        &

        (sy >= 0)
        &
        (sy < height)

        &

        (rx >= 0)
        &
        (rx < width)

        &

        (ry >= 0)
        &
        (ry < height)
    )


    valid = np.zeros(
        count,
        dtype=bool
    )


    indices = np.flatnonzero(
        inside
    )


    if len(indices) > 0:

        valid[
            indices
        ] = (

            patch_mask[
                sy[
                    indices
                ],
                sx[
                    indices
                ]
            ]

            &

            patch_mask[
                ry[
                    indices
                ],
                rx[
                    indices
                ]
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
# PRIOR FILTER
# ============================================================

def filter_by_seed_transform(
    source_points,
    reference_points,
    confidence,
    seed_transform
):

    if len(source_points) == 0:

        return (
            source_points,
            reference_points,
            confidence,
            np.empty(
                (0,),
                dtype=np.float64
            )
        )


    predicted_reference = transform_points(
        source_points,
        seed_transform
    )


    prior_residual = np.linalg.norm(

        predicted_reference
        -
        reference_points,

        axis=1
    )


    accepted = (
        prior_residual
        <= MAX_PRIOR_RESIDUAL_PX
    )


    return (
        source_points[
            accepted
        ],
        reference_points[
            accepted
        ],
        confidence[
            accepted
        ],
        prior_residual[
            accepted
        ]
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_matches(
    source_points,
    reference_points,
    confidence,
    prior_residual
):

    if len(source_points) == 0:

        return (
            source_points,
            reference_points,
            confidence,
            prior_residual
        )


    # Prefer:
    # 1. smaller coarse-prior residual
    # 2. larger LoFTR confidence

    order = np.lexsort(
        (
            -confidence,
            prior_residual
        )
    )


    source_points = (
        source_points[
            order
        ]
    )


    reference_points = (
        reference_points[
            order
        ]
    )


    confidence = (
        confidence[
            order
        ]
    )


    prior_residual = (
        prior_residual[
            order
        ]
    )


    selected_source = []
    selected_reference = []
    selected_confidence = []
    selected_prior = []


    seen = set()


    radius = float(
        DEDUPLICATION_RADIUS_PX
    )


    for (
        source_point,
        reference_point,
        score,
        residual

    ) in zip(
        source_points,
        reference_points,
        confidence,
        prior_residual
    ):

        key = (

            int(
                round(
                    source_point[0]
                    / radius
                )
            ),

            int(
                round(
                    source_point[1]
                    / radius
                )
            ),

            int(
                round(
                    reference_point[0]
                    / radius
                )
            ),

            int(
                round(
                    reference_point[1]
                    / radius
                )
            )
        )


        if key in seen:

            continue


        seen.add(
            key
        )


        selected_source.append(
            source_point
        )


        selected_reference.append(
            reference_point
        )


        selected_confidence.append(
            score
        )


        selected_prior.append(
            residual
        )


    return (
        np.asarray(
            selected_source,
            dtype=np.float32
        ),

        np.asarray(
            selected_reference,
            dtype=np.float32
        ),

        np.asarray(
            selected_confidence,
            dtype=np.float32
        ),

        np.asarray(
            selected_prior,
            dtype=np.float64
        )
    )


# ============================================================
# RANSAC
# ============================================================

def estimate_affine(
    source_points,
    reference_points
):

    if len(source_points) < 3:

        return (
            None,
            np.zeros(
                len(source_points),
                dtype=bool
            )
        )


    transform, mask = cv2.estimateAffine2D(

        source_points,

        reference_points,

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
        transform is None
        or
        mask is None
    ):

        return (
            None,
            np.zeros(
                len(source_points),
                dtype=bool
            )
        )


    return (
        transform.astype(
            np.float64
        ),
        mask.reshape(
            -1
        ).astype(
            bool
        )
    )


# ============================================================
# RESIDUALS
# ============================================================

def compute_residuals(
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
# COVERAGE
# ============================================================

def coverage_metrics(
    points,
    valid_cells,
    width,
    height
):

    occupied = np.zeros_like(
        valid_cells
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
            valid_cells
        )
    )


    occupied_valid = int(
        np.count_nonzero(
            occupied
            &
            valid_cells
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
            else 0.0,

        "full_grid_coverage":
            float(
                np.count_nonzero(
                    occupied
                )
                /
                (
                    GRID_ROWS
                    * GRID_COLS
                )
            )
    }


# ============================================================
# UNIFORM SELECTION
# ============================================================

def uniform_select(
    source_points,
    reference_points,
    confidence,
    residuals,
    width,
    height
):

    groups = {}


    for index, point in enumerate(
        source_points
    ):

        key = point_cell(
            point,
            width,
            height
        )


        groups.setdefault(
            key,
            []
        ).append(
            index
        )


    selected = []


    for indices in (
        groups.values()
    ):

        indices = np.asarray(
            indices,
            dtype=np.int64
        )


        order = np.lexsort(
            (
                -confidence[
                    indices
                ],
                residuals[
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
# AFFINE DIAGNOSTICS
# ============================================================

def affine_diagnostics(
    transform
):

    linear = transform[
        :,
        :2
    ]


    translation = transform[
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


    return {

        "translation_x_px":
            float(
                translation[0]
            ),

        "translation_y_px":
            float(
                translation[1]
            ),

        "translation_magnitude_px":
            float(
                np.linalg.norm(
                    translation
                )
            ),

        "rotation_deg":
            rotation_deg,

        "scale_x":
            scale_x,

        "scale_y":
            scale_y,

        "determinant":
            float(
                np.linalg.det(
                    linear
                )
            ),

        "identity_deviation":
            float(
                np.linalg.norm(
                    linear
                    -
                    np.eye(2)
                )
            )
    }


# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check(
    diagnostics
):

    translation_ok = (
        diagnostics[
            "translation_magnitude_px"
        ]
        <= 10.0
    )


    rotation_ok = (
        abs(
            diagnostics[
                "rotation_deg"
            ]
        )
        <= 2.0
    )


    scale_x_ok = (
        0.97
        <= diagnostics[
            "scale_x"
        ]
        <= 1.03
    )


    scale_y_ok = (
        0.97
        <= diagnostics[
            "scale_y"
        ]
        <= 1.03
    )


    determinant_ok = (
        diagnostics[
            "determinant"
        ]
        > 0.0
    )


    accepted = (

        translation_ok
        and
        rotation_ok
        and
        scale_x_ok
        and
        scale_y_ok
        and
        determinant_ok
    )


    return {

        "accepted":
            bool(
                accepted
            ),

        "translation_ok":
            bool(
                translation_ok
            ),

        "rotation_ok":
            bool(
                rotation_ok
            ),

        "scale_x_ok":
            bool(
                scale_x_ok
            ),

        "scale_y_ok":
            bool(
                scale_y_ok
            ),

        "determinant_ok":
            bool(
                determinant_ok
            )
    }


# ============================================================
# DRAW MATCHES
# ============================================================

def draw_matches(
    source,
    reference,
    source_points,
    reference_points,
    output_path,
    max_matches=600
):

    source_bgr = cv2.cvtColor(
        source,
        cv2.COLOR_GRAY2BGR
    )


    reference_bgr = cv2.cvtColor(
        reference,
        cv2.COLOR_GRAY2BGR
    )


    height = max(
        source.shape[0],
        reference.shape[0]
    )


    source_width = (
        source.shape[1]
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
        source.shape[1]:
    ] = reference_bgr


    if len(source_points) == 0:

        cv2.imwrite(
            str(output_path),
            canvas
        )

        return


    count = min(
        len(source_points),
        max_matches
    )


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
        str(output_path),
        canvas
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 002 PRIOR-GUIDED LoFTR"
    )

    print("=" * 80)


    # ========================================================
    # LOAD PAIR
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


    if (
        source.shape
        != reference.shape
        or
        source.shape
        != mask_image.shape
    ):

        raise RuntimeError(
            "Pair images/mask have different shapes."
        )


    height, width = (
        source.shape
    )


    matching_mask = build_matching_mask(
        mask_image
    )


    cv2.imwrite(
        str(
            MATCHING_MASK_OUTPUT
        ),
        (
            matching_mask.astype(
                np.uint8
            )
            * 255
        )
    )


    print(
        "\nPair shape:",
        source.shape
    )


    print(
        "Matching valid ratio:",
        float(
            matching_mask.mean()
        )
    )


    # ========================================================
    # LOAD CREDIBLE GLOBAL LoFTR PRIOR
    # ========================================================

    if not SEED_METRICS_PATH.exists():

        raise RuntimeError(
            "Global LoFTR-gradient metrics not found:\n"
            f"{SEED_METRICS_PATH}"
        )


    with open(
        SEED_METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        seed_metrics = json.load(
            file
        )


    seed_transform = np.asarray(
        seed_metrics[
            "affine_transform"
        ],
        dtype=np.float64
    )


    print("\n")
    print("=" * 80)
    print("GLOBAL COARSE PRIOR")
    print("=" * 80)


    print(
        seed_transform
    )


    print(
        "Prior RMSE:",
        seed_metrics.get(
            "ransac_reprojection_rmse_px"
        )
    )


    print(
        "Prior inliers:",
        seed_metrics.get(
            "inliers"
        )
    )


    # ========================================================
    # BASELINE + GRADIENT REPRESENTATIONS
    # ========================================================

    source_baseline = (
        source.copy()
    )


    reference_baseline = (
        reference.copy()
    )


    source_baseline[
        ~matching_mask
    ] = 0


    reference_baseline[
        ~matching_mask
    ] = 0


    source_gradient = gradient_preprocess(
        source,
        matching_mask
    )


    reference_gradient = gradient_preprocess(
        reference,
        matching_mask
    )


    # ========================================================
    # VALID CELLS
    # ========================================================

    grid_valid = valid_grid_cells(
        matching_mask
    )


    valid_cell_count = int(
        np.count_nonzero(
            grid_valid
        )
    )


    print(
        "\nValid 8x8 cells:",
        valid_cell_count
    )


    # ========================================================
    # LoFTR
    # ========================================================

    matcher = LoFTRMatcher(

        max_dimension=
            LOFTR_MAX_DIMENSION,

        confidence_threshold=
            LOFTR_CONFIDENCE_THRESHOLD
    )


    all_source = []
    all_reference = []
    all_confidence = []
    all_prior_residual = []
    all_mode = []


    cell_visual = cv2.cvtColor(
        source,
        cv2.COLOR_GRAY2BGR
    )


    processed_cells = 0
    cells_with_matches = set()


    start_time = time.perf_counter()


    # ========================================================
    # PROCESS EACH VALID 8x8 CELL
    # ========================================================

    for row in range(
        GRID_ROWS
    ):

        for col in range(
            GRID_COLS
        ):

            if not grid_valid[
                row,
                col
            ]:

                continue


            (
                x0,
                y0,
                x1,
                y1

            ) = patch_for_cell(
                row,
                col,
                width,
                height
            )


            patch_mask = matching_mask[
                y0:y1,
                x0:x1
            ]


            patch_valid_ratio = float(
                patch_mask.mean()
            )


            if (
                patch_valid_ratio
                <
                MIN_PATCH_VALID_RATIO
            ):

                continue


            processed_cells += 1


            cell_match_count = 0


            # =================================================
            # RUN BOTH APPEARANCE REPRESENTATIONS
            # =================================================

            modes = [

                (
                    "baseline",
                    source_baseline,
                    reference_baseline
                ),

                (
                    "gradient",
                    source_gradient,
                    reference_gradient
                )
            ]


            for (
                mode_name,
                source_representation,
                reference_representation

            ) in modes:

                source_patch = (
                    source_representation[
                        y0:y1,
                        x0:x1
                    ]
                    .copy()
                )


                reference_patch = (
                    reference_representation[
                        y0:y1,
                        x0:x1
                    ]
                    .copy()
                )


                source_patch[
                    ~patch_mask
                ] = 0


                reference_patch[
                    ~patch_mask
                ] = 0


                try:

                    (
                        patch_source,
                        patch_reference,
                        patch_confidence

                    ) = matcher.match(

                        source_patch,

                        reference_patch
                    )


                except Exception as error:

                    print(
                        f"Cell ({row},{col}) "
                        f"{mode_name} error: {error}"
                    )

                    continue


                (
                    patch_source,
                    patch_reference,
                    patch_confidence

                ) = filter_patch_mask(

                    patch_source,

                    patch_reference,

                    patch_confidence,

                    patch_mask
                )


                if len(
                    patch_source
                ) == 0:

                    continue


                # Convert patch-local → full Pair 002 coords.

                global_source = (
                    patch_source.copy()
                )


                global_reference = (
                    patch_reference.copy()
                )


                global_source[
                    :,
                    0
                ] += x0


                global_source[
                    :,
                    1
                ] += y0


                global_reference[
                    :,
                    0
                ] += x0


                global_reference[
                    :,
                    1
                ] += y0


                # =============================================
                # CRITICAL COARSE-TO-FINE GEOMETRIC FILTER
                # =============================================

                (
                    global_source,
                    global_reference,
                    patch_confidence,
                    prior_residual

                ) = filter_by_seed_transform(

                    global_source,

                    global_reference,

                    patch_confidence,

                    seed_transform
                )


                if len(
                    global_source
                ) == 0:

                    continue


                cell_match_count += len(
                    global_source
                )


                all_source.append(
                    global_source
                )


                all_reference.append(
                    global_reference
                )


                all_confidence.append(
                    patch_confidence
                )


                all_prior_residual.append(
                    prior_residual
                )


                all_mode.extend(
                    [
                        mode_name
                    ]
                    * len(
                        global_source
                    )
                )


            if cell_match_count > 0:

                cells_with_matches.add(
                    (
                        row,
                        col
                    )
                )


                (
                    cx0,
                    cy0,
                    cx1,
                    cy1

                ) = cell_bounds(
                    row,
                    col,
                    width,
                    height
                )


                cv2.rectangle(
                    cell_visual,
                    (
                        cx0,
                        cy0
                    ),
                    (
                        cx1 - 1,
                        cy1 - 1
                    ),
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )


                print(
                    f"Cell ({row},{col}): "
                    f"{cell_match_count} prior-consistent matches"
                )


            else:

                (
                    cx0,
                    cy0,
                    cx1,
                    cy1

                ) = cell_bounds(
                    row,
                    col,
                    width,
                    height
                )


                cv2.rectangle(
                    cell_visual,
                    (
                        cx0,
                        cy0
                    ),
                    (
                        cx1 - 1,
                        cy1 - 1
                    ),
                    (
                        0,
                        0,
                        255
                    ),
                    1
                )


    matcher_runtime = (
        time.perf_counter()
        -
        start_time
    )


    cv2.imwrite(
        str(
            CELL_COVERAGE_OUTPUT
        ),
        cell_visual
    )


    print("\n")
    print("=" * 80)
    print("LOCAL SEARCH COMPLETE")
    print("=" * 80)


    print(
        "Processed valid cells:",
        processed_cells
    )


    print(
        "Cells with prior-consistent matches:",
        len(
            cells_with_matches
        )
    )


    # ========================================================
    # MERGE
    # ========================================================

    if len(
        all_source
    ) == 0:

        raise RuntimeError(
            "No prior-consistent local matches found."
        )


    source_points = np.vstack(
        all_source
    ).astype(
        np.float32
    )


    reference_points = np.vstack(
        all_reference
    ).astype(
        np.float32
    )


    confidence = np.concatenate(
        all_confidence
    ).astype(
        np.float32
    )


    prior_residual = np.concatenate(
        all_prior_residual
    ).astype(
        np.float64
    )


    print(
        "\nPrior-gated matches before deduplication:",
        len(
            source_points
        )
    )


    # ========================================================
    # DEDUPLICATE
    # ========================================================

    (
        source_points,
        reference_points,
        confidence,
        prior_residual

    ) = deduplicate_matches(

        source_points,

        reference_points,

        confidence,

        prior_residual
    )


    print(
        "Prior-gated matches after deduplication:",
        len(
            source_points
        )
    )


    # ========================================================
    # SAVE ALL GATED MATCHES
    # ========================================================

    all_df = pd.DataFrame(
        {
            "source_x":
                source_points[:, 0],

            "source_y":
                source_points[:, 1],

            "reference_x":
                reference_points[:, 0],

            "reference_y":
                reference_points[:, 1],

            "confidence":
                confidence,

            "prior_residual_px":
                prior_residual
        }
    )


    all_df.to_csv(
        ALL_MATCHES_OUTPUT,
        index=False
    )


    draw_matches(
        source,
        reference,
        source_points,
        reference_points,
        CANDIDATE_VIS_OUTPUT
    )


    # ========================================================
    # FINAL RANSAC
    # ========================================================

    transform, inlier_mask = estimate_affine(

        source_points,

        reference_points
    )


    if transform is None:

        raise RuntimeError(
            "Final affine RANSAC failed."
        )


    source_inliers = (
        source_points[
            inlier_mask
        ]
    )


    reference_inliers = (
        reference_points[
            inlier_mask
        ]
    )


    confidence_inliers = (
        confidence[
            inlier_mask
        ]
    )


    prior_inliers = (
        prior_residual[
            inlier_mask
        ]
    )


    residuals = compute_residuals(

        source_inliers,

        reference_inliers,

        transform
    )


    inlier_count = int(
        len(
            source_inliers
        )
    )


    inlier_ratio = float(

        inlier_count

        /
        len(
            source_points
        )
    )


    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )


    median_residual = float(
        np.median(
            residuals
        )
    )


    mean_residual = float(
        np.mean(
            residuals
        )
    )


    # ========================================================
    # COVERAGE
    # ========================================================

    coverage = coverage_metrics(

        source_inliers,

        grid_valid,

        width,

        height
    )


    # ========================================================
    # AFFINE SANITY
    # ========================================================

    diagnostics = affine_diagnostics(
        transform
    )


    sanity = sanity_check(
        diagnostics
    )


    # ========================================================
    # UNIFORM SELECTION
    # ========================================================

    uniform_indices = uniform_select(

        source_inliers,

        reference_inliers,

        confidence_inliers,

        residuals,

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


    residual_uniform = (
        residuals[
            uniform_indices
        ]
    )


    uniform_coverage = coverage_metrics(

        source_uniform,

        grid_valid,

        width,

        height
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\n")
    print("=" * 80)
    print("PRIOR-GUIDED LoFTR RESULT")
    print("=" * 80)


    print(
        "Candidate matches:",
        len(
            source_points
        )
    )


    print(
        "RANSAC inliers:",
        inlier_count
    )


    print(
        "Inlier ratio:",
        inlier_ratio
    )


    print(
        "RANSAC reprojection RMSE:",
        rmse,
        "px"
    )


    print(
        "Median residual:",
        median_residual,
        "px"
    )


    print(
        "Mean residual:",
        mean_residual,
        "px"
    )


    print(
        "\nValid grid cells:",
        coverage[
            "valid_cells"
        ]
    )


    print(
        "Occupied valid cells:",
        coverage[
            "occupied_valid_cells"
        ]
    )


    print(
        "Valid-region coverage:",
        coverage[
            "valid_region_coverage"
        ]
    )


    print(
        "\nEstimated affine:"
    )


    print(
        transform
    )


    print(
        "\nTranslation magnitude:",
        diagnostics[
            "translation_magnitude_px"
        ],
        "px"
    )


    print(
        "Rotation:",
        diagnostics[
            "rotation_deg"
        ],
        "deg"
    )


    print(
        "Scale X:",
        diagnostics[
            "scale_x"
        ]
    )


    print(
        "Scale Y:",
        diagnostics[
            "scale_y"
        ]
    )


    print(
        "Geometric sanity accepted:",
        sanity[
            "accepted"
        ]
    )


    print(
        "\nUniform correspondences:",
        len(
            source_uniform
        )
    )


    print(
        "Uniform valid-region coverage:",
        uniform_coverage[
            "valid_region_coverage"
        ]
    )


    # ========================================================
    # SAVE INLIER TABLE
    # ========================================================

    inlier_df = pd.DataFrame(
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

            "prior_residual_px":
                prior_inliers,

            "final_residual_px":
                residuals
        }
    )


    inlier_df.to_csv(
        INLIER_OUTPUT,
        index=False
    )


    # ========================================================
    # SAVE UNIFORM TABLE
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
                residual_uniform
        }
    )


    uniform_df.to_csv(
        UNIFORM_OUTPUT,
        index=False
    )


    # ========================================================
    # VISUALIZATIONS
    # ========================================================

    draw_matches(
        source,
        reference,
        source_inliers,
        reference_inliers,
        INLIER_VIS_OUTPUT
    )


    draw_matches(
        source,
        reference,
        source_uniform,
        reference_uniform,
        UNIFORM_VIS_OUTPUT
    )


    # ========================================================
    # REGISTRATION
    # ========================================================

    registered = cv2.warpAffine(

        source,

        transform.astype(
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
                "prior_guided_cellwise_"
                "loftr_baseline_gradient_fusion"
            ),

        "coarse_prior_source":
            str(
                SEED_METRICS_PATH
            ),

        "coarse_prior_transform":
            seed_transform.tolist(),

        "patch_size":
            PATCH_SIZE,

        "prior_gate_px":
            MAX_PRIOR_RESIDUAL_PX,

        "valid_grid_cells":
            valid_cell_count,

        "processed_cells":
            processed_cells,

        "cells_with_local_matches":
            len(
                cells_with_matches
            ),

        "candidate_matches":
            int(
                len(
                    source_points
                )
            ),

        "ransac_inliers":
            inlier_count,

        "inlier_ratio":
            inlier_ratio,

        # Self-consistency only, NOT GT accuracy.
        "ransac_reprojection_rmse_px":
            rmse,

        "ransac_median_residual_px":
            median_residual,

        "ransac_mean_residual_px":
            mean_residual,

        "coverage":
            coverage,

        "uniform_correspondences":
            int(
                len(
                    source_uniform
                )
            ),

        "uniform_coverage":
            uniform_coverage,

        "affine_transform":
            transform.tolist(),

        "affine_diagnostics":
            diagnostics,

        "geometric_sanity":
            sanity,

        "runtime_sec":
            float(
                matcher_runtime
            )
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


    if sanity[
        "accepted"
    ]:

        print(
            "PRIOR-GUIDED LoFTR RESULT: ACCEPTED"
        )

    else:

        print(
            "PRIOR-GUIDED LoFTR RESULT: REJECTED"
        )


    print("=" * 80)


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_OUTPUT
    )


    print(
        "\nInlier visualization:"
    )

    print(
        INLIER_VIS_OUTPUT
    )


    print(
        "\nUniform visualization:"
    )

    print(
        UNIFORM_VIS_OUTPUT
    )


    print(
        "\nRegistered overlay:"
    )

    print(
        OVERLAY_OUTPUT
    )


if __name__ == "__main__":

    main()