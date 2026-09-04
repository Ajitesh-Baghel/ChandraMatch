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
# EXISTING CHANDRAMATCH MATCHER
# ============================================================

from src.matching.loftr_matcher import (
    LoFTRMatcher
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


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_002"
    / "guided_loftr"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "guided_loftr_summary.json"
)


CORRESPONDENCE_OUTPUT = (
    OUTPUT_DIR
    / "guided_loftr_correspondences.csv"
)


UNIFORM_OUTPUT = (
    OUTPUT_DIR
    / "guided_loftr_uniform_correspondences.csv"
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


TILE_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "tile_coverage.png"
)


MATCHING_MASK_OUTPUT = (
    OUTPUT_DIR
    / "matching_mask.png"
)


# ============================================================
# SETTINGS
# ============================================================

# Local patch size.
PATCH_SIZE = 320


# Distance between neighboring patch centres.
#
# PATCH_SIZE = 320
# STRIDE = 160
#
# means 50% overlap.
STRIDE = 160


# Require this fraction of each patch to contain valid data.
MIN_PATCH_VALID_RATIO = 0.12


# LoFTR
LOFTR_MAX_DIMENSION = 320

LOFTR_CONFIDENCE_THRESHOLD = 0.10


# Since both images are already geolocated onto the same
# TMC grid, enormous displacements are impossible for a
# correct local match.
MAX_INITIAL_DISPLACEMENT_PX = 20.0


# Remove duplicate correspondences from overlapping tiles.
DEDUPLICATION_RADIUS_PX = 2.0


# RANSAC
RANSAC_THRESHOLD_PX = 3.0

RANSAC_MAX_ITERS = 20000

RANSAC_CONFIDENCE = 0.999

RANSAC_REFINE_ITERS = 20


# Matching mask erosion.
MASK_EROSION_RADIUS = 4


# Coverage grid.
GRID_ROWS = 8

GRID_COLS = 8

MAX_UNIFORM_PER_CELL = 5


# ============================================================
# IMAGE PREPROCESSING
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
# TILE POSITIONS
# ============================================================

def generate_tile_positions(
    width,
    height
):

    positions = []


    half = (
        PATCH_SIZE // 2
    )


    centre_x_values = list(
        range(
            half,
            width - half + 1,
            STRIDE
        )
    )


    centre_y_values = list(
        range(
            half,
            height - half + 1,
            STRIDE
        )
    )


    # Make sure right/bottom edges are also represented.

    final_x = (
        width - half
    )

    final_y = (
        height - half
    )


    if (
        final_x >= half
        and
        final_x not in centre_x_values
    ):

        centre_x_values.append(
            final_x
        )


    if (
        final_y >= half
        and
        final_y not in centre_y_values
    ):

        centre_y_values.append(
            final_y
        )


    for centre_y in (
        centre_y_values
    ):

        for centre_x in (
            centre_x_values
        ):

            x0 = (
                centre_x
                - half
            )

            y0 = (
                centre_y
                - half
            )


            x1 = (
                x0
                + PATCH_SIZE
            )

            y1 = (
                y0
                + PATCH_SIZE
            )


            positions.append(
                (
                    x0,
                    y0,
                    x1,
                    y1
                )
            )


    return positions


# ============================================================
# FILTER PATCH MATCHES
# ============================================================

def filter_patch_matches(
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


    # --------------------------------------------------------
    # GEOSPATIAL PRIOR
    # --------------------------------------------------------

    displacement = np.linalg.norm(

        reference_points
        -
        source_points,

        axis=1
    )


    valid &= (
        displacement
        <= MAX_INITIAL_DISPLACEMENT_PX
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
# DEDUPLICATE OVERLAPPING TILE MATCHES
# ============================================================

def deduplicate_matches(
    source_points,
    reference_points,
    confidence
):

    if len(source_points) == 0:

        return (
            source_points,
            reference_points,
            confidence
        )


    # Highest confidence first.

    order = np.argsort(
        -confidence
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


    selected_source = []

    selected_reference = []

    selected_confidence = []


    # Quantized source/reference pair key.
    #
    # Using both points avoids accidentally deleting nearby
    # but genuinely different correspondences.

    seen = set()


    radius = float(
        DEDUPLICATION_RADIUS_PX
    )


    for source_point, reference_point, score in zip(

        source_points,
        reference_points,
        confidence
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
        )
    )


# ============================================================
# RANSAC
# ============================================================

def run_ransac(
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


    transform, inlier_mask = (
        cv2.estimateAffine2D(

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
    )


    if (
        transform is None
        or
        inlier_mask is None
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

        inlier_mask.reshape(
            -1
        ).astype(
            bool
        )
    )


# ============================================================
# RESIDUALS
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
# GRID COVERAGE
# ============================================================

def get_valid_grid(
    mask
):

    height, width = (
        mask.shape
    )


    valid_grid = np.zeros(
        (
            GRID_ROWS,
            GRID_COLS
        ),
        dtype=bool
    )


    for row in range(
        GRID_ROWS
    ):

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


        for col in range(
            GRID_COLS
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


            valid_grid[
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


    return valid_grid


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


def coverage_metrics(
    points,
    mask
):

    height, width = (
        mask.shape
    )


    occupied = np.zeros(
        (
            GRID_ROWS,
            GRID_COLS
        ),
        dtype=bool
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


    valid_grid = get_valid_grid(
        mask
    )


    valid_cells = int(
        np.count_nonzero(
            valid_grid
        )
    )


    occupied_valid = int(
        np.count_nonzero(
            occupied
            &
            valid_grid
        )
    )


    occupied_total = int(
        np.count_nonzero(
            occupied
        )
    )


    return {

        "valid_grid_cells":
            valid_cells,

        "occupied_valid_cells":
            occupied_valid,

        "occupied_total_cells":
            occupied_total,

        "valid_region_coverage":
            float(
                occupied_valid
                /
                valid_cells
            )
            if valid_cells > 0
            else 0.0,

        "full_grid_coverage":
            float(
                occupied_total
                /
                (
                    GRID_ROWS
                    * GRID_COLS
                )
            )
    }


# ============================================================
# UNIFORM MATCH SELECTION
# ============================================================

def select_uniform(
    source_points,
    reference_points,
    confidence,
    residuals,
    mask
):

    height, width = (
        mask.shape
    )


    cells = {}


    for index, point in enumerate(
        source_points
    ):

        cell = point_cell(
            point,
            width,
            height
        )


        cells.setdefault(
            cell,
            []
        ).append(
            index
        )


    selected = []


    for indices in (
        cells.values()
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


        chosen = indices[
            order[
                :MAX_UNIFORM_PER_CELL
            ]
        ]


        selected.extend(
            chosen.tolist()
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
            linear[
                :,
                0
            ]
        )
    )


    scale_y = float(
        np.linalg.norm(
            linear[
                :,
                1
            ]
        )
    )


    rotation = math.degrees(

        math.atan2(
            linear[1, 0],
            linear[0, 0]
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
            float(
                rotation
            ),

        "scale_x":
            scale_x,

        "scale_y":
            scale_y,

        "determinant":
            float(
                np.linalg.det(
                    linear
                )
            )
    }


# ============================================================
# GEOMETRIC SANITY
# ============================================================

def geometric_sanity(
    diagnostics
):

    translation_ok = (
        diagnostics[
            "translation_magnitude_px"
        ]
        <= 20.0
    )


    rotation_ok = (
        abs(
            diagnostics[
                "rotation_deg"
            ]
        )
        <= 5.0
    )


    scale_x_ok = (
        0.95
        <= diagnostics[
            "scale_x"
        ]
        <= 1.05
    )


    scale_y_ok = (
        0.95
        <= diagnostics[
            "scale_y"
        ]
        <= 1.05
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
# VISUALIZATION
# ============================================================

def draw_matches(
    source,
    reference,
    source_points,
    reference_points,
    output_path,
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


    count = min(
        len(source_points),
        max_matches
    )


    if count == 0:

        cv2.imwrite(
            str(
                output_path
            ),
            canvas
        )

        return


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
        str(
            output_path
        ),
        canvas
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 002 GUIDED TILED LoFTR"
    )

    print("=" * 80)


    # ========================================================
    # LOAD
    # ========================================================

    source = cv2.imread(
        str(
            SOURCE_PATH
        ),
        cv2.IMREAD_GRAYSCALE
    )


    reference = cv2.imread(
        str(
            REFERENCE_PATH
        ),
        cv2.IMREAD_GRAYSCALE
    )


    mask_image = cv2.imread(
        str(
            VALID_MASK_PATH
        ),
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
            "Could not load Pair 002 files."
        )


    if (
        source.shape
        != reference.shape
        or
        source.shape
        != mask_image.shape
    ):

        raise RuntimeError(
            "Pair 002 image/mask shapes differ."
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


    # ========================================================
    # GRADIENT REPRESENTATION
    # ========================================================

    source_gradient = gradient_preprocess(
        source,
        matching_mask
    )


    reference_gradient = gradient_preprocess(
        reference,
        matching_mask
    )


    height, width = (
        source.shape
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
    # LoFTR
    # ========================================================

    matcher = LoFTRMatcher(

        max_dimension=
            LOFTR_MAX_DIMENSION,

        confidence_threshold=
            LOFTR_CONFIDENCE_THRESHOLD
    )


    # ========================================================
    # TILES
    # ========================================================

    tiles = generate_tile_positions(
        width,
        height
    )


    print(
        "\nCandidate tiles:",
        len(
            tiles
        )
    )


    tile_visual = cv2.cvtColor(
        source,
        cv2.COLOR_GRAY2BGR
    )


    all_source = []

    all_reference = []

    all_confidence = []


    processed_tiles = 0

    matched_tiles = 0


    start_time = time.perf_counter()


    for tile_index, (
        x0,
        y0,
        x1,
        y1

    ) in enumerate(
        tiles,
        start=1
    ):

        patch_mask = matching_mask[
            y0:y1,
            x0:x1
        ]


        valid_ratio = float(
            patch_mask.mean()
        )


        if (
            valid_ratio
            <
            MIN_PATCH_VALID_RATIO
        ):

            continue


        processed_tiles += 1


        source_patch = (
            source_gradient[
                y0:y1,
                x0:x1
            ]
            .copy()
        )


        reference_patch = (
            reference_gradient[
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
                f"Tile {tile_index}: matcher error:",
                error
            )

            continue


        (
            patch_source,
            patch_reference,
            patch_confidence

        ) = filter_patch_matches(

            patch_source,

            patch_reference,

            patch_confidence,

            patch_mask
        )


        if len(
            patch_source
        ) == 0:

            cv2.rectangle(
                tile_visual,
                (
                    x0,
                    y0
                ),
                (
                    x1 - 1,
                    y1 - 1
                ),
                (
                    0,
                    0,
                    255
                ),
                1
            )

            continue


        matched_tiles += 1


        # Convert tile coordinates to global coordinates.

        patch_source[
            :,
            0
        ] += x0


        patch_source[
            :,
            1
        ] += y0


        patch_reference[
            :,
            0
        ] += x0


        patch_reference[
            :,
            1
        ] += y0


        all_source.append(
            patch_source
        )


        all_reference.append(
            patch_reference
        )


        all_confidence.append(
            patch_confidence
        )


        cv2.rectangle(
            tile_visual,
            (
                x0,
                y0
            ),
            (
                x1 - 1,
                y1 - 1
            ),
            (
                0,
                255,
                0
            ),
            2
        )


        print(
            f"Tile {tile_index}: "
            f"{len(patch_source)} accepted matches"
        )


    matcher_runtime = (
        time.perf_counter()
        -
        start_time
    )


    cv2.imwrite(
        str(
            TILE_COVERAGE_OUTPUT
        ),
        tile_visual
    )


    print(
        "\nProcessed valid tiles:",
        processed_tiles
    )


    print(
        "Tiles producing matches:",
        matched_tiles
    )


    # ========================================================
    # MERGE
    # ========================================================

    if len(
        all_source
    ) == 0:

        raise RuntimeError(
            "No guided LoFTR matches were produced."
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


    print(
        "\nMatches before deduplication:",
        len(
            source_points
        )
    )


    (
        source_points,
        reference_points,
        confidence

    ) = deduplicate_matches(

        source_points,

        reference_points,

        confidence
    )


    print(
        "Matches after deduplication:",
        len(
            source_points
        )
    )


    draw_matches(
        source,
        reference,
        source_points,
        reference_points,
        CANDIDATE_VIS_OUTPUT
    )


    # ========================================================
    # RANSAC
    # ========================================================

    transform, inlier_mask = run_ransac(

        source_points,

        reference_points
    )


    if transform is None:

        raise RuntimeError(
            "RANSAC failed."
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


    residuals = compute_residuals(

        source_inliers,

        reference_inliers,

        transform
    )


    inlier_count = len(
        source_inliers
    )


    inlier_ratio = (

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


    print("\n")
    print("=" * 80)
    print("GUIDED LoFTR RESULT")
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


    # ========================================================
    # COVERAGE
    # ========================================================

    coverage = coverage_metrics(

        source_inliers,

        matching_mask
    )


    print(
        "Valid grid cells:",
        coverage[
            "valid_grid_cells"
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


    # ========================================================
    # AFFINE SANITY
    # ========================================================

    diagnostics = affine_diagnostics(
        transform
    )


    sanity = geometric_sanity(
        diagnostics
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


    # ========================================================
    # UNIFORM SELECTION
    # ========================================================

    uniform_indices = select_uniform(

        source_inliers,

        reference_inliers,

        confidence_inliers,

        residuals,

        matching_mask
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

        matching_mask
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
    # SAVE TABLES
    # ========================================================

    correspondence_df = pd.DataFrame(
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

            "geometric_residual_px":
                residuals
        }
    )


    correspondence_df.to_csv(
        CORRESPONDENCE_OUTPUT,
        index=False
    )


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

            "geometric_residual_px":
                residual_uniform
        }
    )


    uniform_df.to_csv(
        UNIFORM_OUTPUT,
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
            "guided_tiled_loftr_gradient",

        "patch_size":
            PATCH_SIZE,

        "stride":
            STRIDE,

        "processed_tiles":
            processed_tiles,

        "matched_tiles":
            matched_tiles,

        "candidate_matches":
            int(
                len(
                    source_points
                )
            ),

        "ransac_inliers":
            int(
                inlier_count
            ),

        "inlier_ratio":
            float(
                inlier_ratio
            ),

        # Again: model self-consistency only.
        "ransac_reprojection_rmse_px":
            rmse,

        "ransac_mean_residual_px":
            mean_residual,

        "ransac_median_residual_px":
            median_residual,

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

        "matcher_runtime_sec":
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
            "GUIDED LoFTR GEOMETRIC RESULT: ACCEPTED"
        )

    else:

        print(
            "GUIDED LoFTR GEOMETRIC RESULT: REJECTED"
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