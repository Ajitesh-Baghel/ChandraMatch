import sys
import json
import math
import time
import inspect
import traceback

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))


# ============================================================
# CHANDRAMATCH MATCHERS
# ============================================================

from src.matching.lightglue_matcher import (
    SuperPointLightGlueMatcher
)

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

PAIR_METADATA_PATH = (
    PAIR_DIR
    / "pair_metadata.json"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_002"
    / "benchmarks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "pair002_benchmarks.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "pair002_benchmarks.json"
)

SUMMARY_MD = (
    OUTPUT_DIR
    / "pair002_benchmarks.md"
)


# ============================================================
# SETTINGS
# ============================================================

RANSAC_THRESHOLD_PX = 3.0

RANSAC_MAX_ITERS = 10000

RANSAC_CONFIDENCE = 0.999

RANSAC_REFINE_ITERS = 20


# Prevent matching against artificial black-mask edges.
MASK_EROSION_RADIUS = 4


# SIFT
SIFT_MAX_FEATURES = 12000

SIFT_RATIO_THRESHOLD = 0.80


# LightGlue
LIGHTGLUE_MAX_KEYPOINTS = 4096


# LoFTR
LOFTR_MAX_DIMENSION = 840

LOFTR_CONFIDENCE_THRESHOLD = 0.20


# Spatial distribution
GRID_ROWS = 8

GRID_COLS = 8

MAX_UNIFORM_PER_CELL = 5


# Visualization
MAX_VISUALIZED_MATCHES = 300


# ============================================================
# IMAGE HELPERS
# ============================================================

def ensure_grayscale_uint8(image):

    if image is None:

        raise RuntimeError(
            "Image could not be loaded."
        )

    if image.ndim == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    if image.dtype == np.uint8:

        return image

    image = image.astype(
        np.float32
    )

    finite = np.isfinite(
        image
    )

    if not np.any(finite):

        return np.zeros(
            image.shape,
            dtype=np.uint8
        )

    values = image[
        finite
    ]

    low, high = np.percentile(
        values,
        [1, 99]
    )

    if high <= low:

        return np.zeros(
            image.shape,
            dtype=np.uint8
        )

    image = np.clip(
        image,
        low,
        high
    )

    image = (
        (
            image
            - low
        )
        /
        (
            high
            - low
        )
        * 255.0
    )

    image[
        ~finite
    ] = 0

    return image.astype(
        np.uint8
    )


def normalize_uint8(
    image,
    mask
):

    image = image.astype(
        np.float32
    )

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
        image,
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
# PREPROCESSING
# ============================================================

def baseline_preprocess(
    image,
    mask
):

    output = image.copy()

    output[
        ~mask
    ] = 0

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


def preprocess_pair(
    source,
    reference,
    mask,
    mode
):

    if mode == "baseline":

        return (
            baseline_preprocess(
                source,
                mask
            ),
            baseline_preprocess(
                reference,
                mask
            )
        )

    if mode == "gradient":

        return (
            gradient_preprocess(
                source,
                mask
            ),
            gradient_preprocess(
                reference,
                mask
            )
        )

    raise ValueError(
        f"Unknown preprocessing mode: {mode}"
    )


# ============================================================
# VALID MASK
# ============================================================

def build_matching_mask(
    valid_mask
):

    binary = (
        valid_mask > 0
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
# SIFT
# ============================================================

def run_sift(
    source,
    reference,
    mask
):

    sift = cv2.SIFT_create(
        nfeatures=SIFT_MAX_FEATURES
    )

    mask_uint8 = (
        mask.astype(
            np.uint8
        )
        * 255
    )

    (
        source_keypoints,
        source_descriptors

    ) = sift.detectAndCompute(
        source,
        mask_uint8
    )

    (
        reference_keypoints,
        reference_descriptors

    ) = sift.detectAndCompute(
        reference,
        mask_uint8
    )

    if (
        source_descriptors is None
        or
        reference_descriptors is None
    ):

        return (
            np.empty(
                (0, 2),
                dtype=np.float32
            ),
            np.empty(
                (0, 2),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.float32
            ),
            {
                "source_keypoints":
                    len(source_keypoints),

                "reference_keypoints":
                    len(reference_keypoints)
            }
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    knn_matches = matcher.knnMatch(
        source_descriptors,
        reference_descriptors,
        k=2
    )

    accepted = []

    for pair in knn_matches:

        if len(pair) < 2:

            continue

        first, second = pair

        if (
            first.distance
            <
            SIFT_RATIO_THRESHOLD
            * second.distance
        ):

            accepted.append(
                first
            )

    source_points = np.array(
        [
            source_keypoints[
                match.queryIdx
            ].pt

            for match in accepted
        ],
        dtype=np.float32
    ).reshape(
        -1,
        2
    )

    reference_points = np.array(
        [
            reference_keypoints[
                match.trainIdx
            ].pt

            for match in accepted
        ],
        dtype=np.float32
    ).reshape(
        -1,
        2
    )

    confidence = np.array(
        [
            1.0
            /
            (
                1.0
                + match.distance
            )

            for match in accepted
        ],
        dtype=np.float32
    )

    return (
        source_points,
        reference_points,
        confidence,
        {
            "source_keypoints":
                len(source_keypoints),

            "reference_keypoints":
                len(reference_keypoints)
        }
    )


# ============================================================
# MATCHER CONSTRUCTION
# ============================================================

def construct_matcher(
    matcher_class,
    preferred_kwargs
):

    try:

        signature = inspect.signature(
            matcher_class.__init__
        )

        supported = {}

        for key, value in preferred_kwargs.items():

            if key in signature.parameters:

                supported[
                    key
                ] = value

        return matcher_class(
            **supported
        )

    except Exception:

        return matcher_class()


# ============================================================
# MATCHER OUTPUT NORMALIZATION
# ============================================================

def normalize_match_output(
    result
):

    source_points = None

    reference_points = None

    confidence = None


    if isinstance(
        result,
        dict
    ):

        source_keys = [
            "source_points",
            "points0",
            "keypoints0",
            "mkpts0",
            "mkpts0_f"
        ]

        reference_keys = [
            "reference_points",
            "points1",
            "keypoints1",
            "mkpts1",
            "mkpts1_f"
        ]

        confidence_keys = [
            "confidence",
            "scores",
            "matching_scores",
            "mconf"
        ]

        for key in source_keys:

            if key in result:

                source_points = result[
                    key
                ]

                break

        for key in reference_keys:

            if key in result:

                reference_points = result[
                    key
                ]

                break

        for key in confidence_keys:

            if key in result:

                confidence = result[
                    key
                ]

                break

    elif isinstance(
        result,
        (tuple, list)
    ):

        if len(result) >= 2:

            source_points = result[0]

            reference_points = result[1]

        if len(result) >= 3:

            confidence = result[2]

    else:

        raise RuntimeError(
            "Unknown matcher output type: "
            f"{type(result)}"
        )


    if (
        source_points is None
        or
        reference_points is None
    ):

        raise RuntimeError(
            "Could not extract match points "
            "from matcher output."
        )


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


    if confidence is None:

        confidence = np.ones(
            len(source_points),
            dtype=np.float32
        )

    else:

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


    return (
        source_points[
            :count
        ],
        reference_points[
            :count
        ],
        confidence[
            :count
        ]
    )


# ============================================================
# FILTER MATCHES USING VALID MASK
# ============================================================

def filter_matches_by_mask(
    source_points,
    reference_points,
    confidence,
    mask
):

    if len(source_points) == 0:

        return (
            source_points,
            reference_points,
            confidence
        )


    height, width = mask.shape


    source_x = np.rint(
        source_points[:, 0]
    ).astype(
        np.int64
    )

    source_y = np.rint(
        source_points[:, 1]
    ).astype(
        np.int64
    )

    reference_x = np.rint(
        reference_points[:, 0]
    ).astype(
        np.int64
    )

    reference_y = np.rint(
        reference_points[:, 1]
    ).astype(
        np.int64
    )


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


    valid = np.zeros(
        len(source_points),
        dtype=bool
    )

    indices = np.flatnonzero(
        inside
    )


    if len(indices) > 0:

        valid[
            indices
        ] = (

            mask[
                source_y[
                    indices
                ],
                source_x[
                    indices
                ]
            ]

            &

            mask[
                reference_y[
                    indices
                ],
                reference_x[
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


    transform, inlier_mask = cv2.estimateAffine2D(

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
# AFFINE POINT TRANSFORM
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


def compute_residuals(
    source_points,
    reference_points,
    transform
):

    predicted = transform_points(
        source_points,
        transform
    )


    differences = (
        predicted
        -
        reference_points
    )


    return np.linalg.norm(
        differences,
        axis=1
    )


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def point_to_grid_cell(
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


def build_valid_grid_cells(
    mask
):

    height, width = mask.shape


    valid_cells = np.zeros(
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


            valid_cells[
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


    return valid_cells


def compute_spatial_coverage(
    points,
    mask
):

    height, width = mask.shape


    occupied = np.zeros(
        (
            GRID_ROWS,
            GRID_COLS
        ),
        dtype=bool
    )


    for point in points:

        row, col = point_to_grid_cell(
            point,
            width,
            height
        )

        occupied[
            row,
            col
        ] = True


    valid_cells = build_valid_grid_cells(
        mask
    )


    occupied_total = int(
        np.count_nonzero(
            occupied
        )
    )


    valid_total = int(
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

        "occupied_cells":
            occupied_total,

        "valid_grid_cells":
            valid_total,

        "occupied_valid_cells":
            occupied_valid,

        "full_grid_coverage":
            float(
                occupied_total
                /
                (
                    GRID_ROWS
                    * GRID_COLS
                )
            ),

        "valid_region_coverage":
            float(
                occupied_valid
                /
                valid_total
            )
            if valid_total > 0
            else 0.0
    }


# ============================================================
# UNIFORM MATCH SELECTION
# ============================================================

def select_uniform_matches(
    source_points,
    reference_points,
    confidence,
    residuals,
    mask
):

    height, width = mask.shape


    cells = {}


    for index, point in enumerate(
        source_points
    ):

        row, col = point_to_grid_cell(
            point,
            width,
            height
        )


        key = (
            row,
            col
        )


        cells.setdefault(
            key,
            []
        ).append(
            index
        )


    selected = []


    for indices in cells.values():

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


    selected = np.asarray(
        selected,
        dtype=np.int64
    )


    if len(selected) > 0:

        selected = selected[
            np.argsort(
                residuals[
                    selected
                ]
            )
        ]


    return selected


# ============================================================
# AFFINE DIAGNOSTICS
# ============================================================

def affine_diagnostics(
    transform
):

    if transform is None:

        return {}


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


    determinant = float(
        np.linalg.det(
            linear
        )
    )


    identity_deviation = float(
        np.linalg.norm(
            linear
            -
            np.eye(2)
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

        "approx_rotation_deg":
            float(
                rotation
            ),

        "approx_scale_x":
            scale_x,

        "approx_scale_y":
            scale_y,

        "linear_determinant":
            determinant,

        "linear_identity_deviation":
            identity_deviation
    }


# ============================================================
# MATCH VISUALIZATION
# ============================================================

def draw_matches(
    source,
    reference,
    source_points,
    reference_points,
    output_path,
    max_matches=MAX_VISUALIZED_MATCHES
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


    source_width = source.shape[1]

    reference_width = reference.shape[1]


    canvas = np.zeros(
        (
            height,
            source_width
            + reference_width,
            3
        ),
        dtype=np.uint8
    )


    canvas[
        :source.shape[0],
        :source_width
    ] = source_bgr


    canvas[
        :reference.shape[0],
        source_width:
        source_width + reference_width
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

            source_point = (
                int(
                    round(
                        source_points[
                            index,
                            0
                        ]
                    )
                ),
                int(
                    round(
                        source_points[
                            index,
                            1
                        ]
                    )
                )
            )


            reference_point = (
                int(
                    round(
                        reference_points[
                            index,
                            0
                        ]
                    )
                )
                + source_width,
                int(
                    round(
                        reference_points[
                            index,
                            1
                        ]
                    )
                )
            )


            cv2.line(
                canvas,
                source_point,
                reference_point,
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
                source_point,
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
                reference_point,
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
# REGISTERED OUTPUT
# ============================================================

def save_registered_outputs(
    source,
    reference,
    mask,
    transform,
    registered_output,
    overlay_output
):

    if transform is None:

        return


    height, width = reference.shape


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
        ~mask
    ] = 0


    cv2.imwrite(
        str(
            registered_output
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
        ~mask
    ] = 0


    cv2.imwrite(
        str(
            overlay_output
        ),
        overlay
    )


# ============================================================
# SAVE CORRESPONDENCE TABLE
# ============================================================

def save_correspondence_csv(
    path,
    source_points,
    reference_points,
    confidence,
    residuals
):

    dataframe = pd.DataFrame(
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

            "geometric_residual_px":
                residuals
        }
    )


    dataframe.to_csv(
        path,
        index=False
    )


# ============================================================
# MATCHERS
# ============================================================

def execute_matcher(
    matcher_name,
    source,
    reference,
    mask,
    run_dir
):

    # --------------------------------------------------------
    # SIFT
    # --------------------------------------------------------

    if matcher_name == "sift":

        return run_sift(
            source,
            reference,
            mask
        )


    # --------------------------------------------------------
    # SUPERPOINT + LIGHTGLUE
    #
    # IMPORTANT:
    # Existing ChandraMatch LightGlue matcher expects paths,
    # not NumPy arrays.
    # --------------------------------------------------------

    if matcher_name == "lightglue":

        matcher = construct_matcher(

            SuperPointLightGlueMatcher,

            {
                "max_keypoints":
                    LIGHTGLUE_MAX_KEYPOINTS,

                "device":
                    None
            }
        )


        source_temp_path = (
            run_dir
            / "lightglue_source_input.png"
        )


        reference_temp_path = (
            run_dir
            / "lightglue_reference_input.png"
        )


        cv2.imwrite(
            str(
                source_temp_path
            ),
            source
        )


        cv2.imwrite(
            str(
                reference_temp_path
            ),
            reference
        )


        result = matcher.match(
            source_temp_path,
            reference_temp_path
        )


        (
            source_points,
            reference_points,
            confidence

        ) = normalize_match_output(
            result
        )


        return (
            source_points,
            reference_points,
            confidence,
            {}
        )


    # --------------------------------------------------------
    # LoFTR
    # --------------------------------------------------------

    if matcher_name == "loftr":

        matcher = construct_matcher(

            LoFTRMatcher,

            {
                "max_dimension":
                    LOFTR_MAX_DIMENSION,

                "confidence_threshold":
                    LOFTR_CONFIDENCE_THRESHOLD,

                "device":
                    None
            }
        )


        result = matcher.match(
            source,
            reference
        )


        (
            source_points,
            reference_points,
            confidence

        ) = normalize_match_output(
            result
        )


        return (
            source_points,
            reference_points,
            confidence,
            {}
        )


    raise ValueError(
        f"Unknown matcher: {matcher_name}"
    )


# ============================================================
# SINGLE BENCHMARK
# ============================================================

def run_benchmark(
    matcher_name,
    preprocessing_name,
    source_original,
    reference_original,
    matching_mask
):

    run_name = (
        f"{matcher_name}_{preprocessing_name}"
    )


    run_dir = (
        OUTPUT_DIR
        / run_name
    )


    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print("\n")
    print("=" * 80)

    print(
        f"RUN: {run_name}"
    )

    print("=" * 80)


    (
        source,
        reference

    ) = preprocess_pair(

        source_original,

        reference_original,

        matching_mask,

        preprocessing_name
    )


    cv2.imwrite(
        str(
            run_dir
            / "source_preprocessed.png"
        ),
        source
    )


    cv2.imwrite(
        str(
            run_dir
            / "reference_preprocessed.png"
        ),
        reference
    )


    # --------------------------------------------------------
    # MATCHER
    # --------------------------------------------------------

    start = time.perf_counter()


    (
        source_points,
        reference_points,
        confidence,
        matcher_extra

    ) = execute_matcher(

        matcher_name,

        source,

        reference,

        matching_mask,

        run_dir
    )


    matcher_runtime = (
        time.perf_counter()
        -
        start
    )


    raw_candidate_count = int(
        len(source_points)
    )


    print(
        "Raw matcher correspondences:",
        raw_candidate_count
    )


    # --------------------------------------------------------
    # MASK FILTER
    # --------------------------------------------------------

    (
        source_points,
        reference_points,
        confidence

    ) = filter_matches_by_mask(

        source_points,

        reference_points,

        confidence,

        matching_mask
    )


    candidate_count = int(
        len(source_points)
    )


    print(
        "Mask-valid correspondences:",
        candidate_count
    )


    draw_matches(

        source,

        reference,

        source_points,

        reference_points,

        run_dir
        / "candidate_matches.png"
    )


    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    ransac_start = time.perf_counter()


    transform, inlier_mask = run_ransac(

        source_points,

        reference_points
    )


    ransac_runtime = (
        time.perf_counter()
        -
        ransac_start
    )


    total_runtime = (
        matcher_runtime
        +
        ransac_runtime
    )


    inlier_count = int(
        np.count_nonzero(
            inlier_mask
        )
    )


    inlier_ratio = (
        inlier_count
        /
        candidate_count

        if candidate_count > 0

        else 0.0
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
    # RANSAC FAILURE
    # --------------------------------------------------------

    if (
        transform is None
        or
        inlier_count < 3
    ):

        result = {

            "pair_id":
                "pair_002",

            "matcher":
                matcher_name,

            "preprocessing":
                preprocessing_name,

            "status":
                "ransac_failed",

            "raw_candidate_matches":
                raw_candidate_count,

            "mask_valid_candidate_matches":
                candidate_count,

            "inliers":
                inlier_count,

            "inlier_ratio":
                float(
                    inlier_ratio
                ),

            "matcher_runtime_sec":
                float(
                    matcher_runtime
                ),

            "ransac_runtime_sec":
                float(
                    ransac_runtime
                ),

            "total_runtime_sec":
                float(
                    total_runtime
                )
        }


        result.update(
            matcher_extra
        )


        with open(
            run_dir
            / "metrics.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4
            )


        return result


    # --------------------------------------------------------
    # INLIERS
    # --------------------------------------------------------

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


    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )


    mean_residual = float(
        np.mean(
            residuals
        )
    )


    median_residual = float(
        np.median(
            residuals
        )
    )


    max_residual = float(
        np.max(
            residuals
        )
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


    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    coverage = compute_spatial_coverage(

        source_inliers,

        matching_mask
    )


    print(
        "8x8 full-grid coverage:",
        coverage[
            "full_grid_coverage"
        ]
    )


    print(
        "8x8 valid-region coverage:",
        coverage[
            "valid_region_coverage"
        ]
    )


    # --------------------------------------------------------
    # UNIFORM SUBSET
    # --------------------------------------------------------

    uniform_indices = select_uniform_matches(

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


    uniform_coverage = compute_spatial_coverage(

        source_uniform,

        matching_mask
    )


    print(
        "Uniform correspondences:",
        len(source_uniform)
    )


    print(
        "Uniform valid-region coverage:",
        uniform_coverage[
            "valid_region_coverage"
        ]
    )


    # --------------------------------------------------------
    # SAVE TABLES
    # --------------------------------------------------------

    save_correspondence_csv(

        run_dir
        / "inlier_correspondences.csv",

        source_inliers,

        reference_inliers,

        confidence_inliers,

        residuals
    )


    save_correspondence_csv(

        run_dir
        / "uniform_correspondences.csv",

        source_uniform,

        reference_uniform,

        confidence_uniform,

        residual_uniform
    )


    # --------------------------------------------------------
    # SAVE VISUALIZATIONS
    # --------------------------------------------------------

    draw_matches(

        source,

        reference,

        source_inliers,

        reference_inliers,

        run_dir
        / "inlier_matches.png"
    )


    draw_matches(

        source,

        reference,

        source_uniform,

        reference_uniform,

        run_dir
        / "uniform_matches.png"
    )


    save_registered_outputs(

        source_original,

        reference_original,

        matching_mask,

        transform,

        run_dir
        / "registered_source.png",

        run_dir
        / "registered_overlay.png"
    )


    # --------------------------------------------------------
    # AFFINE DIAGNOSTICS
    # --------------------------------------------------------

    affine_info = affine_diagnostics(
        transform
    )


    print(
        "Estimated transform:"
    )

    print(
        transform
    )


    print(
        "Translation magnitude:",
        affine_info.get(
            "translation_magnitude_px"
        ),
        "px"
    )


    print(
        "Approx rotation:",
        affine_info.get(
            "approx_rotation_deg"
        ),
        "deg"
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "pair_id":
            "pair_002",

        "matcher":
            matcher_name,

        "preprocessing":
            preprocessing_name,

        "status":
            "success",

        "raw_candidate_matches":
            raw_candidate_count,

        "mask_valid_candidate_matches":
            candidate_count,

        "inliers":
            inlier_count,

        "inlier_ratio":
            float(
                inlier_ratio
            ),

        # Self-consistency of the fitted RANSAC affine model.
        # NOT independent ground-truth accuracy.
        "ransac_reprojection_rmse_px":
            rmse,

        "ransac_mean_residual_px":
            mean_residual,

        "ransac_median_residual_px":
            median_residual,

        "ransac_max_residual_px":
            max_residual,

        "grid_rows":
            GRID_ROWS,

        "grid_cols":
            GRID_COLS,

        "occupied_grid_cells":
            coverage[
                "occupied_cells"
            ],

        "valid_grid_cells":
            coverage[
                "valid_grid_cells"
            ],

        "full_grid_coverage":
            coverage[
                "full_grid_coverage"
            ],

        "valid_region_coverage":
            coverage[
                "valid_region_coverage"
            ],

        "uniform_correspondences":
            int(
                len(source_uniform)
            ),

        "uniform_full_grid_coverage":
            uniform_coverage[
                "full_grid_coverage"
            ],

        "uniform_valid_region_coverage":
            uniform_coverage[
                "valid_region_coverage"
            ],

        "matcher_runtime_sec":
            float(
                matcher_runtime
            ),

        "ransac_runtime_sec":
            float(
                ransac_runtime
            ),

        "total_runtime_sec":
            float(
                total_runtime
            ),

        "affine_transform":
            transform.tolist(),

        "affine_diagnostics":
            affine_info
    }


    result.update(
        matcher_extra
    )


    with open(
        run_dir
        / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )


    return result


# ============================================================
# MARKDOWN SUMMARY
# ============================================================

def write_markdown_summary(
    dataframe,
    path
):

    lines = [

        "# Pair 002 Benchmark",

        "",

        (
            "Real Chandrayaan-2 OHRC ↔ TMC-2 "
            "cross-sensor correspondence benchmark."
        ),

        "",

        (
            "**Important:** "
            "`ransac_reprojection_rmse_px` measures "
            "self-consistency of the fitted affine model. "
            "It is not independent ground-truth "
            "registration accuracy."
        ),

        ""
    ]


    columns = [

        "matcher",
        "preprocessing",
        "status",
        "mask_valid_candidate_matches",
        "inliers",
        "inlier_ratio",
        "ransac_reprojection_rmse_px",
        "ransac_median_residual_px",
        "valid_region_coverage",
        "uniform_correspondences",
        "total_runtime_sec"
    ]


    existing = [

        column

        for column in columns

        if column in dataframe.columns
    ]


    table = dataframe[
        existing
    ]


    lines.append(
        table.to_markdown(
            index=False
        )
    )


    lines.append(
        ""
    )


    path.write_text(

        "\n".join(
            lines
        ),

        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 002 MATCHING BENCHMARK"
    )

    print("=" * 80)


    # ========================================================
    # LOAD PAIR
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


    valid_mask_image = cv2.imread(
        str(
            VALID_MASK_PATH
        ),
        cv2.IMREAD_GRAYSCALE
    )


    if source is None:

        raise RuntimeError(
            f"Could not load:\n"
            f"{SOURCE_PATH}"
        )


    if reference is None:

        raise RuntimeError(
            f"Could not load:\n"
            f"{REFERENCE_PATH}"
        )


    if valid_mask_image is None:

        raise RuntimeError(
            f"Could not load:\n"
            f"{VALID_MASK_PATH}"
        )


    source = ensure_grayscale_uint8(
        source
    )


    reference = ensure_grayscale_uint8(
        reference
    )


    if (
        source.shape
        != reference.shape
    ):

        raise RuntimeError(
            "Source and reference shapes differ."
        )


    if (
        source.shape
        != valid_mask_image.shape
    ):

        raise RuntimeError(
            "Validity-mask shape differs "
            "from image shape."
        )


    # ========================================================
    # MATCHING MASK
    # ========================================================

    matching_mask = build_matching_mask(
        valid_mask_image
    )


    print(
        "\nPair shape:",
        source.shape
    )


    print(
        "Original valid ratio:",
        float(
            np.mean(
                valid_mask_image > 0
            )
        )
    )


    print(
        "Matching-mask ratio after erosion:",
        float(
            np.mean(
                matching_mask
            )
        )
    )


    cv2.imwrite(

        str(
            OUTPUT_DIR
            / "matching_mask.png"
        ),

        (
            matching_mask.astype(
                np.uint8
            )
            * 255
        )
    )


    # ========================================================
    # SIX EXPERIMENTS
    # ========================================================

    experiments = [

        (
            "sift",
            "baseline"
        ),

        (
            "sift",
            "gradient"
        ),

        (
            "lightglue",
            "baseline"
        ),

        (
            "lightglue",
            "gradient"
        ),

        (
            "loftr",
            "baseline"
        ),

        (
            "loftr",
            "gradient"
        )
    ]


    results = []


    for (
        matcher_name,
        preprocessing_name

    ) in experiments:

        try:

            result = run_benchmark(

                matcher_name,

                preprocessing_name,

                source,

                reference,

                matching_mask
            )


        except Exception as error:

            print("\n")
            print(
                "ERROR DURING RUN:"
            )

            print(
                matcher_name,
                preprocessing_name
            )

            print(
                error
            )


            traceback.print_exc()


            result = {

                "pair_id":
                    "pair_002",

                "matcher":
                    matcher_name,

                "preprocessing":
                    preprocessing_name,

                "status":
                    "error",

                "error":
                    str(
                        error
                    )
            }


        results.append(
            result
        )


    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


    # ========================================================
    # SAVE CSV
    # ========================================================

    rows = []


    for result in results:

        row = {}

        for key, value in result.items():

            if isinstance(
                value,
                (
                    dict,
                    list
                )
            ):

                row[
                    key
                ] = json.dumps(
                    value
                )

            else:

                row[
                    key
                ] = value


        rows.append(
            row
        )


    dataframe = pd.DataFrame(
        rows
    )


    dataframe.to_csv(
        SUMMARY_CSV,
        index=False
    )


    # ========================================================
    # SAVE MARKDOWN
    # ========================================================

    try:

        write_markdown_summary(
            dataframe,
            SUMMARY_MD
        )

    except Exception as error:

        print(
            "\nMarkdown summary warning:",
            error
        )


    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n")

    print("=" * 110)

    print(
        "PAIR 002 BENCHMARK SUMMARY"
    )

    print("=" * 110)


    summary_columns = [

        "matcher",
        "preprocessing",
        "status",
        "mask_valid_candidate_matches",
        "inliers",
        "inlier_ratio",
        "ransac_reprojection_rmse_px",
        "ransac_median_residual_px",
        "valid_region_coverage",
        "uniform_correspondences",
        "total_runtime_sec"
    ]


    available_columns = [

        column

        for column in summary_columns

        if column in dataframe.columns
    ]


    if available_columns:

        print(
            dataframe[
                available_columns
            ].to_string(
                index=False
            )
        )


    print("\n")

    print("=" * 110)

    print(
        "OUTPUTS"
    )

    print("=" * 110)


    print(
        "\nCSV:"
    )

    print(
        SUMMARY_CSV
    )


    print(
        "\nJSON:"
    )

    print(
        SUMMARY_JSON
    )


    print(
        "\nMarkdown:"
    )

    print(
        SUMMARY_MD
    )


    print(
        "\nDetailed run folders:"
    )

    print(
        OUTPUT_DIR
    )


    print("\n")

    print("=" * 110)

    print(
        "PAIR 002 BENCHMARK COMPLETE"
    )

    print("=" * 110)


if __name__ == "__main__":

    main()