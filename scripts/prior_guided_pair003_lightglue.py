from pathlib import Path
import csv
import gc
import json
import math
import time

import cv2
import numpy as np
import torch


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_003"
)

SOURCE_PATH = (
    PAIR_DIR
    / "source.png"
)

REFERENCE_PATH = (
    PAIR_DIR
    / "reference.png"
)

MASK_PATH = (
    PAIR_DIR
    / "valid_mask.png"
)

GLOBAL_JSON = (
    ROOT
    / "results"
    / "pair_003"
    / "global_benchmark"
    / "pair003_global_benchmark.json"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_003"
    / "prior_guided_lightglue"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_JSON = (
    OUTPUT_DIR
    / "pair003_prior_guided_lightglue.json"
)

MATCH_CSV = (
    OUTPUT_DIR
    / "pair003_prior_guided_matches.csv"
)

INLIER_VIS = (
    OUTPUT_DIR
    / "pair003_prior_guided_inliers.png"
)

UNIFORM_VIS = (
    OUTPUT_DIR
    / "pair003_prior_guided_uniform.png"
)

GRADIENT_SOURCE = (
    OUTPUT_DIR
    / "source_gradient.png"
)

GRADIENT_REFERENCE = (
    OUTPUT_DIR
    / "reference_gradient.png"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

GRID_ROWS = 6
GRID_COLS = 12

PATCH_SIZE = 320

MIN_CELL_SAFE_RATIO = 0.05

BOUNDARY_MARGIN_PX = 8.0

PRIOR_TOLERANCE_PX = 6.0

LIGHTGLUE_MAX_KEYPOINTS = 2048

RANSAC_THRESHOLD_PX = 2.5
RANSAC_MAX_ITERS = 20000
RANSAC_CONFIDENCE = 0.999

MAX_UNIFORM_PER_CELL = 8

DEDUP_QUANTIZATION_PX = 2.0


# ============================================================
# BASIC IO
# ============================================================

def load_gray(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load:\n{path}"
        )

    return image


# ============================================================
# GRADIENT IMAGE
# ============================================================

def percentile_to_uint8(
    image,
    mask,
    low_percentile=1,
    high_percentile=99
):

    output = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    values = image[
        mask
    ]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:

        return output

    low, high = np.percentile(
        values,
        [
            low_percentile,
            high_percentile
        ]
    )

    if high <= low:

        return output

    normalized = (
        image.astype(np.float32)
        -
        float(low)
    ) / (
        float(high)
        -
        float(low)
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    output[
        mask
    ] = (
        normalized[
            mask
        ]
        *
        255.0
    ).astype(
        np.uint8
    )

    return output


def make_gradient_image(
    image,
    valid_mask
):

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        1.0
    )

    gx = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    gradient = percentile_to_uint8(
        magnitude,
        valid_mask
    )

    gradient[
        ~valid_mask
    ] = 0

    return gradient


# ============================================================
# SAFE INTERIOR
# ============================================================

def make_safe_mask(
    valid_mask
):

    binary = (
        valid_mask.astype(
            np.uint8
        )
    )

    distance = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5
    )

    return (
        distance
        >=
        BOUNDARY_MARGIN_PX
    )


# ============================================================
# PRIOR
# ============================================================

def load_global_prior():

    payload = json.loads(
        GLOBAL_JSON.read_text(
            encoding="utf-8"
        )
    )

    for result in payload[
        "results"
    ]:

        if (
            result.get("matcher")
            ==
            "lightglue"
            and
            result.get("representation")
            ==
            "gradient"
            and
            result.get("status")
            ==
            "ok"
            and
            result.get("transform_sanity")
            is True
        ):

            matrix = np.asarray(
                result[
                    "affine_matrix"
                ],
                dtype=np.float64
            )

            return (
                matrix,
                result
            )

    raise RuntimeError(
        "No sane global LightGlue-gradient "
        "prior found."
    )


def transform_points(
    points,
    matrix
):

    if len(points) == 0:

        return np.zeros(
            (0, 2),
            dtype=np.float32
        )

    homogeneous = np.column_stack(
        [
            points,
            np.ones(
                len(points),
                dtype=np.float64
            )
        ]
    )

    transformed = (
        homogeneous
        @
        matrix.T
    )

    return transformed.astype(
        np.float32
    )


# ============================================================
# PATCH HELPERS
# ============================================================

def cell_bounds(
    row,
    col,
    width,
    height
):

    x0 = int(
        col
        *
        width
        /
        GRID_COLS
    )

    x1 = int(
        (
            col + 1
        )
        *
        width
        /
        GRID_COLS
    )

    y0 = int(
        row
        *
        height
        /
        GRID_ROWS
    )

    y1 = int(
        (
            row + 1
        )
        *
        height
        /
        GRID_ROWS
    )

    return (
        x0,
        y0,
        x1,
        y1
    )


def extract_patch(
    image,
    center_x,
    center_y,
    size
):

    h, w = image.shape

    half = (
        size
        /
        2.0
    )

    x0 = max(
        0,
        int(
            math.floor(
                center_x
                -
                half
            )
        )
    )

    y0 = max(
        0,
        int(
            math.floor(
                center_y
                -
                half
            )
        )
    )

    x1 = min(
        w,
        x0 + size
    )

    y1 = min(
        h,
        y0 + size
    )

    # If clipped on right/bottom,
    # try shifting back to preserve size.

    if (
        x1 - x0
        <
        size
    ):

        x0 = max(
            0,
            x1 - size
        )

    if (
        y1 - y0
        <
        size
    ):

        y0 = max(
            0,
            y1 - size
        )

    patch = image[
        y0:y1,
        x0:x1
    ]

    return (
        patch,
        x0,
        y0
    )


# ============================================================
# MASK FILTER
# ============================================================

def points_inside_mask(
    points,
    mask
):

    if len(points) == 0:

        return np.zeros(
            0,
            dtype=bool
        )

    h, w = mask.shape

    x = np.rint(
        points[:, 0]
    ).astype(
        np.int32
    )

    y = np.rint(
        points[:, 1]
    ).astype(
        np.int32
    )

    inside = (
        (x >= 0)
        &
        (x < w)
        &
        (y >= 0)
        &
        (y < h)
    )

    result = np.zeros(
        len(points),
        dtype=bool
    )

    indices = np.flatnonzero(
        inside
    )

    result[
        indices
    ] = mask[
        y[
            indices
        ],
        x[
            indices
        ]
    ]

    return result


# ============================================================
# LIGHTGLUE
# ============================================================

def create_lightglue():

    from lightglue import (
        SuperPoint,
        LightGlue
    )

    extractor = (
        SuperPoint(
            max_num_keypoints=
                LIGHTGLUE_MAX_KEYPOINTS
        )
        .eval()
        .to(
            DEVICE
        )
    )

    matcher = (
        LightGlue(
            features="superpoint"
        )
        .eval()
        .to(
            DEVICE
        )
    )

    return (
        extractor,
        matcher
    )


def image_tensor(
    image
):

    tensor = torch.from_numpy(
        image.astype(
            np.float32
        )
        /
        255.0
    )

    return (
        tensor
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )


def match_patch(
    source_patch,
    reference_patch,
    extractor,
    matcher
):

    from lightglue.utils import rbd

    if (
        source_patch.size == 0
        or
        reference_patch.size == 0
    ):

        return (
            np.zeros(
                (0, 2),
                dtype=np.float32
            ),
            np.zeros(
                (0, 2),
                dtype=np.float32
            ),
            np.zeros(
                0,
                dtype=np.float32
            )
        )

    tensor0 = image_tensor(
        source_patch
    )

    tensor1 = image_tensor(
        reference_patch
    )

    with torch.inference_mode():

        features0 = extractor.extract(
            tensor0
        )

        features1 = extractor.extract(
            tensor1
        )

        output = matcher(
            {
                "image0":
                    features0,

                "image1":
                    features1
            }
        )

    features0 = rbd(
        features0
    )

    features1 = rbd(
        features1
    )

    output = rbd(
        output
    )

    matches = output[
        "matches"
    ]

    if matches.numel() == 0:

        points0 = np.zeros(
            (0, 2),
            dtype=np.float32
        )

        points1 = np.zeros(
            (0, 2),
            dtype=np.float32
        )

        scores = np.zeros(
            0,
            dtype=np.float32
        )

    else:

        points0 = (
            features0[
                "keypoints"
            ][
                matches[:, 0]
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        points1 = (
            features1[
                "keypoints"
            ][
                matches[:, 1]
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        if "scores" in output:

            scores = (
                output[
                    "scores"
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

        else:

            scores = np.ones(
                len(points0),
                dtype=np.float32
            )

    del (
        tensor0,
        tensor1,
        features0,
        features1,
        output
    )

    if DEVICE == "cuda":

        torch.cuda.empty_cache()

    return (
        points0,
        points1,
        scores
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_matches(
    points0,
    points1,
    scores,
    prior_residuals,
    representations
):

    if len(points0) == 0:

        return (
            points0,
            points1,
            scores,
            prior_residuals,
            representations
        )

    quality = (
        scores
        /
        (
            1.0
            +
            prior_residuals
        )
    )

    order = np.argsort(
        -quality
    )

    used = set()

    selected = []

    q = (
        DEDUP_QUANTIZATION_PX
    )

    for index in order:

        p0 = points0[
            index
        ]

        p1 = points1[
            index
        ]

        key = (
            int(
                round(
                    p0[0]
                    /
                    q
                )
            ),
            int(
                round(
                    p0[1]
                    /
                    q
                )
            ),
            int(
                round(
                    p1[0]
                    /
                    q
                )
            ),
            int(
                round(
                    p1[1]
                    /
                    q
                )
            )
        )

        if key in used:

            continue

        used.add(
            key
        )

        selected.append(
            int(index)
        )

    selected = np.asarray(
        selected,
        dtype=np.int32
    )

    return (
        points0[
            selected
        ],
        points1[
            selected
        ],
        scores[
            selected
        ],
        prior_residuals[
            selected
        ],
        representations[
            selected
        ]
    )


# ============================================================
# AFFINE METRICS
# ============================================================

def affine_parameters(
    matrix
):

    a = float(
        matrix[0, 0]
    )

    b = float(
        matrix[0, 1]
    )

    c = float(
        matrix[1, 0]
    )

    d = float(
        matrix[1, 1]
    )

    tx = float(
        matrix[0, 2]
    )

    ty = float(
        matrix[1, 2]
    )

    scale_x = math.sqrt(
        a * a
        +
        c * c
    )

    scale_y = math.sqrt(
        b * b
        +
        d * d
    )

    rotation = math.degrees(
        math.atan2(
            c,
            a
        )
    )

    translation = math.sqrt(
        tx * tx
        +
        ty * ty
    )

    determinant = (
        a * d
        -
        b * c
    )

    axis_dot = (
        (
            a * b
            +
            c * d
        )
        /
        (
            scale_x
            *
            scale_y
        )
        if (
            scale_x > 0
            and
            scale_y > 0
        )
        else float("nan")
    )

    return {
        "tx_px":
            tx,

        "ty_px":
            ty,

        "translation_px":
            translation,

        "rotation_deg":
            rotation,

        "scale_x":
            scale_x,

        "scale_y":
            scale_y,

        "determinant":
            determinant,

        "normalized_axis_dot":
            axis_dot
    }


def transform_sanity(
    parameters
):

    return bool(
        parameters[
            "determinant"
        ] > 0

        and

        parameters[
            "translation_px"
        ] <= 25.0

        and

        abs(
            parameters[
                "rotation_deg"
            ]
        ) <= 5.0

        and

        0.95
        <=
        parameters[
            "scale_x"
        ]
        <=
        1.05

        and

        0.95
        <=
        parameters[
            "scale_y"
        ]
        <=
        1.05

        and

        abs(
            parameters[
                "normalized_axis_dot"
            ]
        ) <= 0.10
    )


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def get_cell(
    point,
    width,
    height
):

    x, y = point

    col = min(
        GRID_COLS - 1,
        max(
            0,
            int(
                x
                /
                width
                *
                GRID_COLS
            )
        )
    )

    row = min(
        GRID_ROWS - 1,
        max(
            0,
            int(
                y
                /
                height
                *
                GRID_ROWS
            )
        )
    )

    return (
        row,
        col
    )


def valid_cells(
    mask
):

    h, w = mask.shape

    cells = set()

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
                w,
                h
            )

            cell_mask = mask[
                y0:y1,
                x0:x1
            ]

            if (
                cell_mask.size > 0
                and
                float(
                    cell_mask.mean()
                )
                >=
                MIN_CELL_SAFE_RATIO
            ):

                cells.add(
                    (
                        row,
                        col
                    )
                )

    return cells


def spatial_metrics(
    points,
    mask
):

    h, w = mask.shape

    available = valid_cells(
        mask
    )

    occupied = set()

    for point in points:

        cell = get_cell(
            point,
            w,
            h
        )

        if cell in available:

            occupied.add(
                cell
            )

    coverage = (
        len(occupied)
        /
        len(available)
        if len(available) > 0
        else 0.0
    )

    return {
        "valid_cells":
            len(available),

        "occupied_valid_cells":
            len(occupied),

        "coverage":
            float(
                coverage
            )
    }


# ============================================================
# UNIFORM SELECT
# ============================================================

def uniform_select(
    points0,
    points1,
    scores,
    residuals,
    mask
):

    if len(points0) == 0:

        empty = np.zeros(
            0,
            dtype=np.int32
        )

        return empty

    h, w = mask.shape

    quality = (
        scores
        /
        (
            1.0
            +
            residuals
        )
    )

    order = np.argsort(
        -quality
    )

    count = {}

    selected = []

    for index in order:

        cell = get_cell(
            points0[
                index
            ],
            w,
            h
        )

        current = count.get(
            cell,
            0
        )

        if (
            current
            >=
            MAX_UNIFORM_PER_CELL
        ):

            continue

        count[
            cell
        ] = (
            current + 1
        )

        selected.append(
            int(index)
        )

    return np.asarray(
        selected,
        dtype=np.int32
    )


# ============================================================
# VISUALIZATION
# ============================================================

def save_match_visualization(
    source,
    reference,
    points0,
    points1,
    path,
    max_draw=250
):

    source_bgr = cv2.cvtColor(
        source,
        cv2.COLOR_GRAY2BGR
    )

    reference_bgr = cv2.cvtColor(
        reference,
        cv2.COLOR_GRAY2BGR
    )

    h = max(
        source.shape[0],
        reference.shape[0]
    )

    w0 = source.shape[1]
    w1 = reference.shape[1]

    canvas = np.zeros(
        (
            h,
            w0 + w1,
            3
        ),
        dtype=np.uint8
    )

    canvas[
        :source.shape[0],
        :w0
    ] = source_bgr

    canvas[
        :reference.shape[0],
        w0:w0 + w1
    ] = reference_bgr

    if len(points0) > max_draw:

        indices = np.linspace(
            0,
            len(points0) - 1,
            max_draw
        ).astype(
            np.int32
        )

    else:

        indices = np.arange(
            len(points0)
        )

    for index in indices:

        p0 = points0[
            index
        ]

        p1 = points1[
            index
        ]

        a = (
            int(
                round(
                    p0[0]
                )
            ),
            int(
                round(
                    p0[1]
                )
            )
        )

        b = (
            int(
                round(
                    p1[0]
                )
            )
            +
            w0,
            int(
                round(
                    p1[1]
                )
            )
        )

        cv2.line(
            canvas,
            a,
            b,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.circle(
            canvas,
            a,
            2,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            canvas,
            b,
            2,
            (0, 255, 255),
            -1,
            cv2.LINE_AA
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
        "CHANDRAMATCH - PAIR 003 "
        "PRIOR-GUIDED LIGHTGLUE"
    )

    print("=" * 80)


    print(
        "\nDevice:",
        DEVICE
    )

    if DEVICE == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )


    # ========================================================
    # LOAD
    # ========================================================

    source = load_gray(
        SOURCE_PATH
    )

    reference = load_gray(
        REFERENCE_PATH
    )

    mask_image = load_gray(
        MASK_PATH
    )

    valid_mask = (
        mask_image > 0
    )

    safe_mask = make_safe_mask(
        valid_mask
    )


    if (
        source.shape
        !=
        reference.shape
        or
        source.shape
        !=
        valid_mask.shape
    ):

        raise RuntimeError(
            "Image/mask shapes disagree."
        )


    h, w = source.shape


    print(
        "\nShape:",
        source.shape
    )

    print(
        "Safe interior pixels:",
        int(
            np.count_nonzero(
                safe_mask
            )
        )
    )


    # ========================================================
    # PRIOR
    # ========================================================

    prior_matrix, global_result = (
        load_global_prior()
    )


    print("\n")
    print("=" * 80)

    print(
        "GLOBAL PRIOR"
    )

    print("=" * 80)


    print(
        prior_matrix
    )

    print(
        "Global inliers:",
        global_result[
            "inliers"
        ]
    )

    print(
        "Global coverage:",
        global_result[
            "spatial"
        ][
            "coverage"
        ]
    )

    print(
        "Global RMSE:",
        global_result[
            "ransac_reprojection_rmse_px"
        ]
    )


    # ========================================================
    # REPRESENTATIONS
    # ========================================================

    source_baseline = source.copy()

    reference_baseline = reference.copy()

    source_baseline[
        ~valid_mask
    ] = 0

    reference_baseline[
        ~valid_mask
    ] = 0


    source_gradient = make_gradient_image(
        source,
        valid_mask
    )

    reference_gradient = make_gradient_image(
        reference,
        valid_mask
    )


    cv2.imwrite(
        str(
            GRADIENT_SOURCE
        ),
        source_gradient
    )

    cv2.imwrite(
        str(
            GRADIENT_REFERENCE
        ),
        reference_gradient
    )


    representations = {
        "baseline":
            (
                source_baseline,
                reference_baseline
            ),

        "gradient":
            (
                source_gradient,
                reference_gradient
            )
    }


    # ========================================================
    # LIGHTGLUE
    # ========================================================

    print(
        "\nLoading LightGlue..."
    )

    extractor, matcher = (
        create_lightglue()
    )

    print(
        "LightGlue ready."
    )


    # ========================================================
    # LOCAL MATCHING
    # ========================================================

    available_cells = valid_cells(
        safe_mask
    )


    print("\n")
    print("=" * 80)

    print(
        "LOCAL PRIOR-GUIDED MATCHING"
    )

    print("=" * 80)


    print(
        "Valid grid cells:",
        len(
            available_cells
        )
    )

    print(
        "Grid:",
        GRID_ROWS,
        "x",
        GRID_COLS
    )

    print(
        "Patch size:",
        PATCH_SIZE
    )

    print(
        "Prior tolerance:",
        PRIOR_TOLERANCE_PX,
        "px"
    )


    all_points0 = []

    all_points1 = []

    all_scores = []

    all_prior_residuals = []

    all_representations = []

    cell_summary = []


    start_total = time.perf_counter()


    for row in range(
        GRID_ROWS
    ):

        for col in range(
            GRID_COLS
        ):

            if (
                row,
                col
            ) not in available_cells:

                continue


            (
                cell_x0,
                cell_y0,
                cell_x1,
                cell_y1

            ) = cell_bounds(
                row,
                col,
                w,
                h
            )


            center_source = np.array(
                [
                    [
                        (
                            cell_x0
                            +
                            cell_x1
                        )
                        /
                        2.0,

                        (
                            cell_y0
                            +
                            cell_y1
                        )
                        /
                        2.0
                    ]
                ],
                dtype=np.float32
            )


            center_reference = transform_points(
                center_source,
                prior_matrix
            )[0]


            cell_counts = {
                "row":
                    row,

                "col":
                    col,

                "baseline_raw":
                    0,

                "baseline_prior_consistent":
                    0,

                "gradient_raw":
                    0,

                "gradient_prior_consistent":
                    0
            }


            for representation_name, (
                image0,
                image1

            ) in representations.items():


                (
                    patch0,
                    offset0_x,
                    offset0_y

                ) = extract_patch(
                    image0,
                    center_source[0, 0],
                    center_source[0, 1],
                    PATCH_SIZE
                )


                (
                    patch1,
                    offset1_x,
                    offset1_y

                ) = extract_patch(
                    image1,
                    center_reference[0],
                    center_reference[1],
                    PATCH_SIZE
                )


                (
                    local0,
                    local1,
                    scores

                ) = match_patch(
                    patch0,
                    patch1,
                    extractor,
                    matcher
                )


                cell_counts[
                    f"{representation_name}_raw"
                ] = int(
                    len(
                        local0
                    )
                )


                if len(local0) == 0:

                    continue


                global0 = local0.copy()

                global1 = local1.copy()


                global0[
                    :,
                    0
                ] += offset0_x

                global0[
                    :,
                    1
                ] += offset0_y


                global1[
                    :,
                    0
                ] += offset1_x

                global1[
                    :,
                    1
                ] += offset1_y


                # --------------------------------------------
                # Keep only matches belonging to this
                # source cell's core region.
                # --------------------------------------------

                inside_core = (
                    (
                        global0[:, 0]
                        >=
                        cell_x0
                    )
                    &
                    (
                        global0[:, 0]
                        <
                        cell_x1
                    )
                    &
                    (
                        global0[:, 1]
                        >=
                        cell_y0
                    )
                    &
                    (
                        global0[:, 1]
                        <
                        cell_y1
                    )
                )


                safe = (
                    points_inside_mask(
                        global0,
                        safe_mask
                    )
                    &
                    points_inside_mask(
                        global1,
                        safe_mask
                    )
                )


                predicted = transform_points(
                    global0,
                    prior_matrix
                )


                prior_residual = np.linalg.norm(
                    predicted
                    -
                    global1,
                    axis=1
                )


                prior_consistent = (
                    prior_residual
                    <=
                    PRIOR_TOLERANCE_PX
                )


                keep = (
                    inside_core
                    &
                    safe
                    &
                    prior_consistent
                )


                global0 = global0[
                    keep
                ]

                global1 = global1[
                    keep
                ]

                scores = scores[
                    keep
                ]

                prior_residual = (
                    prior_residual[
                        keep
                    ]
                )


                cell_counts[
                    f"{representation_name}"
                    "_prior_consistent"
                ] = int(
                    len(
                        global0
                    )
                )


                if len(global0) == 0:

                    continue


                all_points0.append(
                    global0
                )

                all_points1.append(
                    global1
                )

                all_scores.append(
                    scores
                )

                all_prior_residuals.append(
                    prior_residual.astype(
                        np.float32
                    )
                )

                all_representations.append(
                    np.full(
                        len(
                            global0
                        ),
                        representation_name,
                        dtype="<U16"
                    )
                )


            cell_summary.append(
                cell_counts
            )


            print(
                f"Cell ({row},{col}) "
                f"| base "
                f"{cell_counts['baseline_raw']}"
                f" -> "
                f"{cell_counts['baseline_prior_consistent']} "
                f"| grad "
                f"{cell_counts['gradient_raw']}"
                f" -> "
                f"{cell_counts['gradient_prior_consistent']}"
            )


    runtime = (
        time.perf_counter()
        -
        start_total
    )


    if len(
        all_points0
    ) == 0:

        raise RuntimeError(
            "No prior-consistent local matches."
        )


    points0 = np.concatenate(
        all_points0,
        axis=0
    )

    points1 = np.concatenate(
        all_points1,
        axis=0
    )

    scores = np.concatenate(
        all_scores,
        axis=0
    )

    prior_residuals = np.concatenate(
        all_prior_residuals,
        axis=0
    )

    representations_array = np.concatenate(
        all_representations,
        axis=0
    )


    print("\n")
    print("=" * 80)

    print(
        "MERGED LOCAL MATCHES"
    )

    print("=" * 80)


    print(
        "Before deduplication:",
        len(
            points0
        )
    )


    (
        points0,
        points1,
        scores,
        prior_residuals,
        representations_array

    ) = deduplicate_matches(
        points0,
        points1,
        scores,
        prior_residuals,
        representations_array
    )


    print(
        "After deduplication:",
        len(
            points0
        )
    )


    print(
        "Baseline retained:",
        int(
            np.count_nonzero(
                representations_array
                ==
                "baseline"
            )
        )
    )

    print(
        "Gradient retained:",
        int(
            np.count_nonzero(
                representations_array
                ==
                "gradient"
            )
        )
    )


    # ========================================================
    # RANSAC
    # ========================================================

    if len(
        points0
    ) < 3:

        raise RuntimeError(
            "Insufficient matches for RANSAC."
        )


    matrix, inlier_mask = (
        cv2.estimateAffine2D(
            points0.astype(
                np.float32
            ),
            points1.astype(
                np.float32
            ),
            method=cv2.RANSAC,
            ransacReprojThreshold=
                RANSAC_THRESHOLD_PX,
            maxIters=
                RANSAC_MAX_ITERS,
            confidence=
                RANSAC_CONFIDENCE,
            refineIters=25
        )
    )


    if (
        matrix is None
        or
        inlier_mask is None
    ):

        raise RuntimeError(
            "Final affine RANSAC failed."
        )


    inlier_mask = (
        inlier_mask.ravel()
        >
        0
    )


    inlier_points0 = points0[
        inlier_mask
    ]

    inlier_points1 = points1[
        inlier_mask
    ]

    inlier_scores = scores[
        inlier_mask
    ]

    inlier_prior_residuals = (
        prior_residuals[
            inlier_mask
        ]
    )

    inlier_representations = (
        representations_array[
            inlier_mask
        ]
    )


    predicted = transform_points(
        inlier_points0,
        matrix
    )


    residuals = np.linalg.norm(
        predicted
        -
        inlier_points1,
        axis=1
    )


    parameters = affine_parameters(
        matrix
    )


    spatial = spatial_metrics(
        inlier_points0,
        safe_mask
    )


    uniform_indices = uniform_select(
        inlier_points0,
        inlier_points1,
        inlier_scores,
        residuals,
        safe_mask
    )


    uniform_points0 = (
        inlier_points0[
            uniform_indices
        ]
    )

    uniform_points1 = (
        inlier_points1[
            uniform_indices
        ]
    )


    # ========================================================
    # RESULTS
    # ========================================================

    inlier_count = int(
        len(
            inlier_points0
        )
    )

    candidate_count = int(
        len(
            points0
        )
    )


    print("\n")
    print("=" * 80)

    print(
        "FINAL PRIOR-GUIDED RESULT"
    )

    print("=" * 80)


    print(
        "Candidate matches:",
        candidate_count
    )

    print(
        "RANSAC inliers:",
        inlier_count
    )

    print(
        "Inlier ratio:",
        (
            inlier_count
            /
            candidate_count
        )
    )

    print(
        "RANSAC RMSE:",
        float(
            np.sqrt(
                np.mean(
                    residuals
                    **
                    2
                )
            )
        ),
        "px"
    )

    print(
        "Median residual:",
        float(
            np.median(
                residuals
            )
        ),
        "px"
    )

    print(
        "Mean residual:",
        float(
            np.mean(
                residuals
            )
        ),
        "px"
    )

    print(
        "Coverage:",
        spatial[
            "coverage"
        ]
    )

    print(
        "Occupied valid cells:",
        spatial[
            "occupied_valid_cells"
        ],
        "/",
        spatial[
            "valid_cells"
        ]
    )

    print(
        "Uniform inliers:",
        len(
            uniform_points0
        )
    )

    print(
        "Affine:"
    )

    print(
        matrix
    )

    print(
        "Transform:"
    )

    print(
        parameters
    )

    print(
        "Transform sanity:",
        transform_sanity(
            parameters
        )
    )


    # ========================================================
    # VISUALIZATIONS
    # ========================================================

    save_match_visualization(
        source,
        reference,
        inlier_points0,
        inlier_points1,
        INLIER_VIS,
        max_draw=250
    )


    save_match_visualization(
        source,
        reference,
        uniform_points0,
        uniform_points1,
        UNIFORM_VIS,
        max_draw=250
    )


    # ========================================================
    # MATCH CSV
    # ========================================================

    with MATCH_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "source_x",
                "source_y",
                "reference_x",
                "reference_y",
                "score",
                "prior_residual_px",
                "final_residual_px",
                "representation"
            ]
        )

        for index in range(
            len(
                inlier_points0
            )
        ):

            writer.writerow(
                [
                    float(
                        inlier_points0[
                            index,
                            0
                        ]
                    ),

                    float(
                        inlier_points0[
                            index,
                            1
                        ]
                    ),

                    float(
                        inlier_points1[
                            index,
                            0
                        ]
                    ),

                    float(
                        inlier_points1[
                            index,
                            1
                        ]
                    ),

                    float(
                        inlier_scores[
                            index
                        ]
                    ),

                    float(
                        inlier_prior_residuals[
                            index
                        ]
                    ),

                    float(
                        residuals[
                            index
                        ]
                    ),

                    str(
                        inlier_representations[
                            index
                        ]
                    )
                ]
            )


    # ========================================================
    # JSON
    # ========================================================

    result = {

        "pair_id":
            "pair_003",

        "method":
            "prior_guided_local_lightglue",

        "device":
            DEVICE,

        "grid": {
            "rows":
                GRID_ROWS,

            "cols":
                GRID_COLS,

            "valid_cells":
                len(
                    available_cells
                )
        },

        "patch_size":
            PATCH_SIZE,

        "prior_tolerance_px":
            PRIOR_TOLERANCE_PX,

        "prior": {
            "matrix":
                prior_matrix.tolist(),

            "global_inliers":
                global_result[
                    "inliers"
                ],

            "global_coverage":
                global_result[
                    "spatial"
                ][
                    "coverage"
                ]
        },

        "local_matching": {
            "before_dedup":
                int(
                    sum(
                        len(x)
                        for x in all_points0
                    )
                ),

            "after_dedup":
                candidate_count,

            "baseline_retained":
                int(
                    np.count_nonzero(
                        representations_array
                        ==
                        "baseline"
                    )
                ),

            "gradient_retained":
                int(
                    np.count_nonzero(
                        representations_array
                        ==
                        "gradient"
                    )
                )
        },

        "ransac": {
            "threshold_px":
                RANSAC_THRESHOLD_PX,

            "inliers":
                inlier_count,

            "inlier_ratio":
                float(
                    inlier_count
                    /
                    candidate_count
                ),

            "reprojection_rmse_px":
                float(
                    np.sqrt(
                        np.mean(
                            residuals
                            **
                            2
                        )
                    )
                ),

            "reprojection_median_px":
                float(
                    np.median(
                        residuals
                    )
                ),

            "reprojection_mean_px":
                float(
                    np.mean(
                        residuals
                    )
                )
        },

        "affine_matrix":
            matrix.tolist(),

        "transform":
            parameters,

        "transform_sanity":
            transform_sanity(
                parameters
            ),

        "spatial":
            spatial,

        "uniform_inliers":
            int(
                len(
                    uniform_points0
                )
            ),

        "runtime_seconds":
            float(
                runtime
            ),

        "cell_summary":
            cell_summary,

        "metric_note":
            (
                "RANSAC reprojection RMSE is "
                "correspondence self-consistency, "
                "not independent ground-truth "
                "registration accuracy."
            )
    }


    RESULT_JSON.write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )


    print("\n")
    print("=" * 80)

    print(
        "PAIR 003 PRIOR-GUIDED MATCHING COMPLETE"
    )

    print("=" * 80)


    print(
        "\nJSON:"
    )

    print(
        RESULT_JSON
    )

    print(
        "\nMatches:"
    )

    print(
        MATCH_CSV
    )

    print(
        "\nInlier visualization:"
    )

    print(
        INLIER_VIS
    )

    print(
        "\nUniform visualization:"
    )

    print(
        UNIFORM_VIS
    )


if __name__ == "__main__":

    main()