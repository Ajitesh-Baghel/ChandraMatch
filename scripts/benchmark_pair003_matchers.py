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


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_003"
    / "global_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_JSON = (
    OUTPUT_DIR
    / "pair003_global_benchmark.json"
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "pair003_global_benchmark.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANSAC_THRESHOLD_PX = 2.5
RANSAC_MAX_ITERS = 10000
RANSAC_CONFIDENCE = 0.999

SIFT_MAX_FEATURES = 12000
SIFT_RATIO = 0.80

LIGHTGLUE_MAX_KEYPOINTS = 8192

LOFTR_CONFIDENCE = 0.05

# Try highest first.
# If RTX 4070 Laptop runs out of memory,
# automatically retry smaller input.
LOFTR_MAX_DIMENSIONS = [
    840,
    640,
    512
]

# Keep matches away from shared no-data boundaries.
#
# This is important because source/reference have the
# same final valid mask. Matching mask boundaries would
# artificially inflate performance.
BOUNDARY_MARGIN_PX = 8.0

# Coverage grid
GRID_ROWS = 6
GRID_COLS = 12

# Final spatially distributed product
MAX_UNIFORM_PER_CELL = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def load_gray(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read:\n{path}"
        )

    return image


def percentile_to_uint8(
    image,
    mask,
    p_low=1,
    p_high=99
):

    output = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    values = image[
        mask
    ]

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:

        return output

    low, high = np.percentile(
        values,
        [
            p_low,
            p_high
        ]
    )

    if high <= low:

        return output

    normalized = (
        image.astype(
            np.float32
        )
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


# ============================================================
# GRADIENT REPRESENTATION
# ============================================================

def make_gradient_image(
    image,
    valid_mask
):

    # Small blur reduces pixel-scale noise without
    # trying to "fix" the IIRS image.

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
        valid_mask,
        1,
        99
    )

    gradient[
        ~valid_mask
    ] = 0

    return gradient


# ============================================================
# SAFE INTERIOR MASK
# ============================================================

def make_safe_mask(
    mask
):

    binary = (
        mask > 0
    ).astype(
        np.uint8
    )

    distance = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5
    )

    safe = (
        distance
        >=
        BOUNDARY_MARGIN_PX
    )

    return safe


# ============================================================
# POINT MASK FILTER
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

    inside_image = (
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

    good_indices = np.flatnonzero(
        inside_image
    )

    result[
        good_indices
    ] = mask[
        y[
            good_indices
        ],
        x[
            good_indices
        ]
    ]

    return result


# ============================================================
# AFFINE METRICS
# ============================================================

def affine_parameters(
    matrix
):

    if matrix is None:

        return None

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

    rotation_deg = math.degrees(
        math.atan2(
            c,
            a
        )
    )

    translation_px = math.sqrt(
        tx * tx
        +
        ty * ty
    )

    determinant = (
        a * d
        -
        b * c
    )

    if (
        scale_x > 0
        and
        scale_y > 0
    ):

        axis_dot = (
            a * b
            +
            c * d
        ) / (
            scale_x
            *
            scale_y
        )

    else:

        axis_dot = float(
            "nan"
        )

    return {
        "tx_px":
            tx,

        "ty_px":
            ty,

        "translation_px":
            translation_px,

        "rotation_deg":
            rotation_deg,

        "scale_x":
            scale_x,

        "scale_y":
            scale_y,

        "determinant":
            determinant,

        "normalized_axis_dot":
            float(
                axis_dot
            )
    }


def transform_sanity(
    parameters
):

    if parameters is None:

        return False

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
# COVERAGE
# ============================================================

def grid_cell_index(
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
                max(
                    width,
                    1
                )
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
                max(
                    height,
                    1
                )
                *
                GRID_ROWS
            )
        )
    )

    return (
        row,
        col
    )


def valid_grid_cells(
    mask
):

    h, w = mask.shape

    cells = set()

    for row in range(
        GRID_ROWS
    ):

        y0 = int(
            row
            *
            h
            /
            GRID_ROWS
        )

        y1 = int(
            (
                row + 1
            )
            *
            h
            /
            GRID_ROWS
        )

        for col in range(
            GRID_COLS
        ):

            x0 = int(
                col
                *
                w
                /
                GRID_COLS
            )

            x1 = int(
                (
                    col + 1
                )
                *
                w
                /
                GRID_COLS
            )

            cell_mask = mask[
                y0:y1,
                x0:x1
            ]

            # Require at least 5% useful interior pixels.

            if (
                cell_mask.size > 0
                and
                float(
                    cell_mask.mean()
                )
                >=
                0.05
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

    valid_cells = valid_grid_cells(
        mask
    )

    occupied = set()

    for point in points:

        cell = grid_cell_index(
            point,
            w,
            h
        )

        if cell in valid_cells:

            occupied.add(
                cell
            )

    if len(
        valid_cells
    ) > 0:

        coverage = (
            len(
                occupied
            )
            /
            len(
                valid_cells
            )
        )

    else:

        coverage = 0.0

    return {
        "grid_rows":
            GRID_ROWS,

        "grid_cols":
            GRID_COLS,

        "valid_cells":
            len(
                valid_cells
            ),

        "occupied_valid_cells":
            len(
                occupied
            ),

        "coverage":
            float(
                coverage
            )
    }


# ============================================================
# UNIFORM SELECTION
# ============================================================

def uniform_select(
    points0,
    points1,
    scores,
    mask
):

    if len(
        points0
    ) == 0:

        return (
            points0,
            points1,
            scores
        )

    h, w = mask.shape

    order = np.argsort(
        -scores
    )

    counts = {}

    selected = []

    for index in order:

        point = points0[
            index
        ]

        cell = grid_cell_index(
            point,
            w,
            h
        )

        count = counts.get(
            cell,
            0
        )

        if count >= MAX_UNIFORM_PER_CELL:

            continue

        counts[
            cell
        ] = (
            count + 1
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
        ]
    )


# ============================================================
# RANSAC
# ============================================================

def evaluate_matches(
    points0,
    points1,
    scores,
    safe_mask
):

    result = {
        "candidate_matches":
            int(
                len(
                    points0
                )
            ),

        "ransac_threshold_px":
            RANSAC_THRESHOLD_PX
    }

    if len(
        points0
    ) < 3:

        result[
            "status"
        ] = "insufficient_matches"

        return (
            result,
            None
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
            refineIters=20
        )
    )

    if (
        matrix is None
        or
        inlier_mask is None
    ):

        result[
            "status"
        ] = "ransac_failed"

        return (
            result,
            None
        )

    inlier_mask = (
        inlier_mask.ravel()
        >
        0
    )

    inlier_count = int(
        np.count_nonzero(
            inlier_mask
        )
    )

    if inlier_count == 0:

        result[
            "status"
        ] = "zero_inliers"

        return (
            result,
            None
        )

    prediction = cv2.transform(
        points0.reshape(
            -1,
            1,
            2
        ).astype(
            np.float32
        ),
        matrix
    ).reshape(
        -1,
        2
    )

    residuals = np.linalg.norm(
        prediction
        -
        points1,
        axis=1
    )

    inlier_residuals = residuals[
        inlier_mask
    ]

    inlier_points0 = points0[
        inlier_mask
    ]

    inlier_points1 = points1[
        inlier_mask
    ]

    inlier_scores = scores[
        inlier_mask
    ]

    parameters = affine_parameters(
        matrix
    )

    spatial = spatial_metrics(
        inlier_points0,
        safe_mask
    )

    (
        uniform_points0,
        uniform_points1,
        uniform_scores

    ) = uniform_select(
        inlier_points0,
        inlier_points1,
        inlier_scores,
        safe_mask
    )

    result.update(
        {
            "status":
                "ok",

            "inliers":
                inlier_count,

            "inlier_ratio":
                float(
                    inlier_count
                    /
                    len(
                        points0
                    )
                ),

            # IMPORTANT:
            # This is RANSAC self-consistency residual,
            # NOT independent ground-truth accuracy.

            "ransac_reprojection_rmse_px":
                float(
                    np.sqrt(
                        np.mean(
                            inlier_residuals
                            **
                            2
                        )
                    )
                ),

            "ransac_reprojection_mean_px":
                float(
                    np.mean(
                        inlier_residuals
                    )
                ),

            "ransac_reprojection_median_px":
                float(
                    np.median(
                        inlier_residuals
                    )
                ),

            "ransac_reprojection_max_px":
                float(
                    np.max(
                        inlier_residuals
                    )
                ),

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
                )
        }
    )

    data = {
        "matrix":
            matrix,

        "inlier_points0":
            inlier_points0,

        "inlier_points1":
            inlier_points1,

        "inlier_scores":
            inlier_scores,

        "uniform_points0":
            uniform_points0,

        "uniform_points1":
            uniform_points1,

        "uniform_scores":
            uniform_scores
    }

    return (
        result,
        data
    )


# ============================================================
# MATCH VISUALIZATION
# ============================================================

def save_match_visualization(
    source,
    reference,
    points0,
    points1,
    path,
    max_draw=120
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

    if len(
        points0
    ) > max_draw:

        indices = np.linspace(
            0,
            len(
                points0
            ) - 1,
            max_draw
        ).astype(
            np.int32
        )

    else:

        indices = np.arange(
            len(
                points0
            )
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
            a,
            2,
            (
                0,
                255,
                255
            ),
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            canvas,
            b,
            2,
            (
                0,
                255,
                255
            ),
            -1,
            cv2.LINE_AA
        )

    cv2.imwrite(
        str(path),
        canvas
    )


# ============================================================
# SIFT
# ============================================================

def match_sift(
    image0,
    image1,
    safe_mask
):

    start = time.perf_counter()

    sift = cv2.SIFT_create(
        nfeatures=
            SIFT_MAX_FEATURES
    )

    mask_u8 = (
        safe_mask.astype(
            np.uint8
        )
        *
        255
    )

    keypoints0, descriptors0 = (
        sift.detectAndCompute(
            image0,
            mask_u8
        )
    )

    keypoints1, descriptors1 = (
        sift.detectAndCompute(
            image1,
            mask_u8
        )
    )

    if (
        descriptors0 is None
        or
        descriptors1 is None
    ):

        return (
            np.zeros(
                (
                    0,
                    2
                ),
                dtype=np.float32
            ),
            np.zeros(
                (
                    0,
                    2
                ),
                dtype=np.float32
            ),
            np.zeros(
                0,
                dtype=np.float32
            ),
            {
                "keypoints_source":
                    len(
                        keypoints0
                    ),

                "keypoints_reference":
                    len(
                        keypoints1
                    ),

                "runtime_seconds":
                    float(
                        time.perf_counter()
                        -
                        start
                    )
            }
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    knn = matcher.knnMatch(
        descriptors0,
        descriptors1,
        k=2
    )

    accepted = []

    for pair in knn:

        if len(
            pair
        ) < 2:

            continue

        m, n = pair

        if (
            m.distance
            <
            SIFT_RATIO
            *
            n.distance
        ):

            accepted.append(
                m
            )

    points0 = np.array(
        [
            keypoints0[
                m.queryIdx
            ].pt
            for m in accepted
        ],
        dtype=np.float32
    )

    points1 = np.array(
        [
            keypoints1[
                m.trainIdx
            ].pt
            for m in accepted
        ],
        dtype=np.float32
    )

    scores = np.array(
        [
            1.0
            /
            (
                float(
                    m.distance
                )
                +
                1e-6
            )
            for m in accepted
        ],
        dtype=np.float32
    )

    runtime = (
        time.perf_counter()
        -
        start
    )

    return (
        points0,
        points1,
        scores,
        {
            "keypoints_source":
                len(
                    keypoints0
                ),

            "keypoints_reference":
                len(
                    keypoints1
                ),

            "ratio_threshold":
                SIFT_RATIO,

            "runtime_seconds":
                float(
                    runtime
                )
        }
    )


# ============================================================
# LIGHTGLUE
# ============================================================

def image_to_lightglue_tensor(
    image
):

    tensor = torch.from_numpy(
        image.astype(
            np.float32
        )
        /
        255.0
    )

    # C,H,W
    tensor = tensor.unsqueeze(
        0
    )

    # SuperPoint can operate on grayscale.

    return tensor.to(
        DEVICE
    )


def create_lightglue():

    from lightglue import (
        LightGlue,
        SuperPoint
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


def match_lightglue(
    image0,
    image1,
    safe_mask,
    extractor,
    matcher
):

    from lightglue.utils import rbd

    if DEVICE == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.perf_counter()

    tensor0 = image_to_lightglue_tensor(
        image0
    )

    tensor1 = image_to_lightglue_tensor(
        image1
    )

    with torch.inference_mode():

        feats0 = extractor.extract(
            tensor0
        )

        feats1 = extractor.extract(
            tensor1
        )

        matches01 = matcher(
            {
                "image0":
                    feats0,

                "image1":
                    feats1
            }
        )

    if DEVICE == "cuda":

        torch.cuda.synchronize()

    runtime = (
        time.perf_counter()
        -
        start
    )

    feats0, feats1, matches01 = [
        rbd(x)
        for x
        in (
            feats0,
            feats1,
            matches01
        )
    ]

    matches = matches01[
        "matches"
    ]

    if matches.numel() == 0:

        points0 = np.zeros(
            (
                0,
                2
            ),
            dtype=np.float32
        )

        points1 = points0.copy()

        scores = np.zeros(
            0,
            dtype=np.float32
        )

    else:

        keypoints0 = feats0[
            "keypoints"
        ]

        keypoints1 = feats1[
            "keypoints"
        ]

        points0 = (
            keypoints0[
                matches[
                    :,
                    0
                ]
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        points1 = (
            keypoints1[
                matches[
                    :,
                    1
                ]
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        if (
            "scores"
            in
            matches01
        ):

            scores = (
                matches01[
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
                len(
                    points0
                ),
                dtype=np.float32
            )

    mask_good = (
        points_inside_mask(
            points0,
            safe_mask
        )
        &
        points_inside_mask(
            points1,
            safe_mask
        )
    )

    points0 = points0[
        mask_good
    ]

    points1 = points1[
        mask_good
    ]

    scores = scores[
        mask_good
    ]

    metadata = {
        "max_keypoints":
            LIGHTGLUE_MAX_KEYPOINTS,

        "device":
            DEVICE,

        "runtime_seconds":
            float(
                runtime
            )
    }

    del (
        tensor0,
        tensor1,
        feats0,
        feats1,
        matches01
    )

    if DEVICE == "cuda":

        torch.cuda.empty_cache()

    return (
        points0,
        points1,
        scores,
        metadata
    )


# ============================================================
# LOFTR
# ============================================================

def create_loftr():

    from kornia.feature import LoFTR

    matcher = (
        LoFTR(
            pretrained="outdoor"
        )
        .eval()
        .to(
            DEVICE
        )
    )

    return matcher


def resize_for_loftr(
    image,
    max_dimension
):

    h, w = image.shape

    scale = min(
        1.0,
        float(
            max_dimension
        )
        /
        max(
            h,
            w
        )
    )

    new_h = max(
        8,
        int(
            round(
                h
                *
                scale
                /
                8
            )
            *
            8
        )
    )

    new_w = max(
        8,
        int(
            round(
                w
                *
                scale
                /
                8
            )
            *
            8
        )
    )

    resized = cv2.resize(
        image,
        (
            new_w,
            new_h
        ),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1.0
            else
            cv2.INTER_LINEAR
        )
    )

    scale_x_back = (
        w
        /
        new_w
    )

    scale_y_back = (
        h
        /
        new_h
    )

    return (
        resized,
        scale_x_back,
        scale_y_back
    )


def to_loftr_tensor(
    image
):

    tensor = torch.from_numpy(
        image.astype(
            np.float32
        )
        /
        255.0
    )

    tensor = (
        tensor[
            None,
            None,
            :,
            :
        ]
        .to(
            DEVICE
        )
    )

    return tensor


def match_loftr_once(
    image0,
    image1,
    safe_mask,
    matcher,
    max_dimension
):

    resized0, sx0, sy0 = (
        resize_for_loftr(
            image0,
            max_dimension
        )
    )

    resized1, sx1, sy1 = (
        resize_for_loftr(
            image1,
            max_dimension
        )
    )

    tensor0 = to_loftr_tensor(
        resized0
    )

    tensor1 = to_loftr_tensor(
        resized1
    )

    if DEVICE == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        output = matcher(
            {
                "image0":
                    tensor0,

                "image1":
                    tensor1
            }
        )

    if DEVICE == "cuda":

        torch.cuda.synchronize()

    runtime = (
        time.perf_counter()
        -
        start
    )

    points0 = (
        output[
            "keypoints0"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    points1 = (
        output[
            "keypoints1"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    confidence = (
        output[
            "confidence"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    # Scale coordinates back to original image.

    points0[
        :,
        0
    ] *= sx0

    points0[
        :,
        1
    ] *= sy0

    points1[
        :,
        0
    ] *= sx1

    points1[
        :,
        1
    ] *= sy1

    confidence_good = (
        confidence
        >=
        LOFTR_CONFIDENCE
    )

    points0 = points0[
        confidence_good
    ]

    points1 = points1[
        confidence_good
    ]

    confidence = confidence[
        confidence_good
    ]

    mask_good = (
        points_inside_mask(
            points0,
            safe_mask
        )
        &
        points_inside_mask(
            points1,
            safe_mask
        )
    )

    points0 = points0[
        mask_good
    ]

    points1 = points1[
        mask_good
    ]

    confidence = confidence[
        mask_good
    ]

    metadata = {
        "device":
            DEVICE,

        "max_dimension":
            max_dimension,

        "resized_source_shape":
            list(
                resized0.shape
            ),

        "resized_reference_shape":
            list(
                resized1.shape
            ),

        "confidence_threshold":
            LOFTR_CONFIDENCE,

        "runtime_seconds":
            float(
                runtime
            )
    }

    del (
        tensor0,
        tensor1,
        output
    )

    if DEVICE == "cuda":

        torch.cuda.empty_cache()

    return (
        points0,
        points1,
        confidence,
        metadata
    )


def match_loftr(
    image0,
    image1,
    safe_mask,
    matcher
):

    last_exception = None

    for max_dimension in (
        LOFTR_MAX_DIMENSIONS
    ):

        try:

            return match_loftr_once(
                image0,
                image1,
                safe_mask,
                matcher,
                max_dimension
            )

        except RuntimeError as exc:

            last_exception = exc

            message = str(
                exc
            ).lower()

            if (
                "out of memory"
                not in message
                and
                "cuda"
                not in message
            ):

                raise

            print(
                f"LoFTR failed at "
                f"max_dimension={max_dimension}."
            )

            print(
                "Retrying smaller..."
            )

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    raise RuntimeError(
        "LoFTR failed at all configured sizes."
    ) from last_exception


# ============================================================
# BENCHMARK ONE METHOD
# ============================================================

def run_benchmark(
    matcher_name,
    representation_name,
    image0,
    image1,
    safe_mask,
    match_function
):

    print("\n")
    print("=" * 80)

    print(
        f"{matcher_name.upper()} "
        f"- "
        f"{representation_name.upper()}"
    )

    print("=" * 80)

    (
        points0,
        points1,
        scores,
        matcher_metadata

    ) = match_function(
        image0,
        image1,
        safe_mask
    )

    print(
        "Candidate matches after filtering:",
        len(
            points0
        )
    )

    result, data = evaluate_matches(
        points0,
        points1,
        scores,
        safe_mask
    )

    result[
        "matcher"
    ] = matcher_name

    result[
        "representation"
    ] = representation_name

    result[
        "matcher_metadata"
    ] = matcher_metadata

    if (
        result.get(
            "status"
        )
        ==
        "ok"
    ):

        print(
            "Inliers:",
            result[
                "inliers"
            ]
        )

        print(
            "Inlier ratio:",
            result[
                "inlier_ratio"
            ]
        )

        print(
            "RANSAC reprojection RMSE:",
            result[
                "ransac_reprojection_rmse_px"
            ],
            "px"
        )

        print(
            "Median residual:",
            result[
                "ransac_reprojection_median_px"
            ],
            "px"
        )

        print(
            "Coverage:",
            result[
                "spatial"
            ][
                "coverage"
            ]
        )

        print(
            "Uniform inliers:",
            result[
                "uniform_inliers"
            ]
        )

        print(
            "Affine:"
        )

        print(
            np.asarray(
                result[
                    "affine_matrix"
                ]
            )
        )

        print(
            "Transform:"
        )

        print(
            result[
                "transform"
            ]
        )

        print(
            "Transform sanity:",
            result[
                "transform_sanity"
            ]
        )

        output_name = (
            f"{matcher_name}_"
            f"{representation_name}_"
            f"inliers.png"
        )

        save_match_visualization(
            image0,
            image1,
            data[
                "uniform_points0"
            ],
            data[
                "uniform_points1"
            ],
            OUTPUT_DIR
            /
            output_name
        )

    else:

        print(
            "Status:",
            result.get(
                "status"
            )
        )

    return result


# ============================================================
# CSV
# ============================================================

def write_csv(
    results
):

    fields = [
        "matcher",
        "representation",
        "status",
        "candidate_matches",
        "inliers",
        "inlier_ratio",
        "ransac_reprojection_rmse_px",
        "ransac_reprojection_median_px",
        "coverage",
        "uniform_inliers",
        "translation_px",
        "rotation_deg",
        "scale_x",
        "scale_y",
        "transform_sanity",
        "runtime_seconds"
    ]

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for result in results:

            transform = (
                result.get(
                    "transform"
                )
                or
                {}
            )

            spatial = (
                result.get(
                    "spatial"
                )
                or
                {}
            )

            metadata = (
                result.get(
                    "matcher_metadata"
                )
                or
                {}
            )

            row = {
                "matcher":
                    result.get(
                        "matcher"
                    ),

                "representation":
                    result.get(
                        "representation"
                    ),

                "status":
                    result.get(
                        "status"
                    ),

                "candidate_matches":
                    result.get(
                        "candidate_matches"
                    ),

                "inliers":
                    result.get(
                        "inliers"
                    ),

                "inlier_ratio":
                    result.get(
                        "inlier_ratio"
                    ),

                "ransac_reprojection_rmse_px":
                    result.get(
                        "ransac_reprojection_rmse_px"
                    ),

                "ransac_reprojection_median_px":
                    result.get(
                        "ransac_reprojection_median_px"
                    ),

                "coverage":
                    spatial.get(
                        "coverage"
                    ),

                "uniform_inliers":
                    result.get(
                        "uniform_inliers"
                    ),

                "translation_px":
                    transform.get(
                        "translation_px"
                    ),

                "rotation_deg":
                    transform.get(
                        "rotation_deg"
                    ),

                "scale_x":
                    transform.get(
                        "scale_x"
                    ),

                "scale_y":
                    transform.get(
                        "scale_y"
                    ),

                "transform_sanity":
                    result.get(
                        "transform_sanity"
                    ),

                "runtime_seconds":
                    metadata.get(
                        "runtime_seconds"
                    )
            }

            writer.writerow(
                row
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 003 GLOBAL MATCHER BENCHMARK"
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

    for path in [
        SOURCE_PATH,
        REFERENCE_PATH,
        MASK_PATH
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing:\n{path}"
            )


    # ========================================================
    # LOAD CANONICAL PAIR
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
            "Pair images/mask have different shapes."
        )

    safe_mask = make_safe_mask(
        mask_image
    )

    print(
        "\nShape:",
        source.shape
    )

    print(
        "Valid pixels:",
        int(
            np.count_nonzero(
                valid_mask
            )
        )
    )

    print(
        "Valid ratio:",
        float(
            valid_mask.mean()
        )
    )

    print(
        "Safe interior pixels:",
        int(
            np.count_nonzero(
                safe_mask
            )
        )
    )

    print(
        "Safe interior ratio:",
        float(
            safe_mask.mean()
        )
    )

    print(
        "Boundary exclusion:",
        BOUNDARY_MARGIN_PX,
        "px"
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
            OUTPUT_DIR
            /
            "source_gradient.png"
        ),
        source_gradient
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            /
            "reference_gradient.png"
        ),
        reference_gradient
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            /
            "safe_interior_mask.png"
        ),
        (
            safe_mask.astype(
                np.uint8
            )
            *
            255
        )
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
    # INITIALIZE DEEP MODELS ONCE
    # ========================================================

    print("\nLoading LightGlue...")

    lightglue_extractor, lightglue_matcher = (
        create_lightglue()
    )

    print(
        "LightGlue ready."
    )


    print("\nLoading LoFTR...")

    loftr_matcher = (
        create_loftr()
    )

    print(
        "LoFTR ready."
    )


    # ========================================================
    # RUN
    # ========================================================

    results = []


    for representation_name, (
        image0,
        image1

    ) in representations.items():

        # ----------------------------------------------------
        # SIFT
        # ----------------------------------------------------

        result = run_benchmark(
            "sift",
            representation_name,
            image0,
            image1,
            safe_mask,
            lambda a, b, m:
                match_sift(
                    a,
                    b,
                    m
                )
        )

        results.append(
            result
        )


        # ----------------------------------------------------
        # LIGHTGLUE
        # ----------------------------------------------------

        result = run_benchmark(
            "lightglue",
            representation_name,
            image0,
            image1,
            safe_mask,
            lambda a, b, m:
                match_lightglue(
                    a,
                    b,
                    m,
                    lightglue_extractor,
                    lightglue_matcher
                )
        )

        results.append(
            result
        )


        # ----------------------------------------------------
        # LOFTR
        # ----------------------------------------------------

        result = run_benchmark(
            "loftr",
            representation_name,
            image0,
            image1,
            safe_mask,
            lambda a, b, m:
                match_loftr(
                    a,
                    b,
                    m,
                    loftr_matcher
                )
        )

        results.append(
            result
        )


    # ========================================================
    # SAVE
    # ========================================================

    payload = {
        "pair_id":
            "pair_003",

        "source_sensor":
            "IIRS",

        "reference_sensor":
            "TMC-2",

        "image_shape":
            list(
                source.shape
            ),

        "boundary_margin_px":
            BOUNDARY_MARGIN_PX,

        "ransac_threshold_px":
            RANSAC_THRESHOLD_PX,

        "metric_note":
            (
                "RANSAC reprojection RMSE is an "
                "internal correspondence self-consistency "
                "residual. It is not independent "
                "ground-truth registration accuracy."
            ),

        "results":
            results
    }

    RESULTS_JSON.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )

    write_csv(
        results
    )


    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    print("\n")
    print("=" * 120)

    print(
        "PAIR 003 GLOBAL BENCHMARK SUMMARY"
    )

    print("=" * 120)

    header = (
        f"{'Matcher':<12}"
        f"{'Rep':<11}"
        f"{'Matches':>9}"
        f"{'Inliers':>9}"
        f"{'Ratio':>10}"
        f"{'RMSE':>10}"
        f"{'Median':>10}"
        f"{'Coverage':>11}"
        f"{'Uniform':>10}"
        f"{'Sane':>8}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for result in results:

        if (
            result.get(
                "status"
            )
            ==
            "ok"
        ):

            print(
                f"{result['matcher']:<12}"
                f"{result['representation']:<11}"
                f"{result['candidate_matches']:>9}"
                f"{result['inliers']:>9}"
                f"{result['inlier_ratio']:>10.3f}"
                f"{result['ransac_reprojection_rmse_px']:>10.3f}"
                f"{result['ransac_reprojection_median_px']:>10.3f}"
                f"{result['spatial']['coverage']:>11.3f}"
                f"{result['uniform_inliers']:>10}"
                f"{str(result['transform_sanity']):>8}"
            )

        else:

            print(
                f"{result['matcher']:<12}"
                f"{result['representation']:<11}"
                f"{result.get('candidate_matches', 0):>9}"
                f"{'-':>9}"
                f"{'-':>10}"
                f"{'-':>10}"
                f"{'-':>10}"
                f"{'-':>11}"
                f"{'-':>10}"
                f"{result.get('status', 'failed'):>8}"
            )


    print("\n")
    print("=" * 80)

    print(
        "PAIR 003 GLOBAL BENCHMARK COMPLETE"
    )

    print("=" * 80)

    print(
        "\nJSON:"
    )

    print(
        RESULTS_JSON
    )

    print(
        "\nCSV:"
    )

    print(
        RESULTS_CSV
    )

    print(
        "\nMatch visualizations:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()