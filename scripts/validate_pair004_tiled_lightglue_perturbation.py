from pathlib import Path
import json
import math
import time

import cv2
import numpy as np
import rasterio
from scipy import ndimage

import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pair_004"
    / "canonical"
)

SOURCE_PATH = (
    PAIR_DIR
    / "lro_nac_source.tif"
)

REFERENCE_PATH = (
    PAIR_DIR
    / "tmc2_reference.tif"
)

COMMON_MASK_PATH = (
    PAIR_DIR
    / "common_valid_mask.tif"
)

BASELINE_METRICS_PATH = (
    ROOT
    / "results"
    / "pair_004"
    / "tiled_lightglue"
    / "pair004_tiled_lightglue_metrics.json"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_004"
    / "tiled_lightglue_perturbation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TILE_SIZE = 768
TILE_STRIDE = 512

# Much wider than final real-pair pipeline.
# Known perturbation is NOT used to center this window.
SEARCH_MARGIN = 256

MIN_VALID_PIXELS = 8000

MAX_KEYPOINTS_PER_TILE = 4096

MASK_EROSION_PIXELS = 5

RANSAC_THRESHOLD_PX = 3.0

GT_GRID_STEP = 64


# ============================================================
# PERTURBATIONS
# ============================================================

PERTURBATIONS = [
    {
        "id": "SHIFT_1",
        "tx": 64.0,
        "ty": -48.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
    },

    {
        "id": "SHIFT_ROT_2",
        "tx": -120.0,
        "ty": 80.0,
        "rotation_deg": 0.35,
        "scale": 1.0,
    },

    {
        "id": "SHIFT_ROT_SCALE_3",
        "tx": 80.0,
        "ty": 60.0,
        "rotation_deg": -0.50,
        "scale": 1.002,
    },
]


# ============================================================
# IO
# ============================================================


def read_band(path):
    with rasterio.open(path) as ds:
        return (
            ds.read(1),
            ds.read_masks(1) > 0,
            ds.nodata,
        )


def load_baseline_affine():
    with open(
        BASELINE_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metrics = json.load(f)

    return np.asarray(
        metrics["affine"],
        dtype=np.float64,
    )


# ============================================================
# IMAGE PREPARATION
# ============================================================


def robust_uint8(data, mask):
    out = np.zeros(
        data.shape,
        dtype=np.uint8,
    )

    values = data[mask]
    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return out

    low, high = np.percentile(
        values,
        [2.0, 98.0],
    )

    if high <= low:
        high = low + 1.0

    scaled = (
        data.astype(np.float32)
        - float(low)
    ) / float(high - low)

    scaled = np.clip(
        scaled,
        0.0,
        1.0,
    )

    out[mask] = np.round(
        scaled[mask] * 255.0
    ).astype(np.uint8)

    return out


def gradient_uint8(image, mask):
    img = (
        image.astype(np.float32)
        / 255.0
    )

    img = cv2.GaussianBlur(
        img,
        (5, 5),
        0.8,
    )

    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = np.sqrt(
        gx * gx + gy * gy
    )

    out = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    values = magnitude[mask]

    if values.size == 0:
        return out

    high = np.percentile(
        values,
        99.0,
    )

    if high <= 0:
        return out

    magnitude = np.clip(
        magnitude / high,
        0.0,
        1.0,
    )

    out[mask] = np.round(
        magnitude[mask] * 255.0
    ).astype(np.uint8)

    return out


# ============================================================
# AFFINE
# ============================================================


def make_forward_affine(
    width,
    height,
    tx,
    ty,
    rotation_deg,
    scale,
):
    center = (
        (width - 1) / 2.0,
        (height - 1) / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        rotation_deg,
        scale,
    ).astype(np.float64)

    matrix[0, 2] += tx
    matrix[1, 2] += ty

    return matrix


def to_homogeneous(matrix):
    result = np.eye(
        3,
        dtype=np.float64,
    )

    result[:2, :] = matrix

    return result


def from_homogeneous(matrix):
    return matrix[:2, :].copy()


def compose(A, B):
    """
    A o B:
        x -> B(x) -> A(B(x))
    """
    return from_homogeneous(
        to_homogeneous(A)
        @ to_homogeneous(B)
    )


def affine_points(
    matrix,
    points,
):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    return (
        points
        @ matrix[:, :2].T
        + matrix[:, 2]
    )


def decompose_affine(matrix):
    A = matrix[:, :2]

    c0 = A[:, 0]
    c1 = A[:, 1]

    sx = float(
        np.linalg.norm(c0)
    )

    sy = float(
        np.linalg.norm(c1)
    )

    rotation = float(
        math.degrees(
            math.atan2(
                A[1, 0],
                A[0, 0],
            )
        )
    )

    tx = float(
        matrix[0, 2]
    )

    ty = float(
        matrix[1, 2]
    )

    determinant = float(
        np.linalg.det(A)
    )

    if sx > 0 and sy > 0:
        axis_dot = float(
            np.dot(c0, c1)
            / (sx * sy)
        )
    else:
        axis_dot = float("nan")

    return {
        "scale_x": sx,
        "scale_y": sy,
        "rotation_deg": rotation,

        "translation_x_px": tx,
        "translation_y_px": ty,

        "translation_magnitude_px": float(
            math.hypot(tx, ty)
        ),

        "determinant": determinant,
        "axis_dot": axis_dot,
    }


# ============================================================
# MASK
# ============================================================


def build_matcher_mask(
    source,
    source_mask,
    reference,
    reference_mask,
    common,
):
    science_valid = (
        common
        & source_mask
        & reference_mask
        & np.isfinite(source)
        & np.isfinite(reference)
    )

    matcher_pre = (
        science_valid
        & (reference > 0)
    )

    matcher = ndimage.binary_erosion(
        matcher_pre,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
        iterations=MASK_EROSION_PIXELS,
        border_value=0,
    )

    return (
        science_valid,
        matcher,
    )


def points_inside_mask(
    points,
    mask,
):
    if len(points) == 0:
        return np.zeros(
            0,
            dtype=bool,
        )

    h, w = mask.shape

    x = np.rint(
        points[:, 0]
    ).astype(np.int64)

    y = np.rint(
        points[:, 1]
    ).astype(np.int64)

    inside = (
        (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )

    result = np.zeros(
        len(points),
        dtype=bool,
    )

    ids = np.where(
        inside
    )[0]

    result[ids] = mask[
        y[ids],
        x[ids],
    ]

    return result


# ============================================================
# TILE UTILITIES
# ============================================================


def tile_starts(length):
    starts = list(
        range(
            0,
            max(
                1,
                length
                - TILE_SIZE
                + 1,
            ),
            TILE_STRIDE,
        )
    )

    final_start = max(
        0,
        length - TILE_SIZE,
    )

    if (
        len(starts) == 0
        or starts[-1]
        != final_start
    ):
        starts.append(
            final_start
        )

    return sorted(
        set(starts)
    )


def baseline_reference_bbox(
    baseline_affine,
    x0,
    y0,
    x1,
    y1,
    width,
    height,
):
    """
    IMPORTANT:

    Uses ONLY the original real-pair baseline.

    It does NOT know the synthetic perturbation.
    """

    corners = np.asarray(
        [
            [x0, y0],
            [x1, y0],
            [x0, y1],
            [x1, y1],
        ],
        dtype=np.float64,
    )

    predicted = affine_points(
        baseline_affine,
        corners,
    )

    rx0 = int(
        math.floor(
            predicted[:, 0].min()
            - SEARCH_MARGIN
        )
    )

    rx1 = int(
        math.ceil(
            predicted[:, 0].max()
            + SEARCH_MARGIN
        )
    )

    ry0 = int(
        math.floor(
            predicted[:, 1].min()
            - SEARCH_MARGIN
        )
    )

    ry1 = int(
        math.ceil(
            predicted[:, 1].max()
            + SEARCH_MARGIN
        )
    )

    rx0 = max(
        0,
        rx0,
    )

    ry0 = max(
        0,
        ry0,
    )

    rx1 = min(
        width,
        rx1,
    )

    ry1 = min(
        height,
        ry1,
    )

    return (
        rx0,
        ry0,
        rx1,
        ry1,
    )


# ============================================================
# LIGHTGLUE
# ============================================================


def pad_to_multiple_of_8(image):
    h, w = image.shape

    ph = (
        int(
            math.ceil(h / 8.0)
        )
        * 8
    )

    pw = (
        int(
            math.ceil(w / 8.0)
        )
        * 8
    )

    padded = np.zeros(
        (ph, pw),
        dtype=image.dtype,
    )

    padded[:h, :w] = image

    return (
        padded,
        h,
        w,
    )


def image_tensor(
    image,
    device,
):
    return torch.from_numpy(
        image.astype(np.float32)
        / 255.0
    ).unsqueeze(0).to(device)


def run_lightglue_crop(
    extractor,
    matcher,
    source_crop,
    reference_crop,
    device,
):
    (
        source_pad,
        source_h,
        source_w,
    ) = pad_to_multiple_of_8(
        source_crop
    )

    (
        reference_pad,
        reference_h,
        reference_w,
    ) = pad_to_multiple_of_8(
        reference_crop
    )

    image0 = image_tensor(
        source_pad,
        device,
    )

    image1 = image_tensor(
        reference_pad,
        device,
    )

    with torch.inference_mode():

        feats0 = extractor.extract(
            image0
        )

        feats1 = extractor.extract(
            image1
        )

        prediction = matcher(
            {
                "image0": feats0,
                "image1": feats1,
            }
        )

    feats0 = rbd(feats0)
    feats1 = rbd(feats1)
    prediction = rbd(
        prediction
    )

    kp0 = (
        feats0["keypoints"]
        .detach()
        .cpu()
        .numpy()
    )

    kp1 = (
        feats1["keypoints"]
        .detach()
        .cpu()
        .numpy()
    )

    if "matches" in prediction:

        matches = (
            prediction["matches"]
            .detach()
            .cpu()
            .numpy()
        )

        if matches.size == 0:
            return (
                np.empty(
                    (0, 2),
                    dtype=np.float32,
                ),
                np.empty(
                    (0, 2),
                    dtype=np.float32,
                ),
            )

        source_points = kp0[
            matches[:, 0]
        ]

        reference_points = kp1[
            matches[:, 1]
        ]

    elif "matches0" in prediction:

        matches0 = (
            prediction["matches0"]
            .detach()
            .cpu()
            .numpy()
        )

        ids0 = np.where(
            matches0 >= 0
        )[0]

        ids1 = matches0[
            ids0
        ]

        source_points = kp0[
            ids0
        ]

        reference_points = kp1[
            ids1
        ]

    else:
        raise RuntimeError(
            "Unknown LightGlue output."
        )

    valid = (
        (source_points[:, 0] >= 0)
        & (source_points[:, 0] < source_w)
        & (source_points[:, 1] >= 0)
        & (source_points[:, 1] < source_h)

        & (reference_points[:, 0] >= 0)
        & (reference_points[:, 0] < reference_w)
        & (reference_points[:, 1] >= 0)
        & (reference_points[:, 1] < reference_h)
    )

    return (
        source_points[
            valid
        ].astype(
            np.float32
        ),

        reference_points[
            valid
        ].astype(
            np.float32
        ),
    )


# ============================================================
# DEDUPLICATION
# ============================================================


def deduplicate_matches(
    source_points,
    reference_points,
):
    if len(source_points) == 0:
        return (
            source_points,
            reference_points,
        )

    seen = {}
    selected = []

    for i, (
        source,
        reference,
    ) in enumerate(
        zip(
            source_points,
            reference_points,
        )
    ):

        key = (
            round(
                float(source[0])
                * 2
            ),

            round(
                float(source[1])
                * 2
            ),

            round(
                float(reference[0])
                * 2
            ),

            round(
                float(reference[1])
                * 2
            ),
        )

        if key in seen:
            continue

        seen[key] = True
        selected.append(i)

    selected = np.asarray(
        selected,
        dtype=np.int64,
    )

    return (
        source_points[
            selected
        ],

        reference_points[
            selected
        ],
    )


# ============================================================
# TRANSFORM RESPONSE METRIC
# ============================================================


def transform_response_rmse(
    recovered,
    expected,
    source_mask,
    reference_mask,
):
    h, w = source_mask.shape

    points = []

    for y in range(
        0,
        h,
        GT_GRID_STEP,
    ):
        for x in range(
            0,
            w,
            GT_GRID_STEP,
        ):

            if not source_mask[
                y,
                x
            ]:
                continue

            expected_point = (
                affine_points(
                    expected,
                    [[x, y]],
                )[0]
            )

            ex = int(
                round(
                    expected_point[0]
                )
            )

            ey = int(
                round(
                    expected_point[1]
                )
            )

            if (
                ex < 0
                or ex >= w
                or ey < 0
                or ey >= h
            ):
                continue

            if not reference_mask[
                ey,
                ex
            ]:
                continue

            points.append(
                [x, y]
            )

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if len(points) == 0:
        return None, 0

    recovered_points = (
        affine_points(
            recovered,
            points,
        )
    )

    expected_points = (
        affine_points(
            expected,
            points,
        )
    )

    errors = np.linalg.norm(
        recovered_points
        - expected_points,
        axis=1,
    )

    return (
        float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        ),
        int(len(points)),
    )


# ============================================================
# RUN ONE CONTROL
# ============================================================


def run_control(
    perturbation,
    source_intensity,
    reference_gradient,
    matcher_valid,
    baseline_affine,
    extractor,
    lightglue,
    device,
):
    print("\n")
    print("=" * 78)
    print(
        perturbation["id"]
    )
    print("=" * 78)

    h, w = (
        source_intensity.shape
    )

    # --------------------------------------------------------
    # Known source perturbation
    # --------------------------------------------------------

    forward = make_forward_affine(
        w,
        h,

        perturbation["tx"],
        perturbation["ty"],

        perturbation[
            "rotation_deg"
        ],

        perturbation[
            "scale"
        ],
    )

    inverse = (
        cv2.invertAffineTransform(
            forward
        )
    )

    # Real baseline:
    #
    # original NAC -> TMC = B
    #
    # Perturbed source:
    #
    # original NAC -> perturbed NAC = W
    #
    # Therefore expected:
    #
    # perturbed NAC -> TMC = B o inverse(W)
    #
    expected_affine = compose(
        baseline_affine,
        inverse,
    )

    print(
        "\nInjected original->perturbed NAC:"
    )
    print(forward)

    print(
        "\nExpected perturbed NAC->TMC:"
    )
    print(expected_affine)

    # --------------------------------------------------------
    # Warp source
    # --------------------------------------------------------

    transformed_intensity = (
        cv2.warpAffine(
            source_intensity,
            forward,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=(
                cv2.BORDER_CONSTANT
            ),
            borderValue=0,
        )
    )

    transformed_mask = (
        cv2.warpAffine(
            matcher_valid.astype(
                np.uint8
            ),
            forward,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=(
                cv2.BORDER_CONSTANT
            ),
            borderValue=0,
        )
        > 0
    )

    transformed_mask = (
        ndimage.binary_erosion(
            transformed_mask,
            structure=np.ones(
                (3, 3),
                dtype=bool,
            ),
            iterations=2,
            border_value=0,
        )
    )

    transformed_gradient = (
        gradient_uint8(
            transformed_intensity,
            transformed_mask,
        )
    )

    transformed_gradient[
        ~transformed_mask
    ] = 0

    # --------------------------------------------------------
    # Native tiled LightGlue
    #
    # Notice:
    # - uses original BASELINE affine for search centres
    # - does NOT use expected_affine
    # - does NOT apply a prior-distance gate
    # --------------------------------------------------------

    xs = tile_starts(
        w
    )

    ys = tile_starts(
        h
    )

    total_tiles = (
        len(xs)
        * len(ys)
    )

    all_source = []
    all_reference = []

    raw_total = 0
    mask_total = 0
    tiles_run = 0

    start_time = (
        time.perf_counter()
    )

    tile_number = 0

    for y0 in ys:
        for x0 in xs:

            tile_number += 1

            x1 = min(
                w,
                x0 + TILE_SIZE,
            )

            y1 = min(
                h,
                y0 + TILE_SIZE,
            )

            source_tile_mask = (
                transformed_mask[
                    y0:y1,
                    x0:x1,
                ]
            )

            valid_pixels = int(
                source_tile_mask.sum()
            )

            if (
                valid_pixels
                < MIN_VALID_PIXELS
            ):
                continue

            (
                rx0,
                ry0,
                rx1,
                ry1,
            ) = baseline_reference_bbox(
                baseline_affine,
                x0,
                y0,
                x1,
                y1,
                w,
                h,
            )

            if (
                rx1 <= rx0
                or ry1 <= ry0
            ):
                continue

            source_crop = (
                transformed_gradient[
                    y0:y1,
                    x0:x1,
                ]
            )

            reference_crop = (
                reference_gradient[
                    ry0:ry1,
                    rx0:rx1,
                ]
            )

            (
                source_local,
                reference_local,
            ) = run_lightglue_crop(
                extractor,
                lightglue,

                source_crop,
                reference_crop,

                device,
            )

            raw_count = len(
                source_local
            )

            raw_total += (
                raw_count
            )

            if raw_count == 0:
                continue

            source_native = (
                source_local
                + np.asarray(
                    [x0, y0],
                    dtype=np.float32,
                )
            )

            reference_native = (
                reference_local
                + np.asarray(
                    [rx0, ry0],
                    dtype=np.float32,
                )
            )

            mask_keep = (
                points_inside_mask(
                    source_native,
                    transformed_mask,
                )
                &
                points_inside_mask(
                    reference_native,
                    matcher_valid,
                )
            )

            source_native = (
                source_native[
                    mask_keep
                ]
            )

            reference_native = (
                reference_native[
                    mask_keep
                ]
            )

            mask_total += len(
                source_native
            )

            if len(
                source_native
            ) > 0:

                all_source.append(
                    source_native
                )

                all_reference.append(
                    reference_native
                )

            tiles_run += 1

            print(
                f"Tile "
                f"{tile_number:02d}/"
                f"{total_tiles:02d} "
                f"valid={valid_pixels:,} "
                f"raw={raw_count:<4} "
                f"kept={len(source_native)}"
            )

    runtime = (
        time.perf_counter()
        - start_time
    )

    if len(
        all_source
    ) == 0:
        raise RuntimeError(
            "No LightGlue matches."
        )

    source_points = (
        np.concatenate(
            all_source,
            axis=0,
        )
    )

    reference_points = (
        np.concatenate(
            all_reference,
            axis=0,
        )
    )

    before_dedup = len(
        source_points
    )

    (
        source_points,
        reference_points,
    ) = deduplicate_matches(
        source_points,
        reference_points,
    )

    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    if len(
        source_points
    ) < 4:
        raise RuntimeError(
            "Too few matches."
        )

    recovered_affine, inlier_mask = (
        cv2.estimateAffine2D(
            source_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=(
                RANSAC_THRESHOLD_PX
            ),
            maxIters=30000,
            confidence=0.999,
            refineIters=100,
        )
    )

    if (
        recovered_affine is None
        or inlier_mask is None
    ):
        raise RuntimeError(
            "RANSAC failed."
        )

    inlier_mask = (
        inlier_mask.ravel() > 0
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

    predicted = affine_points(
        recovered_affine,
        source_inliers,
    )

    self_errors = np.linalg.norm(
        predicted
        - reference_inliers,
        axis=1,
    )

    self_rmse = float(
        np.sqrt(
            np.mean(
                self_errors ** 2
            )
        )
    )

    # --------------------------------------------------------
    # Perturbation-response evaluation
    # --------------------------------------------------------

    response_rmse, eval_points = (
        transform_response_rmse(
            recovered_affine,
            expected_affine,
            transformed_mask,
            matcher_valid,
        )
    )

    # Compensate recovered solution by known injected warp.
    # Should return to original baseline.
    compensated_affine = compose(
        recovered_affine,
        forward,
    )

    compensated_rmse, _ = (
        transform_response_rmse(
            compensated_affine,
            baseline_affine,
            matcher_valid,
            matcher_valid,
        )
    )

    identity = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    identity_rmse, _ = (
        transform_response_rmse(
            recovered_affine,
            identity,
            transformed_mask,
            matcher_valid,
        )
    )

    print("\n")
    print("-" * 78)

    print(
        "Raw tiled matches       :",
        raw_total,
    )

    print(
        "Mask-filtered matches   :",
        mask_total,
    )

    print(
        "Before dedup            :",
        before_dedup,
    )

    print(
        "Distinct candidates     :",
        len(source_points),
    )

    print(
        "RANSAC inliers          :",
        len(source_inliers),
    )

    print(
        "Inlier ratio            :",
        f"{len(source_inliers) / len(source_points):.6f}",
    )

    print(
        "Self RMSE px            :",
        f"{self_rmse:.6f}",
    )

    print(
        "\nPERTURBATION RESPONSE RMSE:",
        f"{response_rmse:.6f} px",
    )

    print(
        "COMPENSATED BASELINE RMSE:",
        f"{compensated_rmse:.6f} px",
    )

    print(
        "RECOVERED VS IDENTITY    :",
        f"{identity_rmse:.6f} px",
    )

    print(
        "Evaluation points        :",
        eval_points,
    )

    print(
        "Runtime s                :",
        f"{runtime:.3f}",
    )

    print(
        "\nExpected affine:"
    )

    print(
        expected_affine
    )

    print(
        "\nRecovered affine:"
    )

    print(
        recovered_affine
    )

    print(
        "\nExpected parameters:"
    )

    print(
        json.dumps(
            decompose_affine(
                expected_affine
            ),
            indent=2,
        )
    )

    print(
        "\nRecovered parameters:"
    )

    print(
        json.dumps(
            decompose_affine(
                recovered_affine
            ),
            indent=2,
        )
    )

    return {
        "control_id":
            perturbation["id"],

        "injected_transform":
            perturbation,

        "raw_matches":
            int(raw_total),

        "mask_filtered_matches":
            int(mask_total),

        "distinct_candidates":
            int(
                len(source_points)
            ),

        "ransac_inliers":
            int(
                len(source_inliers)
            ),

        "ransac_inlier_ratio":
            float(
                len(source_inliers)
                / len(source_points)
            ),

        "ransac_self_rmse_px":
            self_rmse,

        "perturbation_response_rmse_px":
            float(
                response_rmse
            ),

        "compensated_baseline_rmse_px":
            float(
                compensated_rmse
            ),

        "recovered_vs_identity_rmse_px":
            float(
                identity_rmse
            ),

        "evaluation_points":
            int(eval_points),

        "expected_affine":
            expected_affine.tolist(),

        "recovered_affine":
            recovered_affine.tolist(),

        "expected_parameters":
            decompose_affine(
                expected_affine
            ),

        "recovered_parameters":
            decompose_affine(
                recovered_affine
            ),

        "runtime_s":
            float(runtime),
    }


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 78)

    print(
        "PAIR 004 — TILED LIGHTGLUE "
        "CONTROLLED PERTURBATION VALIDATION"
    )

    print("=" * 78)

    source, source_mask, _ = (
        read_band(
            SOURCE_PATH
        )
    )

    reference, reference_mask, _ = (
        read_band(
            REFERENCE_PATH
        )
    )

    common_data, _, _ = (
        read_band(
            COMMON_MASK_PATH
        )
    )

    common = (
        common_data > 0
    )

    (
        science_valid,
        matcher_valid,
    ) = build_matcher_mask(
        source,
        source_mask,
        reference,
        reference_mask,
        common,
    )

    print(
        "\nShape:",
        source.shape,
    )

    print(
        "Science-valid pixels:",
        f"{int(science_valid.sum()):,}",
    )

    print(
        "Matcher-valid pixels:",
        f"{int(matcher_valid.sum()):,}",
    )

    # --------------------------------------------------------
    # Original representations
    # --------------------------------------------------------

    source_intensity = (
        robust_uint8(
            source,
            matcher_valid,
        )
    )

    reference_intensity = (
        robust_uint8(
            reference,
            matcher_valid,
        )
    )

    source_intensity[
        ~matcher_valid
    ] = 0

    reference_intensity[
        ~matcher_valid
    ] = 0

    reference_gradient = (
        gradient_uint8(
            reference_intensity,
            matcher_valid,
        )
    )

    reference_gradient[
        ~matcher_valid
    ] = 0

    # --------------------------------------------------------
    # Real-pair baseline
    # --------------------------------------------------------

    baseline_affine = (
        load_baseline_affine()
    )

    print(
        "\nFrozen tiled-LightGlue "
        "real-pair baseline:"
    )

    print(
        baseline_affine
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    extractor = SuperPoint(
        max_num_keypoints=(
            MAX_KEYPOINTS_PER_TILE
        ),
    ).eval().to(device)

    lightglue = LightGlue(
        features="superpoint",
    ).eval().to(device)

    # --------------------------------------------------------
    # Run controls
    # --------------------------------------------------------

    results = []

    for perturbation in (
        PERTURBATIONS
    ):

        result = run_control(
            perturbation,

            source_intensity,
            reference_gradient,

            matcher_valid,

            baseline_affine,

            extractor,
            lightglue,

            device,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        OUT_DIR
        / "pair004_tiled_lightglue_perturbation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for result in results:

        print(
            f"{result['control_id']:<20} "
            f"inliers="
            f"{result['ransac_inliers']:<5} "
            f"self="
            f"{result['ransac_self_rmse_px']:.3f}px "
            f"response="
            f"{result['perturbation_response_rmse_px']:.3f}px "
            f"compensated="
            f"{result['compensated_baseline_rmse_px']:.3f}px "
            f"identity="
            f"{result['recovered_vs_identity_rmse_px']:.1f}px"
        )

    response_values = np.asarray(
        [
            result[
                "perturbation_response_rmse_px"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    print(
        "\nMean perturbation-response RMSE:",
        f"{response_values.mean():.6f}px",
    )

    print(
        "Worst perturbation-response RMSE:",
        f"{response_values.max():.6f}px",
    )

    print(
        "\nOutput:"
    )

    print(
        output_path
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This validates whether the real cross-sensor "
        "matcher follows known injected source motion."
    )

    print(
        "It still does NOT provide independent absolute "
        "ground truth for the unperturbed real NAC↔TMC pair."
    )

    print(
        "\nPAIR 004 TILED LIGHTGLUE "
        "PERTURBATION VALIDATION COMPLETE."
    )


if __name__ == "__main__":
    main()