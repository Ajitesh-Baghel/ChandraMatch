from pathlib import Path
import csv
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
    / "pair_006"
    / "canonical"
)

SOURCE_PATH = (
    PAIR_DIR
    / "source_lro_nac.tif"
)

REFERENCE_PATH = (
    PAIR_DIR
    / "reference_tmc2.tif"
)

COMMON_MASK_PATH = (
    PAIR_DIR
    / "matcher_mask.tif"
)

GLOBAL_RESULTS_PATH = (
    ROOT
    / "results"
    / "pair_006"
    / "global_benchmark"
    / "pair006_global_benchmark.json"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_006"
    / "offset_scan"
    / "offset_p0768"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PIXEL_SIZE_M = 5.0

# Native-resolution tiling
TILE_SIZE = 768
TILE_STRIDE = 512

# Search around global LightGlue affine
SEARCH_MARGIN = 128

# Skip nearly empty footprint tiles
MIN_VALID_PIXELS = 8000

# Generous prior gate.
# This is NOT being used to prove correctness.
# It just prevents completely unrelated distant matches.
PRIOR_GATE_PX = 80.0

# Native-grid RANSAC
RANSAC_THRESHOLD_PX = 3.0

MASK_EROSION_PIXELS = 5

GRID_ROWS = 8
GRID_COLS = 8

MAX_KEYPOINTS_PER_TILE = 4096


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
        gx * gx
        + gy * gy
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

    # TMC zero-DN shadows remain science-valid,
    # but are excluded from feature extraction.
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
        matcher_pre,
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
# GLOBAL PRIOR
# ============================================================


def load_global_affine():
    with open(
        GLOBAL_RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        results = json.load(f)

    for result in results:
        if (
            result["method"]
            == "lightglue"
            and
            result["representation"]
            == "gradient"
        ):
            if result["affine"] is None:
                raise RuntimeError(
                    "Global LightGlue gradient "
                    "affine is missing."
                )

            return np.asarray(
                result["affine"],
                dtype=np.float64,
            )

    raise RuntimeError(
        "Global LightGlue gradient "
        "result not found."
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


# ============================================================
# TILING
# ============================================================


def tile_starts(length):
    starts = list(
        range(
            0,
            max(
                1,
                length - TILE_SIZE + 1,
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
        or starts[-1] != final_start
    ):
        starts.append(
            final_start
        )

    return sorted(
        set(starts)
    )


def predicted_reference_bbox(
    affine,
    x0,
    y0,
    x1,
    y1,
    width,
    height,
):
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
        affine,
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
# LIGHTGLUE HELPERS
# ============================================================


def pad_to_multiple_of_8(image):
    h, w = image.shape

    padded_h = (
        int(
            math.ceil(
                h / 8.0
            )
        )
        * 8
    )

    padded_w = (
        int(
            math.ceil(
                w / 8.0
            )
        )
        * 8
    )

    padded = np.zeros(
        (padded_h, padded_w),
        dtype=image.dtype,
    )

    padded[
        :h,
        :w,
    ] = image

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
        image.astype(
            np.float32
        ) / 255.0
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

    feats0 = rbd(
        feats0
    )

    feats1 = rbd(
        feats1
    )

    prediction = rbd(
        prediction
    )

    keypoints0 = (
        feats0["keypoints"]
        .detach()
        .cpu()
        .numpy()
    )

    keypoints1 = (
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

        source_points = keypoints0[
            matches[:, 0]
        ]

        reference_points = keypoints1[
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

        source_points = keypoints0[
            ids0
        ]

        reference_points = keypoints1[
            ids1
        ]

    else:
        raise RuntimeError(
            "Unrecognized LightGlue output."
        )

    # Remove detections that landed inside padding.
    valid = (
        (source_points[:, 0] >= 0)
        & (
            source_points[:, 0]
            < source_w
        )
        & (source_points[:, 1] >= 0)
        & (
            source_points[:, 1]
            < source_h
        )
        & (
            reference_points[:, 0]
            >= 0
        )
        & (
            reference_points[:, 0]
            < reference_w
        )
        & (
            reference_points[:, 1]
            >= 0
        )
        & (
            reference_points[:, 1]
            < reference_h
        )
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
    if len(
        source_points
    ) == 0:
        return (
            source_points,
            reference_points,
        )

    best = {}

    for i, (
        source,
        reference,
    ) in enumerate(
        zip(
            source_points,
            reference_points,
        )
    ):
        # 0.5-pixel quantization.
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

        if key not in best:
            best[key] = i

    indices = np.asarray(
        sorted(
            best.values()
        ),
        dtype=np.int64,
    )

    return (
        source_points[
            indices
        ],
        reference_points[
            indices
        ],
    )


# ============================================================
# METRICS
# ============================================================


def decompose_affine(matrix):
    A = matrix[
        :,
        :2,
    ]

    column0 = A[
        :,
        0,
    ]

    column1 = A[
        :,
        1,
    ]

    scale_x = float(
        np.linalg.norm(
            column0
        )
    )

    scale_y = float(
        np.linalg.norm(
            column1
        )
    )

    determinant = float(
        np.linalg.det(A)
    )

    rotation = float(
        math.degrees(
            math.atan2(
                A[1, 0],
                A[0, 0],
            )
        )
    )

    translation_x = float(
        matrix[0, 2]
    )

    translation_y = float(
        matrix[1, 2]
    )

    translation = float(
        math.hypot(
            translation_x,
            translation_y,
        )
    )

    if (
        scale_x > 0
        and scale_y > 0
    ):
        axis_dot = float(
            np.dot(
                column0,
                column1,
            )
            / (
                scale_x
                * scale_y
            )
        )
    else:
        axis_dot = float(
            "nan"
        )

    return {
        "scale_x":
            scale_x,

        "scale_y":
            scale_y,

        "rotation_deg":
            rotation,

        "translation_x_px":
            translation_x,

        "translation_y_px":
            translation_y,

        "translation_magnitude_px":
            translation,

        "determinant":
            determinant,

        "axis_dot":
            axis_dot,
    }


def affine_sanity(info):
    return bool(
        info["determinant"] > 0
        and
        0.90
        <= info["scale_x"]
        <= 1.10
        and
        0.90
        <= info["scale_y"]
        <= 1.10
        and
        abs(
            info["rotation_deg"]
        ) <= 5.0
        and
        info[
            "translation_magnitude_px"
        ] <= 100.0
        and
        abs(
            info["axis_dot"]
        ) <= 0.15
    )


def spatial_coverage(
    points,
    valid_mask,
):
    h, w = (
        valid_mask.shape
    )

    valid_cells = set()

    for gy in range(
        GRID_ROWS
    ):
        y0 = int(
            round(
                gy * h
                / GRID_ROWS
            )
        )

        y1 = int(
            round(
                (gy + 1) * h
                / GRID_ROWS
            )
        )

        for gx in range(
            GRID_COLS
        ):
            x0 = int(
                round(
                    gx * w
                    / GRID_COLS
                )
            )

            x1 = int(
                round(
                    (gx + 1) * w
                    / GRID_COLS
                )
            )

            if np.any(
                valid_mask[
                    y0:y1,
                    x0:x1,
                ]
            ):
                valid_cells.add(
                    (
                        gy,
                        gx,
                    )
                )

    occupied = set()

    for x, y in points:

        gx = min(
            GRID_COLS - 1,
            max(
                0,
                int(
                    x / w
                    * GRID_COLS
                ),
            ),
        )

        gy = min(
            GRID_ROWS - 1,
            max(
                0,
                int(
                    y / h
                    * GRID_ROWS
                ),
            ),
        )

        cell = (
            gy,
            gx,
        )

        if cell in valid_cells:
            occupied.add(
                cell
            )

    denominator = len(
        valid_cells
    )

    numerator = len(
        occupied
    )

    ratio = (
        numerator
        / denominator
        if denominator
        else 0.0
    )

    return (
        numerator,
        denominator,
        ratio,
    )


# ============================================================
# VISUALIZATION
# ============================================================


def save_matches(
    source,
    reference,
    source_points,
    reference_points,
    path,
):
    h, w = source.shape

    max_dim = 1800

    scale = min(
        1.0,
        max_dim
        / max(
            h,
            w,
        ),
    )

    new_w = int(
        round(
            w * scale
        )
    )

    new_h = int(
        round(
            h * scale
        )
    )

    source_small = cv2.resize(
        source,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    reference_small = cv2.resize(
        reference,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    source_rgb = cv2.cvtColor(
        source_small,
        cv2.COLOR_GRAY2BGR,
    )

    reference_rgb = cv2.cvtColor(
        reference_small,
        cv2.COLOR_GRAY2BGR,
    )

    canvas = np.concatenate(
        [
            source_rgb,
            reference_rgb,
        ],
        axis=1,
    )

    if len(
        source_points
    ) > 500:
        ids = np.linspace(
            0,
            len(source_points) - 1,
            500,
        ).astype(
            int
        )

        source_points = (
            source_points[
                ids
            ]
        )

        reference_points = (
            reference_points[
                ids
            ]
        )

    for source_point, reference_point in zip(
        source_points,
        reference_points,
    ):

        x0 = int(
            round(
                source_point[0]
                * scale
            )
        )

        y0 = int(
            round(
                source_point[1]
                * scale
            )
        )

        x1 = int(
            round(
                reference_point[0]
                * scale
            )
        ) + new_w

        y1 = int(
            round(
                reference_point[1]
                * scale
            )
        )

        cv2.circle(
            canvas,
            (
                x0,
                y0,
            ),
            3,
            (
                0,
                255,
                0,
            ),
            -1,
        )

        cv2.circle(
            canvas,
            (
                x1,
                y1,
            ),
            3,
            (
                0,
                255,
                0,
            ),
            -1,
        )

        cv2.line(
            canvas,
            (
                x0,
                y0,
            ),
            (
                x1,
                y1,
            ),
            (
                0,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(path),
        canvas,
    )


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 78)

    print(
        "PAIR 004 — PRIOR-GUIDED "
        "TILED LIGHTGLUE GRADIENT"
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

    if (
        source.shape
        != reference.shape
    ):
        raise RuntimeError(
            "Canonical shapes differ."
        )

    h, w = source.shape

    print(
        "\nCanonical shape:",
        h,
        "x",
        w,
    )

    (
        science_valid,
        matcher_pre,
        matcher_valid,
    ) = build_matcher_mask(
        source,
        source_mask,
        reference,
        reference_mask,
        common,
    )

    print(
        "Science-valid pixels   :",
        f"{int(science_valid.sum()):,}",
    )

    print(
        "Nonzero matcher pixels :",
        f"{int(matcher_pre.sum()):,}",
    )

    print(
        "Eroded matcher pixels  :",
        f"{int(matcher_valid.sum()):,}",
    )

    # --------------------------------------------------------
    # Build gradient representation
    # --------------------------------------------------------

    source_intensity = robust_uint8(
        source,
        matcher_valid,
    )

    reference_intensity = robust_uint8(
        reference,
        matcher_valid,
    )

    source_intensity[
        ~matcher_valid
    ] = 0

    reference_intensity[
        ~matcher_valid
    ] = 0

    source_gradient = gradient_uint8(
        source_intensity,
        matcher_valid,
    )

    reference_gradient = gradient_uint8(
        reference_intensity,
        matcher_valid,
    )

    source_gradient[
        ~matcher_valid
    ] = 0

    reference_gradient[
        ~matcher_valid
    ] = 0

    # --------------------------------------------------------
    # Prior
    # --------------------------------------------------------

    global_affine = (
        load_global_affine()
    )

    # --------------------------------------------------------
    # DIAGNOSTIC OFFSET HYPOTHESIS
    # --------------------------------------------------------

    HYPOTHESIS_OFFSET_X_PX = 768

    global_affine = (
        global_affine.copy()
    )

    global_affine[
        0,
        2,
    ] += HYPOTHESIS_OFFSET_X_PX

    print(
        "\nDiagnostic X-offset hypothesis:",
        HYPOTHESIS_OFFSET_X_PX,
        "px",
    )

    print(
        "\nGlobal LightGlue-gradient affine:"
    )

    print(
        global_affine
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

    if (
        device.type
        == "cuda"
    ):
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
    ).eval().to(
        device
    )

    lightglue = LightGlue(
        features="superpoint",
    ).eval().to(
        device
    )

    # --------------------------------------------------------
    # Tiles
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

    print(
        "\nCandidate tiles:",
        total_tiles,
    )

    all_source = []
    all_reference = []

    raw_total = 0
    mask_total = 0
    prior_total = 0

    tiles_run = 0

    start_time = time.perf_counter()

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

            tile_mask = (
                matcher_valid[
                    y0:y1,
                    x0:x1,
                ]
            )

            valid_pixels = int(
                tile_mask.sum()
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
            ) = predicted_reference_bbox(
                global_affine,
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
                source_gradient[
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

            if (
                raw_count == 0
            ):
                continue

            source_native = (
                source_local
                + np.asarray(
                    [
                        x0,
                        y0,
                    ],
                    dtype=np.float32,
                )
            )

            reference_native = (
                reference_local
                + np.asarray(
                    [
                        rx0,
                        ry0,
                    ],
                    dtype=np.float32,
                )
            )

            mask_keep = (
                points_inside_mask(
                    source_native,
                    matcher_valid,
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
            ) == 0:
                continue

            predicted_reference = (
                affine_points(
                    global_affine,
                    source_native,
                )
            )

            prior_error = (
                np.linalg.norm(
                    predicted_reference
                    - reference_native,
                    axis=1,
                )
            )

            prior_keep = (
                prior_error
                <= PRIOR_GATE_PX
            )

            source_native = (
                source_native[
                    prior_keep
                ]
            )

            reference_native = (
                reference_native[
                    prior_keep
                ]
            )

            prior_total += len(
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
                f"src=({x0},{y0}) "
                f"valid="
                f"{valid_pixels:,} "
                f"raw="
                f"{raw_count} "
                f"kept="
                f"{len(source_native)}"
            )

    runtime = (
        time.perf_counter()
        - start_time
    )

    if len(
        all_source
    ) == 0:
        raise RuntimeError(
            "No tiled LightGlue "
            "correspondences survived."
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
    # DIAGNOSTIC: SAVE ALL DISTINCT PRE-RANSAC CANDIDATES
    # --------------------------------------------------------

    candidate_csv_path = (
        OUT_DIR
        / "pair006_offset_p0768_candidates.csv"
    )

    with open(
        candidate_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "source_x",
                "source_y",
                "reference_x",
                "reference_y",
                "dx_px",
                "dy_px",
                "displacement_px",
            ]
        )

        for src, ref in zip(
            source_points,
            reference_points,
        ):

            dx = float(
                ref[0] - src[0]
            )

            dy = float(
                ref[1] - src[1]
            )

            displacement = float(
                np.hypot(
                    dx,
                    dy,
                )
            )

            writer.writerow(
                [
                    float(src[0]),
                    float(src[1]),
                    float(ref[0]),
                    float(ref[1]),
                    dx,
                    dy,
                    displacement,
                ]
            )

    print(
        "\nSaved distinct pre-RANSAC candidates:"
    )

    print(
        candidate_csv_path
    )


    print("\n")
    print("=" * 78)
    print(
        "TILED MATCH COLLECTION"
    )
    print("=" * 78)

    print(
        "Tiles actually run        :",
        tiles_run,
    )

    print(
        "Raw LightGlue matches     :",
        raw_total,
    )

    print(
        "After matcher-mask filter :",
        mask_total,
    )

    print(
        "After generous prior gate :",
        prior_total,
    )

    print(
        "Before deduplication      :",
        before_dedup,
    )

    print(
        "Distinct correspondences  :",
        len(source_points),
    )

    if len(
        source_points
    ) < 4:
        raise RuntimeError(
            "Too few matches "
            "for RANSAC."
        )

    # --------------------------------------------------------
    # Global RANSAC
    # --------------------------------------------------------

    matrix, inlier_mask = (
        cv2.estimateAffine2D(
            source_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=(
                RANSAC_THRESHOLD_PX
            ),
            maxIters=20000,
            confidence=0.999,
            refineIters=100,
        )
    )

    if (
        matrix is None
        or inlier_mask is None
    ):
        raise RuntimeError(
            "LightGlue RANSAC failed."
        )

    inlier_mask = (
        inlier_mask.ravel()
        > 0
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
        matrix,
        source_inliers,
    )

    residuals = np.linalg.norm(
        predicted
        - reference_inliers,
        axis=1,
    )

    inliers = len(
        source_inliers
    )

    inlier_ratio = (
        inliers
        / len(source_points)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    median = float(
        np.median(
            residuals
        )
    )

    mean = float(
        np.mean(
            residuals
        )
    )

    (
        coverage_n,
        coverage_d,
        coverage_ratio,
    ) = spatial_coverage(
        reference_inliers,
        matcher_valid,
    )

    affine_info = (
        decompose_affine(
            matrix
        )
    )

    sane = affine_sanity(
        affine_info
    )

    print("\n")
    print("=" * 78)
    print(
        "FINAL TILED LIGHTGLUE RESULT"
    )
    print("=" * 78)

    print(
        "Distinct candidates       :",
        len(source_points),
    )

    print(
        "RANSAC inliers            :",
        inliers,
    )

    print(
        "Inlier ratio              :",
        f"{inlier_ratio:.6f}",
    )

    print(
        "RANSAC self RMSE px       :",
        f"{rmse:.6f}",
    )

    print(
        "Median residual px        :",
        f"{median:.6f}",
    )

    print(
        "Mean residual px          :",
        f"{mean:.6f}",
    )

    print(
        "Spatial coverage          :",
        f"{coverage_n}/"
        f"{coverage_d} "
        f"({coverage_ratio:.4f})",
    )

    print(
        "Runtime s                 :",
        f"{runtime:.3f}",
    )

    print(
        "Transform sane            :",
        sane,
    )

    print(
        "\nAffine:"
    )

    print(
        matrix
    )

    print(
        "\nTranslation px:",
        f"{affine_info['translation_magnitude_px']:.6f}",
    )

    print(
        "Rotation deg:",
        f"{affine_info['rotation_deg']:.6f}",
    )

    print(
        "Scale X:",
        f"{affine_info['scale_x']:.8f}",
    )

    print(
        "Scale Y:",
        f"{affine_info['scale_y']:.8f}",
    )

    print(
        "Determinant:",
        f"{affine_info['determinant']:.8f}",
    )

    print(
        "Axis dot:",
        f"{affine_info['axis_dot']:.8f}",
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        OUT_DIR
        / "pair006_tiled_lightglue_inliers.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "source_x",
                "source_y",
                "reference_x",
                "reference_y",
                "ransac_residual_px",
            ]
        )

        for (
            source_point,
            reference_point,
            residual,
        ) in zip(
            source_inliers,
            reference_inliers,
            residuals,
        ):

            writer.writerow(
                [
                    float(
                        source_point[0]
                    ),

                    float(
                        source_point[1]
                    ),

                    float(
                        reference_point[0]
                    ),

                    float(
                        reference_point[1]
                    ),

                    float(
                        residual
                    ),
                ]
            )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    metrics = {
        "pair_id":
            "pair_006",

        "method":
            "prior_guided_tiled_lightglue",

        "representation":
            "gradient_magnitude",

        "tile_size_px":
            TILE_SIZE,

        "tile_stride_px":
            TILE_STRIDE,

        "search_margin_px":
            SEARCH_MARGIN,

        "prior_gate_px":
            PRIOR_GATE_PX,

        "tiles_run":
            int(
                tiles_run
            ),

        "raw_matches":
            int(
                raw_total
            ),

        "mask_filtered_matches":
            int(
                mask_total
            ),

        "prior_filtered_matches":
            int(
                prior_total
            ),

        "distinct_candidates":
            int(
                len(
                    source_points
                )
            ),

        "ransac_inliers":
            int(
                inliers
            ),

        "ransac_inlier_ratio":
            float(
                inlier_ratio
            ),

        "ransac_reprojection_rmse_px":
            rmse,

        "median_residual_px":
            median,

        "mean_residual_px":
            mean,

        "coverage_cells":
            int(
                coverage_n
            ),

        "valid_coverage_cells":
            int(
                coverage_d
            ),

        "coverage_ratio":
            float(
                coverage_ratio
            ),

        "affine":
            matrix.tolist(),

        "affine_parameters":
            affine_info,

        "transform_sane":
            sane,

        "runtime_s":
            float(
                runtime
            ),

        "note": (
            "RANSAC reprojection RMSE "
            "is internal self-consistency "
            "only and is not independent "
            "real-pair ground-truth accuracy."
        ),
    }

    json_path = (
        OUT_DIR
        / "pair006_tiled_lightglue_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    visualization_path = (
        OUT_DIR
        / "pair006_tiled_lightglue_matches.png"
    )

    save_matches(
        source_gradient,
        reference_gradient,
        source_inliers,
        reference_inliers,
        visualization_path,
    )

    print("\n")
    print("=" * 78)
    print(
        "OUTPUTS"
    )
    print("=" * 78)

    print(
        json_path
    )

    print(
        csv_path
    )

    print(
        visualization_path
    )

    print(
        "\nPAIR 004 TILED "
        "LIGHTGLUE COMPLETE."
    )


if __name__ == "__main__":
    main()

