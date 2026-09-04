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
import kornia.feature as KF


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

SOURCE_PATH = PAIR_DIR / "lro_nac_source.tif"
REFERENCE_PATH = PAIR_DIR / "tmc2_reference.tif"
COMMON_MASK_PATH = PAIR_DIR / "common_valid_mask.tif"

GLOBAL_RESULTS_PATH = (
    ROOT
    / "results"
    / "pair_004"
    / "global_benchmark"
    / "pair004_global_benchmark.json"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_004"
    / "tiled_loftr"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PIXEL_SIZE_M = 5.0

# Native-pixel tiles
TILE_SIZE = 768
TILE_STRIDE = 512

# Search around global affine prediction
SEARCH_MARGIN = 96

# Skip nearly empty tiles
MIN_VALID_PIXELS = 8000

# Accept tiled correspondence only when reasonably near
# the global affine prediction.
PRIOR_GATE_PX = 40.0

RANSAC_THRESHOLD_PX = 3.0

MASK_EROSION_PIXELS = 5

GRID_ROWS = 8
GRID_COLS = 8


# ============================================================
# INPUT
# ============================================================


def read_band(path):
    with rasterio.open(path) as ds:
        return (
            ds.read(1),
            ds.read_masks(1) > 0,
            ds.nodata,
        )


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

    temp = (
        data.astype(np.float32)
        - float(low)
    ) / float(high - low)

    temp = np.clip(
        temp,
        0.0,
        1.0,
    )

    out[mask] = np.round(
        temp[mask] * 255.0
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
# GLOBAL AFFINE
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
            result["method"] == "loftr"
            and
            result["representation"]
            == "gradient"
        ):
            matrix = np.asarray(
                result["affine"],
                dtype=np.float64,
            )

            print(
                "\nLoaded global "
                "LoFTR-gradient affine:"
            )

            print(matrix)

            return matrix

    raise RuntimeError(
        "LoFTR gradient result "
        "not found."
    )


def affine_points(matrix, points):
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

    matcher_pre_erosion = (
        science_valid
        & (reference > 0)
    )

    matcher = (
        ndimage.binary_erosion(
            matcher_pre_erosion,
            structure=np.ones(
                (3, 3),
                dtype=bool,
            ),
            iterations=(
                MASK_EROSION_PIXELS
            ),
            border_value=0,
        )
    )

    return (
        science_valid,
        matcher_pre_erosion,
        matcher,
    )


# ============================================================
# LOFTR UTILITIES
# ============================================================


def pad_to_multiple_of_8(image):
    h, w = image.shape

    ph = (
        int(
            math.ceil(h / 8.0)
        ) * 8
    )

    pw = (
        int(
            math.ceil(w / 8.0)
        ) * 8
    )

    padded = np.zeros(
        (ph, pw),
        dtype=image.dtype,
    )

    padded[:h, :w] = image

    return padded, h, w


def to_tensor(image, device):
    return torch.from_numpy(
        image.astype(np.float32)
        / 255.0
    )[None, None].to(device)


def point_mask_valid(points, mask):
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

    ids = np.where(inside)[0]

    result[ids] = mask[
        y[ids],
        x[ids],
    ]

    return result


# ============================================================
# TILE GENERATION
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
        starts.append(final_start)

    return sorted(set(starts))


def predicted_reference_bbox(
    affine,
    x0,
    y0,
    x1,
    y1,
    width,
    height,
):
    corners = np.array(
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

    rx0 = max(0, rx0)
    ry0 = max(0, ry0)

    rx1 = min(width, rx1)
    ry1 = min(height, ry1)

    return (
        rx0,
        ry0,
        rx1,
        ry1,
    )


# ============================================================
# DEDUPLICATION
# ============================================================


def deduplicate_matches(
    source_points,
    reference_points,
    confidence,
):
    """
    Deduplicate overlapping-tile matches.

    Quantization at 0.5 native pixels keeps genuinely
    distinct correspondences while removing repeated matches
    from adjacent tiles.
    """

    if len(source_points) == 0:
        return (
            source_points,
            reference_points,
            confidence,
        )

    best = {}

    for i, (
        src,
        ref,
        conf,
    ) in enumerate(
        zip(
            source_points,
            reference_points,
            confidence,
        )
    ):

        key = (
            round(
                float(src[0]) * 2
            ),
            round(
                float(src[1]) * 2
            ),
            round(
                float(ref[0]) * 2
            ),
            round(
                float(ref[1]) * 2
            ),
        )

        if (
            key not in best
            or conf > best[key][0]
        ):
            best[key] = (
                float(conf),
                i,
            )

    indices = np.array(
        sorted(
            item[1]
            for item in best.values()
        ),
        dtype=np.int64,
    )

    return (
        source_points[indices],
        reference_points[indices],
        confidence[indices],
    )


# ============================================================
# METRICS
# ============================================================


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

    determinant = float(
        np.linalg.det(A)
    )

    rotation = math.degrees(
        math.atan2(
            A[1, 0],
            A[0, 0],
        )
    )

    translation_x = float(
        matrix[0, 2]
    )

    translation_y = float(
        matrix[1, 2]
    )

    translation = math.hypot(
        translation_x,
        translation_y,
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
        "rotation_deg": float(
            rotation
        ),
        "translation_x_px":
            translation_x,
        "translation_y_px":
            translation_y,
        "translation_magnitude_px":
            float(translation),
        "determinant": determinant,
        "axis_dot": axis_dot,
    }


def spatial_coverage(
    points,
    valid_mask,
):
    h, w = valid_mask.shape

    valid_cells = set()

    for gy in range(GRID_ROWS):
        y0 = int(
            round(
                gy * h / GRID_ROWS
            )
        )

        y1 = int(
            round(
                (gy + 1)
                * h / GRID_ROWS
            )
        )

        for gx in range(
            GRID_COLS
        ):
            x0 = int(
                round(
                    gx
                    * w
                    / GRID_COLS
                )
            )

            x1 = int(
                round(
                    (gx + 1)
                    * w
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
                    (gy, gx)
                )

    occupied = set()

    for x, y in points:

        gx = min(
            GRID_COLS - 1,
            max(
                0,
                int(
                    x
                    / w
                    * GRID_COLS
                ),
            ),
        )

        gy = min(
            GRID_ROWS - 1,
            max(
                0,
                int(
                    y
                    / h
                    * GRID_ROWS
                ),
            ),
        )

        cell = (gy, gx)

        if cell in valid_cells:
            occupied.add(cell)

    return (
        len(occupied),
        len(valid_cells),
        (
            len(occupied)
            / len(valid_cells)
            if valid_cells
            else 0.0
        ),
    )


# ============================================================
# VISUALIZATION
# ============================================================


def save_matches(
    source,
    reference,
    src_pts,
    ref_pts,
    path,
):
    h, w = source.shape

    display_max = 1800

    scale = min(
        1.0,
        display_max / max(h, w),
    )

    dw = int(
        round(w * scale)
    )

    dh = int(
        round(h * scale)
    )

    src_small = cv2.resize(
        source,
        (dw, dh),
        interpolation=cv2.INTER_AREA,
    )

    ref_small = cv2.resize(
        reference,
        (dw, dh),
        interpolation=cv2.INTER_AREA,
    )

    src_rgb = cv2.cvtColor(
        src_small,
        cv2.COLOR_GRAY2BGR,
    )

    ref_rgb = cv2.cvtColor(
        ref_small,
        cv2.COLOR_GRAY2BGR,
    )

    canvas = np.concatenate(
        [
            src_rgb,
            ref_rgb,
        ],
        axis=1,
    )

    # Display at most 500 lines.
    if len(src_pts) > 500:
        ids = np.linspace(
            0,
            len(src_pts) - 1,
            500,
        ).astype(int)

        src_pts = src_pts[ids]
        ref_pts = ref_pts[ids]

    for p0, p1 in zip(
        src_pts,
        ref_pts,
    ):
        x0 = int(
            round(
                p0[0] * scale
            )
        )

        y0 = int(
            round(
                p0[1] * scale
            )
        )

        x1 = int(
            round(
                p1[0] * scale
            )
        ) + dw

        y1 = int(
            round(
                p1[1] * scale
            )
        )

        cv2.circle(
            canvas,
            (x0, y0),
            3,
            (0, 255, 0),
            -1,
        )

        cv2.circle(
            canvas,
            (x1, y1),
            3,
            (0, 255, 0),
            -1,
        )

        cv2.line(
            canvas,
            (x0, y0),
            (x1, y1),
            (0, 255, 255),
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
    print("=" * 74)
    print(
        "PAIR 004 — PRIOR-GUIDED "
        "TILED LOFTR GRADIENT"
    )
    print("=" * 74)

    source, source_mask, _ = (
        read_band(SOURCE_PATH)
    )

    reference, reference_mask, _ = (
        read_band(REFERENCE_PATH)
    )

    common_data, _, _ = (
        read_band(COMMON_MASK_PATH)
    )

    common = common_data > 0

    if source.shape != reference.shape:
        raise RuntimeError(
            "Canonical raster shapes differ."
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
        matcher_mask,
    ) = build_matcher_mask(
        source,
        source_mask,
        reference,
        reference_mask,
        common,
    )

    print(
        "Science-valid pixels       :",
        f"{int(science_valid.sum()):,}",
    )

    print(
        "Nonzero matcher pixels     :",
        f"{int(matcher_pre.sum()):,}",
    )

    print(
        "Eroded matcher pixels      :",
        f"{int(matcher_mask.sum()):,}",
    )

    source_i = robust_uint8(
        source,
        matcher_mask,
    )

    reference_i = robust_uint8(
        reference,
        matcher_mask,
    )

    source_g = gradient_uint8(
        source_i,
        matcher_mask,
    )

    reference_g = gradient_uint8(
        reference_i,
        matcher_mask,
    )

    source_g[~matcher_mask] = 0
    reference_g[~matcher_mask] = 0

    global_affine = (
        load_global_affine()
    )

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

    matcher = KF.LoFTR(
        pretrained="outdoor",
    ).eval().to(device)

    xs = tile_starts(w)
    ys = tile_starts(h)

    total_tiles = len(xs) * len(ys)

    print(
        "\nCandidate tiles:",
        total_tiles,
    )

    all_src = []
    all_ref = []
    all_conf = []

    tiles_run = 0
    raw_total = 0
    mask_kept_total = 0
    prior_kept_total = 0

    start_all = time.perf_counter()

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
                matcher_mask[
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

            src_crop = source_g[
                y0:y1,
                x0:x1,
            ]

            ref_crop = reference_g[
                ry0:ry1,
                rx0:rx1,
            ]

            src_pad, sh, sw = (
                pad_to_multiple_of_8(
                    src_crop
                )
            )

            ref_pad, rh, rw = (
                pad_to_multiple_of_8(
                    ref_crop
                )
            )

            image0 = to_tensor(
                src_pad,
                device,
            )

            image1 = to_tensor(
                ref_pad,
                device,
            )

            with torch.inference_mode():

                pred = matcher(
                    {
                        "image0": image0,
                        "image1": image1,
                    }
                )

            src_local = (
                pred["keypoints0"]
                .detach()
                .cpu()
                .numpy()
            )

            ref_local = (
                pred["keypoints1"]
                .detach()
                .cpu()
                .numpy()
            )

            if "confidence" in pred:
                confidence = (
                    pred["confidence"]
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                confidence = np.ones(
                    len(src_local),
                    dtype=np.float32,
                )

            raw_count = len(src_local)

            raw_total += raw_count

            if raw_count == 0:
                continue

            # Remove points in padded area.
            crop_keep = (
                (src_local[:, 0] >= 0)
                & (src_local[:, 0] < sw)
                & (src_local[:, 1] >= 0)
                & (src_local[:, 1] < sh)
                & (ref_local[:, 0] >= 0)
                & (ref_local[:, 0] < rw)
                & (ref_local[:, 1] >= 0)
                & (ref_local[:, 1] < rh)
            )

            src_local = src_local[
                crop_keep
            ]

            ref_local = ref_local[
                crop_keep
            ]

            confidence = confidence[
                crop_keep
            ]

            if len(src_local) == 0:
                continue

            src_native = (
                src_local
                + np.array(
                    [x0, y0],
                    dtype=np.float32,
                )
            )

            ref_native = (
                ref_local
                + np.array(
                    [rx0, ry0],
                    dtype=np.float32,
                )
            )

            mask_keep = (
                point_mask_valid(
                    src_native,
                    matcher_mask,
                )
                &
                point_mask_valid(
                    ref_native,
                    matcher_mask,
                )
            )

            src_native = src_native[
                mask_keep
            ]

            ref_native = ref_native[
                mask_keep
            ]

            confidence = confidence[
                mask_keep
            ]

            mask_kept_total += len(
                src_native
            )

            if len(src_native) == 0:
                continue

            predicted_ref = (
                affine_points(
                    global_affine,
                    src_native,
                )
            )

            prior_error = np.linalg.norm(
                predicted_ref
                - ref_native,
                axis=1,
            )

            prior_keep = (
                prior_error
                <= PRIOR_GATE_PX
            )

            src_native = src_native[
                prior_keep
            ]

            ref_native = ref_native[
                prior_keep
            ]

            confidence = confidence[
                prior_keep
            ]

            prior_kept_total += len(
                src_native
            )

            if len(src_native) > 0:

                all_src.append(
                    src_native.astype(
                        np.float32
                    )
                )

                all_ref.append(
                    ref_native.astype(
                        np.float32
                    )
                )

                all_conf.append(
                    confidence.astype(
                        np.float32
                    )
                )

            tiles_run += 1

            print(
                f"Tile {tile_number:02d}/"
                f"{total_tiles:02d} "
                f"src=({x0},{y0}) "
                f"valid={valid_pixels:,} "
                f"raw={raw_count} "
                f"kept={len(src_native)}"
            )

    runtime = (
        time.perf_counter()
        - start_all
    )

    if len(all_src) == 0:
        raise RuntimeError(
            "No tiled correspondences "
            "survived."
        )

    src = np.concatenate(
        all_src,
        axis=0,
    )

    ref = np.concatenate(
        all_ref,
        axis=0,
    )

    confidence = np.concatenate(
        all_conf,
        axis=0,
    )

    before_dedup = len(src)

    (
        src,
        ref,
        confidence,
    ) = deduplicate_matches(
        src,
        ref,
        confidence,
    )

    print("\n" + "=" * 74)
    print("TILED MATCH COLLECTION")
    print("=" * 74)

    print(
        "Tiles actually run         :",
        tiles_run,
    )

    print(
        "Raw LoFTR correspondences  :",
        raw_total,
    )

    print(
        "After matcher-mask filter  :",
        mask_kept_total,
    )

    print(
        "After global-prior gate    :",
        prior_kept_total,
    )

    print(
        "Before deduplication       :",
        before_dedup,
    )

    print(
        "Distinct correspondences   :",
        len(src),
    )

    # ========================================================
    # Global RANSAC on all tiled correspondences
    # ========================================================

    if len(src) < 4:
        raise RuntimeError(
            "Too few correspondences "
            "for RANSAC."
        )

    matrix, inlier_mask = (
        cv2.estimateAffine2D(
            src,
            ref,
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
            "RANSAC affine estimation "
            "failed."
        )

    inlier_mask = (
        inlier_mask.ravel() > 0
    )

    src_in = src[inlier_mask]
    ref_in = ref[inlier_mask]

    predicted = affine_points(
        matrix,
        src_in,
    )

    residuals = np.linalg.norm(
        predicted - ref_in,
        axis=1,
    )

    inliers = len(src_in)

    inlier_ratio = (
        inliers / len(src)
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
        ref_in,
        matcher_mask,
    )

    affine_info = decompose_affine(
        matrix
    )

    print("\n" + "=" * 74)
    print("FINAL TILED LOFTR RESULT")
    print("=" * 74)

    print(
        "Distinct candidates        :",
        len(src),
    )

    print(
        "RANSAC inliers             :",
        inliers,
    )

    print(
        "Inlier ratio               :",
        f"{inlier_ratio:.6f}",
    )

    print(
        "RANSAC reprojection RMSE px:",
        f"{rmse:.6f}",
    )

    print(
        "Median residual px         :",
        f"{median:.6f}",
    )

    print(
        "Mean residual px           :",
        f"{mean:.6f}",
    )

    print(
        "Approx RMSE metres         :",
        f"{rmse * PIXEL_SIZE_M:.3f}",
    )

    print(
        "Spatial coverage           :",
        f"{coverage_n}/"
        f"{coverage_d} "
        f"({coverage_ratio:.4f})",
    )

    print(
        "Runtime s                  :",
        f"{runtime:.3f}",
    )

    print("\nAffine:")
    print(matrix)

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

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        OUT_DIR
        / "pair004_tiled_loftr_inliers.csv"
    )

    with open(
        csv_path,
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
                "residual_px",
                "residual_m",
            ]
        )

        for (
            s,
            r,
            residual,
        ) in zip(
            src_in,
            ref_in,
            residuals,
        ):
            writer.writerow(
                [
                    float(s[0]),
                    float(s[1]),
                    float(r[0]),
                    float(r[1]),
                    float(residual),
                    float(
                        residual
                        * PIXEL_SIZE_M
                    ),
                ]
            )

    # ========================================================
    # JSON
    # ========================================================

    result = {
        "pair_id": "pair_004",

        "method":
            "prior_guided_tiled_loftr",

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
            int(tiles_run),

        "raw_correspondences":
            int(raw_total),

        "mask_filtered_correspondences":
            int(mask_kept_total),

        "prior_filtered_correspondences":
            int(prior_kept_total),

        "distinct_candidates":
            int(len(src)),

        "ransac_inliers":
            int(inliers),

        "inlier_ratio":
            float(inlier_ratio),

        "ransac_reprojection_rmse_px":
            rmse,

        "ransac_reprojection_rmse_m":
            rmse * PIXEL_SIZE_M,

        "median_residual_px":
            median,

        "mean_residual_px":
            mean,

        "coverage_cells":
            int(coverage_n),

        "valid_coverage_cells":
            int(coverage_d),

        "coverage_ratio":
            float(coverage_ratio),

        "affine":
            matrix.tolist(),

        "affine_parameters":
            affine_info,

        "runtime_s":
            float(runtime),

        "note": (
            "RANSAC reprojection residual "
            "is internal self-consistency, "
            "not independent ground-truth "
            "registration accuracy."
        ),
    }

    json_path = (
        OUT_DIR
        / "pair004_tiled_loftr_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    save_matches(
        source_g,
        reference_g,
        src_in,
        ref_in,
        OUT_DIR
        / "pair004_tiled_loftr_matches.png",
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / "pair004_matcher_mask.png"
        ),
        (
            matcher_mask.astype(
                np.uint8
            )
            * 255
        ),
    )

    print("\n" + "=" * 74)
    print("OUTPUTS")
    print("=" * 74)

    print(json_path)
    print(csv_path)

    print(
        OUT_DIR
        / "pair004_tiled_loftr_matches.png"
    )

    print("\nIMPORTANT:")
    print(
        "The residual above is "
        "RANSAC/self-consistency only."
    )

    print(
        "It is NOT independent "
        "ground-truth registration accuracy."
    )

    print(
        "\nPAIR 004 TILED LOFTR COMPLETE."
    )


if __name__ == "__main__":
    main()