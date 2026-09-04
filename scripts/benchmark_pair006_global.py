from pathlib import Path
import csv
import json
import math
import time

import cv2
import numpy as np
import rasterio
from scipy import ndimage


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

SOURCE_PATH = PAIR_DIR / "source_lro_nac.tif"
REFERENCE_PATH = PAIR_DIR / "reference_tmc2.tif"
COMMON_MASK_PATH = PAIR_DIR / "matcher_mask.tif"

OUT_DIR = ROOT / "results" / "pair_006" / "global_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_NAME = "LRO NAC"
REFERENCE_NAME = "TMC-2"

# Native canonical grid = 5 m/pixel
PIXEL_SIZE_M = 5.0

# ------------------------------------------------------------
# Benchmark scales
# ------------------------------------------------------------

SIFT_MAX_DIM = 3000
LIGHTGLUE_MAX_DIM = 2048
LOFTR_MAX_DIM = 1600

# RANSAC residual measured on NATIVE 5 m canonical grid
RANSAC_THRESHOLD_PX = 3.0

# Remove features close to sensor/footprint boundaries
MASK_EROSION_PIXELS = 5

# Spatial distribution metric
GRID_ROWS = 8
GRID_COLS = 8


# ============================================================
# OPTIONAL DEEP MATCHER IMPORTS
# ============================================================

try:
    import torch
except ImportError:
    torch = None


LIGHTGLUE_AVAILABLE = False
LOFTR_AVAILABLE = False


try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    LIGHTGLUE_AVAILABLE = True
except Exception as e:
    print("LightGlue import warning:")
    print(e)


try:
    import kornia.feature as KF

    LOFTR_AVAILABLE = True
except Exception as e:
    print("LoFTR import warning:")
    print(e)


# ============================================================
# BASIC UTILITIES
# ============================================================


def read_single_band(path):
    with rasterio.open(path) as ds:
        data = ds.read(1)
        mask = ds.read_masks(1) > 0

        return (
            data,
            mask,
            ds.transform,
            ds.crs,
            ds.nodata,
        )


def robust_uint8(data, mask):
    """
    Convert a science raster to uint8 using robust percentile
    stretching over matcher-valid pixels only.
    """

    out = np.zeros(data.shape, dtype=np.uint8)

    values = data[mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return out

    p2, p98 = np.percentile(values, [2.0, 98.0])

    if p98 <= p2:
        p98 = p2 + 1.0

    temp = (
        data.astype(np.float32) - float(p2)
    ) / float(p98 - p2)

    temp = np.clip(temp, 0.0, 1.0)

    out[mask] = np.round(
        temp[mask] * 255.0
    ).astype(np.uint8)

    return out


def gradient_magnitude_uint8(image, mask):
    """
    Sobel gradient magnitude.

    Gradient magnitude is insensitive to gradient SIGN, which is
    useful when crater illumination direction reverses.
    """

    img = image.astype(np.float32) / 255.0

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

    mag = np.sqrt(gx * gx + gy * gy)

    out = np.zeros(image.shape, dtype=np.uint8)

    values = mag[mask]

    if values.size == 0:
        return out

    high = np.percentile(values, 99.0)

    if high <= 0:
        return out

    mag = np.clip(
        mag / high,
        0.0,
        1.0,
    )

    out[mask] = np.round(
        mag[mask] * 255.0
    ).astype(np.uint8)

    return out


def resize_for_max_dim(image, max_dim, interpolation):
    h, w = image.shape

    current_max = max(h, w)

    if current_max <= max_dim:
        scale = 1.0
        return image.copy(), scale

    scale = max_dim / current_max

    new_w = max(
        1,
        int(round(w * scale)),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation,
    )

    return resized, scale


def points_to_native(points, scale):
    points = np.asarray(points, dtype=np.float32)

    if points.size == 0:
        return points.reshape(0, 2)

    return points / float(scale)


def points_inside_mask(points, mask):
    """
    Return bool array selecting points whose rounded pixel
    location lies inside mask.
    """

    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    h, w = mask.shape

    x = np.rint(points[:, 0]).astype(np.int64)
    y = np.rint(points[:, 1]).astype(np.int64)

    inside = (
        (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )

    result = np.zeros(len(points), dtype=bool)

    indices = np.where(inside)[0]

    result[indices] = mask[
        y[indices],
        x[indices],
    ]

    return result


# ============================================================
# MASK PREPARATION
# ============================================================


def build_matcher_mask(
    source,
    source_mask,
    reference,
    reference_mask,
    common_mask,
):
    """
    Science-valid mask and matcher-valid mask are deliberately
    different.

    We DO NOT call TMC zero DN "nodata".

    But completely black TMC shadow interiors are excluded from
    feature extraction because they contain essentially no local
    texture.
    """

    valid = common_mask.copy()

    valid &= source_mask
    valid &= reference_mask

    valid &= np.isfinite(source)
    valid &= np.isfinite(reference)

    # Matcher only:
    # preserve zero DN in science product, exclude from feature
    # extraction.
    valid &= reference > 0

    structure = np.ones((3, 3), dtype=bool)

    valid_eroded = ndimage.binary_erosion(
        valid,
        structure=structure,
        iterations=MASK_EROSION_PIXELS,
        border_value=0,
    )

    return valid, valid_eroded


# ============================================================
# SIFT
# ============================================================


def run_sift(source_img, reference_img, matcher_mask):
    start = time.perf_counter()

    src_small, scale = resize_for_max_dim(
        source_img,
        SIFT_MAX_DIM,
        cv2.INTER_AREA,
    )

    ref_small, scale_ref = resize_for_max_dim(
        reference_img,
        SIFT_MAX_DIM,
        cv2.INTER_AREA,
    )

    if abs(scale - scale_ref) > 1e-6:
        raise RuntimeError(
            "Source/reference resize scales differ."
        )

    small_mask = cv2.resize(
        matcher_mask.astype(np.uint8),
        (
            src_small.shape[1],
            src_small.shape[0],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    small_mask = (
        small_mask > 0
    ).astype(np.uint8) * 255

    sift = cv2.SIFT_create(
        nfeatures=20000,
        contrastThreshold=0.01,
        edgeThreshold=10,
        sigma=1.6,
    )

    kp0, des0 = sift.detectAndCompute(
        src_small,
        small_mask,
    )

    kp1, des1 = sift.detectAndCompute(
        ref_small,
        small_mask,
    )

    if des0 is None or des1 is None:
        return {
            "source_points": np.empty((0, 2)),
            "reference_points": np.empty((0, 2)),
            "raw_candidates": 0,
            "runtime_s": time.perf_counter() - start,
        }

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    knn = matcher.knnMatch(
        des0,
        des1,
        k=2,
    )

    good = []

    for pair in knn:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.array(
        [
            kp0[m.queryIdx].pt
            for m in good
        ],
        dtype=np.float32,
    )

    ref_pts = np.array(
        [
            kp1[m.trainIdx].pt
            for m in good
        ],
        dtype=np.float32,
    )

    src_pts = points_to_native(
        src_pts,
        scale,
    )

    ref_pts = points_to_native(
        ref_pts,
        scale,
    )

    runtime = time.perf_counter() - start

    return {
        "source_points": src_pts,
        "reference_points": ref_pts,
        "raw_candidates": len(good),
        "runtime_s": runtime,
        "source_keypoints": len(kp0),
        "reference_keypoints": len(kp1),
        "scale": scale,
    }


# ============================================================
# LIGHTGLUE
# ============================================================


def tensor_from_gray(image, device):
    """
    Returns C x H x W tensor, same convention as LightGlue's
    load_image().
    """

    tensor = torch.from_numpy(
        image.astype(np.float32) / 255.0
    )

    return tensor.unsqueeze(0).to(device)


def run_lightglue(
    source_img,
    reference_img,
):
    if not LIGHTGLUE_AVAILABLE:
        raise RuntimeError(
            "LightGlue package is not available."
        )

    if torch is None:
        raise RuntimeError(
            "PyTorch not available."
        )

    start = time.perf_counter()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    src_small, scale = resize_for_max_dim(
        source_img,
        LIGHTGLUE_MAX_DIM,
        cv2.INTER_AREA,
    )

    ref_small, scale_ref = resize_for_max_dim(
        reference_img,
        LIGHTGLUE_MAX_DIM,
        cv2.INTER_AREA,
    )

    if abs(scale - scale_ref) > 1e-6:
        raise RuntimeError(
            "Source/reference resize scales differ."
        )

    extractor = SuperPoint(
        max_num_keypoints=8192,
    ).eval().to(device)

    matcher = LightGlue(
        features="superpoint",
    ).eval().to(device)

    image0 = tensor_from_gray(
        src_small,
        device,
    )

    image1 = tensor_from_gray(
        ref_small,
        device,
    )

    with torch.inference_mode():

        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)

        pred = matcher(
            {
                "image0": feats0,
                "image1": feats1,
            }
        )

    feats0 = rbd(feats0)
    feats1 = rbd(feats1)
    pred = rbd(pred)

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

    # Modern LightGlue output
    if "matches" in pred:

        matches = (
            pred["matches"]
            .detach()
            .cpu()
            .numpy()
        )

        if matches.size == 0:
            src_pts = np.empty(
                (0, 2),
                dtype=np.float32,
            )

            ref_pts = np.empty(
                (0, 2),
                dtype=np.float32,
            )

        else:
            src_pts = keypoints0[
                matches[:, 0]
            ]

            ref_pts = keypoints1[
                matches[:, 1]
            ]

    # Compatibility fallback
    elif "matches0" in pred:

        matches0 = (
            pred["matches0"]
            .detach()
            .cpu()
            .numpy()
        )

        idx0 = np.where(
            matches0 >= 0
        )[0]

        idx1 = matches0[idx0]

        src_pts = keypoints0[idx0]
        ref_pts = keypoints1[idx1]

    else:
        raise RuntimeError(
            "Unrecognized LightGlue output."
        )

    src_pts = points_to_native(
        src_pts,
        scale,
    )

    ref_pts = points_to_native(
        ref_pts,
        scale,
    )

    runtime = time.perf_counter() - start

    del extractor
    del matcher

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "source_points": src_pts,
        "reference_points": ref_pts,
        "raw_candidates": len(src_pts),
        "runtime_s": runtime,
        "source_keypoints": len(keypoints0),
        "reference_keypoints": len(keypoints1),
        "scale": scale,
    }


# ============================================================
# LOFTR
# ============================================================


def run_loftr(
    source_img,
    reference_img,
):
    if not LOFTR_AVAILABLE:
        raise RuntimeError(
            "Kornia LoFTR is not available."
        )

    if torch is None:
        raise RuntimeError(
            "PyTorch not available."
        )

    start = time.perf_counter()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    src_small, scale = resize_for_max_dim(
        source_img,
        LOFTR_MAX_DIM,
        cv2.INTER_AREA,
    )

    ref_small, scale_ref = resize_for_max_dim(
        reference_img,
        LOFTR_MAX_DIM,
        cv2.INTER_AREA,
    )

    if abs(scale - scale_ref) > 1e-6:
        raise RuntimeError(
            "Source/reference resize scales differ."
        )

    image0 = torch.from_numpy(
        src_small.astype(np.float32) / 255.0
    )[None, None].to(device)

    image1 = torch.from_numpy(
        ref_small.astype(np.float32) / 255.0
    )[None, None].to(device)

    matcher = KF.LoFTR(
        pretrained="outdoor",
    ).eval().to(device)

    with torch.inference_mode():

        pred = matcher(
            {
                "image0": image0,
                "image1": image1,
            }
        )

    src_pts = (
        pred["keypoints0"]
        .detach()
        .cpu()
        .numpy()
    )

    ref_pts = (
        pred["keypoints1"]
        .detach()
        .cpu()
        .numpy()
    )

    src_pts = points_to_native(
        src_pts,
        scale,
    )

    ref_pts = points_to_native(
        ref_pts,
        scale,
    )

    runtime = time.perf_counter() - start

    del matcher

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "source_points": src_pts,
        "reference_points": ref_pts,
        "raw_candidates": len(src_pts),
        "runtime_s": runtime,
        "scale": scale,
    }


# ============================================================
# RANSAC / METRICS
# ============================================================


def affine_predict(matrix, points):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    return (
        points @ matrix[:, :2].T
        + matrix[:, 2]
    )


def decompose_affine(matrix):
    a = matrix[:, :2]

    col0 = a[:, 0]
    col1 = a[:, 1]

    scale_x = float(
        np.linalg.norm(col0)
    )

    scale_y = float(
        np.linalg.norm(col1)
    )

    determinant = float(
        np.linalg.det(a)
    )

    if scale_x > 0 and scale_y > 0:

        axis_dot = float(
            np.dot(col0, col1)
            / (scale_x * scale_y)
        )

    else:
        axis_dot = float("nan")

    rotation_deg = float(
        math.degrees(
            math.atan2(
                a[1, 0],
                a[0, 0],
            )
        )
    )

    translation_x = float(
        matrix[0, 2]
    )

    translation_y = float(
        matrix[1, 2]
    )

    translation_mag = float(
        math.hypot(
            translation_x,
            translation_y,
        )
    )

    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "rotation_deg": rotation_deg,
        "translation_x_px": translation_x,
        "translation_y_px": translation_y,
        "translation_magnitude_px": translation_mag,
        "determinant": determinant,
        "axis_dot": axis_dot,
    }


def affine_sanity(info):
    """
    Pair 006 has already been map-projected onto the same 5 m grid.
    A physically plausible residual transform should therefore
    remain relatively close to identity.
    """

    return bool(
        info["determinant"] > 0
        and 0.90 <= info["scale_x"] <= 1.10
        and 0.90 <= info["scale_y"] <= 1.10
        and abs(info["rotation_deg"]) <= 5.0
        and info["translation_magnitude_px"] <= 100.0
        and abs(info["axis_dot"]) <= 0.15
    )


def spatial_coverage(
    points,
    valid_mask,
    rows=GRID_ROWS,
    cols=GRID_COLS,
):
    h, w = valid_mask.shape

    valid_cells = set()

    for gy in range(rows):
        y0 = int(
            round(gy * h / rows)
        )

        y1 = int(
            round((gy + 1) * h / rows)
        )

        for gx in range(cols):
            x0 = int(
                round(gx * w / cols)
            )

            x1 = int(
                round((gx + 1) * w / cols)
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

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            continue

        gx = min(
            cols - 1,
            max(
                0,
                int(x / w * cols),
            ),
        )

        gy = min(
            rows - 1,
            max(
                0,
                int(y / h * rows),
            ),
        )

        cell = (gy, gx)

        if cell in valid_cells:
            occupied.add(cell)

    denominator = len(valid_cells)
    numerator = len(occupied)

    ratio = (
        numerator / denominator
        if denominator > 0
        else 0.0
    )

    return numerator, denominator, ratio


def uniform_subset(
    source_points,
    reference_points,
    residuals,
    valid_mask,
):
    """
    Keep the lowest-residual inlier in each spatial grid cell.

    This is only a diagnostic uniform subset, not a replacement
    for the dense RANSAC inlier set.
    """

    h, w = valid_mask.shape

    best = {}

    for i, (ref, res) in enumerate(
        zip(reference_points, residuals)
    ):

        x, y = ref

        gx = min(
            GRID_COLS - 1,
            max(
                0,
                int(x / w * GRID_COLS),
            ),
        )

        gy = min(
            GRID_ROWS - 1,
            max(
                0,
                int(y / h * GRID_ROWS),
            ),
        )

        key = (gy, gx)

        if (
            key not in best
            or res < best[key][0]
        ):
            best[key] = (
                float(res),
                i,
            )

    selected = sorted(
        item[1]
        for item in best.values()
    )

    return np.asarray(
        selected,
        dtype=np.int64,
    )


def evaluate_matches(
    method,
    representation,
    match_result,
    matcher_mask,
):
    src = np.asarray(
        match_result["source_points"],
        dtype=np.float32,
    )

    ref = np.asarray(
        match_result["reference_points"],
        dtype=np.float32,
    )

    raw_candidates = int(
        match_result["raw_candidates"]
    )

    if len(src) != len(ref):
        raise RuntimeError(
            "Source/reference match count differs."
        )

    # --------------------------------------------------------
    # Remove matches outside the eroded feature-valid mask
    # --------------------------------------------------------

    src_inside = points_inside_mask(
        src,
        matcher_mask,
    )

    ref_inside = points_inside_mask(
        ref,
        matcher_mask,
    )

    keep = src_inside & ref_inside

    src = src[keep]
    ref = ref[keep]

    filtered_candidates = len(src)

    result = {
        "method": method,
        "representation": representation,
        "raw_candidates": raw_candidates,
        "filtered_candidates": filtered_candidates,
        "runtime_s": float(
            match_result["runtime_s"]
        ),
    }

    if filtered_candidates < 4:

        result.update(
            {
                "inliers": 0,
                "inlier_ratio": 0.0,
                "ransac_reprojection_rmse_px": None,
                "median_residual_px": None,
                "mean_residual_px": None,
                "coverage_cells": 0,
                "valid_coverage_cells": 0,
                "coverage_ratio": 0.0,
                "uniform_inliers": 0,
                "affine": None,
                "sanity": False,
            }
        )

        return result, None

    # --------------------------------------------------------
    # Full affine RANSAC
    # --------------------------------------------------------

    matrix, inlier_mask = cv2.estimateAffine2D(
        src,
        ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=10000,
        confidence=0.999,
        refineIters=50,
    )

    if matrix is None or inlier_mask is None:

        result.update(
            {
                "inliers": 0,
                "inlier_ratio": 0.0,
                "ransac_reprojection_rmse_px": None,
                "median_residual_px": None,
                "mean_residual_px": None,
                "coverage_cells": 0,
                "valid_coverage_cells": 0,
                "coverage_ratio": 0.0,
                "uniform_inliers": 0,
                "affine": None,
                "sanity": False,
            }
        )

        return result, None

    inlier_mask = (
        inlier_mask.ravel() > 0
    )

    src_in = src[inlier_mask]
    ref_in = ref[inlier_mask]

    predicted = affine_predict(
        matrix,
        src_in,
    )

    residuals = np.linalg.norm(
        predicted - ref_in,
        axis=1,
    )

    inliers = len(src_in)

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    median = float(
        np.median(residuals)
    )

    mean = float(
        np.mean(residuals)
    )

    coverage_n, coverage_d, coverage_ratio = (
        spatial_coverage(
            ref_in,
            matcher_mask,
        )
    )

    uniform_indices = uniform_subset(
        src_in,
        ref_in,
        residuals,
        matcher_mask,
    )

    affine_info = decompose_affine(
        matrix
    )

    sane = affine_sanity(
        affine_info
    )

    result.update(
        {
            "inliers": int(inliers),
            "inlier_ratio": float(
                inliers
                / filtered_candidates
            ),
            "ransac_reprojection_rmse_px": rmse,
            "median_residual_px": median,
            "mean_residual_px": mean,
            "coverage_cells": int(
                coverage_n
            ),
            "valid_coverage_cells": int(
                coverage_d
            ),
            "coverage_ratio": float(
                coverage_ratio
            ),
            "uniform_inliers": int(
                len(uniform_indices)
            ),
            "affine": matrix.tolist(),
            "affine_parameters": affine_info,
            "sanity": sane,
        }
    )

    detail = {
        "source_points": src,
        "reference_points": ref,
        "inlier_mask": inlier_mask,
        "source_inliers": src_in,
        "reference_inliers": ref_in,
        "residuals": residuals,
        "uniform_indices": uniform_indices,
        "matrix": matrix,
    }

    return result, detail


# ============================================================
# VISUALIZATION
# ============================================================


def save_match_visualization(
    source_img,
    reference_img,
    detail,
    path,
):
    if detail is None:
        return

    uniform_indices = detail[
        "uniform_indices"
    ]

    if len(uniform_indices) == 0:
        return

    src_pts = detail[
        "source_inliers"
    ][uniform_indices]

    ref_pts = detail[
        "reference_inliers"
    ][uniform_indices]

    # Display at manageable resolution
    max_dim = 1800

    src_small, scale = resize_for_max_dim(
        source_img,
        max_dim,
        cv2.INTER_AREA,
    )

    ref_small, scale_ref = resize_for_max_dim(
        reference_img,
        max_dim,
        cv2.INTER_AREA,
    )

    if abs(scale - scale_ref) > 1e-6:
        return

    src_rgb = cv2.cvtColor(
        src_small,
        cv2.COLOR_GRAY2BGR,
    )

    ref_rgb = cv2.cvtColor(
        ref_small,
        cv2.COLOR_GRAY2BGR,
    )

    h = max(
        src_rgb.shape[0],
        ref_rgb.shape[0],
    )

    w0 = src_rgb.shape[1]
    w1 = ref_rgb.shape[1]

    canvas = np.zeros(
        (h, w0 + w1, 3),
        dtype=np.uint8,
    )

    canvas[
        :src_rgb.shape[0],
        :w0,
    ] = src_rgb

    canvas[
        :ref_rgb.shape[0],
        w0:w0 + w1,
    ] = ref_rgb

    for p0, p1 in zip(
        src_pts,
        ref_pts,
    ):

        x0 = int(
            round(p0[0] * scale)
        )

        y0 = int(
            round(p0[1] * scale)
        )

        x1 = int(
            round(p1[0] * scale)
        ) + w0

        y1 = int(
            round(p1[1] * scale)
        )

        cv2.circle(
            canvas,
            (x0, y0),
            4,
            (0, 255, 0),
            -1,
        )

        cv2.circle(
            canvas,
            (x1, y1),
            4,
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
# CSV
# ============================================================


def write_summary_csv(results, path):
    fields = [
        "method",
        "representation",
        "raw_candidates",
        "filtered_candidates",
        "inliers",
        "inlier_ratio",
        "ransac_reprojection_rmse_px",
        "median_residual_px",
        "mean_residual_px",
        "coverage_cells",
        "valid_coverage_cells",
        "coverage_ratio",
        "uniform_inliers",
        "runtime_s",
        "sanity",
        "translation_px",
        "rotation_deg",
        "scale_x",
        "scale_y",
        "determinant",
        "axis_dot",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for r in results:

            params = (
                r.get("affine_parameters")
                or {}
            )

            row = {
                "method": r["method"],
                "representation": r["representation"],
                "raw_candidates": r["raw_candidates"],
                "filtered_candidates": r["filtered_candidates"],
                "inliers": r["inliers"],
                "inlier_ratio": r["inlier_ratio"],
                "ransac_reprojection_rmse_px":
                    r["ransac_reprojection_rmse_px"],
                "median_residual_px":
                    r["median_residual_px"],
                "mean_residual_px":
                    r["mean_residual_px"],
                "coverage_cells":
                    r["coverage_cells"],
                "valid_coverage_cells":
                    r["valid_coverage_cells"],
                "coverage_ratio":
                    r["coverage_ratio"],
                "uniform_inliers":
                    r["uniform_inliers"],
                "runtime_s":
                    r["runtime_s"],
                "sanity":
                    r["sanity"],
                "translation_px":
                    params.get(
                        "translation_magnitude_px"
                    ),
                "rotation_deg":
                    params.get(
                        "rotation_deg"
                    ),
                "scale_x":
                    params.get(
                        "scale_x"
                    ),
                "scale_y":
                    params.get(
                        "scale_y"
                    ),
                "determinant":
                    params.get(
                        "determinant"
                    ),
                "axis_dot":
                    params.get(
                        "axis_dot"
                    ),
            }

            writer.writerow(row)


# ============================================================
# PRINT RESULT
# ============================================================


def fmt(value, decimals=4):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.{decimals}f}"

    return str(value)


def print_result(result):
    print("\n" + "-" * 72)

    print(
        result["method"].upper(),
        "/",
        result["representation"].upper(),
    )

    print("-" * 72)

    print(
        "Raw candidates      :",
        result["raw_candidates"],
    )

    print(
        "Mask-filtered       :",
        result["filtered_candidates"],
    )

    print(
        "RANSAC inliers      :",
        result["inliers"],
    )

    print(
        "Inlier ratio        :",
        fmt(
            result["inlier_ratio"],
            6,
        ),
    )

    print(
        "RANSAC RMSE px      :",
        fmt(
            result[
                "ransac_reprojection_rmse_px"
            ],
            6,
        ),
    )

    print(
        "Median residual px  :",
        fmt(
            result[
                "median_residual_px"
            ],
            6,
        ),
    )

    print(
        "Mean residual px    :",
        fmt(
            result[
                "mean_residual_px"
            ],
            6,
        ),
    )

    print(
        "Spatial coverage    :",
        f'{result["coverage_cells"]}/'
        f'{result["valid_coverage_cells"]}',
        f'({result["coverage_ratio"]:.4f})',
    )

    print(
        "Uniform subset      :",
        result["uniform_inliers"],
    )

    print(
        "Runtime s           :",
        fmt(
            result["runtime_s"],
            3,
        ),
    )

    print(
        "Transform sane      :",
        result["sanity"],
    )

    params = result.get(
        "affine_parameters"
    )

    if params:

        print(
            "Translation px     :",
            fmt(
                params[
                    "translation_magnitude_px"
                ],
                4,
            ),
        )

        print(
            "Rotation deg       :",
            fmt(
                params[
                    "rotation_deg"
                ],
                4,
            ),
        )

        print(
            "Scale X/Y          :",
            fmt(
                params["scale_x"],
                6,
            ),
            "/",
            fmt(
                params["scale_y"],
                6,
            ),
        )

        print(
            "Determinant        :",
            fmt(
                params["determinant"],
                6,
            ),
        )

        print(
            "Axis dot           :",
            fmt(
                params["axis_dot"],
                6,
            ),
        )

        print("Affine:")

        matrix = np.asarray(
            result["affine"]
        )

        print(matrix)


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 72)
    print("PAIR 006 — GLOBAL CROSS-MISSION MATCHING BENCHMARK")
    print("=" * 72)

    print("\nSource:")
    print(SOURCE_PATH)

    print("\nReference:")
    print(REFERENCE_PATH)

    # --------------------------------------------------------
    # Read canonical pair
    # --------------------------------------------------------

    (
        source,
        source_raster_mask,
        source_transform,
        source_crs,
        source_nodata,
    ) = read_single_band(SOURCE_PATH)

    (
        reference,
        reference_raster_mask,
        reference_transform,
        reference_crs,
        reference_nodata,
    ) = read_single_band(REFERENCE_PATH)

    (
        common_mask_data,
        common_raster_mask,
        _,
        _,
        _,
    ) = read_single_band(COMMON_MASK_PATH)

    common_mask = (
        common_mask_data > 0
    )

    if source.shape != reference.shape:
        raise RuntimeError(
            "Canonical source/reference shapes differ."
        )

    if common_mask.shape != source.shape:
        raise RuntimeError(
            "Common mask shape differs."
        )

    print("\nShape:", source.shape)
    print("Pixel resolution:", PIXEL_SIZE_M, "m")

    # --------------------------------------------------------
    # Matcher mask
    # --------------------------------------------------------

    science_mask, matcher_mask = (
        build_matcher_mask(
            source,
            source_raster_mask,
            reference,
            reference_raster_mask,
            common_mask,
        )
    )

    print(
        "\nScience-valid pixels :",
        f"{int(science_mask.sum()):,}",
    )

    print(
        "Matcher-valid pixels :",
        f"{int(matcher_mask.sum()):,}",
    )

    print(
        "Excluded for matching:",
        f"{int(science_mask.sum() - matcher_mask.sum()):,}",
    )

    # Save matcher mask
    cv2.imwrite(
        str(
            OUT_DIR
            / "matcher_valid_mask.png"
        ),
        (
            matcher_mask.astype(np.uint8)
            * 255
        ),
    )

    # --------------------------------------------------------
    # Intensity representation
    # --------------------------------------------------------

    source_intensity = robust_uint8(
        source,
        matcher_mask,
    )

    reference_intensity = robust_uint8(
        reference,
        matcher_mask,
    )

    source_intensity[
        ~matcher_mask
    ] = 0

    reference_intensity[
        ~matcher_mask
    ] = 0

    # --------------------------------------------------------
    # Gradient representation
    # --------------------------------------------------------

    source_gradient = (
        gradient_magnitude_uint8(
            source_intensity,
            matcher_mask,
        )
    )

    reference_gradient = (
        gradient_magnitude_uint8(
            reference_intensity,
            matcher_mask,
        )
    )

    source_gradient[
        ~matcher_mask
    ] = 0

    reference_gradient[
        ~matcher_mask
    ] = 0

    cv2.imwrite(
        str(
            OUT_DIR
            / "source_intensity.png"
        ),
        source_intensity,
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / "reference_intensity.png"
        ),
        reference_intensity,
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / "source_gradient.png"
        ),
        source_gradient,
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / "reference_gradient.png"
        ),
        reference_gradient,
    )

    representations = {
        "intensity": (
            source_intensity,
            reference_intensity,
        ),
        "gradient": (
            source_gradient,
            reference_gradient,
        ),
    }

    results = []

    # --------------------------------------------------------
    # Run benchmark
    # --------------------------------------------------------

    for representation, (
        src_img,
        ref_img,
    ) in representations.items():

        print("\n")
        print("=" * 72)
        print(
            "REPRESENTATION:",
            representation.upper(),
        )
        print("=" * 72)

        # ====================================================
        # SIFT
        # ====================================================

        try:

            raw = run_sift(
                src_img,
                ref_img,
                matcher_mask,
            )

            result, detail = evaluate_matches(
                "sift",
                representation,
                raw,
                matcher_mask,
            )

            results.append(result)

            print_result(result)

            save_match_visualization(
                src_img,
                ref_img,
                detail,
                OUT_DIR
                / (
                    f"sift_{representation}"
                    "_matches.png"
                ),
            )

        except Exception as e:

            print("\nSIFT FAILED:")
            print(repr(e))

        # ====================================================
        # LIGHTGLUE
        # ====================================================

        try:

            raw = run_lightglue(
                src_img,
                ref_img,
            )

            result, detail = evaluate_matches(
                "lightglue",
                representation,
                raw,
                matcher_mask,
            )

            results.append(result)

            print_result(result)

            save_match_visualization(
                src_img,
                ref_img,
                detail,
                OUT_DIR
                / (
                    f"lightglue_{representation}"
                    "_matches.png"
                ),
            )

        except Exception as e:

            print("\nLIGHTGLUE FAILED:")
            print(repr(e))

        # ====================================================
        # LOFTR
        # ====================================================

        try:

            raw = run_loftr(
                src_img,
                ref_img,
            )

            result, detail = evaluate_matches(
                "loftr",
                representation,
                raw,
                matcher_mask,
            )

            results.append(result)

            print_result(result)

            save_match_visualization(
                src_img,
                ref_img,
                detail,
                OUT_DIR
                / (
                    f"loftr_{representation}"
                    "_matches.png"
                ),
            )

        except Exception as e:

            print("\nLOFTR FAILED:")
            print(repr(e))

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    json_path = (
        OUT_DIR
        / "pair006_global_benchmark.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    csv_path = (
        OUT_DIR
        / "pair006_global_benchmark.csv"
    )

    write_summary_csv(
        results,
        csv_path,
    )

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("PAIR 006 BENCHMARK SUMMARY")
    print("=" * 72)

    ranked = sorted(
        results,
        key=lambda r: (
            not r["sanity"],
            -r["inliers"],
            -r["coverage_ratio"],
            (
                r[
                    "ransac_reprojection_rmse_px"
                ]
                if r[
                    "ransac_reprojection_rmse_px"
                ]
                is not None
                else 1e9
            ),
        ),
    )

    for i, result in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"{i}. "
            f"{result['method']:<10} "
            f"{result['representation']:<10} "
            f"inliers={result['inliers']:<5} "
            f"ratio={result['inlier_ratio']:.4f} "
            f"RMSE={fmt(result['ransac_reprojection_rmse_px'], 4)} "
            f"coverage="
            f"{result['coverage_cells']}/"
            f"{result['valid_coverage_cells']} "
            f"sane={result['sanity']}"
        )

    print("\nIMPORTANT:")
    print(
        "RANSAC reprojection RMSE is an internal "
        "self-consistency residual."
    )

    print(
        "It is NOT independent ground-truth "
        "registration accuracy."
    )

    print("\nOutputs:")
    print(json_path)
    print(csv_path)
    print(OUT_DIR)

    print("\nPAIR 006 GLOBAL BENCHMARK COMPLETE.")


if __name__ == "__main__":
    main()

