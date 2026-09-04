from pathlib import Path
import json
import math
import time

import cv2
import numpy as np
import rasterio
import torch

import kornia.feature as KF

from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pair_005"
    / "canonical"
)

SOURCE_PATH = (
    CANONICAL_DIR
    / "kaguya_tc_source.tif"
)

REFERENCE_PATH = (
    CANONICAL_DIR
    / "tmc2_reference.tif"
)

SCIENCE_MASK_PATH = (
    CANONICAL_DIR
    / "common_valid_mask.tif"
)

MATCHER_MASK_PATH = (
    CANONICAL_DIR
    / "matcher_valid_mask.tif"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_005"
    / "global_benchmark"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PARAMETERS
# ============================================================

MAX_GLOBAL_DIM = 2048

SIFT_FEATURES = 12000

SUPERPOINT_FEATURES = 8192

RANSAC_THRESHOLD_PX = 3.0

BOUNDARY_MARGIN_PX = 8

GRID_ROWS = 8
GRID_COLS = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# NORMALIZATION
# ============================================================

def robust_normalize(
    image,
    mask,
    low=1.0,
    high=99.0,
):

    output = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    valid = (
        mask
        & np.isfinite(image)
    )

    if not np.any(valid):
        return output

    values = image[valid]

    lo = float(
        np.percentile(
            values,
            low,
        )
    )

    hi = float(
        np.percentile(
            values,
            high,
        )
    )

    if hi <= lo:
        hi = lo + 1.0

    scaled = (
        image.astype(np.float32)
        - lo
    ) / (
        hi - lo
    )

    scaled = np.clip(
        scaled,
        0.0,
        1.0,
    )

    output[valid] = (
        scaled[valid]
        * 255.0
    ).astype(np.uint8)

    return output


def gradient_representation(
    image_u8,
    mask,
):

    img = (
        image_u8.astype(np.float32)
        / 255.0
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

    mag = cv2.magnitude(
        gx,
        gy,
    )

    output = np.zeros_like(
        image_u8
    )

    valid = (
        mask
        & np.isfinite(mag)
    )

    if not np.any(valid):
        return output

    values = mag[valid]

    lo = float(
        np.percentile(
            values,
            1.0,
        )
    )

    hi = float(
        np.percentile(
            values,
            99.0,
        )
    )

    if hi <= lo:
        hi = lo + 1e-6

    normalized = (
        mag - lo
    ) / (
        hi - lo
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    output[valid] = (
        normalized[valid]
        * 255.0
    ).astype(np.uint8)

    return output


# ============================================================
# RESIZE
# ============================================================

def resize_global(
    image,
    max_dim,
    interpolation,
):

    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_dim / max(h, w),
    )

    if scale >= 1.0:

        return (
            image.copy(),
            1.0,
        )

    new_w = max(
        1,
        int(
            round(
                w * scale
            )
        ),
    )

    new_h = max(
        1,
        int(
            round(
                h * scale
            )
        ),
    )

    resized = cv2.resize(
        image,
        (
            new_w,
            new_h,
        ),
        interpolation=interpolation,
    )

    return (
        resized,
        scale,
    )


# ============================================================
# MASK FILTERING
# ============================================================

def point_mask_valid(
    points,
    mask,
    margin=0,
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

    valid = (
        (x >= margin)
        & (x < w - margin)
        & (y >= margin)
        & (y < h - margin)
    )

    result = np.zeros(
        len(points),
        dtype=bool,
    )

    idx = np.where(
        valid
    )[0]

    if len(idx) > 0:

        result[idx] = mask[
            y[idx],
            x[idx],
        ]

    return result


def filter_pair_by_mask(
    source_points,
    reference_points,
    mask,
):

    src_valid = point_mask_valid(
        source_points,
        mask,
        BOUNDARY_MARGIN_PX,
    )

    ref_valid = point_mask_valid(
        reference_points,
        mask,
        BOUNDARY_MARGIN_PX,
    )

    keep = (
        src_valid
        & ref_valid
    )

    return (
        source_points[keep],
        reference_points[keep],
    )


# ============================================================
# SIFT
# ============================================================

def run_sift(
    source_u8,
    reference_u8,
    scale,
):

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
    )

    k0, d0 = sift.detectAndCompute(
        source_u8,
        None,
    )

    k1, d1 = sift.detectAndCompute(
        reference_u8,
        None,
    )

    if (
        d0 is None
        or d1 is None
        or len(k0) == 0
        or len(k1) == 0
    ):

        return (
            np.empty(
                (0, 2),
                dtype=np.float32,
            ),
            np.empty(
                (0, 2),
                dtype=np.float32,
            ),
            0,
            0,
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
    )

    knn = matcher.knnMatch(
        d0,
        d1,
        k=2,
    )

    good = []

    for pair in knn:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < (
            0.75
            * n.distance
        ):
            good.append(m)

    source_points = np.array(
        [
            k0[
                m.queryIdx
            ].pt
            for m in good
        ],
        dtype=np.float32,
    )

    reference_points = np.array(
        [
            k1[
                m.trainIdx
            ].pt
            for m in good
        ],
        dtype=np.float32,
    )

    if len(source_points) > 0:

        source_points /= scale
        reference_points /= scale

    return (
        source_points,
        reference_points,
        len(k0),
        len(k1),
    )


# ============================================================
# LIGHTGLUE
# ============================================================

def run_lightglue(
    source_u8,
    reference_u8,
    scale,
    extractor,
    matcher,
):

    src = torch.from_numpy(
        source_u8
    ).float()[None, None]

    ref = torch.from_numpy(
        reference_u8
    ).float()[None, None]

    src = (
        src.to(DEVICE)
        / 255.0
    )

    ref = (
        ref.to(DEVICE)
        / 255.0
    )

    with torch.inference_mode():

        feats0 = extractor.extract(
            src
        )

        feats1 = extractor.extract(
            ref
        )

        matches01 = matcher(
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

        matches01 = rbd(
            matches01
        )

    matches = (
        matches01[
            "matches"
        ]
    )

    if matches.numel() == 0:

        return (
            np.empty(
                (0, 2),
                dtype=np.float32,
            ),
            np.empty(
                (0, 2),
                dtype=np.float32,
            ),
            int(
                len(
                    feats0[
                        "keypoints"
                    ]
                )
            ),
            int(
                len(
                    feats1[
                        "keypoints"
                    ]
                )
            ),
        )

    kpts0 = (
        feats0[
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

    kpts1 = (
        feats1[
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

    kpts0 /= scale
    kpts1 /= scale

    return (
        kpts0,
        kpts1,
        int(
            len(
                feats0[
                    "keypoints"
                ]
            )
        ),
        int(
            len(
                feats1[
                    "keypoints"
                ]
            )
        ),
    )


# ============================================================
# LOFTR
# ============================================================

def run_loftr(
    source_u8,
    reference_u8,
    scale,
    matcher,
):

    src = torch.from_numpy(
        source_u8
    ).float()[None, None]

    ref = torch.from_numpy(
        reference_u8
    ).float()[None, None]

    src = (
        src.to(DEVICE)
        / 255.0
    )

    ref = (
        ref.to(DEVICE)
        / 255.0
    )

    with torch.inference_mode():

        result = matcher(
            {
                "image0": src,
                "image1": ref,
            }
        )

    source_points = (
        result[
            "keypoints0"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    reference_points = (
        result[
            "keypoints1"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    if len(source_points) > 0:

        source_points /= scale
        reference_points /= scale

    return (
        source_points,
        reference_points,
    )


# ============================================================
# AFFINE / RANSAC
# ============================================================

def fit_affine(
    source_points,
    reference_points,
):

    if len(source_points) < 3:

        return (
            None,
            np.zeros(
                len(source_points),
                dtype=bool,
            ),
        )

    affine, mask = cv2.estimateAffine2D(
        source_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=(
            RANSAC_THRESHOLD_PX
        ),
        maxIters=10000,
        confidence=0.999,
        refineIters=10,
    )

    if (
        affine is None
        or mask is None
    ):

        return (
            None,
            np.zeros(
                len(source_points),
                dtype=bool,
            ),
        )

    return (
        affine.astype(
            np.float64
        ),
        mask.ravel().astype(bool),
    )


def affine_residuals(
    affine,
    source_points,
    reference_points,
):

    if (
        affine is None
        or len(source_points) == 0
    ):

        return np.empty(
            0,
            dtype=np.float64,
        )

    transformed = cv2.transform(
        source_points[
            None,
            :, :
        ],
        affine,
    )[0]

    residual = np.linalg.norm(
        transformed
        - reference_points,
        axis=1,
    )

    return residual.astype(
        np.float64
    )


# ============================================================
# TRANSFORM PARAMETERS
# ============================================================

def describe_affine(
    affine,
):

    if affine is None:

        return {
            "sane": False,
        }

    a = float(
        affine[0, 0]
    )

    b = float(
        affine[0, 1]
    )

    c = float(
        affine[1, 0]
    )

    d = float(
        affine[1, 1]
    )

    tx = float(
        affine[0, 2]
    )

    ty = float(
        affine[1, 2]
    )

    scale_x = math.sqrt(
        a * a
        + c * c
    )

    scale_y = math.sqrt(
        b * b
        + d * d
    )

    rotation_deg = math.degrees(
        math.atan2(
            c,
            a,
        )
    )

    determinant = (
        a * d
        - b * c
    )

    axis_dot = (
        a * b
        + c * d
    )

    translation = math.sqrt(
        tx * tx
        + ty * ty
    )

    sane = (
        determinant > 0.0
        and 0.85 <= scale_x <= 1.15
        and 0.85 <= scale_y <= 1.15
        and abs(rotation_deg) <= 10.0
        and translation <= 250.0
        and abs(axis_dot) <= 0.25
    )

    return {
        "translation_px":
            translation,
        "tx_px":
            tx,
        "ty_px":
            ty,
        "rotation_deg":
            rotation_deg,
        "scale_x":
            scale_x,
        "scale_y":
            scale_y,
        "determinant":
            determinant,
        "axis_dot":
            axis_dot,
        "sane":
            bool(sane),
    }


# ============================================================
# COVERAGE
# ============================================================

def compute_coverage(
    inlier_reference_points,
    matcher_mask,
):

    h, w = matcher_mask.shape

    valid_cells = set()

    occupied_cells = set()

    for gy in range(
        GRID_ROWS
    ):

        y0 = int(
            round(
                gy
                * h
                / GRID_ROWS
            )
        )

        y1 = int(
            round(
                (gy + 1)
                * h
                / GRID_ROWS
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

            cell_mask = (
                matcher_mask[
                    y0:y1,
                    x0:x1,
                ]
            )

            # Cell counts as usable if at least
            # 1% of it contains matcher-valid terrain.
            if (
                cell_mask.size > 0
                and cell_mask.mean()
                >= 0.01
            ):

                valid_cells.add(
                    (
                        gy,
                        gx,
                    )
                )

    for point in (
        inlier_reference_points
    ):

        x = float(
            point[0]
        )

        y = float(
            point[1]
        )

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

        cell = (
            gy,
            gx,
        )

        if cell in valid_cells:

            occupied_cells.add(
                cell
            )

    denominator = len(
        valid_cells
    )

    numerator = len(
        occupied_cells
    )

    coverage = (
        numerator / denominator
        if denominator > 0
        else 0.0
    )

    return {
        "occupied_cells":
            numerator,
        "valid_cells":
            denominator,
        "coverage":
            coverage,
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_match_visualization(
    path,
    source_u8,
    reference_u8,
    source_points,
    reference_points,
    inlier_mask,
):

    h, w = source_u8.shape

    max_preview_width = 1600

    scale = min(
        1.0,
        max_preview_width
        / (
            2 * w
        ),
    )

    preview_w = max(
        1,
        int(
            round(
                w * scale
            )
        ),
    )

    preview_h = max(
        1,
        int(
            round(
                h * scale
            )
        ),
    )

    src = cv2.resize(
        source_u8,
        (
            preview_w,
            preview_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    ref = cv2.resize(
        reference_u8,
        (
            preview_w,
            preview_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.concatenate(
        [
            cv2.cvtColor(
                src,
                cv2.COLOR_GRAY2BGR,
            ),
            cv2.cvtColor(
                ref,
                cv2.COLOR_GRAY2BGR,
            ),
        ],
        axis=1,
    )

    inlier_indices = np.where(
        inlier_mask
    )[0]

    # Prevent enormous unreadable previews.
    if len(inlier_indices) > 500:

        chosen = np.linspace(
            0,
            len(inlier_indices) - 1,
            500,
        ).astype(int)

        inlier_indices = (
            inlier_indices[
                chosen
            ]
        )

    for idx in inlier_indices:

        p0 = source_points[
            idx
        ]

        p1 = reference_points[
            idx
        ]

        x0 = int(
            round(
                p0[0]
                * scale
            )
        )

        y0 = int(
            round(
                p0[1]
                * scale
            )
        )

        x1 = int(
            round(
                p1[0]
                * scale
                + preview_w
            )
        )

        y1 = int(
            round(
                p1[1]
                * scale
            )
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
                0,
            ),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(path),
        canvas,
    )


# ============================================================
# RESULT PROCESSING
# ============================================================

def evaluate_matches(
    matcher_name,
    representation,
    source_points,
    reference_points,
    raw_match_count,
    matcher_mask,
    source_vis,
    reference_vis,
    runtime,
    extra=None,
):

    filtered_source, filtered_reference = (
        filter_pair_by_mask(
            source_points,
            reference_points,
            matcher_mask,
        )
    )

    affine, inlier_mask = (
        fit_affine(
            filtered_source,
            filtered_reference,
        )
    )

    inlier_count = int(
        inlier_mask.sum()
    )

    candidate_count = len(
        filtered_source
    )

    inlier_ratio = (
        inlier_count
        / candidate_count
        if candidate_count > 0
        else 0.0
    )

    residuals = (
        affine_residuals(
            affine,
            filtered_source,
            filtered_reference,
        )
    )

    if inlier_count > 0:

        inlier_residuals = (
            residuals[
                inlier_mask
            ]
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    inlier_residuals
                    ** 2
                )
            )
        )

        median = float(
            np.median(
                inlier_residuals
            )
        )

        mean = float(
            np.mean(
                inlier_residuals
            )
        )

        coverage = compute_coverage(
            filtered_reference[
                inlier_mask
            ],
            matcher_mask,
        )

    else:

        rmse = None
        median = None
        mean = None

        coverage = {
            "occupied_cells": 0,
            "valid_cells": 0,
            "coverage": 0.0,
        }

    transform = describe_affine(
        affine
    )

    result = {
        "matcher":
            matcher_name,
        "representation":
            representation,
        "raw_matches":
            int(
                raw_match_count
            ),
        "mask_filtered_candidates":
            int(
                candidate_count
            ),
        "ransac_inliers":
            inlier_count,
        "inlier_ratio":
            float(
                inlier_ratio
            ),
        "ransac_reprojection_rmse_px":
            rmse,
        "ransac_median_residual_px":
            median,
        "ransac_mean_residual_px":
            mean,
        "coverage":
            coverage,
        "affine":
            (
                affine.tolist()
                if affine is not None
                else None
            ),
        "transform":
            transform,
        "runtime_seconds":
            float(
                runtime
            ),
    }

    if extra is not None:

        result[
            "extra"
        ] = extra

    visualization_path = (
        OUT_DIR
        / (
            f"{matcher_name.lower()}_"
            f"{representation}_matches.png"
        )
    )

    save_match_visualization(
        visualization_path,
        source_vis,
        reference_vis,
        filtered_source,
        filtered_reference,
        inlier_mask,
    )

    result[
        "visualization"
    ] = str(
        visualization_path
    )

    return result


# ============================================================
# PRINTING
# ============================================================

def print_result(
    result,
):

    print("\n" + "-" * 78)

    print(
        result[
            "matcher"
        ],
        "/",
        result[
            "representation"
        ],
    )

    print("-" * 78)

    print(
        "Raw matches:",
        result[
            "raw_matches"
        ],
    )

    print(
        "Mask-filtered candidates:",
        result[
            "mask_filtered_candidates"
        ],
    )

    print(
        "RANSAC inliers:",
        result[
            "ransac_inliers"
        ],
    )

    print(
        "Inlier ratio:",
        f"{result['inlier_ratio']:.6f}",
    )

    rmse = result[
        "ransac_reprojection_rmse_px"
    ]

    if rmse is not None:

        print(
            "RANSAC self RMSE:",
            f"{rmse:.6f} px",
        )

        print(
            "Median residual:",
            f"{result['ransac_median_residual_px']:.6f} px",
        )

    else:

        print(
            "RANSAC self RMSE: N/A"
        )

    coverage = result[
        "coverage"
    ]

    print(
        "Spatial coverage:",
        f"{coverage['occupied_cells']}/"
        f"{coverage['valid_cells']}"
        f" = {coverage['coverage']:.6f}",
    )

    print(
        "Runtime:",
        f"{result['runtime_seconds']:.3f} s",
    )

    transform = result[
        "transform"
    ]

    print(
        "Transform sane:",
        transform.get(
            "sane",
            False,
        ),
    )

    if (
        result[
            "affine"
        ]
        is not None
    ):

        print(
            "Affine:"
        )

        print(
            np.array(
                result[
                    "affine"
                ]
            )
        )

        print(
            "Translation:",
            f"{transform['translation_px']:.4f} px",
        )

        print(
            "Rotation:",
            f"{transform['rotation_deg']:.6f} deg",
        )

        print(
            "Scale X/Y:",
            f"{transform['scale_x']:.8f}",
            "/",
            f"{transform['scale_y']:.8f}",
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "PAIR 005 — GLOBAL MATCHING BENCHMARK"
    )
    print("=" * 78)

    print(
        "Device:",
        DEVICE,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # LOAD CANONICAL PAIR
    # --------------------------------------------------------

    with rasterio.open(
        SOURCE_PATH
    ) as ds:

        source = ds.read(
            1
        ).astype(
            np.float32
        )

    with rasterio.open(
        REFERENCE_PATH
    ) as ds:

        reference = ds.read(
            1
        ).astype(
            np.float32
        )

    with rasterio.open(
        SCIENCE_MASK_PATH
    ) as ds:

        science_mask = (
            ds.read(1)
            > 0
        )

    with rasterio.open(
        MATCHER_MASK_PATH
    ) as ds:

        matcher_mask = (
            ds.read(1)
            > 0
        )

    if (
        source.shape
        != reference.shape
        or source.shape
        != science_mask.shape
        or source.shape
        != matcher_mask.shape
    ):

        raise RuntimeError(
            "Canonical rasters/masks have "
            "different shapes."
        )

    h, w = source.shape

    print(
        "\nCanonical shape:",
        h,
        "x",
        w,
    )

    print(
        "Science valid:",
        f"{int(science_mask.sum()):,}",
    )

    print(
        "Matcher valid:",
        f"{int(matcher_mask.sum()):,}",
    )

    # --------------------------------------------------------
    # REPRESENTATIONS
    # --------------------------------------------------------

    source_intensity = robust_normalize(
        source,
        science_mask,
    )

    reference_intensity = robust_normalize(
        reference,
        science_mask,
    )

    source_gradient = gradient_representation(
        source_intensity,
        matcher_mask,
    )

    reference_gradient = gradient_representation(
        reference_intensity,
        matcher_mask,
    )

    source_intensity[
        ~science_mask
    ] = 0

    reference_intensity[
        ~science_mask
    ] = 0

    source_gradient[
        ~matcher_mask
    ] = 0

    reference_gradient[
        ~matcher_mask
    ] = 0

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

    # --------------------------------------------------------
    # GLOBAL RESIZE SCALE
    # --------------------------------------------------------

    _, scale = resize_global(
        source_intensity,
        MAX_GLOBAL_DIM,
        cv2.INTER_AREA,
    )

    print(
        "\nGlobal resize scale:",
        f"{scale:.8f}",
    )

    print(
        "Effective matching resolution:",
        f"{10.0 / scale:.3f} m/pixel",
    )

    # --------------------------------------------------------
    # LOAD DEEP MATCHERS ONCE
    # --------------------------------------------------------

    print("\nLoading SuperPoint + LightGlue...")

    extractor = SuperPoint(
        max_num_keypoints=(
            SUPERPOINT_FEATURES
        )
    ).eval().to(
        DEVICE
    )

    lightglue = LightGlue(
        features="superpoint"
    ).eval().to(
        DEVICE
    )

    print(
        "Loading LoFTR..."
    )

    loftr = KF.LoFTR(
        pretrained="outdoor"
    ).eval().to(
        DEVICE
    )

    results = []

    # --------------------------------------------------------
    # BENCHMARK EACH REPRESENTATION
    # --------------------------------------------------------

    for representation, (
        source_full,
        reference_full,
    ) in representations.items():

        print("\n" + "=" * 78)
        print(
            "REPRESENTATION:",
            representation.upper(),
        )
        print("=" * 78)

        source_small, scale_source = (
            resize_global(
                source_full,
                MAX_GLOBAL_DIM,
                cv2.INTER_AREA,
            )
        )

        reference_small, scale_reference = (
            resize_global(
                reference_full,
                MAX_GLOBAL_DIM,
                cv2.INTER_AREA,
            )
        )

        if abs(
            scale_source
            - scale_reference
        ) > 1e-9:

            raise RuntimeError(
                "Source/reference resize "
                "scales differ."
            )

        current_scale = (
            scale_source
        )

        print(
            "Matching shape:",
            source_small.shape,
        )

        # ====================================================
        # SIFT
        # ====================================================

        print(
            "\nRunning SIFT..."
        )

        start = time.perf_counter()

        (
            sift_src,
            sift_ref,
            sift_k0,
            sift_k1,
        ) = run_sift(
            source_small,
            reference_small,
            current_scale,
        )

        runtime = (
            time.perf_counter()
            - start
        )

        sift_result = evaluate_matches(
            "SIFT",
            representation,
            sift_src,
            sift_ref,
            len(sift_src),
            matcher_mask,
            source_full,
            reference_full,
            runtime,
            extra={
                "source_keypoints":
                    sift_k0,
                "reference_keypoints":
                    sift_k1,
            },
        )

        results.append(
            sift_result
        )

        print_result(
            sift_result
        )

        # ====================================================
        # LIGHTGLUE
        # ====================================================

        print(
            "\nRunning LightGlue..."
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        (
            lg_src,
            lg_ref,
            lg_k0,
            lg_k1,
        ) = run_lightglue(
            source_small,
            reference_small,
            current_scale,
            extractor,
            lightglue,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        runtime = (
            time.perf_counter()
            - start
        )

        lg_result = evaluate_matches(
            "LightGlue",
            representation,
            lg_src,
            lg_ref,
            len(lg_src),
            matcher_mask,
            source_full,
            reference_full,
            runtime,
            extra={
                "source_keypoints":
                    lg_k0,
                "reference_keypoints":
                    lg_k1,
            },
        )

        results.append(
            lg_result
        )

        print_result(
            lg_result
        )

        # ====================================================
        # LOFTR
        # ====================================================

        print(
            "\nRunning LoFTR..."
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        (
            loftr_src,
            loftr_ref,
        ) = run_loftr(
            source_small,
            reference_small,
            current_scale,
            loftr,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        runtime = (
            time.perf_counter()
            - start
        )

        loftr_result = evaluate_matches(
            "LoFTR",
            representation,
            loftr_src,
            loftr_ref,
            len(
                loftr_src
            ),
            matcher_mask,
            source_full,
            reference_full,
            runtime,
        )

        results.append(
            loftr_result
        )

        print_result(
            loftr_result
        )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    output = {
        "pair_id":
            "pair_005",
        "source":
            "SELENE/Kaguya TC",
        "reference":
            "Chandrayaan-2 TMC-2",
        "canonical_resolution_m":
            10.0,
        "canonical_shape": [
            h,
            w,
        ],
        "max_global_dimension":
            MAX_GLOBAL_DIM,
        "ransac_threshold_px":
            RANSAC_THRESHOLD_PX,
        "coverage_grid": [
            GRID_ROWS,
            GRID_COLS,
        ],
        "device":
            DEVICE,
        "results":
            results,
        "metric_note": (
            "ransac_reprojection_rmse_px "
            "is an internal/self-consistency "
            "residual and is NOT independent "
            "ground-truth registration accuracy."
        ),
    }

    json_path = (
        OUT_DIR
        / "pair005_global_benchmark.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 78)
    print(
        "PAIR 005 GLOBAL BENCHMARK SUMMARY"
    )
    print("=" * 78)

    print(
        "\n"
        f"{'Matcher':<12}"
        f"{'Rep':<12}"
        f"{'Cand':>8}"
        f"{'Inlier':>8}"
        f"{'Ratio':>10}"
        f"{'RMSE':>10}"
        f"{'Coverage':>14}"
        f"{'Sane':>8}"
    )

    print(
        "-" * 82
    )

    for result in results:

        rmse = (
            result[
                "ransac_reprojection_rmse_px"
            ]
        )

        rmse_text = (
            f"{rmse:.4f}"
            if rmse is not None
            else "N/A"
        )

        coverage = result[
            "coverage"
        ]

        coverage_text = (
            f"{coverage['occupied_cells']}/"
            f"{coverage['valid_cells']}"
        )

        print(
            f"{result['matcher']:<12}"
            f"{result['representation']:<12}"
            f"{result['mask_filtered_candidates']:>8}"
            f"{result['ransac_inliers']:>8}"
            f"{result['inlier_ratio']:>10.4f}"
            f"{rmse_text:>10}"
            f"{coverage_text:>14}"
            f"{str(result['transform'].get('sane', False)):>8}"
        )

    print("\nSaved:")
    print(
        json_path
    )

    print("\n" + "=" * 78)
    print(
        "PAIR 005 GLOBAL BENCHMARK COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()