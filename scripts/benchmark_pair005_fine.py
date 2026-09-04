from pathlib import Path
import gc
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

COARSE_DIR = (
    ROOT
    / "results"
    / "pair_005"
    / "coarse_registration"
)

SOURCE_PATH = (
    COARSE_DIR
    / "kaguya_registered_coarse_to_tmc2.tif"
)

REFERENCE_PATH = (
    CANONICAL_DIR
    / "tmc2_reference.tif"
)

REGISTERED_COMMON_PATH = (
    COARSE_DIR
    / "registered_common_valid_mask.tif"
)

REFERENCE_MATCHER_MASK_PATH = (
    CANONICAL_DIR
    / "matcher_valid_mask.tif"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_005"
    / "fine_benchmark"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PARAMETERS
# ============================================================

GLOBAL_MAX_DIM = 1280
LOFTR_FALLBACK_MAX_DIM = 1024

SIFT_FEATURES = 10000
SUPERPOINT_FEATURES = 4096

RANSAC_THRESHOLD_PX = 3.0

GRID_ROWS = 8
GRID_COLS = 8

MASK_EROSION_RADIUS = 4
BOUNDARY_MARGIN = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Coarse transform already accepted by visual QA.
#
# Original Kaguya canonical pixel
#            ↓
# TMC canonical pixel
COARSE_AFFINE = np.array(
    [
        [
            1.00186985,
            -0.000798945593,
            423.414579,
        ],
        [
            -0.000295021855,
            1.00234560,
            -2.80028799,
        ],
    ],
    dtype=np.float64,
)


# ============================================================
# IMAGE HELPERS
# ============================================================

def robust_normalize(
    image,
    mask,
    low=1.0,
    high=99.0,
):
    out = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    valid = (
        mask
        & np.isfinite(image)
    )

    if not np.any(valid):
        return out

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

    temp = (
        image.astype(np.float32)
        - lo
    ) / (
        hi - lo
    )

    temp = np.clip(
        temp,
        0.0,
        1.0,
    )

    out[valid] = (
        temp[valid]
        * 255.0
    ).astype(np.uint8)

    return out


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

    out = np.zeros_like(
        image_u8
    )

    valid = (
        mask
        & np.isfinite(mag)
    )

    if not np.any(valid):
        return out

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

    temp = (
        mag - lo
    ) / (
        hi - lo
    )

    temp = np.clip(
        temp,
        0.0,
        1.0,
    )

    out[valid] = (
        temp[valid]
        * 255.0
    ).astype(np.uint8)

    return out


def resize_to_max(
    image,
    max_dim,
):
    h, w = image.shape

    scale = min(
        1.0,
        max_dim / max(h, w),
    )

    if scale >= 1.0:
        return image.copy(), 1.0

    nw = int(
        round(w * scale)
    )

    nh = int(
        round(h * scale)
    )

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


# ============================================================
# MASK
# ============================================================

def build_fine_mask(
    registered_common,
    reference_matcher,
):
    base = (
        registered_common
        & reference_matcher
    )

    k = (
        2 * MASK_EROSION_RADIUS
        + 1
    )

    kernel = np.ones(
        (k, k),
        dtype=np.uint8,
    )

    eroded = cv2.erode(
        base.astype(np.uint8),
        kernel,
        iterations=1,
    ) > 0

    return eroded


def mask_points(
    src_pts,
    ref_pts,
    mask,
):
    if len(src_pts) == 0:
        return (
            src_pts,
            ref_pts,
        )

    h, w = mask.shape

    sx = np.rint(
        src_pts[:, 0]
    ).astype(np.int64)

    sy = np.rint(
        src_pts[:, 1]
    ).astype(np.int64)

    rx = np.rint(
        ref_pts[:, 0]
    ).astype(np.int64)

    ry = np.rint(
        ref_pts[:, 1]
    ).astype(np.int64)

    valid = (
        (sx >= BOUNDARY_MARGIN)
        & (sx < w - BOUNDARY_MARGIN)
        & (sy >= BOUNDARY_MARGIN)
        & (sy < h - BOUNDARY_MARGIN)
        & (rx >= BOUNDARY_MARGIN)
        & (rx < w - BOUNDARY_MARGIN)
        & (ry >= BOUNDARY_MARGIN)
        & (ry < h - BOUNDARY_MARGIN)
    )

    idx = np.where(valid)[0]

    keep = np.zeros(
        len(src_pts),
        dtype=bool,
    )

    if len(idx) > 0:
        keep[idx] = (
            mask[
                sy[idx],
                sx[idx],
            ]
            & mask[
                ry[idx],
                rx[idx],
            ]
        )

    return (
        src_pts[keep],
        ref_pts[keep],
    )


# ============================================================
# SIFT
# ============================================================

def run_sift(
    src,
    ref,
    scale,
):
    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
    )

    k0, d0 = sift.detectAndCompute(
        src,
        None,
    )

    k1, d1 = sift.detectAndCompute(
        ref,
        None,
    )

    if (
        d0 is None
        or d1 is None
    ):
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            0,
            0,
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
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

        if m.distance < 0.75 * n.distance:
            good.append(m)

    p0 = np.array(
        [
            k0[m.queryIdx].pt
            for m in good
        ],
        dtype=np.float32,
    )

    p1 = np.array(
        [
            k1[m.trainIdx].pt
            for m in good
        ],
        dtype=np.float32,
    )

    if len(p0) > 0:
        p0 /= scale
        p1 /= scale

    return (
        p0,
        p1,
        len(k0),
        len(k1),
    )


# ============================================================
# LIGHTGLUE
# ============================================================

def run_lightglue(
    src,
    ref,
    scale,
    extractor,
    matcher,
):
    t0 = (
        torch.from_numpy(src)
        .float()[None, None]
        .to(DEVICE)
        / 255.0
    )

    t1 = (
        torch.from_numpy(ref)
        .float()[None, None]
        .to(DEVICE)
        / 255.0
    )

    with torch.inference_mode():
        f0 = extractor.extract(t0)
        f1 = extractor.extract(t1)

        matches = matcher(
            {
                "image0": f0,
                "image1": f1,
            }
        )

        f0 = rbd(f0)
        f1 = rbd(f1)
        matches = rbd(matches)

    pairs = matches["matches"]

    if pairs.numel() == 0:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            len(f0["keypoints"]),
            len(f1["keypoints"]),
        )

    p0 = (
        f0["keypoints"][
            pairs[:, 0]
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    p1 = (
        f1["keypoints"][
            pairs[:, 1]
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    p0 /= scale
    p1 /= scale

    return (
        p0,
        p1,
        len(f0["keypoints"]),
        len(f1["keypoints"]),
    )


# ============================================================
# LOFTR
# ============================================================

def run_loftr_once(
    src_full,
    ref_full,
    max_dim,
    matcher,
):
    src, scale0 = resize_to_max(
        src_full,
        max_dim,
    )

    ref, scale1 = resize_to_max(
        ref_full,
        max_dim,
    )

    if abs(scale0 - scale1) > 1e-9:
        raise RuntimeError(
            "LoFTR resize mismatch."
        )

    t0 = (
        torch.from_numpy(src)
        .float()[None, None]
        .to(DEVICE)
        / 255.0
    )

    t1 = (
        torch.from_numpy(ref)
        .float()[None, None]
        .to(DEVICE)
        / 255.0
    )

    with torch.inference_mode():
        out = matcher(
            {
                "image0": t0,
                "image1": t1,
            }
        )

    p0 = (
        out["keypoints0"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    p1 = (
        out["keypoints1"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if len(p0) > 0:
        p0 /= scale0
        p1 /= scale0

    return (
        p0,
        p1,
        scale0,
        src.shape,
    )


def run_loftr_safe(
    src,
    ref,
    matcher,
):
    try:
        return (
            *run_loftr_once(
                src,
                ref,
                GLOBAL_MAX_DIM,
                matcher,
            ),
            GLOBAL_MAX_DIM,
        )

    except RuntimeError as exc:
        text = str(exc).lower()

        if (
            "out of memory" not in text
            and "memoryallocation" not in text
        ):
            raise

        print(
            "\nLoFTR OOM at",
            GLOBAL_MAX_DIM,
            "— retrying at",
            LOFTR_FALLBACK_MAX_DIM,
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            *run_loftr_once(
                src,
                ref,
                LOFTR_FALLBACK_MAX_DIM,
                matcher,
            ),
            LOFTR_FALLBACK_MAX_DIM,
        )


# ============================================================
# GEOMETRY
# ============================================================

def fit_affine(
    src_pts,
    ref_pts,
):
    if len(src_pts) < 3:
        return (
            None,
            np.zeros(
                len(src_pts),
                dtype=bool,
            ),
        )

    M, mask = cv2.estimateAffine2D(
        src_pts,
        ref_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=(
            RANSAC_THRESHOLD_PX
        ),
        maxIters=10000,
        confidence=0.999,
        refineIters=10,
    )

    if (
        M is None
        or mask is None
    ):
        return (
            None,
            np.zeros(
                len(src_pts),
                dtype=bool,
            ),
        )

    return (
        M.astype(np.float64),
        mask.ravel().astype(bool),
    )


def residuals(
    M,
    src_pts,
    ref_pts,
):
    if M is None:
        return np.empty(
            0,
            dtype=np.float64,
        )

    pred = cv2.transform(
        src_pts[None],
        M,
    )[0]

    return np.linalg.norm(
        pred - ref_pts,
        axis=1,
    )


def describe_affine(
    M,
):
    if M is None:
        return {
            "fine_sane": False,
        }

    a, b, tx = M[0]
    c, d, ty = M[1]

    sx = math.sqrt(
        a * a + c * c
    )

    sy = math.sqrt(
        b * b + d * d
    )

    rot = math.degrees(
        math.atan2(c, a)
    )

    det = (
        a * d
        - b * c
    )

    dot = (
        a * b
        + c * d
    )

    translation = math.hypot(
        tx,
        ty,
    )

    fine_sane = (
        det > 0
        and 0.98 <= sx <= 1.02
        and 0.98 <= sy <= 1.02
        and abs(rot) <= 2.0
        and translation <= 30.0
        and abs(dot) <= 0.05
    )

    return {
        "tx_px": float(tx),
        "ty_px": float(ty),
        "translation_px":
            float(translation),
        "rotation_deg":
            float(rot),
        "scale_x":
            float(sx),
        "scale_y":
            float(sy),
        "determinant":
            float(det),
        "axis_dot":
            float(dot),
        "fine_sane":
            bool(fine_sane),
    }


def compose_affines(
    fine,
    coarse,
):
    if fine is None:
        return None

    Hf = np.eye(
        3,
        dtype=np.float64,
    )

    Hc = np.eye(
        3,
        dtype=np.float64,
    )

    Hf[:2] = fine
    Hc[:2] = coarse

    total = (
        Hf @ Hc
    )

    return total[:2]


# ============================================================
# COVERAGE
# ============================================================

def coverage(
    inlier_ref,
    mask,
):
    h, w = mask.shape

    valid_cells = set()
    occupied = set()

    for gy in range(
        GRID_ROWS
    ):
        y0 = int(
            gy * h / GRID_ROWS
        )

        y1 = int(
            (gy + 1)
            * h
            / GRID_ROWS
        )

        for gx in range(
            GRID_COLS
        ):
            x0 = int(
                gx * w / GRID_COLS
            )

            x1 = int(
                (gx + 1)
                * w
                / GRID_COLS
            )

            cell = mask[
                y0:y1,
                x0:x1,
            ]

            if (
                cell.size
                and cell.mean() >= 0.01
            ):
                valid_cells.add(
                    (gy, gx)
                )

    for x, y in inlier_ref:
        gx = int(
            x / w
            * GRID_COLS
        )

        gy = int(
            y / h
            * GRID_ROWS
        )

        gx = np.clip(
            gx,
            0,
            GRID_COLS - 1,
        )

        gy = np.clip(
            gy,
            0,
            GRID_ROWS - 1,
        )

        if (
            int(gy),
            int(gx),
        ) in valid_cells:
            occupied.add(
                (
                    int(gy),
                    int(gx),
                )
            )

    denom = len(
        valid_cells
    )

    num = len(
        occupied
    )

    return {
        "occupied_cells": num,
        "valid_cells": denom,
        "coverage":
            num / denom
            if denom
            else 0.0,
    }


# ============================================================
# MATCH VISUALIZATION
# ============================================================

def save_matches(
    path,
    src,
    ref,
    src_pts,
    ref_pts,
    inliers,
):
    h, w = src.shape

    preview_scale = min(
        1.0,
        1800.0 / (2 * w),
    )

    pw = int(
        round(
            w * preview_scale
        )
    )

    ph = int(
        round(
            h * preview_scale
        )
    )

    a = cv2.resize(
        src,
        (pw, ph),
        interpolation=cv2.INTER_AREA,
    )

    b = cv2.resize(
        ref,
        (pw, ph),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.concatenate(
        [
            cv2.cvtColor(
                a,
                cv2.COLOR_GRAY2BGR,
            ),
            cv2.cvtColor(
                b,
                cv2.COLOR_GRAY2BGR,
            ),
        ],
        axis=1,
    )

    ids = np.where(
        inliers
    )[0]

    if len(ids) > 400:
        take = np.linspace(
            0,
            len(ids) - 1,
            400,
        ).astype(int)

        ids = ids[take]

    for i in ids:
        p0 = src_pts[i]
        p1 = ref_pts[i]

        x0 = int(
            round(
                p0[0]
                * preview_scale
            )
        )

        y0 = int(
            round(
                p0[1]
                * preview_scale
            )
        )

        x1 = int(
            round(
                p1[0]
                * preview_scale
                + pw
            )
        )

        y1 = int(
            round(
                p1[1]
                * preview_scale
            )
        )

        cv2.line(
            canvas,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(path),
        canvas,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    matcher_name,
    representation,
    raw_src,
    raw_ref,
    fine_mask,
    full_src_vis,
    full_ref_vis,
    runtime,
    extra=None,
):
    src_pts, ref_pts = mask_points(
        raw_src,
        raw_ref,
        fine_mask,
    )

    M, inliers = fit_affine(
        src_pts,
        ref_pts,
    )

    n_candidates = len(
        src_pts
    )

    n_inliers = int(
        inliers.sum()
    )

    ratio = (
        n_inliers
        / n_candidates
        if n_candidates
        else 0.0
    )

    r = residuals(
        M,
        src_pts,
        ref_pts,
    )

    if n_inliers:
        rin = r[inliers]

        rmse = float(
            np.sqrt(
                np.mean(
                    rin ** 2
                )
            )
        )

        med = float(
            np.median(rin)
        )

        mean = float(
            np.mean(rin)
        )

        cov = coverage(
            ref_pts[inliers],
            fine_mask,
        )

    else:
        rmse = None
        med = None
        mean = None

        cov = {
            "occupied_cells": 0,
            "valid_cells": 0,
            "coverage": 0.0,
        }

    desc = describe_affine(M)

    total_affine = compose_affines(
        M,
        COARSE_AFFINE,
    )

    result = {
        "matcher":
            matcher_name,
        "representation":
            representation,
        "raw_matches":
            int(len(raw_src)),
        "mask_filtered_candidates":
            int(n_candidates),
        "ransac_inliers":
            n_inliers,
        "inlier_ratio":
            float(ratio),
        "ransac_reprojection_rmse_px":
            rmse,
        "ransac_median_residual_px":
            med,
        "ransac_mean_residual_px":
            mean,
        "coverage":
            cov,
        "fine_affine":
            M.tolist()
            if M is not None
            else None,
        "fine_transform":
            desc,
        "composed_original_kaguya_to_tmc_affine":
            (
                total_affine.tolist()
                if total_affine
                is not None
                else None
            ),
        "runtime_seconds":
            float(runtime),
    }

    if extra is not None:
        result["extra"] = extra

    out_img = (
        OUT_DIR
        / (
            matcher_name.lower()
            + "_"
            + representation
            + "_matches.png"
        )
    )

    save_matches(
        out_img,
        full_src_vis,
        full_ref_vis,
        src_pts,
        ref_pts,
        inliers,
    )

    result[
        "visualization"
    ] = str(out_img)

    return result


def print_result(
    r,
):
    print("\n" + "-" * 78)

    print(
        r["matcher"],
        "/",
        r["representation"],
    )

    print("-" * 78)

    print(
        "Raw matches:",
        r["raw_matches"],
    )

    print(
        "Mask-filtered:",
        r[
            "mask_filtered_candidates"
        ],
    )

    print(
        "RANSAC inliers:",
        r["ransac_inliers"],
    )

    print(
        "Inlier ratio:",
        f"{r['inlier_ratio']:.6f}",
    )

    if (
        r[
            "ransac_reprojection_rmse_px"
        ]
        is not None
    ):
        print(
            "RANSAC self RMSE:",
            f"{r['ransac_reprojection_rmse_px']:.6f} px",
        )

        print(
            "Median residual:",
            f"{r['ransac_median_residual_px']:.6f} px",
        )

    c = r["coverage"]

    print(
        "Coverage:",
        f"{c['occupied_cells']}/"
        f"{c['valid_cells']}"
        f" = {c['coverage']:.6f}",
    )

    print(
        "Runtime:",
        f"{r['runtime_seconds']:.3f} s",
    )

    print(
        "Fine transform sane:",
        r[
            "fine_transform"
        ].get(
            "fine_sane",
            False,
        ),
    )

    if r["fine_affine"] is not None:
        print("Fine affine:")

        print(
            np.array(
                r["fine_affine"]
            )
        )

        d = r[
            "fine_transform"
        ]

        print(
            "Residual translation:",
            f"{d['translation_px']:.4f} px",
        )

        print(
            "Residual rotation:",
            f"{d['rotation_deg']:.6f} deg",
        )

        print(
            "Residual scale X/Y:",
            f"{d['scale_x']:.8f}",
            "/",
            f"{d['scale_y']:.8f}",
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print(
        "PAIR 005 — FINE MATCHING BENCHMARK"
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
    # LOAD
    # --------------------------------------------------------

    with rasterio.open(
        SOURCE_PATH
    ) as ds:
        source = (
            ds.read(1)
            .astype(np.float32)
        )

    with rasterio.open(
        REFERENCE_PATH
    ) as ds:
        reference = (
            ds.read(1)
            .astype(np.float32)
        )

    with rasterio.open(
        REGISTERED_COMMON_PATH
    ) as ds:
        registered_common = (
            ds.read(1) > 0
        )

    with rasterio.open(
        REFERENCE_MATCHER_MASK_PATH
    ) as ds:
        ref_matcher = (
            ds.read(1) > 0
        )

    fine_mask = build_fine_mask(
        registered_common,
        ref_matcher,
    )

    h, w = source.shape

    print(
        "\nShape:",
        h,
        "x",
        w,
    )

    print(
        "Registered common:",
        f"{int(registered_common.sum()):,}",
    )

    print(
        "Fine matcher mask:",
        f"{int(fine_mask.sum()):,}",
    )

    # --------------------------------------------------------
    # REPRESENTATIONS
    # --------------------------------------------------------

    src_int = robust_normalize(
        source,
        registered_common,
    )

    ref_int = robust_normalize(
        reference,
        registered_common,
    )

    src_grad = gradient_representation(
        src_int,
        fine_mask,
    )

    ref_grad = gradient_representation(
        ref_int,
        fine_mask,
    )

    reps = {
        "intensity":
            (src_int, ref_int),
        "gradient":
            (src_grad, ref_grad),
    }

    _, common_scale = resize_to_max(
        src_int,
        GLOBAL_MAX_DIM,
    )

    print(
        "\nGlobal max dimension:",
        GLOBAL_MAX_DIM,
    )

    print(
        "Global scale:",
        f"{common_scale:.8f}",
    )

    print(
        "Approx effective resolution:",
        f"{10.0 / common_scale:.3f} m/pixel",
    )

    results = []

    # ========================================================
    # SIFT + LIGHTGLUE FIRST
    # ========================================================

    print(
        "\nLoading SuperPoint + LightGlue..."
    )

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

    for rep_name, (
        src_full,
        ref_full,
    ) in reps.items():

        print("\n" + "=" * 78)
        print(
            "REPRESENTATION:",
            rep_name.upper(),
        )
        print("=" * 78)

        src_small, scale0 = (
            resize_to_max(
                src_full,
                GLOBAL_MAX_DIM,
            )
        )

        ref_small, scale1 = (
            resize_to_max(
                ref_full,
                GLOBAL_MAX_DIM,
            )
        )

        if abs(
            scale0 - scale1
        ) > 1e-9:
            raise RuntimeError(
                "Resize scales differ."
            )

        print(
            "Matching shape:",
            src_small.shape,
        )

        # SIFT
        print(
            "\nRunning SIFT..."
        )

        start = time.perf_counter()

        (
            p0,
            p1,
            k0,
            k1,
        ) = run_sift(
            src_small,
            ref_small,
            scale0,
        )

        result = evaluate(
            "SIFT",
            rep_name,
            p0,
            p1,
            fine_mask,
            src_full,
            ref_full,
            time.perf_counter()
            - start,
            extra={
                "source_keypoints": k0,
                "reference_keypoints": k1,
                "max_dimension":
                    GLOBAL_MAX_DIM,
            },
        )

        results.append(result)
        print_result(result)

        # LightGlue
        print(
            "\nRunning LightGlue..."
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        (
            p0,
            p1,
            k0,
            k1,
        ) = run_lightglue(
            src_small,
            ref_small,
            scale0,
            extractor,
            lightglue,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        result = evaluate(
            "LightGlue",
            rep_name,
            p0,
            p1,
            fine_mask,
            src_full,
            ref_full,
            time.perf_counter()
            - start,
            extra={
                "source_keypoints": k0,
                "reference_keypoints": k1,
                "max_dimension":
                    GLOBAL_MAX_DIM,
            },
        )

        results.append(result)
        print_result(result)

    # --------------------------------------------------------
    # FREE VRAM BEFORE LOFTR
    # --------------------------------------------------------

    print(
        "\nFreeing LightGlue models before LoFTR..."
    )

    del extractor
    del lightglue

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # LOFTR
    # ========================================================

    print(
        "Loading LoFTR..."
    )

    loftr = KF.LoFTR(
        pretrained="outdoor"
    ).eval().to(
        DEVICE
    )

    for rep_name, (
        src_full,
        ref_full,
    ) in reps.items():

        print("\n" + "=" * 78)

        print(
            "LOFTR:",
            rep_name.upper(),
        )

        print("=" * 78)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        (
            p0,
            p1,
            loftr_scale,
            loftr_shape,
            used_max_dim,
        ) = run_loftr_safe(
            src_full,
            ref_full,
            loftr,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        result = evaluate(
            "LoFTR",
            rep_name,
            p0,
            p1,
            fine_mask,
            src_full,
            ref_full,
            time.perf_counter()
            - start,
            extra={
                "max_dimension":
                    used_max_dim,
                "matching_shape":
                    list(
                        loftr_shape
                    ),
                "scale":
                    float(
                        loftr_scale
                    ),
            },
        )

        results.append(result)
        print_result(result)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = {
        "pair_id":
            "pair_005",
        "stage":
            "fine_global_benchmark",
        "source":
            "coarse-registered SELENE/Kaguya TC",
        "reference":
            "Chandrayaan-2 TMC-2",
        "canonical_resolution_m":
            10.0,
        "coarse_affine":
            COARSE_AFFINE.tolist(),
        "global_max_dimension":
            GLOBAL_MAX_DIM,
        "loftr_fallback_max_dimension":
            LOFTR_FALLBACK_MAX_DIM,
        "ransac_threshold_px":
            RANSAC_THRESHOLD_PX,
        "fine_mask_pixels":
            int(
                fine_mask.sum()
            ),
        "results":
            results,
        "metric_note":
            (
                "RANSAC reprojection RMSE is "
                "internal correspondence consistency, "
                "not independent absolute registration "
                "ground-truth accuracy."
            ),
    }

    json_path = (
        OUT_DIR
        / "pair005_fine_benchmark.json"
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
    print("=" * 90)
    print(
        "PAIR 005 FINE BENCHMARK SUMMARY"
    )
    print("=" * 90)

    print(
        f"{'Matcher':<12}"
        f"{'Rep':<12}"
        f"{'Cand':>8}"
        f"{'Inlier':>9}"
        f"{'Ratio':>10}"
        f"{'RMSE':>10}"
        f"{'Coverage':>13}"
        f"{'Shift':>10}"
        f"{'Sane':>8}"
    )

    print("-" * 92)

    for r in results:
        rmse = (
            r[
                "ransac_reprojection_rmse_px"
            ]
        )

        rmse_txt = (
            f"{rmse:.4f}"
            if rmse is not None
            else "N/A"
        )

        cov = r["coverage"]

        cov_txt = (
            f"{cov['occupied_cells']}/"
            f"{cov['valid_cells']}"
        )

        shift = (
            r[
                "fine_transform"
            ].get(
                "translation_px",
                None,
            )
        )

        shift_txt = (
            f"{shift:.3f}"
            if shift is not None
            else "N/A"
        )

        print(
            f"{r['matcher']:<12}"
            f"{r['representation']:<12}"
            f"{r['mask_filtered_candidates']:>8}"
            f"{r['ransac_inliers']:>9}"
            f"{r['inlier_ratio']:>10.4f}"
            f"{rmse_txt:>10}"
            f"{cov_txt:>13}"
            f"{shift_txt:>10}"
            f"{str(r['fine_transform'].get('fine_sane', False)):>8}"
        )

    print("\nSaved:")
    print(json_path)

    print("\n" + "=" * 78)
    print(
        "PAIR 005 FINE BENCHMARK COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()