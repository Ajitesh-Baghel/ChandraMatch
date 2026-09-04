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

COMMON_MASK_PATH = (
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
    / "fine_perturbation_validation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PARAMETERS
# ============================================================

MAX_DIM = 1280

SIFT_FEATURES = 10000
SUPERPOINT_FEATURES = 4096

RANSAC_THRESHOLD_PX = 3.0

MASK_EROSION_RADIUS = 4
BOUNDARY_MARGIN = 8

GRID_ROWS = 8
GRID_COLS = 8

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

    f = (
        image_u8.astype(np.float32)
        / 255.0
    )

    gx = cv2.Sobel(
        f,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        f,
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

    nw = max(
        1,
        int(round(w * scale)),
    )

    nh = max(
        1,
        int(round(h * scale)),
    )

    return (
        cv2.resize(
            image,
            (nw, nh),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def erode_mask(
    mask,
):

    k = (
        2 * MASK_EROSION_RADIUS
        + 1
    )

    kernel = np.ones(
        (k, k),
        dtype=np.uint8,
    )

    return (
        cv2.erode(
            mask.astype(np.uint8),
            kernel,
            iterations=1,
        )
        > 0
    )


# ============================================================
# AFFINE HELPERS
# ============================================================

def affine_to_h(
    M,
):

    H = np.eye(
        3,
        dtype=np.float64,
    )

    H[:2] = M

    return H


def h_to_affine(
    H,
):

    return (
        H[:2]
        .astype(np.float64)
    )


def compose(
    A,
    B,
):
    """
    Apply B first, then A.
    """

    return h_to_affine(
        affine_to_h(A)
        @ affine_to_h(B)
    )


def inverse_affine(
    M,
):

    return h_to_affine(
        np.linalg.inv(
            affine_to_h(M)
        )
    )


def transform_points(
    M,
    points,
):

    if len(points) == 0:

        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    return cv2.transform(
        points.astype(
            np.float64
        )[None],
        M.astype(
            np.float64
        ),
    )[0]


def affine_difference_rmse(
    A,
    B,
    points,
):

    if (
        A is None
        or B is None
        or len(points) == 0
    ):
        return None

    pa = transform_points(
        A,
        points,
    )

    pb = transform_points(
        B,
        points,
    )

    d = np.linalg.norm(
        pa - pb,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(
                d ** 2
            )
        )
    )


def describe_affine(
    M,
):

    if M is None:
        return None

    a, b, tx = M[0]
    c, d, ty = M[1]

    sx = math.sqrt(
        a * a
        + c * c
    )

    sy = math.sqrt(
        b * b
        + d * d
    )

    rotation = math.degrees(
        math.atan2(
            c,
            a,
        )
    )

    return {
        "tx_px": float(tx),
        "ty_px": float(ty),
        "translation_px":
            float(
                math.hypot(
                    tx,
                    ty,
                )
            ),
        "rotation_deg":
            float(rotation),
        "scale_x":
            float(sx),
        "scale_y":
            float(sy),
        "determinant":
            float(
                a * d
                - b * c
            ),
    }


# ============================================================
# KNOWN PERTURBATIONS
# ============================================================

def make_perturbations(
    width,
    height,
):

    center = (
        width / 2.0,
        height / 2.0,
    )

    # --------------------------------------------------------
    # 1. PURE TRANSLATION
    # --------------------------------------------------------

    shift1 = np.array(
        [
            [
                1.0,
                0.0,
                64.0,
            ],
            [
                0.0,
                1.0,
                -48.0,
            ],
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 2. SHIFT + ROTATION
    # --------------------------------------------------------

    shift_rot2 = (
        cv2.getRotationMatrix2D(
            center,
            0.40,
            1.0,
        )
        .astype(np.float64)
    )

    shift_rot2[
        0,
        2
    ] += -96.0

    shift_rot2[
        1,
        2
    ] += 72.0

    # --------------------------------------------------------
    # 3. SHIFT + ROTATION + SCALE
    # --------------------------------------------------------

    shift_rot_scale3 = (
        cv2.getRotationMatrix2D(
            center,
            -0.55,
            1.004,
        )
        .astype(np.float64)
    )

    shift_rot_scale3[
        0,
        2
    ] += 80.0

    shift_rot_scale3[
        1,
        2
    ] += -60.0

    return {
        "SHIFT1":
            shift1,

        "SHIFT_ROT2":
            shift_rot2,

        "SHIFT_ROT_SCALE3":
            shift_rot_scale3,
    }


# ============================================================
# PERTURB IMAGE
# ============================================================

def perturb_source(
    source,
    valid,
    P,
):

    h, w = source.shape

    clean = np.nan_to_num(
        source,
        nan=0.0,
    ).astype(np.float32)

    warped = cv2.warpAffine(
        clean,
        P,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_valid = (
        cv2.warpAffine(
            valid.astype(np.uint8),
            P,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )

    warped[
        ~warped_valid
    ] = np.nan

    return (
        warped,
        warped_valid,
    )


# ============================================================
# POINT MASK FILTER
# ============================================================

def filter_matches(
    source_points,
    reference_points,
    valid_mask,
):

    if len(source_points) == 0:

        return (
            source_points,
            reference_points,
        )

    h, w = valid_mask.shape

    sx = np.rint(
        source_points[:, 0]
    ).astype(np.int64)

    sy = np.rint(
        source_points[:, 1]
    ).astype(np.int64)

    rx = np.rint(
        reference_points[:, 0]
    ).astype(np.int64)

    ry = np.rint(
        reference_points[:, 1]
    ).astype(np.int64)

    inside = (
        (sx >= BOUNDARY_MARGIN)
        & (sx < w - BOUNDARY_MARGIN)
        & (sy >= BOUNDARY_MARGIN)
        & (sy < h - BOUNDARY_MARGIN)
        & (rx >= BOUNDARY_MARGIN)
        & (rx < w - BOUNDARY_MARGIN)
        & (ry >= BOUNDARY_MARGIN)
        & (ry < h - BOUNDARY_MARGIN)
    )

    ids = np.where(
        inside
    )[0]

    keep = np.zeros(
        len(source_points),
        dtype=bool,
    )

    if len(ids) > 0:

        keep[ids] = (
            valid_mask[
                sy[ids],
                sx[ids],
            ]
            & valid_mask[
                ry[ids],
                rx[ids],
            ]
        )

    return (
        source_points[keep],
        reference_points[keep],
    )


# ============================================================
# SIFT
# ============================================================

def run_sift(
    source,
    reference,
):

    src_small, scale0 = (
        resize_to_max(
            source,
            MAX_DIM,
        )
    )

    ref_small, scale1 = (
        resize_to_max(
            reference,
            MAX_DIM,
        )
    )

    if abs(
        scale0 - scale1
    ) > 1e-9:

        raise RuntimeError(
            "SIFT scale mismatch."
        )

    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
    )

    k0, d0 = sift.detectAndCompute(
        src_small,
        None,
    )

    k1, d1 = sift.detectAndCompute(
        ref_small,
        None,
    )

    if (
        d0 is None
        or d1 is None
    ):

        return (
            np.empty(
                (0, 2),
                np.float32,
            ),
            np.empty(
                (0, 2),
                np.float32,
            ),
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

        if (
            m.distance
            < 0.75
            * n.distance
        ):
            good.append(m)

    p0 = np.array(
        [
            k0[
                m.queryIdx
            ].pt
            for m in good
        ],
        dtype=np.float32,
    )

    p1 = np.array(
        [
            k1[
                m.trainIdx
            ].pt
            for m in good
        ],
        dtype=np.float32,
    )

    if len(p0) > 0:

        p0 /= scale0
        p1 /= scale0

    return (
        p0,
        p1,
    )


# ============================================================
# LIGHTGLUE
# ============================================================

def run_lightglue(
    source,
    reference,
    extractor,
    matcher,
):

    src_small, scale0 = (
        resize_to_max(
            source,
            MAX_DIM,
        )
    )

    ref_small, scale1 = (
        resize_to_max(
            reference,
            MAX_DIM,
        )
    )

    if abs(
        scale0 - scale1
    ) > 1e-9:

        raise RuntimeError(
            "LightGlue scale mismatch."
        )

    t0 = (
        torch.from_numpy(
            src_small
        )
        .float()[None, None]
        .to(DEVICE)
        / 255.0
    )

    t1 = (
        torch.from_numpy(
            ref_small
        )
        .float()[None, None]
        .to(DEVICE)
        / 255.0
    )

    with torch.inference_mode():

        f0 = extractor.extract(
            t0
        )

        f1 = extractor.extract(
            t1
        )

        matches = matcher(
            {
                "image0": f0,
                "image1": f1,
            }
        )

        f0 = rbd(f0)
        f1 = rbd(f1)
        matches = rbd(
            matches
        )

    pairs = matches[
        "matches"
    ]

    if pairs.numel() == 0:

        return (
            np.empty(
                (0, 2),
                np.float32,
            ),
            np.empty(
                (0, 2),
                np.float32,
            ),
        )

    p0 = (
        f0[
            "keypoints"
        ][
            pairs[:, 0]
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    p1 = (
        f1[
            "keypoints"
        ][
            pairs[:, 1]
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    p0 /= scale0
    p1 /= scale0

    return (
        p0,
        p1,
    )


# ============================================================
# LOFTR
# ============================================================

def run_loftr(
    source,
    reference,
    matcher,
):

    src_small, scale0 = (
        resize_to_max(
            source,
            MAX_DIM,
        )
    )

    ref_small, scale1 = (
        resize_to_max(
            reference,
            MAX_DIM,
        )
    )

    if abs(
        scale0 - scale1
    ) > 1e-9:

        raise RuntimeError(
            "LoFTR scale mismatch."
        )

    t0 = (
        torch.from_numpy(
            src_small
        )
        .float()[None, None]
        .to(DEVICE)
        / 255.0
    )

    t1 = (
        torch.from_numpy(
            ref_small
        )
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
        out[
            "keypoints0"
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    p1 = (
        out[
            "keypoints1"
        ]
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
    )


# ============================================================
# RANSAC
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

    M, inliers = cv2.estimateAffine2D(
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
        M is None
        or inliers is None
    ):

        return (
            None,
            np.zeros(
                len(source_points),
                dtype=bool,
            ),
        )

    return (
        M.astype(
            np.float64
        ),
        inliers.ravel().astype(bool),
    )


def self_rmse(
    M,
    source_points,
    reference_points,
    inliers,
):

    if (
        M is None
        or int(inliers.sum()) == 0
    ):

        return None

    pred = transform_points(
        M,
        source_points[
            inliers
        ],
    )

    truth = (
        reference_points[
            inliers
        ]
    )

    errors = np.linalg.norm(
        pred - truth,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )


# ============================================================
# COVERAGE
# ============================================================

def compute_coverage(
    points,
    valid_mask,
):

    h, w = valid_mask.shape

    valid_cells = set()
    occupied = set()

    for gy in range(
        GRID_ROWS
    ):

        y0 = int(
            gy
            * h
            / GRID_ROWS
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
                gx
                * w
                / GRID_COLS
            )

            x1 = int(
                (gx + 1)
                * w
                / GRID_COLS
            )

            cell = valid_mask[
                y0:y1,
                x0:x1,
            ]

            if (
                cell.size > 0
                and cell.mean()
                >= 0.01
            ):

                valid_cells.add(
                    (
                        gy,
                        gx,
                    )
                )

    for x, y in points:

        gx = int(
            x
            / w
            * GRID_COLS
        )

        gy = int(
            y
            / h
            * GRID_ROWS
        )

        gx = int(
            np.clip(
                gx,
                0,
                GRID_COLS - 1,
            )
        )

        gy = int(
            np.clip(
                gy,
                0,
                GRID_ROWS - 1,
            )
        )

        if (
            gy,
            gx,
        ) in valid_cells:

            occupied.add(
                (
                    gy,
                    gx,
                )
            )

    n_valid = len(
        valid_cells
    )

    n_occ = len(
        occupied
    )

    return {
        "occupied_cells":
            n_occ,

        "valid_cells":
            n_valid,

        "coverage":
            (
                n_occ
                / n_valid
                if n_valid
                else 0.0
            ),
    }


# ============================================================
# SAMPLE POINTS FOR TRANSFORM COMPARISON
# ============================================================

def sample_mask_points(
    mask,
    max_points=5000,
):

    ys, xs = np.where(
        mask
    )

    if len(xs) == 0:

        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    if len(xs) > max_points:

        ids = np.linspace(
            0,
            len(xs) - 1,
            max_points,
        ).astype(np.int64)

        xs = xs[ids]
        ys = ys[ids]

    return np.column_stack(
        [
            xs,
            ys,
        ]
    ).astype(np.float64)


# ============================================================
# ONE MATCHING RUN
# ============================================================

def evaluate_run(
    matcher_name,
    representation,
    source_repr,
    reference_repr,
    valid_mask,
    run_function,
):

    start = time.perf_counter()

    p0, p1 = run_function(
        source_repr,
        reference_repr,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    raw_count = len(
        p0
    )

    p0, p1 = filter_matches(
        p0,
        p1,
        valid_mask,
    )

    M, inliers = fit_affine(
        p0,
        p1,
    )

    n_candidates = len(
        p0
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

    rmse = self_rmse(
        M,
        p0,
        p1,
        inliers,
    )

    if n_inliers:

        coverage = compute_coverage(
            p1[inliers],
            valid_mask,
        )

    else:

        coverage = {
            "occupied_cells": 0,
            "valid_cells": 0,
            "coverage": 0.0,
        }

    return {
        "matcher":
            matcher_name,

        "representation":
            representation,

        "raw_matches":
            int(raw_count),

        "candidates":
            int(n_candidates),

        "inliers":
            int(n_inliers),

        "inlier_ratio":
            float(ratio),

        "self_rmse_px":
            rmse,

        "coverage":
            coverage,

        "affine":
            (
                M.tolist()
                if M is not None
                else None
            ),

        "transform":
            describe_affine(M),

        "runtime_seconds":
            float(runtime),

        "_M":
            M,
    }


# ============================================================
# VALIDATE ONE MATCHER
# ============================================================

def validate_matcher(
    matcher_name,
    representation,
    source,
    reference,
    base_common,
    reference_matcher,
    run_function,
):

    print("\n")
    print("=" * 78)
    print(
        matcher_name,
        "/",
        representation.upper(),
    )
    print("=" * 78)

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    base_mask = erode_mask(
        base_common
        & reference_matcher
    )

    source_intensity = robust_normalize(
        source,
        base_common,
    )

    reference_intensity = robust_normalize(
        reference,
        base_common,
    )

    if representation == "gradient":

        source_repr = gradient_representation(
            source_intensity,
            base_mask,
        )

        reference_repr = gradient_representation(
            reference_intensity,
            base_mask,
        )

    else:

        source_repr = (
            source_intensity
        )

        reference_repr = (
            reference_intensity
        )

    baseline = evaluate_run(
        matcher_name,
        representation,
        source_repr,
        reference_repr,
        base_mask,
        run_function,
    )

    M_base = baseline[
        "_M"
    ]

    if M_base is None:

        raise RuntimeError(
            matcher_name
            + " baseline failed."
        )

    print("\nBASELINE")

    print(
        "Candidates:",
        baseline[
            "candidates"
        ],
    )

    print(
        "Inliers:",
        baseline[
            "inliers"
        ],
    )

    print(
        "Self RMSE:",
        baseline[
            "self_rmse_px"
        ],
    )

    print(
        "Coverage:",
        baseline[
            "coverage"
        ],
    )

    print(
        "Affine:"
    )

    print(
        M_base
    )

    perturbations = (
        make_perturbations(
            source.shape[1],
            source.shape[0],
        )
    )

    tests = []

    base_sample_points = (
        sample_mask_points(
            base_mask
        )
    )

    for (
        test_name,
        P,
    ) in perturbations.items():

        print("\n" + "-" * 78)
        print(test_name)
        print("-" * 78)

        pert_source, pert_valid = (
            perturb_source(
                source,
                base_common,
                P,
            )
        )

        validation_mask = erode_mask(
            pert_valid
            & reference_matcher
        )

        pert_intensity = robust_normalize(
            pert_source,
            pert_valid,
        )

        # Reference normalization over the same
        # physical region used in this experiment.
        ref_intensity = robust_normalize(
            reference,
            validation_mask,
        )

        if representation == "gradient":

            pert_repr = (
                gradient_representation(
                    pert_intensity,
                    validation_mask,
                )
            )

            ref_repr = (
                gradient_representation(
                    ref_intensity,
                    validation_mask,
                )
            )

        else:

            pert_repr = pert_intensity
            ref_repr = ref_intensity

        result = evaluate_run(
            matcher_name,
            representation,
            pert_repr,
            ref_repr,
            validation_mask,
            run_function,
        )

        M_pert = result[
            "_M"
        ]

        # ----------------------------------------------------
        # EXPECTED TRANSFORM
        #
        # Baseline:
        #
        #     x_ref = M_base * x_source
        #
        # Perturbed source:
        #
        #     x_pert = P * x_source
        #
        # Therefore:
        #
        #     x_ref =
        #       M_base * inv(P) * x_pert
        #
        # ----------------------------------------------------

        expected = compose(
            M_base,
            inverse_affine(P),
        )

        if M_pert is not None:

            compensated = compose(
                M_pert,
                P,
            )

            pert_sample_points = (
                sample_mask_points(
                    validation_mask
                )
            )

            response_rmse = (
                affine_difference_rmse(
                    M_pert,
                    expected,
                    pert_sample_points,
                )
            )

            compensated_rmse = (
                affine_difference_rmse(
                    compensated,
                    M_base,
                    base_sample_points,
                )
            )

            identity = np.array(
                [
                    [
                        1.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        1.0,
                        0.0,
                    ],
                ],
                dtype=np.float64,
            )

            identity_error = (
                affine_difference_rmse(
                    M_pert,
                    identity,
                    pert_sample_points,
                )
            )

        else:

            compensated = None
            response_rmse = None
            compensated_rmse = None
            identity_error = None

        passed = (
            response_rmse is not None
            and response_rmse <= 3.0
            and result[
                "inliers"
            ] >= 30
            and result[
                "coverage"
            ][
                "coverage"
            ] >= 0.50
        )

        print(
            "Known perturbation:"
        )

        print(P)

        print(
            "\nExpected perturbed-source → reference:"
        )

        print(expected)

        print(
            "\nRecovered:"
        )

        print(M_pert)

        print(
            "\nCandidates:",
            result[
                "candidates"
            ],
        )

        print(
            "Inliers:",
            result[
                "inliers"
            ],
        )

        print(
            "Inlier ratio:",
            f"{result['inlier_ratio']:.6f}",
        )

        print(
            "Self RMSE:",
            result[
                "self_rmse_px"
            ],
        )

        print(
            "Coverage:",
            result[
                "coverage"
            ],
        )

        print(
            "\nPERTURBATION RESPONSE RMSE:",
            response_rmse,
            "px",
        )

        print(
            "COMPENSATED BASELINE RMSE:",
            compensated_rmse,
            "px",
        )

        print(
            "DISTANCE FROM IDENTITY:",
            identity_error,
            "px",
        )

        print(
            "PASS:",
            passed,
        )

        clean_result = {
            k: v
            for k, v
            in result.items()
            if k != "_M"
        }

        clean_result.update(
            {
                "test":
                    test_name,

                "known_perturbation":
                    P.tolist(),

                "expected_affine":
                    expected.tolist(),

                "recovered_affine":
                    (
                        M_pert.tolist()
                        if M_pert is not None
                        else None
                    ),

                "compensated_affine":
                    (
                        compensated.tolist()
                        if compensated
                        is not None
                        else None
                    ),

                "perturbation_response_rmse_px":
                    response_rmse,

                "compensated_baseline_rmse_px":
                    compensated_rmse,

                "distance_from_identity_px":
                    identity_error,

                "pass":
                    bool(passed),
            }
        )

        tests.append(
            clean_result
        )

    clean_baseline = {
        k: v
        for k, v
        in baseline.items()
        if k != "_M"
    }

    response_values = [
        t[
            "perturbation_response_rmse_px"
        ]
        for t in tests
        if t[
            "perturbation_response_rmse_px"
        ]
        is not None
    ]

    return {
        "matcher":
            matcher_name,

        "representation":
            representation,

        "baseline":
            clean_baseline,

        "tests":
            tests,

        "all_tests_pass":
            all(
                t["pass"]
                for t in tests
            ),

        "mean_response_rmse_px":
            (
                float(
                    np.mean(
                        response_values
                    )
                )
                if response_values
                else None
            ),

        "worst_response_rmse_px":
            (
                float(
                    np.max(
                        response_values
                    )
                )
                if response_values
                else None
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "PAIR 005 — CROSS-SENSOR "
        "FINE PERTURBATION VALIDATION"
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
    # LOAD DATA
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
        COMMON_MASK_PATH
    ) as ds:

        base_common = (
            ds.read(1)
            > 0
        )

    with rasterio.open(
        REFERENCE_MATCHER_MASK_PATH
    ) as ds:

        reference_matcher = (
            ds.read(1)
            > 0
        )

    print(
        "\nShape:",
        source.shape,
    )

    print(
        "Base common:",
        f"{int(base_common.sum()):,}",
    )

    results = []

    # ========================================================
    # SIFT GRADIENT
    # ========================================================

    def sift_runner(
        a,
        b,
    ):
        return run_sift(
            a,
            b,
        )

    results.append(
        validate_matcher(
            "SIFT",
            "gradient",
            source,
            reference,
            base_common,
            reference_matcher,
            sift_runner,
        )
    )

    # ========================================================
    # LIGHTGLUE GRADIENT
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

    def lightglue_runner(
        a,
        b,
    ):
        return run_lightglue(
            a,
            b,
            extractor,
            lightglue,
        )

    results.append(
        validate_matcher(
            "LightGlue",
            "gradient",
            source,
            reference,
            base_common,
            reference_matcher,
            lightglue_runner,
        )
    )

    del extractor
    del lightglue

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # LOFTR INTENSITY
    # ========================================================

    print(
        "\nLoading LoFTR..."
    )

    loftr = KF.LoFTR(
        pretrained="outdoor"
    ).eval().to(
        DEVICE
    )

    def loftr_runner(
        a,
        b,
    ):
        return run_loftr(
            a,
            b,
            loftr,
        )

    results.append(
        validate_matcher(
            "LoFTR",
            "intensity",
            source,
            reference,
            base_common,
            reference_matcher,
            loftr_runner,
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = {
        "pair_id":
            "pair_005",

        "stage":
            "fine_cross_sensor_perturbation_validation",

        "source":
            "coarse-registered SELENE/Kaguya TC",

        "reference":
            "Chandrayaan-2 TMC-2",

        "canonical_resolution_m":
            10.0,

        "max_matching_dimension":
            MAX_DIM,

        "pass_rule":
            {
                "maximum_response_rmse_px":
                    3.0,

                "minimum_inliers":
                    30,

                "minimum_spatial_coverage":
                    0.50,
            },

        "results":
            results,

        "interpretation_note":
            (
                "Perturbation-response error measures whether "
                "the matcher follows known injected motion in "
                "the real cross-sensor pair. It is stronger "
                "robustness evidence than RANSAC self residual, "
                "but it is not independent absolute ground-truth "
                "accuracy for the unperturbed real registration."
            ),
    }

    output_path = (
        OUT_DIR
        / "pair005_fine_perturbation_validation.json"
    )

    with open(
        output_path,
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
        "PAIR 005 PERTURBATION VALIDATION SUMMARY"
    )
    print("=" * 78)

    for result in results:

        print(
            "\n",
            result[
                "matcher"
            ],
            "/",
            result[
                "representation"
            ],
        )

        print(
            "All tests pass:",
            result[
                "all_tests_pass"
            ],
        )

        print(
            "Mean response RMSE:",
            result[
                "mean_response_rmse_px"
            ],
        )

        print(
            "Worst response RMSE:",
            result[
                "worst_response_rmse_px"
            ],
        )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print("\n" + "=" * 78)
    print(
        "PAIR 005 VALIDATION COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()