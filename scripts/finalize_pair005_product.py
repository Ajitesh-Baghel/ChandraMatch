from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np
import rasterio
from PIL import Image


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

OUT_DIR = (
    ROOT
    / "results"
    / "pair_005"
    / "final_product"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ORIGINAL_SOURCE_PATH = (
    CANONICAL_DIR
    / "kaguya_tc_source.tif"
)

COARSE_SOURCE_PATH = (
    COARSE_DIR
    / "kaguya_registered_coarse_to_tmc2.tif"
)

REFERENCE_PATH = (
    CANONICAL_DIR
    / "tmc2_reference.tif"
)

COMMON_PATH = (
    COARSE_DIR
    / "registered_common_valid_mask.tif"
)

MATCHER_PATH = (
    CANONICAL_DIR
    / "matcher_valid_mask.tif"
)


# ============================================================
# FINAL TRANSFORMS
# ============================================================

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


FINE_AFFINE = np.array(
    [
        [
            9.99970995e-01,
            9.52740430e-05,
            -7.46950517e-02,
        ],
        [
            1.56173491e-05,
            1.00002608e00,
            -8.25130637e-02,
        ],
    ],
    dtype=np.float64,
)


# Fine benchmark parameters
MAX_DIM = 1280
SIFT_FEATURES = 10000
RANSAC_THRESHOLD = 3.0

GRID_ROWS = 8
GRID_COLS = 8

UNIFORM_PER_CELL = 4


# ============================================================
# AFFINE HELPERS
# ============================================================

def affine_to_h(M):

    H = np.eye(
        3,
        dtype=np.float64,
    )

    H[:2] = M

    return H


def compose_affines(A, B):
    """
    Apply B first, then A.
    """

    H = (
        affine_to_h(A)
        @ affine_to_h(B)
    )

    return H[:2]


def inverse_affine(M):

    H = np.linalg.inv(
        affine_to_h(M)
    )

    return H[:2]


def transform_points(M, points):

    if len(points) == 0:

        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    return cv2.transform(
        points.astype(
            np.float64
        )[None],
        M,
    )[0]


FINAL_AFFINE = compose_affines(
    FINE_AFFINE,
    COARSE_AFFINE,
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

    result = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    valid = (
        mask
        & np.isfinite(image)
    )

    if not np.any(valid):
        return result

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

    result[valid] = (
        scaled[valid]
        * 255.0
    ).astype(np.uint8)

    return result


def gradient_representation(
    image,
    mask,
):

    f = (
        image.astype(np.float32)
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

    result = np.zeros_like(
        image
    )

    valid = (
        mask
        & np.isfinite(mag)
    )

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

    norm = (
        mag - lo
    ) / (
        hi - lo
    )

    norm = np.clip(
        norm,
        0.0,
        1.0,
    )

    result[valid] = (
        norm[valid]
        * 255
    ).astype(np.uint8)

    return result


def resize_to_max(
    image,
    max_dim,
):

    h, w = image.shape

    scale = min(
        1.0,
        max_dim / max(h, w),
    )

    if scale >= 1:

        return (
            image.copy(),
            1.0,
        )

    nw = int(
        round(w * scale)
    )

    nh = int(
        round(h * scale)
    )

    return (
        cv2.resize(
            image,
            (nw, nh),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def resize_preview(
    image,
    max_dim=2048,
):

    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_dim / max(h, w),
    )

    if scale >= 1:
        return image

    return cv2.resize(
        image,
        (
            int(round(w * scale)),
            int(round(h * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def save_gray(
    path,
    image,
):

    Image.fromarray(
        resize_preview(image)
    ).save(path)


def save_rgb(
    path,
    image,
):

    Image.fromarray(
        resize_preview(image)
    ).save(path)


# ============================================================
# REPRODUCE FINAL SIFT-GRADIENT CORRESPONDENCES
# ============================================================

def run_final_sift(
    source,
    reference,
    matcher_mask,
):

    src_int = robust_normalize(
        source,
        matcher_mask,
    )

    ref_int = robust_normalize(
        reference,
        matcher_mask,
    )

    src_grad = gradient_representation(
        src_int,
        matcher_mask,
    )

    ref_grad = gradient_representation(
        ref_int,
        matcher_mask,
    )

    src_small, scale0 = resize_to_max(
        src_grad,
        MAX_DIM,
    )

    ref_small, scale1 = resize_to_max(
        ref_grad,
        MAX_DIM,
    )

    if abs(
        scale0 - scale1
    ) > 1e-9:

        raise RuntimeError(
            "Source/reference resize mismatch."
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

    p0 /= scale0
    p1 /= scale0

    return (
        p0,
        p1,
        src_grad,
        ref_grad,
    )


def filter_by_mask(
    p0,
    p1,
    mask,
):

    h, w = mask.shape

    x0 = np.rint(
        p0[:, 0]
    ).astype(int)

    y0 = np.rint(
        p0[:, 1]
    ).astype(int)

    x1 = np.rint(
        p1[:, 0]
    ).astype(int)

    y1 = np.rint(
        p1[:, 1]
    ).astype(int)

    inside = (
        (x0 >= 8)
        & (x0 < w - 8)
        & (y0 >= 8)
        & (y0 < h - 8)
        & (x1 >= 8)
        & (x1 < w - 8)
        & (y1 >= 8)
        & (y1 < h - 8)
    )

    ids = np.where(
        inside
    )[0]

    keep = np.zeros(
        len(p0),
        dtype=bool,
    )

    keep[ids] = (
        mask[
            y0[ids],
            x0[ids],
        ]
        & mask[
            y1[ids],
            x1[ids],
        ]
    )

    return (
        p0[keep],
        p1[keep],
    )


# ============================================================
# RANSAC
# ============================================================

def fit_affine(
    p0,
    p1,
):

    M, inlier_mask = (
        cv2.estimateAffine2D(
            p0,
            p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=(
                RANSAC_THRESHOLD
            ),
            maxIters=10000,
            confidence=0.999,
            refineIters=10,
        )
    )

    if (
        M is None
        or inlier_mask is None
    ):

        raise RuntimeError(
            "Final SIFT affine failed."
        )

    return (
        M.astype(np.float64),
        inlier_mask.ravel().astype(bool),
    )


def calculate_residuals(
    M,
    p0,
    p1,
):

    pred = transform_points(
        M,
        p0,
    )

    return np.linalg.norm(
        pred - p1,
        axis=1,
    )


# ============================================================
# UNIFORM SUBSET
# ============================================================

def select_uniform(
    source_points,
    reference_points,
    residuals,
    mask,
):

    h, w = mask.shape

    selected = []

    for gy in range(
        GRID_ROWS
    ):

        y0 = int(
            gy * h / GRID_ROWS
        )

        y1 = int(
            (gy + 1)
            * h / GRID_ROWS
        )

        for gx in range(
            GRID_COLS
        ):

            x0 = int(
                gx * w / GRID_COLS
            )

            x1 = int(
                (gx + 1)
                * w / GRID_COLS
            )

            valid_fraction = (
                mask[
                    y0:y1,
                    x0:x1,
                ].mean()
            )

            if valid_fraction < 0.01:
                continue

            ids = np.where(
                (reference_points[:, 0] >= x0)
                & (reference_points[:, 0] < x1)
                & (reference_points[:, 1] >= y0)
                & (reference_points[:, 1] < y1)
            )[0]

            if len(ids) == 0:
                continue

            # Lowest residual matches from each cell.
            order = ids[
                np.argsort(
                    residuals[ids]
                )
            ]

            selected.extend(
                order[
                    :UNIFORM_PER_CELL
                ].tolist()
            )

    return np.array(
        selected,
        dtype=np.int64,
    )


# ============================================================
# FILE WRITERS
# ============================================================

def write_registered(
    path,
    data,
    profile,
):

    profile = profile.copy()

    profile.update(
        {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
        }
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            data.astype(np.float32),
            1,
        )


def write_csv(
    path,
    original_source,
    coarse_source,
    reference,
    residuals,
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "source_x_original_kaguya_px",
                "source_y_original_kaguya_px",
                "source_x_coarse_registered_px",
                "source_y_coarse_registered_px",
                "reference_x_tmc_px",
                "reference_y_tmc_px",
                "ransac_residual_px",
            ]
        )

        for (
            orig,
            coarse,
            ref,
            residual,
        ) in zip(
            original_source,
            coarse_source,
            reference,
            residuals,
        ):

            writer.writerow(
                [
                    float(orig[0]),
                    float(orig[1]),
                    float(coarse[0]),
                    float(coarse[1]),
                    float(ref[0]),
                    float(ref[1]),
                    float(residual),
                ]
            )


# ============================================================
# VISUALIZATION
# ============================================================

def checkerboard(
    source,
    reference,
    valid,
    block=256,
):

    yy, xx = np.indices(
        source.shape
    )

    choose_source = (
        (
            xx // block
            + yy // block
        )
        % 2 == 0
    )

    result = np.where(
        choose_source,
        source,
        reference,
    )

    result[
        ~valid
    ] = 0

    return result.astype(
        np.uint8
    )


def draw_uniform_matches(
    path,
    source,
    reference,
    p0,
    p1,
):

    h, w = source.shape

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

    left = cv2.resize(
        source,
        (pw, ph),
        interpolation=cv2.INTER_AREA,
    )

    right = cv2.resize(
        reference,
        (pw, ph),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.concatenate(
        [
            cv2.cvtColor(
                left,
                cv2.COLOR_GRAY2BGR,
            ),
            cv2.cvtColor(
                right,
                cv2.COLOR_GRAY2BGR,
            ),
        ],
        axis=1,
    )

    for a, b in zip(
        p0,
        p1,
    ):

        a = (
            int(round(a[0] * preview_scale)),
            int(round(a[1] * preview_scale)),
        )

        b = (
            int(
                round(
                    b[0]
                    * preview_scale
                    + pw
                )
            ),
            int(round(b[1] * preview_scale)),
        )

        cv2.line(
            canvas,
            a,
            b,
            (0, 255, 0),
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
        "PAIR 005 — FINAL PRODUCT"
    )
    print("=" * 78)

    print(
        "\nCoarse affine:"
    )
    print(
        COARSE_AFFINE
    )

    print(
        "\nFine SIFT-gradient affine:"
    )
    print(
        FINE_AFFINE
    )

    print(
        "\nComposed original Kaguya → TMC affine:"
    )
    print(
        FINAL_AFFINE
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    with rasterio.open(
        ORIGINAL_SOURCE_PATH
    ) as ds:

        original_source = (
            ds.read(1)
            .astype(np.float32)
        )

    with rasterio.open(
        COARSE_SOURCE_PATH
    ) as ds:

        coarse_source = (
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

        reference_profile = (
            ds.profile.copy()
        )

    with rasterio.open(
        COMMON_PATH
    ) as ds:

        common = (
            ds.read(1) > 0
        )

    with rasterio.open(
        MATCHER_PATH
    ) as ds:

        matcher = (
            ds.read(1) > 0
        )

    fine_mask = (
        common
        & matcher
    )

    # A small erosion matching the validation stage.
    kernel = np.ones(
        (9, 9),
        dtype=np.uint8,
    )

    fine_mask = (
        cv2.erode(
            fine_mask.astype(np.uint8),
            kernel,
            iterations=1,
        )
        > 0
    )

    print(
        "\nFine valid mask:",
        f"{int(fine_mask.sum()):,}",
    )

    # --------------------------------------------------------
    # RE-RUN FINAL SIFT
    # --------------------------------------------------------

    (
        p0,
        p1,
        src_grad,
        ref_grad,
    ) = run_final_sift(
        coarse_source,
        reference,
        fine_mask,
    )

    print(
        "Raw SIFT gradient matches:",
        len(p0),
    )

    p0, p1 = filter_by_mask(
        p0,
        p1,
        fine_mask,
    )

    print(
        "Mask-filtered candidates:",
        len(p0),
    )

    M_check, inliers = fit_affine(
        p0,
        p1,
    )

    p0_in = p0[inliers]
    p1_in = p1[inliers]

    residual = calculate_residuals(
        M_check,
        p0_in,
        p1_in,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    print(
        "RANSAC inliers:",
        len(p0_in),
    )

    print(
        "Inlier ratio:",
        len(p0_in)
        / len(p0),
    )

    print(
        "Self RMSE:",
        rmse,
    )

    print(
        "Recovered fine affine:"
    )

    print(
        M_check
    )

    # --------------------------------------------------------
    # USE REPRODUCED FINE AFFINE
    #
    # This avoids relying on manually pasted rounded values.
    # --------------------------------------------------------

    final_affine = compose_affines(
        M_check,
        COARSE_AFFINE,
    )

    print(
        "\nFinal reproduced composed affine:"
    )

    print(
        final_affine
    )

    # --------------------------------------------------------
    # FINAL WARP — ORIGINAL KAGUYA → TMC
    # --------------------------------------------------------

    h, w = reference.shape

    source_clean = np.nan_to_num(
        original_source,
        nan=0.0,
    ).astype(np.float32)

    final_registered = cv2.warpAffine(
        source_clean,
        final_affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    original_valid = (
        np.isfinite(
            original_source
        )
    )

    warped_valid = (
        cv2.warpAffine(
            original_valid.astype(
                np.uint8
            ),
            final_affine,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )

    reference_valid = (
        np.isfinite(
            reference
        )
    )

    final_common = (
        warped_valid
        & reference_valid
    )

    final_registered[
        ~warped_valid
    ] = np.nan

    print(
        "\nFinal registered common pixels:",
        f"{int(final_common.sum()):,}",
    )

    # --------------------------------------------------------
    # ORIGINAL COORDINATES FOR MATCH CSV
    # --------------------------------------------------------

    coarse_inverse = inverse_affine(
        COARSE_AFFINE
    )

    original_points = transform_points(
        coarse_inverse,
        p0_in,
    )

    # --------------------------------------------------------
    # UNIFORM SUBSET
    # --------------------------------------------------------

    uniform_ids = select_uniform(
        p0_in,
        p1_in,
        residual,
        fine_mask,
    )

    uniform_p0 = (
        p0_in[
            uniform_ids
        ]
    )

    uniform_p1 = (
        p1_in[
            uniform_ids
        ]
    )

    uniform_original = (
        original_points[
            uniform_ids
        ]
    )

    uniform_residual = (
        residual[
            uniform_ids
        ]
    )

    print(
        "Uniform final points:",
        len(uniform_ids),
    )

    # --------------------------------------------------------
    # WRITE RASTER
    # --------------------------------------------------------

    registered_path = (
        OUT_DIR
        / "registered_kaguya_tc_to_tmc2.tif"
    )

    write_registered(
        registered_path,
        final_registered,
        reference_profile,
    )

    # --------------------------------------------------------
    # WRITE CSVs
    # --------------------------------------------------------

    all_csv = (
        OUT_DIR
        / "pair005_all_match_points.csv"
    )

    uniform_csv = (
        OUT_DIR
        / "pair005_uniform_match_points.csv"
    )

    write_csv(
        all_csv,
        original_points,
        p0_in,
        p1_in,
        residual,
    )

    write_csv(
        uniform_csv,
        uniform_original,
        uniform_p0,
        uniform_p1,
        uniform_residual,
    )

    # --------------------------------------------------------
    # VISUAL PRODUCTS
    # --------------------------------------------------------

    registered_vis = robust_normalize(
        final_registered,
        final_common,
    )

    reference_vis = robust_normalize(
        reference,
        final_common,
    )

    overlay = (
        0.5
        * registered_vis.astype(np.float32)
        + 0.5
        * reference_vis.astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    overlay[
        ~final_common
    ] = 0

    check = checkerboard(
        registered_vis,
        reference_vis,
        final_common,
    )

    rgb = np.zeros(
        (
            h,
            w,
            3,
        ),
        dtype=np.uint8,
    )

    rgb[..., 0] = reference_vis
    rgb[..., 1] = registered_vis
    rgb[..., 2] = registered_vis

    rgb[
        ~final_common
    ] = 0

    save_gray(
        OUT_DIR
        / "registered_kaguya_preview.png",
        registered_vis,
    )

    save_gray(
        OUT_DIR
        / "tmc2_reference_preview.png",
        reference_vis,
    )

    save_gray(
        OUT_DIR
        / "registered_overlay_50_50.png",
        overlay,
    )

    save_gray(
        OUT_DIR
        / "registered_checkerboard.png",
        check,
    )

    save_rgb(
        OUT_DIR
        / "registered_red_cyan.png",
        rgb,
    )

    draw_uniform_matches(
        OUT_DIR
        / "pair005_final_matches.png",
        src_grad,
        ref_grad,
        uniform_p0,
        uniform_p1,
    )

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    a, b, tx = final_affine[0]
    c, d, ty = final_affine[1]

    total_transform = {
        "tx_px":
            float(tx),

        "ty_px":
            float(ty),

        "translation_px":
            float(
                math.hypot(
                    tx,
                    ty,
                )
            ),

        "approx_translation_m":
            float(
                math.hypot(
                    tx,
                    ty,
                )
                * 10.0
            ),

        "rotation_deg":
            float(
                math.degrees(
                    math.atan2(
                        c,
                        a,
                    )
                )
            ),

        "scale_x":
            float(
                math.sqrt(
                    a * a
                    + c * c
                )
            ),

        "scale_y":
            float(
                math.sqrt(
                    b * b
                    + d * d
                )
            ),
    }

    metrics = {
        "pair_id":
            "pair_005",

        "source":
            "SELENE/Kaguya TC",

        "reference":
            "Chandrayaan-2 TMC-2",

        "canonical_resolution_m":
            10.0,

        "final_method":
            (
                "coarse SIFT intensity affine "
                "+ fine SIFT gradient affine"
            ),

        "coarse_affine":
            COARSE_AFFINE.tolist(),

        "fine_affine_reproduced":
            M_check.tolist(),

        "final_composed_affine":
            final_affine.tolist(),

        "final_transform":
            total_transform,

        "fine_correspondence_metrics":
            {
                "candidates":
                    int(len(p0)),

                "ransac_inliers":
                    int(len(p0_in)),

                "inlier_ratio":
                    float(
                        len(p0_in)
                        / len(p0)
                    ),

                "ransac_reprojection_rmse_px":
                    rmse,

                "median_residual_px":
                    float(
                        np.median(
                            residual
                        )
                    ),

                "uniform_match_points":
                    int(
                        len(
                            uniform_ids
                        )
                    ),
            },

        "controlled_motion_validation":
            {
                "sift_gradient": {
                    "all_tests_pass":
                        True,

                    "mean_response_rmse_px":
                        0.25353194789979316,

                    "worst_response_rmse_px":
                        0.2649830031162702,
                },

                "loftr_intensity_corroboration": {
                    "all_tests_pass":
                        True,

                    "mean_response_rmse_px":
                        0.40380381149269545,

                    "worst_response_rmse_px":
                        0.4923566908485119,
                },

                "lightglue_gradient": {
                    "all_tests_pass":
                        False,
                },
            },

        "interpretation": {
            "ransac_note":
                (
                    "RANSAC residual is internal "
                    "correspondence consistency and "
                    "not independent absolute "
                    "ground-truth registration accuracy."
                ),

            "perturbation_note":
                (
                    "Known-motion response validates "
                    "that the cross-sensor matcher "
                    "follows injected terrain motion. "
                    "It does not provide independent "
                    "absolute ground truth for the "
                    "unperturbed real pair."
                ),
        },
    }

    metrics_path = (
        OUT_DIR
        / "pair005_final_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print(
        "PAIR 005 FINAL OUTPUTS"
    )
    print("=" * 78)

    for path in [
        registered_path,
        all_csv,
        uniform_csv,
        metrics_path,
        OUT_DIR
        / "registered_overlay_50_50.png",
        OUT_DIR
        / "registered_checkerboard.png",
        OUT_DIR
        / "registered_red_cyan.png",
        OUT_DIR
        / "pair005_final_matches.png",
    ]:

        print(path)

    print("\n" + "=" * 78)
    print(
        "PAIR 005 FINALIZATION COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()