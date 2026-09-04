from pathlib import Path
import csv
import json
import math

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
    / "pair_004"
    / "canonical"
)

SOURCE_PATH = PAIR_DIR / "lro_nac_source.tif"
REFERENCE_PATH = PAIR_DIR / "tmc2_reference.tif"
COMMON_MASK_PATH = PAIR_DIR / "common_valid_mask.tif"

INPUT_CSV = (
    ROOT
    / "results"
    / "pair_004"
    / "tiled_loftr"
    / "pair004_tiled_loftr_inliers.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_004"
    / "tiled_loftr_refined"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_M = 5.0

# Local refinement configuration
PATCH_SIZE = 31
PATCH_HALF = PATCH_SIZE // 2

SEARCH_RADIUS = 4

SEARCH_SIZE = (
    PATCH_SIZE
    + 2 * SEARCH_RADIUS
)

MIN_TEMPLATE_STD = 7.0

MIN_NCC = 0.40
MIN_PEAK_MARGIN = 0.025

MAX_CORRECTION_PX = 4.5

# Forward → backward consistency
MAX_CYCLE_ERROR_PX = 0.75

# Strict RANSAC after refinement
RANSAC_THRESHOLD_PX = 1.5

GRID_ROWS = 8
GRID_COLS = 8

MASK_EROSION_PIXELS = 5


# ============================================================
# INPUT
# ============================================================


def read_band(path):
    with rasterio.open(path) as ds:
        return (
            ds.read(1),
            ds.read_masks(1) > 0,
        )


def load_matches(path):
    src = []
    ref = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            src.append(
                [
                    float(row["source_x"]),
                    float(row["source_y"]),
                ]
            )

            ref.append(
                [
                    float(row["reference_x"]),
                    float(row["reference_y"]),
                ]
            )

    return (
        np.asarray(
            src,
            dtype=np.float32,
        ),
        np.asarray(
            ref,
            dtype=np.float32,
        ),
    )


# ============================================================
# IMAGE REPRESENTATION
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

    mag = np.sqrt(
        gx * gx
        + gy * gy
    )

    out = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    values = mag[mask]

    if values.size == 0:
        return out

    high = np.percentile(
        values,
        99.0,
    )

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


def build_matcher_mask(
    source,
    source_mask,
    reference,
    reference_mask,
    common,
):
    science = (
        common
        & source_mask
        & reference_mask
        & np.isfinite(source)
        & np.isfinite(reference)
    )

    feature_valid = (
        science
        & (reference > 0)
    )

    matcher = (
        ndimage.binary_erosion(
            feature_valid,
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

    return science, matcher


# ============================================================
# PATCH UTILITIES
# ============================================================


def window_fully_valid(
    mask,
    center,
    half_size,
):
    x = float(center[0])
    y = float(center[1])

    x0 = int(
        math.floor(
            x - half_size
        )
    )

    x1 = int(
        math.ceil(
            x + half_size
        )
    )

    y0 = int(
        math.floor(
            y - half_size
        )
    )

    y1 = int(
        math.ceil(
            y + half_size
        )
    )

    h, w = mask.shape

    if (
        x0 < 0
        or y0 < 0
        or x1 >= w
        or y1 >= h
    ):
        return False

    return bool(
        np.all(
            mask[
                y0:y1 + 1,
                x0:x1 + 1,
            ]
        )
    )


def get_patch(
    image,
    center,
    size,
):
    return cv2.getRectSubPix(
        image,
        (size, size),
        (
            float(center[0]),
            float(center[1]),
        ),
    )


# ============================================================
# SUBPIXEL PEAK
# ============================================================


def quadratic_offset(
    left,
    center,
    right,
):
    denominator = (
        left
        - 2.0 * center
        + right
    )

    if abs(denominator) < 1e-10:
        return 0.0

    delta = (
        0.5
        * (left - right)
        / denominator
    )

    return float(
        np.clip(
            delta,
            -1.0,
            1.0,
        )
    )


def refine_one_direction(
    template_image,
    search_image,
    template_center,
    search_center,
    valid_mask,
):
    """
    Match a PATCH_SIZE template centred at template_center
    against a SEARCH_SIZE window centred at search_center.

    Returns refined point in search_image coordinates.
    """

    # Template itself must be fully feature-valid.
    if not window_fully_valid(
        valid_mask,
        template_center,
        PATCH_HALF,
    ):
        return None

    # Entire search window must be feature-valid.
    search_half = (
        PATCH_HALF
        + SEARCH_RADIUS
    )

    if not window_fully_valid(
        valid_mask,
        search_center,
        search_half,
    ):
        return None

    template = get_patch(
        template_image,
        template_center,
        PATCH_SIZE,
    )

    search = get_patch(
        search_image,
        search_center,
        SEARCH_SIZE,
    )

    template = template.astype(
        np.float32
    )

    search = search.astype(
        np.float32
    )

    template_std = float(
        template.std()
    )

    if template_std < MIN_TEMPLATE_STD:
        return None

    response = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED,
    )

    _, peak_value, _, peak_location = (
        cv2.minMaxLoc(response)
    )

    peak_x = int(
        peak_location[0]
    )

    peak_y = int(
        peak_location[1]
    )

    rows, cols = response.shape

    # Need neighbours for quadratic subpixel interpolation.
    if (
        peak_x <= 0
        or peak_y <= 0
        or peak_x >= cols - 1
        or peak_y >= rows - 1
    ):
        return None

    if peak_value < MIN_NCC:
        return None

    # --------------------------------------------------------
    # Peak uniqueness
    # --------------------------------------------------------

    response_second = (
        response.copy()
    )

    sx0 = max(
        0,
        peak_x - 1,
    )

    sx1 = min(
        cols,
        peak_x + 2,
    )

    sy0 = max(
        0,
        peak_y - 1,
    )

    sy1 = min(
        rows,
        peak_y + 2,
    )

    response_second[
        sy0:sy1,
        sx0:sx1,
    ] = -1.0

    second_peak = float(
        np.max(
            response_second
        )
    )

    peak_margin = (
        float(peak_value)
        - second_peak
    )

    if peak_margin < MIN_PEAK_MARGIN:
        return None

    # --------------------------------------------------------
    # Quadratic subpixel peak
    # --------------------------------------------------------

    center_value = float(
        response[
            peak_y,
            peak_x,
        ]
    )

    dx = quadratic_offset(
        float(
            response[
                peak_y,
                peak_x - 1,
            ]
        ),
        center_value,
        float(
            response[
                peak_y,
                peak_x + 1,
            ]
        ),
    )

    dy = quadratic_offset(
        float(
            response[
                peak_y - 1,
                peak_x,
            ]
        ),
        center_value,
        float(
            response[
                peak_y + 1,
                peak_x,
            ]
        ),
    )

    # SEARCH_RADIUS is where a zero-shift peak should occur.
    shift_x = (
        peak_x
        + dx
        - SEARCH_RADIUS
    )

    shift_y = (
        peak_y
        + dy
        - SEARCH_RADIUS
    )

    correction = math.hypot(
        shift_x,
        shift_y,
    )

    if correction > MAX_CORRECTION_PX:
        return None

    refined = np.array(
        [
            float(search_center[0])
            + shift_x,

            float(search_center[1])
            + shift_y,
        ],
        dtype=np.float32,
    )

    return {
        "point": refined,
        "ncc": float(
            peak_value
        ),
        "peak_margin": float(
            peak_margin
        ),
        "correction_px": float(
            correction
        ),
        "shift_x": float(
            shift_x
        ),
        "shift_y": float(
            shift_y
        ),
    }


# ============================================================
# AFFINE / METRICS
# ============================================================


def affine_predict(
    matrix,
    points,
):
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

    tx = float(
        matrix[0, 2]
    )

    ty = float(
        matrix[1, 2]
    )

    translation = float(
        math.hypot(
            tx,
            ty,
        )
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
        "translation_magnitude_px":
            translation,
        "determinant": determinant,
        "axis_dot": axis_dot,
    }


def spatial_coverage(
    points,
    valid_mask,
):
    h, w = valid_mask.shape

    valid_cells = set()

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

        cell = (
            gy,
            gx,
        )

        if cell in valid_cells:
            occupied.add(cell)

    denominator = len(
        valid_cells
    )

    numerator = len(
        occupied
    )

    return (
        numerator,
        denominator,
        (
            numerator
            / denominator
            if denominator
            else 0.0
        ),
    )


# ============================================================
# VISUALIZATION
# ============================================================


def save_visualization(
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
        max_dim / max(h, w),
    )

    dw = int(
        round(
            w * scale
        )
    )

    dh = int(
        round(
            h * scale
        )
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

    if len(source_points) > 500:

        ids = np.linspace(
            0,
            len(source_points) - 1,
            500,
        ).astype(int)

        source_points = (
            source_points[ids]
        )

        reference_points = (
            reference_points[ids]
        )

    for src, ref in zip(
        source_points,
        reference_points,
    ):

        x0 = int(
            round(
                src[0] * scale
            )
        )

        y0 = int(
            round(
                src[1] * scale
            )
        )

        x1 = int(
            round(
                ref[0] * scale
            )
        ) + dw

        y1 = int(
            round(
                ref[1] * scale
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
    print("=" * 76)
    print(
        "PAIR 004 — STRICT SUBPIXEL "
        "REFINEMENT OF TILED LOFTR"
    )
    print("=" * 76)

    source, source_mask = (
        read_band(SOURCE_PATH)
    )

    reference, reference_mask = (
        read_band(REFERENCE_PATH)
    )

    common_data, _ = (
        read_band(
            COMMON_MASK_PATH
        )
    )

    common = (
        common_data > 0
    )

    (
        source_points,
        reference_points,
    ) = load_matches(
        INPUT_CSV
    )

    print(
        "\nInput tiled-LoFTR inliers:",
        len(source_points),
    )

    (
        science_mask,
        matcher_mask,
    ) = build_matcher_mask(
        source,
        source_mask,
        reference,
        reference_mask,
        common,
    )

    print(
        "Science-valid pixels      :",
        f"{int(science_mask.sum()):,}",
    )

    print(
        "Matcher-valid pixels      :",
        f"{int(matcher_mask.sum()):,}",
    )

    # --------------------------------------------------------
    # Gradient representations
    # --------------------------------------------------------

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

    source_g[
        ~matcher_mask
    ] = 0

    reference_g[
        ~matcher_mask
    ] = 0

    # --------------------------------------------------------
    # Strict local refinement
    # --------------------------------------------------------

    accepted_src = []
    accepted_ref = []

    ncc_values = []
    peak_margins = []
    corrections = []
    cycle_errors = []

    rejected_forward = 0
    rejected_backward = 0
    rejected_cycle = 0

    for i, (
        src,
        ref,
    ) in enumerate(
        zip(
            source_points,
            reference_points,
        )
    ):

        forward = (
            refine_one_direction(
                source_g,
                reference_g,
                src,
                ref,
                matcher_mask,
            )
        )

        if forward is None:

            rejected_forward += 1
            continue

        refined_ref = (
            forward["point"]
        )

        # ----------------------------------------------------
        # Backward consistency:
        # refined reference patch must lead back to original
        # source coordinate.
        # ----------------------------------------------------

        backward = (
            refine_one_direction(
                reference_g,
                source_g,
                refined_ref,
                src,
                matcher_mask,
            )
        )

        if backward is None:

            rejected_backward += 1
            continue

        returned_src = (
            backward["point"]
        )

        cycle_error = float(
            np.linalg.norm(
                returned_src
                - src
            )
        )

        if (
            cycle_error
            > MAX_CYCLE_ERROR_PX
        ):

            rejected_cycle += 1
            continue

        accepted_src.append(
            src
        )

        accepted_ref.append(
            refined_ref
        )

        ncc_values.append(
            forward["ncc"]
        )

        peak_margins.append(
            forward[
                "peak_margin"
            ]
        )

        corrections.append(
            forward[
                "correction_px"
            ]
        )

        cycle_errors.append(
            cycle_error
        )

    accepted_src = np.asarray(
        accepted_src,
        dtype=np.float32,
    )

    accepted_ref = np.asarray(
        accepted_ref,
        dtype=np.float32,
    )

    ncc_values = np.asarray(
        ncc_values,
        dtype=np.float32,
    )

    peak_margins = np.asarray(
        peak_margins,
        dtype=np.float32,
    )

    corrections = np.asarray(
        corrections,
        dtype=np.float32,
    )

    cycle_errors = np.asarray(
        cycle_errors,
        dtype=np.float32,
    )

    print("\n" + "=" * 76)
    print("REFINEMENT GATING")
    print("=" * 76)

    print(
        "Input correspondences :",
        len(source_points),
    )

    print(
        "Rejected forward      :",
        rejected_forward,
    )

    print(
        "Rejected backward     :",
        rejected_backward,
    )

    print(
        "Rejected cycle        :",
        rejected_cycle,
    )

    print(
        "Accepted refined      :",
        len(accepted_src),
    )

    if len(accepted_src) < 4:

        raise RuntimeError(
            "Too few correspondences "
            "survived strict refinement."
        )

    print(
        "Acceptance ratio      :",
        f"{len(accepted_src) / len(source_points):.6f}",
    )

    print(
        "\nForward NCC median    :",
        f"{np.median(ncc_values):.6f}",
    )

    print(
        "Forward NCC mean      :",
        f"{np.mean(ncc_values):.6f}",
    )

    print(
        "Peak margin median    :",
        f"{np.median(peak_margins):.6f}",
    )

    print(
        "Correction median px  :",
        f"{np.median(corrections):.6f}",
    )

    print(
        "Correction mean px    :",
        f"{np.mean(corrections):.6f}",
    )

    print(
        "Cycle error median px :",
        f"{np.median(cycle_errors):.6f}",
    )

    # --------------------------------------------------------
    # RANSAC on refined correspondences
    # --------------------------------------------------------

    matrix, inlier_mask = (
        cv2.estimateAffine2D(
            accepted_src,
            accepted_ref,
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
            "Refined RANSAC failed."
        )

    inlier_mask = (
        inlier_mask.ravel() > 0
    )

    src_in = (
        accepted_src[
            inlier_mask
        ]
    )

    ref_in = (
        accepted_ref[
            inlier_mask
        ]
    )

    ncc_in = (
        ncc_values[
            inlier_mask
        ]
    )

    correction_in = (
        corrections[
            inlier_mask
        ]
    )

    cycle_in = (
        cycle_errors[
            inlier_mask
        ]
    )

    predicted = affine_predict(
        matrix,
        src_in,
    )

    residuals = np.linalg.norm(
        predicted - ref_in,
        axis=1,
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

    affine_info = (
        decompose_affine(
            matrix
        )
    )

    print("\n" + "=" * 76)
    print("REFINED RESULT")
    print("=" * 76)

    print(
        "Accepted candidates        :",
        len(accepted_src),
    )

    print(
        "RANSAC inliers             :",
        len(src_in),
    )

    print(
        "Inlier ratio               :",
        f"{len(src_in) / len(accepted_src):.6f}",
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
        "Residual × 5m nominal      :",
        f"{rmse * PIXEL_SIZE_M:.3f} m",
    )

    print(
        "Spatial coverage           :",
        f"{coverage_n}/"
        f"{coverage_d} "
        f"({coverage_ratio:.4f})",
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

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        OUT_DIR
        / "pair004_refined_inliers.csv"
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
                "refined_reference_x",
                "refined_reference_y",
                "ncc",
                "local_correction_px",
                "cycle_error_px",
                "ransac_residual_px",
            ]
        )

        for (
            src,
            ref,
            ncc,
            correction,
            cycle,
            residual,
        ) in zip(
            src_in,
            ref_in,
            ncc_in,
            correction_in,
            cycle_in,
            residuals,
        ):

            writer.writerow(
                [
                    float(src[0]),
                    float(src[1]),
                    float(ref[0]),
                    float(ref[1]),
                    float(ncc),
                    float(correction),
                    float(cycle),
                    float(residual),
                ]
            )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    metrics = {
        "pair_id": "pair_004",

        "input_method":
            "prior_guided_tiled_loftr_gradient",

        "refinement":
            "strict_local_gradient_ncc_subpixel",

        "input_correspondences":
            int(len(source_points)),

        "accepted_refined":
            int(len(accepted_src)),

        "refinement_acceptance_ratio":
            float(
                len(accepted_src)
                / len(source_points)
            ),

        "ransac_inliers":
            int(len(src_in)),

        "inlier_ratio":
            float(
                len(src_in)
                / len(accepted_src)
            ),

        "ransac_reprojection_rmse_px":
            rmse,

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

        "median_ncc":
            float(
                np.median(
                    ncc_values
                )
            ),

        "median_local_correction_px":
            float(
                np.median(
                    corrections
                )
            ),

        "median_cycle_error_px":
            float(
                np.median(
                    cycle_errors
                )
            ),

        "affine":
            matrix.tolist(),

        "affine_parameters":
            affine_info,

        "parameters": {
            "patch_size":
                PATCH_SIZE,

            "search_radius_px":
                SEARCH_RADIUS,

            "minimum_template_std":
                MIN_TEMPLATE_STD,

            "minimum_ncc":
                MIN_NCC,

            "minimum_peak_margin":
                MIN_PEAK_MARGIN,

            "maximum_correction_px":
                MAX_CORRECTION_PX,

            "maximum_cycle_error_px":
                MAX_CYCLE_ERROR_PX,

            "ransac_threshold_px":
                RANSAC_THRESHOLD_PX,
        },

        "note": (
            "Reported RANSAC residual is "
            "internal/self-consistency only. "
            "It is not independent "
            "ground-truth registration accuracy."
        ),
    }

    json_path = (
        OUT_DIR
        / "pair004_refined_metrics.json"
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
        / "pair004_refined_matches.png"
    )

    save_visualization(
        source_g,
        reference_g,
        src_in,
        ref_in,
        visualization_path,
    )

    print("\n" + "=" * 76)
    print("OUTPUTS")
    print("=" * 76)

    print(json_path)
    print(csv_path)
    print(
        visualization_path
    )

    print("\nIMPORTANT:")
    print(
        "Do not interpret the residual "
        "as independent registration accuracy."
    )

    print(
        "\nPAIR 004 STRICT REFINEMENT COMPLETE."
    )


if __name__ == "__main__":
    main()