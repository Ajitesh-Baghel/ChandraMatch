from pathlib import Path
import csv
import json
import math

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

NAC_PATH = PAIR_DIR / "lro_nac_source.tif"
NAC_MASK_PATH = PAIR_DIR / "lro_nac_valid_mask.tif"

OUT_DIR = (
    ROOT
    / "results"
    / "pair_004"
    / "controlled_gt"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PIXEL_SIZE_M = 5.0

LOFTR_MAX_DIM = 1600

RANSAC_THRESHOLD_PX = 3.0

MASK_EROSION_PIXELS = 5

GT_EVAL_GRID_STEP = 64


# ------------------------------------------------------------
# Three controlled affine perturbations
#
# These are deliberately subpixel + rotation + scale mixtures.
# ------------------------------------------------------------

CONTROLS = [
    {
        "id": "CTRL004_1",
        "translation_x_px": 3.35,
        "translation_y_px": -2.60,
        "rotation_deg": 0.18,
        "scale": 1.0015,
    },
    {
        "id": "CTRL004_2",
        "translation_x_px": -8.70,
        "translation_y_px": 5.45,
        "rotation_deg": -0.42,
        "scale": 0.9975,
    },
    {
        "id": "CTRL004_3",
        "translation_x_px": 14.25,
        "translation_y_px": -10.80,
        "rotation_deg": 0.75,
        "scale": 1.0045,
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
    image_f = (
        image.astype(np.float32)
        / 255.0
    )

    image_f = cv2.GaussianBlur(
        image_f,
        (5, 5),
        0.8,
    )

    gx = cv2.Sobel(
        image_f,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image_f,
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
        magnitude[mask]
        * 255.0
    ).astype(np.uint8)

    return out


# ============================================================
# AFFINE UTILITIES
# ============================================================


def affine_predict(matrix, points):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    return (
        points
        @ matrix[:, :2].T
        + matrix[:, 2]
    )


def make_control_affine(
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
# RESIZE
# ============================================================


def resize_max_dim(
    image,
    max_dim,
    interpolation,
):
    h, w = image.shape

    current = max(h, w)

    if current <= max_dim:
        return image.copy(), 1.0

    scale = max_dim / current

    nw = max(
        1,
        int(round(w * scale)),
    )

    nh = max(
        1,
        int(round(h * scale)),
    )

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=interpolation,
    )

    return resized, scale


# ============================================================
# POINT MASK
# ============================================================


def points_inside_mask(points, mask):
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
# LOFTR
# ============================================================


def run_loftr(
    matcher,
    source_image,
    reference_image,
    device,
):
    src_small, src_scale = (
        resize_max_dim(
            source_image,
            LOFTR_MAX_DIM,
            cv2.INTER_AREA,
        )
    )

    ref_small, ref_scale = (
        resize_max_dim(
            reference_image,
            LOFTR_MAX_DIM,
            cv2.INTER_AREA,
        )
    )

    if abs(
        src_scale - ref_scale
    ) > 1e-8:
        raise RuntimeError(
            "Resize scales differ."
        )

    image0 = torch.from_numpy(
        src_small.astype(
            np.float32
        ) / 255.0
    )[None, None].to(device)

    image1 = torch.from_numpy(
        ref_small.astype(
            np.float32
        ) / 255.0
    )[None, None].to(device)

    with torch.inference_mode():
        prediction = matcher(
            {
                "image0": image0,
                "image1": image1,
            }
        )

    source_points = (
        prediction["keypoints0"]
        .detach()
        .cpu()
        .numpy()
    )

    reference_points = (
        prediction["keypoints1"]
        .detach()
        .cpu()
        .numpy()
    )

    source_points = (
        source_points
        / src_scale
    )

    reference_points = (
        reference_points
        / ref_scale
    )

    return (
        source_points.astype(
            np.float32
        ),
        reference_points.astype(
            np.float32
        ),
    )


# ============================================================
# KNOWN-GT TRANSFORM EVALUATION
# ============================================================


def gt_transform_error(
    estimated,
    ground_truth,
    transformed_mask,
    reference_mask,
):
    h, w = transformed_mask.shape

    points = []

    for y in range(
        0,
        h,
        GT_EVAL_GRID_STEP,
    ):
        for x in range(
            0,
            w,
            GT_EVAL_GRID_STEP,
        ):
            if not transformed_mask[
                y,
                x,
            ]:
                continue

            gt_target = affine_predict(
                ground_truth,
                np.array(
                    [[x, y]],
                    dtype=np.float64,
                ),
            )[0]

            gx = int(
                round(
                    gt_target[0]
                )
            )

            gy = int(
                round(
                    gt_target[1]
                )
            )

            if (
                gx < 0
                or gx >= w
                or gy < 0
                or gy >= h
            ):
                continue

            if not reference_mask[
                gy,
                gx,
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
        return None

    predicted_est = affine_predict(
        estimated,
        points,
    )

    predicted_gt = affine_predict(
        ground_truth,
        points,
    )

    error = np.linalg.norm(
        predicted_est
        - predicted_gt,
        axis=1,
    )

    return {
        "evaluation_points":
            int(len(points)),

        "ground_truth_registration_rmse_px":
            float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),

        "ground_truth_registration_median_px":
            float(
                np.median(error)
            ),

        "ground_truth_registration_mean_px":
            float(
                np.mean(error)
            ),

        "ground_truth_registration_max_px":
            float(
                np.max(error)
            ),
    }


# ============================================================
# VISUALIZATION
# ============================================================


def save_matches(
    source_image,
    reference_image,
    source_points,
    reference_points,
    path,
):
    h, w = source_image.shape

    max_dim = 1800

    scale = min(
        1.0,
        max_dim / max(h, w),
    )

    dw = int(
        round(w * scale)
    )

    dh = int(
        round(h * scale)
    )

    src = cv2.resize(
        source_image,
        (dw, dh),
        interpolation=cv2.INTER_AREA,
    )

    ref = cv2.resize(
        reference_image,
        (dw, dh),
        interpolation=cv2.INTER_AREA,
    )

    src = cv2.cvtColor(
        src,
        cv2.COLOR_GRAY2BGR,
    )

    ref = cv2.cvtColor(
        ref,
        cv2.COLOR_GRAY2BGR,
    )

    canvas = np.concatenate(
        [src, ref],
        axis=1,
    )

    if len(source_points) > 300:
        ids = np.linspace(
            0,
            len(source_points) - 1,
            300,
        ).astype(int)

        source_points = (
            source_points[ids]
        )

        reference_points = (
            reference_points[ids]
        )

    for p0, p1 in zip(
        source_points,
        reference_points,
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
# RUN ONE CONTROL
# ============================================================


def run_control(
    control,
    original_intensity,
    original_mask,
    original_gradient,
    matcher,
    device,
):
    control_id = control["id"]

    print("\n")
    print("=" * 76)
    print(control_id)
    print("=" * 76)

    h, w = original_intensity.shape

    forward_affine = (
        make_control_affine(
            w,
            h,
            control[
                "translation_x_px"
            ],
            control[
                "translation_y_px"
            ],
            control[
                "rotation_deg"
            ],
            control[
                "scale"
            ],
        )
    )

    # --------------------------------------------------------
    # forward_affine:
    # original NAC -> synthetically transformed NAC
    #
    # We match:
    # transformed NAC -> original NAC
    #
    # Therefore this inverse matrix is the ACTUAL GT mapping.
    # --------------------------------------------------------

    ground_truth_affine = (
        cv2.invertAffineTransform(
            forward_affine
        )
    )

    transformed_intensity = (
        cv2.warpAffine(
            original_intensity,
            forward_affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )

    transformed_mask = (
        cv2.warpAffine(
            original_mask.astype(
                np.uint8
            ),
            forward_affine,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
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
            iterations=(
                MASK_EROSION_PIXELS
            ),
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
    # LoFTR
    # --------------------------------------------------------

    source_points, reference_points = (
        run_loftr(
            matcher,
            transformed_gradient,
            original_gradient,
            device,
        )
    )

    raw_matches = len(
        source_points
    )

    valid = (
        points_inside_mask(
            source_points,
            transformed_mask,
        )
        &
        points_inside_mask(
            reference_points,
            original_mask,
        )
    )

    source_points = (
        source_points[valid]
    )

    reference_points = (
        reference_points[valid]
    )

    filtered_matches = len(
        source_points
    )

    print(
        "Raw LoFTR matches       :",
        raw_matches,
    )

    print(
        "Mask-filtered matches   :",
        filtered_matches,
    )

    if filtered_matches < 4:
        raise RuntimeError(
            f"{control_id}: "
            "too few matches."
        )

    # --------------------------------------------------------
    # RANSAC pipeline estimate
    # --------------------------------------------------------

    estimated_affine, inlier_mask = (
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
        estimated_affine is None
        or inlier_mask is None
    ):
        raise RuntimeError(
            f"{control_id}: "
            "RANSAC failed."
        )

    inlier_mask = (
        inlier_mask.ravel() > 0
    )

    src_in = source_points[
        inlier_mask
    ]

    ref_in = reference_points[
        inlier_mask
    ]

    # --------------------------------------------------------
    # SELF-CONSISTENCY residual
    # --------------------------------------------------------

    ransac_prediction = (
        affine_predict(
            estimated_affine,
            src_in,
        )
    )

    self_error = np.linalg.norm(
        ransac_prediction
        - ref_in,
        axis=1,
    )

    self_rmse = float(
        np.sqrt(
            np.mean(
                self_error ** 2
            )
        )
    )

    # --------------------------------------------------------
    # TRUE GT correspondence error for accepted matches
    # --------------------------------------------------------

    true_reference = affine_predict(
        ground_truth_affine,
        src_in,
    )

    match_gt_error = np.linalg.norm(
        ref_in
        - true_reference,
        axis=1,
    )

    match_gt_rmse = float(
        np.sqrt(
            np.mean(
                match_gt_error ** 2
            )
        )
    )

    match_gt_median = float(
        np.median(
            match_gt_error
        )
    )

    # --------------------------------------------------------
    # TRUE estimated-transform-vs-GT error
    #
    # This is our strongest controlled registration metric.
    # --------------------------------------------------------

    transform_gt = gt_transform_error(
        estimated_affine,
        ground_truth_affine,
        transformed_mask,
        original_mask,
    )

    if transform_gt is None:
        raise RuntimeError(
            "No GT evaluation points."
        )

    estimated_params = (
        decompose_affine(
            estimated_affine
        )
    )

    ground_truth_params = (
        decompose_affine(
            ground_truth_affine
        )
    )

    print(
        "RANSAC inliers          :",
        len(src_in),
    )

    print(
        "Inlier ratio            :",
        f"{len(src_in) / filtered_matches:.6f}",
    )

    print(
        "\nRANSAC self RMSE px     :",
        f"{self_rmse:.6f}",
    )

    print(
        "Matched-point GT RMSE px:",
        f"{match_gt_rmse:.6f}",
    )

    print(
        "Matched-point GT median :",
        f"{match_gt_median:.6f}",
    )

    print(
        "\nGROUND-TRUTH "
        "REGISTRATION RMSE PX:",
        f"{transform_gt['ground_truth_registration_rmse_px']:.6f}",
    )

    print(
        "GROUND-TRUTH "
        "REGISTRATION RMSE M :",
        f"{transform_gt['ground_truth_registration_rmse_px'] * PIXEL_SIZE_M:.4f}",
    )

    print(
        "GT evaluation points    :",
        transform_gt[
            "evaluation_points"
        ],
    )

    print("\nKnown GT affine:")
    print(
        ground_truth_affine
    )

    print("\nRecovered affine:")
    print(
        estimated_affine
    )

    print(
        "\nKnown GT parameters:"
    )
    print(
        json.dumps(
            ground_truth_params,
            indent=2,
        )
    )

    print(
        "\nRecovered parameters:"
    )
    print(
        json.dumps(
            estimated_params,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # SAVE CONTROL OUTPUTS
    # --------------------------------------------------------

    control_dir = (
        OUT_DIR / control_id
    )

    control_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(
            control_dir
            / "transformed_gradient.png"
        ),
        transformed_gradient,
    )

    save_matches(
        transformed_gradient,
        original_gradient,
        src_in,
        ref_in,
        control_dir
        / "matches.png",
    )

    csv_path = (
        control_dir
        / "inliers_with_gt.csv"
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
                "matched_reference_x",
                "matched_reference_y",
                "true_reference_x",
                "true_reference_y",
                "ground_truth_error_px",
                "ransac_self_residual_px",
            ]
        )

        for (
            src,
            ref,
            true_ref,
            gt_err,
            self_err,
        ) in zip(
            src_in,
            ref_in,
            true_reference,
            match_gt_error,
            self_error,
        ):
            writer.writerow(
                [
                    float(src[0]),
                    float(src[1]),
                    float(ref[0]),
                    float(ref[1]),
                    float(
                        true_ref[0]
                    ),
                    float(
                        true_ref[1]
                    ),
                    float(gt_err),
                    float(self_err),
                ]
            )

    result = {
        "control_id":
            control_id,

        "synthetic_forward_transform":
            control,

        "raw_matches":
            int(raw_matches),

        "filtered_matches":
            int(filtered_matches),

        "ransac_inliers":
            int(len(src_in)),

        "ransac_inlier_ratio":
            float(
                len(src_in)
                / filtered_matches
            ),

        # ----------------------------------------------------
        # Internal/self-consistency metric
        # ----------------------------------------------------

        "ransac_reprojection_rmse_px":
            self_rmse,

        # ----------------------------------------------------
        # Actual known-GT metrics
        # ----------------------------------------------------

        "matched_point_ground_truth_rmse_px":
            match_gt_rmse,

        "matched_point_ground_truth_median_px":
            match_gt_median,

        "ground_truth_registration_rmse_px":
            transform_gt[
                "ground_truth_registration_rmse_px"
            ],

        "ground_truth_registration_rmse_m":
            (
                transform_gt[
                    "ground_truth_registration_rmse_px"
                ]
                * PIXEL_SIZE_M
            ),

        "ground_truth_registration_median_px":
            transform_gt[
                "ground_truth_registration_median_px"
            ],

        "ground_truth_registration_mean_px":
            transform_gt[
                "ground_truth_registration_mean_px"
            ],

        "ground_truth_registration_max_px":
            transform_gt[
                "ground_truth_registration_max_px"
            ],

        "gt_evaluation_points":
            transform_gt[
                "evaluation_points"
            ],

        "ground_truth_affine":
            ground_truth_affine.tolist(),

        "estimated_affine":
            estimated_affine.tolist(),

        "ground_truth_affine_parameters":
            ground_truth_params,

        "estimated_affine_parameters":
            estimated_params,

        "important_note": (
            "ground_truth_registration_rmse_px "
            "is evaluated against the explicitly "
            "known synthetic affine transformation "
            "and is therefore an independent "
            "controlled ground-truth metric. "
            "ransac_reprojection_rmse_px is only "
            "internal self-consistency."
        ),
    }

    with open(
        control_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    return result


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 76)
    print(
        "PAIR 004 — CONTROLLED "
        "KNOWN-GROUND-TRUTH VALIDATION"
    )
    print("=" * 76)

    nac, nac_raster_mask = (
        read_band(NAC_PATH)
    )

    nac_valid_data, _ = (
        read_band(NAC_MASK_PATH)
    )

    valid = (
        nac_raster_mask
        & (nac_valid_data > 0)
        & np.isfinite(nac)
    )

    valid = ndimage.binary_erosion(
        valid,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
        iterations=(
            MASK_EROSION_PIXELS
        ),
        border_value=0,
    )

    print(
        "\nImage shape:",
        nac.shape,
    )

    print(
        "Valid pixels:",
        f"{int(valid.sum()):,}",
    )

    intensity = robust_uint8(
        nac,
        valid,
    )

    intensity[
        ~valid
    ] = 0

    gradient = gradient_uint8(
        intensity,
        valid,
    )

    gradient[
        ~valid
    ] = 0

    cv2.imwrite(
        str(
            OUT_DIR
            / "original_gradient.png"
        ),
        gradient,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
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

    results = []

    for control in CONTROLS:
        result = run_control(
            control,
            intensity,
            valid,
            gradient,
            matcher,
            device,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_path = (
        OUT_DIR
        / "pair004_controlled_gt_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\n")
    print("=" * 76)
    print("CONTROLLED GT SUMMARY")
    print("=" * 76)

    for result in results:
        print(
            f"{result['control_id']}: "
            f"inliers={result['ransac_inliers']}, "
            f"self_RMSE="
            f"{result['ransac_reprojection_rmse_px']:.4f}px, "
            f"GT_registration_RMSE="
            f"{result['ground_truth_registration_rmse_px']:.4f}px, "
            f"GT_match_RMSE="
            f"{result['matched_point_ground_truth_rmse_px']:.4f}px"
        )

    gt_values = np.array(
        [
            r[
                "ground_truth_registration_rmse_px"
            ]
            for r in results
        ],
        dtype=np.float64,
    )

    print(
        "\nMean controlled GT "
        "registration RMSE:",
        f"{gt_values.mean():.6f} px",
    )

    print(
        "Worst controlled GT "
        "registration RMSE:",
        f"{gt_values.max():.6f} px",
    )

    print(
        "\nThis GT registration RMSE "
        "is a genuine known-transform "
        "controlled metric."
    )

    print(
        "Do NOT transfer this number "
        "directly to the real NAC↔TMC pair."
    )

    print(
        "\nSummary:"
    )
    print(
        summary_path
    )

    print(
        "\nPAIR 004 CONTROLLED "
        "VALIDATION COMPLETE."
    )


if __name__ == "__main__":
    main()