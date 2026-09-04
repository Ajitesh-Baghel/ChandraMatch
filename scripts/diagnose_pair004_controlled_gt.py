from pathlib import Path
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
    / "controlled_gt_diagnostic"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

LOFTR_MAX_DIM = 1600
SIFT_MAX_DIM = 3000

RANSAC_THRESHOLD_PX = 3.0

MASK_EROSION_PIXELS = 5

GT_GRID_STEP = 64


CONTROLS = [
    {
        "id": "CTRL004_1",
        "tx": 3.35,
        "ty": -2.60,
        "rotation_deg": 0.18,
        "scale": 1.0015,
    },
    {
        "id": "CTRL004_2",
        "tx": -8.70,
        "ty": 5.45,
        "rotation_deg": -0.42,
        "scale": 0.9975,
    },
    {
        "id": "CTRL004_3",
        "tx": 14.25,
        "ty": -10.80,
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
    out = np.zeros(data.shape, dtype=np.uint8)

    values = data[mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return out

    lo, hi = np.percentile(values, [2.0, 98.0])

    if hi <= lo:
        hi = lo + 1.0

    scaled = (
        data.astype(np.float32) - float(lo)
    ) / float(hi - lo)

    scaled = np.clip(scaled, 0.0, 1.0)

    out[mask] = np.round(
        scaled[mask] * 255.0
    ).astype(np.uint8)

    return out


def gradient_uint8(image, mask):
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

    hi = np.percentile(values, 99.0)

    if hi <= 0:
        return out

    mag = np.clip(
        mag / hi,
        0.0,
        1.0,
    )

    out[mask] = np.round(
        mag[mask] * 255.0
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

    M = cv2.getRotationMatrix2D(
        center,
        rotation_deg,
        scale,
    ).astype(np.float64)

    M[0, 2] += tx
    M[1, 2] += ty

    return M


def transform_points(M, points):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    return (
        points @ M[:, :2].T
        + M[:, 2]
    )


# ============================================================
# RESIZE
# ============================================================

def resize_max(image, max_dim, interpolation):
    h, w = image.shape

    if max(h, w) <= max_dim:
        return image.copy(), 1.0

    scale = max_dim / max(h, w)

    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=interpolation,
    )

    return resized, scale


# ============================================================
# MASK FILTER
# ============================================================

def points_inside(points, mask):
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

    ids = np.where(inside)[0]

    result[ids] = mask[
        y[ids],
        x[ids],
    ]

    return result


# ============================================================
# SIFT
# ============================================================

def run_sift(source, reference, source_mask, reference_mask):
    src_small, scale = resize_max(
        source,
        SIFT_MAX_DIM,
        cv2.INTER_AREA,
    )

    ref_small, scale_ref = resize_max(
        reference,
        SIFT_MAX_DIM,
        cv2.INTER_AREA,
    )

    if abs(scale - scale_ref) > 1e-8:
        raise RuntimeError("SIFT scales differ.")

    src_mask_small = cv2.resize(
        source_mask.astype(np.uint8),
        (src_small.shape[1], src_small.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) * 255

    ref_mask_small = cv2.resize(
        reference_mask.astype(np.uint8),
        (ref_small.shape[1], ref_small.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ) * 255

    sift = cv2.SIFT_create(
        nfeatures=20000,
        contrastThreshold=0.01,
        edgeThreshold=10,
    )

    kp0, des0 = sift.detectAndCompute(
        src_small,
        src_mask_small,
    )

    kp1, des1 = sift.detectAndCompute(
        ref_small,
        ref_mask_small,
    )

    if des0 is None or des1 is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    bf = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    knn = bf.knnMatch(
        des0,
        des1,
        k=2,
    )

    good = []

    for pair in knn:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    src = np.asarray(
        [kp0[m.queryIdx].pt for m in good],
        dtype=np.float32,
    )

    ref = np.asarray(
        [kp1[m.trainIdx].pt for m in good],
        dtype=np.float32,
    )

    return (
        src / scale,
        ref / scale,
    )


# ============================================================
# LOFTR
# ============================================================

def run_loftr(
    matcher,
    source,
    reference,
    device,
):
    src_small, scale = resize_max(
        source,
        LOFTR_MAX_DIM,
        cv2.INTER_AREA,
    )

    ref_small, scale_ref = resize_max(
        reference,
        LOFTR_MAX_DIM,
        cv2.INTER_AREA,
    )

    if abs(scale - scale_ref) > 1e-8:
        raise RuntimeError("LoFTR scales differ.")

    image0 = torch.from_numpy(
        src_small.astype(np.float32) / 255.0
    )[None, None].to(device)

    image1 = torch.from_numpy(
        ref_small.astype(np.float32) / 255.0
    )[None, None].to(device)

    with torch.inference_mode():
        pred = matcher(
            {
                "image0": image0,
                "image1": image1,
            }
        )

    src = (
        pred["keypoints0"]
        .detach()
        .cpu()
        .numpy()
        / scale
    )

    ref = (
        pred["keypoints1"]
        .detach()
        .cpu()
        .numpy()
        / scale
    )

    return (
        src.astype(np.float32),
        ref.astype(np.float32),
    )


# ============================================================
# GT TRANSFORM ERROR
# ============================================================

def transform_rmse(
    estimated,
    ground_truth,
    source_mask,
    reference_mask,
):
    h, w = source_mask.shape

    pts = []

    for y in range(0, h, GT_GRID_STEP):
        for x in range(0, w, GT_GRID_STEP):
            if not source_mask[y, x]:
                continue

            gt = transform_points(
                ground_truth,
                [[x, y]],
            )[0]

            gx = int(round(gt[0]))
            gy = int(round(gt[1]))

            if (
                gx < 0
                or gx >= w
                or gy < 0
                or gy >= h
            ):
                continue

            if not reference_mask[gy, gx]:
                continue

            pts.append([x, y])

    pts = np.asarray(
        pts,
        dtype=np.float64,
    )

    est = transform_points(
        estimated,
        pts,
    )

    gt = transform_points(
        ground_truth,
        pts,
    )

    errors = np.linalg.norm(
        est - gt,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(errors ** 2)
        )
    )


# ============================================================
# MATCH DIAGNOSTIC
# ============================================================

def evaluate(
    name,
    src,
    ref,
    gt_affine,
    source_mask,
    reference_mask,
):
    valid = (
        points_inside(src, source_mask)
        & points_inside(ref, reference_mask)
    )

    src = src[valid]
    ref = ref[valid]

    print("\n" + "-" * 72)
    print(name)
    print("-" * 72)

    print("Filtered matches:", len(src))

    if len(src) < 4:
        print("Too few matches.")
        return

    true_ref = transform_points(
        gt_affine,
        src,
    )

    gt_error = np.linalg.norm(
        ref - true_ref,
        axis=1,
    )

    identity_error = np.linalg.norm(
        ref - src,
        axis=1,
    )

    print(
        "Raw-match GT median px :",
        f"{np.median(gt_error):.6f}",
    )

    print(
        "Raw-match GT mean px   :",
        f"{np.mean(gt_error):.6f}",
    )

    print(
        "GT-consistent <1px     :",
        f"{np.mean(gt_error < 1.0):.4f}",
    )

    print(
        "GT-consistent <3px     :",
        f"{np.mean(gt_error < 3.0):.4f}",
    )

    print(
        "Identity-consistent <1 :",
        f"{np.mean(identity_error < 1.0):.4f}",
    )

    print(
        "Identity-consistent <3 :",
        f"{np.mean(identity_error < 3.0):.4f}",
    )

    M, inlier_mask = cv2.estimateAffine2D(
        src,
        ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=20000,
        confidence=0.999,
        refineIters=100,
    )

    if M is None:
        print("RANSAC FAILED")
        return

    inlier_mask = inlier_mask.ravel() > 0

    src_in = src[inlier_mask]
    ref_in = ref[inlier_mask]

    pred = transform_points(
        M,
        src_in,
    )

    self_error = np.linalg.norm(
        pred - ref_in,
        axis=1,
    )

    gt_rmse = transform_rmse(
        M,
        gt_affine,
        source_mask,
        reference_mask,
    )

    print(
        "RANSAC inliers         :",
        len(src_in),
    )

    print(
        "RANSAC inlier ratio    :",
        f"{len(src_in) / len(src):.6f}",
    )

    print(
        "RANSAC self RMSE px    :",
        f"{np.sqrt(np.mean(self_error ** 2)):.6f}",
    )

    print(
        "TRUE transform GT RMSE :",
        f"{gt_rmse:.6f}",
    )

    print("\nRecovered affine:")
    print(M)

    print("\nExpected GT affine:")
    print(gt_affine)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("PAIR 004 — CONTROLLED GT DIAGNOSTIC")
    print("=" * 76)

    nac, raster_mask = read_band(
        NAC_PATH
    )

    nac_mask_data, _ = read_band(
        NAC_MASK_PATH
    )

    valid = (
        raster_mask
        & (nac_mask_data > 0)
        & np.isfinite(nac)
    )

    valid = ndimage.binary_erosion(
        valid,
        structure=np.ones(
            (3, 3),
            dtype=bool,
        ),
        iterations=MASK_EROSION_PIXELS,
        border_value=0,
    )

    print("\nShape:", nac.shape)
    print(
        "Valid pixels:",
        f"{int(valid.sum()):,}",
    )

    intensity = robust_uint8(
        nac,
        valid,
    )

    intensity[~valid] = 0

    gradient = gradient_uint8(
        intensity,
        valid,
    )

    gradient[~valid] = 0

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    loftr = KF.LoFTR(
        pretrained="outdoor"
    ).eval().to(device)

    h, w = intensity.shape

    for control in CONTROLS:
        print("\n")
        print("=" * 76)
        print(control["id"])
        print("=" * 76)

        forward = make_forward_affine(
            w,
            h,
            control["tx"],
            control["ty"],
            control["rotation_deg"],
            control["scale"],
        )

        gt = cv2.invertAffineTransform(
            forward
        )

        # ----------------------------------------------------
        # Algebraic sanity check:
        # original -> transformed -> original
        # ----------------------------------------------------

        test_points = np.asarray(
            [
                [w * 0.25, h * 0.25],
                [w * 0.50, h * 0.50],
                [w * 0.75, h * 0.75],
                [w * 0.75, h * 0.25],
            ],
            dtype=np.float64,
        )

        warped_pts = transform_points(
            forward,
            test_points,
        )

        roundtrip = transform_points(
            gt,
            warped_pts,
        )

        roundtrip_error = np.linalg.norm(
            roundtrip - test_points,
            axis=1,
        )

        print(
            "Affine roundtrip max error:",
            f"{roundtrip_error.max():.12f}px",
        )

        print("\nForward affine:")
        print(forward)

        print("\nExpected transformed->original GT:")
        print(gt)

        # ----------------------------------------------------
        # Synthetic transformed image
        # ----------------------------------------------------

        transformed_intensity = cv2.warpAffine(
            intensity,
            forward,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        transformed_mask = (
            cv2.warpAffine(
                valid.astype(np.uint8),
                forward,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )

        transformed_mask = ndimage.binary_erosion(
            transformed_mask,
            structure=np.ones(
                (3, 3),
                dtype=bool,
            ),
            iterations=MASK_EROSION_PIXELS,
            border_value=0,
        )

        transformed_gradient = gradient_uint8(
            transformed_intensity,
            transformed_mask,
        )

        transformed_gradient[
            ~transformed_mask
        ] = 0

        # ====================================================
        # SIFT INTENSITY
        # ====================================================

        src, ref = run_sift(
            transformed_intensity,
            intensity,
            transformed_mask,
            valid,
        )

        evaluate(
            "SIFT / INTENSITY",
            src,
            ref,
            gt,
            transformed_mask,
            valid,
        )

        # ====================================================
        # SIFT GRADIENT
        # ====================================================

        src, ref = run_sift(
            transformed_gradient,
            gradient,
            transformed_mask,
            valid,
        )

        evaluate(
            "SIFT / GRADIENT",
            src,
            ref,
            gt,
            transformed_mask,
            valid,
        )

        # ====================================================
        # LOFTR GRADIENT
        # ====================================================

        src, ref = run_loftr(
            loftr,
            transformed_gradient,
            gradient,
            device,
        )

        evaluate(
            "LOFTR / GRADIENT",
            src,
            ref,
            gt,
            transformed_mask,
            valid,
        )

    print("\n")
    print("=" * 76)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 76)

    print(
        "\nInterpretation:"
    )

    print(
        "If SIFT recovers GT but LoFTR does not, "
        "the synthetic affine machinery is correct "
        "and LoFTR is producing position-biased matches."
    )

    print(
        "If BOTH fail similarly, we investigate the "
        "synthetic warp / coordinate convention instead."
    )


if __name__ == "__main__":
    main()