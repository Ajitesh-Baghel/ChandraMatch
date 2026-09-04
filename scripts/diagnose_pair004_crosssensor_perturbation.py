from pathlib import Path
import json
import math

import cv2
import numpy as np
import rasterio
from scipy import ndimage

import torch
import kornia.feature as KF

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
    / "pair_004"
    / "canonical"
)

SOURCE_PATH = (
    PAIR_DIR
    / "lro_nac_source.tif"
)

REFERENCE_PATH = (
    PAIR_DIR
    / "tmc2_reference.tif"
)

COMMON_MASK_PATH = (
    PAIR_DIR
    / "common_valid_mask.tif"
)

BASELINE_JSON = (
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
    / "crosssensor_perturbation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOFTR_MAX_DIM = 1600
LIGHTGLUE_MAX_DIM = 2048

MASK_EROSION = 5

RANSAC_THRESHOLD_PX = 3.0

GT_GRID_STEP = 64


# ============================================================
# CONTROLLED PERTURBATIONS
# ============================================================

PERTURBATIONS = [
    {
        "id": "SHIFT_1",
        "tx": 64.0,
        "ty": -48.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
    },
    {
        "id": "SHIFT_ROT_2",
        "tx": -120.0,
        "ty": 80.0,
        "rotation_deg": 0.35,
        "scale": 1.0,
    },
    {
        "id": "SHIFT_ROT_SCALE_3",
        "tx": 80.0,
        "ty": 60.0,
        "rotation_deg": -0.50,
        "scale": 1.002,
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


def load_baseline_affine(
    method,
    representation="gradient",
):
    with open(
        BASELINE_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        results = json.load(f)

    for result in results:
        if (
            result["method"] == method
            and
            result["representation"]
            == representation
        ):
            if result["affine"] is None:
                raise RuntimeError(
                    f"{method} has no baseline affine."
                )

            return np.asarray(
                result["affine"],
                dtype=np.float64,
            )

    raise RuntimeError(
        f"Baseline not found: "
        f"{method}/{representation}"
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


# ============================================================
# AFFINE UTILITIES
# ============================================================

def to_homogeneous(M):
    H = np.eye(
        3,
        dtype=np.float64,
    )

    H[:2, :] = M

    return H


def from_homogeneous(H):
    return H[:2, :].copy()


def compose(A, B):
    """
    A o B

    Point:
        x -> B(x) -> A(B(x))
    """

    return from_homogeneous(
        to_homogeneous(A)
        @ to_homogeneous(B)
    )


def affine_predict(M, points):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    return (
        points
        @ M[:, :2].T
        + M[:, 2]
    )


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


def decompose_affine(M):
    A = M[:, :2]

    c0 = A[:, 0]
    c1 = A[:, 1]

    sx = float(
        np.linalg.norm(c0)
    )

    sy = float(
        np.linalg.norm(c1)
    )

    rotation = float(
        math.degrees(
            math.atan2(
                A[1, 0],
                A[0, 0],
            )
        )
    )

    tx = float(M[0, 2])
    ty = float(M[1, 2])

    return {
        "scale_x": sx,
        "scale_y": sy,
        "rotation_deg": rotation,
        "translation_x_px": tx,
        "translation_y_px": ty,
        "translation_magnitude_px": float(
            math.hypot(tx, ty)
        ),
        "determinant": float(
            np.linalg.det(A)
        ),
    }


# ============================================================
# RESIZE
# ============================================================

def resize_max(
    image,
    max_dim,
):
    h, w = image.shape

    if max(h, w) <= max_dim:
        return image.copy(), 1.0

    scale = (
        max_dim
        / max(h, w)
    )

    new_w = int(
        round(w * scale)
    )

    new_h = int(
        round(h * scale)
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


# ============================================================
# MASK
# ============================================================

def points_inside(
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

    ids = np.where(inside)[0]

    result[ids] = mask[
        y[ids],
        x[ids],
    ]

    return result


# ============================================================
# LIGHTGLUE
# ============================================================

def tensor_lightglue(
    image,
    device,
):
    return torch.from_numpy(
        image.astype(np.float32)
        / 255.0
    ).unsqueeze(0).to(device)


def run_lightglue(
    extractor,
    matcher,
    source,
    reference,
    device,
):
    source_small, scale = (
        resize_max(
            source,
            LIGHTGLUE_MAX_DIM,
        )
    )

    reference_small, scale_ref = (
        resize_max(
            reference,
            LIGHTGLUE_MAX_DIM,
        )
    )

    if abs(
        scale - scale_ref
    ) > 1e-8:
        raise RuntimeError(
            "LightGlue resize scales differ."
        )

    image0 = tensor_lightglue(
        source_small,
        device,
    )

    image1 = tensor_lightglue(
        reference_small,
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

    feats0 = rbd(feats0)
    feats1 = rbd(feats1)
    prediction = rbd(
        prediction
    )

    kp0 = (
        feats0["keypoints"]
        .detach()
        .cpu()
        .numpy()
    )

    kp1 = (
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

        src = kp0[
            matches[:, 0]
        ]

        ref = kp1[
            matches[:, 1]
        ]

    else:

        matches0 = (
            prediction["matches0"]
            .detach()
            .cpu()
            .numpy()
        )

        idx0 = np.where(
            matches0 >= 0
        )[0]

        idx1 = matches0[idx0]

        src = kp0[idx0]
        ref = kp1[idx1]

    return (
        (src / scale).astype(
            np.float32
        ),
        (ref / scale).astype(
            np.float32
        ),
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
    source_small, scale = (
        resize_max(
            source,
            LOFTR_MAX_DIM,
        )
    )

    reference_small, scale_ref = (
        resize_max(
            reference,
            LOFTR_MAX_DIM,
        )
    )

    if abs(
        scale - scale_ref
    ) > 1e-8:
        raise RuntimeError(
            "LoFTR resize scales differ."
        )

    image0 = torch.from_numpy(
        source_small.astype(
            np.float32
        ) / 255.0
    )[None, None].to(device)

    image1 = torch.from_numpy(
        reference_small.astype(
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

    src = (
        prediction[
            "keypoints0"
        ]
        .detach()
        .cpu()
        .numpy()
        / scale
    )

    ref = (
        prediction[
            "keypoints1"
        ]
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
# TRANSFORM COMPARISON
# ============================================================

def transform_difference_rmse(
    recovered,
    expected,
    source_mask,
    reference_mask,
):
    h, w = source_mask.shape

    points = []

    for y in range(
        0,
        h,
        GT_GRID_STEP,
    ):
        for x in range(
            0,
            w,
            GT_GRID_STEP,
        ):

            if not source_mask[
                y,
                x,
            ]:
                continue

            expected_point = (
                affine_predict(
                    expected,
                    [[x, y]],
                )[0]
            )

            ex = int(
                round(
                    expected_point[0]
                )
            )

            ey = int(
                round(
                    expected_point[1]
                )
            )

            if (
                ex < 0
                or ex >= w
                or ey < 0
                or ey >= h
            ):
                continue

            if not reference_mask[
                ey,
                ex,
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
        return None, 0

    recovered_pts = (
        affine_predict(
            recovered,
            points,
        )
    )

    expected_pts = (
        affine_predict(
            expected,
            points,
        )
    )

    error = np.linalg.norm(
        recovered_pts
        - expected_pts,
        axis=1,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    return rmse, len(points)


# ============================================================
# EVALUATE MATCHER
# ============================================================

def evaluate(
    method_name,
    source_points,
    reference_points,
    source_mask,
    reference_mask,
    baseline_affine,
    forward_warp,
):
    valid = (
        points_inside(
            source_points,
            source_mask,
        )
        &
        points_inside(
            reference_points,
            reference_mask,
        )
    )

    source_points = (
        source_points[valid]
    )

    reference_points = (
        reference_points[valid]
    )

    print("\n" + "-" * 74)
    print(method_name)
    print("-" * 74)

    print(
        "Filtered matches:",
        len(source_points),
    )

    if len(source_points) < 4:
        print(
            "Too few matches."
        )
        return None

    recovered, inlier_mask = (
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
        recovered is None
        or inlier_mask is None
    ):
        print(
            "RANSAC failed."
        )
        return None

    inlier_mask = (
        inlier_mask.ravel() > 0
    )

    src_in = (
        source_points[
            inlier_mask
        ]
    )

    ref_in = (
        reference_points[
            inlier_mask
        ]
    )

    prediction = (
        affine_predict(
            recovered,
            src_in,
        )
    )

    residual = np.linalg.norm(
        prediction - ref_in,
        axis=1,
    )

    self_rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    # --------------------------------------------------------
    # If:
    #
    # original NAC -> TMC = B
    #
    # and:
    #
    # original NAC -> transformed NAC = W
    #
    # then:
    #
    # transformed NAC -> TMC
    #
    # must be:
    #
    # B o inverse(W)
    # --------------------------------------------------------

    inverse_warp = (
        cv2.invertAffineTransform(
            forward_warp
        )
    )

    expected = compose(
        baseline_affine,
        inverse_warp,
    )

    response_rmse, eval_points = (
        transform_difference_rmse(
            recovered,
            expected,
            source_mask,
            reference_mask,
        )
    )

    identity = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    identity_rmse, _ = (
        transform_difference_rmse(
            recovered,
            identity,
            source_mask,
            reference_mask,
        )
    )

    # Another equivalent diagnostic:
    #
    # recovered o W should return to the
    # original baseline B.
    compensated = compose(
        recovered,
        forward_warp,
    )

    compensated_rmse, _ = (
        transform_difference_rmse(
            compensated,
            baseline_affine,
            source_mask,
            reference_mask,
        )
    )

    print(
        "RANSAC inliers:",
        len(src_in),
    )

    print(
        "Inlier ratio:",
        f"{len(src_in) / len(source_points):.6f}",
    )

    print(
        "Self RMSE px:",
        f"{self_rmse:.6f}",
    )

    print(
        "\nPERTURBATION RESPONSE RMSE:",
        (
            f"{response_rmse:.6f} px"
            if response_rmse
            is not None
            else "N/A"
        ),
    )

    print(
        "Compensated-baseline RMSE:",
        (
            f"{compensated_rmse:.6f} px"
            if compensated_rmse
            is not None
            else "N/A"
        ),
    )

    print(
        "Recovered-vs-identity RMSE:",
        (
            f"{identity_rmse:.6f} px"
            if identity_rmse
            is not None
            else "N/A"
        ),
    )

    print(
        "Evaluation points:",
        eval_points,
    )

    print(
        "\nExpected affine:"
    )

    print(expected)

    print(
        "\nRecovered affine:"
    )

    print(recovered)

    print(
        "\nExpected parameters:"
    )

    print(
        json.dumps(
            decompose_affine(
                expected
            ),
            indent=2,
        )
    )

    print(
        "\nRecovered parameters:"
    )

    print(
        json.dumps(
            decompose_affine(
                recovered
            ),
            indent=2,
        )
    )

    return {
        "method":
            method_name,

        "matches":
            int(
                len(source_points)
            ),

        "inliers":
            int(
                len(src_in)
            ),

        "inlier_ratio":
            float(
                len(src_in)
                / len(source_points)
            ),

        "self_rmse_px":
            self_rmse,

        "perturbation_response_rmse_px":
            response_rmse,

        "compensated_baseline_rmse_px":
            compensated_rmse,

        "recovered_vs_identity_rmse_px":
            identity_rmse,

        "evaluation_points":
            int(eval_points),

        "expected_affine":
            expected.tolist(),

        "recovered_affine":
            recovered.tolist(),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print(
        "PAIR 004 — REAL CROSS-SENSOR "
        "CONTROLLED PERTURBATION DIAGNOSTIC"
    )
    print("=" * 78)

    source, source_raster_mask = (
        read_band(
            SOURCE_PATH
        )
    )

    reference, reference_raster_mask = (
        read_band(
            REFERENCE_PATH
        )
    )

    common_data, _ = (
        read_band(
            COMMON_MASK_PATH
        )
    )

    common = (
        common_data > 0
    )

    science_valid = (
        common
        & source_raster_mask
        & reference_raster_mask
        & np.isfinite(source)
        & np.isfinite(reference)
    )

    matcher_valid = (
        science_valid
        & (reference > 0)
    )

    matcher_valid = (
        ndimage.binary_erosion(
            matcher_valid,
            structure=np.ones(
                (3, 3),
                dtype=bool,
            ),
            iterations=(
                MASK_EROSION
            ),
            border_value=0,
        )
    )

    print(
        "\nShape:",
        source.shape,
    )

    print(
        "Science-valid:",
        f"{int(science_valid.sum()):,}",
    )

    print(
        "Matcher-valid:",
        f"{int(matcher_valid.sum()):,}",
    )

    source_intensity = (
        robust_uint8(
            source,
            matcher_valid,
        )
    )

    reference_intensity = (
        robust_uint8(
            reference,
            matcher_valid,
        )
    )

    source_intensity[
        ~matcher_valid
    ] = 0

    reference_intensity[
        ~matcher_valid
    ] = 0

    reference_gradient = (
        gradient_uint8(
            reference_intensity,
            matcher_valid,
        )
    )

    reference_gradient[
        ~matcher_valid
    ] = 0

    # --------------------------------------------------------
    # Original global baselines
    # --------------------------------------------------------

    baseline_loftr = (
        load_baseline_affine(
            "loftr",
            "gradient",
        )
    )

    baseline_lightglue = (
        load_baseline_affine(
            "lightglue",
            "gradient",
        )
    )

    print(
        "\nOriginal LoFTR-gradient baseline:"
    )
    print(
        baseline_loftr
    )

    print(
        "\nOriginal LightGlue-gradient baseline:"
    )
    print(
        baseline_lightglue
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

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    loftr = KF.LoFTR(
        pretrained="outdoor",
    ).eval().to(device)

    extractor = SuperPoint(
        max_num_keypoints=8192,
    ).eval().to(device)

    lightglue = LightGlue(
        features="superpoint",
    ).eval().to(device)

    h, w = source.shape

    all_results = []

    # ========================================================
    # PERTURBATIONS
    # ========================================================

    for perturbation in (
        PERTURBATIONS
    ):

        print("\n")
        print("=" * 78)
        print(
            perturbation["id"]
        )
        print("=" * 78)

        forward = (
            make_forward_affine(
                w,
                h,
                perturbation["tx"],
                perturbation["ty"],
                perturbation[
                    "rotation_deg"
                ],
                perturbation[
                    "scale"
                ],
            )
        )

        print(
            "\nInjected original->"
            "transformed NAC affine:"
        )

        print(forward)

        transformed_intensity = (
            cv2.warpAffine(
                source_intensity,
                forward,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=(
                    cv2.BORDER_CONSTANT
                ),
                borderValue=0,
            )
        )

        transformed_mask = (
            cv2.warpAffine(
                matcher_valid.astype(
                    np.uint8
                ),
                forward,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=(
                    cv2.BORDER_CONSTANT
                ),
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
                iterations=2,
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

        # ====================================================
        # LIGHTGLUE
        # ====================================================

        src_lg, ref_lg = (
            run_lightglue(
                extractor,
                lightglue,
                transformed_gradient,
                reference_gradient,
                device,
            )
        )

        result_lg = evaluate(
            "lightglue_gradient",
            src_lg,
            ref_lg,
            transformed_mask,
            matcher_valid,
            baseline_lightglue,
            forward,
        )

        # ====================================================
        # LOFTR
        # ====================================================

        src_lf, ref_lf = (
            run_loftr(
                loftr,
                transformed_gradient,
                reference_gradient,
                device,
            )
        )

        result_lf = evaluate(
            "loftr_gradient",
            src_lf,
            ref_lf,
            transformed_mask,
            matcher_valid,
            baseline_loftr,
            forward,
        )

        all_results.append(
            {
                "perturbation":
                    perturbation,

                "forward_affine":
                    forward.tolist(),

                "lightglue":
                    result_lg,

                "loftr":
                    result_lf,
            }
        )

    # ========================================================
    # SAVE
    # ========================================================

    output_json = (
        OUT_DIR
        / "pair004_crosssensor_perturbation.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for item in all_results:

        print(
            "\n"
            + item[
                "perturbation"
            ]["id"]
        )

        for key in [
            "lightglue",
            "loftr",
        ]:

            result = item[key]

            if result is None:
                print(
                    f"  {key}: FAILED"
                )
                continue

            print(
                f"  {key:<10} "
                f"inliers="
                f"{result['inliers']:<5} "
                f"response_RMSE="
                f"{result['perturbation_response_rmse_px']:.4f}px "
                f"identity_RMSE="
                f"{result['recovered_vs_identity_rmse_px']:.4f}px"
            )

    print(
        "\nOutput:"
    )

    print(
        output_json
    )

    print(
        "\nINTERPRETATION:"
    )

    print(
        "Small perturbation-response RMSE means "
        "the matcher actually followed the known "
        "synthetic displacement of the real NAC terrain."
    )

    print(
        "A recovered transform that remains much "
        "closer to identity instead indicates "
        "position-biased matching."
    )

    print(
        "\nPAIR 004 CROSS-SENSOR "
        "PERTURBATION DIAGNOSTIC COMPLETE."
    )


if __name__ == "__main__":
    main()