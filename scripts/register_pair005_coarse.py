from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pair_005"
    / "canonical"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_005"
    / "coarse_registration"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


# SIFT global result recovered in the previous benchmark.
#
# Source = Kaguya
# Reference = TMC
#
# This maps Kaguya pixel coordinates → TMC pixel coordinates.
AFFINE = np.array(
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

    normalized = (
        image.astype(np.float32)
        - lo
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


def gradient_u8(
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

    out = np.zeros(
        image.shape,
        dtype=np.uint8,
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

    out[valid] = (
        norm[valid]
        * 255.0
    ).astype(np.uint8)

    return out


def resize_preview(
    image,
    max_dimension=2048,
):

    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_dimension / max(h, w),
    )

    if scale >= 1.0:
        return image

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

    return cv2.resize(
        image,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )


def save_gray(
    path,
    image,
):

    Image.fromarray(
        resize_preview(
            image
        )
    ).save(
        path
    )


def save_rgb(
    path,
    image,
):

    Image.fromarray(
        resize_preview(
            image
        )
    ).save(
        path
    )


def write_registered_tif(
    path,
    data,
    reference_profile,
):

    profile = (
        reference_profile.copy()
    )

    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
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
            data.astype(
                np.float32
            ),
            1,
        )


def write_mask(
    path,
    mask,
    reference_profile,
):

    profile = (
        reference_profile.copy()
    )

    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "deflate",
            "tiled": True,
        }
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            mask.astype(
                np.uint8
            ),
            1,
        )


def checkerboard(
    a,
    b,
    valid,
    block=256,
):

    h, w = a.shape

    yy, xx = np.indices(
        (
            h,
            w,
        )
    )

    choose_a = (
        (
            xx // block
            + yy // block
        )
        % 2
        == 0
    )

    result = np.where(
        choose_a,
        a,
        b,
    ).astype(
        np.uint8
    )

    result[
        ~valid
    ] = 0

    return result


def red_cyan(
    source,
    reference,
    valid,
):

    # Reference TMC = red
    # Kaguya = cyan

    rgb = np.zeros(
        (
            source.shape[0],
            source.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    rgb[..., 0] = (
        reference
    )

    rgb[..., 1] = (
        source
    )

    rgb[..., 2] = (
        source
    )

    rgb[
        ~valid
    ] = 0

    return rgb


def compute_gradient_ncc(
    source,
    reference,
    valid,
):

    source_grad = gradient_u8(
        source,
        valid,
    ).astype(
        np.float32
    )

    reference_grad = gradient_u8(
        reference,
        valid,
    ).astype(
        np.float32
    )

    use = (
        valid
        & (
            (
                source_grad > 0
            )
            | (
                reference_grad > 0
            )
        )
    )

    if int(
        use.sum()
    ) < 100:

        return None

    a = (
        source_grad[
            use
        ]
    )

    b = (
        reference_grad[
            use
        ]
    )

    a = (
        a - a.mean()
    )

    b = (
        b - b.mean()
    )

    denom = (
        np.sqrt(
            np.sum(
                a * a
            )
        )
        * np.sqrt(
            np.sum(
                b * b
            )
        )
    )

    if denom <= 0:
        return None

    return float(
        np.sum(
            a * b
        )
        / denom
    )


def main():

    print("=" * 78)
    print(
        "PAIR 005 — COARSE SIFT REGISTRATION"
    )
    print("=" * 78)

    print(
        "\nAffine Kaguya → TMC:"
    )

    print(
        AFFINE
    )

    with rasterio.open(
        SOURCE_PATH
    ) as src_ds:

        source = src_ds.read(
            1
        ).astype(
            np.float32
        )

    with rasterio.open(
        REFERENCE_PATH
    ) as ref_ds:

        reference = ref_ds.read(
            1
        ).astype(
            np.float32
        )

        reference_profile = (
            ref_ds.profile.copy()
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

    if not (
        source.shape
        == reference.shape
        == science_mask.shape
        == matcher_mask.shape
    ):

        raise RuntimeError(
            "Canonical input shapes differ."
        )

    h, w = source.shape

    print(
        "\nCanonical shape:",
        h,
        "x",
        w,
    )

    # --------------------------------------------------------
    # SOURCE VALID MASK
    # --------------------------------------------------------

    source_valid = (
        science_mask
        & np.isfinite(source)
    )

    # --------------------------------------------------------
    # WARP KAGUYA → TMC PIXEL FRAME
    #
    # OpenCV warpAffine with WARP_INVERSE_MAP omitted expects M
    # describing source → destination geometry.
    # --------------------------------------------------------

    source_for_warp = np.nan_to_num(
        source,
        nan=0.0,
    ).astype(
        np.float32
    )

    registered = cv2.warpAffine(
        source_for_warp,
        AFFINE,
        (
            w,
            h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    registered_valid = cv2.warpAffine(
        source_valid.astype(
            np.uint8
        ),
        AFFINE,
        (
            w,
            h,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    # Reference must independently contain science data.
    reference_valid = (
        science_mask
        & np.isfinite(reference)
    )

    registered_common = (
        registered_valid
        & reference_valid
    )

    registered[
        ~registered_valid
    ] = np.nan

    print("\n" + "=" * 78)
    print(
        "REGISTERED VALIDITY"
    )
    print("=" * 78)

    print(
        "Source valid before warp:",
        f"{int(source_valid.sum()):,}",
    )

    print(
        "Source valid after warp:",
        f"{int(registered_valid.sum()):,}",
    )

    print(
        "Registered common valid:",
        f"{int(registered_common.sum()):,}",
    )

    print(
        "Registered common / reference science:",
        f"{registered_common.sum() / reference_valid.sum():.8f}",
    )

    # --------------------------------------------------------
    # SAVE REGISTERED RASTER
    # --------------------------------------------------------

    registered_path = (
        OUT_DIR
        / "kaguya_registered_coarse_to_tmc2.tif"
    )

    valid_path = (
        OUT_DIR
        / "registered_common_valid_mask.tif"
    )

    write_registered_tif(
        registered_path,
        registered,
        reference_profile,
    )

    write_mask(
        valid_path,
        registered_common,
        reference_profile,
    )

    # --------------------------------------------------------
    # VISUAL NORMALIZATION
    # --------------------------------------------------------

    source_before_vis = robust_normalize(
        source,
        science_mask,
    )

    source_after_vis = robust_normalize(
        registered,
        registered_common,
    )

    reference_before_vis = robust_normalize(
        reference,
        science_mask,
    )

    reference_after_vis = robust_normalize(
        reference,
        registered_common,
    )

    # --------------------------------------------------------
    # BEFORE QA
    # --------------------------------------------------------

    before_checker = checkerboard(
        source_before_vis,
        reference_before_vis,
        science_mask,
    )

    before_rc = red_cyan(
        source_before_vis,
        reference_before_vis,
        science_mask,
    )

    # --------------------------------------------------------
    # AFTER QA
    # --------------------------------------------------------

    after_checker = checkerboard(
        source_after_vis,
        reference_after_vis,
        registered_common,
    )

    after_rc = red_cyan(
        source_after_vis,
        reference_after_vis,
        registered_common,
    )

    after_overlay = (
        0.5
        * source_after_vis.astype(
            np.float32
        )
        + 0.5
        * reference_after_vis.astype(
            np.float32
        )
    )

    after_overlay = np.clip(
        after_overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    after_overlay[
        ~registered_common
    ] = 0

    # --------------------------------------------------------
    # SAVE PREVIEWS
    # --------------------------------------------------------

    save_gray(
        OUT_DIR
        / "before_checkerboard.png",
        before_checker,
    )

    save_rgb(
        OUT_DIR
        / "before_red_cyan.png",
        before_rc,
    )

    save_gray(
        OUT_DIR
        / "after_checkerboard.png",
        after_checker,
    )

    save_rgb(
        OUT_DIR
        / "after_red_cyan.png",
        after_rc,
    )

    save_gray(
        OUT_DIR
        / "after_overlay_50_50.png",
        after_overlay,
    )

    save_gray(
        OUT_DIR
        / "registered_kaguya_preview.png",
        source_after_vis,
    )

    # --------------------------------------------------------
    # SIMPLE INDEPENDENT IMAGE-SIMILARITY DIAGNOSTIC
    #
    # Not ground truth, but useful before/after evidence.
    # --------------------------------------------------------

    before_ncc = compute_gradient_ncc(
        source_before_vis,
        reference_before_vis,
        matcher_mask,
    )

    registered_matcher = cv2.warpAffine(
        matcher_mask.astype(
            np.uint8
        ),
        AFFINE,
        (
            w,
            h,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    after_metric_mask = (
        registered_common
        & registered_matcher
    )

    after_ncc = compute_gradient_ncc(
        source_after_vis,
        reference_after_vis,
        after_metric_mask,
    )

    print("\n" + "=" * 78)
    print(
        "GRADIENT NCC DIAGNOSTIC"
    )
    print("=" * 78)

    print(
        "Before registration:",
        before_ncc,
    )

    print(
        "After registration :",
        after_ncc,
    )

    if (
        before_ncc is not None
        and after_ncc is not None
    ):

        print(
            "Change:",
            after_ncc
            - before_ncc,
        )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "pair_id":
            "pair_005",
        "stage":
            "coarse_registration",
        "source":
            "SELENE/Kaguya TC",
        "reference":
            "Chandrayaan-2 TMC-2",
        "canonical_resolution_m":
            10.0,
        "affine_source_to_reference":
            AFFINE.tolist(),
        "coarse_translation": {
            "tx_px":
                float(
                    AFFINE[
                        0,
                        2
                    ]
                ),
            "ty_px":
                float(
                    AFFINE[
                        1,
                        2
                    ]
                ),
            "approx_tx_m":
                float(
                    AFFINE[
                        0,
                        2
                    ]
                    * 10.0
                ),
            "approx_ty_m":
                float(
                    AFFINE[
                        1,
                        2
                    ]
                    * 10.0
                ),
        },
        "sift_global_evidence": {
            "candidates":
                2332,
            "ransac_inliers":
                2226,
            "inlier_ratio":
                0.954545,
            "ransac_self_rmse_px":
                0.921611,
            "coverage":
                48 / 54,
        },
        "independent_lightglue_evidence": {
            "tx_px":
                422.902,
            "sift_lightglue_tx_difference_px":
                abs(
                    float(
                        AFFINE[
                            0,
                            2
                        ]
                    )
                    - 422.902
                ),
        },
        "registered_validity": {
            "registered_source_pixels":
                int(
                    registered_valid.sum()
                ),
            "common_pixels":
                int(
                    registered_common.sum()
                ),
        },
        "gradient_ncc": {
            "before":
                before_ncc,
            "after":
                after_ncc,
        },
        "metric_note":
            (
                "SIFT RANSAC residual and gradient NCC "
                "are not independent absolute ground-truth "
                "registration accuracy."
            ),
    }

    metrics_path = (
        OUT_DIR
        / "pair005_coarse_registration_metrics.json"
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

    print("\n" + "=" * 78)
    print(
        "OUTPUTS"
    )
    print("=" * 78)

    for path in [
        registered_path,
        valid_path,
        OUT_DIR
        / "before_checkerboard.png",
        OUT_DIR
        / "before_red_cyan.png",
        OUT_DIR
        / "after_checkerboard.png",
        OUT_DIR
        / "after_red_cyan.png",
        OUT_DIR
        / "after_overlay_50_50.png",
        metrics_path,
    ]:

        print(
            path
        )

    print("\n" + "=" * 78)
    print(
        "PAIR 005 COARSE REGISTRATION COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()