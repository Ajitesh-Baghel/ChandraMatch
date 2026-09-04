from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


ROOT = Path(__file__).resolve().parents[1]

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pair_007"
    / "canonical"
)

SOURCE_PATH = (
    PAIR_DIR
    / "source_lro_nac.tif"
)

REFERENCE_PATH = (
    PAIR_DIR
    / "reference_tmc2.tif"
)

MASK_PATH = (
    PAIR_DIR
    / "matcher_mask.tif"
)

GLOBAL_JSON = (
    ROOT
    / "results"
    / "pair_007"
    / "global_benchmark"
    / "pair007_global_benchmark.json"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_007"
    / "large_offset_diagnostic"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def normalize_u8(image, mask):
    valid = (
        mask
        & np.isfinite(image)
    )

    out = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    if not np.any(valid):
        return out

    lo, hi = np.percentile(
        image[valid],
        [1, 99],
    )

    if hi <= lo:
        hi = lo + 1.0

    f = (
        image.astype(np.float32)
        - lo
    ) / (
        hi - lo
    )

    f = np.clip(
        f,
        0.0,
        1.0,
    )

    out[valid] = (
        f[valid] * 255.0
    ).astype(np.uint8)

    return out


def gradient(image):
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

    return cv2.magnitude(
        gx,
        gy,
    )


def ncc(a, b, mask):
    valid = (
        mask
        & np.isfinite(a)
        & np.isfinite(b)
    )

    if valid.sum() < 100:
        return None

    x = a[valid].astype(
        np.float64
    )

    y = b[valid].astype(
        np.float64
    )

    x -= x.mean()
    y -= y.mean()

    denom = (
        np.sqrt(
            np.sum(x * x)
            * np.sum(y * y)
        )
    )

    if denom <= 1e-12:
        return None

    return float(
        np.sum(x * y)
        / denom
    )


def find_loftr_intensity_affine():
    with open(
        GLOBAL_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    for result in data:
        if (
            str(
                result.get(
                    "method",
                    ""
                )
            ).lower()
            == "loftr"
            and
            str(
                result.get(
                    "representation",
                    ""
                )
            ).lower()
            == "intensity"
        ):
            return np.asarray(
                result["affine"],
                dtype=np.float64,
            )

    raise RuntimeError(
        "LoFTR intensity affine "
        "not found in Pair007 "
        "global benchmark."
    )


def warp_affine(
    image,
    affine,
    shape,
    interpolation,
):
    return cv2.warpAffine(
        image,
        affine.astype(
            np.float64
        ),
        (
            shape[1],
            shape[0],
        ),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def make_checkerboard(
    a,
    b,
    block=128,
):
    h, w = a.shape

    result = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    for y in range(
        0,
        h,
        block,
    ):
        for x in range(
            0,
            w,
            block,
        ):
            y1 = min(
                y + block,
                h,
            )

            x1 = min(
                x + block,
                w,
            )

            gy = y // block
            gx = x // block

            if (
                gx + gy
            ) % 2 == 0:
                result[
                    y:y1,
                    x:x1
                ] = a[
                    y:y1,
                    x:x1
                ]
            else:
                result[
                    y:y1,
                    x:x1
                ] = b[
                    y:y1,
                    x:x1
                ]

    return result


def local_grid_ncc(
    a,
    b,
    mask,
    rows=4,
    cols=8,
):
    h, w = a.shape

    values = []

    for gy in range(rows):
        y0 = (
            gy * h // rows
        )

        y1 = (
            (gy + 1)
            * h // rows
        )

        for gx in range(cols):
            x0 = (
                gx * w // cols
            )

            x1 = (
                (gx + 1)
                * w // cols
            )

            cell_mask = mask[
                y0:y1,
                x0:x1
            ]

            if (
                cell_mask.sum()
                < 1000
            ):
                continue

            score = ncc(
                a[
                    y0:y1,
                    x0:x1
                ],
                b[
                    y0:y1,
                    x0:x1
                ],
                cell_mask,
            )

            if score is not None:
                values.append(
                    {
                        "row": gy,
                        "col": gx,
                        "ncc": score,
                        "pixels":
                            int(
                                cell_mask.sum()
                            ),
                    }
                )

    return values


def main():
    print("=" * 78)
    print(
        "PAIR 007 — LARGE-OFFSET "
        "INDEPENDENT DIAGNOSTIC"
    )
    print("=" * 78)

    with rasterio.open(
        SOURCE_PATH
    ) as ds:
        source = (
            ds.read(1)
            .astype(np.float32)
        )

        profile = (
            ds.profile.copy()
        )

        source_nodata = (
            ds.nodata
        )

    with rasterio.open(
        REFERENCE_PATH
    ) as ds:
        reference = (
            ds.read(1)
            .astype(np.float32)
        )

    with rasterio.open(
        MASK_PATH
    ) as ds:
        base_mask = (
            ds.read(1)
            > 0
        )

    affine = (
        find_loftr_intensity_affine()
    )

    print()
    print(
        "Global LoFTR-intensity affine:"
    )

    print(
        affine
    )

    print()

    source_valid = (
        np.isfinite(source)
    )

    if source_nodata is not None:
        source_valid &= (
            source
            != source_nodata
        )

    reference_valid = (
        np.isfinite(reference)
    )

    # --------------------------------------------------------
    # Warp source + validity
    # --------------------------------------------------------

    registered = warp_affine(
        np.nan_to_num(
            source,
            nan=0.0,
        ),
        affine,
        source.shape,
        cv2.INTER_LINEAR,
    )

    warped_valid = warp_affine(
        source_valid.astype(
            np.uint8
        ),
        affine,
        source.shape,
        cv2.INTER_NEAREST,
    ) > 0

    common_after = (
        warped_valid
        & reference_valid
        & base_mask
    )

    common_before = (
        source_valid
        & reference_valid
        & base_mask
    )

    # --------------------------------------------------------
    # Normalization + gradients
    # --------------------------------------------------------

    src_u8 = normalize_u8(
        source,
        common_before,
    )

    ref_before_u8 = normalize_u8(
        reference,
        common_before,
    )

    reg_u8 = normalize_u8(
        registered,
        common_after,
    )

    ref_after_u8 = normalize_u8(
        reference,
        common_after,
    )

    src_grad = gradient(
        src_u8
    )

    ref_before_grad = gradient(
        ref_before_u8
    )

    reg_grad = gradient(
        reg_u8
    )

    ref_after_grad = gradient(
        ref_after_u8
    )

    before_ncc = ncc(
        src_grad,
        ref_before_grad,
        common_before,
    )

    after_ncc = ncc(
        reg_grad,
        ref_after_grad,
        common_after,
    )

    print(
        "Common pixels before:",
        f"{int(common_before.sum()):,}",
    )

    print(
        "Common pixels after :",
        f"{int(common_after.sum()):,}",
    )

    print()

    print(
        "Gradient NCC before:",
        before_ncc,
    )

    print(
        "Gradient NCC after :",
        after_ncc,
    )

    if (
        before_ncc is not None
        and after_ncc is not None
    ):
        print(
            "NCC change        :",
            after_ncc
            - before_ncc,
        )

    # --------------------------------------------------------
    # Local NCC
    # --------------------------------------------------------

    local_before = (
        local_grid_ncc(
            src_grad,
            ref_before_grad,
            common_before,
        )
    )

    local_after = (
        local_grid_ncc(
            reg_grad,
            ref_after_grad,
            common_after,
        )
    )

    before_values = [
        x["ncc"]
        for x in local_before
    ]

    after_values = [
        x["ncc"]
        for x in local_after
    ]

    print()
    print(
        "Local valid cells before:",
        len(before_values),
    )

    print(
        "Local valid cells after :",
        len(after_values),
    )

    if before_values:
        print(
            "Local median NCC before:",
            float(
                np.median(
                    before_values
                )
            ),
        )

    if after_values:
        print(
            "Local median NCC after :",
            float(
                np.median(
                    after_values
                )
            ),
        )

        print(
            "Local NCC > 0.20 after:",
            sum(
                x > 0.20
                for x
                in after_values
            ),
            "/",
            len(after_values),
        )

    # --------------------------------------------------------
    # GeoTIFF
    # --------------------------------------------------------

    profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
    )

    output_tif = (
        OUT_DIR
        / "registered_loftr_candidate.tif"
    )

    with rasterio.open(
        output_tif,
        "w",
        **profile,
    ) as ds:
        ds.write(
            registered.astype(
                np.float32
            ),
            1,
        )

    # --------------------------------------------------------
    # QA visuals
    # --------------------------------------------------------

    overlay = (
        0.5
        * reg_u8.astype(
            np.float32
        )
        + 0.5
        * ref_after_u8.astype(
            np.float32
        )
    ).astype(
        np.uint8
    )

    checker = make_checkerboard(
        ref_after_u8,
        reg_u8,
    )

    red_cyan = np.zeros(
        (
            source.shape[0],
            source.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    red_cyan[:, :, 2] = (
        ref_after_u8
    )

    red_cyan[:, :, 1] = (
        reg_u8
    )

    red_cyan[:, :, 0] = (
        reg_u8
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / "registered_overlay.png"
        ),
        overlay,
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / "registered_checkerboard.png"
        ),
        checker,
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / "registered_red_cyan.png"
        ),
        red_cyan,
    )

    metrics = {
        "pair_id":
            "pair_007",

        "diagnostic":
            "large_offset_visual_and_similarity",

        "affine":
            affine.tolist(),

        "common_pixels_before":
            int(
                common_before.sum()
            ),

        "common_pixels_after":
            int(
                common_after.sum()
            ),

        "gradient_ncc_before":
            before_ncc,

        "gradient_ncc_after":
            after_ncc,

        "gradient_ncc_change":
            (
                None
                if (
                    before_ncc is None
                    or after_ncc is None
                )
                else
                float(
                    after_ncc
                    - before_ncc
                )
            ),

        "local_before":
            local_before,

        "local_after":
            local_after,

        "note":
            (
                "This is an independent "
                "appearance/alignment diagnostic. "
                "It does not convert Pair007 into "
                "a validated registration and does "
                "not replace perturbation testing."
            ),
    }

    metrics_path = (
        OUT_DIR
        / "pair007_large_offset_diagnostic.json"
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

    print()
    print("Outputs:")

    print(
        output_tif
    )

    print(
        OUT_DIR
        / "registered_overlay.png"
    )

    print(
        OUT_DIR
        / "registered_checkerboard.png"
    )

    print(
        OUT_DIR
        / "registered_red_cyan.png"
    )

    print(
        metrics_path
    )

    print()
    print(
        "PAIR007 DIAGNOSTIC COMPLETE."
    )


if __name__ == "__main__":
    main()