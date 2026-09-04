from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

KAGUYA = (
    ROOT
    / "data"
    / "raw"
    / "kaguya_tc"
    / "2008_02_17"
    / "DTMTCO_03_01597S711E1002PS_img.lbl"
)

TMC = (
    ROOT
    / "data"
    / "raw"
    / "tmc2"
    / "2023_10_25"
    / "data"
    / "derived"
    / "20231025"
    / "ch2_tmc_ndn_20231025T1956513800_d_oth_d18.tif"
)

OUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pair_005"
    / "canonical"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


KAGUYA_PRODUCT = (
    "DTMTCO_03_01597S711E1002PS_IMG"
)

KAGUYA_SCALE_FACTOR = 0.01
KAGUYA_OFFSET = 0.0

EROSION_RADIUS = 5


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

    lo = np.percentile(
        values,
        low,
    )

    hi = np.percentile(
        values,
        high,
    )

    if hi <= lo:
        hi = lo + 1.0

    scaled = (
        image - lo
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


def resize_preview(
    image,
    max_dimension=1800,
):

    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_dimension / max(h, w),
    )

    if scale >= 1.0:
        return image

    new_w = int(
        round(w * scale)
    )

    new_h = int(
        round(h * scale)
    )

    return cv2.resize(
        image,
        (new_w, new_h),
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


def write_float_tif(
    path,
    data,
    crs,
    transform,
):

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            data.astype(np.float32),
            1,
        )


def write_mask_tif(
    path,
    mask,
    crs,
    transform,
):

    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            mask.astype(np.uint8),
            1,
        )


def derive_tmc_swath_footprint(
    tmc_on_grid,
    kaguya_valid,
):
    """
    Infer the actual TMC observation swath.

    TMC has no explicit nodata value:
        - zeros outside the image swath are fill
        - zeros inside the swath can be real lunar shadow

    We therefore use positive TMC pixels only as SEEDS for locating
    the image swath, then derive the enclosing footprint.

    A convex hull is suitable here because this local section of the
    push-broom TMC observation is approximately a continuous strip.
    """

    seed = (
        kaguya_valid
        & np.isfinite(tmc_on_grid)
        & (tmc_on_grid > 0)
    )

    seed_u8 = (
        seed.astype(np.uint8)
        * 255
    )

    # Bridge tiny gaps in illuminated support without trying
    # to fill the large genuine shadow regions.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    cleaned = cv2.morphologyEx(
        seed_u8,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError(
            "Could not derive TMC footprint: "
            "no positive-data contours found."
        )

    # Ignore tiny isolated fragments.
    areas = np.array(
        [
            cv2.contourArea(c)
            for c in contours
        ],
        dtype=np.float64,
    )

    max_area = float(
        np.max(areas)
    )

    major_contours = [
        c
        for c, area in zip(
            contours,
            areas,
        )
        if area >= max(
            1000.0,
            0.005 * max_area,
        )
    ]

    if not major_contours:
        raise RuntimeError(
            "Could not identify major TMC contours."
        )

    all_points = np.vstack(
        major_contours
    )

    hull = cv2.convexHull(
        all_points
    )

    footprint = np.zeros(
        seed.shape,
        dtype=np.uint8,
    )

    cv2.fillConvexPoly(
        footprint,
        hull,
        1,
    )

    footprint = (
        footprint > 0
    )

    # Kaguya itself has an irregular valid polygon.
    footprint &= kaguya_valid

    return (
        seed,
        footprint,
        hull,
        len(contours),
        len(major_contours),
    )


def crop_to_mask(
    mask,
):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError(
            "Cannot crop: overlap mask is empty."
        )

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1

    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    return (
        x0,
        y0,
        x1,
        y1,
    )


def main():

    print("=" * 78)
    print(
        "PAIR 005 — BUILD CORRECTED "
        "KAGUYA TC ↔ TMC-2 CANONICAL"
    )
    print("=" * 78)

    with rasterio.open(
        KAGUYA
    ) as kag, rasterio.open(
        TMC
    ) as tmc:

        print("\n" + "=" * 78)
        print("INPUT")
        print("=" * 78)

        print(
            "Kaguya shape:",
            kag.height,
            "x",
            kag.width,
        )

        print(
            "Kaguya resolution:",
            kag.res,
        )

        print(
            "Kaguya bounds:",
            kag.bounds,
        )

        print(
            "\nTMC shape:",
            tmc.height,
            "x",
            tmc.width,
        )

        print(
            "TMC resolution:",
            tmc.res,
        )

        print(
            "TMC nodata:",
            tmc.nodata,
        )

        # ------------------------------------------------------------
        # KAGUYA
        # ------------------------------------------------------------

        kag_raw = kag.read(1)

        kag_valid = (
            kag.read_masks(1) > 0
        )

        kag_valid &= np.isfinite(
            kag_raw
        )

        kag_radiance = (
            kag_raw.astype(
                np.float32
            )
            * KAGUYA_SCALE_FACTOR
            + KAGUYA_OFFSET
        )

        # ------------------------------------------------------------
        # TMC → KAGUYA NATIVE 10 m GRID
        # ------------------------------------------------------------

        print("\n" + "=" * 78)
        print("TMC → KAGUYA 10 m GRID")
        print("=" * 78)

        tmc_10m = np.full(
            (
                kag.height,
                kag.width,
            ),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(
                tmc,
                1,
            ),
            destination=tmc_10m,
            src_transform=tmc.transform,
            src_crs=tmc.crs,
            src_nodata=None,
            dst_transform=kag.transform,
            dst_crs=kag.crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )

        # ------------------------------------------------------------
        # RECOVER TRUE TMC SWATH
        # ------------------------------------------------------------

        print("\n" + "=" * 78)
        print("DERIVING TMC SWATH FOOTPRINT")
        print("=" * 78)

        (
            tmc_positive_seed,
            tmc_footprint,
            hull,
            raw_contours,
            major_contours,
        ) = derive_tmc_swath_footprint(
            tmc_10m,
            kag_valid,
        )

        print(
            "Positive TMC seed pixels:",
            f"{int(tmc_positive_seed.sum()):,}",
        )

        print(
            "Raw external contours:",
            raw_contours,
        )

        print(
            "Major contours retained:",
            major_contours,
        )

        print(
            "Derived TMC footprint pixels:",
            f"{int(tmc_footprint.sum()):,}",
        )

        print(
            "Hull vertices:",
            len(hull),
        )

        # ------------------------------------------------------------
        # TRUE SCIENCE OVERLAP
        #
        # Important:
        # zeros INSIDE tmc_footprint are retained as science data.
        # ------------------------------------------------------------

        science_full = (
            kag_valid
            & tmc_footprint
            & np.isfinite(tmc_10m)
        )

        if not np.any(
            science_full
        ):
            raise RuntimeError(
                "Corrected science overlap is empty."
            )

        positive_inside = (
            science_full
            & (tmc_10m > 0)
        )

        shadow_inside = (
            science_full
            & (tmc_10m == 0)
        )

        print("\n" + "=" * 78)
        print("CORRECTED SCIENCE OVERLAP")
        print("=" * 78)

        print(
            "Kaguya valid pixels:",
            f"{int(kag_valid.sum()):,}",
        )

        print(
            "Science-overlap pixels:",
            f"{int(science_full.sum()):,}",
        )

        print(
            "Overlap / Kaguya-valid:",
            f"{science_full.sum() / kag_valid.sum():.8f}",
        )

        print(
            "Positive TMC pixels inside footprint:",
            f"{int(positive_inside.sum()):,}",
        )

        print(
            "Zero TMC pixels retained as shadow:",
            f"{int(shadow_inside.sum()):,}",
        )

        print(
            "Shadow fraction within footprint:",
            f"{shadow_inside.sum() / science_full.sum():.8f}",
        )

        # ------------------------------------------------------------
        # CROP TO TRUE OVERLAP BOUNDING BOX
        # ------------------------------------------------------------

        (
            x0,
            y0,
            x1,
            y1,
        ) = crop_to_mask(
            science_full
        )

        print("\n" + "=" * 78)
        print("OVERLAP CROP")
        print("=" * 78)

        print(
            "Full-grid bbox:",
            f"x={x0}..{x1 - 1}, "
            f"y={y0}..{y1 - 1}",
        )

        width = (
            x1 - x0
        )

        height = (
            y1 - y0
        )

        print(
            "Canonical shape:",
            height,
            "x",
            width,
        )

        crop_window = Window(
            x0,
            y0,
            width,
            height,
        )

        canonical_transform = (
            window_transform(
                crop_window,
                kag.transform,
            )
        )

        kag_crop = (
            kag_radiance[
                y0:y1,
                x0:x1,
            ]
            .copy()
        )

        tmc_crop = (
            tmc_10m[
                y0:y1,
                x0:x1,
            ]
            .copy()
        )

        science = (
            science_full[
                y0:y1,
                x0:x1,
            ]
            .copy()
        )

        footprint_crop = (
            tmc_footprint[
                y0:y1,
                x0:x1,
            ]
            .copy()
        )

        # NaN outside actual science overlap.
        kag_crop[
            ~science
        ] = np.nan

        tmc_crop[
            ~science
        ] = np.nan

        # ------------------------------------------------------------
        # MATCHER MASK
        #
        # Science mask retains shadows.
        # Matcher excludes exact-zero TMC patches.
        # ------------------------------------------------------------

        matcher_base = (
            science
            & (tmc_crop > 0)
        )

        kernel_size = (
            2 * EROSION_RADIUS
            + 1
        )

        erosion_kernel = np.ones(
            (
                kernel_size,
                kernel_size,
            ),
            dtype=np.uint8,
        )

        matcher = (
            cv2.erode(
                matcher_base.astype(
                    np.uint8
                ),
                erosion_kernel,
                iterations=1,
            )
            > 0
        )

        total = (
            height
            * width
        )

        science_count = int(
            science.sum()
        )

        matcher_base_count = int(
            matcher_base.sum()
        )

        matcher_count = int(
            matcher.sum()
        )

        shadow_count = int(
            np.sum(
                science
                & (tmc_crop == 0)
            )
        )

        print("\n" + "=" * 78)
        print("CANONICAL VALIDITY")
        print("=" * 78)

        print(
            "Total canonical pixels:",
            f"{total:,}",
        )

        print(
            "Science valid:",
            f"{science_count:,}",
            f"({science_count / total:.8f})",
        )

        print(
            "TMC zeros retained as shadow:",
            f"{shadow_count:,}",
            f"({shadow_count / science_count:.8f})",
        )

        print(
            "Matcher base:",
            f"{matcher_base_count:,}",
        )

        print(
            f"Matcher after {EROSION_RADIUS}px erosion:",
            f"{matcher_count:,}",
        )

        print(
            "Matcher / science:",
            f"{matcher_count / science_count:.8f}",
        )

        # ------------------------------------------------------------
        # WRITE
        # ------------------------------------------------------------

        print("\n" + "=" * 78)
        print("WRITING CORRECTED CANONICAL PRODUCTS")
        print("=" * 78)

        files = {
            "kaguya": (
                OUT_DIR
                / "kaguya_tc_source.tif"
            ),
            "tmc": (
                OUT_DIR
                / "tmc2_reference.tif"
            ),
            "science": (
                OUT_DIR
                / "common_valid_mask.tif"
            ),
            "matcher": (
                OUT_DIR
                / "matcher_valid_mask.tif"
            ),
            "footprint": (
                OUT_DIR
                / "tmc_swath_footprint_mask.tif"
            ),
        }

        write_float_tif(
            files["kaguya"],
            kag_crop,
            kag.crs,
            canonical_transform,
        )

        write_float_tif(
            files["tmc"],
            tmc_crop,
            kag.crs,
            canonical_transform,
        )

        write_mask_tif(
            files["science"],
            science,
            kag.crs,
            canonical_transform,
        )

        write_mask_tif(
            files["matcher"],
            matcher,
            kag.crs,
            canonical_transform,
        )

        write_mask_tif(
            files["footprint"],
            footprint_crop,
            kag.crs,
            canonical_transform,
        )

        for path in files.values():
            print(path)

        # ------------------------------------------------------------
        # PREVIEWS
        # ------------------------------------------------------------

        kag_vis = robust_normalize(
            kag_crop,
            science,
        )

        tmc_vis = robust_normalize(
            tmc_crop,
            science,
        )

        kag_vis[
            ~science
        ] = 0

        tmc_vis[
            ~science
        ] = 0

        save_gray(
            OUT_DIR
            / "kaguya_tc_preview.png",
            kag_vis,
        )

        save_gray(
            OUT_DIR
            / "tmc2_reference_preview.png",
            tmc_vis,
        )

        divider = np.full(
            (
                height,
                10,
            ),
            255,
            dtype=np.uint8,
        )

        side = np.concatenate(
            [
                kag_vis,
                divider,
                tmc_vis,
            ],
            axis=1,
        )

        save_gray(
            OUT_DIR
            / "canonical_side_by_side.png",
            side,
        )

        # Red/cyan:
        # reference TMC = red
        # source Kaguya = cyan
        rgb = np.zeros(
            (
                height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

        rgb[..., 0] = (
            tmc_vis
        )

        rgb[..., 1] = (
            kag_vis
        )

        rgb[..., 2] = (
            kag_vis
        )

        rgb[
            ~science
        ] = 0

        save_rgb(
            OUT_DIR
            / "canonical_red_cyan.png",
            rgb,
        )

        # 50/50 gray overlay
        overlay = (
            0.5
            * kag_vis.astype(
                np.float32
            )
            + 0.5
            * tmc_vis.astype(
                np.float32
            )
        )

        overlay = np.clip(
            overlay,
            0,
            255,
        ).astype(np.uint8)

        overlay[
            ~science
        ] = 0

        save_gray(
            OUT_DIR
            / "canonical_overlay_50_50.png",
            overlay,
        )

        # Checkerboard
        yy, xx = np.indices(
            (
                height,
                width,
            )
        )

        block = 256

        choose_kag = (
            (
                xx // block
                + yy // block
            )
            % 2
            == 0
        )

        checker = np.where(
            choose_kag,
            kag_vis,
            tmc_vis,
        ).astype(np.uint8)

        checker[
            ~science
        ] = 0

        save_gray(
            OUT_DIR
            / "canonical_checkerboard.png",
            checker,
        )

        save_gray(
            OUT_DIR
            / "matcher_valid_mask_preview.png",
            (
                matcher.astype(
                    np.uint8
                )
                * 255
            ),
        )

        save_gray(
            OUT_DIR
            / "tmc_swath_footprint_preview.png",
            (
                footprint_crop.astype(
                    np.uint8
                )
                * 255
            ),
        )

        # ------------------------------------------------------------
        # METADATA
        # ------------------------------------------------------------

        bounds = rasterio.transform.array_bounds(
            height,
            width,
            canonical_transform,
        )

        metadata = {
            "pair_id": "pair_005",
            "source": {
                "mission": "SELENE/Kaguya",
                "instrument": "Terrain Camera (TC)",
                "product": KAGUYA_PRODUCT,
                "observation_time":
                    "2008-02-17T03:51:38.114635",
                "native_resolution_m": 10.0,
                "incidence_angle_deg": 77.267,
                "emission_angle_deg": 14.105,
                "phase_angle_deg": 87.070,
                "solar_azimuth_deg": 311.334,
            },
            "reference": {
                "mission": "Chandrayaan-2",
                "instrument": "TMC-2",
                "observation_date": "2023-10-25",
                "native_resolution_m": 5.0,
                "incidence_angle_deg": 82.949194,
                "sun_azimuth_deg": 286.401572,
            },
            "canonical": {
                "width": width,
                "height": height,
                "resolution_m": 10.0,
                "bounds": {
                    "left": bounds[0],
                    "bottom": bounds[1],
                    "right": bounds[2],
                    "top": bounds[3],
                },
                "crs_wkt": kag.crs.to_wkt(),
            },
            "validity": {
                "science_valid_pixels":
                    science_count,
                "science_valid_ratio":
                    science_count / total,
                "tmc_shadow_zero_pixels":
                    shadow_count,
                "matcher_base_pixels":
                    matcher_base_count,
                "matcher_valid_pixels":
                    matcher_count,
                "matcher_valid_ratio_science":
                    matcher_count
                    / science_count,
                "matcher_erosion_radius_px":
                    EROSION_RADIUS,
            },
            "tmc_swath_recovery": {
                "method":
                    "positive-data support + "
                    "major external contours + "
                    "convex hull",
                "positive_seed_pixels":
                    int(
                        tmc_positive_seed.sum()
                    ),
                "raw_external_contours":
                    raw_contours,
                "major_contours":
                    major_contours,
                "footprint_pixels_full_grid":
                    int(
                        tmc_footprint.sum()
                    ),
                "note":
                    "TMC zero outside reconstructed "
                    "swath is fill; TMC zero inside "
                    "swath is retained as possible "
                    "lunar shadow.",
            },
            "geometry": {
                "native_resolution_ratio": 2.0,
                "incidence_difference_deg":
                    abs(
                        82.949194
                        - 77.267
                    ),
                "solar_azimuth_difference_deg":
                    min(
                        abs(
                            286.401572
                            - 311.334
                        ),
                        360.0
                        - abs(
                            286.401572
                            - 311.334
                        ),
                    ),
            },
        }

        metadata_path = (
            OUT_DIR
            / "metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
            )

        print(metadata_path)

        print("\n" + "=" * 78)
        print("PREVIEWS")
        print("=" * 78)

        for name in [
            "canonical_side_by_side.png",
            "canonical_red_cyan.png",
            "canonical_checkerboard.png",
            "matcher_valid_mask_preview.png",
            "tmc_swath_footprint_preview.png",
        ]:

            print(
                OUT_DIR
                / name
            )

        print("\n" + "=" * 78)
        print(
            "PAIR 005 CORRECTED "
            "CANONICAL BUILD COMPLETE"
        )
        print("=" * 78)


if __name__ == "__main__":
    main()