from pathlib import Path
import json
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject, Resampling


ROOT = Path(r"E:\SIH26166")

TMC_PATH = (
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

NAC_PATH = (
    ROOT
    / "data"
    / "processed"
    / "pair_006"
    / "lro_nac_isis"
    / "M141704260RE_map_5m.tif"
)

OUT = ROOT / "data" / "processed" / "pair_006" / "canonical"
OUT.mkdir(parents=True, exist_ok=True)


def full_pixel_window(bounds, transform):
    w = from_bounds(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
        transform=transform,
    )

    c0 = math.floor(w.col_off)
    r0 = math.floor(w.row_off)

    c1 = math.ceil(w.col_off + w.width)
    r1 = math.ceil(w.row_off + w.height)

    return Window(
        col_off=c0,
        row_off=r0,
        width=c1 - c0,
        height=r1 - r0,
    )


def write_tif(path, array, transform, crs, nodata=None, dtype=None):
    if dtype is None:
        dtype = array.dtype

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
    }

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def robust_preview(arr, valid):
    out = np.zeros(arr.shape, dtype=np.uint8)

    vals = arr[valid & np.isfinite(arr)]

    if vals.size == 0:
        return out

    lo, hi = np.percentile(vals, [2, 98])

    if hi <= lo:
        hi = lo + 1.0

    scaled = (arr.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled, 0, 1)

    out = (scaled * 255).astype(np.uint8)
    out[~valid] = 0

    return out


def largest_swath_hull(positive_mask):
    """
    Recover approximate geometric TMC swath footprint.

    TMC has nodata=None and can contain genuine zero-valued shadows.
    Therefore TMC > 0 is used as a seed for the footprint, not as the
    final science-validity definition.
    """

    seed = positive_mask.astype(np.uint8) * 255

    # Stabilize small holes/gaps before finding the main footprint.
    kernel = np.ones((21, 21), np.uint8)
    closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise RuntimeError("Could not recover TMC geometric footprint.")

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)

    footprint = np.zeros_like(seed)
    cv2.fillConvexPoly(footprint, hull, 255)

    return footprint > 0, hull


def main():
    print("=" * 80)
    print("PAIR 006 — CANONICAL PAIR BUILDER")
    print("=" * 80)

    with rasterio.open(TMC_PATH) as tmc, rasterio.open(NAC_PATH) as nac:

        # ----------------------------------------------------------
        # 1. Build target grid from TMC.
        # ----------------------------------------------------------

        target_window = full_pixel_window(
            nac.bounds,
            tmc.transform,
        )

        target_transform = tmc.window_transform(target_window)

        width = int(target_window.width)
        height = int(target_window.height)

        print("\nTARGET TMC GRID")
        print("  window :", target_window)
        print("  shape  :", height, "x", width)
        print("  res    :", tmc.res)
        print("  transform:")
        print(target_transform)

        # ----------------------------------------------------------
        # 2. Read TMC directly on its native grid.
        # ----------------------------------------------------------

        tmc_arr = tmc.read(
            1,
            window=target_window,
            boundless=True,
            fill_value=0,
        )

        tmc_finite = np.isfinite(tmc_arr)
        tmc_positive = tmc_finite & (tmc_arr > 0)

        print("\nTMC")
        print(
            "  positive pixels:",
            int(tmc_positive.sum()),
            "/",
            tmc_positive.size,
            f"= {tmc_positive.mean():.6f}",
        )

        # ----------------------------------------------------------
        # 3. Reproject NAC to EXACT TMC grid.
        # ----------------------------------------------------------

        nac_on_tmc = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(nac, 1),
            destination=nac_on_tmc,
            src_transform=nac.transform,
            src_crs=nac.crs,
            src_nodata=nac.nodata,
            dst_transform=target_transform,
            dst_crs=tmc.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        # Reproject validity independently using nearest neighbour.
        nac_native = nac.read(1)

        nac_native_valid = np.isfinite(nac_native)

        if nac.nodata is not None:
            nac_native_valid &= nac_native != nac.nodata

        nac_valid_u8 = nac_native_valid.astype(np.uint8)

        nac_valid_on_tmc = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        reproject(
            source=nac_valid_u8,
            destination=nac_valid_on_tmc,
            src_transform=nac.transform,
            src_crs=nac.crs,
            dst_transform=target_transform,
            dst_crs=tmc.crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        nac_valid = (
            (nac_valid_on_tmc > 0)
            & np.isfinite(nac_on_tmc)
        )

        print("\nNAC ON TMC GRID")
        print(
            "  valid pixels:",
            int(nac_valid.sum()),
            "/",
            nac_valid.size,
            f"= {nac_valid.mean():.6f}",
        )

        # ----------------------------------------------------------
        # 4. Recover geometric TMC swath.
        #
        # We deliberately distinguish:
        #
        # geometric footprint:
        #   where the TMC observation exists, including real shadows.
        #
        # positive support:
        #   TMC > 0, useful for matcher support.
        # ----------------------------------------------------------

        tmc_footprint, hull = largest_swath_hull(tmc_positive)

        print("\nTMC GEOMETRIC FOOTPRINT")
        print(
            "  footprint pixels:",
            int(tmc_footprint.sum()),
            "/",
            tmc_footprint.size,
            f"= {tmc_footprint.mean():.6f}",
        )

        # ----------------------------------------------------------
        # 5. Masks
        # ----------------------------------------------------------

        science_mask = nac_valid & tmc_footprint

        # Conservative matcher support:
        # exclude TMC zeros, even though some may be real shadows.
        matcher_base = nac_valid & tmc_positive

        # Same 5-pixel boundary erosion philosophy as Pair004.
        erosion_kernel = np.ones((11, 11), np.uint8)

        matcher_mask = cv2.erode(
            matcher_base.astype(np.uint8),
            erosion_kernel,
            iterations=1,
        ) > 0

        print("\nCOMMON SUPPORT")
        print(
            "  science:",
            int(science_mask.sum()),
            "/",
            science_mask.size,
            f"= {science_mask.mean():.6f}",
        )

        print(
            "  matcher base:",
            int(matcher_base.sum()),
            "/",
            matcher_base.size,
            f"= {matcher_base.mean():.6f}",
        )

        print(
            "  matcher after 5px erosion:",
            int(matcher_mask.sum()),
            "/",
            matcher_mask.size,
            f"= {matcher_mask.mean():.6f}",
        )

        if matcher_mask.sum() == 0:
            raise RuntimeError(
                "No common matcher support found."
            )

        # ----------------------------------------------------------
        # 6. Valid-support bounding box statistics
        # ----------------------------------------------------------

        ys, xs = np.where(matcher_mask)

        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())

        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1

        print("\nCOMMON MATCHER BOUNDING BOX")
        print(
            f"  x: {min_x} -> {max_x} "
            f"({bbox_width} px)"
        )
        print(
            f"  y: {min_y} -> {max_y} "
            f"({bbox_height} px)"
        )

        # ----------------------------------------------------------
        # 7. Save canonical rasters
        # ----------------------------------------------------------

        source_path = OUT / "source_lro_nac.tif"
        reference_path = OUT / "reference_tmc2.tif"

        science_path = OUT / "science_mask.tif"
        matcher_path = OUT / "matcher_mask.tif"
        footprint_path = OUT / "tmc_footprint_mask.tif"

        write_tif(
            source_path,
            nac_on_tmc,
            target_transform,
            tmc.crs,
            nodata=np.nan,
            dtype="float32",
        )

        write_tif(
            reference_path,
            tmc_arr,
            target_transform,
            tmc.crs,
            nodata=None,
            dtype=tmc_arr.dtype,
        )

        write_tif(
            science_path,
            science_mask.astype(np.uint8),
            target_transform,
            tmc.crs,
            nodata=0,
            dtype="uint8",
        )

        write_tif(
            matcher_path,
            matcher_mask.astype(np.uint8),
            target_transform,
            tmc.crs,
            nodata=0,
            dtype="uint8",
        )

        write_tif(
            footprint_path,
            tmc_footprint.astype(np.uint8),
            target_transform,
            tmc.crs,
            nodata=0,
            dtype="uint8",
        )

        # ----------------------------------------------------------
        # 8. Preview images
        # ----------------------------------------------------------

        nac_preview = robust_preview(
            nac_on_tmc,
            science_mask,
        )

        # Use positive values for contrast calculation but display
        # whole geometric science footprint.
        tmc_contrast_valid = science_mask & tmc_positive

        tmc_preview = robust_preview(
            tmc_arr.astype(np.float32),
            tmc_contrast_valid,
        )

        cv2.imwrite(
            str(OUT / "source_lro_nac_preview.png"),
            nac_preview,
        )

        cv2.imwrite(
            str(OUT / "reference_tmc2_preview.png"),
            tmc_preview,
        )

        cv2.imwrite(
            str(OUT / "science_mask_preview.png"),
            science_mask.astype(np.uint8) * 255,
        )

        cv2.imwrite(
            str(OUT / "matcher_mask_preview.png"),
            matcher_mask.astype(np.uint8) * 255,
        )

        cv2.imwrite(
            str(OUT / "tmc_footprint_preview.png"),
            tmc_footprint.astype(np.uint8) * 255,
        )

        # Side-by-side
        side = np.concatenate(
            [tmc_preview, nac_preview],
            axis=1,
        )

        cv2.imwrite(
            str(OUT / "side_by_side_tmc_nac.png"),
            side,
        )

        # Red/cyan diagnostic.
        rgb = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        # OpenCV is BGR:
        # TMC -> red
        # NAC -> cyan
        rgb[:, :, 2] = tmc_preview
        rgb[:, :, 1] = nac_preview
        rgb[:, :, 0] = nac_preview

        rgb[~science_mask] = 0

        cv2.imwrite(
            str(OUT / "unregistered_red_cyan.png"),
            rgb,
        )

        # ----------------------------------------------------------
        # 9. Save metadata
        # ----------------------------------------------------------

        metadata = {
            "pair_id": "pair_006",
            "source_sensor": "LRO NAC",
            "source_product": "M141704260RE",
            "reference_sensor": "Chandrayaan-2 TMC-2",
            "reference_product":
                "ch2_tmc_ndn_20231025T1956513800_d_oth_d18",

            "canonical_resolution_m": 5.0,

            "width": width,
            "height": height,

            "tmc_positive_pixels":
                int(tmc_positive.sum()),

            "tmc_positive_ratio":
                float(tmc_positive.mean()),

            "tmc_footprint_pixels":
                int(tmc_footprint.sum()),

            "tmc_footprint_ratio":
                float(tmc_footprint.mean()),

            "nac_valid_pixels":
                int(nac_valid.sum()),

            "nac_valid_ratio":
                float(nac_valid.mean()),

            "science_valid_pixels":
                int(science_mask.sum()),

            "science_valid_ratio":
                float(science_mask.mean()),

            "matcher_base_pixels":
                int(matcher_base.sum()),

            "matcher_base_ratio":
                float(matcher_base.mean()),

            "matcher_pixels":
                int(matcher_mask.sum()),

            "matcher_ratio":
                float(matcher_mask.mean()),

            "matcher_bbox": {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "width": bbox_width,
                "height": bbox_height,
            },

            "transform": list(target_transform)[:6],

            "source_original_bounds": {
                "left": nac.bounds.left,
                "bottom": nac.bounds.bottom,
                "right": nac.bounds.right,
                "top": nac.bounds.top,
            },

            "target_grid_bounds": {
                "left": target_transform.c,
                "top": target_transform.f,
                "right":
                    target_transform.c
                    + width * target_transform.a,
                "bottom":
                    target_transform.f
                    + height * target_transform.e,
            },

            "notes": [
                "NAC was reprojected onto the native TMC 5 m grid.",
                "TMC > 0 is matcher support, not science validity.",
                "TMC geometric footprint is recovered from the main positive-support hull so internal zero-valued shadows can remain scientifically valid.",
                "Matcher mask uses common NAC/TMC positive support followed by 5 px erosion.",
                "No image-registration transform has been applied at this stage.",
            ],
        }

        with open(
            OUT / "pair006_canonical_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)

        print("\nOUTPUTS")
        print(" ", source_path)
        print(" ", reference_path)
        print(" ", science_path)
        print(" ", matcher_path)
        print(" ", footprint_path)

        print("\nPreview files written to:")
        print(" ", OUT)

        print("\nPAIR 006 canonical build complete.")


if __name__ == "__main__":
    main()