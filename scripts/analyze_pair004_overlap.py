from pathlib import Path
import json
import math

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

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
    / "pair_004"
    / "lro_nac_isis"
    / "M185196277LE_map_5m.cub"
)

OUT_DIR = ROOT / "results" / "pair_004" / "overlap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def valid_mask(data, raster_mask, nodata):
    """
    Build a conservative validity mask using:
      - rasterio's internal mask
      - finite numeric values
      - explicit nodata if present
    """
    valid = raster_mask > 0
    valid &= np.isfinite(data)

    if nodata is not None and np.isfinite(nodata):
        valid &= data != nodata

    return valid


def save_mask_png(mask, path):
    arr = (mask.astype(np.uint8) * 255)
    Image.fromarray(arr).save(path)


def save_mask_tif(mask, path, transform, crs):
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "nodata": 0,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def main():
    print("=" * 70)
    print("PAIR 004 — EXACT TMC-2 / LRO NAC OVERLAP")
    print("=" * 70)

    print("\nTMC:")
    print(TMC_PATH)

    print("\nNAC:")
    print(NAC_PATH)

    if not TMC_PATH.exists():
        raise FileNotFoundError(TMC_PATH)

    if not NAC_PATH.exists():
        raise FileNotFoundError(NAC_PATH)

    with rasterio.open(TMC_PATH) as tmc, rasterio.open(NAC_PATH) as nac:

        print("\n" + "=" * 70)
        print("DATASET INFORMATION")
        print("=" * 70)

        print("\nTMC")
        print("Driver:", tmc.driver)
        print("Shape:", tmc.height, "x", tmc.width)
        print("CRS:", tmc.crs)
        print("Transform:", tmc.transform)
        print("Bounds:", tmc.bounds)
        print("Nodata:", tmc.nodata)
        print("Dtype:", tmc.dtypes[0])

        print("\nNAC")
        print("Driver:", nac.driver)
        print("Shape:", nac.height, "x", nac.width)
        print("CRS:", nac.crs)
        print("Transform:", nac.transform)
        print("Bounds:", nac.bounds)
        print("Nodata:", nac.nodata)
        print("Dtype:", nac.dtypes[0])

        print("\nCRS exactly equal:", tmc.crs == nac.crs)

        # Both datasets have already been put in the same lunar
        # south-polar stereographic coordinate system.
        #
        # Calculate projected bounding-box intersection.
        left = max(tmc.bounds.left, nac.bounds.left)
        right = min(tmc.bounds.right, nac.bounds.right)
        bottom = max(tmc.bounds.bottom, nac.bounds.bottom)
        top = min(tmc.bounds.top, nac.bounds.top)

        if left >= right or bottom >= top:
            raise RuntimeError("No projected bounding-box overlap.")

        intersection_area = (right - left) * (top - bottom)
        nac_bbox_area = (
            (nac.bounds.right - nac.bounds.left)
            * (nac.bounds.top - nac.bounds.bottom)
        )

        bbox_overlap_fraction = intersection_area / nac_bbox_area

        print("\n" + "=" * 70)
        print("PROJECTED BOUNDING-BOX OVERLAP")
        print("=" * 70)

        print("Intersection:")
        print(" left  =", left)
        print(" right =", right)
        print(" bottom=", bottom)
        print(" top   =", top)

        print(
            f"\nNAC bounding-box overlap fraction: "
            f"{bbox_overlap_fraction:.6f}"
        )
        print(
            f"NAC bounding-box overlap percent: "
            f"{bbox_overlap_fraction * 100:.3f}%"
        )

        # --------------------------------------------------------------
        # Build an overlap window aligned EXACTLY to the TMC grid.
        # --------------------------------------------------------------

        wf = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=tmc.transform,
        )

        col0 = max(0, math.floor(wf.col_off))
        row0 = max(0, math.floor(wf.row_off))

        col1 = min(
            tmc.width,
            math.ceil(wf.col_off + wf.width)
        )

        row1 = min(
            tmc.height,
            math.ceil(wf.row_off + wf.height)
        )

        width = col1 - col0
        height = row1 - row0

        window = Window(
            col_off=col0,
            row_off=row0,
            width=width,
            height=height,
        )

        dst_transform = window_transform(window, tmc.transform)

        print("\n" + "=" * 70)
        print("TMC-ALIGNED OVERLAP GRID")
        print("=" * 70)

        print("Window:", window)
        print("Shape:", height, "x", width)
        print("Pixels:", height * width)
        print("Transform:")
        print(dst_transform)

        # --------------------------------------------------------------
        # Read TMC crop
        # --------------------------------------------------------------

        tmc_data = tmc.read(1, window=window)
        tmc_raster_mask = tmc.read_masks(1, window=window)

        tmc_valid = valid_mask(
            tmc_data,
            tmc_raster_mask,
            tmc.nodata,
        )

        # --------------------------------------------------------------
        # Reproject NAC to the exact TMC pixel grid.
        #
        # Reproject IMAGE and VALIDITY MASK separately.
        # This avoids interpolating invalid pixels into valid regions.
        # --------------------------------------------------------------

        nac_data = nac.read(1)
        nac_raster_mask = nac.read_masks(1)

        nac_valid_native = valid_mask(
            nac_data,
            nac_raster_mask,
            nac.nodata,
        )

        nac_on_tmc = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=nac_data,
            destination=nac_on_tmc,
            src_transform=nac.transform,
            src_crs=nac.crs,
            src_nodata=nac.nodata,
            dst_transform=dst_transform,
            dst_crs=tmc.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        nac_valid_on_tmc = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        reproject(
            source=nac_valid_native.astype(np.uint8),
            destination=nac_valid_on_tmc,
            src_transform=nac.transform,
            src_crs=nac.crs,
            src_nodata=0,
            dst_transform=dst_transform,
            dst_crs=tmc.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        nac_valid = nac_valid_on_tmc > 0
        nac_valid &= np.isfinite(nac_on_tmc)

        # --------------------------------------------------------------
        # Common validity
        # --------------------------------------------------------------

        common_valid = tmc_valid & nac_valid

        total_pixels = common_valid.size

        tmc_valid_count = int(tmc_valid.sum())
        nac_valid_count = int(nac_valid.sum())
        common_valid_count = int(common_valid.sum())

        tmc_valid_ratio = tmc_valid_count / total_pixels
        nac_valid_ratio = nac_valid_count / total_pixels
        common_valid_ratio = common_valid_count / total_pixels

        if nac_valid_count > 0:
            common_of_nac = common_valid_count / nac_valid_count
        else:
            common_of_nac = 0.0

        print("\n" + "=" * 70)
        print("VALID PIXEL OVERLAP")
        print("=" * 70)

        print(f"Grid pixels       : {total_pixels:,}")
        print(
            f"TMC valid pixels  : {tmc_valid_count:,} "
            f"({tmc_valid_ratio:.6f})"
        )
        print(
            f"NAC valid pixels  : {nac_valid_count:,} "
            f"({nac_valid_ratio:.6f})"
        )
        print(
            f"COMMON valid      : {common_valid_count:,} "
            f"({common_valid_ratio:.6f})"
        )
        print(
            f"Common / NAC valid: {common_of_nac:.6f} "
            f"({common_of_nac * 100:.3f}%)"
        )

        # --------------------------------------------------------------
        # Largest connected common-valid component
        # --------------------------------------------------------------

        structure = np.ones((3, 3), dtype=np.uint8)

        labels, num_components = ndimage.label(
            common_valid,
            structure=structure,
        )

        print("\nConnected components:", num_components)

        if num_components > 0:
            component_sizes = np.bincount(labels.ravel())
            component_sizes[0] = 0

            largest_label = int(component_sizes.argmax())
            largest_size = int(component_sizes[largest_label])

            largest_component = labels == largest_label

            rows, cols = np.where(largest_component)

            r0 = int(rows.min())
            r1 = int(rows.max())
            c0 = int(cols.min())
            c1 = int(cols.max())

            largest_fraction_common = (
                largest_size / common_valid_count
                if common_valid_count
                else 0.0
            )

            print(
                f"Largest component : {largest_size:,} pixels"
            )
            print(
                f"Fraction of common: {largest_fraction_common:.6f}"
            )
            print(
                f"Component rows    : {r0} -> {r1}"
            )
            print(
                f"Component cols    : {c0} -> {c1}"
            )

            component_window = Window(
                col_off=col0 + c0,
                row_off=row0 + r0,
                width=(c1 - c0 + 1),
                height=(r1 - r0 + 1),
            )

            component_transform = window_transform(
                component_window,
                tmc.transform,
            )

            component_left = component_transform.c
            component_top = component_transform.f
            component_right = (
                component_left
                + component_transform.a * component_window.width
            )
            component_bottom = (
                component_top
                + component_transform.e * component_window.height
            )

            print("\nLargest component projected bounds:")
            print(" left  =", component_left)
            print(" right =", component_right)
            print(" bottom=", component_bottom)
            print(" top   =", component_top)

        else:
            largest_label = 0
            largest_size = 0
            largest_fraction_common = 0.0
            largest_component = np.zeros_like(
                common_valid,
                dtype=bool,
            )

            r0 = r1 = c0 = c1 = None
            component_left = None
            component_right = None
            component_bottom = None
            component_top = None

        # --------------------------------------------------------------
        # Save masks
        # --------------------------------------------------------------

        save_mask_png(
            tmc_valid,
            OUT_DIR / "tmc_valid_mask.png",
        )

        save_mask_png(
            nac_valid,
            OUT_DIR / "nac_valid_mask.png",
        )

        save_mask_png(
            common_valid,
            OUT_DIR / "common_valid_mask.png",
        )

        save_mask_png(
            largest_component,
            OUT_DIR / "largest_common_component.png",
        )

        save_mask_tif(
            common_valid,
            OUT_DIR / "common_valid_mask.tif",
            dst_transform,
            tmc.crs,
        )

        # --------------------------------------------------------------
        # JSON summary
        # --------------------------------------------------------------

        summary = {
            "tmc_path": str(TMC_PATH),
            "nac_path": str(NAC_PATH),

            "tmc_shape": [
                tmc.height,
                tmc.width,
            ],

            "nac_shape": [
                nac.height,
                nac.width,
            ],

            "tmc_resolution_m": [
                float(abs(tmc.transform.a)),
                float(abs(tmc.transform.e)),
            ],

            "nac_resolution_m": [
                float(abs(nac.transform.a)),
                float(abs(nac.transform.e)),
            ],

            "nac_bounds": {
                "left": float(nac.bounds.left),
                "bottom": float(nac.bounds.bottom),
                "right": float(nac.bounds.right),
                "top": float(nac.bounds.top),
            },

            "intersection_bounds": {
                "left": float(left),
                "bottom": float(bottom),
                "right": float(right),
                "top": float(top),
            },

            "nac_bbox_overlap_fraction": float(
                bbox_overlap_fraction
            ),

            "analysis_grid_shape": [
                int(height),
                int(width),
            ],

            "analysis_grid_pixels": int(total_pixels),

            "tmc_valid_pixels": tmc_valid_count,
            "nac_valid_pixels": nac_valid_count,
            "common_valid_pixels": common_valid_count,

            "tmc_valid_ratio": float(tmc_valid_ratio),
            "nac_valid_ratio": float(nac_valid_ratio),
            "common_valid_ratio": float(common_valid_ratio),

            "common_fraction_of_nac_valid": float(
                common_of_nac
            ),

            "connected_components": int(num_components),

            "largest_component_pixels": int(
                largest_size
            ),

            "largest_component_fraction_of_common": float(
                largest_fraction_common
            ),

            "largest_component_local_pixel_bounds": {
                "row_min": r0,
                "row_max": r1,
                "col_min": c0,
                "col_max": c1,
            },

            "largest_component_projected_bounds": {
                "left": component_left,
                "bottom": component_bottom,
                "right": component_right,
                "top": component_top,
            },
        }

        summary_path = (
            OUT_DIR / "pair004_overlap_summary.json"
        )

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("OUTPUTS")
        print("=" * 70)

        print(summary_path)
        print(OUT_DIR / "tmc_valid_mask.png")
        print(OUT_DIR / "nac_valid_mask.png")
        print(OUT_DIR / "common_valid_mask.png")
        print(OUT_DIR / "largest_common_component.png")
        print(OUT_DIR / "common_valid_mask.tif")

        print("\nDONE.")


if __name__ == "__main__":
    main()