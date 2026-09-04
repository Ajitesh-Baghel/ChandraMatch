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

PAIR_DIR = ROOT / "data" / "processed" / "pair_004" / "canonical"
PREVIEW_DIR = PAIR_DIR / "previews"

PAIR_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


TMC_OUT = PAIR_DIR / "tmc2_reference.tif"
NAC_OUT = PAIR_DIR / "lro_nac_source.tif"
COMMON_MASK_OUT = PAIR_DIR / "common_valid_mask.tif"
NAC_MASK_OUT = PAIR_DIR / "lro_nac_valid_mask.tif"
METADATA_OUT = PAIR_DIR / "metadata.json"

NAC_OUTPUT_NODATA = -9999.0


def raster_valid_mask(data, raster_mask, nodata):
    valid = raster_mask > 0
    valid &= np.isfinite(data)

    if nodata is not None:
        try:
            if np.isfinite(nodata):
                valid &= data != nodata
        except TypeError:
            pass

    return valid


def normalize_preview(data, mask, p_low=2.0, p_high=98.0):
    """
    Robustly normalize a single-band image to uint8.

    Only valid pixels participate in percentile estimation.
    Invalid pixels are black.
    """
    out = np.zeros(data.shape, dtype=np.uint8)

    values = data[mask]

    if values.size == 0:
        return out

    values = values[np.isfinite(values)]

    if values.size == 0:
        return out

    low = np.percentile(values, p_low)
    high = np.percentile(values, p_high)

    if high <= low:
        high = low + 1.0

    scaled = (data.astype(np.float32) - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)

    out[mask] = np.round(
        scaled[mask] * 255.0
    ).astype(np.uint8)

    return out


def save_preview(arr, path, max_width=2200):
    image = Image.fromarray(arr)

    if image.width > max_width:
        ratio = max_width / image.width
        new_height = max(
            1,
            int(round(image.height * ratio))
        )

        image = image.resize(
            (max_width, new_height),
            Image.Resampling.LANCZOS,
        )

    image.save(path)


def save_single_band_tif(
    path,
    data,
    transform,
    crs,
    dtype,
    nodata=None,
):
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


def main():
    print("=" * 72)
    print("PAIR 004 — BUILD CANONICAL TMC-2 / LRO NAC PAIR")
    print("=" * 72)

    if not TMC_PATH.exists():
        raise FileNotFoundError(TMC_PATH)

    if not NAC_PATH.exists():
        raise FileNotFoundError(NAC_PATH)

    print("\nReference:")
    print(TMC_PATH)

    print("\nSource:")
    print(NAC_PATH)

    with rasterio.open(TMC_PATH) as tmc, rasterio.open(NAC_PATH) as nac:

        # --------------------------------------------------------------
        # 1. Projected bounding-box intersection
        # --------------------------------------------------------------

        left = max(tmc.bounds.left, nac.bounds.left)
        right = min(tmc.bounds.right, nac.bounds.right)
        bottom = max(tmc.bounds.bottom, nac.bounds.bottom)
        top = min(tmc.bounds.top, nac.bounds.top)

        if left >= right or bottom >= top:
            raise RuntimeError(
                "TMC and NAC have no projected overlap."
            )

        # --------------------------------------------------------------
        # 2. Make initial overlap window on EXACT TMC pixel grid
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
            math.ceil(wf.col_off + wf.width),
        )

        row1 = min(
            tmc.height,
            math.ceil(wf.row_off + wf.height),
        )

        initial_width = col1 - col0
        initial_height = row1 - row0

        initial_window = Window(
            col_off=col0,
            row_off=row0,
            width=initial_width,
            height=initial_height,
        )

        initial_transform = window_transform(
            initial_window,
            tmc.transform,
        )

        print("\n" + "=" * 72)
        print("INITIAL TMC-ALIGNED GRID")
        print("=" * 72)

        print("Window:", initial_window)
        print(
            "Shape:",
            initial_height,
            "x",
            initial_width,
        )

        print("Transform:")
        print(initial_transform)

        # --------------------------------------------------------------
        # 3. Read TMC reference crop
        # --------------------------------------------------------------

        tmc_data = tmc.read(
            1,
            window=initial_window,
        )

        tmc_raster_mask = tmc.read_masks(
            1,
            window=initial_window,
        )

        tmc_valid = raster_valid_mask(
            tmc_data,
            tmc_raster_mask,
            tmc.nodata,
        )

        # --------------------------------------------------------------
        # 4. Read NAC and place it onto the exact TMC grid
        # --------------------------------------------------------------

        nac_data_native = nac.read(1)
        nac_mask_native = nac.read_masks(1)

        nac_valid_native = raster_valid_mask(
            nac_data_native,
            nac_mask_native,
            nac.nodata,
        )

        nac_on_tmc = np.full(
            (initial_height, initial_width),
            NAC_OUTPUT_NODATA,
            dtype=np.float32,
        )

        reproject(
            source=nac_data_native,
            destination=nac_on_tmc,
            src_transform=nac.transform,
            src_crs=nac.crs,
            src_nodata=nac.nodata,
            dst_transform=initial_transform,
            dst_crs=tmc.crs,
            dst_nodata=NAC_OUTPUT_NODATA,
            resampling=Resampling.bilinear,
        )

        nac_valid_on_tmc_u8 = np.zeros(
            (initial_height, initial_width),
            dtype=np.uint8,
        )

        reproject(
            source=nac_valid_native.astype(np.uint8),
            destination=nac_valid_on_tmc_u8,
            src_transform=nac.transform,
            src_crs=nac.crs,
            src_nodata=0,
            dst_transform=initial_transform,
            dst_crs=tmc.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        nac_valid = nac_valid_on_tmc_u8 > 0
        nac_valid &= np.isfinite(nac_on_tmc)
        nac_valid &= nac_on_tmc != NAC_OUTPUT_NODATA

        # --------------------------------------------------------------
        # 5. Check potential TMC fill pixels
        # --------------------------------------------------------------

        tmc_zero_total = int(
            np.count_nonzero(tmc_data == 0)
        )

        tmc_zero_under_nac = int(
            np.count_nonzero(
                (tmc_data == 0) & nac_valid
            )
        )

        print("\n" + "=" * 72)
        print("TMC ZERO-DN CHECK")
        print("=" * 72)

        print(
            f"TMC zero pixels in bounding window : "
            f"{tmc_zero_total:,}"
        )

        print(
            f"TMC zero pixels under valid NAC    : "
            f"{tmc_zero_under_nac:,}"
        )

        if tmc_zero_under_nac > 0:
            print(
                "\nWARNING: Zero-valued TMC pixels occur "
                "under the NAC footprint."
            )
            print(
                "They are NOT automatically being treated "
                "as nodata."
            )

        # --------------------------------------------------------------
        # 6. Common validity
        # --------------------------------------------------------------

        common_valid = tmc_valid & nac_valid

        common_count = int(common_valid.sum())

        if common_count == 0:
            raise RuntimeError(
                "No common valid pixels found."
            )

        # --------------------------------------------------------------
        # 7. Find largest connected common-valid component
        # --------------------------------------------------------------

        structure = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        labels, num_components = ndimage.label(
            common_valid,
            structure=structure,
        )

        component_sizes = np.bincount(labels.ravel())

        if component_sizes.size <= 1:
            raise RuntimeError(
                "No connected common-valid component."
            )

        component_sizes[0] = 0

        largest_label = int(
            component_sizes.argmax()
        )

        largest_size = int(
            component_sizes[largest_label]
        )

        largest_component = (
            labels == largest_label
        )

        rows, cols = np.where(largest_component)

        local_r0 = int(rows.min())
        local_r1 = int(rows.max())
        local_c0 = int(cols.min())
        local_c1 = int(cols.max())

        print("\n" + "=" * 72)
        print("COMMON COMPONENT")
        print("=" * 72)

        print(
            f"Connected components : {num_components}"
        )

        print(
            f"Largest component     : "
            f"{largest_size:,} pixels"
        )

        print(
            "Local rows            :",
            local_r0,
            "->",
            local_r1,
        )

        print(
            "Local cols            :",
            local_c0,
            "->",
            local_c1,
        )

        # --------------------------------------------------------------
        # 8. Crop to bounding box of entire largest component
        #
        # We intentionally preserve the ENTIRE continuous NAC strip.
        # We are not selecting an arbitrary "easy" matching patch.
        # --------------------------------------------------------------

        canonical_height = (
            local_r1 - local_r0 + 1
        )

        canonical_width = (
            local_c1 - local_c0 + 1
        )

        canonical_window = Window(
            col_off=col0 + local_c0,
            row_off=row0 + local_r0,
            width=canonical_width,
            height=canonical_height,
        )

        canonical_transform = window_transform(
            canonical_window,
            tmc.transform,
        )

        row_slice = slice(
            local_r0,
            local_r1 + 1,
        )

        col_slice = slice(
            local_c0,
            local_c1 + 1,
        )

        tmc_final = tmc_data[
            row_slice,
            col_slice,
        ]

        nac_final = nac_on_tmc[
            row_slice,
            col_slice,
        ]

        tmc_valid_final = tmc_valid[
            row_slice,
            col_slice,
        ]

        nac_valid_final = nac_valid[
            row_slice,
            col_slice,
        ]

        common_final = common_valid[
            row_slice,
            col_slice,
        ]

        # --------------------------------------------------------------
        # Make sure invalid NAC pixels have one consistent fill value.
        # --------------------------------------------------------------

        nac_final = nac_final.astype(np.float32)
        nac_final[~nac_valid_final] = NAC_OUTPUT_NODATA

        final_pixels = int(
            canonical_height * canonical_width
        )

        final_common_count = int(
            common_final.sum()
        )

        final_common_ratio = (
            final_common_count / final_pixels
        )

        nac_final_count = int(
            nac_valid_final.sum()
        )

        tmc_final_count = int(
            tmc_valid_final.sum()
        )

        print("\n" + "=" * 72)
        print("CANONICAL PAIR")
        print("=" * 72)

        print(
            "Shape:",
            canonical_height,
            "x",
            canonical_width,
        )

        print(
            f"Total pixels       : "
            f"{final_pixels:,}"
        )

        print(
            f"TMC valid          : "
            f"{tmc_final_count:,}"
        )

        print(
            f"NAC valid          : "
            f"{nac_final_count:,}"
        )

        print(
            f"Common valid       : "
            f"{final_common_count:,}"
        )

        print(
            f"Common valid ratio : "
            f"{final_common_ratio:.6f} "
            f"({final_common_ratio * 100:.3f}%)"
        )

        # --------------------------------------------------------------
        # 9. Bounds
        # --------------------------------------------------------------

        canonical_left = canonical_transform.c
        canonical_top = canonical_transform.f

        canonical_right = (
            canonical_left
            + canonical_transform.a
            * canonical_width
        )

        canonical_bottom = (
            canonical_top
            + canonical_transform.e
            * canonical_height
        )

        print("\nProjected bounds:")
        print(" left   =", canonical_left)
        print(" right  =", canonical_right)
        print(" bottom =", canonical_bottom)
        print(" top    =", canonical_top)

        # --------------------------------------------------------------
        # 10. Write canonical science rasters
        # --------------------------------------------------------------

        print("\nWriting canonical rasters...")

        save_single_band_tif(
            TMC_OUT,
            tmc_final,
            canonical_transform,
            tmc.crs,
            "uint16",
        )

        save_single_band_tif(
            NAC_OUT,
            nac_final,
            canonical_transform,
            tmc.crs,
            "float32",
            nodata=NAC_OUTPUT_NODATA,
        )

        save_single_band_tif(
            COMMON_MASK_OUT,
            common_final.astype(np.uint8),
            canonical_transform,
            tmc.crs,
            "uint8",
            nodata=0,
        )

        save_single_band_tif(
            NAC_MASK_OUT,
            nac_valid_final.astype(np.uint8),
            canonical_transform,
            tmc.crs,
            "uint8",
            nodata=0,
        )

        # --------------------------------------------------------------
        # 11. Preview normalization
        #
        # Use COMMON validity for both so percentile scaling is
        # calculated over exactly the same geographic support.
        # --------------------------------------------------------------

        print("Building previews...")

        tmc_preview = normalize_preview(
            tmc_final,
            common_final,
        )

        nac_preview = normalize_preview(
            nac_final,
            common_final,
        )

        save_preview(
            tmc_preview,
            PREVIEW_DIR / "tmc2_reference.png",
        )

        save_preview(
            nac_preview,
            PREVIEW_DIR / "lro_nac_source.png",
        )

        save_preview(
            common_final.astype(np.uint8) * 255,
            PREVIEW_DIR / "common_valid_mask.png",
        )

        # Side-by-side
        side_by_side = np.concatenate(
            [tmc_preview, nac_preview],
            axis=1,
        )

        save_preview(
            side_by_side,
            PREVIEW_DIR / "pair004_side_by_side.png",
            max_width=3600,
        )

        # 50/50 overlay
        overlay = np.zeros(
            tmc_preview.shape,
            dtype=np.uint8,
        )

        overlay_float = (
            0.5 * tmc_preview.astype(np.float32)
            + 0.5 * nac_preview.astype(np.float32)
        )

        overlay[common_final] = np.round(
            overlay_float[common_final]
        ).astype(np.uint8)

        save_preview(
            overlay,
            PREVIEW_DIR / "pair004_overlay_50_50.png",
        )

        # Checkerboard
        checker = np.zeros_like(
            tmc_preview,
            dtype=np.uint8,
        )

        block = 128

        yy, xx = np.indices(
            tmc_preview.shape
        )

        choose_tmc = (
            ((xx // block) + (yy // block)) % 2
            == 0
        )

        checker[
            choose_tmc & common_final
        ] = tmc_preview[
            choose_tmc & common_final
        ]

        checker[
            (~choose_tmc) & common_final
        ] = nac_preview[
            (~choose_tmc) & common_final
        ]

        save_preview(
            checker,
            PREVIEW_DIR / "pair004_checkerboard.png",
        )

        # --------------------------------------------------------------
        # 12. Metadata
        # --------------------------------------------------------------

        metadata = {
            "pair_id": "pair_004",

            "description": (
                "Chandrayaan-2 TMC-2 reference versus "
                "LRO LROC NAC source"
            ),

            "reference_sensor": "Chandrayaan-2 TMC-2",
            "source_sensor": "LRO LROC NAC Left",

            "source_product_id": "M185196277LE",

            "tmc_input": str(TMC_PATH),
            "nac_input": str(NAC_PATH),

            "reference_output": str(TMC_OUT),
            "source_output": str(NAC_OUT),
            "common_mask_output": str(
                COMMON_MASK_OUT
            ),

            "canonical_shape": [
                canonical_height,
                canonical_width,
            ],

            "pixel_resolution_m": 5.0,

            "projection": {
                "name": "PolarStereographic",
                "target": "Moon",
                "center_latitude_deg": -90.0,
                "center_longitude_deg": 0.0,
                "moon_radius_m": 1737400.0,
            },

            "transform": [
                canonical_transform.a,
                canonical_transform.b,
                canonical_transform.c,
                canonical_transform.d,
                canonical_transform.e,
                canonical_transform.f,
            ],

            "bounds": {
                "left": float(canonical_left),
                "bottom": float(canonical_bottom),
                "right": float(canonical_right),
                "top": float(canonical_top),
            },

            "total_pixels": final_pixels,

            "tmc_valid_pixels": tmc_final_count,
            "nac_valid_pixels": nac_final_count,

            "common_valid_pixels": (
                final_common_count
            ),

            "common_valid_ratio": float(
                final_common_ratio
            ),

            "connected_components_before_crop": int(
                num_components
            ),

            "largest_component_pixels": int(
                largest_size
            ),

            "tmc_zero_pixels_initial_window": (
                tmc_zero_total
            ),

            "tmc_zero_pixels_under_valid_nac": (
                tmc_zero_under_nac
            ),

            "nac_output_nodata": (
                NAC_OUTPUT_NODATA
            ),

            "processing": {
                "nac_native_processing": [
                    "lronac2isis",
                    "spiceinit",
                    "lronaccal",
                    "lronacecho",
                    "cam2map",
                ],

                "nac_map_resolution_m": 5.0,

                "canonical_grid": (
                    "Exact TMC-2 pixel grid"
                ),

                "nac_grid_alignment_resampling": (
                    "bilinear"
                ),

                "mask_grid_alignment_resampling": (
                    "nearest"
                ),
            },
        }

        with open(
            METADATA_OUT,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

    print("\n" + "=" * 72)
    print("OUTPUTS")
    print("=" * 72)

    print(TMC_OUT)
    print(NAC_OUT)
    print(COMMON_MASK_OUT)
    print(NAC_MASK_OUT)
    print(METADATA_OUT)

    print("\nPreviews:")
    print(
        PREVIEW_DIR / "tmc2_reference.png"
    )
    print(
        PREVIEW_DIR / "lro_nac_source.png"
    )
    print(
        PREVIEW_DIR / "common_valid_mask.png"
    )
    print(
        PREVIEW_DIR / "pair004_side_by_side.png"
    )
    print(
        PREVIEW_DIR / "pair004_overlay_50_50.png"
    )
    print(
        PREVIEW_DIR / "pair004_checkerboard.png"
    )

    print("\nPAIR 004 CANONICAL BUILD COMPLETE.")


if __name__ == "__main__":
    main()