from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds


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
    / "results"
    / "pair_005"
    / "overlap"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def print_bounds(name, b):
    print(
        f"{name}: "
        f"left={b.left:.6f}, "
        f"bottom={b.bottom:.6f}, "
        f"right={b.right:.6f}, "
        f"top={b.top:.6f}"
    )


def main():

    print("=" * 78)
    print("PAIR 005 — KAGUYA TC ↔ TMC-2 EXACT OVERLAP")
    print("=" * 78)

    print("\nKaguya:")
    print(KAGUYA)

    print("\nTMC:")
    print(TMC)

    with rasterio.open(KAGUYA) as kag, rasterio.open(TMC) as tmc:

        print("\n" + "=" * 78)
        print("RASTER GEOMETRY")
        print("=" * 78)

        print(
            "\nKaguya shape:",
            kag.height,
            "x",
            kag.width,
        )

        print(
            "Kaguya resolution:",
            kag.res,
        )

        print_bounds(
            "Kaguya bounds",
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

        print_bounds(
            "TMC bounds",
            tmc.bounds,
        )

        print("\nCRS exact equality:")
        print(
            kag.crs == tmc.crs
        )

        print("\nKaguya CRS:")
        print(kag.crs)

        print("\nTMC CRS:")
        print(tmc.crs)

        # ------------------------------------------------------------
        # KAGUYA SCIENCE VALIDITY
        #
        # GDAL correctly recognises TC ortho DN=0 as nodata.
        # We deliberately use the GDAL mask instead of assuming anything
        # about DQA at this stage.
        # ------------------------------------------------------------

        kag_data = kag.read(1)

        kag_valid = (
            kag.read_masks(1) > 0
        )

        kag_valid &= np.isfinite(
            kag_data
        )

        print("\n" + "=" * 78)
        print("KAGUYA VALIDITY")
        print("=" * 78)

        print(
            "Total Kaguya pixels:",
            f"{kag_data.size:,}",
        )

        print(
            "Valid Kaguya pixels:",
            f"{int(kag_valid.sum()):,}",
        )

        print(
            "Valid ratio:",
            f"{kag_valid.mean():.8f}",
        )

        # ------------------------------------------------------------
        # PROJECT TMC ONTO THE EXACT KAGUYA 10 m GRID
        #
        # Kaguya is the coarser sensor:
        #
        #       TMC    = 5 m
        #       Kaguya = 10 m
        #
        # Therefore we downsample TMC to Kaguya.
        #
        # We DO NOT upsample Kaguya to 5 m.
        # ------------------------------------------------------------

        print("\n" + "=" * 78)
        print("REPROJECTING TMC → KAGUYA 10 m GRID")
        print("=" * 78)

        tmc_on_kaguya = np.full(
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
            destination=tmc_on_kaguya,
            src_transform=tmc.transform,
            src_crs=tmc.crs,
            src_nodata=tmc.nodata,
            dst_transform=kag.transform,
            dst_crs=kag.crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )

        tmc_spatial_valid = np.isfinite(
            tmc_on_kaguya
        )

        print(
            "TMC-covered Kaguya-grid pixels:",
            f"{int(tmc_spatial_valid.sum()):,}",
        )

        print(
            "TMC grid coverage ratio:",
            f"{tmc_spatial_valid.mean():.8f}",
        )

        # ------------------------------------------------------------
        # SCIENCE OVERLAP
        #
        # IMPORTANT:
        #
        # TMC DN=0 is NOT automatically nodata.
        # In Pair 004 we verified that many zero values correspond to
        # legitimate deep lunar shadows.
        #
        # So science validity here is:
        #
        #   valid Kaguya radiance
        #   AND
        #   spatially covered by TMC
        #
        # ------------------------------------------------------------

        common_valid = (
            kag_valid
            & tmc_spatial_valid
        )

        common_count = int(
            common_valid.sum()
        )

        print("\n" + "=" * 78)
        print("COMMON SCIENCE OVERLAP")
        print("=" * 78)

        print(
            "Common valid pixels:",
            f"{common_count:,}",
        )

        print(
            "Common / full Kaguya raster:",
            f"{common_count / kag_data.size:.8f}",
        )

        print(
            "Common / Kaguya-valid:",
            f"{common_count / max(int(kag_valid.sum()), 1):.8f}",
        )

        # ------------------------------------------------------------
        # ZERO / SHADOW DIAGNOSTIC
        # ------------------------------------------------------------

        if common_count > 0:

            tmc_common = (
                tmc_on_kaguya[
                    common_valid
                ]
            )

            kag_common = (
                kag_data[
                    common_valid
                ]
            )

            tmc_zero = int(
                np.sum(
                    tmc_common == 0
                )
            )

            print("\n" + "=" * 78)
            print("INTENSITY DIAGNOSTICS")
            print("=" * 78)

            print(
                "TMC exact-zero pixels inside common overlap:",
                f"{tmc_zero:,}",
            )

            print(
                "TMC zero fraction of common:",
                f"{tmc_zero / common_count:.8f}",
            )

            print(
                "\nTMC common min/max:",
                float(
                    np.min(tmc_common)
                ),
                "/",
                float(
                    np.max(tmc_common)
                ),
            )

            print(
                "TMC common median:",
                float(
                    np.median(tmc_common)
                ),
            )

            print(
                "\nKaguya common raw-DN min/max:",
                int(
                    np.min(kag_common)
                ),
                "/",
                int(
                    np.max(kag_common)
                ),
            )

            print(
                "Kaguya common median:",
                float(
                    np.median(kag_common)
                ),
            )

        # ------------------------------------------------------------
        # CONNECTED COMPONENT ANALYSIS
        # ------------------------------------------------------------

        print("\n" + "=" * 78)
        print("CONNECTED OVERLAP COMPONENTS")
        print("=" * 78)

        binary = (
            common_valid.astype(
                np.uint8
            )
        )

        nlabels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        components = []

        for label_id in range(
            1,
            nlabels,
        ):

            x = int(
                stats[
                    label_id,
                    cv2.CC_STAT_LEFT,
                ]
            )

            y = int(
                stats[
                    label_id,
                    cv2.CC_STAT_TOP,
                ]
            )

            w = int(
                stats[
                    label_id,
                    cv2.CC_STAT_WIDTH,
                ]
            )

            h = int(
                stats[
                    label_id,
                    cv2.CC_STAT_HEIGHT,
                ]
            )

            area = int(
                stats[
                    label_id,
                    cv2.CC_STAT_AREA,
                ]
            )

            components.append(
                {
                    "label": label_id,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": area,
                }
            )

        components.sort(
            key=lambda c: c["area"],
            reverse=True,
        )

        print(
            "Non-background components:",
            len(components),
        )

        largest = None

        if components:

            largest = components[0]

            print(
                "Largest component pixels:",
                f"{largest['area']:,}",
            )

            print(
                "Largest / common overlap:",
                f"{largest['area'] / common_count:.8f}",
            )

            print(
                "Pixel bbox:",
                f"x={largest['x']}.."
                f"{largest['x'] + largest['width'] - 1}, "
                f"y={largest['y']}.."
                f"{largest['y'] + largest['height'] - 1}",
            )

            win = Window(
                largest["x"],
                largest["y"],
                largest["width"],
                largest["height"],
            )

            left, bottom, right, top = (
                window_bounds(
                    win,
                    kag.transform,
                )
            )

            print(
                "Projected bbox:",
                f"left={left:.6f}, "
                f"bottom={bottom:.6f}, "
                f"right={right:.6f}, "
                f"top={top:.6f}",
            )

            largest[
                "projected_bounds"
            ] = {
                "left": left,
                "bottom": bottom,
                "right": right,
                "top": top,
            }

        # ------------------------------------------------------------
        # SAVE SUMMARY
        # ------------------------------------------------------------

        summary = {
            "pair": "pair_005",
            "source": "SELENE/Kaguya TC",
            "reference": "Chandrayaan-2 TMC-2",
            "kaguya_product": (
                "DTMTCO_03_01597S711E1002PS_IMG"
            ),
            "kaguya_observation_time": (
                "2008-02-17T03:51:38.114635"
            ),
            "tmc_observation_date": (
                "2023-10-25"
            ),
            "common_grid_resolution_m": 10.0,
            "kaguya_shape": [
                kag.height,
                kag.width,
            ],
            "kaguya_bounds": {
                "left": kag.bounds.left,
                "bottom": kag.bounds.bottom,
                "right": kag.bounds.right,
                "top": kag.bounds.top,
            },
            "kaguya_valid_pixels": int(
                kag_valid.sum()
            ),
            "kaguya_valid_ratio": float(
                kag_valid.mean()
            ),
            "tmc_spatial_coverage_pixels": int(
                tmc_spatial_valid.sum()
            ),
            "common_valid_pixels": common_count,
            "common_fraction_full_kaguya": float(
                common_count
                / kag_data.size
            ),
            "common_fraction_kaguya_valid": float(
                common_count
                / max(
                    int(
                        kag_valid.sum()
                    ),
                    1,
                )
            ),
            "largest_component": largest,
            "number_of_components": len(
                components
            ),
            "mask_policy": {
                "kaguya": (
                    "GDAL/PDS nodata mask; "
                    "TC DN=0 excluded"
                ),
                "tmc": (
                    "spatial coverage only; "
                    "TMC DN=0 retained as "
                    "potential lunar shadow"
                ),
                "dqa": (
                    "not used until flag "
                    "semantics are decoded"
                ),
            },
        }

        json_path = (
            OUT_DIR
            / "pair005_overlap_summary.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
            )

        print("\n" + "=" * 78)

        print(
            "Saved:"
        )

        print(
            json_path
        )

        print("\n" + "=" * 78)
        print("PAIR 005 OVERLAP ANALYSIS COMPLETE")
        print("=" * 78)


if __name__ == "__main__":
    main()