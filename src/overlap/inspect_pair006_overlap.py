from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


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


def intersection_bounds(a, b):
    left = max(a.left, b.left)
    bottom = max(a.bottom, b.bottom)
    right = min(a.right, b.right)
    top = min(a.top, b.top)

    if left >= right or bottom >= top:
        return None

    return left, bottom, right, top


def main():
    print("=" * 80)
    print("PAIR 006 — EXACT NAC ↔ TMC OVERLAP INSPECTION")
    print("=" * 80)

    with rasterio.open(TMC_PATH) as tmc, rasterio.open(NAC_PATH) as nac:
        print("\nTMC")
        print("  path :", TMC_PATH)
        print("  size :", tmc.width, "x", tmc.height)
        print("  res  :", tmc.res)
        print("  CRS  :", tmc.crs)
        print("  bounds:", tmc.bounds)

        print("\nLRO NAC")
        print("  path :", NAC_PATH)
        print("  size :", nac.width, "x", nac.height)
        print("  res  :", nac.res)
        print("  CRS  :", nac.crs)
        print("  bounds:", nac.bounds)
        print("  nodata:", nac.nodata)

        # Transform NAC bounds into TMC CRS only if necessary.
        if nac.crs != tmc.crs:
            print("\nCRS objects differ — transforming NAC bounds into TMC CRS.")

            nac_bounds_tmc = transform_bounds(
                nac.crs,
                tmc.crs,
                *nac.bounds,
                densify_pts=21,
            )
        else:
            print("\nCRS objects match.")
            nac_bounds_tmc = (
                nac.bounds.left,
                nac.bounds.bottom,
                nac.bounds.right,
                nac.bounds.top,
            )

        class Bounds:
            def __init__(self, values):
                self.left, self.bottom, self.right, self.top = values

        overlap = intersection_bounds(
            tmc.bounds,
            Bounds(nac_bounds_tmc),
        )

        if overlap is None:
            print("\nERROR: No raster-bound intersection.")
            return

        left, bottom, right, top = overlap

        print("\nBOUNDING-BOX INTERSECTION")
        print(f"  left   = {left:.3f}")
        print(f"  bottom = {bottom:.3f}")
        print(f"  right  = {right:.3f}")
        print(f"  top    = {top:.3f}")
        print(f"  width  = {right - left:.3f} m")
        print(f"  height = {top - bottom:.3f} m")

        # ------------------------------------------------------------
        # TMC pixels inside overlap
        # ------------------------------------------------------------
        tmc_window = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=tmc.transform,
        )

        tmc_window = tmc_window.round_offsets().round_lengths()

        tmc_arr = tmc.read(
            1,
            window=tmc_window,
            boundless=True,
            fill_value=0,
        )

        # ------------------------------------------------------------
        # NAC pixels
        #
        # Pair006 NAC was deliberately projected at the same 5 m grid
        # family. Read complete mapped crop for first overlap QA.
        # ------------------------------------------------------------
        nac_arr = nac.read(1)

        if nac.nodata is not None:
            nac_valid = (
                np.isfinite(nac_arr)
                & (nac_arr != nac.nodata)
            )
        else:
            nac_valid = np.isfinite(nac_arr)

        # TMC has nodata=None. Positive DN is used here ONLY as matcher
        # support diagnostic. Do not interpret every zero as invalid
        # lunar terrain because true shadows may also be zero.
        tmc_positive = np.isfinite(tmc_arr) & (tmc_arr > 0)

        print("\nTMC OVERLAP WINDOW")
        print("  shape:", tmc_arr.shape)
        print(
            "  positive support:",
            int(tmc_positive.sum()),
            "/",
            tmc_positive.size,
            f"= {tmc_positive.mean():.6f}",
        )

        print("\nNAC VALIDITY")
        print("  shape:", nac_arr.shape)
        print(
            "  valid:",
            int(nac_valid.sum()),
            "/",
            nac_valid.size,
            f"= {nac_valid.mean():.6f}",
        )

        print("\nNAC DN RANGE")
        if np.any(nac_valid):
            vals = nac_arr[nac_valid]
            print("  min   :", float(np.min(vals)))
            print("  max   :", float(np.max(vals)))
            print("  mean  :", float(np.mean(vals)))
            print("  median:", float(np.median(vals)))

        print("\nTMC DN RANGE")
        if np.any(tmc_positive):
            vals = tmc_arr[tmc_positive]
            print("  min   :", float(np.min(vals)))
            print("  max   :", float(np.max(vals)))
            print("  mean  :", float(np.mean(vals)))
            print("  median:", float(np.median(vals)))

        print("\nSUMMARY")
        print(
            f"  NAC valid ratio       : {nac_valid.mean():.6f}"
        )
        print(
            f"  TMC positive ratio    : {tmc_positive.mean():.6f}"
        )

        if tmc_positive.mean() > 0.95:
            print(
                "  TMC support assessment : EXCELLENT "
                "(candidate lies well inside TMC swath)"
            )
        elif tmc_positive.mean() > 0.75:
            print(
                "  TMC support assessment : GOOD"
            )
        else:
            print(
                "  TMC support assessment : PARTIAL — inspect footprint"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()