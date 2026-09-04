from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]

KAGUYA_DIR = (
    ROOT
    / "data"
    / "raw"
    / "kaguya_tc"
    / "2008_02_17"
)

PRODUCT = "DTMTCO_03_01597S711E1002PS"

FILES = {
    "TC_ORTHO": KAGUYA_DIR / f"{PRODUCT}_img.lbl",
    "DTM": KAGUYA_DIR / f"{PRODUCT}_dtm.lbl",
    "DQA": KAGUYA_DIR / f"{PRODUCT}_dqa.lbl",
}


def inspect(name, path):
    print("\n")
    print("=" * 76)
    print(name)
    print("=" * 76)

    print("Path:")
    print(path)

    try:
        with rasterio.open(path) as ds:

            print("\nOPEN SUCCESS")

            print("Driver       :", ds.driver)
            print("Width        :", ds.width)
            print("Height       :", ds.height)
            print("Bands        :", ds.count)
            print("Dtypes       :", ds.dtypes)
            print("Nodata       :", ds.nodata)

            print("\nCRS:")
            print(ds.crs)

            print("\nTransform:")
            print(ds.transform)

            print("\nBounds:")
            print(ds.bounds)

            data = ds.read(1)

            raster_mask = (
                ds.read_masks(1) > 0
            )

            finite = np.isfinite(data)

            valid = (
                raster_mask
                & finite
            )

            print(
                "\nRaster-mask valid:",
                f"{int(raster_mask.sum()):,}",
                "/",
                f"{data.size:,}",
                f"({raster_mask.mean():.6f})",
            )

            print(
                "Finite pixels:",
                f"{int(finite.sum()):,}",
                "/",
                f"{data.size:,}",
                f"({finite.mean():.6f})",
            )

            print(
                "Combined valid:",
                f"{int(valid.sum()):,}",
                "/",
                f"{data.size:,}",
                f"({valid.mean():.6f})",
            )

            if np.any(valid):

                values = data[valid]

                print("\nValid-value statistics:")

                print(
                    "Minimum:",
                    float(
                        np.min(values)
                    ),
                )

                print(
                    "Maximum:",
                    float(
                        np.max(values)
                    ),
                )

                print(
                    "Mean:",
                    float(
                        np.mean(values)
                    ),
                )

                print(
                    "Median:",
                    float(
                        np.median(values)
                    ),
                )

                print(
                    "Zero pixels:",
                    f"{int(np.sum(values == 0)):,}",
                )

                percentiles = np.percentile(
                    values,
                    [
                        0.1,
                        1,
                        2,
                        50,
                        98,
                        99,
                        99.9,
                    ],
                )

                print("\nPercentiles:")

                for p, value in zip(
                    [
                        0.1,
                        1,
                        2,
                        50,
                        98,
                        99,
                        99.9,
                    ],
                    percentiles,
                ):

                    print(
                        f"  {p:5.1f}% : "
                        f"{float(value):.6f}"
                    )

            print("\nCorner map coordinates:")

            corners = [
                ("UL", 0, 0),
                (
                    "UR",
                    ds.width - 1,
                    0,
                ),
                (
                    "LL",
                    0,
                    ds.height - 1,
                ),
                (
                    "LR",
                    ds.width - 1,
                    ds.height - 1,
                ),
            ]

            for (
                label,
                col,
                row,
            ) in corners:

                x, y = ds.xy(
                    row,
                    col,
                    offset="center",
                )

                print(
                    f"{label}: "
                    f"x={x:.6f}, "
                    f"y={y:.6f}"
                )

    except Exception as exc:

        print("\nOPEN FAILED")

        print(
            type(exc).__name__
            + ":",
            exc,
        )


def main():

    print("=" * 76)

    print(
        "PAIR 005 — KAGUYA TC "
        "PDS3 INSPECTION"
    )

    print("=" * 76)

    for name, path in FILES.items():

        inspect(
            name,
            path,
        )

    print("\n")
    print("=" * 76)

    print(
        "PAIR 005 INSPECTION COMPLETE"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()