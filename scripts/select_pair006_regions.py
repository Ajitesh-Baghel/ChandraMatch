from pathlib import Path
import math

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

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


# ============================================================
# CONSTANTS
# ============================================================

MOON_RADIUS_M = 1_737_400.0

# Pair004 LRO NAC experiment centre.
PAIR004_LAT = -70.715607
PAIR004_LON = 100.313597

TARGET_WIDTH = 2500

N_SECTIONS = 12

SUPPORT_KERNEL = 31


# ============================================================
# LUNAR SOUTH-POLAR STEREOGRAPHIC → LAT/LON
# ============================================================

def polar_stereo_to_lonlat(
    x,
    y,
):
    """
    Inverse spherical south-polar stereographic projection.

    Projection used by the TMC product:

        Moon radius      = 1,737,400 m
        latitude origin  = -90 deg
        central meridian = 0 deg
        false easting    = 0
        false northing   = 0

    For this south-polar convention:

        rho = 2 R tan(pi/4 + lat/2)

        x = rho sin(lon)
        y = rho cos(lon)
    """

    rho = math.hypot(
        x,
        y,
    )

    if rho < 1e-12:

        return (
            -90.0,
            0.0,
        )

    latitude_rad = (
        2.0
        * math.atan(
            rho
            / (
                2.0
                * MOON_RADIUS_M
            )
        )
        - math.pi / 2.0
    )

    longitude_rad = math.atan2(
        x,
        y,
    )

    latitude = math.degrees(
        latitude_rad
    )

    longitude = math.degrees(
        longitude_rad
    )

    # Normalize to [-180, 180].
    longitude = (
        (
            longitude
            + 180.0
        )
        % 360.0
        - 180.0
    )

    return (
        latitude,
        longitude,
    )


# ============================================================
# DISTANCE ON MOON
# ============================================================

def lunar_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    dphi = math.radians(
        lat2 - lat1
    )

    dlon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            dphi / 2.0
        ) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            dlon / 2.0
        ) ** 2
    )

    a = min(
        1.0,
        max(
            0.0,
            a,
        ),
    )

    c = (
        2.0
        * math.atan2(
            math.sqrt(a),
            math.sqrt(
                1.0 - a
            ),
        )
    )

    return (
        MOON_RADIUS_M
        * c
        / 1000.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)

    print(
        "PAIR 006 — SELECT NEW "
        "TMC GENERALIZATION REGIONS"
    )

    print("=" * 78)

    with rasterio.open(
        TMC
    ) as ds:

        print("\nTMC:")
        print(TMC)

        print(
            "\nShape:",
            ds.height,
            "x",
            ds.width,
        )

        print(
            "Resolution:",
            ds.res,
        )

        print(
            "Bounds:",
            ds.bounds,
        )

        print(
            "CRS:"
        )

        print(
            ds.crs
        )

        # ----------------------------------------------------
        # REDUCED REPRESENTATION
        # ----------------------------------------------------

        scale = (
            ds.width
            / TARGET_WIDTH
        )

        target_height = max(
            1,
            int(
                round(
                    ds.height
                    / scale
                )
            ),
        )

        print(
            "\nSampling TMC as:",
            target_height,
            "x",
            TARGET_WIDTH,
        )

        small = ds.read(
            1,
            out_shape=(
                target_height,
                TARGET_WIDTH,
            ),
            resampling=(
                Resampling.average
            ),
        ).astype(
            np.float32
        )

        positive = (
            np.isfinite(
                small
            )
            & (
                small > 0
            )
        )

        print(
            "Positive sampled pixels:",
            f"{int(positive.sum()):,}",
        )

        print(
            "Positive sampled ratio:",
            f"{positive.mean():.6f}",
        )

        # ----------------------------------------------------
        # LOCAL SUPPORT
        #
        # High support means the candidate lies well inside
        # illuminated TMC terrain rather than on the swath edge.
        # ----------------------------------------------------

        support = cv2.boxFilter(
            positive.astype(
                np.float32
            ),
            ddepth=-1,
            ksize=(
                SUPPORT_KERNEL,
                SUPPORT_KERNEL,
            ),
            normalize=True,
            borderType=(
                cv2.BORDER_CONSTANT
            ),
        )

        # ----------------------------------------------------
        # DIVIDE ALONG THE LONG TMC DIMENSION
        # ----------------------------------------------------

        candidates = []

        for section in range(
            N_SECTIONS
        ):

            x0 = int(
                section
                * TARGET_WIDTH
                / N_SECTIONS
            )

            x1 = int(
                (
                    section + 1
                )
                * TARGET_WIDTH
                / N_SECTIONS
            )

            section_support = (
                support[
                    :,
                    x0:x1,
                ]
            )

            if (
                section_support.size
                == 0
                or np.max(
                    section_support
                )
                <= 0
            ):

                continue

            local_y, local_x = (
                np.unravel_index(
                    np.argmax(
                        section_support
                    ),
                    section_support.shape,
                )
            )

            x_small = (
                x0
                + local_x
            )

            y_small = (
                local_y
            )

            support_value = float(
                support[
                    y_small,
                    x_small,
                ]
            )

            # -----------------------------------------------
            # Reduced pixel → original raster pixel
            # -----------------------------------------------

            col = (
                (
                    x_small
                    + 0.5
                )
                * ds.width
                / TARGET_WIDTH
                - 0.5
            )

            row = (
                (
                    y_small
                    + 0.5
                )
                * ds.height
                / target_height
                - 0.5
            )

            # -----------------------------------------------
            # Original pixel → projected coordinate
            # -----------------------------------------------

            x_map, y_map = ds.xy(
                row,
                col,
                offset="center",
            )

            x_map = float(
                x_map
            )

            y_map = float(
                y_map
            )

            # -----------------------------------------------
            # Lunar projected coordinate → lon/lat
            # -----------------------------------------------

            lat, lon = (
                polar_stereo_to_lonlat(
                    x_map,
                    y_map,
                )
            )

            distance_pair004 = (
                lunar_distance_km(
                    PAIR004_LAT,
                    PAIR004_LON,
                    lat,
                    lon,
                )
            )

            candidates.append(
                {
                    "section":
                        section + 1,

                    "row":
                        float(row),

                    "col":
                        float(col),

                    "support":
                        support_value,

                    "x":
                        x_map,

                    "y":
                        y_map,

                    "lat":
                        lat,

                    "lon":
                        lon,

                    "distance_pair004_km":
                        distance_pair004,
                }
            )

        # ----------------------------------------------------
        # PRINT IN SWATH ORDER
        # ----------------------------------------------------

        print("\n")
        print("=" * 78)

        print(
            "CANDIDATES ALONG TMC SWATH"
        )

        print("=" * 78)

        for c in candidates:

            print(
                f"\nCandidate "
                f"{c['section']:02d}"
            )

            print(
                "  Original pixel:",
                f"row={c['row']:.1f}",
                f"col={c['col']:.1f}",
            )

            print(
                "  Projected:",
                f"x={c['x']:.1f} m",
                f"y={c['y']:.1f} m",
            )

            print(
                "  Lunar Lat/Lon:",
                f"{c['lat']:.6f},",
                f"{c['lon']:.6f}",
            )

            print(
                "  Local positive support:",
                f"{c['support']:.4f}",
            )

            print(
                "  Distance from Pair004:",
                f"{c['distance_pair004_km']:.2f} km",
            )

        # ----------------------------------------------------
        # RANK CANDIDATES
        #
        # We prefer:
        #
        #   high image support
        #   +
        #   far from Pair004
        #
        # This is not the final scene selection because LRO
        # NAC availability still needs to be checked.
        # ----------------------------------------------------

        eligible = [
            c
            for c in candidates
            if c[
                "support"
            ] >= 0.70
        ]

        if not eligible:

            eligible = (
                candidates.copy()
            )

        ranked = sorted(
            eligible,
            key=lambda c: (
                c[
                    "distance_pair004_km"
                ],
                c[
                    "support"
                ],
            ),
            reverse=True,
        )

        print("\n")
        print("=" * 78)

        print(
            "PAIR 006 SEARCH PRIORITY"
        )

        print("=" * 78)

        print(
            "\nPair004 reference centre:"
        )

        print(
            f"  lat = "
            f"{PAIR004_LAT:.6f}"
        )

        print(
            f"  lon = "
            f"{PAIR004_LON:.6f}"
        )

        print(
            "\nBest geographically independent "
            "candidates:"
        )

        for rank, c in enumerate(
            ranked[:5],
            start=1,
        ):

            print(
                f"\n#{rank} "
                f"Candidate "
                f"{c['section']:02d}"
            )

            print(
                f"  Lat/Lon: "
                f"{c['lat']:.6f}, "
                f"{c['lon']:.6f}"
            )

            print(
                f"  TMC support: "
                f"{c['support']:.4f}"
            )

            print(
                f"  Separation from Pair004: "
                f"{c['distance_pair004_km']:.2f} km"
            )

            # Useful initial LROC search box.
            print(
                "  Suggested LROC search box:"
            )

            print(
                f"    North = "
                f"{c['lat'] + 0.40:.4f}"
            )

            print(
                f"    South = "
                f"{c['lat'] - 0.40:.4f}"
            )

            print(
                f"    West  = "
                f"{c['lon'] - 0.60:.4f}"
            )

            print(
                f"    East  = "
                f"{c['lon'] + 0.60:.4f}"
            )

        print("\n")
        print("=" * 78)

        print(
            "PAIR 006 REGION SELECTION COMPLETE"
        )

        print("=" * 78)


if __name__ == "__main__":
    main()