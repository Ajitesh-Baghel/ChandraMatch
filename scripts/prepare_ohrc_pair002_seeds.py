import sys
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.ingestion.ohrc_product import (
    open_ohrc_image,
    read_ohrc_window
)

from src.ingestion.ohrc_geometry import (
    OHRCGeometryGrid
)


# ============================================================
# PATHS
# ============================================================

PRODUCT_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "ohrc"
    / "2021_12_28"
)


DATA_DIR = (
    PRODUCT_ROOT
    / "data"
    / "calibrated"
    / "20211228"
)


GEOMETRY_DIR = (
    PRODUCT_ROOT
    / "geometry"
    / "calibrated"
    / "20211228"
)


IMG_PATH = next(
    DATA_DIR.glob(
        "*.img"
    )
)


IMAGE_XML_PATH = next(
    DATA_DIR.glob(
        "*.xml"
    )
)


GEOMETRY_CSV_PATH = next(
    GEOMETRY_DIR.glob(
        "*.csv"
    )
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "ohrc"
    / "pair002_seeds"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_JSON = (
    OUTPUT_DIR
    / "seed_regions.json"
)


# ============================================================
# SETTINGS
# ============================================================

PREVIEW_SIZE = 4096


# Three positions along-track.
#
# Avoid extreme beginning/end of image.

SEED_FRACTIONS = {

    "SEED_A": 0.35,

    "SEED_B": 0.60,

    "SEED_C": 0.80
}


# Portal search window.
#
# We only need to find a TMC product intersecting the
# central OHRC terrain.
#
# Portal limits AOI to <= 5 degrees.

SEARCH_LAT_HALF_SPAN = 0.20

SEARCH_LON_HALF_SPAN = 2.40


# ============================================================
# LOAD PRODUCT
# ============================================================

print("=" * 80)

print(
    "CHANDRAMATCH - OHRC PAIR 002 SEED GENERATOR"
)

print("=" * 80)


image, metadata = (
    open_ohrc_image(
        IMG_PATH,
        IMAGE_XML_PATH
    )
)


geometry = OHRCGeometryGrid(
    GEOMETRY_CSV_PATH
)


height, width = image.shape


print(
    "\nOHRC native size:"
)

print(
    width,
    "x",
    height
)


print(
    "Native resolution:",
    metadata[
        "pixel_resolution_m"
    ],
    "m/pixel"
)


# ============================================================
# SEARCH BOX HELPER
# ============================================================

def make_search_boxes(
    longitude,
    latitude
):

    lat_min = max(
        -90.0,
        latitude
        - SEARCH_LAT_HALF_SPAN
    )

    lat_max = min(
        90.0,
        latitude
        + SEARCH_LAT_HALF_SPAN
    )


    lon_min = (
        longitude
        - SEARCH_LON_HALF_SPAN
    )

    lon_max = (
        longitude
        + SEARCH_LON_HALF_SPAN
    )


    # --------------------------------------------------------
    # Normal case
    # --------------------------------------------------------

    if (
        lon_min >= 0.0
        and
        lon_max < 360.0
    ):

        return [
            {
                "min_lat":
                    lat_min,

                "max_lat":
                    lat_max,

                "min_lon":
                    lon_min,

                "max_lon":
                    lon_max
            }
        ]


    # --------------------------------------------------------
    # Longitude wraps below 0
    # --------------------------------------------------------

    if lon_min < 0.0:

        return [

            {
                "min_lat":
                    lat_min,

                "max_lat":
                    lat_max,

                "min_lon":
                    0.0,

                "max_lon":
                    lon_max
            },

            {
                "min_lat":
                    lat_min,

                "max_lat":
                    lat_max,

                "min_lon":
                    lon_min
                    + 360.0,

                "max_lon":
                    360.0
            }
        ]


    # --------------------------------------------------------
    # Longitude wraps above 360
    # --------------------------------------------------------

    return [

        {
            "min_lat":
                lat_min,

            "max_lat":
                lat_max,

            "min_lon":
                lon_min,

            "max_lon":
                360.0
        },

        {
            "min_lat":
                lat_min,

            "max_lat":
                lat_max,

            "min_lon":
                0.0,

            "max_lon":
                lon_max
                - 360.0
        }
    ]


# ============================================================
# GENERATE SEEDS
# ============================================================

results = {}


for seed_name, fraction in (
    SEED_FRACTIONS.items()
):

    print("\n")
    print("=" * 80)

    print(
        seed_name
    )

    print("=" * 80)


    center_x = (
        width // 2
    )


    center_y = int(
        round(
            fraction
            *
            (height - 1)
        )
    )


    longitude, latitude = (
        geometry.pixel_to_lonlat(
            center_x,
            center_y
        )
    )


    print(
        "Image coordinate:"
    )

    print(
        "  Pixel:",
        center_x
    )

    print(
        "  Scan:",
        center_y
    )


    print(
        "\nLunar coordinate:"
    )

    print(
        "  Longitude:",
        longitude
    )

    print(
        "  Latitude:",
        latitude
    )


    # --------------------------------------------------------
    # PREVIEW WINDOW
    # --------------------------------------------------------

    x0 = int(
        center_x
        -
        PREVIEW_SIZE // 2
    )


    y0 = int(
        center_y
        -
        PREVIEW_SIZE // 2
    )


    x0 = max(
        0,
        min(
            x0,
            width
            - PREVIEW_SIZE
        )
    )


    y0 = max(
        0,
        min(
            y0,
            height
            - PREVIEW_SIZE
        )
    )


    preview = read_ohrc_window(

        image,

        x0,
        y0,

        PREVIEW_SIZE,
        PREVIEW_SIZE
    )


    preview_path = (

        OUTPUT_DIR

        / f"{seed_name.lower()}_4096.png"
    )


    cv2.imwrite(
        str(
            preview_path
        ),
        preview
    )


    # --------------------------------------------------------
    # WINDOW FOOTPRINT
    # --------------------------------------------------------

    footprint = (
        geometry.window_footprint(

            x0,
            y0,

            PREVIEW_SIZE,
            PREVIEW_SIZE,

            samples_per_edge=25
        )
    )


    print(
        "\n4096 × 4096 preview footprint:"
    )

    print(
        "  Latitude:",
        footprint[
            "latitude_min"
        ],
        "→",
        footprint[
            "latitude_max"
        ]
    )

    print(
        "  Local longitude:",
        footprint[
            "longitude_min_local"
        ],
        "→",
        footprint[
            "longitude_max_local"
        ]
    )

    print(
        "  Latitude span:",
        footprint[
            "latitude_span_deg"
        ]
    )

    print(
        "  Longitude span:",
        footprint[
            "longitude_span_deg"
        ]
    )


    # --------------------------------------------------------
    # PORTAL SEARCH BOX
    # --------------------------------------------------------

    search_boxes = (
        make_search_boxes(
            longitude,
            latitude
        )
    )


    print(
        "\nTMC-2 portal search:"
    )


    for index, box in enumerate(
        search_boxes,
        start=1
    ):

        print(
            f"\n  Search box {index}:"
        )

        print(
            "    Min Lat:",
            box[
                "min_lat"
            ]
        )

        print(
            "    Max Lat:",
            box[
                "max_lat"
            ]
        )

        print(
            "    Min Lon:",
            box[
                "min_lon"
            ]
        )

        print(
            "    Max Lon:",
            box[
                "max_lon"
            ]
        )


    results[
        seed_name
    ] = {

        "fraction_along_track":
            fraction,

        "center_pixel":
            center_x,

        "center_scan":
            center_y,

        "center_longitude":
            longitude,

        "center_latitude":
            latitude,

        "preview_x":
            x0,

        "preview_y":
            y0,

        "preview_width":
            PREVIEW_SIZE,

        "preview_height":
            PREVIEW_SIZE,

        "preview_path":
            str(
                preview_path.resolve()
            ),

        "footprint":
            footprint,

        "tmc_search_boxes":
            search_boxes
    }


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


print("\n")
print("=" * 80)

print(
    "PAIR 002 SEEDS GENERATED"
)

print("=" * 80)


print(
    "\nSaved:"
)

print(
    OUTPUT_JSON
)


print(
    "\nPreview images:"
)

for path in sorted(
    OUTPUT_DIR.glob(
        "seed_*.png"
    )
):

    print(
        "-",
        path
    )

def lon_360_to_180(longitude):
    """
    Convert longitude from [0, 360)
    to [-180, 180).
    """

    return (
        (longitude + 180.0)
        % 360.0
        - 180.0
    )