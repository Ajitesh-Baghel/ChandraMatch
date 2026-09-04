import sys
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.windows import Window

from pyproj import CRS as PyCRS
from pyproj import Transformer


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(ROOT)
)


# ============================================================
# CHANDRAMATCH MODULES
# ============================================================

from src.ingestion.tmc2_metadata import (
    inspect_tmc2_product
)


# ============================================================
# TMC-2 CANDIDATE
#
# Usage:
#
# python scripts\inspect_pair002_tmc_candidate.py ^
#     data\raw\tmc2\2022_01_09
#
# If no path is supplied, the old 2023 candidate is used.
# ============================================================

if len(sys.argv) >= 2:

    PRODUCT_ROOT = Path(
        sys.argv[1]
    )

    if not PRODUCT_ROOT.is_absolute():

        PRODUCT_ROOT = (
            ROOT
            / PRODUCT_ROOT
        )

else:

    PRODUCT_ROOT = (
        ROOT
        / "data"
        / "raw"
        / "tmc2"
        / "2023_05_12"
    )


PRODUCT_ROOT = PRODUCT_ROOT.resolve()

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_002"
    / "tmc_candidate"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


METADATA_OUTPUT = (
    OUTPUT_DIR
    / "tmc_candidate_metadata.json"
)


PREVIEW_OUTPUT = (
    OUTPUT_DIR
    / "tmc_candidate_preview.png"
)


SEED_WINDOW_OUTPUT = (
    OUTPUT_DIR
    / "tmc_seed_neighbourhood.png"
)


# ============================================================
# OHRC SEED C
#
# Coordinates derived from the OHRC geometry grid.
#
# OHRC longitude convention:
#     0 ... 360 degrees
#
# Portal / conventional geographic representation:
#     -180 ... +180 degrees
# ============================================================

OHRC_SEED_C_LONGITUDE_360 = (
    227.63395217227017
)

OHRC_SEED_C_LATITUDE = (
    -89.42078231045038
)


OHRC_SEED_C_LONGITUDE_180 = (

    (
        OHRC_SEED_C_LONGITUDE_360
        + 180.0
    )

    % 360.0

    - 180.0
)


# ============================================================
# LOCAL VALIDITY SEARCH
#
# 400 TMC pixels × 5 m/pixel ≈ 2 km radius.
#
# This helps avoid rejecting a potentially useful product
# just because the exact projected centre lies on one fill
# pixel while valid swath data exists nearby.
# ============================================================

LOCAL_SEARCH_RADIUS_PIXELS = 400


# ============================================================
# FIND TIFF
# ============================================================

if not PRODUCT_ROOT.exists():

    raise RuntimeError(
        f"TMC candidate folder does not exist:\n"
        f"{PRODUCT_ROOT}"
    )


tif_files = sorted(
    PRODUCT_ROOT.rglob(
        "*.tif"
    )
)


if not tif_files:

    raise RuntimeError(
        f"No TIFF found under:\n"
        f"{PRODUCT_ROOT}"
    )


print("=" * 80)
print(
    "CHANDRAMATCH - PAIR 002 TMC-2 CANDIDATE INSPECTOR"
)
print("=" * 80)


print(
    "\nCandidate directory:"
)

print(
    PRODUCT_ROOT
)


print(
    "\nTIFF files found:",
    len(tif_files)
)


for index, path in enumerate(
    tif_files,
    start=1
):

    print(
        f"[{index}] {path}"
    )


# For a normal downloaded product there should be one
# actual derived TIFF.

TIF_PATH = tif_files[0]


print("\nUsing:")

print(
    TIF_PATH
)


# ============================================================
# READ TMC PRODUCT METADATA
# ============================================================

try:

    (
        record,
        xml_fields

    ) = inspect_tmc2_product(
        TIF_PATH
    )


except Exception as error:

    print("\n")
    print(
        "WARNING: standard TMC metadata parser failed."
    )

    print(
        error
    )

    record = {}

    xml_fields = []


# ============================================================
# VARIABLES FILLED DURING INSPECTION
# ============================================================

preview_valid_ratio = None

is_geographic = False

projected_x = None
projected_y = None

projected_inside_bbox = False

projected_row = None
projected_col = None

seed_pixel_value = None

valid_pixel_at_seed = False

local_valid_ratio = 0.0
local_valid_pixels = 0
local_total_pixels = 0

candidate_accepted = False

projection_name = None

latitude_of_origin = None

central_meridian = None


# ============================================================
# RASTER INSPECTION
# ============================================================

with rasterio.open(
    TIF_PATH
) as src:

    print("\n")
    print("=" * 80)
    print(
        "RASTER INFORMATION"
    )
    print("=" * 80)


    print(
        "Width:",
        src.width
    )

    print(
        "Height:",
        src.height
    )

    print(
        "Bands:",
        src.count
    )

    print(
        "Dtype:",
        src.dtypes
    )

    print(
        "NoData:",
        src.nodata
    )

    print(
        "Resolution:",
        src.res
    )


    print(
        "\nCRS:"
    )

    print(
        src.crs
    )


    print(
        "\nTransform:"
    )

    print(
        src.transform
    )


    print(
        "\nBounds:"
    )

    print(
        "Left:",
        src.bounds.left
    )

    print(
        "Bottom:",
        src.bounds.bottom
    )

    print(
        "Right:",
        src.bounds.right
    )

    print(
        "Top:",
        src.bounds.top
    )


    # ========================================================
    # CRS DETAILS
    # ========================================================

    is_geographic = bool(
        src.crs
        and
        src.crs.is_geographic
    )


    print(
        "\nCRS is geographic:",
        is_geographic
    )


    if src.crs:

        try:

            crs_dict = src.crs.to_dict()

            projection_name = (
                crs_dict.get(
                    "proj"
                )
            )

            latitude_of_origin = (
                crs_dict.get(
                    "lat_0"
                )
            )

            central_meridian = (
                crs_dict.get(
                    "lon_0"
                )
            )


            print(
                "Projection:",
                projection_name
            )

            print(
                "Latitude of origin:",
                latitude_of_origin
            )

            print(
                "Central meridian:",
                central_meridian
            )


        except Exception as error:

            print(
                "Could not decode CRS parameters:",
                error
            )


    # ========================================================
    # RASTER TAGS
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "RASTER TAGS"
    )
    print("=" * 80)


    tags = src.tags()


    if tags:

        for key, value in (
            tags.items()
        ):

            print(
                f"{key}: {value}"
            )

    else:

        print(
            "No raster tags."
        )


    # ========================================================
    # LOW-RES FULL PRODUCT PREVIEW
    # ========================================================

    max_preview_dimension = 1600


    preview_scale = min(

        1.0,

        max_preview_dimension
        /
        max(
            src.width,
            src.height
        )
    )


    preview_width = max(

        1,

        int(
            round(
                src.width
                * preview_scale
            )
        )
    )


    preview_height = max(

        1,

        int(
            round(
                src.height
                * preview_scale
            )
        )
    )


    preview_raw = src.read(

        1,

        out_shape=(
            preview_height,
            preview_width
        ),

        resampling=
            Resampling.average
    )


    # ========================================================
    # PREVIEW VALIDITY
    # ========================================================

    preview_valid = (
        preview_raw > 0
    )


    preview_valid_ratio = float(
        preview_valid.mean()
    )


    print("\n")
    print("=" * 80)
    print(
        "PREVIEW VALIDITY"
    )
    print("=" * 80)


    print(
        "Preview shape:",
        preview_raw.shape
    )


    print(
        "Non-zero valid ratio:",
        preview_valid_ratio
    )


    preview = np.zeros(
        preview_raw.shape,
        dtype=np.uint8
    )


    values = preview_raw[
        preview_valid
    ]


    if values.size > 0:

        low, high = np.percentile(
            values,
            [1, 99]
        )


        print(
            "Valid minimum:",
            float(
                np.min(values)
            )
        )

        print(
            "Valid maximum:",
            float(
                np.max(values)
            )
        )

        print(
            "1st percentile:",
            float(low)
        )

        print(
            "99th percentile:",
            float(high)
        )


        if high > low:

            work = np.clip(

                preview_raw.astype(
                    np.float32
                ),

                low,
                high
            )


            work = (

                (
                    work
                    - low
                )

                /

                (
                    high
                    - low
                )

                * 255.0
            )


            preview[
                preview_valid
            ] = (

                work[
                    preview_valid
                ]
                .astype(
                    np.uint8
                )
            )


    cv2.imwrite(

        str(
            PREVIEW_OUTPUT
        ),

        preview
    )


    print(
        "\nPreview saved:"
    )

    print(
        PREVIEW_OUTPUT
    )


    # ========================================================
    # OHRC SEED INFORMATION
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "OHRC SEED C"
    )
    print("=" * 80)


    print(
        "Longitude [0,360):",
        OHRC_SEED_C_LONGITUDE_360
    )


    print(
        "Longitude [-180,180):",
        OHRC_SEED_C_LONGITUDE_180
    )


    print(
        "Latitude:",
        OHRC_SEED_C_LATITUDE
    )


    # ========================================================
    # OHRC GEOGRAPHIC COORDINATE → TMC CRS
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "OHRC → TMC CRS TRANSFORMATION"
    )
    print("=" * 80)


    if src.crs is None:

        print(
            "TMC raster has no CRS."
        )


    else:

        try:

            tmc_crs = PyCRS.from_wkt(
                src.crs.to_wkt()
            )


            lunar_geographic_crs = (
                tmc_crs.geodetic_crs
            )


            print(
                "Lunar geographic CRS:"
            )

            print(
                lunar_geographic_crs
            )


            transformer = (
                Transformer.from_crs(

                    lunar_geographic_crs,

                    tmc_crs,

                    always_xy=True
                )
            )


            (
                projected_x,
                projected_y

            ) = transformer.transform(

                OHRC_SEED_C_LONGITUDE_180,

                OHRC_SEED_C_LATITUDE
            )


            projected_x = float(
                projected_x
            )

            projected_y = float(
                projected_y
            )


            print(
                "\nProjected OHRC Seed C:"
            )

            print(
                "  X:",
                projected_x
            )

            print(
                "  Y:",
                projected_y
            )


            # =================================================
            # BOUNDING-BOX CHECK
            # =================================================

            projected_inside_bbox = bool(

                np.isfinite(
                    projected_x
                )

                and

                np.isfinite(
                    projected_y
                )

                and

                src.bounds.left
                <= projected_x
                <= src.bounds.right

                and

                src.bounds.bottom
                <= projected_y
                <= src.bounds.top
            )


            print(
                "\nProjected seed inside TMC bounding box:",
                projected_inside_bbox
            )


            # =================================================
            # PIXEL CHECK
            # =================================================

            if projected_inside_bbox:

                (
                    projected_row,
                    projected_col

                ) = src.index(

                    projected_x,
                    projected_y
                )


                projected_row = int(
                    projected_row
                )

                projected_col = int(
                    projected_col
                )


                print(
                    "\nProjected TMC pixel:"
                )

                print(
                    "  Column:",
                    projected_col
                )

                print(
                    "  Row:",
                    projected_row
                )


                if (

                    0
                    <= projected_row
                    < src.height

                    and

                    0
                    <= projected_col
                    < src.width
                ):

                    pixel_window = Window(

                        projected_col,

                        projected_row,

                        1,

                        1
                    )


                    seed_pixel_value = (
                        src.read(

                            1,

                            window=
                                pixel_window
                        )[0, 0]
                    )


                    seed_pixel_value = float(
                        seed_pixel_value
                    )


                    valid_pixel_at_seed = bool(
                        seed_pixel_value > 0
                    )


                    print(
                        "\nPixel value at exact OHRC seed:",
                        seed_pixel_value
                    )

                    print(
                        "Exact seed pixel valid:",
                        valid_pixel_at_seed
                    )


                    # =========================================
                    # LOCAL NEIGHBOURHOOD VALIDITY
                    # =========================================

                    radius = (
                        LOCAL_SEARCH_RADIUS_PIXELS
                    )


                    x0 = max(

                        0,

                        projected_col
                        - radius
                    )


                    y0 = max(

                        0,

                        projected_row
                        - radius
                    )


                    x1 = min(

                        src.width,

                        projected_col
                        + radius
                        + 1
                    )


                    y1 = min(

                        src.height,

                        projected_row
                        + radius
                        + 1
                    )


                    local_window = Window(

                        x0,
                        y0,

                        x1 - x0,
                        y1 - y0
                    )


                    local_data = src.read(

                        1,

                        window=
                            local_window
                    )


                    local_valid = (
                        local_data > 0
                    )


                    local_valid_pixels = int(
                        np.count_nonzero(
                            local_valid
                        )
                    )


                    local_total_pixels = int(
                        local_valid.size
                    )


                    if local_total_pixels > 0:

                        local_valid_ratio = float(

                            local_valid_pixels

                            /

                            local_total_pixels
                        )


                    print("\n")
                    print(
                        "Local seed neighbourhood:"
                    )

                    print(
                        "  Radius:",
                        radius,
                        "TMC pixels"
                    )

                    print(
                        "  Approx physical radius:",
                        radius
                        * abs(
                            src.res[0]
                        ),
                        "metres"
                    )

                    print(
                        "  Valid pixels:",
                        local_valid_pixels
                    )

                    print(
                        "  Total pixels:",
                        local_total_pixels
                    )

                    print(
                        "  Valid ratio:",
                        local_valid_ratio
                    )


                    # =========================================
                    # SAVE LOCAL PREVIEW
                    # =========================================

                    local_preview = np.zeros(

                        local_data.shape,

                        dtype=np.uint8
                    )


                    local_values = local_data[
                        local_valid
                    ]


                    if local_values.size > 0:

                        local_low, local_high = (
                            np.percentile(

                                local_values,

                                [1, 99]
                            )
                        )


                        if local_high > local_low:

                            local_work = np.clip(

                                local_data.astype(
                                    np.float32
                                ),

                                local_low,

                                local_high
                            )


                            local_work = (

                                (
                                    local_work
                                    - local_low
                                )

                                /

                                (
                                    local_high
                                    - local_low
                                )

                                * 255.0
                            )


                            local_preview[
                                local_valid
                            ] = (

                                local_work[
                                    local_valid
                                ]
                                .astype(
                                    np.uint8
                                )
                            )


                    # Mark exact projected seed.

                    local_seed_x = (
                        projected_col
                        - x0
                    )

                    local_seed_y = (
                        projected_row
                        - y0
                    )


                    if (

                        0
                        <= local_seed_x
                        < local_preview.shape[1]

                        and

                        0
                        <= local_seed_y
                        < local_preview.shape[0]
                    ):

                        cv2.drawMarker(

                            local_preview,

                            (
                                int(
                                    local_seed_x
                                ),

                                int(
                                    local_seed_y
                                )
                            ),

                            255,

                            markerType=
                                cv2.MARKER_CROSS,

                            markerSize=31,

                            thickness=2
                        )


                    cv2.imwrite(

                        str(
                            SEED_WINDOW_OUTPUT
                        ),

                        local_preview
                    )


                    print(
                        "\nSeed neighbourhood saved:"
                    )

                    print(
                        SEED_WINDOW_OUTPUT
                    )


        except Exception as error:

            print(
                "\nCRS transformation failed:"
            )

            print(
                error
            )


# ============================================================
# PRODUCT METADATA
# ============================================================

print("\n")
print("=" * 80)
print(
    "TMC-2 PRODUCT METADATA"
)
print("=" * 80)


important_fields = [

    "product_id",
    "acquisition_date",

    "sun_azimuth_deg",
    "sun_elevation_deg",
    "incidence_angle_deg",

    "width",
    "height",

    "resolution_x",
    "resolution_y",

    "left",
    "bottom",
    "right",
    "top",

    "crs",

    "xml_path"
]


for key in important_fields:

    print(
        f"{key}: "
        f"{record.get(key)}"
    )


# ============================================================
# AUTOMATIC CANDIDATE DECISION
# ============================================================

#
# ACCEPT:
#
# 1. OHRC seed transforms into this TMC raster's bbox
#
# AND
#
# 2. either:
#       exact seed pixel contains real data
#    OR
#       valid TMC data exists in the local ~2 km region.
#
# The neighbourhood condition is useful because the actual
# observation footprint can be narrow and the geometry
# datasets are not guaranteed to line up at exactly one pixel.
#

local_data_available = bool(
    local_valid_pixels > 0
)


candidate_accepted = bool(

    projected_inside_bbox

    and

    (
        valid_pixel_at_seed
        or
        local_data_available
    )
)


print("\n")
print("=" * 80)


if candidate_accepted:

    print(
        "PAIR 002 CANDIDATE STATUS: ACCEPT"
    )

    print(
        "The OHRC Seed C location intersects "
        "valid TMC-2 imagery."
    )


else:

    print(
        "PAIR 002 CANDIDATE STATUS: REJECT"
    )


    if not projected_inside_bbox:

        print(
            "Reason:"
        )

        print(
            "OHRC Seed C does not lie inside "
            "this TMC-2 projected raster."
        )


    elif not local_data_available:

        print(
            "Reason:"
        )

        print(
            "The geographic location falls inside "
            "the raster bounds, but no valid TMC-2 "
            "observation pixels were found around "
            "the OHRC seed."
        )


print("=" * 80)


# ============================================================
# STRUCTURED REPORT
# ============================================================

report = {

    "tmc_candidate": {

        "product_root":
            str(
                PRODUCT_ROOT.resolve()
            ),

        "tif_path":
            str(
                TIF_PATH.resolve()
            ),

        "product_id":
            record.get(
                "product_id"
            ),

        "acquisition_date":
            record.get(
                "acquisition_date"
            ),

        "sun_azimuth_deg":
            record.get(
                "sun_azimuth_deg"
            ),

        "sun_elevation_deg":
            record.get(
                "sun_elevation_deg"
            ),

        "pixel_resolution":
            [
                record.get(
                    "resolution_x"
                ),

                record.get(
                    "resolution_y"
                )
            ],

        "preview_valid_ratio":
            preview_valid_ratio,

        "projection_name":
            projection_name,

        "latitude_of_origin":
            latitude_of_origin,

        "central_meridian":
            central_meridian
    },


    "ohrc_seed_c": {

        "longitude_360":
            OHRC_SEED_C_LONGITUDE_360,

        "longitude_180":
            OHRC_SEED_C_LONGITUDE_180,

        "latitude":
            OHRC_SEED_C_LATITUDE
    },


    "projection_test": {

        "projected_x":
            projected_x,

        "projected_y":
            projected_y,

        "inside_tmc_bbox":
            projected_inside_bbox,

        "tmc_pixel_row":
            projected_row,

        "tmc_pixel_col":
            projected_col,

        "exact_seed_pixel_value":
            seed_pixel_value,

        "exact_seed_pixel_valid":
            valid_pixel_at_seed,

        "local_search_radius_pixels":
            LOCAL_SEARCH_RADIUS_PIXELS,

        "local_valid_pixels":
            local_valid_pixels,

        "local_total_pixels":
            local_total_pixels,

        "local_valid_ratio":
            local_valid_ratio
    },


    "candidate_accepted":
        candidate_accepted
}


with open(
    METADATA_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(

        report,

        file,

        indent=4,

        ensure_ascii=False,

        default=str
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 80)

print(
    "PAIR 002 TMC CANDIDATE INSPECTION COMPLETE"
)

print("=" * 80)


print(
    "\nMetadata saved:"
)

print(
    METADATA_OUTPUT
)


print(
    "\nFull-product preview:"
)

print(
    PREVIEW_OUTPUT
)


if projected_inside_bbox:

    print(
        "\nSeed-neighbourhood preview:"
    )

    print(
        SEED_WINDOW_OUTPUT
    )