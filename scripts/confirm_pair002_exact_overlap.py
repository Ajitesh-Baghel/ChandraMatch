import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio

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
# PATHS
# ============================================================

OHRC_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "ohrc"
    / "2021_12_28"
)


TMC_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "tmc2"
    / "2023_10_25"
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_002"
    / "exact_overlap"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CSV_OUTPUT = (
    OUTPUT_DIR
    / "exact_overlap_points.csv"
)


MASK_OUTPUT = (
    OUTPUT_DIR
    / "exact_overlap_mask.png"
)


LARGEST_COMPONENT_OUTPUT = (
    OUTPUT_DIR
    / "largest_overlap_component.png"
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "exact_overlap_summary.json"
)


# ============================================================
# FIND INPUT FILES
# ============================================================

tmc_files = sorted(
    TMC_ROOT.rglob(
        "*.tif"
    )
)


if not tmc_files:

    raise RuntimeError(
        f"No TMC TIFF found under:\n"
        f"{TMC_ROOT}"
    )


TMC_TIF = tmc_files[0]


geometry_files = sorted(
    (
        OHRC_ROOT
        / "geometry"
        / "calibrated"
    ).rglob(
        "*.csv"
    )
)


if not geometry_files:

    raise RuntimeError(
        f"No OHRC geometry CSV found under:\n"
        f"{OHRC_ROOT}"
    )


OHRC_GEOMETRY_CSV = (
    geometry_files[0]
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)

print(
    "CHANDRAMATCH - PAIR 002 EXACT OVERLAP CONFIRMATION"
)

print("=" * 80)


print(
    "\nOHRC geometry:"
)

print(
    OHRC_GEOMETRY_CSV
)


print(
    "\nTMC product:"
)

print(
    TMC_TIF
)


# ============================================================
# LOAD OHRC GEOMETRY
# ============================================================

geometry = pd.read_csv(
    OHRC_GEOMETRY_CSV
)


required_columns = {
    "Longitude",
    "Latitude",
    "Pixel",
    "Scan"
}


missing = (
    required_columns
    - set(
        geometry.columns
    )
)


if missing:

    raise RuntimeError(
        f"Missing geometry columns: {missing}"
    )


longitude_360 = (
    geometry[
        "Longitude"
    ].to_numpy(
        dtype=np.float64
    )
)


latitude = (
    geometry[
        "Latitude"
    ].to_numpy(
        dtype=np.float64
    )
)


ohrc_pixel = (
    geometry[
        "Pixel"
    ].to_numpy(
        dtype=np.int64
    )
)


ohrc_scan = (
    geometry[
        "Scan"
    ].to_numpy(
        dtype=np.int64
    )
)


# ============================================================
# LONGITUDE 0..360 → -180..180
# ============================================================

longitude_180 = (

    (
        longitude_360
        + 180.0
    )

    % 360.0

    - 180.0
)


print(
    "\nOHRC geometry points:",
    len(
        geometry
    )
)


# ============================================================
# OPEN TMC
# ============================================================

with rasterio.open(
    TMC_TIF
) as src:

    print("\n")
    print("=" * 80)
    print("TMC PRODUCT")
    print("=" * 80)


    print(
        "Shape:",
        src.height,
        "x",
        src.width
    )


    print(
        "Resolution:",
        src.res
    )


    print(
        "Bounds:",
        src.bounds
    )


    print(
        "CRS:"
    )

    print(
        src.crs
    )


    if src.crs is None:

        raise RuntimeError(
            "TMC product has no CRS."
        )


    # ========================================================
    # LUNAR GEOGRAPHIC → TMC CRS
    # ========================================================

    tmc_crs = PyCRS.from_wkt(
        src.crs.to_wkt()
    )


    lunar_geographic_crs = (
        tmc_crs.geodetic_crs
    )


    transformer = Transformer.from_crs(

        lunar_geographic_crs,

        tmc_crs,

        always_xy=True
    )


    (
        projected_x,
        projected_y

    ) = transformer.transform(

        longitude_180,

        latitude
    )


    projected_x = np.asarray(
        projected_x,
        dtype=np.float64
    )


    projected_y = np.asarray(
        projected_y,
        dtype=np.float64
    )


    # ========================================================
    # PROJECTED COORDINATES → TMC PIXELS
    # ========================================================

    inverse_transform = (
        ~src.transform
    )


    (
        cols_float,
        rows_float

    ) = inverse_transform * (

        projected_x,
        projected_y
    )


    cols_float = np.asarray(
        cols_float,
        dtype=np.float64
    )


    rows_float = np.asarray(
        rows_float,
        dtype=np.float64
    )


    rows = np.floor(
        rows_float
    ).astype(
        np.int64
    )


    cols = np.floor(
        cols_float
    ).astype(
        np.int64
    )


    finite = (

        np.isfinite(
            projected_x
        )

        &

        np.isfinite(
            projected_y
        )
    )


    inside = (

        finite

        &

        (
            rows >= 0
        )

        &

        (
            rows < src.height
        )

        &

        (
            cols >= 0
        )

        &

        (
            cols < src.width
        )
    )


    inside_indices = (
        np.flatnonzero(
            inside
        )
    )


    print(
        "\nOHRC geometry points inside TMC raster:",
        len(
            inside_indices
        ),
        "/",
        len(
            geometry
        )
    )


    if len(
        inside_indices
    ) == 0:

        raise RuntimeError(
            "No OHRC geometry points fall inside "
            "the TMC raster."
        )


    # ========================================================
    # READ ONLY THE PART OF THE TMC IMAGE COVERING OHRC
    #
    # This avoids loading the entire ~12.5 GB TIFF.
    # ========================================================

    min_row = int(
        rows[
            inside_indices
        ].min()
    )


    max_row = int(
        rows[
            inside_indices
        ].max()
    )


    min_col = int(
        cols[
            inside_indices
        ].min()
    )


    max_col = int(
        cols[
            inside_indices
        ].max()
    )


    # Small safety padding.

    PADDING = 2


    min_row = max(
        0,
        min_row - PADDING
    )


    max_row = min(
        src.height - 1,
        max_row + PADDING
    )


    min_col = max(
        0,
        min_col - PADDING
    )


    max_col = min(
        src.width - 1,
        max_col + PADDING
    )


    window_width = (
        max_col
        - min_col
        + 1
    )


    window_height = (
        max_row
        - min_row
        + 1
    )


    print("\n")
    print("=" * 80)
    print("TMC OHRC-TRACK WINDOW")
    print("=" * 80)


    print(
        "Rows:",
        min_row,
        "to",
        max_row
    )


    print(
        "Columns:",
        min_col,
        "to",
        max_col
    )


    print(
        "Window shape:",
        window_height,
        "x",
        window_width
    )


    estimated_mb = (

        window_height
        * window_width
        * 2

        / 1024
        / 1024
    )


    print(
        "Approx uint16 memory:",
        estimated_mb,
        "MB"
    )


    window = Window(

        min_col,
        min_row,

        window_width,
        window_height
    )


    print(
        "\nReading native TMC window..."
    )


    tmc_data = src.read(

        1,

        window=window
    )


    print(
        "Native TMC window loaded."
    )


# ============================================================
# SAMPLE EXACT TMC VALUES
# ============================================================

tmc_values = np.zeros(
    len(
        geometry
    ),
    dtype=np.uint16
)


local_rows = (
    rows[
        inside_indices
    ]
    - min_row
)


local_cols = (
    cols[
        inside_indices
    ]
    - min_col
)


tmc_values[
    inside_indices
] = tmc_data[

    local_rows,

    local_cols
]


# ============================================================
# EXACT VALID OVERLAP
#
# In these TMC products:
#     0 = fill / invalid
#     >0 = image data
# ============================================================

exact_overlap = (

    inside

    &

    (
        tmc_values > 0
    )
)


exact_indices = (
    np.flatnonzero(
        exact_overlap
    )
)


exact_count = int(
    len(
        exact_indices
    )
)


exact_ratio = float(

    exact_count

    /

    len(
        geometry
    )
)


print("\n")
print("=" * 80)
print("EXACT FULL-RESOLUTION OVERLAP")
print("=" * 80)


print(
    "Exact OHRC geometry points on valid TMC pixels:",
    exact_count
)


print(
    "Total OHRC geometry points:",
    len(
        geometry
    )
)


print(
    "Exact overlap ratio:",
    exact_ratio
)


print(
    "Exact overlap percentage:",
    exact_ratio * 100.0,
    "%"
)


# ============================================================
# BUILD REGULAR OHRC GEOMETRY MASK
# ============================================================

unique_scans = np.sort(
    np.unique(
        ohrc_scan
    )
)


unique_pixels = np.sort(
    np.unique(
        ohrc_pixel
    )
)


print(
    "\nOHRC geometry grid:"
)

print(
    "  Scan samples:",
    len(
        unique_scans
    )
)


print(
    "  Pixel samples:",
    len(
        unique_pixels
    )
)


grid_mask = np.zeros(

    (
        len(
            unique_scans
        ),

        len(
            unique_pixels
        )
    ),

    dtype=np.uint8
)


scan_indices = np.searchsorted(

    unique_scans,

    ohrc_scan
)


pixel_indices = np.searchsorted(

    unique_pixels,

    ohrc_pixel
)


grid_mask[

    scan_indices[
        exact_indices
    ],

    pixel_indices[
        exact_indices
    ]

] = 255


# ============================================================
# CONNECTED COMPONENT ANALYSIS
#
# This finds the largest contiguous overlapping part of the
# OHRC strip. That is what we want for Pair 002 extraction.
# ============================================================

binary_mask = (
    grid_mask > 0
).astype(
    np.uint8
)


(
    component_count,
    labels,
    stats,
    centroids

) = cv2.connectedComponentsWithStats(

    binary_mask,

    connectivity=8
)


print("\n")
print("=" * 80)
print("CONNECTED OVERLAP COMPONENTS")
print("=" * 80)


print(
    "Components found:",
    max(
        0,
        component_count - 1
    )
)


largest_component_label = None
largest_component_area = 0


if component_count > 1:

    component_areas = (

        stats[
            1:,
            cv2.CC_STAT_AREA
        ]
    )


    largest_component_label = (

        int(
            np.argmax(
                component_areas
            )
        )

        + 1
    )


    largest_component_area = int(

        stats[
            largest_component_label,
            cv2.CC_STAT_AREA
        ]
    )


    print(
        "Largest component geometry samples:",
        largest_component_area
    )


# ============================================================
# MAP LARGEST COMPONENT BACK TO OHRC GEOMETRY POINTS
# ============================================================

largest_component_points = np.zeros(

    len(
        geometry
    ),

    dtype=bool
)


if largest_component_label is not None:

    point_labels = labels[

        scan_indices,

        pixel_indices
    ]


    largest_component_points = (

        point_labels
        == largest_component_label
    )


    component_indices = np.flatnonzero(

        largest_component_points
        & exact_overlap
    )


else:

    component_indices = np.array(
        [],
        dtype=np.int64
    )


# ============================================================
# REPORT COMPONENT RANGE
# ============================================================

if len(
    component_indices
) > 0:

    component_scan_min = int(
        ohrc_scan[
            component_indices
        ].min()
    )


    component_scan_max = int(
        ohrc_scan[
            component_indices
        ].max()
    )


    component_pixel_min = int(
        ohrc_pixel[
            component_indices
        ].min()
    )


    component_pixel_max = int(
        ohrc_pixel[
            component_indices
        ].max()
    )


    component_lat_min = float(
        latitude[
            component_indices
        ].min()
    )


    component_lat_max = float(
        latitude[
            component_indices
        ].max()
    )


    component_lon_min = float(
        longitude_360[
            component_indices
        ].min()
    )


    component_lon_max = float(
        longitude_360[
            component_indices
        ].max()
    )


    print("\n")
    print("=" * 80)
    print("LARGEST CONTIGUOUS OVERLAP")
    print("=" * 80)


    print(
        "OHRC scan range:",
        component_scan_min,
        "to",
        component_scan_max
    )


    print(
        "OHRC pixel range:",
        component_pixel_min,
        "to",
        component_pixel_max
    )


    print(
        "Latitude range:",
        component_lat_min,
        "to",
        component_lat_max
    )


    print(
        "Longitude [0,360) range:",
        component_lon_min,
        "to",
        component_lon_max
    )


else:

    component_scan_min = None
    component_scan_max = None

    component_pixel_min = None
    component_pixel_max = None

    component_lat_min = None
    component_lat_max = None

    component_lon_min = None
    component_lon_max = None


# ============================================================
# SAVE FULL POINT TABLE
# ============================================================

output_table = (
    geometry.copy()
)


output_table[
    "Longitude180"
] = longitude_180


output_table[
    "TMC_X"
] = projected_x


output_table[
    "TMC_Y"
] = projected_y


output_table[
    "TMC_Row"
] = rows


output_table[
    "TMC_Col"
] = cols


output_table[
    "Inside_TMC"
] = inside


output_table[
    "TMC_Value"
] = tmc_values


output_table[
    "Exact_Valid_Overlap"
] = exact_overlap


output_table[
    "Largest_Component"
] = largest_component_points


output_table.to_csv(

    CSV_OUTPUT,

    index=False
)


# ============================================================
# SAVE EXACT OVERLAP MASK
# ============================================================

scale = 5


mask_preview = cv2.resize(

    grid_mask,

    None,

    fx=scale,
    fy=scale,

    interpolation=
        cv2.INTER_NEAREST
)


cv2.imwrite(

    str(
        MASK_OUTPUT
    ),

    mask_preview
)


# ============================================================
# SAVE LARGEST COMPONENT MASK
# ============================================================

component_mask = np.zeros_like(
    grid_mask
)


if largest_component_label is not None:

    component_mask[
        labels
        == largest_component_label
    ] = 255


component_preview = cv2.resize(

    component_mask,

    None,

    fx=scale,
    fy=scale,

    interpolation=
        cv2.INTER_NEAREST
)


cv2.imwrite(

    str(
        LARGEST_COMPONENT_OUTPUT
    ),

    component_preview
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "ohrc_product":
        "ch2_ohr_ncp_20211228T2209123959",

    "tmc_product":
        "ch2_tmc_ndn_20231025T1956513800_d_oth_d18",

    "ohrc_geometry_points":
        int(
            len(
                geometry
            )
        ),

    "exact_overlap_points":
        exact_count,

    "exact_overlap_ratio":
        exact_ratio,

    "exact_overlap_percent":
        exact_ratio * 100.0,

    "connected_components":
        int(
            max(
                0,
                component_count - 1
            )
        ),

    "largest_component_points":
        int(
            largest_component_area
        ),

    "largest_component": {

        "scan_min":
            component_scan_min,

        "scan_max":
            component_scan_max,

        "pixel_min":
            component_pixel_min,

        "pixel_max":
            component_pixel_max,

        "latitude_min":
            component_lat_min,

        "latitude_max":
            component_lat_max,

        "longitude_360_min":
            component_lon_min,

        "longitude_360_max":
            component_lon_max
    }
}


with open(

    SUMMARY_OUTPUT,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        summary,

        file,

        indent=4
    )


# ============================================================
# FINAL DECISION
# ============================================================

print("\n")
print("=" * 80)


if exact_count > 0:

    print(
        "PAIR 002 EXACT OVERLAP: CONFIRMED"
    )

    print(
        "OHRC and TMC-2 share native valid imagery."
    )


else:

    print(
        "PAIR 002 EXACT OVERLAP: NOT CONFIRMED"
    )


print("=" * 80)


print(
    "\nCSV:"
)

print(
    CSV_OUTPUT
)


print(
    "\nExact overlap mask:"
)

print(
    MASK_OUTPUT
)


print(
    "\nLargest component mask:"
)

print(
    LARGEST_COMPONENT_OUTPUT
)


print(
    "\nSummary:"
)

print(
    SUMMARY_OUTPUT
)