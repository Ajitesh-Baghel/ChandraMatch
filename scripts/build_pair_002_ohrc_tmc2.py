import sys
import json
import math
import xml.etree.ElementTree as ET

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio

from rasterio.windows import Window
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(ROOT)
)


# ============================================================
# INPUT PATHS
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


EXACT_OVERLAP_CSV = (
    ROOT
    / "results"
    / "pair_002"
    / "exact_overlap"
    / "exact_overlap_points.csv"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_002"
)


RESULT_DIR = (
    ROOT
    / "results"
    / "pair_002"
    / "build"
)


PAIR_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SOURCE_TIF_OUTPUT = (
    PAIR_DIR
    / "source_ohrc_on_tmc_grid.tif"
)


REFERENCE_TIF_OUTPUT = (
    PAIR_DIR
    / "reference_tmc2.tif"
)


VALID_MASK_TIF_OUTPUT = (
    PAIR_DIR
    / "valid_mask.tif"
)


SOURCE_PNG_OUTPUT = (
    PAIR_DIR
    / "source.png"
)


REFERENCE_PNG_OUTPUT = (
    PAIR_DIR
    / "reference.png"
)


VALID_MASK_PNG_OUTPUT = (
    PAIR_DIR
    / "valid_mask.png"
)


PAIR_COMPARISON_OUTPUT = (
    PAIR_DIR
    / "pair_comparison.png"
)


OVERLAY_OUTPUT = (
    PAIR_DIR
    / "overlay_50_50.png"
)


DIFFERENCE_OUTPUT = (
    PAIR_DIR
    / "difference.png"
)


NATIVE_OHRC_PREVIEW_OUTPUT = (
    RESULT_DIR
    / "ohrc_native_overlap_preview.png"
)


RAW_TMC_MASK_OUTPUT = (
    RESULT_DIR
    / "tmc_raw_nonzero_mask.png"
)


CLEAN_TMC_MASK_OUTPUT = (
    RESULT_DIR
    / "tmc_clean_valid_mask.png"
)


METADATA_OUTPUT = (
    PAIR_DIR
    / "pair_metadata.json"
)


# ============================================================
# SETTINGS
# ============================================================

# Padding around TMC geometry component.
TMC_PADDING_PIXELS = 20


# Padding around native OHRC crop.
OHRC_PADDING_PIXELS = 20


# Extra OHRC geometry samples used to construct the inverse
# geolocation interpolation.
GEOMETRY_PADDING_NATIVE_PIXELS = 300


# Trim empty borders around final common support.
FINAL_MASK_PADDING_PIXELS = 10


# Maximum internal zero component treated as a small hole.
#
# At TMC resolution:
#
# 1 pixel = 5 m × 5 m
#
# 100 pixels ≈ 2500 m².
#
# Larger dark regions remain untouched.
MAX_HOLE_AREA = 100


# Small morphological closing kernel.
MASK_CLOSE_KERNEL_SIZE = 5


# ============================================================
# HELPERS
# ============================================================

def local_name(tag):
    """
    Remove an XML namespace from a tag.
    """

    return tag.split("}")[-1]


def find_xml_value(root, names):
    """
    Return the first XML value matching any local tag name.
    """

    names = set(names)

    for element in root.iter():

        if local_name(element.tag) in names:

            if element.text is not None:

                text = element.text.strip()

                if text:
                    return text

    return None


def find_all_axis_dimensions(root):
    """
    Read PDS4 Axis_Array dimensions.
    """

    dimensions = {}

    for element in root.iter():

        if local_name(element.tag) != "Axis_Array":
            continue

        axis_name = None
        axis_elements = None

        for child in element:

            name = local_name(child.tag)

            if name == "axis_name":

                axis_name = (
                    child.text.strip()
                    if child.text
                    else None
                )

            elif name == "elements":

                axis_elements = (
                    int(child.text.strip())
                    if child.text
                    else None
                )

        if (
            axis_name is not None
            and
            axis_elements is not None
        ):

            dimensions[
                axis_name.lower()
            ] = axis_elements

    return dimensions


def pds_dtype_to_numpy(data_type):
    """
    Minimal PDS4 datatype conversion.
    """

    mapping = {

        "UnsignedByte":
            np.dtype("uint8"),

        "UnsignedLSB2":
            np.dtype("<u2"),

        "UnsignedMSB2":
            np.dtype(">u2"),

        "SignedByte":
            np.dtype("int8"),

        "SignedLSB2":
            np.dtype("<i2"),

        "SignedMSB2":
            np.dtype(">i2")
    }


    if data_type not in mapping:

        raise RuntimeError(
            "Unsupported PDS datatype: "
            f"{data_type}"
        )


    return mapping[
        data_type
    ]


def normalize_uint8(image, mask):
    """
    Robust 1st-99th percentile normalization.
    """

    output = np.zeros(
        image.shape,
        dtype=np.uint8
    )


    values = image[
        mask
    ]


    if values.size == 0:
        return output


    low, high = np.percentile(
        values,
        [1, 99]
    )


    if high <= low:
        return output


    work = np.clip(

        image.astype(
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


    output[
        mask
    ] = (

        work[
            mask
        ]
        .astype(
            np.uint8
        )
    )


    return output


def circular_angle_difference(a, b):
    """
    Smallest circular angular difference in degrees.
    """

    if (
        a is None
        or
        b is None
    ):

        return None


    difference = abs(
        a - b
    ) % 360.0


    return min(
        difference,
        360.0 - difference
    )


def parse_float(value):

    if value is None:
        return None

    try:

        return float(
            value
        )

    except Exception:

        return None


def column_to_bool(series):
    """
    Robust conversion of CSV boolean columns.
    """

    if series.dtype == bool:
        return series


    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
    )


def clean_tmc_valid_mask(tmc_image):
    """
    Build a cleaner TMC swath-support mask.

    Raw TMC products use zero extensively around the acquired
    swath, but small isolated zero regions can also appear
    inside otherwise valid terrain.

    Strategy:

    1. Start from image > 0.
    2. Apply a small morphological close.
    3. Find zero connected components.
    4. Preserve all components touching the image boundary.
    5. Preserve large internal zero regions.
    6. Fill only small enclosed holes.

    This prevents the common mask from developing thousands
    of tiny artificial holes while preserving genuine swath
    gaps and exterior fill.
    """

    raw_mask = (
        tmc_image > 0
    ).astype(
        np.uint8
    )


    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (
            MASK_CLOSE_KERNEL_SIZE,
            MASK_CLOSE_KERNEL_SIZE
        )
    )


    cleaned = cv2.morphologyEx(

        raw_mask,

        cv2.MORPH_CLOSE,

        kernel,

        iterations=1
    )


    inverse = (
        1 - cleaned
    ).astype(
        np.uint8
    )


    (
        component_count,
        labels,
        stats,
        _

    ) = cv2.connectedComponentsWithStats(

        inverse,

        connectivity=8
    )


    holes_filled = 0
    pixels_filled = 0


    for label in range(
        1,
        component_count
    ):

        area = int(

            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )


        left = int(

            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )


        top = int(

            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )


        width = int(

            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )


        height = int(

            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )


        touches_border = (

            left <= 0

            or

            top <= 0

            or

            left + width
            >= inverse.shape[1]

            or

            top + height
            >= inverse.shape[0]
        )


        if (

            not touches_border

            and

            area <= MAX_HOLE_AREA
        ):

            cleaned[
                labels == label
            ] = 1


            holes_filled += 1
            pixels_filled += area


    return (
        raw_mask.astype(bool),
        cleaned.astype(bool),
        holes_filled,
        pixels_filled
    )


# ============================================================
# FIND OHRC IMAGE + LABEL
# ============================================================

ohrc_img_files = sorted(

    (
        OHRC_ROOT
        / "data"
        / "calibrated"
    ).rglob(
        "*.img"
    )
)


if not ohrc_img_files:

    raise RuntimeError(
        "No OHRC calibrated IMG found."
    )


OHRC_IMG = (
    ohrc_img_files[0]
)


OHRC_XML = (
    OHRC_IMG.with_suffix(
        ".xml"
    )
)


if not OHRC_XML.exists():

    raise RuntimeError(
        f"OHRC XML not found:\n"
        f"{OHRC_XML}"
    )


# ============================================================
# FIND TMC TIFF + LABEL
# ============================================================

tmc_files = sorted(
    TMC_ROOT.rglob(
        "*.tif"
    )
)


if not tmc_files:

    raise RuntimeError(
        "No TMC TIFF found."
    )


TMC_TIF = (
    tmc_files[0]
)


TMC_XML = (
    TMC_TIF.with_suffix(
        ".xml"
    )
)


# ============================================================
# CHECK EXACT OVERLAP TABLE
# ============================================================

if not EXACT_OVERLAP_CSV.exists():

    raise RuntimeError(
        "Exact overlap CSV does not exist:\n"
        f"{EXACT_OVERLAP_CSV}\n\n"
        "Run confirm_pair002_exact_overlap.py first."
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)

print(
    "CHANDRAMATCH - BUILD PAIR 002"
)

print(
    "OHRC ↔ TMC-2"
)

print("=" * 80)


print(
    "\nOHRC image:"
)

print(
    OHRC_IMG
)


print(
    "\nTMC image:"
)

print(
    TMC_TIF
)


print(
    "\nExact overlap table:"
)

print(
    EXACT_OVERLAP_CSV
)


# ============================================================
# READ OHRC LABEL
# ============================================================

ohrc_tree = ET.parse(
    OHRC_XML
)


ohrc_root_xml = (
    ohrc_tree.getroot()
)


ohrc_dimensions = (
    find_all_axis_dimensions(
        ohrc_root_xml
    )
)


OHRC_HEIGHT = (
    ohrc_dimensions.get(
        "line"
    )
)


OHRC_WIDTH = (
    ohrc_dimensions.get(
        "sample"
    )
)


if (
    OHRC_HEIGHT is None
    or
    OHRC_WIDTH is None
):

    raise RuntimeError(
        "Could not read OHRC dimensions "
        "from PDS XML."
    )


ohrc_data_type_text = (
    find_xml_value(
        ohrc_root_xml,
        ["data_type"]
    )
)


if ohrc_data_type_text is None:

    raise RuntimeError(
        "Could not read OHRC datatype."
    )


OHRC_DTYPE = (
    pds_dtype_to_numpy(
        ohrc_data_type_text
    )
)


ohrc_offset_text = (
    find_xml_value(
        ohrc_root_xml,
        ["offset"]
    )
)


OHRC_OFFSET = (

    int(
        ohrc_offset_text
    )

    if ohrc_offset_text

    else 0
)


OHRC_RESOLUTION = parse_float(

    find_xml_value(
        ohrc_root_xml,
        [
            "pixel_resolution",
            "ground_sampling_distance"
        ]
    )
)


OHRC_SUN_AZIMUTH = parse_float(

    find_xml_value(
        ohrc_root_xml,
        ["sun_azimuth"]
    )
)


OHRC_SUN_ELEVATION = parse_float(

    find_xml_value(
        ohrc_root_xml,
        ["sun_elevation"]
    )
)


OHRC_START_TIME = (
    find_xml_value(
        ohrc_root_xml,
        ["start_date_time"]
    )
)


print("\n")
print("=" * 80)
print("OHRC PRODUCT")
print("=" * 80)


print(
    "Shape:",
    OHRC_HEIGHT,
    "x",
    OHRC_WIDTH
)


print(
    "Datatype:",
    OHRC_DTYPE
)


print(
    "Offset:",
    OHRC_OFFSET
)


print(
    "Pixel resolution:",
    OHRC_RESOLUTION
)


print(
    "Sun azimuth:",
    OHRC_SUN_AZIMUTH
)


print(
    "Sun elevation:",
    OHRC_SUN_ELEVATION
)


# ============================================================
# VALIDATE OHRC BINARY SIZE
# ============================================================

expected_ohrc_bytes = (

    OHRC_OFFSET

    +

    OHRC_HEIGHT
    * OHRC_WIDTH
    * OHRC_DTYPE.itemsize
)


actual_ohrc_bytes = (
    OHRC_IMG.stat().st_size
)


print(
    "Expected bytes:",
    expected_ohrc_bytes
)


print(
    "Actual bytes:",
    actual_ohrc_bytes
)


if actual_ohrc_bytes < expected_ohrc_bytes:

    raise RuntimeError(
        "OHRC IMG is smaller than expected."
    )


# ============================================================
# OPEN OHRC MEMORY MAP
# ============================================================

ohrc_memmap = np.memmap(

    OHRC_IMG,

    dtype=OHRC_DTYPE,

    mode="r",

    offset=OHRC_OFFSET,

    shape=(
        OHRC_HEIGHT,
        OHRC_WIDTH
    )
)


print(
    "OHRC memory map opened."
)


# ============================================================
# READ EXACT OVERLAP TABLE
# ============================================================

overlap_table = pd.read_csv(
    EXACT_OVERLAP_CSV
)


required_columns = {

    "Longitude",
    "Latitude",
    "Pixel",
    "Scan",

    "TMC_X",
    "TMC_Y",
    "TMC_Row",
    "TMC_Col",

    "Exact_Valid_Overlap",
    "Largest_Component"
}


missing = (

    required_columns

    - set(
        overlap_table.columns
    )
)


if missing:

    raise RuntimeError(
        "Missing overlap CSV columns: "
        f"{missing}"
    )


overlap_table[
    "Exact_Valid_Overlap"
] = column_to_bool(

    overlap_table[
        "Exact_Valid_Overlap"
    ]
)


overlap_table[
    "Largest_Component"
] = column_to_bool(

    overlap_table[
        "Largest_Component"
    ]
)


component = overlap_table[

    overlap_table[
        "Largest_Component"
    ]

    &

    overlap_table[
        "Exact_Valid_Overlap"
    ]

].copy()


if len(
    component
) < 20:

    raise RuntimeError(
        "Largest overlap component has too "
        "few geometry control points."
    )


print("\n")
print("=" * 80)
print("OVERLAP COMPONENT")
print("=" * 80)


print(
    "Geometry samples:",
    len(
        component
    )
)


component_scan_min = int(

    component[
        "Scan"
    ].min()
)


component_scan_max = int(

    component[
        "Scan"
    ].max()
)


component_pixel_min = int(

    component[
        "Pixel"
    ].min()
)


component_pixel_max = int(

    component[
        "Pixel"
    ].max()
)


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


# ============================================================
# GEOMETRY POINTS USED FOR INTERPOLATION
# ============================================================

mapping_scan_min = max(

    0,

    component_scan_min
    - GEOMETRY_PADDING_NATIVE_PIXELS
)


mapping_scan_max = min(

    OHRC_HEIGHT - 1,

    component_scan_max
    + GEOMETRY_PADDING_NATIVE_PIXELS
)


mapping_pixel_min = max(

    0,

    component_pixel_min
    - GEOMETRY_PADDING_NATIVE_PIXELS
)


mapping_pixel_max = min(

    OHRC_WIDTH - 1,

    component_pixel_max
    + GEOMETRY_PADDING_NATIVE_PIXELS
)


mapping_points = overlap_table[

    (
        overlap_table[
            "Scan"
        ]
        >= mapping_scan_min
    )

    &

    (
        overlap_table[
            "Scan"
        ]
        <= mapping_scan_max
    )

    &

    (
        overlap_table[
            "Pixel"
        ]
        >= mapping_pixel_min
    )

    &

    (
        overlap_table[
            "Pixel"
        ]
        <= mapping_pixel_max
    )

].copy()


print(
    "\nGeometry interpolation samples:",
    len(
        mapping_points
    )
)


if len(
    mapping_points
) < 20:

    raise RuntimeError(
        "Too few geometry interpolation points."
    )


# ============================================================
# OPEN TMC PRODUCT
# ============================================================

with rasterio.open(
    TMC_TIF
) as tmc:

    print("\n")
    print("=" * 80)
    print("TMC PRODUCT")
    print("=" * 80)


    print(
        "Shape:",
        tmc.height,
        "x",
        tmc.width
    )


    print(
        "Resolution:",
        tmc.res
    )


    print(
        "CRS:"
    )

    print(
        tmc.crs
    )


    TMC_RESOLUTION = abs(
        float(
            tmc.res[0]
        )
    )


    TMC_SUN_AZIMUTH = None
    TMC_SUN_ELEVATION = None
    TMC_START_TIME = None


    if TMC_XML.exists():

        tmc_xml_tree = ET.parse(
            TMC_XML
        )


        tmc_xml_root = (
            tmc_xml_tree.getroot()
        )


        TMC_SUN_AZIMUTH = parse_float(

            find_xml_value(
                tmc_xml_root,
                ["sun_azimuth"]
            )
        )


        TMC_SUN_ELEVATION = parse_float(

            find_xml_value(
                tmc_xml_root,
                ["sun_elevation"]
            )
        )


        TMC_START_TIME = (
            find_xml_value(
                tmc_xml_root,
                ["start_date_time"]
            )
        )


    print(
        "Sun azimuth:",
        TMC_SUN_AZIMUTH
    )


    print(
        "Sun elevation:",
        TMC_SUN_ELEVATION
    )


    # ========================================================
    # DETERMINE TMC WINDOW
    # ========================================================

    component_rows = (
        component[
            "TMC_Row"
        ].to_numpy(
            dtype=np.int64
        )
    )


    component_cols = (
        component[
            "TMC_Col"
        ].to_numpy(
            dtype=np.int64
        )
    )


    tmc_row_min = max(

        0,

        int(
            component_rows.min()
        )
        - TMC_PADDING_PIXELS
    )


    tmc_row_max = min(

        tmc.height - 1,

        int(
            component_rows.max()
        )
        + TMC_PADDING_PIXELS
    )


    tmc_col_min = max(

        0,

        int(
            component_cols.min()
        )
        - TMC_PADDING_PIXELS
    )


    tmc_col_max = min(

        tmc.width - 1,

        int(
            component_cols.max()
        )
        + TMC_PADDING_PIXELS
    )


    tmc_window_width = (

        tmc_col_max
        - tmc_col_min
        + 1
    )


    tmc_window_height = (

        tmc_row_max
        - tmc_row_min
        + 1
    )


    print("\n")
    print("=" * 80)
    print("PAIR 002 TMC WINDOW")
    print("=" * 80)


    print(
        "Rows:",
        tmc_row_min,
        "to",
        tmc_row_max
    )


    print(
        "Columns:",
        tmc_col_min,
        "to",
        tmc_col_max
    )


    print(
        "Shape:",
        tmc_window_height,
        "x",
        tmc_window_width
    )


    tmc_window = Window(

        tmc_col_min,
        tmc_row_min,

        tmc_window_width,
        tmc_window_height
    )


    tmc_crop = tmc.read(

        1,

        window=tmc_window
    )


    tmc_window_transform = (
        tmc.window_transform(
            tmc_window
        )
    )


    # ========================================================
    # OUTPUT TMC GRID COORDINATES
    # ========================================================

    local_cols = (

        np.arange(
            tmc_window_width,
            dtype=np.float64
        )

        + 0.5
    )


    local_rows = (

        np.arange(
            tmc_window_height,
            dtype=np.float64
        )

        + 0.5
    )


    col_grid, row_grid = np.meshgrid(

        local_cols,
        local_rows
    )


    x_grid = (

        tmc_window_transform.c

        +

        tmc_window_transform.a
        * col_grid

        +

        tmc_window_transform.b
        * row_grid
    )


    y_grid = (

        tmc_window_transform.f

        +

        tmc_window_transform.d
        * col_grid

        +

        tmc_window_transform.e
        * row_grid
    )


    # ========================================================
    # INVERSE OHRC GEOLOCATION
    #
    # TMC projected X/Y
    #      ↓
    # OHRC native pixel / scan
    # ========================================================

    control_xy = np.column_stack(
        (

            mapping_points[
                "TMC_X"
            ].to_numpy(
                dtype=np.float64
            ),

            mapping_points[
                "TMC_Y"
            ].to_numpy(
                dtype=np.float64
            )
        )
    )


    control_pixel = (
        mapping_points[
            "Pixel"
        ].to_numpy(
            dtype=np.float64
        )
    )


    control_scan = (
        mapping_points[
            "Scan"
        ].to_numpy(
            dtype=np.float64
        )
    )


    # Remove duplicate projected coordinates.

    _, unique_indices = np.unique(

        np.round(
            control_xy,
            decimals=6
        ),

        axis=0,

        return_index=True
    )


    control_xy = (
        control_xy[
            unique_indices
        ]
    )


    control_pixel = (
        control_pixel[
            unique_indices
        ]
    )


    control_scan = (
        control_scan[
            unique_indices
        ]
    )


    print(
        "\nUnique geometry controls:",
        len(
            control_xy
        )
    )


    print(
        "Building Delaunay triangulation..."
    )


    triangulation = Delaunay(

        control_xy,

        qhull_options="QJ"
    )


    print(
        "Building inverse OHRC geolocation interpolators..."
    )


    pixel_interpolator = (
        LinearNDInterpolator(

            triangulation,

            control_pixel,

            fill_value=np.nan
        )
    )


    scan_interpolator = (
        LinearNDInterpolator(

            triangulation,

            control_scan,

            fill_value=np.nan
        )
    )


    print(
        "Interpolating output grid..."
    )


    mapped_pixel = pixel_interpolator(

        x_grid,
        y_grid
    )


    mapped_scan = scan_interpolator(

        x_grid,
        y_grid
    )


    mapped_pixel = np.asarray(
        mapped_pixel,
        dtype=np.float64
    )


    mapped_scan = np.asarray(
        mapped_scan,
        dtype=np.float64
    )


    geometry_valid = (

        np.isfinite(
            mapped_pixel
        )

        &

        np.isfinite(
            mapped_scan
        )

        &

        (
            mapped_pixel >= 0
        )

        &

        (
            mapped_pixel
            <= OHRC_WIDTH - 1
        )

        &

        (
            mapped_scan >= 0
        )

        &

        (
            mapped_scan
            <= OHRC_HEIGHT - 1
        )
    )


    print(
        "Geolocatable TMC-grid pixels:",
        int(
            np.count_nonzero(
                geometry_valid
            )
        )
    )


    # ========================================================
    # DETERMINE NATIVE OHRC CROP
    # ========================================================

    valid_pixels = (
        mapped_pixel[
            geometry_valid
        ]
    )


    valid_scans = (
        mapped_scan[
            geometry_valid
        ]
    )


    if valid_pixels.size == 0:

        raise RuntimeError(
            "No output pixels could be mapped "
            "into OHRC coordinates."
        )


    ohrc_crop_col_min = max(

        0,

        int(
            math.floor(
                valid_pixels.min()
            )
        )
        - OHRC_PADDING_PIXELS
    )


    ohrc_crop_col_max = min(

        OHRC_WIDTH - 1,

        int(
            math.ceil(
                valid_pixels.max()
            )
        )
        + OHRC_PADDING_PIXELS
    )


    ohrc_crop_row_min = max(

        0,

        int(
            math.floor(
                valid_scans.min()
            )
        )
        - OHRC_PADDING_PIXELS
    )


    ohrc_crop_row_max = min(

        OHRC_HEIGHT - 1,

        int(
            math.ceil(
                valid_scans.max()
            )
        )
        + OHRC_PADDING_PIXELS
    )


    print("\n")
    print("=" * 80)
    print("NATIVE OHRC CROP")
    print("=" * 80)


    print(
        "Rows / scans:",
        ohrc_crop_row_min,
        "to",
        ohrc_crop_row_max
    )


    print(
        "Columns / pixels:",
        ohrc_crop_col_min,
        "to",
        ohrc_crop_col_max
    )


    print(
        "Shape:",
        (
            ohrc_crop_row_max
            - ohrc_crop_row_min
            + 1
        ),
        "x",
        (
            ohrc_crop_col_max
            - ohrc_crop_col_min
            + 1
        )
    )


    # ========================================================
    # READ REQUIRED OHRC CROP
    # ========================================================

    ohrc_native_crop = np.asarray(

        ohrc_memmap[

            ohrc_crop_row_min:
            ohrc_crop_row_max + 1,

            ohrc_crop_col_min:
            ohrc_crop_col_max + 1
        ]
    )


    # ========================================================
    # SAVE NATIVE OHRC PREVIEW
    # ========================================================

    native_preview_scale = min(

        1.0,

        1600.0
        /
        max(
            ohrc_native_crop.shape
        )
    )


    native_preview = cv2.resize(

        ohrc_native_crop,

        (

            max(
                1,
                int(
                    round(
                        ohrc_native_crop.shape[1]
                        * native_preview_scale
                    )
                )
            ),

            max(
                1,
                int(
                    round(
                        ohrc_native_crop.shape[0]
                        * native_preview_scale
                    )
                )
            )
        ),

        interpolation=
            cv2.INTER_AREA
    )


    cv2.imwrite(

        str(
            NATIVE_OHRC_PREVIEW_OUTPUT
        ),

        native_preview
    )


    # ========================================================
    # CV2 REMAP TABLES
    # ========================================================

    map_x = (

        mapped_pixel
        - ohrc_crop_col_min
    ).astype(
        np.float32
    )


    map_y = (

        mapped_scan
        - ohrc_crop_row_min
    ).astype(
        np.float32
    )


    map_x[
        ~geometry_valid
    ] = -1.0


    map_y[
        ~geometry_valid
    ] = -1.0


    # ========================================================
    # RESAMPLE OHRC ONTO TMC 5-M GRID
    # ========================================================

    print(
        "\nResampling OHRC onto TMC grid..."
    )


    ohrc_warped = cv2.remap(

        ohrc_native_crop,

        map_x,

        map_y,

        interpolation=
            cv2.INTER_LINEAR,

        borderMode=
            cv2.BORDER_CONSTANT,

        borderValue=0
    )


    print(
        "OHRC reprojection complete."
    )


    # ========================================================
    # CLEAN TMC VALIDITY MASK
    # ========================================================

    print("\n")
    print("=" * 80)
    print("TMC VALIDITY MASK CLEANING")
    print("=" * 80)


    (
        tmc_raw_valid,
        tmc_valid,
        holes_filled,
        pixels_filled

    ) = clean_tmc_valid_mask(
        tmc_crop
    )


    raw_valid_count = int(
        np.count_nonzero(
            tmc_raw_valid
        )
    )


    clean_valid_count = int(
        np.count_nonzero(
            tmc_valid
        )
    )


    raw_valid_ratio = float(

        raw_valid_count

        /

        tmc_raw_valid.size
    )


    clean_valid_ratio = float(

        clean_valid_count

        /

        tmc_valid.size
    )


    print(
        "Raw TMC non-zero pixels:",
        raw_valid_count
    )


    print(
        "Raw TMC non-zero ratio:",
        raw_valid_ratio
    )


    print(
        "Clean TMC valid pixels:",
        clean_valid_count
    )


    print(
        "Clean TMC valid ratio:",
        clean_valid_ratio
    )


    print(
        "Small holes filled:",
        holes_filled
    )


    print(
        "Zero pixels restored as support:",
        pixels_filled
    )


    # ========================================================
    # SAVE RAW / CLEAN TMC MASK DEBUG IMAGES
    # ========================================================

    cv2.imwrite(

        str(
            RAW_TMC_MASK_OUTPUT
        ),

        (
            tmc_raw_valid.astype(
                np.uint8
            )
            * 255
        )
    )


    cv2.imwrite(

        str(
            CLEAN_TMC_MASK_OUTPUT
        ),

        (
            tmc_valid.astype(
                np.uint8
            )
            * 255
        )
    )


    # ========================================================
    # COMMON VALID SUPPORT
    # ========================================================

    common_valid = (

        geometry_valid

        &

        tmc_valid
    )


    common_valid_count = int(
        np.count_nonzero(
            common_valid
        )
    )


    common_valid_ratio = float(

        common_valid_count

        /

        common_valid.size
    )


    print("\n")
    print("=" * 80)
    print("INITIAL COMMON GRID")
    print("=" * 80)


    print(
        "Shape:",
        common_valid.shape
    )


    print(
        "Common valid pixels:",
        common_valid_count
    )


    print(
        "Common valid ratio:",
        common_valid_ratio
    )


    if common_valid_count == 0:

        raise RuntimeError(
            "No common valid pixels after "
            "OHRC reprojection."
        )


    # ========================================================
    # TRIM EMPTY BORDERS
    # ========================================================

    valid_rows, valid_cols = (
        np.nonzero(
            common_valid
        )
    )


    final_row_min = max(

        0,

        int(
            valid_rows.min()
        )
        - FINAL_MASK_PADDING_PIXELS
    )


    final_row_max = min(

        common_valid.shape[0] - 1,

        int(
            valid_rows.max()
        )
        + FINAL_MASK_PADDING_PIXELS
    )


    final_col_min = max(

        0,

        int(
            valid_cols.min()
        )
        - FINAL_MASK_PADDING_PIXELS
    )


    final_col_max = min(

        common_valid.shape[1] - 1,

        int(
            valid_cols.max()
        )
        + FINAL_MASK_PADDING_PIXELS
    )


    final_window = Window(

        final_col_min,
        final_row_min,

        final_col_max
        - final_col_min
        + 1,

        final_row_max
        - final_row_min
        + 1
    )


    final_transform = rasterio.windows.transform(

        final_window,

        tmc_window_transform
    )


    ohrc_final = ohrc_warped[

        final_row_min:
        final_row_max + 1,

        final_col_min:
        final_col_max + 1

    ].copy()


    tmc_final = tmc_crop[

        final_row_min:
        final_row_max + 1,

        final_col_min:
        final_col_max + 1

    ].copy()


    valid_final = common_valid[

        final_row_min:
        final_row_max + 1,

        final_col_min:
        final_col_max + 1

    ].copy()


    # ========================================================
    # MASK OUTSIDE TRUE COMMON SUPPORT
    # ========================================================

    ohrc_final[
        ~valid_final
    ] = 0


    tmc_final[
        ~valid_final
    ] = 0


    final_valid_count = int(
        np.count_nonzero(
            valid_final
        )
    )


    final_valid_ratio = float(

        final_valid_count

        /

        valid_final.size
    )


    print("\n")
    print("=" * 80)
    print("FINAL PAIR 002")
    print("=" * 80)


    print(
        "Shape:",
        ohrc_final.shape
    )


    print(
        "Valid pixels:",
        final_valid_count
    )


    print(
        "Valid ratio:",
        final_valid_ratio
    )


    # ========================================================
    # SAVE SOURCE GEO-TIFF
    # ========================================================

    source_profile = (
        tmc.profile.copy()
    )


    source_profile.update(

        driver="GTiff",

        height=
            ohrc_final.shape[0],

        width=
            ohrc_final.shape[1],

        count=1,

        dtype="uint8",

        transform=
            final_transform,

        compress="deflate",

        BIGTIFF="IF_SAFER"
    )


    with rasterio.open(

        SOURCE_TIF_OUTPUT,

        "w",

        **source_profile

    ) as dst:

        dst.write(

            ohrc_final.astype(
                np.uint8
            ),

            1
        )


    # ========================================================
    # SAVE REFERENCE GEO-TIFF
    # ========================================================

    reference_profile = (
        tmc.profile.copy()
    )


    reference_profile.update(

        driver="GTiff",

        height=
            tmc_final.shape[0],

        width=
            tmc_final.shape[1],

        count=1,

        transform=
            final_transform,

        compress="deflate",

        BIGTIFF="IF_SAFER"
    )


    with rasterio.open(

        REFERENCE_TIF_OUTPUT,

        "w",

        **reference_profile

    ) as dst:

        dst.write(
            tmc_final,
            1
        )


    # ========================================================
    # SAVE VALID MASK GEO-TIFF
    # ========================================================

    mask_profile = (
        tmc.profile.copy()
    )


    mask_profile.update(

        driver="GTiff",

        height=
            valid_final.shape[0],

        width=
            valid_final.shape[1],

        count=1,

        dtype="uint8",

        transform=
            final_transform,

        compress="deflate",

        BIGTIFF="IF_SAFER"
    )


    with rasterio.open(

        VALID_MASK_TIF_OUTPUT,

        "w",

        **mask_profile

    ) as dst:

        dst.write(

            (
                valid_final.astype(
                    np.uint8
                )
                * 255
            ),

            1
        )


    # ========================================================
    # NORMALIZED DISPLAY IMAGES
    # ========================================================

    source_display = normalize_uint8(

        ohrc_final,

        valid_final
    )


    reference_display = normalize_uint8(

        tmc_final,

        valid_final
    )


    valid_display = (

        valid_final.astype(
            np.uint8
        )

        * 255
    )


    cv2.imwrite(

        str(
            SOURCE_PNG_OUTPUT
        ),

        source_display
    )


    cv2.imwrite(

        str(
            REFERENCE_PNG_OUTPUT
        ),

        reference_display
    )


    cv2.imwrite(

        str(
            VALID_MASK_PNG_OUTPUT
        ),

        valid_display
    )


    # ========================================================
    # SIDE-BY-SIDE COMPARISON
    # ========================================================

    source_labelled = cv2.cvtColor(

        source_display,

        cv2.COLOR_GRAY2BGR
    )


    reference_labelled = cv2.cvtColor(

        reference_display,

        cv2.COLOR_GRAY2BGR
    )


    cv2.putText(

        source_labelled,

        "OHRC -> TMC grid",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        reference_labelled,

        "TMC-2 reference",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    comparison = np.hstack(
        (
            source_labelled,
            reference_labelled
        )
    )


    cv2.imwrite(

        str(
            PAIR_COMPARISON_OUTPUT
        ),

        comparison
    )


    # ========================================================
    # 50 / 50 OVERLAY
    # ========================================================

    overlay = cv2.addWeighted(

        source_display,
        0.5,

        reference_display,
        0.5,

        0
    )


    overlay[
        ~valid_final
    ] = 0


    cv2.imwrite(

        str(
            OVERLAY_OUTPUT
        ),

        overlay
    )


    # ========================================================
    # INTENSITY DIFFERENCE VISUALIZATION
    #
    # This is NOT an accuracy metric.
    # ========================================================

    difference = cv2.absdiff(

        source_display,

        reference_display
    )


    difference[
        ~valid_final
    ] = 0


    cv2.imwrite(

        str(
            DIFFERENCE_OUTPUT
        ),

        difference
    )


    # ========================================================
    # FINAL GLOBAL TMC PIXEL RANGE
    # ========================================================

    final_global_row_min = (

        tmc_row_min
        + final_row_min
    )


    final_global_row_max = (

        tmc_row_min
        + final_row_max
    )


    final_global_col_min = (

        tmc_col_min
        + final_col_min
    )


    final_global_col_max = (

        tmc_col_min
        + final_col_max
    )


    # ========================================================
    # FINAL MAP BOUNDS
    # ========================================================

    final_bounds = rasterio.transform.array_bounds(

        ohrc_final.shape[0],

        ohrc_final.shape[1],

        final_transform
    )


    # ========================================================
    # PAIR CHALLENGE METADATA
    # ========================================================

    scale_ratio = None


    if (
        OHRC_RESOLUTION is not None
        and
        OHRC_RESOLUTION > 0
    ):

        scale_ratio = (

            TMC_RESOLUTION

            /

            OHRC_RESOLUTION
        )


    sun_azimuth_difference = (
        circular_angle_difference(

            OHRC_SUN_AZIMUTH,

            TMC_SUN_AZIMUTH
        )
    )


    sun_elevation_difference = None


    if (
        OHRC_SUN_ELEVATION
        is not None

        and

        TMC_SUN_ELEVATION
        is not None
    ):

        sun_elevation_difference = abs(

            OHRC_SUN_ELEVATION

            -

            TMC_SUN_ELEVATION
        )


    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "pair_id":
            "pair_002",

        "purpose":
            (
                "Real cross-sensor, cross-scale, "
                "illumination-varying correspondence"
            ),

        "source": {

            "sensor":
                "OHRC",

            "product":
                OHRC_IMG.stem,

            "acquisition_time":
                OHRC_START_TIME,

            "native_shape": [

                int(
                    OHRC_HEIGHT
                ),

                int(
                    OHRC_WIDTH
                )
            ],

            "native_resolution_m":
                OHRC_RESOLUTION,

            "sun_azimuth_deg":
                OHRC_SUN_AZIMUTH,

            "sun_elevation_deg":
                OHRC_SUN_ELEVATION,

            "native_crop": {

                "scan_min":
                    int(
                        ohrc_crop_row_min
                    ),

                "scan_max":
                    int(
                        ohrc_crop_row_max
                    ),

                "pixel_min":
                    int(
                        ohrc_crop_col_min
                    ),

                "pixel_max":
                    int(
                        ohrc_crop_col_max
                    )
            }
        },


        "reference": {

            "sensor":
                "TMC-2",

            "product":
                TMC_TIF.stem,

            "acquisition_time":
                TMC_START_TIME,

            "native_resolution_m":
                TMC_RESOLUTION,

            "sun_azimuth_deg":
                TMC_SUN_AZIMUTH,

            "sun_elevation_deg":
                TMC_SUN_ELEVATION,

            "native_tmc_window": {

                "row_min":
                    int(
                        final_global_row_min
                    ),

                "row_max":
                    int(
                        final_global_row_max
                    ),

                "col_min":
                    int(
                        final_global_col_min
                    ),

                "col_max":
                    int(
                        final_global_col_max
                    )
            }
        },


        "challenge": {

            "nominal_scale_ratio":
                scale_ratio,

            "sun_azimuth_difference_deg":
                sun_azimuth_difference,

            "sun_elevation_difference_deg":
                sun_elevation_difference
        },


        "overlap": {

            "largest_component_geometry_points":
                int(
                    len(
                        component
                    )
                ),

            "component_scan_min":
                int(
                    component_scan_min
                ),

            "component_scan_max":
                int(
                    component_scan_max
                ),

            "component_pixel_min":
                int(
                    component_pixel_min
                ),

            "component_pixel_max":
                int(
                    component_pixel_max
                )
        },


        "tmc_validity_mask": {

            "raw_nonzero_pixels":
                int(
                    raw_valid_count
                ),

            "raw_nonzero_ratio":
                float(
                    raw_valid_ratio
                ),

            "clean_valid_pixels":
                int(
                    clean_valid_count
                ),

            "clean_valid_ratio":
                float(
                    clean_valid_ratio
                ),

            "morphological_close_kernel":
                int(
                    MASK_CLOSE_KERNEL_SIZE
                ),

            "maximum_internal_hole_area":
                int(
                    MAX_HOLE_AREA
                ),

            "small_holes_filled":
                int(
                    holes_filled
                ),

            "pixels_added_to_support":
                int(
                    pixels_filled
                )
        },


        "canonical_pair": {

            "grid":
                (
                    "TMC-2 native south-polar "
                    "stereographic grid"
                ),

            "resolution_m":
                TMC_RESOLUTION,

            "shape": [

                int(
                    ohrc_final.shape[0]
                ),

                int(
                    ohrc_final.shape[1]
                )
            ],

            "valid_pixels":
                int(
                    final_valid_count
                ),

            "valid_ratio":
                float(
                    final_valid_ratio
                ),

            "bounds": {

                "left":
                    float(
                        final_bounds[0]
                    ),

                "bottom":
                    float(
                        final_bounds[1]
                    ),

                "right":
                    float(
                        final_bounds[2]
                    ),

                "top":
                    float(
                        final_bounds[3]
                    )
            },

            "crs":
                str(
                    tmc.crs
                ),

            "resampling":
                (
                    "bilinear OHRC -> "
                    "TMC native 5 m grid"
                ),

            "geolocation_interpolation":
                (
                    "Delaunay + LinearNDInterpolator "
                    "using OHRC geometry samples"
                )
        }
    }


    with open(

        METADATA_OUTPUT,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metadata,

            file,

            indent=4
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("PAIR 002 BUILD COMPLETE")
print("=" * 80)


print(
    "\nSource OHRC-on-TMC-grid:"
)

print(
    SOURCE_TIF_OUTPUT
)


print(
    "\nReference TMC:"
)

print(
    REFERENCE_TIF_OUTPUT
)


print(
    "\nValid mask:"
)

print(
    VALID_MASK_TIF_OUTPUT
)


print(
    "\nRaw TMC mask debug image:"
)

print(
    RAW_TMC_MASK_OUTPUT
)


print(
    "\nClean TMC mask debug image:"
)

print(
    CLEAN_TMC_MASK_OUTPUT
)


print(
    "\nPair comparison:"
)

print(
    PAIR_COMPARISON_OUTPUT
)


print(
    "\nOverlay:"
)

print(
    OVERLAY_OUTPUT
)


print(
    "\nDifference visualization:"
)

print(
    DIFFERENCE_OUTPUT
)


print(
    "\nMetadata:"
)

print(
    METADATA_OUTPUT
)


print("\n")
print("=" * 80)

print(
    "PAIR 002 READY FOR VISUAL VALIDATION"
)

print("=" * 80)