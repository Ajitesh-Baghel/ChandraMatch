import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio

from rasterio.enums import Resampling
from pyproj import CRS as PyCRS
from pyproj import Transformer
from scipy.spatial import cKDTree


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
    / "full_overlap_check"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OVERLAY_OUTPUT = (
    OUTPUT_DIR
    / "ohrc_track_on_tmc.png"
)


CSV_OUTPUT = (
    OUTPUT_DIR
    / "ohrc_geometry_overlap.csv"
)


# ============================================================
# FIND TMC TIFF
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


# ============================================================
# FIND OHRC GEOMETRY CSV
# ============================================================

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
    "CHANDRAMATCH - FULL OHRC ↔ TMC-2 OVERLAP CHECK"
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


missing_columns = (
    required_columns
    - set(
        geometry.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "Missing OHRC geometry columns: "
        f"{missing_columns}"
    )


# ============================================================
# OHRC COORDINATES
# ============================================================

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
# CONVERT LONGITUDE
#
# OHRC:
#     0 ... 360
#
# Geographic convention used by pyproj:
#     -180 ... +180
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


print(
    "OHRC latitude range:",
    float(
        latitude.min()
    ),
    "to",
    float(
        latitude.max()
    )
)


print(
    "OHRC longitude [0,360) range:",
    float(
        longitude_360.min()
    ),
    "to",
    float(
        longitude_360.max()
    )
)


# ============================================================
# OPEN TMC PRODUCT
# ============================================================

with rasterio.open(
    TMC_TIF
) as src:

    print("\n")

    print("=" * 80)

    print(
        "TMC PRODUCT"
    )

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
            "TMC GeoTIFF does not have a CRS."
        )


    # ========================================================
    # BUILD LUNAR GEOGRAPHIC → TMC CRS TRANSFORMER
    # ========================================================

    tmc_crs = (
        PyCRS.from_wkt(
            src.crs.to_wkt()
        )
    )


    lunar_geographic_crs = (
        tmc_crs.geodetic_crs
    )


    transformer = (
        Transformer.from_crs(

            lunar_geographic_crs,

            tmc_crs,

            always_xy=True
        )
    )


    # ========================================================
    # PROJECT ALL OHRC GEOMETRY POINTS
    # ========================================================

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
    # CHECK FINITE COORDINATES
    # ========================================================

    finite = (

        np.isfinite(
            projected_x
        )

        &

        np.isfinite(
            projected_y
        )
    )


    # ========================================================
    # CHECK TMC RECTANGULAR BOUNDS
    # ========================================================

    inside_bbox = (

        finite

        &

        (
            projected_x
            >= src.bounds.left
        )

        &

        (
            projected_x
            <= src.bounds.right
        )

        &

        (
            projected_y
            >= src.bounds.bottom
        )

        &

        (
            projected_y
            <= src.bounds.top
        )
    )


    inside_bbox_count = int(
        np.count_nonzero(
            inside_bbox
        )
    )


    print(
        "\nOHRC points inside TMC rectangular bounds:",
        inside_bbox_count,
        "/",
        len(
            inside_bbox
        )
    )


    # ========================================================
    # CREATE COARSE TMC VALID-DATA REPRESENTATION
    #
    # Rasterio cannot use Resampling.max for direct reads.
    #
    # We therefore use average resampling into FLOAT32.
    # Float32 is important because a small amount of valid
    # imagery inside an otherwise-zero downsample region can
    # survive as a fractional positive value instead of being
    # rounded back to zero.
    # ========================================================

    MAX_PREVIEW_WIDTH = 4000


    preview_scale = min(

        1.0,

        MAX_PREVIEW_WIDTH
        / src.width
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


    print(
        "\nBuilding coarse valid-data mask:",
        preview_width,
        "x",
        preview_height
    )


    coarse = src.read(

        1,

        out_shape=(
            preview_height,
            preview_width
        ),

        resampling=
            Resampling.average,

        out_dtype=
            "float32"
    )


    # ========================================================
    # VALID DATA MASK
    #
    # TMC fill appears as zero in this product.
    # ========================================================

    valid_mask = (
        coarse > 0
    )


    coarse_valid_ratio = float(
        valid_mask.mean()
    )


    print(
        "Coarse valid ratio:",
        coarse_valid_ratio
    )


    # ========================================================
    # PROJECTED COORDINATES → FULL-RES TMC PIXELS
    # ========================================================

    inverse_transform = (
        ~src.transform
    )


    (
        full_cols,
        full_rows

    ) = inverse_transform * (

        projected_x,
        projected_y
    )


    full_cols = np.asarray(
        full_cols,
        dtype=np.float64
    )


    full_rows = np.asarray(
        full_rows,
        dtype=np.float64
    )


    # ========================================================
    # FULL-RES PIXELS → COARSE MASK PIXELS
    # ========================================================

    coarse_cols_float = (

        full_cols

        * preview_width

        / src.width
    )


    coarse_rows_float = (

        full_rows

        * preview_height

        / src.height
    )


    coarse_cols = np.floor(
        coarse_cols_float
    ).astype(
        np.int64
    )


    coarse_rows = np.floor(
        coarse_rows_float
    ).astype(
        np.int64
    )


    # ========================================================
    # CHECK WHICH OHRC POINTS MAP INTO COARSE IMAGE
    # ========================================================

    inside_preview = (

        inside_bbox

        &

        (
            coarse_cols
            >= 0
        )

        &

        (
            coarse_cols
            < preview_width
        )

        &

        (
            coarse_rows
            >= 0
        )

        &

        (
            coarse_rows
            < preview_height
        )
    )


    # ========================================================
    # CHECK COARSE VALID SWATH OVERLAP
    # ========================================================

    coarse_overlap = np.zeros(
        len(
            geometry
        ),
        dtype=bool
    )


    valid_geometry_indices = (
        np.flatnonzero(
            inside_preview
        )
    )


    coarse_overlap[
        valid_geometry_indices
    ] = valid_mask[

        coarse_rows[
            valid_geometry_indices
        ],

        coarse_cols[
            valid_geometry_indices
        ]
    ]


    overlap_count = int(
        np.count_nonzero(
            coarse_overlap
        )
    )


    print("\n")

    print("=" * 80)

    print(
        "COARSE OVERLAP RESULT"
    )

    print("=" * 80)


    print(
        "OHRC geometry points on valid TMC swath:",
        overlap_count
    )


    if overlap_count > 0:

        overlap_indices = (
            np.flatnonzero(
                coarse_overlap
            )
        )


        print(
            "\nOHRC overlapping scan range:"
        )

        print(
            int(
                ohrc_scan[
                    overlap_indices
                ].min()
            ),
            "to",
            int(
                ohrc_scan[
                    overlap_indices
                ].max()
            )
        )


        print(
            "OHRC overlapping pixel range:"
        )

        print(
            int(
                ohrc_pixel[
                    overlap_indices
                ].min()
            ),
            "to",
            int(
                ohrc_pixel[
                    overlap_indices
                ].max()
            )
        )


        print(
            "Overlapping latitude range:"
        )

        print(
            float(
                latitude[
                    overlap_indices
                ].min()
            ),
            "to",
            float(
                latitude[
                    overlap_indices
                ].max()
            )
        )


        print(
            "Overlapping longitude [0,360) range:"
        )

        print(
            float(
                longitude_360[
                    overlap_indices
                ].min()
            ),
            "to",
            float(
                longitude_360[
                    overlap_indices
                ].max()
            )
        )


    # ========================================================
    # FIND NEAREST VALID TMC DATA TO ENTIRE OHRC TRACK
    # ========================================================

    valid_rows, valid_cols = (
        np.nonzero(
            valid_mask
        )
    )


    if len(
        valid_rows
    ) == 0:

        raise RuntimeError(
            "TMC product contains no valid pixels "
            "in coarse representation."
        )


    # ========================================================
    # COARSE VALID PIXELS → APPROX FULL TMC PIXELS
    # ========================================================

    full_valid_cols = (

        (
            valid_cols.astype(
                np.float64
            )

            + 0.5
        )

        * src.width

        / preview_width
    )


    full_valid_rows = (

        (
            valid_rows.astype(
                np.float64
            )

            + 0.5
        )

        * src.height

        / preview_height
    )


    # ========================================================
    # FULL TMC PIXELS → PROJECTED METRIC COORDINATES
    # ========================================================

    (
        valid_x,
        valid_y

    ) = src.transform * (

        full_valid_cols,
        full_valid_rows
    )


    valid_x = np.asarray(
        valid_x,
        dtype=np.float64
    )


    valid_y = np.asarray(
        valid_y,
        dtype=np.float64
    )


    valid_xy = np.column_stack(
        (
            valid_x,
            valid_y
        )
    )


    # ========================================================
    # FINITE OHRC PROJECTED POINTS
    # ========================================================

    finite_indices = (
        np.flatnonzero(
            finite
        )
    )


    ohrc_xy = np.column_stack(
        (

            projected_x[
                finite_indices
            ],

            projected_y[
                finite_indices
            ]
        )
    )


    print(
        "\nBuilding nearest-valid-pixel search..."
    )


    # ========================================================
    # KD-TREE
    # ========================================================

    tree = cKDTree(
        valid_xy
    )


    (
        nearest_distance,
        nearest_index

    ) = tree.query(

        ohrc_xy,

        k=1
    )


    # ========================================================
    # CLOSEST OHRC GEOMETRY SAMPLE
    # ========================================================

    best_local_index = int(
        np.argmin(
            nearest_distance
        )
    )


    best_global_index = int(
        finite_indices[
            best_local_index
        ]
    )


    minimum_distance_m = float(
        nearest_distance[
            best_local_index
        ]
    )


    nearest_tmc_index = int(
        nearest_index[
            best_local_index
        ]
    )


    nearest_tmc_x = float(
        valid_x[
            nearest_tmc_index
        ]
    )


    nearest_tmc_y = float(
        valid_y[
            nearest_tmc_index
        ]
    )


    print("\n")

    print("=" * 80)

    print(
        "NEAREST TMC VALID DATA TO OHRC"
    )

    print("=" * 80)


    print(
        "Approx minimum distance:",
        minimum_distance_m,
        "metres"
    )


    print(
        "Approx minimum distance:",
        minimum_distance_m
        / 1000.0,
        "km"
    )


    print(
        "\nNearest OHRC geometry sample:"
    )


    print(
        "  OHRC pixel:",
        int(
            ohrc_pixel[
                best_global_index
            ]
        )
    )


    print(
        "  OHRC scan:",
        int(
            ohrc_scan[
                best_global_index
            ]
        )
    )


    print(
        "  Latitude:",
        float(
            latitude[
                best_global_index
            ]
        )
    )


    print(
        "  Longitude 0..360:",
        float(
            longitude_360[
                best_global_index
            ]
        )
    )


    print(
        "  Longitude -180..180:",
        float(
            longitude_180[
                best_global_index
            ]
        )
    )


    print(
        "\nNearest coarse valid TMC location:"
    )


    print(
        "  X:",
        nearest_tmc_x
    )


    print(
        "  Y:",
        nearest_tmc_y
    )


    # ========================================================
    # SAVE OVERLAP TABLE
    # ========================================================

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
        "Inside_TMC_BBox"
    ] = inside_bbox


    output_table[
        "Coarse_Valid_Overlap"
    ] = coarse_overlap


    output_table.to_csv(

        CSV_OUTPUT,

        index=False
    )


    # ========================================================
    # BUILD DISPLAY IMAGE
    # ========================================================

    valid_values = coarse[
        valid_mask
    ]


    display = np.zeros(
        coarse.shape,
        dtype=np.uint8
    )


    if valid_values.size > 0:

        low, high = np.percentile(
            valid_values,
            [1, 99]
        )


        if high > low:

            normalized = np.clip(

                coarse.astype(
                    np.float32
                ),

                low,
                high
            )


            normalized = (

                (
                    normalized
                    - low
                )

                /

                (
                    high
                    - low
                )

                * 255.0
            )


            display[
                valid_mask
            ] = (

                normalized[
                    valid_mask
                ]
                .astype(
                    np.uint8
                )
            )


    # Convert to BGR so we can draw coloured points.

    overlay = cv2.cvtColor(

        display,

        cv2.COLOR_GRAY2BGR
    )


    # ========================================================
    # DRAW OHRC GEOMETRY TRACK
    #
    # Red:
    #     OHRC geometry inside TMC bbox but not on valid TMC.
    #
    # Green:
    #     OHRC geometry coinciding with coarse valid TMC.
    # ========================================================

    draw_indices = (
        np.flatnonzero(
            inside_preview
        )
    )


    for index in draw_indices:

        x = int(
            coarse_cols[
                index
            ]
        )

        y = int(
            coarse_rows[
                index
            ]
        )


        if coarse_overlap[
            index
        ]:

            cv2.circle(

                overlay,

                (
                    x,
                    y
                ),

                3,

                (
                    0,
                    255,
                    0
                ),

                -1
            )

        else:

            cv2.circle(

                overlay,

                (
                    x,
                    y
                ),

                1,

                (
                    0,
                    0,
                    255
                ),

                -1
            )


    # ========================================================
    # MARK NEAREST OHRC POINT
    # ========================================================

    nearest_ohrc_x = int(
        coarse_cols[
            best_global_index
        ]
    )


    nearest_ohrc_y = int(
        coarse_rows[
            best_global_index
        ]
    )


    if (

        0
        <= nearest_ohrc_x
        < preview_width

        and

        0
        <= nearest_ohrc_y
        < preview_height
    ):

        cv2.drawMarker(

            overlay,

            (
                nearest_ohrc_x,
                nearest_ohrc_y
            ),

            (
                255,
                255,
                255
            ),

            markerType=
                cv2.MARKER_CROSS,

            markerSize=25,

            thickness=2
        )


    # ========================================================
    # MARK NEAREST VALID TMC CELL
    # ========================================================

    nearest_valid_coarse_col = int(
        valid_cols[
            nearest_tmc_index
        ]
    )


    nearest_valid_coarse_row = int(
        valid_rows[
            nearest_tmc_index
        ]
    )


    cv2.drawMarker(

        overlay,

        (
            nearest_valid_coarse_col,
            nearest_valid_coarse_row
        ),

        (
            255,
            255,
            0
        ),

        markerType=
            cv2.MARKER_DIAMOND,

        markerSize=25,

        thickness=2
    )


    # ========================================================
    # SAVE OVERLAY
    # ========================================================

    cv2.imwrite(

        str(
            OVERLAY_OUTPUT
        ),

        overlay
    )


# ============================================================
# FINAL DECISION
# ============================================================

print("\n")

print("=" * 80)


if overlap_count > 0:

    print(
        "FULL OHRC ↔ TMC RESULT: POTENTIAL OVERLAP FOUND"
    )


    print(
        "At least part of the OHRC geometry track "
        "falls on the TMC valid-data footprint."
    )


else:

    print(
        "FULL OHRC ↔ TMC RESULT: NO OVERLAP"
    )


    print(
        "No sampled OHRC geometry locations fall "
        "on the TMC valid-data footprint."
    )


print("=" * 80)


# ============================================================
# OUTPUT FILES
# ============================================================

print(
    "\nOverlap table:"
)

print(
    CSV_OUTPUT
)


print(
    "\nOverlay image:"
)

print(
    OVERLAY_OUTPUT
)