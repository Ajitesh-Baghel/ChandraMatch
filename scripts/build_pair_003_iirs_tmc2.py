from pathlib import Path
import json
import math

import cv2
import numpy as np
import rasterio

from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# IIRS PRODUCT
# ============================================================

IIRS_DIR = (
    ROOT
    / "data"
    / "raw"
    / "iirs"
    / "2024_01_15"
    / "data"
    / "derived"
    / "20240115"
)

PREFIX = (
    "ch2_iir_ndi_20240115T2100076733"
)

RFL_QUB = (
    IIRS_DIR
    / f"{PREFIX}_d_rfl_d18_srd.qub"
)

LOC_IMG = (
    IIRS_DIR
    / f"{PREFIX}_d_loc_d18_ard.img"
)


# ============================================================
# TMC
# ============================================================

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


# ============================================================
# EXACT OVERLAP
# ============================================================

OVERLAP_NPZ = (
    ROOT
    / "results"
    / "pair_003"
    / "exact_overlap"
    / "exact_overlap_masks.npz"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_003"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DEBUG_DIR = (
    ROOT
    / "results"
    / "pair_003"
    / "build"
)

DEBUG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IIRS DIMENSIONS
# ============================================================

IIRS_BANDS = 256
IIRS_LINES = 14695
IIRS_SAMPLES = 250

FLOAT32 = np.dtype("<f4")


# ============================================================
# SPECTRAL REPRESENTATION
#
# Bands 1-9:
# 712.3 -> 847.2 nm
# ============================================================

OPTICAL_BANDS = list(
    range(9)
)


# ============================================================
# COMMON GRID
# ============================================================

# Native IIRS nominal resolution from metadata.

COMMON_RESOLUTION_M = 63.37


# Geometry interpolation density.
#
# We do not need millions of Delaunay points because
# the IIRS geolocation surface is smooth.

GEOMETRY_LINE_STEP = 4
GEOMETRY_SAMPLE_STEP = 2


# Add a little map-space padding around the component.

GRID_PADDING_PIXELS = 3


# ============================================================
# MOON
# ============================================================

MOON_RADIUS_M = 1737400.0


# ============================================================
# OUTPUT FILES
# ============================================================

SOURCE_TIF = (
    OUTPUT_DIR
    / "source_iirs_on_common_grid.tif"
)

REFERENCE_TIF = (
    OUTPUT_DIR
    / "reference_tmc2_on_common_grid.tif"
)

MASK_TIF = (
    OUTPUT_DIR
    / "valid_mask.tif"
)


SOURCE_PNG = (
    OUTPUT_DIR
    / "source.png"
)

REFERENCE_PNG = (
    OUTPUT_DIR
    / "reference.png"
)

MASK_PNG = (
    OUTPUT_DIR
    / "valid_mask.png"
)

COMPARISON_PNG = (
    OUTPUT_DIR
    / "pair_comparison.png"
)

OVERLAY_PNG = (
    OUTPUT_DIR
    / "overlay_50_50.png"
)

DIFFERENCE_PNG = (
    OUTPUT_DIR
    / "difference.png"
)

METADATA_JSON = (
    OUTPUT_DIR
    / "pair_metadata.json"
)


# ============================================================
# HELPERS
# ============================================================

def percentile_stretch(
    image,
    mask,
    p_low=1,
    p_high=99
):

    output = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    values = image[
        mask
    ]

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:

        return output, None, None


    low, high = np.percentile(
        values,
        [
            p_low,
            p_high
        ]
    )


    if high <= low:

        return (
            output,
            float(low),
            float(high)
        )


    normalized = (
        image - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )


    output[
        mask
    ] = (
        normalized[
            mask
        ]
        *
        255.0
    ).astype(
        np.uint8
    )


    return (
        output,
        float(low),
        float(high)
    )


def save_tiff(
    path,
    image,
    transform,
    crs,
    dtype,
    nodata=None
):

    profile = {
        "driver":
            "GTiff",

        "height":
            image.shape[0],

        "width":
            image.shape[1],

        "count":
            1,

        "dtype":
            dtype,

        "crs":
            crs,

        "transform":
            transform,

        "compress":
            "deflate"
    }


    if nodata is not None:

        profile[
            "nodata"
        ] = nodata


    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            image.astype(
                dtype
            ),
            1
        )


def circular_angle_difference(
    a,
    b
):

    difference = abs(
        a - b
    ) % 360.0

    return min(
        difference,
        360.0 - difference
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - BUILD PAIR 003 IIRS / TMC-2"
    )

    print("=" * 80)


    # ========================================================
    # FILE CHECK
    # ========================================================

    for path in [
        RFL_QUB,
        LOC_IMG,
        TMC_PATH,
        OVERLAP_NPZ
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing:\n{path}"
            )


    # ========================================================
    # OPEN IIRS CUBES
    # ========================================================

    rfl = np.memmap(
        RFL_QUB,
        dtype=FLOAT32,
        mode="r",
        shape=(
            IIRS_BANDS,
            IIRS_LINES,
            IIRS_SAMPLES
        ),
        order="C"
    )


    loc = np.memmap(
        LOC_IMG,
        dtype=FLOAT32,
        mode="r",
        shape=(
            4,
            IIRS_LINES,
            IIRS_SAMPLES
        ),
        order="C"
    )


    print(
        "\nIIRS reflectance:",
        rfl.shape
    )

    print(
        "IIRS location:",
        loc.shape
    )


    # ========================================================
    # LOAD EXACT OVERLAP
    # ========================================================

    overlap_data = np.load(
        OVERLAP_NPZ
    )


    largest_mask = (
        overlap_data[
            "largest_component_mask"
        ].astype(
            bool
        )
    )


    component_lines, component_samples = (
        np.where(
            largest_mask
        )
    )


    if component_lines.size == 0:

        raise RuntimeError(
            "Largest overlap component is empty."
        )


    line_min = int(
        component_lines.min()
    )

    line_max = int(
        component_lines.max()
    )

    sample_min = int(
        component_samples.min()
    )

    sample_max = int(
        component_samples.max()
    )


    print("\n")
    print("=" * 80)

    print(
        "LARGEST EXACT OVERLAP COMPONENT"
    )

    print("=" * 80)


    print(
        "Native IIRS lines:",
        line_min,
        "->",
        line_max
    )

    print(
        "Native IIRS samples:",
        sample_min,
        "->",
        sample_max
    )

    print(
        "Component pixels:",
        int(
            component_lines.size
        )
    )


    # ========================================================
    # CROP
    # ========================================================

    crop_line0 = max(
        0,
        line_min - 2
    )

    crop_line1 = min(
        IIRS_LINES,
        line_max + 3
    )

    crop_sample0 = max(
        0,
        sample_min - 2
    )

    crop_sample1 = min(
        IIRS_SAMPLES,
        sample_max + 3
    )


    crop_height = (
        crop_line1
        -
        crop_line0
    )

    crop_width = (
        crop_sample1
        -
        crop_sample0
    )


    print(
        "\nIIRS crop:"
    )

    print(
        "Lines:",
        crop_line0,
        "->",
        crop_line1 - 1
    )

    print(
        "Samples:",
        crop_sample0,
        "->",
        crop_sample1 - 1
    )

    print(
        "Shape:",
        (
            crop_height,
            crop_width
        )
    )


    component_crop = (
        largest_mask[
            crop_line0:crop_line1,
            crop_sample0:crop_sample1
        ]
    )


    # ========================================================
    # BUILD IIRS OPTICAL REPRESENTATION
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "BUILDING IIRS 712-847 nm REPRESENTATION"
    )

    print("=" * 80)


    optical_sum = np.zeros(
        (
            crop_height,
            crop_width
        ),
        dtype=np.float32
    )


    optical_count = np.zeros(
        (
            crop_height,
            crop_width
        ),
        dtype=np.uint8
    )


    for band_index in OPTICAL_BANDS:

        print(
            "Reading IIRS band:",
            band_index + 1
        )


        band = np.asarray(
            rfl[
                band_index,
                crop_line0:crop_line1,
                crop_sample0:crop_sample1
            ],
            dtype=np.float32
        )


        valid = (
            np.isfinite(
                band
            )
            &
            (
                band >= 0
            )
            &
            (
                band < 5.0
            )
        )


        optical_sum[
            valid
        ] += band[
            valid
        ]


        optical_count[
            valid
        ] += 1


    optical = np.full(
        (
            crop_height,
            crop_width
        ),
        np.nan,
        dtype=np.float32
    )


    optical_valid = (
        optical_count > 0
    )


    optical[
        optical_valid
    ] = (
        optical_sum[
            optical_valid
        ]
        /
        optical_count[
            optical_valid
        ]
    )


    print(
        "Native optical valid ratio:",
        float(
            optical_valid.mean()
        )
    )


    # ========================================================
    # PROJECT IIRS GEOLOCATION
    # ========================================================

    with rasterio.open(
        TMC_PATH
    ) as tmc:

        tmc_crs = (
            tmc.crs
        )


        lunar_geographic = (
            CRS.from_proj4(
                f"+proj=longlat "
                f"+R={MOON_RADIUS_M} "
                f"+no_defs"
            )
        )


        transformer = (
            Transformer.from_crs(
                lunar_geographic,
                tmc_crs,
                always_xy=True
            )
        )


        longitude = np.asarray(
            loc[
                0,
                crop_line0:crop_line1,
                crop_sample0:crop_sample1
            ],
            dtype=np.float64
        )


        latitude = np.asarray(
            loc[
                1,
                crop_line0:crop_line1,
                crop_sample0:crop_sample1
            ],
            dtype=np.float64
        )


        x, y = transformer.transform(
            longitude,
            latitude
        )


        x = np.asarray(
            x,
            dtype=np.float64
        )

        y = np.asarray(
            y,
            dtype=np.float64
        )


        # ====================================================
        # BOUNDS FROM EXACT COMPONENT ONLY
        # ====================================================

        component_xy_valid = (
            component_crop
            &
            np.isfinite(x)
            &
            np.isfinite(y)
        )


        component_x = (
            x[
                component_xy_valid
            ]
        )

        component_y = (
            y[
                component_xy_valid
            ]
        )


        if component_x.size == 0:

            raise RuntimeError(
                "No projected component coordinates."
            )


        x_min = float(
            component_x.min()
        )

        x_max = float(
            component_x.max()
        )

        y_min = float(
            component_y.min()
        )

        y_max = float(
            component_y.max()
        )


        resolution = (
            COMMON_RESOLUTION_M
        )


        padding_m = (
            GRID_PADDING_PIXELS
            *
            resolution
        )


        x_min -= padding_m
        x_max += padding_m

        y_min -= padding_m
        y_max += padding_m


        # Align the common grid.

        x_min = (
            math.floor(
                x_min
                /
                resolution
            )
            *
            resolution
        )


        x_max = (
            math.ceil(
                x_max
                /
                resolution
            )
            *
            resolution
        )


        y_min = (
            math.floor(
                y_min
                /
                resolution
            )
            *
            resolution
        )


        y_max = (
            math.ceil(
                y_max
                /
                resolution
            )
            *
            resolution
        )


        output_width = int(
            round(
                (
                    x_max
                    -
                    x_min
                )
                /
                resolution
            )
        )


        output_height = int(
            round(
                (
                    y_max
                    -
                    y_min
                )
                /
                resolution
            )
        )


        output_transform = (
            from_origin(
                x_min,
                y_max,
                resolution,
                resolution
            )
        )


        print("\n")
        print("=" * 80)

        print(
            "COMMON MAP GRID"
        )

        print("=" * 80)


        print(
            "Resolution:",
            resolution,
            "m/pixel"
        )

        print(
            "Bounds:"
        )

        print(
            (
                x_min,
                y_min,
                x_max,
                y_max
            )
        )

        print(
            "Shape:",
            (
                output_height,
                output_width
            )
        )


        # ====================================================
        # BUILD SPARSE IIRS INVERSE GEOLOCATION MODEL
        # ====================================================

        local_rows = np.arange(
            crop_height,
            dtype=np.float64
        )


        local_cols = np.arange(
            crop_width,
            dtype=np.float64
        )


        sampled_rows = np.arange(
            0,
            crop_height,
            GEOMETRY_LINE_STEP
        )


        if (
            sampled_rows[-1]
            !=
            crop_height - 1
        ):

            sampled_rows = np.append(
                sampled_rows,
                crop_height - 1
            )


        sampled_cols = np.arange(
            0,
            crop_width,
            GEOMETRY_SAMPLE_STEP
        )


        if (
            sampled_cols[-1]
            !=
            crop_width - 1
        ):

            sampled_cols = np.append(
                sampled_cols,
                crop_width - 1
            )


        sample_col_grid, sample_row_grid = (
            np.meshgrid(
                sampled_cols,
                sampled_rows
            )
        )


        sample_x = (
            x[
                sample_row_grid,
                sample_col_grid
            ]
        )


        sample_y = (
            y[
                sample_row_grid,
                sample_col_grid
            ]
        )


        geometry_good = (
            np.isfinite(
                sample_x
            )
            &
            np.isfinite(
                sample_y
            )
        )


        points = np.column_stack(
            [
                sample_x[
                    geometry_good
                ],
                sample_y[
                    geometry_good
                ]
            ]
        )


        row_values = (
            sample_row_grid[
                geometry_good
            ].astype(
                np.float64
            )
        )


        col_values = (
            sample_col_grid[
                geometry_good
            ].astype(
                np.float64
            )
        )


        print(
            "\nGeometry interpolation points:",
            points.shape[0]
        )


        triangulation = (
            Delaunay(
                points
            )
        )


        row_interpolator = (
            LinearNDInterpolator(
                triangulation,
                row_values,
                fill_value=np.nan
            )
        )


        col_interpolator = (
            LinearNDInterpolator(
                triangulation,
                col_values,
                fill_value=np.nan
            )
        )


        # ====================================================
        # COMMON GRID PIXEL CENTERS
        # ====================================================

        grid_cols = np.arange(
            output_width,
            dtype=np.float64
        )


        grid_rows = np.arange(
            output_height,
            dtype=np.float64
        )


        grid_col_mesh, grid_row_mesh = (
            np.meshgrid(
                grid_cols,
                grid_rows
            )
        )


        grid_x = (
            x_min
            +
            (
                grid_col_mesh
                +
                0.5
            )
            *
            resolution
        )


        grid_y = (
            y_max
            -
            (
                grid_row_mesh
                +
                0.5
            )
            *
            resolution
        )


        query_points = np.column_stack(
            [
                grid_x.ravel(),
                grid_y.ravel()
            ]
        )


        map_y = (
            row_interpolator(
                query_points
            )
            .reshape(
                output_height,
                output_width
            )
            .astype(
                np.float32
            )
        )


        map_x = (
            col_interpolator(
                query_points
            )
            .reshape(
                output_height,
                output_width
            )
            .astype(
                np.float32
            )
        )


        mapping_valid = (
            np.isfinite(
                map_x
            )
            &
            np.isfinite(
                map_y
            )
            &
            (
                map_x >= 0
            )
            &
            (
                map_x <= crop_width - 1
            )
            &
            (
                map_y >= 0
            )
            &
            (
                map_y <= crop_height - 1
            )
        )


        print(
            "Geolocatable common-grid pixels:",
            int(
                np.count_nonzero(
                    mapping_valid
                )
            )
        )


        # ====================================================
        # REMAP IIRS TO COMMON GRID
        # ====================================================

        optical_for_remap = np.nan_to_num(
            optical,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(
            np.float32
        )


        source_grid = cv2.remap(
            optical_for_remap,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )


        source_native_valid = (
            optical_valid.astype(
                np.uint8
            )
            *
            255
        )


        source_valid_grid = (
            cv2.remap(
                source_native_valid,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            >
            0
        )


        # ====================================================
        # REMAP EXACT COMPONENT SUPPORT
        # ====================================================

        component_u8 = (
            component_crop.astype(
                np.uint8
            )
            *
            255
        )


        component_grid = (
            cv2.remap(
                component_u8,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            >
            0
        )


        # ====================================================
        # RESAMPLE TMC -> COMMON ~63 m GRID
        # ====================================================

        reference_grid = np.zeros(
            (
                output_height,
                output_width
            ),
            dtype=np.float32
        )


        print(
            "\nResampling TMC 5 m ->",
            resolution,
            "m..."
        )


        reproject(
            source=rasterio.band(
                tmc,
                1
            ),

            destination=
                reference_grid,

            src_transform=
                tmc.transform,

            src_crs=
                tmc.crs,

            src_nodata=
                0,

            dst_transform=
                output_transform,

            dst_crs=
                tmc_crs,

            dst_nodata=
                0,

            resampling=
                Resampling.average
        )


        reference_valid = (
            np.isfinite(
                reference_grid
            )
            &
            (
                reference_grid > 0
            )
        )


        # ====================================================
        # FINAL COMMON VALID MASK
        # ====================================================

        source_valid = (
            mapping_valid
            &
            source_valid_grid
            &
            np.isfinite(
                source_grid
            )
            &
            (
                source_grid >= 0
            )
        )


        common_mask = (
            component_grid
            &
            source_valid
            &
            reference_valid
        )


        print(
            "\nInitial common-grid valid pixels:",
            int(
                np.count_nonzero(
                    common_mask
                )
            )
        )

        print(
            "Initial common-grid valid ratio:",
            float(
                common_mask.mean()
            )
        )


        if not np.any(
            common_mask
        ):

            raise RuntimeError(
                "No common valid pixels after resampling."
            )


        # ====================================================
        # CROP OUTPUT TO COMMON SUPPORT BOUNDING BOX
        # ====================================================

        valid_rows, valid_cols = (
            np.where(
                common_mask
            )
        )


        final_row0 = max(
            0,
            int(
                valid_rows.min()
            )
            -
            2
        )

        final_row1 = min(
            output_height,
            int(
                valid_rows.max()
            )
            +
            3
        )

        final_col0 = max(
            0,
            int(
                valid_cols.min()
            )
            -
            2
        )

        final_col1 = min(
            output_width,
            int(
                valid_cols.max()
            )
            +
            3
        )


        source_final = (
            source_grid[
                final_row0:final_row1,
                final_col0:final_col1
            ]
        )


        reference_final = (
            reference_grid[
                final_row0:final_row1,
                final_col0:final_col1
            ]
        )


        mask_final = (
            common_mask[
                final_row0:final_row1,
                final_col0:final_col1
            ]
        )


        final_transform = (
            output_transform
            *
            rasterio.Affine.translation(
                final_col0,
                final_row0
            )
        )


        # Zero outside common valid support.

        source_final = (
            source_final.copy()
        )

        reference_final = (
            reference_final.copy()
        )


        source_final[
            ~mask_final
        ] = 0


        reference_final[
            ~mask_final
        ] = 0


        print("\n")
        print("=" * 80)

        print(
            "FINAL PAIR 003"
        )

        print("=" * 80)


        print(
            "Shape:",
            source_final.shape
        )

        print(
            "Valid pixels:",
            int(
                np.count_nonzero(
                    mask_final
                )
            )
        )

        print(
            "Valid ratio:",
            float(
                mask_final.mean()
            )
        )


        # ====================================================
        # SAVE GEOTIFFS
        # ====================================================

        save_tiff(
            SOURCE_TIF,
            source_final,
            final_transform,
            tmc_crs,
            "float32",
            nodata=0
        )


        save_tiff(
            REFERENCE_TIF,
            reference_final,
            final_transform,
            tmc_crs,
            "float32",
            nodata=0
        )


        save_tiff(
            MASK_TIF,
            (
                mask_final.astype(
                    np.uint8
                )
                *
                255
            ),
            final_transform,
            tmc_crs,
            "uint8",
            nodata=0
        )


        # ====================================================
        # MATCHING PNGS
        # ====================================================

        source_png, source_p01, source_p99 = (
            percentile_stretch(
                source_final,
                mask_final
            )
        )


        reference_png, reference_p01, reference_p99 = (
            percentile_stretch(
                reference_final,
                mask_final
            )
        )


        mask_png = (
            mask_final.astype(
                np.uint8
            )
            *
            255
        )


        cv2.imwrite(
            str(
                SOURCE_PNG
            ),
            source_png
        )


        cv2.imwrite(
            str(
                REFERENCE_PNG
            ),
            reference_png
        )


        cv2.imwrite(
            str(
                MASK_PNG
            ),
            mask_png
        )


        # ====================================================
        # COMPARISON
        # ====================================================

        divider = np.full(
            (
                source_png.shape[0],
                8
            ),
            255,
            dtype=np.uint8
        )


        comparison = np.hstack(
            [
                source_png,
                divider,
                reference_png
            ]
        )


        cv2.imwrite(
            str(
                COMPARISON_PNG
            ),
            comparison
        )


        # ====================================================
        # OVERLAY
        # ====================================================

        overlay = cv2.addWeighted(
            source_png,
            0.5,
            reference_png,
            0.5,
            0
        )


        overlay[
            ~mask_final
        ] = 0


        cv2.imwrite(
            str(
                OVERLAY_PNG
            ),
            overlay
        )


        # ====================================================
        # DIFFERENCE
        # ====================================================

        difference = cv2.absdiff(
            source_png,
            reference_png
        )


        difference[
            ~mask_final
        ] = 0


        cv2.imwrite(
            str(
                DIFFERENCE_PNG
            ),
            difference
        )


        # ====================================================
        # DEBUG SUPPORT IMAGES
        # ====================================================

        cv2.imwrite(
            str(
                DEBUG_DIR
                /
                "component_on_common_grid.png"
            ),
            (
                component_grid.astype(
                    np.uint8
                )
                *
                255
            )
        )


        cv2.imwrite(
            str(
                DEBUG_DIR
                /
                "tmc_valid_on_common_grid.png"
            ),
            (
                reference_valid.astype(
                    np.uint8
                )
                *
                255
            )
        )


        cv2.imwrite(
            str(
                DEBUG_DIR
                /
                "iirs_valid_on_common_grid.png"
            ),
            (
                source_valid.astype(
                    np.uint8
                )
                *
                255
            )
        )


        # ====================================================
        # METADATA
        # ====================================================

        iirs_resolution = (
            63.37
        )

        tmc_resolution = (
            5.0
        )


        iirs_sun_azimuth = (
            30.236921
        )

        iirs_sun_elevation = (
            36.116365
        )


        tmc_sun_azimuth = (
            286.401572
        )

        tmc_sun_elevation = (
            7.050806
        )


        metadata = {

            "pair_id":
                "pair_003",

            "source": {

                "sensor":
                    "IIRS",

                "product":
                    (
                        "ch2_iir_ndi_"
                        "20240115T2100076733_"
                        "d_rfl_d18_srd"
                    ),

                "representation":
                    (
                        "mean surface reflectance "
                        "of IIRS bands 1-9 "
                        "(712.3-847.2 nm)"
                    ),

                "native_resolution_m_per_px":
                    iirs_resolution,

                "sun_azimuth_deg":
                    iirs_sun_azimuth,

                "sun_elevation_deg":
                    iirs_sun_elevation
            },

            "reference": {

                "sensor":
                    "TMC-2",

                "product":
                    TMC_PATH.stem,

                "native_resolution_m_per_px":
                    tmc_resolution,

                "sun_azimuth_deg":
                    tmc_sun_azimuth,

                "sun_elevation_deg":
                    tmc_sun_elevation
            },

            "challenge": {

                "native_resolution_ratio":
                    (
                        iirs_resolution
                        /
                        tmc_resolution
                    ),

                "sun_azimuth_difference_deg":
                    circular_angle_difference(
                        iirs_sun_azimuth,
                        tmc_sun_azimuth
                    ),

                "sun_elevation_difference_deg":
                    abs(
                        iirs_sun_elevation
                        -
                        tmc_sun_elevation
                    )
            },

            "exact_overlap": {

                "largest_component_native_iirs_pixels":
                    int(
                        component_lines.size
                    ),

                "native_iirs_line_range":
                    [
                        line_min,
                        line_max
                    ],

                "native_iirs_sample_range":
                    [
                        sample_min,
                        sample_max
                    ]
            },

            "common_grid": {

                "resolution_m_per_px":
                    resolution,

                "shape": [
                    int(
                        source_final.shape[0]
                    ),
                    int(
                        source_final.shape[1]
                    )
                ],

                "valid_pixels":
                    int(
                        np.count_nonzero(
                            mask_final
                        )
                    ),

                "valid_ratio":
                    float(
                        mask_final.mean()
                    ),

                "crs":
                    str(
                        tmc_crs
                    ),

                "transform":
                    [
                        float(v)
                        for v
                        in final_transform[
                            :6
                        ]
                    ]
            },

            "visualization_stretch": {

                "source_p01":
                    source_p01,

                "source_p99":
                    source_p99,

                "reference_p01":
                    reference_p01,

                "reference_p99":
                    reference_p99
            },

            "note":
                (
                    "IIRS was not upsampled onto the "
                    "native 5 m TMC grid. Both sensors "
                    "were normalized to a common "
                    "63.37 m/pixel map grid, preserving "
                    "the native IIRS information scale."
                )
        }


        METADATA_JSON.write_text(
            json.dumps(
                metadata,
                indent=2
            ),
            encoding="utf-8"
        )


    # ========================================================
    # DONE
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "PAIR 003 BUILD COMPLETE"
    )

    print("=" * 80)


    print(
        "\nSource:"
    )

    print(
        SOURCE_PNG
    )


    print(
        "\nReference:"
    )

    print(
        REFERENCE_PNG
    )


    print(
        "\nMask:"
    )

    print(
        MASK_PNG
    )


    print(
        "\nComparison:"
    )

    print(
        COMPARISON_PNG
    )


    print(
        "\nOverlay:"
    )

    print(
        OVERLAY_PNG
    )


    print(
        "\nMetadata:"
    )

    print(
        METADATA_JSON
    )


if __name__ == "__main__":

    main()