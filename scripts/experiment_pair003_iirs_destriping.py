from pathlib import Path
import json
import math

import cv2
import numpy as np
import rasterio
import torch

from pyproj import CRS, Transformer
from scipy.ndimage import median_filter
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

import benchmark_pair003_matchers as bench


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# IIRS
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
# PAIR 003
# ============================================================

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_003"
)

CANONICAL_SOURCE_PNG = (
    PAIR_DIR
    / "source.png"
)

REFERENCE_PNG = (
    PAIR_DIR
    / "reference.png"
)

MASK_PNG = (
    PAIR_DIR
    / "valid_mask.png"
)

SOURCE_TIF = (
    PAIR_DIR
    / "source_iirs_on_common_grid.tif"
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
    / "results"
    / "pair_003"
    / "destriping_experiment"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_JSON = (
    OUTPUT_DIR
    / "pair003_destriping_experiment.json"
)


# ============================================================
# CONSTANTS
# ============================================================

IIRS_BANDS = 256
IIRS_LINES = 14695
IIRS_SAMPLES = 250

FLOAT32 = np.dtype("<f4")

MOON_RADIUS_M = 1737400.0

GEOMETRY_LINE_STEP = 4
GEOMETRY_SAMPLE_STEP = 2


# ============================================================
# BAND SETS
#
# 1-3:
# 712.3 - 746.0 nm
#
# 1-6:
# 712.3 - 796.6 nm
#
# 1-9:
# 712.3 - 847.2 nm
# ============================================================

BAND_SETS = {

    "bands_1_3": [
        0, 1, 2
    ],

    "bands_1_6": [
        0, 1, 2, 3, 4, 5
    ],

    "bands_1_9": [
        0, 1, 2, 3, 4, 5, 6, 7, 8
    ]
}


WAVELENGTHS_NM = [

    712.3,
    729.2,
    746.0,
    762.9,
    779.7,
    796.6,
    813.4,
    830.3,
    847.2
]


# ============================================================
# DETECTOR DESTRIPING SETTINGS
# ============================================================

# IIRS is a pushbroom instrument.
#
# Detector-to-detector gain variation tends to produce
# stripes along the flight direction.
#
# We estimate a robust median for each native SAMPLE
# column and compare it against a smooth cross-track
# profile.
#
# We only remove the high-frequency column-to-column
# component, preserving large-scale lunar albedo trends.

COLUMN_SMOOTH_WIDTH = 21

MIN_GAIN = 0.70
MAX_GAIN = 1.30


# ============================================================
# HELPERS
# ============================================================

def build_native_mean(
    rfl,
    bands,
    line0,
    line1,
    sample0,
    sample1
):

    height = (
        line1
        -
        line0
    )

    width = (
        sample1
        -
        sample0
    )


    total = np.zeros(
        (
            height,
            width
        ),
        dtype=np.float32
    )


    count = np.zeros(
        (
            height,
            width
        ),
        dtype=np.uint8
    )


    for band_index in bands:

        band = np.asarray(
            rfl[
                band_index,
                line0:line1,
                sample0:sample1
            ],
            dtype=np.float32
        )


        valid = (
            np.isfinite(
                band
            )
            &
            (
                band >= 0.0
            )
            &
            (
                band < 5.0
            )
        )


        total[
            valid
        ] += band[
            valid
        ]


        count[
            valid
        ] += 1


    output = np.full(
        (
            height,
            width
        ),
        np.nan,
        dtype=np.float32
    )


    valid_output = (
        count > 0
    )


    output[
        valid_output
    ] = (
        total[
            valid_output
        ]
        /
        count[
            valid_output
        ]
    )


    return (
        output,
        valid_output
    )


# ============================================================
# FILL MISSING PROFILE
# ============================================================

def fill_profile(
    profile
):

    profile = profile.astype(
        np.float64
    ).copy()


    valid = np.isfinite(
        profile
    )


    if not np.any(
        valid
    ):

        raise RuntimeError(
            "Detector profile contains no valid values."
        )


    indices = np.arange(
        len(
            profile
        )
    )


    profile[
        ~valid
    ] = np.interp(
        indices[
            ~valid
        ],
        indices[
            valid
        ],
        profile[
            valid
        ]
    )


    return profile


# ============================================================
# ROBUST COLUMN-GAIN DESTRIPING
# ============================================================

def detector_column_gain_destripe(
    image,
    valid
):

    working = np.where(
        valid,
        image,
        np.nan
    )


    # Robust median reflectance produced by
    # each detector/sample column.

    column_median = np.nanmedian(
        working,
        axis=0
    )


    column_median = fill_profile(
        column_median
    )


    # Smooth profile represents legitimate
    # low-frequency cross-track brightness.

    smooth_profile = median_filter(
        column_median,
        size=COLUMN_SMOOTH_WIDTH,
        mode="nearest"
    )


    epsilon = 1e-8


    gain = (
        column_median
        /
        np.maximum(
            smooth_profile,
            epsilon
        )
    )


    gain = np.clip(
        gain,
        MIN_GAIN,
        MAX_GAIN
    )


    corrected = np.full(
        image.shape,
        np.nan,
        dtype=np.float32
    )


    corrected[
        valid
    ] = (
        image[
            valid
        ]
        /
        np.broadcast_to(
            gain[
                None,
                :
            ],
            image.shape
        )[
            valid
        ]
    )


    # Preserve overall median brightness.

    original_median = float(
        np.nanmedian(
            image[
                valid
            ]
        )
    )


    corrected_median = float(
        np.nanmedian(
            corrected[
                valid
            ]
        )
    )


    if (
        corrected_median > 0
    ):

        corrected[
            valid
        ] *= (
            original_median
            /
            corrected_median
        )


    stats = {

        "gain_min":
            float(
                np.min(
                    gain
                )
            ),

        "gain_max":
            float(
                np.max(
                    gain
                )
            ),

        "gain_median":
            float(
                np.median(
                    gain
                )
            ),

        "gain_std":
            float(
                np.std(
                    gain
                )
            ),

        "column_profile_min":
            float(
                np.min(
                    column_median
                )
            ),

        "column_profile_max":
            float(
                np.max(
                    column_median
                )
            )
    }


    return (
        corrected,
        gain.astype(
            np.float32
        ),
        stats
    )


# ============================================================
# CREATE INVERSE GEOLOCATION MAP
# ============================================================

def build_common_grid_remap(
    loc,
    line0,
    line1,
    sample0,
    sample1,
    common_transform,
    common_crs,
    output_height,
    output_width
):

    longitude = np.asarray(
        loc[
            0,
            line0:line1,
            sample0:sample1
        ],
        dtype=np.float64
    )


    latitude = np.asarray(
        loc[
            1,
            line0:line1,
            sample0:sample1
        ],
        dtype=np.float64
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
            common_crs,
            always_xy=True
        )
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


    native_height = (
        line1
        -
        line0
    )

    native_width = (
        sample1
        -
        sample0
    )


    sampled_rows = np.arange(
        0,
        native_height,
        GEOMETRY_LINE_STEP
    )


    if (
        sampled_rows[-1]
        !=
        native_height - 1
    ):

        sampled_rows = np.append(
            sampled_rows,
            native_height - 1
        )


    sampled_cols = np.arange(
        0,
        native_width,
        GEOMETRY_SAMPLE_STEP
    )


    if (
        sampled_cols[-1]
        !=
        native_width - 1
    ):

        sampled_cols = np.append(
            sampled_cols,
            native_width - 1
        )


    col_grid, row_grid = np.meshgrid(
        sampled_cols,
        sampled_rows
    )


    sample_x = x[
        row_grid,
        col_grid
    ]


    sample_y = y[
        row_grid,
        col_grid
    ]


    geometry_valid = (
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
                geometry_valid
            ],

            sample_y[
                geometry_valid
            ]
        ]
    )


    row_values = (
        row_grid[
            geometry_valid
        ].astype(
            np.float64
        )
    )


    col_values = (
        col_grid[
            geometry_valid
        ].astype(
            np.float64
        )
    )


    print(
        "Geometry interpolation points:",
        len(
            points
        )
    )


    triangulation = Delaunay(
        points
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


    # Canonical Pair 003 GeoTIFF is north-up.

    if (
        abs(
            common_transform.b
        )
        >
        1e-12

        or

        abs(
            common_transform.d
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Unexpected rotated common-grid transform."
        )


    cols = np.arange(
        output_width,
        dtype=np.float64
    )


    rows = np.arange(
        output_height,
        dtype=np.float64
    )


    col_mesh, row_mesh = np.meshgrid(
        cols,
        rows
    )


    grid_x = (
        common_transform.c
        +
        (
            col_mesh
            +
            0.5
        )
        *
        common_transform.a
    )


    grid_y = (
        common_transform.f
        +
        (
            row_mesh
            +
            0.5
        )
        *
        common_transform.e
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
            map_x
            <=
            native_width - 1
        )
        &
        (
            map_y >= 0
        )
        &
        (
            map_y
            <=
            native_height - 1
        )
    )


    return (
        map_x,
        map_y,
        mapping_valid
    )


# ============================================================
# REMAP NATIVE IMAGE
# ============================================================

def remap_to_common_grid(
    native_image,
    native_valid,
    map_x,
    map_y,
    canonical_mask
):

    source = np.nan_to_num(
        native_image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(
        np.float32
    )


    remapped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


    native_valid_u8 = (
        native_valid.astype(
            np.uint8
        )
        *
        255
    )


    remapped_valid = (
        cv2.remap(
            native_valid_u8,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        >
        0
    )


    final_valid = (
        canonical_mask
        &
        remapped_valid
        &
        np.isfinite(
            remapped
        )
    )


    remapped[
        ~final_valid
    ] = 0


    return (
        remapped,
        final_valid
    )


# ============================================================
# STRETCH FLOAT IMAGE
# ============================================================

def stretch_float(
    image,
    valid
):

    values = image[
        valid
    ]


    values = values[
        np.isfinite(
            values
        )
    ]


    output = np.zeros(
        image.shape,
        dtype=np.uint8
    )


    if values.size == 0:

        return output


    low, high = np.percentile(
        values,
        [
            1,
            99
        ]
    )


    normalized = (
        image
        -
        low
    ) / (
        high
        -
        low
        +
        1e-12
    )


    normalized = np.clip(
        normalized,
        0,
        1
    )


    output[
        valid
    ] = (
        normalized[
            valid
        ]
        *
        255
    ).astype(
        np.uint8
    )


    return output


# ============================================================
# BENCHMARK LIGHTGLUE
# ============================================================

def benchmark_variant(
    variant_name,
    representation_name,
    source,
    reference,
    valid_mask,
    safe_mask,
    extractor,
    matcher
):

    if representation_name == "gradient":

        source_input = bench.make_gradient_image(
            source,
            valid_mask
        )

        reference_input = bench.make_gradient_image(
            reference,
            valid_mask
        )

    else:

        source_input = source.copy()

        reference_input = reference.copy()


    source_input[
        ~valid_mask
    ] = 0


    reference_input[
        ~valid_mask
    ] = 0


    (
        points0,
        points1,
        scores,
        matcher_metadata

    ) = bench.match_lightglue(
        source_input,
        reference_input,
        safe_mask,
        extractor,
        matcher
    )


    result, data = bench.evaluate_matches(
        points0,
        points1,
        scores,
        safe_mask
    )


    result[
        "variant"
    ] = variant_name


    result[
        "representation"
    ] = representation_name


    result[
        "matcher_metadata"
    ] = matcher_metadata


    print(
        f"\n{variant_name} / "
        f"{representation_name}"
    )


    print(
        "Matches:",
        result.get(
            "candidate_matches",
            0
        )
    )


    if (
        result.get(
            "status"
        )
        ==
        "ok"
    ):

        print(
            "Inliers:",
            result[
                "inliers"
            ]
        )

        print(
            "Ratio:",
            result[
                "inlier_ratio"
            ]
        )

        print(
            "RMSE:",
            result[
                "ransac_reprojection_rmse_px"
            ]
        )

        print(
            "Median:",
            result[
                "ransac_reprojection_median_px"
            ]
        )

        print(
            "Coverage:",
            result[
                "spatial"
            ][
                "coverage"
            ]
        )

        print(
            "Uniform:",
            result[
                "uniform_inliers"
            ]
        )

        print(
            "Transform sanity:",
            result[
                "transform_sanity"
            ]
        )

        print(
            "Transform:",
            result[
                "transform"
            ]
        )


        visualization_path = (
            OUTPUT_DIR
            /
            (
                f"{variant_name}_"
                f"{representation_name}_"
                f"inliers.png"
            )
        )


        bench.save_match_visualization(
            source_input,
            reference_input,
            data[
                "uniform_points0"
            ],
            data[
                "uniform_points1"
            ],
            visualization_path,
            max_draw=200
        )


    else:

        print(
            "Status:",
            result.get(
                "status"
            )
        )


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 003 "
        "IIRS DESTRIPING EXPERIMENT"
    )

    print("=" * 80)


    print(
        "\nDevice:",
        bench.DEVICE
    )


    if bench.DEVICE == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )


    # ========================================================
    # CHECK FILES
    # ========================================================

    for path in [

        RFL_QUB,
        LOC_IMG,
        OVERLAP_NPZ,

        CANONICAL_SOURCE_PNG,
        REFERENCE_PNG,
        MASK_PNG,
        SOURCE_TIF
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing:\n{path}"
            )


    # ========================================================
    # LOAD CANONICAL PAIR
    # ========================================================

    canonical_source = bench.load_gray(
        CANONICAL_SOURCE_PNG
    )


    reference = bench.load_gray(
        REFERENCE_PNG
    )


    mask_image = bench.load_gray(
        MASK_PNG
    )


    canonical_mask = (
        mask_image > 0
    )


    safe_mask = bench.make_safe_mask(
        mask_image
    )


    print(
        "\nCanonical shape:",
        canonical_source.shape
    )


    print(
        "Canonical valid pixels:",
        int(
            np.count_nonzero(
                canonical_mask
            )
        )
    )


    # ========================================================
    # EXACT OVERLAP CROP
    # ========================================================

    overlap = np.load(
        OVERLAP_NPZ
    )


    largest_mask = (
        overlap[
            "largest_component_mask"
        ].astype(
            bool
        )
    )


    lines, samples = np.where(
        largest_mask
    )


    line_min = int(
        lines.min()
    )

    line_max = int(
        lines.max()
    )

    sample_min = int(
        samples.min()
    )

    sample_max = int(
        samples.max()
    )


    line0 = max(
        0,
        line_min - 2
    )

    line1 = min(
        IIRS_LINES,
        line_max + 3
    )

    sample0 = max(
        0,
        sample_min - 2
    )

    sample1 = min(
        IIRS_SAMPLES,
        sample_max + 3
    )


    print(
        "\nNative crop:"
    )

    print(
        "Lines:",
        line0,
        "->",
        line1 - 1
    )

    print(
        "Samples:",
        sample0,
        "->",
        sample1 - 1
    )


    # ========================================================
    # MEMMAP
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


    # ========================================================
    # CANONICAL GRID
    # ========================================================

    with rasterio.open(
        SOURCE_TIF
    ) as dataset:

        common_transform = (
            dataset.transform
        )

        common_crs = (
            dataset.crs
        )

        output_height = (
            dataset.height
        )

        output_width = (
            dataset.width
        )


    print("\n")
    print("=" * 80)

    print(
        "BUILDING EXACT COMMON-GRID REMAP"
    )

    print("=" * 80)


    (
        map_x,
        map_y,
        mapping_valid

    ) = build_common_grid_remap(
        loc,

        line0,
        line1,

        sample0,
        sample1,

        common_transform,
        common_crs,

        output_height,
        output_width
    )


    print(
        "Geolocatable canonical pixels:",
        int(
            np.count_nonzero(
                mapping_valid
            )
        )
    )


    # ========================================================
    # BUILD EXPERIMENTAL VARIANTS
    # ========================================================

    variants = {

        "canonical_1_9":
            canonical_source
    }


    variant_metadata = {}


    print("\n")
    print("=" * 80)

    print(
        "BUILDING IIRS REPRESENTATIONS"
    )

    print("=" * 80)


    for band_name, bands in BAND_SETS.items():

        wavelengths = [
            WAVELENGTHS_NM[
                index
            ]
            for index
            in bands
        ]


        print(
            f"\n{band_name}"
        )

        print(
            "Bands:",
            [
                b + 1
                for b in bands
            ]
        )

        print(
            "Wavelengths:",
            wavelengths
        )


        (
            native_mean,
            native_valid

        ) = build_native_mean(
            rfl,
            bands,
            line0,
            line1,
            sample0,
            sample1
        )


        # ----------------------------------------------------
        # RAW SPECTRAL VARIANT
        # ----------------------------------------------------

        (
            raw_grid,
            raw_valid

        ) = remap_to_common_grid(
            native_mean,
            native_valid,
            map_x,
            map_y,
            canonical_mask
        )


        raw_png = stretch_float(
            raw_grid,
            raw_valid
        )


        raw_name = (
            f"{band_name}_raw"
        )


        variants[
            raw_name
        ] = raw_png


        cv2.imwrite(
            str(
                OUTPUT_DIR
                /
                f"{raw_name}.png"
            ),
            raw_png
        )


        # ----------------------------------------------------
        # DETECTOR GAIN DESTRIPING
        # ----------------------------------------------------

        (
            corrected_native,
            gain,
            destripe_stats

        ) = detector_column_gain_destripe(
            native_mean,
            native_valid
        )


        (
            corrected_grid,
            corrected_valid

        ) = remap_to_common_grid(
            corrected_native,
            native_valid,
            map_x,
            map_y,
            canonical_mask
        )


        corrected_png = stretch_float(
            corrected_grid,
            corrected_valid
        )


        corrected_name = (
            f"{band_name}_destriped"
        )


        variants[
            corrected_name
        ] = corrected_png


        cv2.imwrite(
            str(
                OUTPUT_DIR
                /
                f"{corrected_name}.png"
            ),
            corrected_png
        )


        # Gain profile visualization.

        gain_min = float(
            np.min(
                gain
            )
        )

        gain_max = float(
            np.max(
                gain
            )
        )


        gain_vis = np.zeros(
            (
                240,
                len(
                    gain
                )
            ),
            dtype=np.uint8
        )


        if gain_max > gain_min:

            gain_normalized = (
                gain
                -
                gain_min
            ) / (
                gain_max
                -
                gain_min
            )

        else:

            gain_normalized = np.zeros_like(
                gain
            )


        for x_index, value in enumerate(
            gain_normalized
        ):

            y_value = int(
                round(
                    (
                        1.0
                        -
                        float(
                            value
                        )
                    )
                    *
                    220
                )
            )

            y_value += 10


            cv2.circle(
                gain_vis,
                (
                    x_index,
                    y_value
                ),
                1,
                255,
                -1
            )


        gain_vis = cv2.resize(
            gain_vis,
            (
                1000,
                300
            ),
            interpolation=cv2.INTER_NEAREST
        )


        cv2.imwrite(
            str(
                OUTPUT_DIR
                /
                f"{band_name}_gain_profile.png"
            ),
            gain_vis
        )


        variant_metadata[
            band_name
        ] = {

            "bands_1_based":
                [
                    b + 1
                    for b in bands
                ],

            "wavelengths_nm":
                wavelengths,

            "raw_valid_pixels":
                int(
                    np.count_nonzero(
                        raw_valid
                    )
                ),

            "destriped_valid_pixels":
                int(
                    np.count_nonzero(
                        corrected_valid
                    )
                ),

            "destriping":
                destripe_stats
        }


        print(
            "Destriping stats:",
            destripe_stats
        )


    # ========================================================
    # LIGHTGLUE
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "LIGHTGLUE REPRESENTATION BENCHMARK"
    )

    print("=" * 80)


    print(
        "\nLoading LightGlue..."
    )


    extractor, matcher = (
        bench.create_lightglue()
    )


    print(
        "LightGlue ready."
    )


    results = []


    for variant_name, source_variant in variants.items():

        # Test both raw intensity and gradient.

        for representation in [
            "baseline",
            "gradient"
        ]:

            result = benchmark_variant(

                variant_name,
                representation,

                source_variant,
                reference,

                canonical_mask,
                safe_mask,

                extractor,
                matcher
            )


            results.append(
                result
            )


    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    print("\n")
    print("=" * 120)

    print(
        "PAIR 003 IIRS REPRESENTATION SUMMARY"
    )

    print("=" * 120)


    header = (

        f"{'Variant':<28}"
        f"{'Rep':<10}"
        f"{'Matches':>9}"
        f"{'Inliers':>9}"
        f"{'Ratio':>9}"
        f"{'RMSE':>9}"
        f"{'Coverage':>11}"
        f"{'Uniform':>9}"
        f"{'Sane':>8}"
    )


    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )


    for result in results:

        if (
            result.get(
                "status"
            )
            ==
            "ok"
        ):

            print(

                f"{result['variant']:<28}"
                f"{result['representation']:<10}"

                f"{result['candidate_matches']:>9}"
                f"{result['inliers']:>9}"

                f"{result['inlier_ratio']:>9.3f}"

                f"{result['ransac_reprojection_rmse_px']:>9.3f}"

                f"{result['spatial']['coverage']:>11.3f}"

                f"{result['uniform_inliers']:>9}"

                f"{str(result['transform_sanity']):>8}"
            )

        else:

            print(

                f"{result['variant']:<28}"
                f"{result['representation']:<10}"

                f"{result.get('candidate_matches', 0):>9}"

                f"{'-':>9}"
                f"{'-':>9}"
                f"{'-':>9}"
                f"{'-':>11}"
                f"{'-':>9}"

                f"{result.get('status', 'fail'):>8}"
            )


    # ========================================================
    # FIND BEST SANE RESULT
    # ========================================================

    sane_results = [

        result

        for result
        in results

        if (
            result.get(
                "status"
            )
            ==
            "ok"

            and

            result.get(
                "transform_sanity"
            )
            is True
        )
    ]


    if len(
        sane_results
    ) > 0:

        best = max(

            sane_results,

            key=lambda result: (

                result[
                    "spatial"
                ][
                    "coverage"
                ],

                result[
                    "inliers"
                ],

                result[
                    "inlier_ratio"
                ]
            )
        )


        print("\n")
        print("=" * 80)

        print(
            "BEST SANE REPRESENTATION"
        )

        print("=" * 80)


        print(
            "Variant:",
            best[
                "variant"
            ]
        )

        print(
            "Representation:",
            best[
                "representation"
            ]
        )

        print(
            "Inliers:",
            best[
                "inliers"
            ]
        )

        print(
            "Coverage:",
            best[
                "spatial"
            ][
                "coverage"
            ]
        )

        print(
            "RMSE:",
            best[
                "ransac_reprojection_rmse_px"
            ]
        )

        print(
            "Transform:",
            best[
                "transform"
            ]
        )

    else:

        best = None


    # ========================================================
    # SAVE JSON
    # ========================================================

    payload = {

        "pair_id":
            "pair_003",

        "experiment":
            "native_iirs_detector_gain_destriping",

        "native_crop": {

            "line_start":
                line0,

            "line_end_exclusive":
                line1,

            "sample_start":
                sample0,

            "sample_end_exclusive":
                sample1
        },

        "destriping_settings": {

            "column_smooth_width":
                COLUMN_SMOOTH_WIDTH,

            "gain_clip": [
                MIN_GAIN,
                MAX_GAIN
            ],

            "description":
                (
                    "Robust detector/sample-column "
                    "median gain correction relative "
                    "to a smooth cross-track profile. "
                    "Applied in native IIRS geometry "
                    "before common-grid warping."
                )
        },

        "variant_metadata":
            variant_metadata,

        "results":
            results,

        "best_sane_result":
            best,

        "metric_note":
            (
                "RANSAC reprojection RMSE is "
                "correspondence self-consistency, "
                "not independent ground-truth "
                "registration accuracy."
            )
    }


    RESULT_JSON.write_text(

        json.dumps(
            payload,
            indent=2
        ),

        encoding="utf-8"
    )


    print("\n")
    print("=" * 80)

    print(
        "DESTRIPING EXPERIMENT COMPLETE"
    )

    print("=" * 80)


    print(
        "\nResults:"
    )

    print(
        RESULT_JSON
    )


    print(
        "\nImages:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()