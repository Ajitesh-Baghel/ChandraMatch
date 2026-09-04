from pathlib import Path
import csv
import json

import cv2
import numpy as np
import rasterio

from pyproj import CRS, Transformer
from rasterio.windows import Window


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
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_003"
    / "exact_overlap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUMMARY_PATH = (
    OUTPUT_DIR
    / "exact_overlap_summary.json"
)

MASK_NPZ_PATH = (
    OUTPUT_DIR
    / "exact_overlap_masks.npz"
)

MASK_PATH = (
    OUTPUT_DIR
    / "exact_overlap_mask.png"
)

MASK_STRETCHED_PATH = (
    OUTPUT_DIR
    / "exact_overlap_mask_stretched.png"
)

LARGEST_PATH = (
    OUTPUT_DIR
    / "largest_overlap_component.png"
)

LARGEST_STRETCHED_PATH = (
    OUTPUT_DIR
    / "largest_overlap_component_stretched.png"
)

LINE_CSV_PATH = (
    OUTPUT_DIR
    / "overlap_by_iirs_line.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

MOON_RADIUS_M = 1737400.0

IIRS_LINES = 14695
IIRS_SAMPLES = 250
LOC_BANDS = 4

DTYPE = np.dtype("<f4")


# Process a few hundred IIRS lines at once.
# This keeps TMC reads small despite the huge source raster.

CHUNK_LINES = 256


# ============================================================
# OPEN IIRS LOCATION CUBE
# ============================================================

def open_location_cube():

    expected_bytes = (
        LOC_BANDS
        *
        IIRS_LINES
        *
        IIRS_SAMPLES
        *
        DTYPE.itemsize
    )

    actual_bytes = (
        LOC_IMG.stat().st_size
    )

    if actual_bytes != expected_bytes:

        raise RuntimeError(
            "\nIIRS location file size mismatch.\n"
            f"Expected: {expected_bytes}\n"
            f"Actual:   {actual_bytes}"
        )

    return np.memmap(
        LOC_IMG,
        dtype=DTYPE,
        mode="r",
        shape=(
            LOC_BANDS,
            IIRS_LINES,
            IIRS_SAMPLES
        ),
        order="C"
    )


# ============================================================
# VISUALIZATION
# ============================================================

def save_binary_image(
    mask,
    path
):

    image = (
        mask.astype(
            np.uint8
        )
        *
        255
    )

    cv2.imwrite(
        str(path),
        image
    )


def save_stretched(
    mask,
    path
):

    image = (
        mask.astype(
            np.uint8
        )
        *
        255
    )

    # Native IIRS is extremely narrow:
    # 250 × 14695.
    #
    # Stretch width ONLY for visualization.

    target_width = 700
    target_height = 3000

    stretched = cv2.resize(
        image,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imwrite(
        str(path),
        stretched
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 003 EXACT IIRS/TMC OVERLAP"
    )

    print("=" * 80)


    if not LOC_IMG.exists():

        raise FileNotFoundError(
            f"IIRS location image not found:\n{LOC_IMG}"
        )


    if not TMC_PATH.exists():

        raise FileNotFoundError(
            f"TMC raster not found:\n{TMC_PATH}"
        )


    # ========================================================
    # LOAD LOCATION CUBE
    # ========================================================

    loc = open_location_cube()


    print(
        "\nIIRS location cube:"
    )

    print(
        "Shape:",
        loc.shape
    )

    print(
        "Memory mapped:",
        isinstance(
            loc,
            np.memmap
        )
    )


    # ========================================================
    # OUTPUT MASKS
    # ========================================================

    inside_tmc_mask = np.zeros(
        (
            IIRS_LINES,
            IIRS_SAMPLES
        ),
        dtype=bool
    )


    valid_tmc_mask = np.zeros(
        (
            IIRS_LINES,
            IIRS_SAMPLES
        ),
        dtype=bool
    )


    # ========================================================
    # OPEN TMC
    # ========================================================

    with rasterio.open(
        TMC_PATH
    ) as tmc:

        print(
            "\nTMC:"
        )

        print(
            TMC_PATH
        )

        print(
            "Shape:",
            (
                tmc.height,
                tmc.width
            )
        )

        print(
            "Resolution:",
            tmc.res
        )

        print(
            "Bounds:",
            tmc.bounds
        )

        print(
            "CRS:"
        )

        print(
            tmc.crs
        )


        # ====================================================
        # GEOGRAPHIC → TMC CRS
        # ====================================================

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
                tmc.crs,
                always_xy=True
            )
        )


        transform = (
            tmc.transform
        )


        # We expect the existing TMC to be
        # north-up/no rotation.

        if (
            abs(transform.b) > 1e-12
            or
            abs(transform.d) > 1e-12
        ):

            raise RuntimeError(
                "Unexpected rotated/sheared TMC transform."
            )


        # ====================================================
        # PROCESS IIRS IN LINE CHUNKS
        # ====================================================

        total_chunks = int(
            np.ceil(
                IIRS_LINES
                /
                CHUNK_LINES
            )
        )


        print("\n")
        print("=" * 80)

        print(
            "PROJECTING ALL IIRS PIXEL CENTERS"
        )

        print("=" * 80)


        for chunk_index, line0 in enumerate(
            range(
                0,
                IIRS_LINES,
                CHUNK_LINES
            )
        ):

            line1 = min(
                IIRS_LINES,
                line0 + CHUNK_LINES
            )


            longitude = np.asarray(
                loc[
                    0,
                    line0:line1,
                    :
                ],
                dtype=np.float64
            )


            latitude = np.asarray(
                loc[
                    1,
                    line0:line1,
                    :
                ],
                dtype=np.float64
            )


            shape = longitude.shape


            longitude_flat = (
                longitude.ravel()
            )

            latitude_flat = (
                latitude.ravel()
            )


            geometry_valid = (
                np.isfinite(
                    longitude_flat
                )
                &
                np.isfinite(
                    latitude_flat
                )
                &
                (
                    latitude_flat >= -90
                )
                &
                (
                    latitude_flat <= 90
                )
            )


            x = np.full(
                longitude_flat.shape,
                np.nan,
                dtype=np.float64
            )

            y = np.full(
                latitude_flat.shape,
                np.nan,
                dtype=np.float64
            )


            if np.any(
                geometry_valid
            ):

                x_valid, y_valid = (
                    transformer.transform(
                        longitude_flat[
                            geometry_valid
                        ],
                        latitude_flat[
                            geometry_valid
                        ]
                    )
                )


                x[
                    geometry_valid
                ] = x_valid

                y[
                    geometry_valid
                ] = y_valid


            # =================================================
            # PROJECTED COORDINATES -> TMC PIXEL
            # =================================================

            finite_xy = (
                np.isfinite(x)
                &
                np.isfinite(y)
            )


            rows = np.full(
                x.shape,
                -1,
                dtype=np.int32
            )

            cols = np.full(
                x.shape,
                -1,
                dtype=np.int32
            )


            # col = (x - originX) / pixelWidth
            #
            # row = (y - originY) / pixelHeight
            #
            # pixelHeight is negative for north-up image.

            cols_float = (
                (
                    x[
                        finite_xy
                    ]
                    -
                    transform.c
                )
                /
                transform.a
            )


            rows_float = (
                (
                    y[
                        finite_xy
                    ]
                    -
                    transform.f
                )
                /
                transform.e
            )


            cols[
                finite_xy
            ] = np.floor(
                cols_float
            ).astype(
                np.int32
            )


            rows[
                finite_xy
            ] = np.floor(
                rows_float
            ).astype(
                np.int32
            )


            inside = (
                finite_xy
                &
                (
                    rows >= 0
                )
                &
                (
                    rows < tmc.height
                )
                &
                (
                    cols >= 0
                )
                &
                (
                    cols < tmc.width
                )
            )


            inside_tmc_mask[
                line0:line1,
                :
            ] = inside.reshape(
                shape
            )


            # =================================================
            # SAMPLE NATIVE TMC VALIDITY
            # =================================================

            valid_flat = np.zeros(
                inside.shape,
                dtype=bool
            )


            if np.any(
                inside
            ):

                inside_indices = (
                    np.flatnonzero(
                        inside
                    )
                )


                inside_rows = (
                    rows[
                        inside_indices
                    ]
                )

                inside_cols = (
                    cols[
                        inside_indices
                    ]
                )


                row_min = int(
                    inside_rows.min()
                )

                row_max = int(
                    inside_rows.max()
                )

                col_min = int(
                    inside_cols.min()
                )

                col_max = int(
                    inside_cols.max()
                )


                window = Window(
                    col_min,
                    row_min,
                    col_max - col_min + 1,
                    row_max - row_min + 1
                )


                tmc_chunk = tmc.read(
                    1,
                    window=window
                )


                local_rows = (
                    inside_rows
                    -
                    row_min
                )

                local_cols = (
                    inside_cols
                    -
                    col_min
                )


                values = (
                    tmc_chunk[
                        local_rows,
                        local_cols
                    ]
                )


                valid_values = (
                    np.isfinite(
                        values
                    )
                    &
                    (
                        values > 0
                    )
                )


                valid_flat[
                    inside_indices
                ] = valid_values


            valid_tmc_mask[
                line0:line1,
                :
            ] = valid_flat.reshape(
                shape
            )


            print(
                f"Chunk "
                f"{chunk_index + 1:02d}"
                f"/"
                f"{total_chunks:02d} "
                f"| lines "
                f"{line0:05d}"
                f" -> "
                f"{line1 - 1:05d} "
                f"| inside="
                f"{int(np.count_nonzero(inside))} "
                f"| valid="
                f"{int(np.count_nonzero(valid_flat))}"
            )


    # ========================================================
    # GLOBAL COUNTS
    # ========================================================

    total_pixels = (
        IIRS_LINES
        *
        IIRS_SAMPLES
    )


    inside_count = int(
        np.count_nonzero(
            inside_tmc_mask
        )
    )


    common_count = int(
        np.count_nonzero(
            valid_tmc_mask
        )
    )


    inside_ratio = (
        inside_count
        /
        total_pixels
    )


    common_ratio = (
        common_count
        /
        total_pixels
    )


    if inside_count > 0:

        valid_given_inside_ratio = (
            common_count
            /
            inside_count
        )

    else:

        valid_given_inside_ratio = 0.0


    print("\n")
    print("=" * 80)

    print(
        "EXACT CENTER-POINT OVERLAP"
    )

    print("=" * 80)


    print(
        "Total IIRS pixels:",
        total_pixels
    )

    print(
        "IIRS pixel centers inside TMC raster:",
        inside_count
    )

    print(
        "Inside-TMC ratio:",
        inside_ratio
    )

    print(
        "IIRS centers on VALID TMC:",
        common_count
    )

    print(
        "Exact IIRS/TMC overlap ratio:",
        common_ratio
    )

    print(
        "Valid TMC among inside centers:",
        valid_given_inside_ratio
    )


    if common_count == 0:

        print("\n")
        print(
            "PAIR 003 REJECTED: "
            "NO EXACT VALID OVERLAP"
        )

        return


    # ========================================================
    # OVERLAPPING LINE RANGE
    # ========================================================

    line_has_overlap = (
        np.any(
            valid_tmc_mask,
            axis=1
        )
    )


    overlap_lines = (
        np.flatnonzero(
            line_has_overlap
        )
    )


    first_line = int(
        overlap_lines.min()
    )

    last_line = int(
        overlap_lines.max()
    )


    sample_has_overlap = (
        np.any(
            valid_tmc_mask,
            axis=0
        )
    )


    overlap_samples = (
        np.flatnonzero(
            sample_has_overlap
        )
    )


    first_sample = int(
        overlap_samples.min()
    )

    last_sample = int(
        overlap_samples.max()
    )


    print(
        "\nOverall overlapping IIRS line range:",
        first_line,
        "->",
        last_line
    )

    print(
        "Overall overlapping IIRS sample range:",
        first_sample,
        "->",
        last_sample
    )


    # ========================================================
    # CONNECTED COMPONENTS
    #
    # Find the largest spatially contiguous common region
    # on the native IIRS grid.
    # ========================================================

    binary = (
        valid_tmc_mask.astype(
            np.uint8
        )
    )


    (
        component_count,
        labels,
        stats,
        centroids

    ) = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )


    # Background is label 0.

    real_component_count = (
        component_count - 1
    )


    print(
        "\nConnected overlap components:",
        real_component_count
    )


    largest_component_mask = np.zeros_like(
        valid_tmc_mask,
        dtype=bool
    )


    largest_label = None
    largest_area = 0


    if real_component_count > 0:

        component_areas = (
            stats[
                1:,
                cv2.CC_STAT_AREA
            ]
        )


        largest_label = (
            1
            +
            int(
                np.argmax(
                    component_areas
                )
            )
        )


        largest_area = int(
            stats[
                largest_label,
                cv2.CC_STAT_AREA
            ]
        )


        largest_component_mask = (
            labels
            ==
            largest_label
        )


    print(
        "Largest overlap component:",
        largest_area,
        "IIRS pixels"
    )


    # ========================================================
    # LARGEST COMPONENT EXTENT
    # ========================================================

    largest_info = None


    if largest_area > 0:

        largest_lines, largest_samples = (
            np.where(
                largest_component_mask
            )
        )


        largest_line_min = int(
            largest_lines.min()
        )

        largest_line_max = int(
            largest_lines.max()
        )

        largest_sample_min = int(
            largest_samples.min()
        )

        largest_sample_max = int(
            largest_samples.max()
        )


        longitude_all = np.asarray(
            loc[0],
            dtype=np.float32
        )

        latitude_all = np.asarray(
            loc[1],
            dtype=np.float32
        )


        component_longitude = (
            longitude_all[
                largest_component_mask
            ]
        )

        component_latitude = (
            latitude_all[
                largest_component_mask
            ]
        )


        longitude_180 = (
            (
                component_longitude.astype(
                    np.float64
                )
                +
                180.0
            )
            %
            360.0
        ) - 180.0


        largest_info = {

            "area_iirs_pixels":
                largest_area,

            "line_min":
                largest_line_min,

            "line_max":
                largest_line_max,

            "sample_min":
                largest_sample_min,

            "sample_max":
                largest_sample_max,

            "latitude_min":
                float(
                    np.min(
                        component_latitude
                    )
                ),

            "latitude_max":
                float(
                    np.max(
                        component_latitude
                    )
                ),

            "longitude_180_min":
                float(
                    np.min(
                        longitude_180
                    )
                ),

            "longitude_180_max":
                float(
                    np.max(
                        longitude_180
                    )
                )
        }


        print("\n")
        print("=" * 80)

        print(
            "LARGEST CONTIGUOUS OVERLAP"
        )

        print("=" * 80)


        print(
            "IIRS line:",
            largest_line_min,
            "->",
            largest_line_max
        )

        print(
            "IIRS sample:",
            largest_sample_min,
            "->",
            largest_sample_max
        )

        print(
            "Latitude:",
            largest_info[
                "latitude_min"
            ],
            "->",
            largest_info[
                "latitude_max"
            ]
        )

        print(
            "Longitude:",
            largest_info[
                "longitude_180_min"
            ],
            "->",
            largest_info[
                "longitude_180_max"
            ]
        )


    # ========================================================
    # LINE-BY-LINE CSV
    # ========================================================

    with LINE_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(
            f
        )


        writer.writerow(
            [
                "iirs_line",
                "inside_tmc_pixels",
                "valid_tmc_pixels",
                "first_valid_sample",
                "last_valid_sample"
            ]
        )


        for line in range(
            IIRS_LINES
        ):

            inside_line = (
                inside_tmc_mask[
                    line
                ]
            )

            valid_line = (
                valid_tmc_mask[
                    line
                ]
            )


            inside_line_count = int(
                np.count_nonzero(
                    inside_line
                )
            )

            valid_line_count = int(
                np.count_nonzero(
                    valid_line
                )
            )


            if valid_line_count > 0:

                samples = (
                    np.flatnonzero(
                        valid_line
                    )
                )


                first_valid = int(
                    samples.min()
                )

                last_valid = int(
                    samples.max()
                )

            else:

                first_valid = ""
                last_valid = ""


            writer.writerow(
                [
                    line,
                    inside_line_count,
                    valid_line_count,
                    first_valid,
                    last_valid
                ]
            )


    # ========================================================
    # SAVE MASK DATA
    # ========================================================

    np.savez_compressed(
        MASK_NPZ_PATH,

        inside_tmc_mask=
            inside_tmc_mask,

        valid_tmc_mask=
            valid_tmc_mask,

        largest_component_mask=
            largest_component_mask
    )


    # ========================================================
    # SAVE VISUALS
    # ========================================================

    save_binary_image(
        valid_tmc_mask,
        MASK_PATH
    )


    save_stretched(
        valid_tmc_mask,
        MASK_STRETCHED_PATH
    )


    save_binary_image(
        largest_component_mask,
        LARGEST_PATH
    )


    save_stretched(
        largest_component_mask,
        LARGEST_STRETCHED_PATH
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "pair_id":
            "pair_003",

        "source_sensor":
            "IIRS",

        "reference_sensor":
            "TMC-2",

        "iirs_product":
            PREFIX,

        "tmc_product":
            TMC_PATH.stem,

        "iirs_shape": [
            IIRS_LINES,
            IIRS_SAMPLES
        ],

        "total_iirs_pixels":
            total_pixels,

        "inside_tmc_pixels":
            inside_count,

        "inside_tmc_ratio":
            float(
                inside_ratio
            ),

        "valid_tmc_overlap_pixels":
            common_count,

        "exact_overlap_ratio":
            float(
                common_ratio
            ),

        "valid_tmc_given_inside_ratio":
            float(
                valid_given_inside_ratio
            ),

        "overlap_line_range": [
            first_line,
            last_line
        ],

        "overlap_sample_range": [
            first_sample,
            last_sample
        ],

        "connected_components":
            int(
                real_component_count
            ),

        "largest_component":
            largest_info,

        "definition":
            (
                "Exact overlap is measured by "
                "projecting each native IIRS "
                "pixel center into the TMC-2 "
                "native 5 m raster and testing "
                "whether the corresponding TMC "
                "pixel contains a finite value > 0."
            )
    }


    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )


    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "PAIR 003 EXACT OVERLAP CONFIRMED"
    )

    print("=" * 80)


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_PATH
    )


    print(
        "\nExact mask:"
    )

    print(
        MASK_PATH
    )


    print(
        "\nStretched exact mask:"
    )

    print(
        MASK_STRETCHED_PATH
    )


    print(
        "\nLargest component:"
    )

    print(
        LARGEST_PATH
    )


    print(
        "\nLine summary:"
    )

    print(
        LINE_CSV_PATH
    )


if __name__ == "__main__":

    main()  