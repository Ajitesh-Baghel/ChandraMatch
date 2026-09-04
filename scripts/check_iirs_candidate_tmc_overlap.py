from pathlib import Path

import cv2
import numpy as np
import rasterio

from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.transform import rowcol
from rasterio.windows import Window
from shapely.geometry import Polygon


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# EXISTING TMC PRODUCT
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
# IIRS CANDIDATE
# ============================================================

IIRS_PRODUCT_ID = (
    "ch2_iir_ndi_20240115T2100076733_d_rfl_d18_srd"
)


# Refined corners from IIRS XML.
#
# Order:
# UL, UR, LR, LL
#
# IMPORTANT:
# longitude values from ISDA may be 0..360.
# They are normalized later.

IIRS_CORNERS = [

    (
        101.017258,
        -71.936597
    ),

    (
        99.237446,
        -71.946877
    ),

    (
        99.500905,
        -31.410223
    ),

    (
        100.117132,
        -31.402947
    )
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_003"
    / "candidate_screening"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OVERLAY_OUTPUT = (
    OUTPUT_DIR
    / "iirs_candidate_tmc_overlap.png"
)


MASK_OUTPUT = (
    OUTPUT_DIR
    / "iirs_candidate_footprint_mask.png"
)


# ============================================================
# SETTINGS
# ============================================================

MOON_RADIUS_M = 1737400.0


# Keep coarse preview manageable.
MAX_PREVIEW_DIMENSION = 4000


# Add some margin around projected footprint.
WINDOW_PADDING_PIXELS = 100


# ============================================================
# LONGITUDE
# ============================================================

def longitude_180(
    longitude
):

    return (
        (
            longitude
            +
            180.0
        )
        %
        360.0
    ) - 180.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - IIRS/TMC CANDIDATE OVERLAP SCREEN"
    )

    print("=" * 80)


    if not TMC_PATH.exists():

        raise FileNotFoundError(
            f"TMC file not found:\n{TMC_PATH}"
        )


    # ========================================================
    # OPEN TMC
    # ========================================================

    with rasterio.open(
        TMC_PATH
    ) as dataset:

        print(
            "\nTMC:"
        )

        print(
            TMC_PATH
        )


        print(
            "\nTMC shape:",
            (
                dataset.height,
                dataset.width
            )
        )


        print(
            "TMC CRS:"
        )

        print(
            dataset.crs
        )


        print(
            "\nTMC bounds:"
        )

        print(
            dataset.bounds
        )


        print(
            "\nTMC resolution:"
        )

        print(
            dataset.res
        )


        # ====================================================
        # LUNAR GEOGRAPHIC CRS
        # ====================================================

        lunar_geographic = CRS.from_proj4(

            f"+proj=longlat "
            f"+R={MOON_RADIUS_M} "
            f"+no_defs"
        )


        transformer = Transformer.from_crs(

            lunar_geographic,

            dataset.crs,

            always_xy=True
        )


        # ====================================================
        # PROJECT IIRS CORNERS
        # ====================================================

        projected = []


        print("\n")
        print("=" * 80)
        print("PROJECTED IIRS FOOTPRINT")
        print("=" * 80)


        labels = [
            "UL",
            "UR",
            "LR",
            "LL"
        ]


        for (
            label,
            (
                longitude,
                latitude
            )

        ) in zip(
            labels,
            IIRS_CORNERS
        ):

            longitude = longitude_180(
                longitude
            )


            x, y = transformer.transform(

                longitude,
                latitude
            )


            projected.append(
                (
                    float(x),
                    float(y)
                )
            )


            print(
                f"{label}: "
                f"lon={longitude:.6f}, "
                f"lat={latitude:.6f} "
                f"-> "
                f"X={x:.3f}, "
                f"Y={y:.3f}"
            )


        polygon = Polygon(
            projected
        )


        if not polygon.is_valid:

            polygon = polygon.buffer(
                0
            )


        print(
            "\nProjected polygon valid:",
            polygon.is_valid
        )


        print(
            "Projected polygon area:",
            polygon.area,
            "m^2"
        )


        # ====================================================
        # INTERSECTION WITH TMC RECTANGLE
        # ====================================================

        tmc_polygon = Polygon(

            [
                (
                    dataset.bounds.left,
                    dataset.bounds.bottom
                ),

                (
                    dataset.bounds.right,
                    dataset.bounds.bottom
                ),

                (
                    dataset.bounds.right,
                    dataset.bounds.top
                ),

                (
                    dataset.bounds.left,
                    dataset.bounds.top
                )
            ]
        )


        rectangular_intersection = (
            polygon.intersection(
                tmc_polygon
            )
        )


        print("\n")
        print("=" * 80)
        print("RECTANGULAR INTERSECTION")
        print("=" * 80)


        print(
            "Intersects TMC raster bounds:",
            not rectangular_intersection.is_empty
        )


        if rectangular_intersection.is_empty:

            print(
                "\nRESULT: REJECT"
            )

            print(
                "IIRS footprint does not intersect "
                "the TMC raster rectangle."
            )

            return


        print(
            "Intersection area:",
            rectangular_intersection.area,
            "m^2"
        )


        # ====================================================
        # CONVERT INTERSECTION BOUNDS TO TMC WINDOW
        # ====================================================

        (
            min_x,
            min_y,
            max_x,
            max_y

        ) = rectangular_intersection.bounds


        row_a, col_a = rowcol(

            dataset.transform,

            min_x,
            max_y
        )


        row_b, col_b = rowcol(

            dataset.transform,

            max_x,
            min_y
        )


        row0 = max(
            0,
            min(
                row_a,
                row_b
            )
            -
            WINDOW_PADDING_PIXELS
        )


        row1 = min(
            dataset.height,
            max(
                row_a,
                row_b
            )
            +
            WINDOW_PADDING_PIXELS
        )


        col0 = max(
            0,
            min(
                col_a,
                col_b
            )
            -
            WINDOW_PADDING_PIXELS
        )


        col1 = min(
            dataset.width,
            max(
                col_a,
                col_b
            )
            +
            WINDOW_PADDING_PIXELS
        )


        window_height = (
            row1 - row0
        )


        window_width = (
            col1 - col0
        )


        print(
            "\nCandidate TMC window:"
        )

        print(
            f"rows {row0} -> {row1}"
        )

        print(
            f"cols {col0} -> {col1}"
        )

        print(
            "Native window shape:",
            (
                window_height,
                window_width
            )
        )


        if (
            window_height <= 0
            or
            window_width <= 0
        ):

            raise RuntimeError(
                "Invalid candidate TMC window."
            )


        # ====================================================
        # COARSE READ
        # ====================================================

        scale = min(

            1.0,

            MAX_PREVIEW_DIMENSION
            /
            max(
                window_width,
                window_height
            )
        )


        out_width = max(
            1,
            int(
                round(
                    window_width
                    *
                    scale
                )
            )
        )


        out_height = max(
            1,
            int(
                round(
                    window_height
                    *
                    scale
                )
            )
        )


        window = Window(

            col0,

            row0,

            window_width,

            window_height
        )


        tmc = dataset.read(

            1,

            window=
                window,

            out_shape=
                (
                    out_height,
                    out_width
                ),

            resampling=
                rasterio.enums.Resampling.average,

            out_dtype=
                "float32"
        )


        # Approximate transform corresponding to
        # downsampled window.

        window_transform = (
            dataset.window_transform(
                window
            )
        )


        preview_transform = (
            window_transform

            *
            rasterio.Affine.scale(

                window_width
                /
                out_width,

                window_height
                /
                out_height
            )
        )


        print(
            "\nPreview shape:",
            tmc.shape
        )


        # ====================================================
        # TMC VALIDITY
        # ====================================================

        tmc_valid = (
            np.isfinite(
                tmc
            )
            &
            (
                tmc > 0
            )
        )


        print(
            "TMC valid preview ratio:",
            float(
                tmc_valid.mean()
            )
        )


        # ====================================================
        # RASTERIZE IIRS FOOTPRINT
        # ====================================================

        iirs_mask = rasterize(

            [
                (
                    polygon,
                    1
                )
            ],

            out_shape=
                tmc.shape,

            transform=
                preview_transform,

            fill=
                0,

            dtype=
                np.uint8
        ).astype(
            bool
        )


        # ====================================================
        # EXACT QUESTION:
        #
        # Does the projected IIRS footprint contain any
        # non-zero TMC pixels?
        # ====================================================

        common = (
            iirs_mask
            &
            tmc_valid
        )


        iirs_pixels = int(
            np.count_nonzero(
                iirs_mask
            )
        )


        common_pixels = int(
            np.count_nonzero(
                common
            )
        )


        if iirs_pixels > 0:

            overlap_ratio = (
                common_pixels
                /
                iirs_pixels
            )

        else:

            overlap_ratio = 0.0


        print("\n")
        print("=" * 80)
        print("VALID-SWATH INTERSECTION")
        print("=" * 80)


        print(
            "IIRS footprint preview pixels:",
            iirs_pixels
        )


        print(
            "IIRS pixels hitting valid TMC:",
            common_pixels
        )


        print(
            "Approx IIRS-footprint valid-TMC ratio:",
            overlap_ratio
        )


        # ====================================================
        # VISUALIZATION
        # ====================================================

        values = tmc[
            tmc_valid
        ]


        preview = np.zeros(
            tmc.shape,
            dtype=np.uint8
        )


        if values.size > 0:

            low, high = np.percentile(
                values,
                [
                    1,
                    99
                ]
            )


            if high > low:

                normalized = np.clip(

                    (
                        tmc
                        -
                        low
                    )

                    /
                    (
                        high
                        -
                        low
                    ),

                    0,
                    1
                )


                preview[
                    tmc_valid
                ] = (

                    normalized[
                        tmc_valid
                    ]

                    *
                    255
                ).astype(
                    np.uint8
                )


        overlay = cv2.cvtColor(

            preview,

            cv2.COLOR_GRAY2BGR
        )


        # Blue = entire approximate IIRS footprint.

        overlay[
            iirs_mask
        ] = (

            0.65
            *
            overlay[
                iirs_mask
            ]

            +

            0.35
            *
            np.array(
                [
                    255,
                    0,
                    0
                ]
            )

        ).astype(
            np.uint8
        )


        # Green = actual valid TMC under IIRS footprint.

        overlay[
            common
        ] = np.array(
            [
                0,
                255,
                0
            ],
            dtype=np.uint8
        )


        cv2.imwrite(

            str(
                OVERLAY_OUTPUT
            ),

            overlay
        )


        cv2.imwrite(

            str(
                MASK_OUTPUT
            ),

            (
                iirs_mask.astype(
                    np.uint8
                )
                *
                255
            )
        )


        # ====================================================
        # DECISION
        # ====================================================

        print("\n")
        print("=" * 80)


        if common_pixels > 0:

            print(
                "RESULT: POTENTIAL IIRS/TMC OVERLAP FOUND"
            )

            print()
            print(
                "This candidate is worth downloading."
            )

        else:

            print(
                "RESULT: NO VALID TMC OVERLAP"
            )

            print()
            print(
                "Reject this IIRS candidate "
                "without downloading the QUB."
            )


        print("=" * 80)


        print(
            "\nOverlay:"
        )

        print(
            OVERLAY_OUTPUT
        )


if __name__ == "__main__":

    main()