from pathlib import Path
import csv
import json

import cv2
import numpy as np
import rasterio
import torch

import benchmark_pair003_matchers as bench


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_003"
)

SOURCE_PNG = (
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

REFERENCE_TIF = (
    PAIR_DIR
    / "reference_tmc2_on_common_grid.tif"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_003"
    / "final_product"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


REGISTERED_TIF = (
    OUTPUT_DIR
    / "registered_iirs_to_tmc2.tif"
)

REGISTERED_PNG = (
    OUTPUT_DIR
    / "registered_iirs.png"
)

REGISTERED_MASK_TIF = (
    OUTPUT_DIR
    / "registered_valid_mask.tif"
)

OVERLAY_PNG = (
    OUTPUT_DIR
    / "registered_overlay_50_50.png"
)

CHECKERBOARD_PNG = (
    OUTPUT_DIR
    / "registered_checkerboard.png"
)

DIFFERENCE_PNG = (
    OUTPUT_DIR
    / "registered_difference.png"
)

MATCHES_CSV = (
    OUTPUT_DIR
    / "pair003_final_match_points.csv"
)

MATCH_VIS = (
    OUTPUT_DIR
    / "pair003_final_matches.png"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "pair003_final_metrics.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANSAC_THRESHOLD_PX = 2.5

RANSAC_MAX_ITERS = 10000

RANSAC_CONFIDENCE = 0.999

CHECKER_SIZE = 64


# ============================================================
# IMAGE STRETCH
# ============================================================

def stretch_float(
    image,
    mask
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

        return output


    low, high = np.percentile(
        values,
        [
            1,
            99
        ]
    )


    if high <= low:

        return output


    normalized = (
        image.astype(
            np.float32
        )
        -
        float(
            low
        )
    ) / (
        float(
            high
        )
        -
        float(
            low
        )
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
        255
    ).astype(
        np.uint8
    )


    return output


# ============================================================
# CHECKERBOARD
# ============================================================

def make_checkerboard(
    image0,
    image1,
    mask,
    block_size
):

    h, w = image0.shape

    yy, xx = np.indices(
        (
            h,
            w
        )
    )


    selector = (
        (
            xx
            //
            block_size
        )
        +
        (
            yy
            //
            block_size
        )
    ) % 2 == 0


    output = np.where(
        selector,
        image0,
        image1
    ).astype(
        np.uint8
    )


    output[
        ~mask
    ] = 0


    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - FINALIZE PAIR 003 PRODUCT"
    )

    print("=" * 80)


    for path in [

        SOURCE_PNG,
        REFERENCE_PNG,
        MASK_PNG,

        SOURCE_TIF,
        REFERENCE_TIF

    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing:\n{path}"
            )


    # ========================================================
    # LOAD MATCHING IMAGES
    # ========================================================

    source = bench.load_gray(
        SOURCE_PNG
    )

    reference = bench.load_gray(
        REFERENCE_PNG
    )

    mask_image = bench.load_gray(
        MASK_PNG
    )


    valid_mask = (
        mask_image > 0
    )


    safe_mask = bench.make_safe_mask(
        mask_image
    )


    print(
        "\nShape:",
        source.shape
    )


    print(
        "Valid pixels:",
        int(
            np.count_nonzero(
                valid_mask
            )
        )
    )


    # ========================================================
    # GRADIENT REPRESENTATION
    # ========================================================

    source_gradient = (
        bench.make_gradient_image(
            source,
            valid_mask
        )
    )


    reference_gradient = (
        bench.make_gradient_image(
            reference,
            valid_mask
        )
    )


    # ========================================================
    # LIGHTGLUE
    # ========================================================

    print(
        "\nLoading LightGlue..."
    )


    extractor, matcher = (
        bench.create_lightglue()
    )


    print(
        "LightGlue ready."
    )


    (
        points0,
        points1,
        scores,
        matcher_metadata

    ) = bench.match_lightglue(

        source_gradient,
        reference_gradient,

        safe_mask,

        extractor,
        matcher
    )


    print(
        "\nCandidate matches:",
        len(
            points0
        )
    )


    # ========================================================
    # FINAL RANSAC
    # ========================================================

    matrix, inlier_mask = (
        cv2.estimateAffine2D(

            points0.astype(
                np.float32
            ),

            points1.astype(
                np.float32
            ),

            method=
                cv2.RANSAC,

            ransacReprojThreshold=
                RANSAC_THRESHOLD_PX,

            maxIters=
                RANSAC_MAX_ITERS,

            confidence=
                RANSAC_CONFIDENCE,

            refineIters=
                20
        )
    )


    if (
        matrix is None
        or
        inlier_mask is None
    ):

        raise RuntimeError(
            "Final Pair 003 RANSAC failed."
        )


    inlier_mask = (
        inlier_mask.ravel()
        >
        0
    )


    inlier0 = points0[
        inlier_mask
    ]

    inlier1 = points1[
        inlier_mask
    ]

    inlier_scores = scores[
        inlier_mask
    ]


    prediction = cv2.transform(

        inlier0.reshape(
            -1,
            1,
            2
        ).astype(
            np.float32
        ),

        matrix

    ).reshape(
        -1,
        2
    )


    residuals = np.linalg.norm(
        prediction
        -
        inlier1,
        axis=1
    )


    rmse = float(
        np.sqrt(
            np.mean(
                residuals
                **
                2
            )
        )
    )


    mean_residual = float(
        np.mean(
            residuals
        )
    )


    median_residual = float(
        np.median(
            residuals
        )
    )


    parameters = bench.affine_parameters(
        matrix
    )


    sane = bench.transform_sanity(
        parameters
    )


    spatial = bench.spatial_metrics(
        inlier0,
        safe_mask
    )


    (
        uniform0,
        uniform1,
        uniform_scores

    ) = bench.uniform_select(

        inlier0,
        inlier1,
        inlier_scores,
        safe_mask
    )


    print("\n")
    print("=" * 80)

    print(
        "FINAL CORRESPONDENCE SET"
    )

    print("=" * 80)


    print(
        "Candidates:",
        len(
            points0
        )
    )


    print(
        "Inliers:",
        len(
            inlier0
        )
    )


    print(
        "Inlier ratio:",
        len(
            inlier0
        )
        /
        len(
            points0
        )
    )


    print(
        "RMSE:",
        rmse,
        "px"
    )


    print(
        "Median:",
        median_residual,
        "px"
    )


    print(
        "Mean:",
        mean_residual,
        "px"
    )


    print(
        "Coverage:",
        spatial[
            "coverage"
        ]
    )


    print(
        "Occupied valid cells:",
        spatial[
            "occupied_valid_cells"
        ],
        "/",
        spatial[
            "valid_cells"
        ]
    )


    print(
        "Uniform matches:",
        len(
            uniform0
        )
    )


    print(
        "Affine:"
    )

    print(
        matrix
    )


    print(
        "Transform:"
    )

    print(
        parameters
    )


    print(
        "Transform sanity:",
        sane
    )


    if not sane:

        raise RuntimeError(
            "Final transform failed sanity check."
        )


    # ========================================================
    # SAVE MATCH POINTS
    # ========================================================

    with MATCHES_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "source_x_px",
                "source_y_px",

                "reference_x_px",
                "reference_y_px",

                "confidence",

                "ransac_residual_px"
            ]
        )


        for index in range(
            len(
                inlier0
            )
        ):

            writer.writerow(
                [
                    float(
                        inlier0[
                            index,
                            0
                        ]
                    ),

                    float(
                        inlier0[
                            index,
                            1
                        ]
                    ),

                    float(
                        inlier1[
                            index,
                            0
                        ]
                    ),

                    float(
                        inlier1[
                            index,
                            1
                        ]
                    ),

                    float(
                        inlier_scores[
                            index
                        ]
                    ),

                    float(
                        residuals[
                            index
                        ]
                    )
                ]
            )


    # ========================================================
    # MATCH VISUALIZATION
    # ========================================================

    bench.save_match_visualization(

        source,
        reference,

        uniform0,
        uniform1,

        MATCH_VIS,

        max_draw=100
    )


    # ========================================================
    # READ FLOAT GEOTIFFS
    # ========================================================

    with rasterio.open(
        SOURCE_TIF
    ) as src:

        source_float = src.read(
            1
        ).astype(
            np.float32
        )

        source_profile = (
            src.profile.copy()
        )

        source_transform = (
            src.transform
        )

        source_crs = (
            src.crs
        )


    with rasterio.open(
        REFERENCE_TIF
    ) as ref:

        reference_float = ref.read(
            1
        ).astype(
            np.float32
        )


    if (
        source_float.shape
        !=
        reference_float.shape
    ):

        raise RuntimeError(
            "GeoTIFF shapes do not match."
        )


    h, w = (
        source_float.shape
    )


    # ========================================================
    # REGISTER IIRS
    # ========================================================

    registered_float = cv2.warpAffine(

        source_float,

        matrix.astype(
            np.float64
        ),

        (
            w,
            h
        ),

        flags=
            cv2.INTER_LINEAR,

        borderMode=
            cv2.BORDER_CONSTANT,

        borderValue=
            0
    )


    registered_mask = (

        cv2.warpAffine(

            (
                valid_mask.astype(
                    np.uint8
                )
                *
                255
            ),

            matrix.astype(
                np.float64
            ),

            (
                w,
                h
            ),

            flags=
                cv2.INTER_NEAREST,

            borderMode=
                cv2.BORDER_CONSTANT,

            borderValue=
                0
        )

        >
        0
    )


    reference_valid = (
        np.isfinite(
            reference_float
        )
        &
        (
            reference_float > 0
        )
    )


    common_registered_mask = (
        registered_mask
        &
        reference_valid
    )


    registered_float[
        ~common_registered_mask
    ] = 0


    # ========================================================
    # WRITE REGISTERED GEOTIFF
    # ========================================================

    output_profile = (
        source_profile.copy()
    )


    output_profile.update(
        {
            "dtype":
                "float32",

            "count":
                1,

            "nodata":
                0,

            "compress":
                "deflate"
        }
    )


    with rasterio.open(
        REGISTERED_TIF,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            registered_float.astype(
                np.float32
            ),
            1
        )


        dst.update_tags(

            pair_id=
                "pair_003",

            source_sensor=
                "IIRS",

            reference_sensor=
                "TMC-2",

            registration_method=
                "SuperPoint+LightGlue gradient + affine RANSAC",

            correspondence_count=
                str(
                    len(
                        inlier0
                    )
                ),

            ransac_reprojection_rmse_px=
                str(
                    rmse
                )
        )


    # ========================================================
    # REGISTERED MASK GEOTIFF
    # ========================================================

    mask_profile = (
        output_profile.copy()
    )


    mask_profile.update(
        {
            "dtype":
                "uint8",

            "nodata":
                0
        }
    )


    with rasterio.open(
        REGISTERED_MASK_TIF,
        "w",
        **mask_profile
    ) as dst:

        dst.write(

            (
                common_registered_mask.astype(
                    np.uint8
                )
                *
                255
            ),

            1
        )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    registered_png = stretch_float(
        registered_float,
        common_registered_mask
    )


    reference_png = stretch_float(
        reference_float,
        common_registered_mask
    )


    cv2.imwrite(
        str(
            REGISTERED_PNG
        ),
        registered_png
    )


    overlay = cv2.addWeighted(
        registered_png,
        0.5,
        reference_png,
        0.5,
        0
    )


    overlay[
        ~common_registered_mask
    ] = 0


    cv2.imwrite(
        str(
            OVERLAY_PNG
        ),
        overlay
    )


    difference = cv2.absdiff(
        registered_png,
        reference_png
    )


    difference[
        ~common_registered_mask
    ] = 0


    cv2.imwrite(
        str(
            DIFFERENCE_PNG
        ),
        difference
    )


    checkerboard = make_checkerboard(

        registered_png,
        reference_png,

        common_registered_mask,

        CHECKER_SIZE
    )


    cv2.imwrite(
        str(
            CHECKERBOARD_PNG
        ),
        checkerboard
    )


    # ========================================================
    # METRICS
    # ========================================================

    resolution_m = float(
        abs(
            source_transform.a
        )
    )


    summary = {

        "pair_id":
            "pair_003",

        "source_sensor":
            "IIRS",

        "reference_sensor":
            "TMC-2",

        "source_product":
            (
                "ch2_iir_ndi_"
                "20240115T2100076733_"
                "d_rfl_d18_srd"
            ),

        "matching_representation":
            (
                "mean IIRS surface reflectance "
                "bands 1-9 (712.3-847.2 nm), "
                "gradient representation"
            ),

        "matcher":
            "SuperPoint + LightGlue",

        "geometry_model":
            "2D affine RANSAC",

        "candidate_matches":
            int(
                len(
                    points0
                )
            ),

        "inliers":
            int(
                len(
                    inlier0
                )
            ),

        "inlier_ratio":
            float(
                len(
                    inlier0
                )
                /
                len(
                    points0
                )
            ),

        "ransac_reprojection_rmse_px":
            rmse,

        "ransac_reprojection_median_px":
            median_residual,

        "ransac_reprojection_mean_px":
            mean_residual,

        "common_grid_resolution_m_per_px":
            resolution_m,

        "rmse_equivalent_grid_distance_m":
            float(
                rmse
                *
                resolution_m
            ),

        "spatial_coverage":
            spatial,

        "uniform_correspondences":
            int(
                len(
                    uniform0
                )
            ),

        "affine_matrix":
            matrix.tolist(),

        "transform":
            parameters,

        "transform_sanity":
            sane,

        "registered_common_valid_pixels":
            int(
                np.count_nonzero(
                    common_registered_mask
                )
            ),

        "registered_common_valid_ratio":
            float(
                common_registered_mask.mean()
            ),

        "subpixel_refinement": {

            "attempted":
                True,

            "accepted":
                False,

            "reason":
                (
                    "0 of 20 global LightGlue-gradient "
                    "inliers passed conservative local "
                    "NCC refinement gates."
                )
        },

        "important_metric_note":
            (
                "RANSAC reprojection RMSE and its "
                "equivalent grid distance measure "
                "internal correspondence self-consistency. "
                "They are NOT independent ground-truth "
                "registration accuracy."
            )
    }


    SUMMARY_JSON.write_text(

        json.dumps(
            summary,
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
        "PAIR 003 FINAL PRODUCT COMPLETE"
    )

    print("=" * 80)


    print(
        "\nRegistered IIRS:"
    )

    print(
        REGISTERED_TIF
    )


    print(
        "\nMatch points:"
    )

    print(
        MATCHES_CSV
    )


    print(
        "\nMetrics:"
    )

    print(
        SUMMARY_JSON
    )


    print(
        "\nOverlay:"
    )

    print(
        OVERLAY_PNG
    )


    print(
        "\nCheckerboard:"
    )

    print(
        CHECKERBOARD_PNG
    )


    print(
        "\nFinal correspondence visualization:"
    )

    print(
        MATCH_VIS
    )


if __name__ == "__main__":

    main()