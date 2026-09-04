from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np

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

SOURCE_PATH = (
    PAIR_DIR
    / "source.png"
)

REFERENCE_PATH = (
    PAIR_DIR
    / "reference.png"
)

MASK_PATH = (
    PAIR_DIR
    / "valid_mask.png"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_003"
    / "subpixel_refinement"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_JSON = (
    OUTPUT_DIR
    / "pair003_subpixel_refinement.json"
)

MATCH_CSV = (
    OUTPUT_DIR
    / "pair003_refined_matches.csv"
)

INITIAL_VIS = (
    OUTPUT_DIR
    / "initial_lightglue_inliers.png"
)

REFINED_VIS = (
    OUTPUT_DIR
    / "refined_inliers.png"
)

UNIFORM_VIS = (
    OUTPUT_DIR
    / "refined_uniform_inliers.png"
)


# ============================================================
# SETTINGS
# ============================================================

TEMPLATE_RADIUS = 11

SEARCH_RADIUS = 3

MIN_NCC_SCORE = 0.25

MIN_PEAK_MARGIN = 0.008

MAX_CORRECTION_PX = 3.5


INITIAL_RANSAC_THRESHOLD = 2.5

REFINED_RANSAC_THRESHOLD = 1.5

RANSAC_MAX_ITERS = 30000

RANSAC_CONFIDENCE = 0.999


# ============================================================
# QUADRATIC SUBPIXEL PEAK
# ============================================================

def quadratic_peak_offset(
    left,
    center,
    right
):

    denominator = (
        left
        -
        2.0 * center
        +
        right
    )

    if abs(
        denominator
    ) < 1e-9:

        return 0.0

    offset = (
        0.5
        *
        (
            left
            -
            right
        )
        /
        denominator
    )

    return float(
        np.clip(
            offset,
            -1.0,
            1.0
        )
    )


# ============================================================
# LOCAL NCC REFINEMENT
# ============================================================

def refine_one_match(
    source,
    reference,
    source_point,
    reference_point,
    distance_mask
):

    sx = int(
        round(
            source_point[0]
        )
    )

    sy = int(
        round(
            source_point[1]
        )
    )

    rx = int(
        round(
            reference_point[0]
        )
    )

    ry = int(
        round(
            reference_point[1]
        )
    )


    required_distance = (
        TEMPLATE_RADIUS
        +
        SEARCH_RADIUS
        +
        2
    )


    h, w = source.shape


    if (
        sx < 0
        or
        sx >= w
        or
        sy < 0
        or
        sy >= h
        or
        rx < 0
        or
        rx >= w
        or
        ry < 0
        or
        ry >= h
    ):

        return None


    if (
        distance_mask[
            sy,
            sx
        ]
        <
        required_distance

        or

        distance_mask[
            ry,
            rx
        ]
        <
        required_distance
    ):

        return None


    r = TEMPLATE_RADIUS
    s = SEARCH_RADIUS


    template = source[
        sy-r:sy+r+1,
        sx-r:sx+r+1
    ]


    search = reference[
        ry-r-s:ry+r+s+1,
        rx-r-s:rx+r+s+1
    ]


    expected_template = (
        2 * r + 1,
        2 * r + 1
    )

    expected_search = (
        2 * r + 1 + 2 * s,
        2 * r + 1 + 2 * s
    )


    if (
        template.shape
        !=
        expected_template
        or
        search.shape
        !=
        expected_search
    ):

        return None


    response = cv2.matchTemplate(
        search,
        template,
        cv2.TM_CCOEFF_NORMED
    )


    _, best_score, _, best_location = (
        cv2.minMaxLoc(
            response
        )
    )


    best_x = int(
        best_location[0]
    )

    best_y = int(
        best_location[1]
    )


    # ========================================================
    # SECOND INDEPENDENT PEAK
    # ========================================================

    second = response.copy()


    y0 = max(
        0,
        best_y - 1
    )

    y1 = min(
        second.shape[0],
        best_y + 2
    )

    x0 = max(
        0,
        best_x - 1
    )

    x1 = min(
        second.shape[1],
        best_x + 2
    )


    second[
        y0:y1,
        x0:x1
    ] = -np.inf


    second_score = float(
        np.max(
            second
        )
    )


    peak_margin = (
        float(
            best_score
        )
        -
        second_score
    )


    if (
        best_score
        <
        MIN_NCC_SCORE
    ):

        return None


    if (
        peak_margin
        <
        MIN_PEAK_MARGIN
    ):

        return None


    # ========================================================
    # SUBPIXEL QUADRATIC INTERPOLATION
    # ========================================================

    subpixel_x = 0.0

    subpixel_y = 0.0


    if (
        best_x > 0
        and
        best_x
        <
        response.shape[1] - 1
    ):

        subpixel_x = (
            quadratic_peak_offset(
                float(
                    response[
                        best_y,
                        best_x - 1
                    ]
                ),
                float(
                    response[
                        best_y,
                        best_x
                    ]
                ),
                float(
                    response[
                        best_y,
                        best_x + 1
                    ]
                )
            )
        )


    if (
        best_y > 0
        and
        best_y
        <
        response.shape[0] - 1
    ):

        subpixel_y = (
            quadratic_peak_offset(
                float(
                    response[
                        best_y - 1,
                        best_x
                    ]
                ),
                float(
                    response[
                        best_y,
                        best_x
                    ]
                ),
                float(
                    response[
                        best_y + 1,
                        best_x
                    ]
                )
            )
        )


    correction_x = (
        best_x
        -
        SEARCH_RADIUS
        +
        subpixel_x
    )


    correction_y = (
        best_y
        -
        SEARCH_RADIUS
        +
        subpixel_y
    )


    correction_norm = math.sqrt(
        correction_x
        *
        correction_x
        +
        correction_y
        *
        correction_y
    )


    if (
        correction_norm
        >
        MAX_CORRECTION_PX
    ):

        return None


    refined_reference = np.array(
        [
            reference_point[0]
            +
            correction_x,

            reference_point[1]
            +
            correction_y
        ],
        dtype=np.float32
    )


    return {

        "point":
            refined_reference,

        "score":
            float(
                best_score
            ),

        "second_score":
            second_score,

        "peak_margin":
            float(
                peak_margin
            ),

        "correction_x":
            float(
                correction_x
            ),

        "correction_y":
            float(
                correction_y
            ),

        "correction_norm":
            float(
                correction_norm
            )
    }


# ============================================================
# AFFINE
# ============================================================

def fit_affine(
    points0,
    points1,
    threshold
):

    matrix, inlier_mask = (
        cv2.estimateAffine2D(
            points0.astype(
                np.float32
            ),
            points1.astype(
                np.float32
            ),
            method=cv2.RANSAC,
            ransacReprojThreshold=
                threshold,
            maxIters=
                RANSAC_MAX_ITERS,
            confidence=
                RANSAC_CONFIDENCE,
            refineIters=25
        )
    )


    if (
        matrix is None
        or
        inlier_mask is None
    ):

        return None


    inlier_mask = (
        inlier_mask.ravel()
        >
        0
    )


    if (
        np.count_nonzero(
            inlier_mask
        )
        <
        3
    ):

        return None


    inlier0 = points0[
        inlier_mask
    ]

    inlier1 = points1[
        inlier_mask
    ]


    predicted = cv2.transform(
        inlier0.reshape(
            -1,
            1,
            2
        ),
        matrix
    ).reshape(
        -1,
        2
    )


    residuals = np.linalg.norm(
        predicted
        -
        inlier1,
        axis=1
    )


    return {

        "matrix":
            matrix,

        "mask":
            inlier_mask,

        "points0":
            inlier0,

        "points1":
            inlier1,

        "residuals":
            residuals,

        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        residuals
                        **
                        2
                    )
                )
            ),

        "median":
            float(
                np.median(
                    residuals
                )
            ),

        "mean":
            float(
                np.mean(
                    residuals
                )
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 003 "
        "LIGHTGLUE SUBPIXEL REFINEMENT"
    )

    print("=" * 80)


    source = bench.load_gray(
        SOURCE_PATH
    )

    reference = bench.load_gray(
        REFERENCE_PATH
    )

    mask_image = bench.load_gray(
        MASK_PATH
    )


    valid_mask = (
        mask_image > 0
    )


    safe_mask = bench.make_safe_mask(
        mask_image
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
    # REPRODUCE GLOBAL LIGHTGLUE MATCHES
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
        candidate0,
        candidate1,
        candidate_scores,
        matcher_metadata

    ) = bench.match_lightglue(
        source_gradient,
        reference_gradient,
        safe_mask,
        extractor,
        matcher
    )


    print(
        "\nGlobal candidates:",
        len(
            candidate0
        )
    )


    # ========================================================
    # INITIAL RANSAC
    # ========================================================

    initial = fit_affine(
        candidate0,
        candidate1,
        INITIAL_RANSAC_THRESHOLD
    )


    if initial is None:

        raise RuntimeError(
            "Initial LightGlue affine failed."
        )


    initial_parameters = (
        bench.affine_parameters(
            initial[
                "matrix"
            ]
        )
    )


    initial_sane = (
        bench.transform_sanity(
            initial_parameters
        )
    )


    initial_spatial = (
        bench.spatial_metrics(
            initial[
                "points0"
            ],
            safe_mask
        )
    )


    print("\n")
    print("=" * 80)

    print(
        "INITIAL LIGHTGLUE-GRADIENT RESULT"
    )

    print("=" * 80)


    print(
        "Inliers:",
        len(
            initial[
                "points0"
            ]
        )
    )

    print(
        "RMSE:",
        initial[
            "rmse"
        ],
        "px"
    )

    print(
        "Median:",
        initial[
            "median"
        ],
        "px"
    )

    print(
        "Coverage:",
        initial_spatial[
            "coverage"
        ]
    )

    print(
        "Affine:"
    )

    print(
        initial[
            "matrix"
        ]
    )

    print(
        "Transform sanity:",
        initial_sane
    )


    bench.save_match_visualization(
        source,
        reference,
        initial[
            "points0"
        ],
        initial[
            "points1"
        ],
        INITIAL_VIS,
        max_draw=100
    )


    if not initial_sane:

        raise RuntimeError(
            "Reproduced global result is not sane."
        )


    # ========================================================
    # DISTANCE FROM MASK BOUNDARY
    # ========================================================

    distance = cv2.distanceTransform(
        valid_mask.astype(
            np.uint8
        ),
        cv2.DIST_L2,
        5
    )


    # ========================================================
    # LOCAL REFINEMENT
    # ========================================================

    refined_source = []

    refined_reference = []

    refined_quality = []

    refinement_records = []


    print("\n")
    print("=" * 80)

    print(
        "LOCAL SUBPIXEL REFINEMENT"
    )

    print("=" * 80)


    for index, (
        point0,
        point1

    ) in enumerate(
        zip(
            initial[
                "points0"
            ],
            initial[
                "points1"
            ]
        )
    ):

        refined = refine_one_match(

            source_gradient,
            reference_gradient,

            point0,
            point1,

            distance
        )


        if refined is None:

            continue


        refined_source.append(
            point0
        )


        refined_reference.append(
            refined[
                "point"
            ]
        )


        quality = (

            refined[
                "score"
            ]

            *

            (
                1.0
                +
                10.0
                *
                refined[
                    "peak_margin"
                ]
            )
        )


        refined_quality.append(
            quality
        )


        refinement_records.append(
            {
                "original_index":
                    index,

                **{
                    key:
                        value

                    for key, value
                    in refined.items()

                    if key
                    !=
                    "point"
                }
            }
        )


    refined_source = np.asarray(
        refined_source,
        dtype=np.float32
    )


    refined_reference = np.asarray(
        refined_reference,
        dtype=np.float32
    )


    refined_quality = np.asarray(
        refined_quality,
        dtype=np.float32
    )


    print(
        "Input LightGlue inliers:",
        len(
            initial[
                "points0"
            ]
        )
    )


    print(
        "Locally accepted:",
        len(
            refined_source
        )
    )


    if (
        len(
            refined_source
        )
        <
        3
    ):

        raise RuntimeError(
            "Too few locally refined correspondences."
        )


    # ========================================================
    # FINAL RANSAC
    # ========================================================

    final = fit_affine(
        refined_source,
        refined_reference,
        REFINED_RANSAC_THRESHOLD
    )


    if final is None:

        raise RuntimeError(
            "Final refined affine failed."
        )


    final_parameters = (
        bench.affine_parameters(
            final[
                "matrix"
            ]
        )
    )


    final_sane = (
        bench.transform_sanity(
            final_parameters
        )
    )


    final_spatial = (
        bench.spatial_metrics(
            final[
                "points0"
            ],
            safe_mask
        )
    )


    accepted_quality = (
        refined_quality[
            final[
                "mask"
            ]
        ]
    )


    (
        uniform0,
        uniform1,
        uniform_scores

    ) = bench.uniform_select(

        final[
            "points0"
        ],

        final[
            "points1"
        ],

        accepted_quality,

        safe_mask
    )


    print("\n")
    print("=" * 80)

    print(
        "REFINED PAIR 003 RESULT"
    )

    print("=" * 80)


    print(
        "Final RANSAC inliers:",
        len(
            final[
                "points0"
            ]
        )
    )


    print(
        "RMSE:",
        final[
            "rmse"
        ],
        "px"
    )


    print(
        "Median:",
        final[
            "median"
        ],
        "px"
    )


    print(
        "Mean:",
        final[
            "mean"
        ],
        "px"
    )


    print(
        "Coverage:",
        final_spatial[
            "coverage"
        ]
    )


    print(
        "Occupied valid cells:",
        final_spatial[
            "occupied_valid_cells"
        ],
        "/",
        final_spatial[
            "valid_cells"
        ]
    )


    print(
        "Uniform inliers:",
        len(
            uniform0
        )
    )


    print(
        "Affine:"
    )

    print(
        final[
            "matrix"
        ]
    )


    print(
        "Transform:"
    )

    print(
        final_parameters
    )


    print(
        "Transform sanity:",
        final_sane
    )


    # ========================================================
    # ACCEPT / REJECT REFINEMENT
    # ========================================================

    minimum_inliers = max(
        6,
        int(
            math.ceil(
                0.5
                *
                len(
                    initial[
                        "points0"
                    ]
                )
            )
        )
    )


    refinement_accepted = bool(

        final_sane

        and

        len(
            final[
                "points0"
            ]
        )
        >=
        minimum_inliers

        and

        final[
            "rmse"
        ]
        <
        initial[
            "rmse"
        ]

        and

        final_spatial[
            "coverage"
        ]
        >=
        0.20
    )


    print("\n")
    print("=" * 80)

    print(
        "REFINEMENT DECISION"
    )

    print("=" * 80)


    print(
        "Accepted:",
        refinement_accepted
    )


    if refinement_accepted:

        print(
            "Use the refined correspondence set "
            "as the Pair 003 high-precision product."
        )

    else:

        print(
            "Keep the original global "
            "LightGlue-gradient result."
        )


    # ========================================================
    # VISUALS
    # ========================================================

    bench.save_match_visualization(
        source,
        reference,
        final[
            "points0"
        ],
        final[
            "points1"
        ],
        REFINED_VIS,
        max_draw=100
    )


    bench.save_match_visualization(
        source,
        reference,
        uniform0,
        uniform1,
        UNIFORM_VIS,
        max_draw=100
    )


    # ========================================================
    # CSV
    # ========================================================

    with MATCH_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "source_x",
                "source_y",
                "reference_x",
                "reference_y",
                "final_residual_px"
            ]
        )


        for index in range(
            len(
                final[
                    "points0"
                ]
            )
        ):

            writer.writerow(
                [
                    float(
                        final[
                            "points0"
                        ][
                            index,
                            0
                        ]
                    ),

                    float(
                        final[
                            "points0"
                        ][
                            index,
                            1
                        ]
                    ),

                    float(
                        final[
                            "points1"
                        ][
                            index,
                            0
                        ]
                    ),

                    float(
                        final[
                            "points1"
                        ][
                            index,
                            1
                        ]
                    ),

                    float(
                        final[
                            "residuals"
                        ][
                            index
                        ]
                    )
                ]
            )


    # ========================================================
    # JSON
    # ========================================================

    payload = {

        "pair_id":
            "pair_003",

        "method":
            (
                "global_lightglue_gradient_"
                "plus_local_ncc_subpixel_refinement"
            ),

        "initial": {

            "candidate_matches":
                int(
                    len(
                        candidate0
                    )
                ),

            "inliers":
                int(
                    len(
                        initial[
                            "points0"
                        ]
                    )
                ),

            "rmse_px":
                initial[
                    "rmse"
                ],

            "median_px":
                initial[
                    "median"
                ],

            "coverage":
                initial_spatial[
                    "coverage"
                ],

            "affine_matrix":
                initial[
                    "matrix"
                ].tolist(),

            "transform":
                initial_parameters,

            "transform_sanity":
                initial_sane
        },

        "refinement": {

            "template_size":
                2 * TEMPLATE_RADIUS + 1,

            "search_radius_px":
                SEARCH_RADIUS,

            "minimum_ncc_score":
                MIN_NCC_SCORE,

            "minimum_peak_margin":
                MIN_PEAK_MARGIN,

            "accepted_local_refinements":
                int(
                    len(
                        refined_source
                    )
                )
        },

        "final": {

            "inliers":
                int(
                    len(
                        final[
                            "points0"
                        ]
                    )
                ),

            "rmse_px":
                final[
                    "rmse"
                ],

            "median_px":
                final[
                    "median"
                ],

            "mean_px":
                final[
                    "mean"
                ],

            "coverage":
                final_spatial[
                    "coverage"
                ],

            "uniform_inliers":
                int(
                    len(
                        uniform0
                    )
                ),

            "affine_matrix":
                final[
                    "matrix"
                ].tolist(),

            "transform":
                final_parameters,

            "transform_sanity":
                final_sane
        },

        "refinement_accepted":
            refinement_accepted,

        "metric_note":
            (
                "RMSE values are correspondence "
                "self-consistency residuals. "
                "They are not independent "
                "ground-truth registration accuracy."
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
        "PAIR 003 SUBPIXEL REFINEMENT COMPLETE"
    )

    print("=" * 80)


    print(
        "\nJSON:"
    )

    print(
        RESULT_JSON
    )


    print(
        "\nMatches:"
    )

    print(
        MATCH_CSV
    )


    print(
        "\nVisualization:"
    )

    print(
        REFINED_VIS
    )


if __name__ == "__main__":

    main()