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

GLOBAL_JSON = (
    ROOT
    / "results"
    / "pair_003"
    / "global_benchmark"
    / "pair003_global_benchmark.json"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_003"
    / "prior_guided_mind"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_JSON = (
    OUTPUT_DIR
    / "pair003_prior_guided_mind.json"
)

MATCH_CSV = (
    OUTPUT_DIR
    / "pair003_prior_guided_mind_matches.csv"
)

CANDIDATE_VIS = (
    OUTPUT_DIR
    / "candidate_matches.png"
)

INLIER_VIS = (
    OUTPUT_DIR
    / "ransac_inliers.png"
)

UNIFORM_VIS = (
    OUTPUT_DIR
    / "uniform_inliers.png"
)

WARPED_SOURCE_VIS = (
    OUTPUT_DIR
    / "source_warped_by_prior.png"
)

MIND_SOURCE_VIS = (
    OUTPUT_DIR
    / "mind_source_preview.png"
)

MIND_REFERENCE_VIS = (
    OUTPUT_DIR
    / "mind_reference_preview.png"
)

ANCHOR_VIS = (
    OUTPUT_DIR
    / "selected_anchors.png"
)


# ============================================================
# SETTINGS
# ============================================================

GRID_ROWS = 6
GRID_COLS = 12

ANCHORS_PER_CELL = 4

ANCHOR_MIN_SEPARATION_PX = 26

TEMPLATE_RADIUS = 14

SEARCH_RADIUS = 8

REVERSE_SEARCH_RADIUS = 3

BOUNDARY_MARGIN_PX = 8


# MIND descriptor

MIND_SHIFT = 3

MIND_PATCH_SIZE = 5

MIND_EPSILON = 1e-6


# Matching

MIN_DISTINCTNESS_RATIO = 1.015

MAX_REVERSE_ERROR_PX = 1.5


# RANSAC

RANSAC_THRESHOLD_PX = 2.5

RANSAC_MAX_ITERS = 30000

RANSAC_CONFIDENCE = 0.999


# ============================================================
# GLOBAL PRIOR
# ============================================================

def load_global_prior():

    payload = json.loads(
        GLOBAL_JSON.read_text(
            encoding="utf-8"
        )
    )

    for result in payload["results"]:

        if (
            result.get("matcher")
            ==
            "lightglue"

            and

            result.get("representation")
            ==
            "gradient"

            and

            result.get("status")
            ==
            "ok"

            and

            result.get("transform_sanity")
            is True
        ):

            return (
                np.asarray(
                    result[
                        "affine_matrix"
                    ],
                    dtype=np.float64
                ),
                result
            )

    raise RuntimeError(
        "Sane global LightGlue-gradient prior "
        "could not be found."
    )


# ============================================================
# AFFINE HELPERS
# ============================================================

def transform_points(
    points,
    matrix
):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    if len(points) == 0:

        return np.zeros(
            (0, 2),
            dtype=np.float32
        )

    homogeneous = np.column_stack(
        [
            points,
            np.ones(
                len(points),
                dtype=np.float64
            )
        ]
    )

    transformed = (
        homogeneous
        @
        matrix.T
    )

    return transformed.astype(
        np.float32
    )


def invert_affine(
    matrix
):

    full = np.eye(
        3,
        dtype=np.float64
    )

    full[
        :2,
        :
    ] = matrix

    inverse = np.linalg.inv(
        full
    )

    return inverse[
        :2,
        :
    ]


# ============================================================
# MASK
# ============================================================

def distance_from_invalid(
    mask
):

    binary = (
        mask.astype(
            np.uint8
        )
    )

    return cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5
    )


# ============================================================
# MIND DESCRIPTOR
# ============================================================

def shift_image(
    image,
    dx,
    dy
):

    # shifted[y,x] = image[y+dy,x+dx]

    return np.roll(
        image,
        shift=(
            -dy,
            -dx
        ),
        axis=(
            0,
            1
        )
    )


def compute_mind_descriptor(
    image,
    valid_mask
):

    print(
        "Computing MIND descriptor..."
    )

    image = (
        image.astype(
            np.float32
        )
        /
        255.0
    )

    image = cv2.GaussianBlur(
        image,
        (0, 0),
        0.8
    )


    # 8-neighbour self-similarity pattern.

    offsets = [

        (-MIND_SHIFT, 0),
        ( MIND_SHIFT, 0),

        (0, -MIND_SHIFT),
        (0,  MIND_SHIFT),

        (-MIND_SHIFT, -MIND_SHIFT),
        ( MIND_SHIFT, -MIND_SHIFT),

        (-MIND_SHIFT,  MIND_SHIFT),
        ( MIND_SHIFT,  MIND_SHIFT)
    ]


    channels = []


    for dx, dy in offsets:

        shifted = shift_image(
            image,
            dx,
            dy
        )

        difference = (
            image
            -
            shifted
        )

        squared = (
            difference
            *
            difference
        )


        # Patch SSD around each pixel.

        ssd = cv2.boxFilter(
            squared,
            cv2.CV_32F,
            (
                MIND_PATCH_SIZE,
                MIND_PATCH_SIZE
            ),
            normalize=True,
            borderType=cv2.BORDER_REFLECT101
        )


        channels.append(
            ssd
        )


    ssd_stack = np.stack(
        channels,
        axis=0
    )


    # Local normalization:
    # MIND encodes similarity relative to the
    # neighbourhood's own variation.

    variance = np.mean(
        ssd_stack,
        axis=0
    )


    valid_variance = variance[
        valid_mask
    ]


    positive_variance = valid_variance[
        valid_variance
        >
        0
    ]


    if positive_variance.size > 0:

        variance_floor = float(
            np.median(
                positive_variance
            )
            *
            0.05
        )

    else:

        variance_floor = (
            MIND_EPSILON
        )


    variance_floor = max(
        variance_floor,
        MIND_EPSILON
    )


    variance = np.maximum(
        variance,
        variance_floor
    )


    descriptor = np.exp(
        -
        ssd_stack
        /
        variance[
            None,
            :,
            :
        ]
    ).astype(
        np.float32
    )


    # Normalize descriptor vectors.

    norm = np.sqrt(
        np.sum(
            descriptor
            *
            descriptor,
            axis=0
        )
        +
        MIND_EPSILON
    )


    descriptor /= norm[
        None,
        :,
        :
    ]


    descriptor[
        :,
        ~valid_mask
    ] = 0


    return descriptor


# ============================================================
# MIND PREVIEW
# ============================================================

def descriptor_preview(
    descriptor,
    mask
):

    # Mean self-similarity response for visualization only.

    image = np.mean(
        descriptor,
        axis=0
    )


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
        1e-8
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
# GRID
# ============================================================

def cell_bounds(
    row,
    col,
    width,
    height
):

    x0 = int(
        col
        *
        width
        /
        GRID_COLS
    )

    x1 = int(
        (col + 1)
        *
        width
        /
        GRID_COLS
    )

    y0 = int(
        row
        *
        height
        /
        GRID_ROWS
    )

    y1 = int(
        (row + 1)
        *
        height
        /
        GRID_ROWS
    )

    return (
        x0,
        y0,
        x1,
        y1
    )


# ============================================================
# TEXTURE MAP
# ============================================================

def gradient_energy(
    image,
    valid_mask
):

    image = (
        image.astype(
            np.float32
        )
        /
        255.0
    )

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        1.0
    )


    gx = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )


    energy = (
        gx * gx
        +
        gy * gy
    )


    energy = cv2.boxFilter(
        energy,
        cv2.CV_32F,
        (
            21,
            21
        ),
        normalize=True
    )


    energy[
        ~valid_mask
    ] = 0


    return energy


# ============================================================
# ANCHOR SELECTION
# ============================================================

def select_anchors(
    common_valid,
    distance,
    source,
    reference
):

    height, width = (
        common_valid.shape
    )


    source_energy = gradient_energy(
        source,
        common_valid
    )


    reference_energy = gradient_energy(
        reference,
        common_valid
    )


    # Prefer locations textured in BOTH modalities.

    texture = np.sqrt(
        np.maximum(
            source_energy,
            0
        )
        *
        np.maximum(
            reference_energy,
            0
        )
    )


    # Enough support for:
    # MIND neighbourhood
    # + matching template
    # + local displacement search.

    required_distance = (
        TEMPLATE_RADIUS
        +
        SEARCH_RADIUS
        +
        MIND_SHIFT
        +
        MIND_PATCH_SIZE // 2
        +
        2
    )


    anchors = []

    cell_summary = []


    for row in range(
        GRID_ROWS
    ):

        for col in range(
            GRID_COLS
        ):

            (
                x0,
                y0,
                x1,
                y1

            ) = cell_bounds(
                row,
                col,
                width,
                height
            )


            candidate_mask = (

                common_valid[
                    y0:y1,
                    x0:x1
                ]

                &

                (
                    distance[
                        y0:y1,
                        x0:x1
                    ]
                    >=
                    required_distance
                )
            )


            ys, xs = np.where(
                candidate_mask
            )


            if len(xs) == 0:

                continue


            xs = (
                xs
                +
                x0
            )

            ys = (
                ys
                +
                y0
            )


            scores = texture[
                ys,
                xs
            ]


            order = np.argsort(
                -scores
            )


            selected = []


            for index in order:

                x = int(
                    xs[
                        index
                    ]
                )

                y = int(
                    ys[
                        index
                    ]
                )


                sufficiently_far = True


                for existing in selected:

                    dx = (
                        x
                        -
                        existing[
                            0
                        ]
                    )

                    dy = (
                        y
                        -
                        existing[
                            1
                        ]
                    )


                    if (
                        dx * dx
                        +
                        dy * dy
                        <
                        ANCHOR_MIN_SEPARATION_PX
                        *
                        ANCHOR_MIN_SEPARATION_PX
                    ):

                        sufficiently_far = False

                        break


                if not sufficiently_far:

                    continue


                selected.append(
                    (
                        x,
                        y,
                        float(
                            scores[
                                index
                            ]
                        )
                    )
                )


                if (
                    len(
                        selected
                    )
                    >=
                    ANCHORS_PER_CELL
                ):

                    break


            for x, y, score in selected:

                anchors.append(
                    {
                        "x":
                            x,

                        "y":
                            y,

                        "grid_row":
                            row,

                        "grid_col":
                            col,

                        "texture":
                            score
                    }
                )


            cell_summary.append(
                {
                    "row":
                        row,

                    "col":
                        col,

                    "anchors":
                        len(
                            selected
                        )
                }
            )


    return (
        anchors,
        cell_summary
    )


# ============================================================
# MIND LOCAL SEARCH
# ============================================================

def mind_search(
    template_descriptor,
    target_descriptor,
    template_center,
    search_center,
    search_radius
):

    tx = int(
        round(
            template_center[
                0
            ]
        )
    )

    ty = int(
        round(
            template_center[
                1
            ]
        )
    )


    sx = int(
        round(
            search_center[
                0
            ]
        )
    )

    sy = int(
        round(
            search_center[
                1
            ]
        )
    )


    r = (
        TEMPLATE_RADIUS
    )


    template = (
        template_descriptor[
            :,
            ty-r:ty+r+1,
            tx-r:tx+r+1
        ]
    )


    expected_size = (
        2 * r
        +
        1
    )


    if (
        template.shape[
            1
        ]
        !=
        expected_size

        or

        template.shape[
            2
        ]
        !=
        expected_size
    ):

        return None


    target_region = (
        target_descriptor[
            :,
            sy-r-search_radius:
            sy+r+search_radius+1,

            sx-r-search_radius:
            sx+r+search_radius+1
        ]
    )


    expected_region = (
        expected_size
        +
        2
        *
        search_radius
    )


    if (
        target_region.shape[
            1
        ]
        !=
        expected_region

        or

        target_region.shape[
            2
        ]
        !=
        expected_region
    ):

        return None


    # cv2.matchTemplate performs the spatial search
    # efficiently. Sum descriptor-channel SSDs.

    cost_surface = None


    for channel in range(
        template.shape[
            0
        ]
    ):

        channel_cost = cv2.matchTemplate(
            target_region[
                channel
            ],
            template[
                channel
            ],
            cv2.TM_SQDIFF
        )


        if cost_surface is None:

            cost_surface = channel_cost

        else:

            cost_surface += channel_cost


    # Mean squared descriptor difference.

    normalization = float(
        template.shape[
            0
        ]
        *
        template.shape[
            1
        ]
        *
        template.shape[
            2
        ]
    )


    cost_surface /= (
        normalization
    )


    flat_index = int(
        np.argmin(
            cost_surface
        )
    )


    best_y, best_x = np.unravel_index(
        flat_index,
        cost_surface.shape
    )


    best_cost = float(
        cost_surface[
            best_y,
            best_x
        ]
    )


    # --------------------------------------------
    # Second independent local minimum.
    #
    # Ignore 3x3 region around best solution.
    # --------------------------------------------

    second_surface = (
        cost_surface.copy()
    )


    y0 = max(
        0,
        best_y - 1
    )

    y1 = min(
        second_surface.shape[
            0
        ],
        best_y + 2
    )


    x0 = max(
        0,
        best_x - 1
    )

    x1 = min(
        second_surface.shape[
            1
        ],
        best_x + 2
    )


    second_surface[
        y0:y1,
        x0:x1
    ] = np.inf


    second_cost = float(
        np.min(
            second_surface
        )
    )


    distinctness_ratio = (
        second_cost
        /
        (
            best_cost
            +
            1e-12
        )
    )


    displacement_x = (
        best_x
        -
        search_radius
    )


    displacement_y = (
        best_y
        -
        search_radius
    )


    matched_x = (
        sx
        +
        displacement_x
    )


    matched_y = (
        sy
        +
        displacement_y
    )


    return {
        "x":
            int(
                matched_x
            ),

        "y":
            int(
                matched_y
            ),

        "dx":
            int(
                displacement_x
            ),

        "dy":
            int(
                displacement_y
            ),

        "best_cost":
            best_cost,

        "second_cost":
            second_cost,

        "distinctness_ratio":
            float(
                distinctness_ratio
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 003 "
        "PRIOR-GUIDED MIND MATCHING"
    )

    print("=" * 80)


    for path in [

        SOURCE_PATH,
        REFERENCE_PATH,
        MASK_PATH,
        GLOBAL_JSON

    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing:\n{path}"
            )


    # ========================================================
    # LOAD
    # ========================================================

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


    height, width = (
        source.shape
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
    # GLOBAL PRIOR
    # ========================================================

    prior, global_result = (
        load_global_prior()
    )


    inverse_prior = invert_affine(
        prior
    )


    print("\n")
    print("=" * 80)

    print(
        "GLOBAL LIGHTGLUE-GRADIENT PRIOR"
    )

    print("=" * 80)


    print(
        prior
    )


    print(
        "Global inliers:",
        global_result[
            "inliers"
        ]
    )


    print(
        "Global coverage:",
        global_result[
            "spatial"
        ][
            "coverage"
        ]
    )


    print(
        "Global RMSE:",
        global_result[
            "ransac_reprojection_rmse_px"
        ]
    )


    # ========================================================
    # WARP SOURCE BY GLOBAL PRIOR
    # ========================================================

    warped_source = cv2.warpAffine(
        source,
        prior,
        (
            width,
            height
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


    warped_mask = (
        cv2.warpAffine(
            (
                valid_mask.astype(
                    np.uint8
                )
                *
                255
            ),
            prior,
            (
                width,
                height
            ),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        >
        0
    )


    common_valid = (
        valid_mask
        &
        warped_mask
    )


    common_distance = (
        distance_from_invalid(
            common_valid
        )
    )


    common_safe = (
        common_distance
        >=
        BOUNDARY_MARGIN_PX
    )


    print(
        "\nCommon valid pixels:",
        int(
            np.count_nonzero(
                common_valid
            )
        )
    )


    print(
        "Common safe pixels:",
        int(
            np.count_nonzero(
                common_safe
            )
        )
    )


    cv2.imwrite(
        str(
            WARPED_SOURCE_VIS
        ),
        warped_source
    )


    # ========================================================
    # MIND
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "MIND DESCRIPTORS"
    )

    print("=" * 80)


    print(
        "\nSource:"
    )


    source_mind = compute_mind_descriptor(
        warped_source,
        common_valid
    )


    print(
        "Source MIND shape:",
        source_mind.shape
    )


    print(
        "\nReference:"
    )


    reference_mind = compute_mind_descriptor(
        reference,
        common_valid
    )


    print(
        "Reference MIND shape:",
        reference_mind.shape
    )


    cv2.imwrite(
        str(
            MIND_SOURCE_VIS
        ),
        descriptor_preview(
            source_mind,
            common_valid
        )
    )


    cv2.imwrite(
        str(
            MIND_REFERENCE_VIS
        ),
        descriptor_preview(
            reference_mind,
            common_valid
        )
    )


    # ========================================================
    # ANCHORS
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "ANCHOR SELECTION"
    )

    print("=" * 80)


    anchors, cell_summary = (
        select_anchors(
            common_valid,
            common_distance,
            warped_source,
            reference
        )
    )


    print(
        "Selected anchors:",
        len(
            anchors
        )
    )


    print(
        "Grid:",
        GRID_ROWS,
        "x",
        GRID_COLS
    )


    print(
        "Anchors/cell:",
        ANCHORS_PER_CELL
    )


    print(
        "Template:",
        (
            2 * TEMPLATE_RADIUS + 1
        ),
        "x",
        (
            2 * TEMPLATE_RADIUS + 1
        )
    )


    print(
        "Search radius:",
        SEARCH_RADIUS,
        "px"
    )


    anchor_image = cv2.cvtColor(
        warped_source,
        cv2.COLOR_GRAY2BGR
    )


    for anchor in anchors:

        cv2.circle(
            anchor_image,
            (
                anchor[
                    "x"
                ],
                anchor[
                    "y"
                ]
            ),
            3,
            (
                0,
                255,
                255
            ),
            -1
        )


    cv2.imwrite(
        str(
            ANCHOR_VIS
        ),
        anchor_image
    )


    # ========================================================
    # FORWARD + REVERSE MIND SEARCH
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "LOCAL MIND MATCHING"
    )

    print("=" * 80)


    warped_source_points = []

    reference_points = []

    costs = []

    distinctness = []

    reverse_errors = []

    records = []


    forward_distinct = 0

    mutual_matches = 0


    for anchor_index, anchor in enumerate(
        anchors
    ):


        source_point = np.array(
            [
                anchor[
                    "x"
                ],
                anchor[
                    "y"
                ]
            ],
            dtype=np.float32
        )


        # --------------------------------------------
        # FORWARD
        # --------------------------------------------

        forward = mind_search(
            source_mind,
            reference_mind,
            source_point,
            source_point,
            SEARCH_RADIUS
        )


        if forward is None:

            continue


        if (
            forward[
                "distinctness_ratio"
            ]
            <
            MIN_DISTINCTNESS_RATIO
        ):

            continue


        forward_distinct += 1


        reference_point = np.array(
            [
                forward[
                    "x"
                ],
                forward[
                    "y"
                ]
            ],
            dtype=np.float32
        )


        # --------------------------------------------
        # REVERSE
        #
        # Reference template should return to original
        # globally-warped source location.
        # --------------------------------------------

        reverse = mind_search(
            reference_mind,
            source_mind,
            reference_point,
            source_point,
            REVERSE_SEARCH_RADIUS
        )


        if reverse is None:

            continue


        reverse_position = np.array(
            [
                reverse[
                    "x"
                ],
                reverse[
                    "y"
                ]
            ],
            dtype=np.float32
        )


        reverse_error = float(
            np.linalg.norm(
                reverse_position
                -
                source_point
            )
        )


        if (
            reverse_error
            >
            MAX_REVERSE_ERROR_PX
        ):

            continue


        mutual_matches += 1


        warped_source_points.append(
            source_point
        )


        reference_points.append(
            reference_point
        )


        costs.append(
            float(
                forward[
                    "best_cost"
                ]
            )
        )


        distinctness.append(
            float(
                forward[
                    "distinctness_ratio"
                ]
            )
        )


        reverse_errors.append(
            reverse_error
        )


        records.append(
            {
                "anchor_index":
                    anchor_index,

                "grid_row":
                    anchor[
                        "grid_row"
                    ],

                "grid_col":
                    anchor[
                        "grid_col"
                    ],

                "residual_dx":
                    forward[
                        "dx"
                    ],

                "residual_dy":
                    forward[
                        "dy"
                    ],

                "mind_cost":
                    forward[
                        "best_cost"
                    ],

                "distinctness_ratio":
                    forward[
                        "distinctness_ratio"
                    ],

                "reverse_error_px":
                    reverse_error
            }
        )


    print(
        "Forward distinct:",
        forward_distinct,
        "/",
        len(
            anchors
        )
    )


    print(
        "Mutual matches:",
        mutual_matches,
        "/",
        len(
            anchors
        )
    )


    if mutual_matches < 3:

        raise RuntimeError(
            "Too few mutual MIND matches."
        )


    warped_source_points = np.asarray(
        warped_source_points,
        dtype=np.float32
    )


    reference_points = np.asarray(
        reference_points,
        dtype=np.float32
    )


    costs = np.asarray(
        costs,
        dtype=np.float32
    )


    distinctness = np.asarray(
        distinctness,
        dtype=np.float32
    )


    reverse_errors = np.asarray(
        reverse_errors,
        dtype=np.float32
    )


    # Convert warped-source coordinates back to
    # ORIGINAL Pair 003 source coordinates.

    original_source_points = (
        transform_points(
            warped_source_points,
            inverse_prior
        )
    )


    print(
        "\nMIND cost:"
    )


    print(
        "min:",
        float(
            costs.min()
        )
    )

    print(
        "median:",
        float(
            np.median(
                costs
            )
        )
    )

    print(
        "max:",
        float(
            costs.max()
        )
    )


    print(
        "\nDistinctness ratio:"
    )


    print(
        "min:",
        float(
            distinctness.min()
        )
    )

    print(
        "median:",
        float(
            np.median(
                distinctness
            )
        )
    )

    print(
        "max:",
        float(
            distinctness.max()
        )
    )


    bench.save_match_visualization(
        source,
        reference,
        original_source_points,
        reference_points,
        CANDIDATE_VIS,
        max_draw=250
    )


    # ========================================================
    # RANSAC
    # ========================================================

    matrix, inlier_mask = (
        cv2.estimateAffine2D(
            original_source_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=
                RANSAC_THRESHOLD_PX,
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

        raise RuntimeError(
            "MIND RANSAC failed."
        )


    inlier_mask = (
        inlier_mask.ravel()
        >
        0
    )


    inlier_source = (
        original_source_points[
            inlier_mask
        ]
    )


    inlier_reference = (
        reference_points[
            inlier_mask
        ]
    )


    inlier_costs = (
        costs[
            inlier_mask
        ]
    )


    inlier_distinctness = (
        distinctness[
            inlier_mask
        ]
    )


    prediction = transform_points(
        inlier_source,
        matrix
    )


    residuals = np.linalg.norm(
        prediction
        -
        inlier_reference,
        axis=1
    )


    candidate_count = int(
        len(
            original_source_points
        )
    )


    inlier_count = int(
        len(
            inlier_source
        )
    )


    parameters = bench.affine_parameters(
        matrix
    )


    sane = bench.transform_sanity(
        parameters
    )


    spatial = bench.spatial_metrics(
        inlier_source,
        common_safe
    )


    # High score = low MIND cost + distinct peak.

    quality = (
        distinctness[
            inlier_mask
        ]
        /
        (
            inlier_costs
            +
            1e-6
        )
    ).astype(
        np.float32
    )


    (
        uniform_source,
        uniform_reference,
        uniform_scores

    ) = bench.uniform_select(
        inlier_source,
        inlier_reference,
        quality,
        common_safe
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


    median_residual = float(
        np.median(
            residuals
        )
    )


    mean_residual = float(
        np.mean(
            residuals
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "PAIR 003 MIND RESULT"
    )

    print("=" * 80)


    print(
        "Candidate matches:",
        candidate_count
    )


    print(
        "RANSAC inliers:",
        inlier_count
    )


    print(
        "Inlier ratio:",
        inlier_count
        /
        candidate_count
    )


    print(
        "RANSAC reprojection RMSE:",
        rmse,
        "px"
    )


    print(
        "Median residual:",
        median_residual,
        "px"
    )


    print(
        "Mean residual:",
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
        "Uniform inliers:",
        len(
            uniform_source
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


    # ========================================================
    # VISUALIZATION
    # ========================================================

    bench.save_match_visualization(
        source,
        reference,
        inlier_source,
        inlier_reference,
        INLIER_VIS,
        max_draw=250
    )


    bench.save_match_visualization(
        source,
        reference,
        uniform_source,
        uniform_reference,
        UNIFORM_VIS,
        max_draw=250
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

                "mind_cost",

                "distinctness_ratio",

                "reverse_error_px",

                "ransac_inlier"
            ]
        )


        for index in range(
            candidate_count
        ):

            writer.writerow(
                [
                    float(
                        original_source_points[
                            index,
                            0
                        ]
                    ),

                    float(
                        original_source_points[
                            index,
                            1
                        ]
                    ),

                    float(
                        reference_points[
                            index,
                            0
                        ]
                    ),

                    float(
                        reference_points[
                            index,
                            1
                        ]
                    ),

                    float(
                        costs[
                            index
                        ]
                    ),

                    float(
                        distinctness[
                            index
                        ]
                    ),

                    float(
                        reverse_errors[
                            index
                        ]
                    ),

                    int(
                        inlier_mask[
                            index
                        ]
                    )
                ]
            )


    # ========================================================
    # JSON
    # ========================================================

    result = {

        "pair_id":
            "pair_003",

        "method":
            (
                "global_lightglue_gradient_prior_"
                "plus_local_MIND"
            ),

        "global_prior": {

            "matrix":
                prior.tolist(),

            "inliers":
                global_result[
                    "inliers"
                ],

            "coverage":
                global_result[
                    "spatial"
                ][
                    "coverage"
                ],

            "rmse_px":
                global_result[
                    "ransac_reprojection_rmse_px"
                ]
        },

        "mind": {

            "descriptor_channels":
                int(
                    source_mind.shape[
                        0
                    ]
                ),

            "neighbour_shift_px":
                MIND_SHIFT,

            "patch_size":
                MIND_PATCH_SIZE
        },

        "matching": {

            "grid_rows":
                GRID_ROWS,

            "grid_cols":
                GRID_COLS,

            "anchors_per_cell":
                ANCHORS_PER_CELL,

            "selected_anchors":
                len(
                    anchors
                ),

            "template_radius":
                TEMPLATE_RADIUS,

            "search_radius":
                SEARCH_RADIUS,

            "minimum_distinctness_ratio":
                MIN_DISTINCTNESS_RATIO,

            "forward_distinct_matches":
                forward_distinct,

            "mutual_matches":
                mutual_matches,

            "cell_summary":
                cell_summary
        },

        "result": {

            "candidate_matches":
                candidate_count,

            "inliers":
                inlier_count,

            "inlier_ratio":
                float(
                    inlier_count
                    /
                    candidate_count
                ),

            "ransac_reprojection_rmse_px":
                rmse,

            "ransac_reprojection_median_px":
                median_residual,

            "ransac_reprojection_mean_px":
                mean_residual,

            "affine_matrix":
                matrix.tolist(),

            "transform":
                parameters,

            "transform_sanity":
                sane,

            "spatial":
                spatial,

            "uniform_inliers":
                int(
                    len(
                        uniform_source
                    )
                )
        },

        "metric_note":
            (
                "RANSAC reprojection RMSE measures "
                "internal correspondence self-consistency. "
                "It is not independent ground-truth "
                "registration accuracy."
            )
    }


    RESULT_JSON.write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )


    print("\n")
    print("=" * 80)

    print(
        "PAIR 003 PRIOR-GUIDED MIND COMPLETE"
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
        "\nInlier visualization:"
    )

    print(
        INLIER_VIS
    )


    print(
        "\nUniform visualization:"
    )

    print(
        UNIFORM_VIS
    )


if __name__ == "__main__":

    main()