from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np


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
    / "prior_guided_ngf"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_JSON = (
    OUTPUT_DIR
    / "pair003_prior_guided_ngf.json"
)

MATCH_CSV = (
    OUTPUT_DIR
    / "pair003_prior_guided_ngf_matches.csv"
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
    / "source_warped_by_global_prior.png"
)

NGF_SOURCE_VIS = (
    OUTPUT_DIR
    / "warped_source_gradient.png"
)

NGF_REFERENCE_VIS = (
    OUTPUT_DIR
    / "reference_gradient.png"
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

ANCHORS_PER_CELL = 3

TEMPLATE_RADIUS = 18

SEARCH_RADIUS = 8

REVERSE_SEARCH_RADIUS = 2

BOUNDARY_MARGIN_PX = 8

ANCHOR_MIN_SEPARATION_PX = 32

MIN_ORIENTATION_SCORE = 0.58

MIN_PEAK_MARGIN = 0.008

MIN_REVERSE_SCORE = 0.56

MAX_REVERSE_ERROR_PX = 1.5

RANSAC_THRESHOLD_PX = 2.5

RANSAC_MAX_ITERS = 20000

RANSAC_CONFIDENCE = 0.999

MAX_UNIFORM_PER_CELL = 4


# ============================================================
# IO
# ============================================================

def load_gray(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read:\n{path}"
        )

    return image


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
            result.get("matcher") == "lightglue"
            and
            result.get("representation") == "gradient"
            and
            result.get("status") == "ok"
            and
            result.get("transform_sanity") is True
        ):

            matrix = np.asarray(
                result["affine_matrix"],
                dtype=np.float64
            )

            return matrix, result

    raise RuntimeError(
        "Could not find sane "
        "LightGlue-gradient global prior."
    )


# ============================================================
# AFFINE
# ============================================================

def transform_points(
    points,
    matrix
):

    if len(points) == 0:

        return np.zeros(
            (0, 2),
            dtype=np.float32
        )

    homogeneous = np.column_stack(
        [
            points.astype(np.float64),
            np.ones(
                len(points),
                dtype=np.float64
            )
        ]
    )

    output = (
        homogeneous
        @
        matrix.T
    )

    return output.astype(
        np.float32
    )


def invert_affine(matrix):

    full = np.eye(
        3,
        dtype=np.float64
    )

    full[:2, :] = matrix

    inverse = np.linalg.inv(
        full
    )

    return inverse[:2, :]


# ============================================================
# SAFE MASK
# ============================================================

def make_safe_mask(
    mask
):

    binary = (
        mask > 0
    ).astype(
        np.uint8
    )

    distance = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5
    )

    safe = (
        distance
        >=
        BOUNDARY_MARGIN_PX
    )

    return (
        safe,
        distance
    )


# ============================================================
# GRADIENT FIELD
# ============================================================

def gradient_field(
    image,
    valid_mask
):

    image_f = (
        image.astype(
            np.float32
        )
        /
        255.0
    )

    blurred = cv2.GaussianBlur(
        image_f,
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

    magnitude = np.sqrt(
        gx * gx
        +
        gy * gy
    )

    # Stabilizes normalization in very flat regions.
    epsilon = 0.01

    denominator = np.sqrt(
        gx * gx
        +
        gy * gy
        +
        epsilon * epsilon
    )

    ngx = (
        gx
        /
        denominator
    )

    ngy = (
        gy
        /
        denominator
    )

    ngx[
        ~valid_mask
    ] = 0

    ngy[
        ~valid_mask
    ] = 0

    magnitude[
        ~valid_mask
    ] = 0

    return (
        ngx,
        ngy,
        magnitude
    )


def gradient_visualization(
    magnitude,
    mask
):

    output = np.zeros(
        magnitude.shape,
        dtype=np.uint8
    )

    values = magnitude[
        mask
    ]

    if values.size == 0:

        return output

    low, high = np.percentile(
        values,
        [1, 99]
    )

    normalized = (
        magnitude
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
# CELL HELPERS
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


def get_cell(
    point,
    width,
    height
):

    x, y = point

    col = min(
        GRID_COLS - 1,
        max(
            0,
            int(
                x
                /
                width
                *
                GRID_COLS
            )
        )
    )

    row = min(
        GRID_ROWS - 1,
        max(
            0,
            int(
                y
                /
                height
                *
                GRID_ROWS
            )
        )
    )

    return (
        row,
        col
    )


# ============================================================
# ANCHOR SELECTION
# ============================================================

def select_anchors(
    common_safe,
    distance,
    source_magnitude
):

    height, width = (
        common_safe.shape
    )

    required_distance = (
        TEMPLATE_RADIUS
        +
        SEARCH_RADIUS
        +
        2
    )

    # Local texture estimate.
    texture = cv2.boxFilter(
        source_magnitude,
        cv2.CV_32F,
        (25, 25),
        normalize=True
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

            valid_cell = (
                common_safe[
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
                valid_cell
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

            selected_cell = []

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

                keep = True

                for existing in selected_cell:

                    dx = (
                        x
                        -
                        existing[0]
                    )

                    dy = (
                        y
                        -
                        existing[1]
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

                        keep = False

                        break

                if not keep:

                    continue

                selected_cell.append(
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
                        selected_cell
                    )
                    >=
                    ANCHORS_PER_CELL
                ):

                    break

            for x, y, score in selected_cell:

                anchors.append(
                    {
                        "x":
                            x,

                        "y":
                            y,

                        "row":
                            row,

                        "col":
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
                            selected_cell
                        )
                }
            )

    return (
        anchors,
        cell_summary
    )


# ============================================================
# NGF PATCH SCORE
# ============================================================

def ngf_patch_score(
    source_gx,
    source_gy,
    source_mag,
    reference_gx,
    reference_gy,
    reference_mag
):

    # Contrast reversal invariant:
    #
    # cos(theta)^2
    #
    # Same direction -> 1
    # Opposite direction -> 1
    # Orthogonal -> 0

    dot = (
        source_gx
        *
        reference_gx
        +
        source_gy
        *
        reference_gy
    )

    orientation = (
        dot
        *
        dot
    )

    source_scale = (
        np.percentile(
            source_mag,
            90
        )
        +
        1e-6
    )

    reference_scale = (
        np.percentile(
            reference_mag,
            90
        )
        +
        1e-6
    )

    source_weight = np.clip(
        source_mag
        /
        source_scale,
        0,
        1
    )

    reference_weight = np.clip(
        reference_mag
        /
        reference_scale,
        0,
        1
    )

    weight = np.sqrt(
        source_weight
        *
        reference_weight
    )

    denominator = float(
        np.sum(
            weight
        )
    )

    if denominator <= 1e-6:

        return 0.0

    return float(
        np.sum(
            orientation
            *
            weight
        )
        /
        denominator
    )


# ============================================================
# LOCAL SEARCH
# ============================================================

def local_ngf_search(
    template_center,
    search_center,
    template_gx,
    template_gy,
    template_mag,
    target_gx,
    target_gy,
    target_mag,
    radius
):

    tx = int(
        round(
            template_center[0]
        )
    )

    ty = int(
        round(
            template_center[1]
        )
    )

    sx = int(
        round(
            search_center[0]
        )
    )

    sy = int(
        round(
            search_center[1]
        )
    )

    r = TEMPLATE_RADIUS


    template_x = template_gx[
        ty-r:ty+r+1,
        tx-r:tx+r+1
    ]

    template_y = template_gy[
        ty-r:ty+r+1,
        tx-r:tx+r+1
    ]

    template_m = template_mag[
        ty-r:ty+r+1,
        tx-r:tx+r+1
    ]


    expected_shape = (
        2 * r + 1,
        2 * r + 1
    )

    if (
        template_x.shape
        !=
        expected_shape
    ):

        return None


    scores = []


    for dy in range(
        -radius,
        radius + 1
    ):

        for dx in range(
            -radius,
            radius + 1
        ):

            cx = (
                sx + dx
            )

            cy = (
                sy + dy
            )


            target_x = target_gx[
                cy-r:cy+r+1,
                cx-r:cx+r+1
            ]

            target_y = target_gy[
                cy-r:cy+r+1,
                cx-r:cx+r+1
            ]

            target_m = target_mag[
                cy-r:cy+r+1,
                cx-r:cx+r+1
            ]


            if (
                target_x.shape
                !=
                expected_shape
            ):

                continue


            score = ngf_patch_score(
                template_x,
                template_y,
                template_m,
                target_x,
                target_y,
                target_m
            )


            scores.append(
                (
                    score,
                    dx,
                    dy
                )
            )


    if len(
        scores
    ) == 0:

        return None


    scores.sort(
        key=lambda item:
            item[0],
        reverse=True
    )


    best_score, best_dx, best_dy = (
        scores[0]
    )


    # Second-best peak sufficiently separated
    # from the winning displacement.

    second_score = None


    for score, dx, dy in scores[1:]:

        if (
            abs(
                dx
                -
                best_dx
            )
            >
            1

            or

            abs(
                dy
                -
                best_dy
            )
            >
            1
        ):

            second_score = score

            break


    if second_score is None:

        second_score = (
            scores[1][0]
            if len(scores) > 1
            else 0.0
        )


    peak_margin = (
        best_score
        -
        second_score
    )


    return {
        "score":
            float(
                best_score
            ),

        "second_score":
            float(
                second_score
            ),

        "peak_margin":
            float(
                peak_margin
            ),

        "dx":
            int(
                best_dx
            ),

        "dy":
            int(
                best_dy
            ),

        "x":
            int(
                sx
                +
                best_dx
            ),

        "y":
            int(
                sy
                +
                best_dy
            )
    }


# ============================================================
# AFFINE METRICS
# ============================================================

def affine_parameters(
    matrix
):

    a = float(
        matrix[0, 0]
    )

    b = float(
        matrix[0, 1]
    )

    c = float(
        matrix[1, 0]
    )

    d = float(
        matrix[1, 1]
    )

    tx = float(
        matrix[0, 2]
    )

    ty = float(
        matrix[1, 2]
    )


    scale_x = math.sqrt(
        a * a
        +
        c * c
    )

    scale_y = math.sqrt(
        b * b
        +
        d * d
    )


    rotation_deg = math.degrees(
        math.atan2(
            c,
            a
        )
    )


    translation_px = math.sqrt(
        tx * tx
        +
        ty * ty
    )


    determinant = (
        a * d
        -
        b * c
    )


    if (
        scale_x > 0
        and
        scale_y > 0
    ):

        axis_dot = (
            a * b
            +
            c * d
        ) / (
            scale_x
            *
            scale_y
        )

    else:

        axis_dot = float(
            "nan"
        )


    return {
        "tx_px":
            tx,

        "ty_px":
            ty,

        "translation_px":
            translation_px,

        "rotation_deg":
            rotation_deg,

        "scale_x":
            scale_x,

        "scale_y":
            scale_y,

        "determinant":
            determinant,

        "normalized_axis_dot":
            float(
                axis_dot
            )
    }


def transform_sanity(
    parameters
):

    return bool(
        parameters[
            "determinant"
        ] > 0

        and

        parameters[
            "translation_px"
        ] <= 25

        and

        abs(
            parameters[
                "rotation_deg"
            ]
        ) <= 5

        and

        0.95
        <=
        parameters[
            "scale_x"
        ]
        <=
        1.05

        and

        0.95
        <=
        parameters[
            "scale_y"
        ]
        <=
        1.05

        and

        abs(
            parameters[
                "normalized_axis_dot"
            ]
        ) <= 0.10
    )


# ============================================================
# COVERAGE
# ============================================================

def valid_grid_cells(
    mask
):

    height, width = (
        mask.shape
    )

    cells = set()


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


            cell = mask[
                y0:y1,
                x0:x1
            ]


            if (
                cell.size > 0
                and
                float(
                    cell.mean()
                )
                >=
                0.05
            ):

                cells.add(
                    (
                        row,
                        col
                    )
                )

    return cells


def spatial_metrics(
    points,
    mask
):

    height, width = (
        mask.shape
    )

    valid_cells = (
        valid_grid_cells(
            mask
        )
    )

    occupied = set()


    for point in points:

        cell = get_cell(
            point,
            width,
            height
        )

        if cell in valid_cells:

            occupied.add(
                cell
            )


    coverage = (
        len(
            occupied
        )
        /
        len(
            valid_cells
        )
        if len(
            valid_cells
        )
        > 0
        else 0.0
    )


    return {
        "valid_cells":
            len(
                valid_cells
            ),

        "occupied_valid_cells":
            len(
                occupied
            ),

        "coverage":
            float(
                coverage
            )
    }


# ============================================================
# UNIFORM SELECTION
# ============================================================

def uniform_select(
    points0,
    scores,
    mask
):

    if len(
        points0
    ) == 0:

        return np.zeros(
            0,
            dtype=np.int32
        )


    height, width = (
        mask.shape
    )


    order = np.argsort(
        -scores
    )


    counts = {}

    selected = []


    for index in order:

        cell = get_cell(
            points0[
                index
            ],
            width,
            height
        )


        count = counts.get(
            cell,
            0
        )


        if (
            count
            >=
            MAX_UNIFORM_PER_CELL
        ):

            continue


        counts[
            cell
        ] = (
            count + 1
        )


        selected.append(
            int(
                index
            )
        )


    return np.asarray(
        selected,
        dtype=np.int32
    )


# ============================================================
# VISUALIZATION
# ============================================================

def save_match_visualization(
    source,
    reference,
    points0,
    points1,
    path,
    max_draw=250
):

    source_bgr = cv2.cvtColor(
        source,
        cv2.COLOR_GRAY2BGR
    )

    reference_bgr = cv2.cvtColor(
        reference,
        cv2.COLOR_GRAY2BGR
    )


    height = max(
        source.shape[0],
        reference.shape[0]
    )

    width0 = (
        source.shape[1]
    )

    width1 = (
        reference.shape[1]
    )


    canvas = np.zeros(
        (
            height,
            width0
            +
            width1,
            3
        ),
        dtype=np.uint8
    )


    canvas[
        :source.shape[0],
        :width0
    ] = source_bgr


    canvas[
        :reference.shape[0],
        width0:width0+width1
    ] = reference_bgr


    if (
        len(
            points0
        )
        >
        max_draw
    ):

        indices = np.linspace(
            0,
            len(
                points0
            )
            -
            1,
            max_draw
        ).astype(
            np.int32
        )

    else:

        indices = np.arange(
            len(
                points0
            )
        )


    for index in indices:

        p0 = (
            int(
                round(
                    points0[
                        index,
                        0
                    ]
                )
            ),
            int(
                round(
                    points0[
                        index,
                        1
                    ]
                )
            )
        )


        p1 = (
            int(
                round(
                    points1[
                        index,
                        0
                    ]
                )
            )
            +
            width0,

            int(
                round(
                    points1[
                        index,
                        1
                    ]
                )
            )
        )


        cv2.line(
            canvas,
            p0,
            p1,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )


        cv2.circle(
            canvas,
            p0,
            2,
            (0, 255, 255),
            -1
        )


        cv2.circle(
            canvas,
            p1,
            2,
            (0, 255, 255),
            -1
        )


    cv2.imwrite(
        str(
            path
        ),
        canvas
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - PAIR 003 "
        "PRIOR-GUIDED NGF MATCHING"
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
    # LOAD PAIR
    # ========================================================

    source = load_gray(
        SOURCE_PATH
    )

    reference = load_gray(
        REFERENCE_PATH
    )

    mask_image = load_gray(
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


    # ========================================================
    # GLOBAL PRIOR
    # ========================================================

    prior, prior_result = (
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
        "Prior inliers:",
        prior_result[
            "inliers"
        ]
    )


    print(
        "Prior coverage:",
        prior_result[
            "spatial"
        ][
            "coverage"
        ]
    )


    # ========================================================
    # WARP SOURCE USING PRIOR
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


    warped_valid = cv2.warpAffine(
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
    ) > 0


    reference_safe, reference_distance = (
        make_safe_mask(
            valid_mask
        )
    )


    warped_safe, warped_distance = (
        make_safe_mask(
            warped_valid
        )
    )


    common_safe = (
        reference_safe
        &
        warped_safe
    )


    common_distance = np.minimum(
        reference_distance,
        warped_distance
    )


    print(
        "\nCommon safe pixels:",
        int(
            np.count_nonzero(
                common_safe
            )
        )
    )


    print(
        "Common safe ratio:",
        float(
            common_safe.mean()
        )
    )


    cv2.imwrite(
        str(
            WARPED_SOURCE_VIS
        ),
        warped_source
    )


    # ========================================================
    # GRADIENT FIELDS
    # ========================================================

    (
        source_gx,
        source_gy,
        source_mag

    ) = gradient_field(
        warped_source,
        common_safe
    )


    (
        reference_gx,
        reference_gy,
        reference_mag

    ) = gradient_field(
        reference,
        common_safe
    )


    cv2.imwrite(
        str(
            NGF_SOURCE_VIS
        ),
        gradient_visualization(
            source_mag,
            common_safe
        )
    )


    cv2.imwrite(
        str(
            NGF_REFERENCE_VIS
        ),
        gradient_visualization(
            reference_mag,
            common_safe
        )
    )


    # ========================================================
    # SELECT SPATIALLY DISTRIBUTED ANCHORS
    # ========================================================

    (
        anchors,
        cell_summary

    ) = select_anchors(
        common_safe,
        common_distance,
        source_mag
    )


    print("\n")
    print("=" * 80)

    print(
        "ANCHOR SELECTION"
    )

    print("=" * 80)


    print(
        "Selected anchors:",
        len(
            anchors
        )
    )


    print(
        "Template size:",
        2 * TEMPLATE_RADIUS + 1,
        "x",
        2 * TEMPLATE_RADIUS + 1
    )


    print(
        "Forward search:",
        SEARCH_RADIUS,
        "px"
    )


    # Anchor visualization.

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
            (0, 255, 255),
            -1
        )


    cv2.imwrite(
        str(
            ANCHOR_VIS
        ),
        anchor_image
    )


    # ========================================================
    # LOCAL NGF SEARCH
    # ========================================================

    source_points = []

    reference_points = []

    match_scores = []

    match_margins = []

    local_records = []


    accepted_forward = 0

    accepted_reverse = 0


    print("\n")
    print("=" * 80)

    print(
        "LOCAL NGF SEARCH"
    )

    print("=" * 80)


    for index, anchor in enumerate(
        anchors
    ):

        q = np.array(
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


        forward = local_ngf_search(

            q,
            q,

            source_gx,
            source_gy,
            source_mag,

            reference_gx,
            reference_gy,
            reference_mag,

            SEARCH_RADIUS
        )


        if forward is None:

            continue


        if (
            forward[
                "score"
            ]
            <
            MIN_ORIENTATION_SCORE

            or

            forward[
                "peak_margin"
            ]
            <
            MIN_PEAK_MARGIN
        ):

            continue


        accepted_forward += 1


        r = np.array(
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
        # Reverse consistency:
        #
        # Take the matched reference patch and see
        # whether it returns to the original warped
        # source position.
        # --------------------------------------------

        reverse = local_ngf_search(

            r,
            q,

            reference_gx,
            reference_gy,
            reference_mag,

            source_gx,
            source_gy,
            source_mag,

            REVERSE_SEARCH_RADIUS
        )


        if reverse is None:

            continue


        if (
            reverse[
                "score"
            ]
            <
            MIN_REVERSE_SCORE
        ):

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
                q
            )
        )


        if (
            reverse_error
            >
            MAX_REVERSE_ERROR_PX
        ):

            continue


        accepted_reverse += 1


        # --------------------------------------------
        # q is in GLOBAL-PRIOR-WARPED source space.
        #
        # Convert q back to the original Pair 003
        # source coordinates.
        # --------------------------------------------

        source_original = transform_points(
            q.reshape(
                1,
                2
            ),
            inverse_prior
        )[0]


        source_points.append(
            source_original
        )


        reference_points.append(
            r
        )


        match_scores.append(
            float(
                forward[
                    "score"
                ]
            )
        )


        match_margins.append(
            float(
                forward[
                    "peak_margin"
                ]
            )
        )


        local_records.append(
            {
                "anchor_index":
                    index,

                "grid_row":
                    anchor[
                        "row"
                    ],

                "grid_col":
                    anchor[
                        "col"
                    ],

                "warped_source_x":
                    float(
                        q[0]
                    ),

                "warped_source_y":
                    float(
                        q[1]
                    ),

                "residual_dx":
                    forward[
                        "dx"
                    ],

                "residual_dy":
                    forward[
                        "dy"
                    ],

                "ngf_score":
                    forward[
                        "score"
                    ],

                "peak_margin":
                    forward[
                        "peak_margin"
                    ],

                "reverse_score":
                    reverse[
                        "score"
                    ],

                "reverse_error_px":
                    reverse_error
            }
        )


    print(
        "Forward accepted:",
        accepted_forward,
        "/",
        len(
            anchors
        )
    )


    print(
        "Mutual accepted:",
        accepted_reverse,
        "/",
        len(
            anchors
        )
    )


    if len(
        source_points
    ) < 3:

        raise RuntimeError(
            "Too few NGF correspondences "
            "for affine estimation."
        )


    source_points = np.asarray(
        source_points,
        dtype=np.float32
    )


    reference_points = np.asarray(
        reference_points,
        dtype=np.float32
    )


    match_scores = np.asarray(
        match_scores,
        dtype=np.float32
    )


    match_margins = np.asarray(
        match_margins,
        dtype=np.float32
    )


    save_match_visualization(
        source,
        reference,
        source_points,
        reference_points,
        CANDIDATE_VIS
    )


    # ========================================================
    # RANSAC
    # ========================================================

    matrix, inlier_mask = (
        cv2.estimateAffine2D(
            source_points,
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
            "NGF affine RANSAC failed."
        )


    inlier_mask = (
        inlier_mask.ravel()
        >
        0
    )


    inlier_source = (
        source_points[
            inlier_mask
        ]
    )


    inlier_reference = (
        reference_points[
            inlier_mask
        ]
    )


    inlier_scores = (
        match_scores[
            inlier_mask
        ]
    )


    inlier_margins = (
        match_margins[
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


    inlier_count = int(
        len(
            inlier_source
        )
    )


    candidate_count = int(
        len(
            source_points
        )
    )


    parameters = affine_parameters(
        matrix
    )


    spatial = spatial_metrics(
        inlier_source,
        reference_safe
    )


    # Quality = NGF score plus distinctness
    # of the local displacement peak.

    uniform_quality = (
        inlier_scores
        +
        4.0
        *
        inlier_margins
    )


    uniform_indices = uniform_select(
        inlier_source,
        uniform_quality,
        reference_safe
    )


    uniform_source = (
        inlier_source[
            uniform_indices
        ]
    )


    uniform_reference = (
        inlier_reference[
            uniform_indices
        ]
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

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


    print("\n")
    print("=" * 80)

    print(
        "PAIR 003 NGF RESULT"
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
        transform_sanity(
            parameters
        )
    )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    save_match_visualization(
        source,
        reference,
        inlier_source,
        inlier_reference,
        INLIER_VIS
    )


    save_match_visualization(
        source,
        reference,
        uniform_source,
        uniform_reference,
        UNIFORM_VIS
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
                "ngf_score",
                "peak_margin",
                "ransac_inlier"
            ]
        )


        for index in range(
            len(
                source_points
            )
        ):

            writer.writerow(
                [
                    float(
                        source_points[
                            index,
                            0
                        ]
                    ),

                    float(
                        source_points[
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
                        match_scores[
                            index
                        ]
                    ),

                    float(
                        match_margins[
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

    payload = {

        "pair_id":
            "pair_003",

        "method":
            (
                "global_lightglue_prior_"
                "plus_local_ngf_search"
            ),

        "global_prior": {

            "matrix":
                prior.tolist(),

            "inliers":
                prior_result[
                    "inliers"
                ],

            "coverage":
                prior_result[
                    "spatial"
                ][
                    "coverage"
                ]
        },

        "settings": {

            "grid_rows":
                GRID_ROWS,

            "grid_cols":
                GRID_COLS,

            "anchors_per_cell":
                ANCHORS_PER_CELL,

            "template_radius":
                TEMPLATE_RADIUS,

            "search_radius":
                SEARCH_RADIUS,

            "minimum_orientation_score":
                MIN_ORIENTATION_SCORE,

            "minimum_peak_margin":
                MIN_PEAK_MARGIN,

            "reverse_search_radius":
                REVERSE_SEARCH_RADIUS,

            "maximum_reverse_error_px":
                MAX_REVERSE_ERROR_PX,

            "ransac_threshold_px":
                RANSAC_THRESHOLD_PX
        },

        "anchors": {

            "selected":
                len(
                    anchors
                ),

            "forward_accepted":
                accepted_forward,

            "mutual_accepted":
                accepted_reverse,

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
                transform_sanity(
                    parameters
                ),

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
                "RANSAC reprojection RMSE is "
                "correspondence self-consistency. "
                "It is not independent "
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
        "PAIR 003 PRIOR-GUIDED NGF COMPLETE"
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