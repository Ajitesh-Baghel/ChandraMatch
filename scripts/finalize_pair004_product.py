from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np
import rasterio
from affine import Affine
from scipy import ndimage


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pair_004"
    / "canonical"
)

SOURCE_PATH = (
    CANONICAL_DIR
    / "lro_nac_source.tif"
)

REFERENCE_PATH = (
    CANONICAL_DIR
    / "tmc2_reference.tif"
)

COMMON_MASK_PATH = (
    CANONICAL_DIR
    / "common_valid_mask.tif"
)

MATCHES_PATH = (
    ROOT
    / "results"
    / "pair_004"
    / "tiled_lightglue"
    / "pair004_tiled_lightglue_inliers.csv"
)

MATCHER_METRICS_PATH = (
    ROOT
    / "results"
    / "pair_004"
    / "tiled_lightglue"
    / "pair004_tiled_lightglue_metrics.json"
)

VALIDATION_PATH = (
    ROOT
    / "results"
    / "pair_004"
    / "tiled_lightglue_perturbation"
    / "pair004_tiled_lightglue_perturbation.json"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_004"
    / "final_product"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PIXEL_SIZE_M = 5.0

MASK_EROSION_PIXELS = 5

GRID_ROWS = 8
GRID_COLS = 8

# Up to 5 best matches per spatial cell.
UNIFORM_MATCHES_PER_CELL = 5


# ============================================================
# IO
# ============================================================


def read_raster(path):
    with rasterio.open(path) as ds:

        return {
            "data": ds.read(1),
            "mask": ds.read_masks(1) > 0,
            "profile": ds.profile.copy(),
            "transform": ds.transform,
            "crs": ds.crs,
            "nodata": ds.nodata,
            "width": ds.width,
            "height": ds.height,
        }


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_matches(path):
    records = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            records.append(
                {
                    "source_x":
                        float(
                            row[
                                "source_x"
                            ]
                        ),

                    "source_y":
                        float(
                            row[
                                "source_y"
                            ]
                        ),

                    "reference_x":
                        float(
                            row[
                                "reference_x"
                            ]
                        ),

                    "reference_y":
                        float(
                            row[
                                "reference_y"
                            ]
                        ),

                    "ransac_residual_px":
                        float(
                            row[
                                "ransac_residual_px"
                            ]
                        ),
                }
            )

    return records


# ============================================================
# BASIC VALIDATION
# ============================================================


def validate_canonical_pair(
    source,
    reference,
    common,
):
    if (
        source["data"].shape
        != reference["data"].shape
    ):
        raise RuntimeError(
            "Source/reference shapes differ."
        )

    if (
        source["data"].shape
        != common["data"].shape
    ):
        raise RuntimeError(
            "Common mask shape differs."
        )

    if (
        source["transform"]
        != reference["transform"]
    ):
        raise RuntimeError(
            "Source/reference transforms differ."
        )

    if (
        source["width"]
        != reference["width"]
        or
        source["height"]
        != reference["height"]
    ):
        raise RuntimeError(
            "Source/reference raster dimensions differ."
        )


# ============================================================
# MASKS
# ============================================================


def build_masks(
    source,
    reference,
    common,
):
    common_bool = (
        common["data"] > 0
    )

    science_valid = (
        common_bool
        & source["mask"]
        & reference["mask"]
        & np.isfinite(
            source["data"]
        )
        & np.isfinite(
            reference["data"]
        )
    )

    # TMC zero-DN shadow remains SCIENCE VALID.
    #
    # It is excluded only from feature extraction.
    matcher_pre = (
        science_valid
        & (
            reference["data"]
            > 0
        )
    )

    matcher_valid = (
        ndimage.binary_erosion(
            matcher_pre,
            structure=np.ones(
                (3, 3),
                dtype=bool,
            ),
            iterations=(
                MASK_EROSION_PIXELS
            ),
            border_value=0,
        )
    )

    return (
        science_valid,
        matcher_pre,
        matcher_valid,
    )


# ============================================================
# AFFINE UTILITIES
# ============================================================


def affine_points(
    matrix,
    points,
):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    return (
        points
        @ matrix[:, :2].T
        + matrix[:, 2]
    )


def decompose_affine(matrix):
    A = matrix[:, :2]

    c0 = A[:, 0]
    c1 = A[:, 1]

    scale_x = float(
        np.linalg.norm(c0)
    )

    scale_y = float(
        np.linalg.norm(c1)
    )

    rotation_deg = float(
        math.degrees(
            math.atan2(
                A[1, 0],
                A[0, 0],
            )
        )
    )

    tx = float(
        matrix[0, 2]
    )

    ty = float(
        matrix[1, 2]
    )

    determinant = float(
        np.linalg.det(A)
    )

    if (
        scale_x > 0
        and scale_y > 0
    ):
        axis_dot = float(
            np.dot(c0, c1)
            / (
                scale_x
                * scale_y
            )
        )
    else:
        axis_dot = float("nan")

    return {
        "scale_x":
            scale_x,

        "scale_y":
            scale_y,

        "rotation_deg":
            rotation_deg,

        "translation_x_px":
            tx,

        "translation_y_px":
            ty,

        "translation_magnitude_px":
            float(
                math.hypot(
                    tx,
                    ty,
                )
            ),

        "determinant":
            determinant,

        "axis_dot":
            axis_dot,
    }


# ============================================================
# REGISTER NAC TO TMC GRID
# ============================================================


def register_source(
    source_data,
    source_valid,
    reference_valid,
    affine_matrix,
):
    h, w = source_data.shape

    # --------------------------------------------------------
    # Important:
    #
    # We do NOT interpolate the source nodata values.
    #
    # Instead:
    #   1. invalid source = 0
    #   2. warp pixel values
    #   3. warp validity weights
    #   4. divide weighted value by warped coverage
    #
    # This avoids nodata contamination around footprint edges.
    # --------------------------------------------------------

    source_clean = np.where(
        source_valid,
        source_data,
        0.0,
    ).astype(
        np.float32
    )

    source_weight = (
        source_valid.astype(
            np.float32
        )
    )

    warped_value = cv2.warpAffine(
        source_clean,
        affine_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped_weight = cv2.warpAffine(
        source_weight,
        affine_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    registered = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    valid = (
        warped_weight > 0.50
    )

    valid &= reference_valid

    registered[valid] = (
        warped_value[valid]
        / np.maximum(
            warped_weight[valid],
            1e-6,
        )
    )

    return (
        registered,
        valid,
        warped_weight,
    )


# ============================================================
# WRITE REGISTERED GEOTIFF
# ============================================================


def choose_nodata(
    source_nodata,
):
    if (
        source_nodata is not None
        and np.isfinite(
            source_nodata
        )
    ):
        value = np.float32(
            source_nodata
        )

        if np.isfinite(value):
            return float(value)

    return float(
        np.float32(
            -3.4028235e38
        )
    )


def write_registered_geotiff(
    path,
    registered,
    valid,
    reference_profile,
    source_nodata,
):
    nodata = choose_nodata(
        source_nodata
    )

    output = registered.copy()

    output[
        ~valid
    ] = nodata

    profile = (
        reference_profile.copy()
    )

    profile.update(
        {
            "driver":
                "GTiff",

            "dtype":
                "float32",

            "count":
                1,

            "nodata":
                nodata,

            "compress":
                "deflate",

            "predictor":
                3,

            "tiled":
                True,

            "BIGTIFF":
                "IF_SAFER",
        }
    )

    with rasterio.open(
        path,
        "w",
        **profile,
    ) as dst:

        dst.write(
            output.astype(
                np.float32
            ),
            1,
        )

        dst.write_mask(
            valid.astype(
                np.uint8
            )
            * 255
        )

    return nodata


# ============================================================
# PREVIEW NORMALIZATION
# ============================================================


def robust_uint8(
    image,
    valid,
):
    out = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    values = image[
        valid
    ]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return out

    low, high = np.percentile(
        values,
        [2.0, 98.0],
    )

    if high <= low:
        high = low + 1.0

    normalized = (
        image.astype(
            np.float32
        )
        - float(low)
    ) / float(
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    out[
        valid
    ] = np.round(
        normalized[
            valid
        ]
        * 255.0
    ).astype(
        np.uint8
    )

    return out


def gradient_uint8(
    image,
    valid,
):
    image_f = (
        image.astype(
            np.float32
        )
        / 255.0
    )

    image_f = cv2.GaussianBlur(
        image_f,
        (5, 5),
        0.8,
    )

    gx = cv2.Sobel(
        image_f,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image_f,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = np.sqrt(
        gx * gx
        + gy * gy
    )

    out = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    values = magnitude[
        valid
    ]

    if values.size == 0:
        return out

    high = np.percentile(
        values,
        99.0,
    )

    if high <= 0:
        return out

    magnitude = np.clip(
        magnitude / high,
        0.0,
        1.0,
    )

    out[
        valid
    ] = np.round(
        magnitude[
            valid
        ]
        * 255.0
    ).astype(
        np.uint8
    )

    return out


# ============================================================
# OVERLAY / CHECKERBOARD
# ============================================================


def save_overlay(
    registered_preview,
    reference_preview,
    valid,
    path,
):
    reg = cv2.cvtColor(
        registered_preview,
        cv2.COLOR_GRAY2BGR,
    )

    ref = cv2.cvtColor(
        reference_preview,
        cv2.COLOR_GRAY2BGR,
    )

    overlay = cv2.addWeighted(
        reg,
        0.5,
        ref,
        0.5,
        0.0,
    )

    overlay[
        ~valid
    ] = 0

    cv2.imwrite(
        str(path),
        overlay,
    )


def save_checkerboard(
    registered_preview,
    reference_preview,
    valid,
    path,
):
    h, w = (
        registered_preview.shape
    )

    result = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    block = 128

    for y0 in range(
        0,
        h,
        block,
    ):
        for x0 in range(
            0,
            w,
            block,
        ):

            y1 = min(
                h,
                y0 + block,
            )

            x1 = min(
                w,
                x0 + block,
            )

            bx = (
                x0 // block
            )

            by = (
                y0 // block
            )

            if (
                bx + by
            ) % 2 == 0:

                result[
                    y0:y1,
                    x0:x1,
                ] = (
                    registered_preview[
                        y0:y1,
                        x0:x1,
                    ]
                )

            else:

                result[
                    y0:y1,
                    x0:x1,
                ] = (
                    reference_preview[
                        y0:y1,
                        x0:x1,
                    ]
                )

    result[
        ~valid
    ] = 0

    cv2.imwrite(
        str(path),
        result,
    )


# ============================================================
# PIXEL -> MAP COORDINATE
# ============================================================


def pixel_to_map(
    transform,
    x,
    y,
):
    # Pixel centres
    map_x, map_y = (
        transform
        * (
            float(x) + 0.5,
            float(y) + 0.5,
        )
    )

    return (
        float(map_x),
        float(map_y),
    )


# ============================================================
# RECOMPUTE MATCH RESIDUALS
# ============================================================


def recompute_match_residuals(
    records,
    affine_matrix,
):
    if len(
        records
    ) == 0:
        return records

    source_points = np.asarray(
        [
            [
                r[
                    "source_x"
                ],
                r[
                    "source_y"
                ],
            ]
            for r in records
        ],
        dtype=np.float64,
    )

    reference_points = np.asarray(
        [
            [
                r[
                    "reference_x"
                ],
                r[
                    "reference_y"
                ],
            ]
            for r in records
        ],
        dtype=np.float64,
    )

    predicted = affine_points(
        affine_matrix,
        source_points,
    )

    residuals = np.linalg.norm(
        predicted
        - reference_points,
        axis=1,
    )

    for record, residual in zip(
        records,
        residuals,
    ):

        record[
            "final_residual_px"
        ] = float(
            residual
        )

        record[
            "final_residual_nominal_m"
        ] = float(
            residual
            * PIXEL_SIZE_M
        )

    return records


# ============================================================
# SPATIAL GRID
# ============================================================


def valid_grid_cells(
    valid_mask,
):
    h, w = (
        valid_mask.shape
    )

    cells = set()

    for gy in range(
        GRID_ROWS
    ):

        y0 = int(
            round(
                gy
                * h
                / GRID_ROWS
            )
        )

        y1 = int(
            round(
                (gy + 1)
                * h
                / GRID_ROWS
            )
        )

        for gx in range(
            GRID_COLS
        ):

            x0 = int(
                round(
                    gx
                    * w
                    / GRID_COLS
                )
            )

            x1 = int(
                round(
                    (gx + 1)
                    * w
                    / GRID_COLS
                )
            )

            if np.any(
                valid_mask[
                    y0:y1,
                    x0:x1,
                ]
            ):
                cells.add(
                    (
                        gy,
                        gx,
                    )
                )

    return cells


def record_cell(
    record,
    width,
    height,
):
    x = record[
        "reference_x"
    ]

    y = record[
        "reference_y"
    ]

    gx = min(
        GRID_COLS - 1,
        max(
            0,
            int(
                x / width
                * GRID_COLS
            ),
        ),
    )

    gy = min(
        GRID_ROWS - 1,
        max(
            0,
            int(
                y / height
                * GRID_ROWS
            ),
        ),
    )

    return (
        gy,
        gx,
    )


def select_uniform_matches(
    records,
    valid_mask,
):
    h, w = (
        valid_mask.shape
    )

    valid_cells = (
        valid_grid_cells(
            valid_mask
        )
    )

    grouped = {}

    for record in records:

        cell = record_cell(
            record,
            w,
            h,
        )

        if (
            cell
            not in valid_cells
        ):
            continue

        grouped.setdefault(
            cell,
            [],
        ).append(
            record
        )

    selected = []

    for cell in sorted(
        grouped.keys()
    ):

        candidates = sorted(
            grouped[cell],
            key=lambda r:
                r[
                    "final_residual_px"
                ],
        )

        selected.extend(
            candidates[
                :UNIFORM_MATCHES_PER_CELL
            ]
        )

    occupied_cells = {
        record_cell(
            r,
            w,
            h,
        )
        for r in selected
    }

    return (
        selected,
        len(
            occupied_cells
        ),
        len(
            valid_cells
        ),
    )


# ============================================================
# WRITE MATCH CSV
# ============================================================


def write_match_csv(
    path,
    records,
    transform,
):
    fields = [
        "source_x",
        "source_y",

        "reference_x",
        "reference_y",

        "source_map_x_m",
        "source_map_y_m",

        "reference_map_x_m",
        "reference_map_y_m",

        "final_residual_px",

        "final_residual_nominal_m",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:

            (
                source_map_x,
                source_map_y,
            ) = pixel_to_map(
                transform,
                record[
                    "source_x"
                ],
                record[
                    "source_y"
                ],
            )

            (
                reference_map_x,
                reference_map_y,
            ) = pixel_to_map(
                transform,
                record[
                    "reference_x"
                ],
                record[
                    "reference_y"
                ],
            )

            writer.writerow(
                {
                    "source_x":
                        record[
                            "source_x"
                        ],

                    "source_y":
                        record[
                            "source_y"
                        ],

                    "reference_x":
                        record[
                            "reference_x"
                        ],

                    "reference_y":
                        record[
                            "reference_y"
                        ],

                    "source_map_x_m":
                        source_map_x,

                    "source_map_y_m":
                        source_map_y,

                    "reference_map_x_m":
                        reference_map_x,

                    "reference_map_y_m":
                        reference_map_y,

                    "final_residual_px":
                        record[
                            "final_residual_px"
                        ],

                    "final_residual_nominal_m":
                        record[
                            "final_residual_nominal_m"
                        ],
                }
            )


# ============================================================
# MATCH VISUALIZATION
# ============================================================


def save_match_visualization(
    source_gradient,
    reference_gradient,
    records,
    path,
):
    h, w = (
        source_gradient.shape
    )

    max_dim = 1800

    scale = min(
        1.0,
        max_dim
        / max(
            h,
            w,
        ),
    )

    display_w = int(
        round(
            w * scale
        )
    )

    display_h = int(
        round(
            h * scale
        )
    )

    source_small = cv2.resize(
        source_gradient,
        (
            display_w,
            display_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    reference_small = cv2.resize(
        reference_gradient,
        (
            display_w,
            display_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    source_rgb = cv2.cvtColor(
        source_small,
        cv2.COLOR_GRAY2BGR,
    )

    reference_rgb = cv2.cvtColor(
        reference_small,
        cv2.COLOR_GRAY2BGR,
    )

    canvas = np.concatenate(
        [
            source_rgb,
            reference_rgb,
        ],
        axis=1,
    )

    for record in records:

        x0 = int(
            round(
                record[
                    "source_x"
                ]
                * scale
            )
        )

        y0 = int(
            round(
                record[
                    "source_y"
                ]
                * scale
            )
        )

        x1 = int(
            round(
                record[
                    "reference_x"
                ]
                * scale
            )
        ) + display_w

        y1 = int(
            round(
                record[
                    "reference_y"
                ]
                * scale
            )
        )

        cv2.circle(
            canvas,
            (
                x0,
                y0,
            ),
            3,
            (
                0,
                255,
                0,
            ),
            -1,
        )

        cv2.circle(
            canvas,
            (
                x1,
                y1,
            ),
            3,
            (
                0,
                255,
                0,
            ),
            -1,
        )

        cv2.line(
            canvas,
            (
                x0,
                y0,
            ),
            (
                x1,
                y1,
            ),
            (
                0,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(path),
        canvas,
    )


# ============================================================
# VALIDATION SUMMARY
# ============================================================


def summarize_validation(
    validation_results,
):
    response_values = np.asarray(
        [
            float(
                r[
                    "perturbation_response_rmse_px"
                ]
            )
            for r in validation_results
        ],
        dtype=np.float64,
    )

    compensated_values = np.asarray(
        [
            float(
                r[
                    "compensated_baseline_rmse_px"
                ]
            )
            for r in validation_results
        ],
        dtype=np.float64,
    )

    controls = []

    for result in (
        validation_results
    ):

        controls.append(
            {
                "control_id":
                    result[
                        "control_id"
                    ],

                "ransac_inliers":
                    result[
                        "ransac_inliers"
                    ],

                "ransac_inlier_ratio":
                    result[
                        "ransac_inlier_ratio"
                    ],

                "ransac_self_rmse_px":
                    result[
                        "ransac_self_rmse_px"
                    ],

                "perturbation_response_rmse_px":
                    result[
                        "perturbation_response_rmse_px"
                    ],

                "compensated_baseline_rmse_px":
                    result[
                        "compensated_baseline_rmse_px"
                    ],

                "recovered_vs_identity_rmse_px":
                    result[
                        "recovered_vs_identity_rmse_px"
                    ],
            }
        )

    return {
        "controls":
            controls,

        "mean_perturbation_response_rmse_px":
            float(
                response_values.mean()
            ),

        "worst_perturbation_response_rmse_px":
            float(
                response_values.max()
            ),

        "mean_compensated_baseline_rmse_px":
            float(
                compensated_values.mean()
            ),

        "note": (
            "These controlled values measure "
            "whether the real NAC-to-TMC matcher "
            "follows known injected source motion. "
            "They do not provide independent "
            "absolute ground truth for the "
            "unperturbed real pair."
        ),
    }


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 78)

    print(
        "PAIR 004 — FINAL PRODUCT"
    )

    print("=" * 78)

    # --------------------------------------------------------
    # Load canonical pair
    # --------------------------------------------------------

    source = read_raster(
        SOURCE_PATH
    )

    reference = read_raster(
        REFERENCE_PATH
    )

    common = read_raster(
        COMMON_MASK_PATH
    )

    validate_canonical_pair(
        source,
        reference,
        common,
    )

    h, w = (
        source["data"].shape
    )

    print(
        "\nCanonical shape:",
        h,
        "x",
        w,
    )

    print(
        "Resolution:",
        PIXEL_SIZE_M,
        "m/pixel",
    )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    (
        science_valid,
        matcher_pre,
        matcher_valid,
    ) = build_masks(
        source,
        reference,
        common,
    )

    print(
        "\nScience-valid pixels:",
        f"{int(science_valid.sum()):,}",
    )

    print(
        "Nonzero matcher pixels:",
        f"{int(matcher_pre.sum()):,}",
    )

    print(
        "Matcher-valid pixels:",
        f"{int(matcher_valid.sum()):,}",
    )

    # --------------------------------------------------------
    # Load matcher result
    # --------------------------------------------------------

    matcher_metrics = (
        load_json(
            MATCHER_METRICS_PATH
        )
    )

    affine_matrix = np.asarray(
        matcher_metrics[
            "affine"
        ],
        dtype=np.float64,
    )

    affine_info = (
        decompose_affine(
            affine_matrix
        )
    )

    print(
        "\nFinal affine:"
    )

    print(
        affine_matrix
    )

    print(
        "\nRANSAC inliers:",
        matcher_metrics[
            "ransac_inliers"
        ],
    )

    print(
        "RANSAC self RMSE:",
        f"{matcher_metrics['ransac_reprojection_rmse_px']:.6f}px",
    )

    print(
        "Spatial coverage:",
        f"{matcher_metrics['coverage_cells']}/"
        f"{matcher_metrics['valid_coverage_cells']} "
        f"({matcher_metrics['coverage_ratio']:.4f})",
    )

    # --------------------------------------------------------
    # Register source NAC -> TMC grid
    # --------------------------------------------------------

    source_valid = (
        source["mask"]
        & np.isfinite(
            source["data"]
        )
    )

    reference_valid = (
        reference["mask"]
        & np.isfinite(
            reference["data"]
        )
    )

    (
        registered,
        registered_valid,
        warped_weight,
    ) = register_source(
        source["data"],
        source_valid,
        reference_valid,
        affine_matrix,
    )

    print(
        "\nRegistered valid pixels:",
        f"{int(registered_valid.sum()):,}",
    )

    registered_path = (
        OUT_DIR
        / "registered_lro_nac_to_tmc2.tif"
    )

    output_nodata = (
        write_registered_geotiff(
            registered_path,
            registered,
            registered_valid,
            reference["profile"],
            source["nodata"],
        )
    )

    # --------------------------------------------------------
    # Match points
    # --------------------------------------------------------

    records = load_matches(
        MATCHES_PATH
    )

    records = (
        recompute_match_residuals(
            records,
            affine_matrix,
        )
    )

    print(
        "\nLoaded final inliers:",
        len(records),
    )

    all_matches_path = (
        OUT_DIR
        / "pair004_all_match_points.csv"
    )

    write_match_csv(
        all_matches_path,
        records,
        reference["transform"],
    )

    # --------------------------------------------------------
    # Uniform subset
    # --------------------------------------------------------

    (
        uniform_records,
        uniform_cells,
        valid_cells,
    ) = select_uniform_matches(
        records,
        matcher_valid,
    )

    uniform_matches_path = (
        OUT_DIR
        / "pair004_uniform_match_points.csv"
    )

    write_match_csv(
        uniform_matches_path,
        uniform_records,
        reference["transform"],
    )

    print(
        "Uniform matches:",
        len(
            uniform_records
        ),
    )

    print(
        "Uniform occupied cells:",
        f"{uniform_cells}/"
        f"{valid_cells}",
    )

    # --------------------------------------------------------
    # Preview common support
    # --------------------------------------------------------

    visual_valid = (
        registered_valid
        & science_valid
    )

    registered_preview = (
        robust_uint8(
            registered,
            visual_valid,
        )
    )

    reference_preview = (
        robust_uint8(
            reference["data"],
            visual_valid,
        )
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / "registered_lro_nac_preview.png"
        ),
        registered_preview,
    )

    cv2.imwrite(
        str(
            OUT_DIR
            / "tmc2_reference_preview.png"
        ),
        reference_preview,
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay_path = (
        OUT_DIR
        / "registered_overlay_50_50.png"
    )

    save_overlay(
        registered_preview,
        reference_preview,
        visual_valid,
        overlay_path,
    )

    # --------------------------------------------------------
    # Checkerboard
    # --------------------------------------------------------

    checkerboard_path = (
        OUT_DIR
        / "registered_checkerboard.png"
    )

    save_checkerboard(
        registered_preview,
        reference_preview,
        visual_valid,
        checkerboard_path,
    )

    # --------------------------------------------------------
    # Match visualization
    # --------------------------------------------------------

    source_preview = (
        robust_uint8(
            source["data"],
            matcher_valid,
        )
    )

    reference_match_preview = (
        robust_uint8(
            reference["data"],
            matcher_valid,
        )
    )

    source_gradient = (
        gradient_uint8(
            source_preview,
            matcher_valid,
        )
    )

    reference_gradient = (
        gradient_uint8(
            reference_match_preview,
            matcher_valid,
        )
    )

    source_gradient[
        ~matcher_valid
    ] = 0

    reference_gradient[
        ~matcher_valid
    ] = 0

    final_matches_image = (
        OUT_DIR
        / "pair004_final_matches.png"
    )

    save_match_visualization(
        source_gradient,
        reference_gradient,
        uniform_records,
        final_matches_image,
    )

    # --------------------------------------------------------
    # Controlled perturbation validation
    # --------------------------------------------------------

    validation_results = (
        load_json(
            VALIDATION_PATH
        )
    )

    validation_summary = (
        summarize_validation(
            validation_results
        )
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    residual_values = np.asarray(
        [
            r[
                "final_residual_px"
            ]
            for r in records
        ],
        dtype=np.float64,
    )

    uniform_residuals = np.asarray(
        [
            r[
                "final_residual_px"
            ]
            for r in uniform_records
        ],
        dtype=np.float64,
    )

    metrics = {
        "pair_id":
            "pair_004",

        "pair_type":
            "cross_mission_cross_sensor",

        "source":
            {
                "mission":
                    "LRO",

                "instrument":
                    "LROC NAC LEFT",

                "product":
                    "M185196277LE",

                "observation_date":
                    "2012-02-29",
            },

        "reference":
            {
                "mission":
                    "Chandrayaan-2",

                "instrument":
                    "TMC-2",

                "product":
                    "ch2_tmc_ndn_20231025T1956513800_d_oth_d18",

                "observation_date":
                    "2023-10-25",
            },

        "canonical":
            {
                "width":
                    int(w),

                "height":
                    int(h),

                "pixel_resolution_m":
                    PIXEL_SIZE_M,

                "science_valid_pixels":
                    int(
                        science_valid.sum()
                    ),

                "matcher_valid_pixels":
                    int(
                        matcher_valid.sum()
                    ),
            },

        "final_method":
            {
                "representation":
                    "gradient_magnitude",

                "feature_extractor":
                    "SuperPoint",

                "matcher":
                    "LightGlue",

                "strategy":
                    "native_resolution_overlapping_tiles",

                "geometric_estimator":
                    "affine_RANSAC",
            },

        "real_pair_result":
            {
                "distinct_candidates":
                    matcher_metrics[
                        "distinct_candidates"
                    ],

                "ransac_inliers":
                    matcher_metrics[
                        "ransac_inliers"
                    ],

                "ransac_inlier_ratio":
                    matcher_metrics[
                        "ransac_inlier_ratio"
                    ],

                "ransac_reprojection_rmse_px":
                    matcher_metrics[
                        "ransac_reprojection_rmse_px"
                    ],

                "ransac_reprojection_rmse_nominal_m":
                    float(
                        matcher_metrics[
                            "ransac_reprojection_rmse_px"
                        ]
                        * PIXEL_SIZE_M
                    ),

                "median_residual_px":
                    matcher_metrics[
                        "median_residual_px"
                    ],

                "mean_residual_px":
                    matcher_metrics[
                        "mean_residual_px"
                    ],

                "coverage_cells":
                    matcher_metrics[
                        "coverage_cells"
                    ],

                "valid_coverage_cells":
                    matcher_metrics[
                        "valid_coverage_cells"
                    ],

                "coverage_ratio":
                    matcher_metrics[
                        "coverage_ratio"
                    ],

                "affine":
                    affine_matrix.tolist(),

                "affine_parameters":
                    affine_info,

                "transform_sane":
                    matcher_metrics[
                        "transform_sane"
                    ],
            },

        "uniform_match_product":
            {
                "all_ransac_inliers":
                    int(
                        len(records)
                    ),

                "uniform_matches":
                    int(
                        len(
                            uniform_records
                        )
                    ),

                "max_matches_per_grid_cell":
                    UNIFORM_MATCHES_PER_CELL,

                "occupied_grid_cells":
                    int(
                        uniform_cells
                    ),

                "valid_grid_cells":
                    int(
                        valid_cells
                    ),

                "coverage_ratio":
                    float(
                        uniform_cells
                        / valid_cells
                        if valid_cells
                        else 0.0
                    ),

                "uniform_residual_rmse_px":
                    (
                        float(
                            np.sqrt(
                                np.mean(
                                    uniform_residuals
                                    ** 2
                                )
                            )
                        )
                        if uniform_residuals.size
                        else None
                    ),
            },

        "registered_product":
            {
                "filename":
                    registered_path.name,

                "valid_pixels":
                    int(
                        registered_valid.sum()
                    ),

                "nodata_value":
                    output_nodata,

                "target_grid":
                    "TMC-2 canonical 5 m lunar polar stereographic grid",
            },

        "controlled_cross_sensor_perturbation_validation":
            validation_summary,

        "interpretation_notes":
            [
                (
                    "The real-pair RANSAC reprojection RMSE "
                    "is an internal/self-consistency residual."
                ),

                (
                    "The nominal metre conversion is "
                    "pixel residual multiplied by the 5 m "
                    "canonical grid spacing; it is not "
                    "independent absolute registration accuracy."
                ),

                (
                    "Controlled perturbation-response RMSE "
                    "measures whether the real cross-sensor "
                    "matcher follows explicitly injected "
                    "source motion."
                ),

                (
                    "The controlled perturbation experiment "
                    "does not provide independent absolute "
                    "ground truth for the unperturbed "
                    "LRO NAC to TMC-2 pair."
                ),

                (
                    "Zero-DN TMC shadow pixels are retained "
                    "as science-valid data but excluded from "
                    "feature extraction."
                ),
            ],
    }

    metrics_path = (
        OUT_DIR
        / "pair004_final_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL PRINT
    # --------------------------------------------------------

    print("\n")
    print("=" * 78)

    print(
        "PAIR 004 FINAL PRODUCT SUMMARY"
    )

    print("=" * 78)

    print(
        "Method:"
    )

    print(
        "Gradient magnitude "
        "+ tiled SuperPoint/LightGlue "
        "+ affine RANSAC"
    )

    print(
        "\nCandidates:",
        matcher_metrics[
            "distinct_candidates"
        ],
    )

    print(
        "RANSAC inliers:",
        matcher_metrics[
            "ransac_inliers"
        ],
    )

    print(
        "Inlier ratio:",
        f"{matcher_metrics['ransac_inlier_ratio']:.6f}",
    )

    print(
        "RANSAC self RMSE:",
        f"{matcher_metrics['ransac_reprojection_rmse_px']:.6f}px",
    )

    print(
        "Coverage:",
        f"{matcher_metrics['coverage_cells']}/"
        f"{matcher_metrics['valid_coverage_cells']} "
        f"({matcher_metrics['coverage_ratio']:.4f})",
    )

    print(
        "Uniform output points:",
        len(
            uniform_records
        ),
    )

    print(
        "\nPerturbation validation mean:",
        f"{validation_summary['mean_perturbation_response_rmse_px']:.6f}px",
    )

    print(
        "Perturbation validation worst:",
        f"{validation_summary['worst_perturbation_response_rmse_px']:.6f}px",
    )

    print(
        "\nAffine:"
    )

    print(
        affine_matrix
    )

    print("\n")
    print("=" * 78)

    print(
        "OUTPUTS"
    )

    print("=" * 78)

    print(
        registered_path
    )

    print(
        all_matches_path
    )

    print(
        uniform_matches_path
    )

    print(
        metrics_path
    )

    print(
        overlay_path
    )

    print(
        checkerboard_path
    )

    print(
        final_matches_image
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Real-pair RANSAC RMSE remains "
        "an internal self-consistency metric."
    )

    print(
        "Controlled perturbation RMSE validates "
        "motion-following, not absolute unperturbed "
        "NAC-to-TMC ground-truth accuracy."
    )

    print(
        "\nPAIR 004 FINALIZATION COMPLETE."
    )


if __name__ == "__main__":
    main()