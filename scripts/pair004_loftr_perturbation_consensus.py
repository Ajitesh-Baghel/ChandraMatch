from pathlib import Path
import csv
import json
import math
from collections import Counter

import cv2
import numpy as np
import rasterio
from scipy import ndimage
from scipy.spatial import cKDTree

import torch
import kornia.feature as KF


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pair_004"
    / "canonical"
)

SOURCE_PATH = (
    PAIR_DIR
    / "lro_nac_source.tif"
)

REFERENCE_PATH = (
    PAIR_DIR
    / "tmc2_reference.tif"
)

COMMON_MASK_PATH = (
    PAIR_DIR
    / "common_valid_mask.tif"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_004"
    / "loftr_perturbation_consensus"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PIXEL_SIZE_M = 5.0

LOFTR_MAX_DIM = 1600

MASK_EROSION_PIXELS = 5

# Correspondences from different perturbation runs are considered
# the same physical match only if BOTH source and reference
# positions agree to these tolerances.
SOURCE_CONSENSUS_TOL_PX = 8.0
REFERENCE_CONSENSUS_TOL_PX = 8.0

# Identity anchor + at least two independent perturbations.
MIN_RUN_SUPPORT = 3

# Suppress duplicate consensus points.
DEDUP_SOURCE_TOL_PX = 4.0
DEDUP_REFERENCE_TOL_PX = 4.0

RANSAC_THRESHOLD_PX = 3.0

GRID_ROWS = 8
GRID_COLS = 8


# ============================================================
# PERTURBATIONS
# ============================================================

PERTURBATIONS = [
    {
        "id": "IDENTITY",
        "tx": 0.0,
        "ty": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
    },

    # The perturbation on which global LoFTR previously failed.
    {
        "id": "SHIFT_A",
        "tx": 64.0,
        "ty": -48.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
    },

    # Opposite-direction translation.
    {
        "id": "SHIFT_B",
        "tx": -64.0,
        "ty": 48.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
    },

    {
        "id": "SHIFT_ROT_A",
        "tx": -120.0,
        "ty": 80.0,
        "rotation_deg": 0.35,
        "scale": 1.0,
    },

    {
        "id": "SHIFT_ROT_SCALE",
        "tx": 80.0,
        "ty": 60.0,
        "rotation_deg": -0.50,
        "scale": 1.002,
    },

    {
        "id": "MILD_ROT",
        "tx": 40.0,
        "ty": -30.0,
        "rotation_deg": 0.20,
        "scale": 1.0,
    },
]


# ============================================================
# IO
# ============================================================

def read_band(path):
    with rasterio.open(path) as ds:
        return (
            ds.read(1),
            ds.read_masks(1) > 0,
            ds.nodata,
        )


# ============================================================
# IMAGE REPRESENTATION
# ============================================================

def robust_uint8(data, mask):
    out = np.zeros(
        data.shape,
        dtype=np.uint8,
    )

    values = data[mask]
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

    scaled = (
        data.astype(np.float32)
        - float(low)
    ) / float(high - low)

    scaled = np.clip(
        scaled,
        0.0,
        1.0,
    )

    out[mask] = np.round(
        scaled[mask] * 255.0
    ).astype(np.uint8)

    return out


def gradient_uint8(image, mask):
    img = (
        image.astype(np.float32)
        / 255.0
    )

    img = cv2.GaussianBlur(
        img,
        (5, 5),
        0.8,
    )

    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        img,
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

    values = magnitude[mask]

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

    out[mask] = np.round(
        magnitude[mask]
        * 255.0
    ).astype(np.uint8)

    return out


# ============================================================
# AFFINE UTILITIES
# ============================================================

def make_forward_affine(
    width,
    height,
    tx,
    ty,
    rotation_deg,
    scale,
):
    center = (
        (width - 1) / 2.0,
        (height - 1) / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        rotation_deg,
        scale,
    ).astype(np.float64)

    matrix[0, 2] += tx
    matrix[1, 2] += ty

    return matrix


def affine_points(matrix, points):
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

    if scale_x > 0 and scale_y > 0:
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
        "scale_x": scale_x,
        "scale_y": scale_y,
        "rotation_deg": rotation_deg,
        "translation_x_px": tx,
        "translation_y_px": ty,
        "translation_magnitude_px": float(
            math.hypot(tx, ty)
        ),
        "determinant": determinant,
        "axis_dot": axis_dot,
    }


def affine_sanity(info):
    return bool(
        info["determinant"] > 0
        and 0.90 <= info["scale_x"] <= 1.10
        and 0.90 <= info["scale_y"] <= 1.10
        and abs(
            info["rotation_deg"]
        ) <= 5.0
        and info[
            "translation_magnitude_px"
        ] <= 100.0
        and abs(
            info["axis_dot"]
        ) <= 0.15
    )


# ============================================================
# RESIZE
# ============================================================

def resize_max(
    image,
    max_dim,
):
    h, w = image.shape

    if max(h, w) <= max_dim:
        return image.copy(), 1.0

    scale = (
        max_dim
        / max(h, w)
    )

    new_w = max(
        1,
        int(round(w * scale)),
    )

    new_h = max(
        1,
        int(round(h * scale)),
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


# ============================================================
# MASK
# ============================================================

def points_inside_mask(
    points,
    mask,
):
    if len(points) == 0:
        return np.zeros(
            0,
            dtype=bool,
        )

    h, w = mask.shape

    x = np.rint(
        points[:, 0]
    ).astype(np.int64)

    y = np.rint(
        points[:, 1]
    ).astype(np.int64)

    inside = (
        (x >= 0)
        & (x < w)
        & (y >= 0)
        & (y < h)
    )

    result = np.zeros(
        len(points),
        dtype=bool,
    )

    ids = np.where(
        inside
    )[0]

    result[ids] = mask[
        y[ids],
        x[ids],
    ]

    return result


# ============================================================
# LOFTR
# ============================================================

def run_loftr(
    matcher,
    source_image,
    reference_image,
    device,
):
    source_small, scale = (
        resize_max(
            source_image,
            LOFTR_MAX_DIM,
        )
    )

    reference_small, scale_ref = (
        resize_max(
            reference_image,
            LOFTR_MAX_DIM,
        )
    )

    if abs(
        scale - scale_ref
    ) > 1e-8:
        raise RuntimeError(
            "LoFTR resize scales differ."
        )

    image0 = torch.from_numpy(
        source_small.astype(
            np.float32
        ) / 255.0
    )[None, None].to(device)

    image1 = torch.from_numpy(
        reference_small.astype(
            np.float32
        ) / 255.0
    )[None, None].to(device)

    with torch.inference_mode():
        prediction = matcher(
            {
                "image0": image0,
                "image1": image1,
            }
        )

    source_points = (
        prediction[
            "keypoints0"
        ]
        .detach()
        .cpu()
        .numpy()
        / scale
    )

    reference_points = (
        prediction[
            "keypoints1"
        ]
        .detach()
        .cpu()
        .numpy()
        / scale
    )

    return (
        source_points.astype(
            np.float32
        ),
        reference_points.astype(
            np.float32
        ),
    )


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def spatial_coverage(
    points,
    valid_mask,
):
    h, w = valid_mask.shape

    valid_cells = set()

    for gy in range(
        GRID_ROWS
    ):

        y0 = int(
            round(
                gy * h
                / GRID_ROWS
            )
        )

        y1 = int(
            round(
                (gy + 1) * h
                / GRID_ROWS
            )
        )

        for gx in range(
            GRID_COLS
        ):

            x0 = int(
                round(
                    gx * w
                    / GRID_COLS
                )
            )

            x1 = int(
                round(
                    (gx + 1) * w
                    / GRID_COLS
                )
            )

            if np.any(
                valid_mask[
                    y0:y1,
                    x0:x1,
                ]
            ):
                valid_cells.add(
                    (gy, gx)
                )

    occupied = set()

    for x, y in points:

        gx = min(
            GRID_COLS - 1,
            max(
                0,
                int(
                    x / w
                    * GRID_COLS
                ),
            ),
        )

        gy = min(
            GRID_ROWS - 1,
            max(
                0,
                int(
                    y / h
                    * GRID_ROWS
                ),
            ),
        )

        cell = (
            gy,
            gx,
        )

        if cell in valid_cells:
            occupied.add(
                cell
            )

    denominator = len(
        valid_cells
    )

    numerator = len(
        occupied
    )

    ratio = (
        numerator
        / denominator
        if denominator
        else 0.0
    )

    return (
        numerator,
        denominator,
        ratio,
    )


# ============================================================
# RUN ONE PERTURBATION
# ============================================================

def run_perturbation(
    perturbation,
    source_intensity,
    reference_gradient,
    matcher_valid,
    matcher,
    device,
):
    h, w = (
        source_intensity.shape
    )

    forward = (
        make_forward_affine(
            w,
            h,
            perturbation["tx"],
            perturbation["ty"],
            perturbation[
                "rotation_deg"
            ],
            perturbation[
                "scale"
            ],
        )
    )

    inverse = (
        cv2.invertAffineTransform(
            forward
        )
    )

    # --------------------------------------------------------
    # Warp source image and valid support
    # --------------------------------------------------------

    transformed_intensity = (
        cv2.warpAffine(
            source_intensity,
            forward,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=(
                cv2.BORDER_CONSTANT
            ),
            borderValue=0,
        )
    )

    transformed_mask = (
        cv2.warpAffine(
            matcher_valid.astype(
                np.uint8
            ),
            forward,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=(
                cv2.BORDER_CONSTANT
            ),
            borderValue=0,
        )
        > 0
    )

    # Small erosion removes the new warp boundary.
    transformed_mask = (
        ndimage.binary_erosion(
            transformed_mask,
            structure=np.ones(
                (3, 3),
                dtype=bool,
            ),
            iterations=2,
            border_value=0,
        )
    )

    transformed_gradient = (
        gradient_uint8(
            transformed_intensity,
            transformed_mask,
        )
    )

    transformed_gradient[
        ~transformed_mask
    ] = 0

    # --------------------------------------------------------
    # Match perturbed NAC -> original TMC
    # --------------------------------------------------------

    src_transformed, ref = (
        run_loftr(
            matcher,
            transformed_gradient,
            reference_gradient,
            device,
        )
    )

    raw_matches = len(
        src_transformed
    )

    valid = (
        points_inside_mask(
            src_transformed,
            transformed_mask,
        )
        &
        points_inside_mask(
            ref,
            matcher_valid,
        )
    )

    src_transformed = (
        src_transformed[valid]
    )

    ref = ref[valid]

    # --------------------------------------------------------
    # Transform source coordinates BACK to original NAC grid.
    # --------------------------------------------------------

    src_original = (
        affine_points(
            inverse,
            src_transformed,
        )
        .astype(np.float32)
    )

    valid_original = (
        points_inside_mask(
            src_original,
            matcher_valid,
        )
    )

    src_original = (
        src_original[
            valid_original
        ]
    )

    ref = ref[
        valid_original
    ]

    print(
        f"{perturbation['id']:<18} "
        f"raw={raw_matches:<5} "
        f"canonical={len(src_original):<5}"
    )

    return {
        "id":
            perturbation["id"],

        "forward_affine":
            forward,

        "inverse_affine":
            inverse,

        "source_points":
            src_original,

        "reference_points":
            ref,

        "raw_matches":
            int(raw_matches),

        "canonical_matches":
            int(
                len(src_original)
            ),
    }


# ============================================================
# CONSENSUS
# ============================================================

def build_consensus(
    runs,
):
    identity = None

    for run in runs:
        if run["id"] == "IDENTITY":
            identity = run
            break

    if identity is None:
        raise RuntimeError(
            "Identity run missing."
        )

    anchor_src = (
        identity[
            "source_points"
        ]
    )

    anchor_ref = (
        identity[
            "reference_points"
        ]
    )

    # KD trees built over compensated ORIGINAL source positions.
    trees = {}

    for run in runs:

        if run["id"] == "IDENTITY":
            continue

        if len(
            run["source_points"]
        ) == 0:
            continue

        trees[run["id"]] = (
            cKDTree(
                run[
                    "source_points"
                ]
            )
        )

    candidates = []

    # --------------------------------------------------------
    # Identity matches serve only as anchor locations.
    #
    # A match is retained if equivalent content-based
    # correspondences reappear after independent source
    # perturbations.
    # --------------------------------------------------------

    for anchor_index, (
        src0,
        ref0,
    ) in enumerate(
        zip(
            anchor_src,
            anchor_ref,
        )
    ):

        members_src = [
            src0.astype(
                np.float64
            )
        ]

        members_ref = [
            ref0.astype(
                np.float64
            )
        ]

        support_ids = [
            "IDENTITY"
        ]

        for run in runs:

            run_id = run["id"]

            if run_id == "IDENTITY":
                continue

            if run_id not in trees:
                continue

            tree = trees[
                run_id
            ]

            nearby = (
                tree.query_ball_point(
                    src0,
                    r=(
                        SOURCE_CONSENSUS_TOL_PX
                    ),
                )
            )

            if len(nearby) == 0:
                continue

            run_src = (
                run[
                    "source_points"
                ]
            )

            run_ref = (
                run[
                    "reference_points"
                ]
            )

            best_index = None
            best_score = None

            for idx in nearby:

                src_distance = float(
                    np.linalg.norm(
                        run_src[idx]
                        - src0
                    )
                )

                ref_distance = float(
                    np.linalg.norm(
                        run_ref[idx]
                        - ref0
                    )
                )

                if (
                    ref_distance
                    > REFERENCE_CONSENSUS_TOL_PX
                ):
                    continue

                score = (
                    src_distance
                    + ref_distance
                )

                if (
                    best_score is None
                    or score < best_score
                ):
                    best_score = score
                    best_index = idx

            if best_index is None:
                continue

            members_src.append(
                run_src[
                    best_index
                ].astype(
                    np.float64
                )
            )

            members_ref.append(
                run_ref[
                    best_index
                ].astype(
                    np.float64
                )
            )

            support_ids.append(
                run_id
            )

        support = len(
            support_ids
        )

        if (
            support
            < MIN_RUN_SUPPORT
        ):
            continue

        members_src = (
            np.asarray(
                members_src,
                dtype=np.float64,
            )
        )

        members_ref = (
            np.asarray(
                members_ref,
                dtype=np.float64,
            )
        )

        # Median across independent perturbations is robust
        # to one bad localization.
        consensus_src = (
            np.median(
                members_src,
                axis=0,
            )
        )

        consensus_ref = (
            np.median(
                members_ref,
                axis=0,
            )
        )

        src_spread = (
            np.linalg.norm(
                members_src
                - consensus_src,
                axis=1,
            )
        )

        ref_spread = (
            np.linalg.norm(
                members_ref
                - consensus_ref,
                axis=1,
            )
        )

        candidates.append(
            {
                "source":
                    consensus_src,

                "reference":
                    consensus_ref,

                "support":
                    support,

                "support_ids":
                    support_ids,

                "source_spread_median_px":
                    float(
                        np.median(
                            src_spread
                        )
                    ),

                "source_spread_max_px":
                    float(
                        np.max(
                            src_spread
                        )
                    ),

                "reference_spread_median_px":
                    float(
                        np.median(
                            ref_spread
                        )
                    ),

                "reference_spread_max_px":
                    float(
                        np.max(
                            ref_spread
                        )
                    ),
            }
        )

    return candidates


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_consensus(
    candidates,
):
    # Strongest and most stable candidates first.
    ordered = sorted(
        candidates,
        key=lambda c: (
            -c["support"],
            c[
                "source_spread_median_px"
            ]
            + c[
                "reference_spread_median_px"
            ],
        ),
    )

    selected = []

    for candidate in ordered:

        src = candidate[
            "source"
        ]

        ref = candidate[
            "reference"
        ]

        duplicate = False

        for existing in selected:

            src_distance = float(
                np.linalg.norm(
                    src
                    - existing[
                        "source"
                    ]
                )
            )

            ref_distance = float(
                np.linalg.norm(
                    ref
                    - existing[
                        "reference"
                    ]
                )
            )

            if (
                src_distance
                <= DEDUP_SOURCE_TOL_PX
                and
                ref_distance
                <= DEDUP_REFERENCE_TOL_PX
            ):
                duplicate = True
                break

        if not duplicate:
            selected.append(
                candidate
            )

    return selected


# ============================================================
# VISUALIZATION
# ============================================================

def save_matches(
    source,
    reference,
    source_points,
    reference_points,
    path,
):
    h, w = source.shape

    max_dim = 1800

    scale = min(
        1.0,
        max_dim / max(h, w),
    )

    dw = int(
        round(w * scale)
    )

    dh = int(
        round(h * scale)
    )

    src_small = cv2.resize(
        source,
        (dw, dh),
        interpolation=cv2.INTER_AREA,
    )

    ref_small = cv2.resize(
        reference,
        (dw, dh),
        interpolation=cv2.INTER_AREA,
    )

    src_rgb = cv2.cvtColor(
        src_small,
        cv2.COLOR_GRAY2BGR,
    )

    ref_rgb = cv2.cvtColor(
        ref_small,
        cv2.COLOR_GRAY2BGR,
    )

    canvas = np.concatenate(
        [
            src_rgb,
            ref_rgb,
        ],
        axis=1,
    )

    if len(source_points) > 400:

        ids = np.linspace(
            0,
            len(source_points) - 1,
            400,
        ).astype(int)

        source_points = (
            source_points[ids]
        )

        reference_points = (
            reference_points[ids]
        )

    for src, ref in zip(
        source_points,
        reference_points,
    ):

        x0 = int(
            round(
                src[0]
                * scale
            )
        )

        y0 = int(
            round(
                src[1]
                * scale
            )
        )

        x1 = int(
            round(
                ref[0]
                * scale
            )
        ) + dw

        y1 = int(
            round(
                ref[1]
                * scale
            )
        )

        cv2.circle(
            canvas,
            (x0, y0),
            3,
            (0, 255, 0),
            -1,
        )

        cv2.circle(
            canvas,
            (x1, y1),
            3,
            (0, 255, 0),
            -1,
        )

        cv2.line(
            canvas,
            (x0, y0),
            (x1, y1),
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(path),
        canvas,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print(
        "PAIR 004 — LOFTR "
        "PERTURBATION-CONSENSUS FILTER"
    )
    print("=" * 78)

    source, source_mask, _ = (
        read_band(
            SOURCE_PATH
        )
    )

    reference, reference_mask, _ = (
        read_band(
            REFERENCE_PATH
        )
    )

    common_data, _, _ = (
        read_band(
            COMMON_MASK_PATH
        )
    )

    common = (
        common_data > 0
    )

    if (
        source.shape
        != reference.shape
    ):
        raise RuntimeError(
            "Canonical shapes differ."
        )

    h, w = source.shape

    print(
        "\nCanonical shape:",
        h,
        "x",
        w,
    )

    # --------------------------------------------------------
    # Science validity
    # --------------------------------------------------------

    science_valid = (
        common
        & source_mask
        & reference_mask
        & np.isfinite(source)
        & np.isfinite(reference)
    )

    # Completely black TMC shadow interior excluded only from
    # feature extraction, not from science validity.
    matcher_valid = (
        science_valid
        & (reference > 0)
    )

    matcher_valid = (
        ndimage.binary_erosion(
            matcher_valid,
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

    print(
        "Science-valid pixels:",
        f"{int(science_valid.sum()):,}",
    )

    print(
        "Matcher-valid pixels:",
        f"{int(matcher_valid.sum()):,}",
    )

    # --------------------------------------------------------
    # Gradient representation
    # --------------------------------------------------------

    source_intensity = (
        robust_uint8(
            source,
            matcher_valid,
        )
    )

    reference_intensity = (
        robust_uint8(
            reference,
            matcher_valid,
        )
    )

    source_intensity[
        ~matcher_valid
    ] = 0

    reference_intensity[
        ~matcher_valid
    ] = 0

    reference_gradient = (
        gradient_uint8(
            reference_intensity,
            matcher_valid,
        )
    )

    reference_gradient[
        ~matcher_valid
    ] = 0

    cv2.imwrite(
        str(
            OUT_DIR
            / "reference_gradient.png"
        ),
        reference_gradient,
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    matcher = KF.LoFTR(
        pretrained="outdoor",
    ).eval().to(device)

    # --------------------------------------------------------
    # Independent perturbation runs
    # --------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("PERTURBATION RUNS")
    print("=" * 78)

    runs = []

    for perturbation in (
        PERTURBATIONS
    ):

        run = run_perturbation(
            perturbation,
            source_intensity,
            reference_gradient,
            matcher_valid,
            matcher,
            device,
        )

        runs.append(
            run
        )

    # --------------------------------------------------------
    # Consensus
    # --------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("CONSENSUS")
    print("=" * 78)

    candidates = (
        build_consensus(
            runs
        )
    )

    print(
        "Consensus before dedup:",
        len(candidates),
    )

    consensus = (
        deduplicate_consensus(
            candidates
        )
    )

    print(
        "Consensus after dedup :",
        len(consensus),
    )

    if len(consensus) < 4:
        raise RuntimeError(
            "Too few stable "
            "perturbation-consensus matches."
        )

    support_distribution = Counter(
        candidate["support"]
        for candidate in consensus
    )

    print(
        "Support distribution  :",
        dict(
            sorted(
                support_distribution.items()
            )
        ),
    )

    source_points = np.asarray(
        [
            c["source"]
            for c in consensus
        ],
        dtype=np.float32,
    )

    reference_points = np.asarray(
        [
            c["reference"]
            for c in consensus
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Final RANSAC
    # --------------------------------------------------------

    matrix, inlier_mask = (
        cv2.estimateAffine2D(
            source_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=(
                RANSAC_THRESHOLD_PX
            ),
            maxIters=20000,
            confidence=0.999,
            refineIters=100,
        )
    )

    if (
        matrix is None
        or inlier_mask is None
    ):
        raise RuntimeError(
            "Consensus RANSAC failed."
        )

    inlier_mask = (
        inlier_mask.ravel() > 0
    )

    src_in = (
        source_points[
            inlier_mask
        ]
    )

    ref_in = (
        reference_points[
            inlier_mask
        ]
    )

    consensus_inliers = [
        consensus[i]
        for i in np.where(
            inlier_mask
        )[0]
    ]

    predicted = (
        affine_points(
            matrix,
            src_in,
        )
    )

    residuals = np.linalg.norm(
        predicted
        - ref_in,
        axis=1,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuals ** 2
            )
        )
    )

    median = float(
        np.median(
            residuals
        )
    )

    mean = float(
        np.mean(
            residuals
        )
    )

    (
        coverage_n,
        coverage_d,
        coverage_ratio,
    ) = spatial_coverage(
        ref_in,
        matcher_valid,
    )

    affine_info = (
        decompose_affine(
            matrix
        )
    )

    sane = affine_sanity(
        affine_info
    )

    # --------------------------------------------------------
    # Spread diagnostics
    # --------------------------------------------------------

    src_spread_median = float(
        np.median(
            [
                c[
                    "source_spread_median_px"
                ]
                for c in consensus_inliers
            ]
        )
    )

    ref_spread_median = float(
        np.median(
            [
                c[
                    "reference_spread_median_px"
                ]
                for c in consensus_inliers
            ]
        )
    )

    print("\n")
    print("=" * 78)
    print(
        "FINAL PERTURBATION-CONSENSUS RESULT"
    )
    print("=" * 78)

    print(
        "Stable candidates         :",
        len(consensus),
    )

    print(
        "RANSAC inliers            :",
        len(src_in),
    )

    print(
        "Inlier ratio              :",
        f"{len(src_in) / len(consensus):.6f}",
    )

    print(
        "RANSAC self RMSE px       :",
        f"{rmse:.6f}",
    )

    print(
        "Median residual px        :",
        f"{median:.6f}",
    )

    print(
        "Mean residual px          :",
        f"{mean:.6f}",
    )

    print(
        "Spatial coverage          :",
        f"{coverage_n}/"
        f"{coverage_d} "
        f"({coverage_ratio:.4f})",
    )

    print(
        "Median source run spread  :",
        f"{src_spread_median:.6f}px",
    )

    print(
        "Median reference spread   :",
        f"{ref_spread_median:.6f}px",
    )

    print(
        "Transform sane            :",
        sane,
    )

    print("\nAffine:")
    print(matrix)

    print(
        "\nTranslation px:",
        f"{affine_info['translation_magnitude_px']:.6f}",
    )

    print(
        "Rotation deg:",
        f"{affine_info['rotation_deg']:.6f}",
    )

    print(
        "Scale X:",
        f"{affine_info['scale_x']:.8f}",
    )

    print(
        "Scale Y:",
        f"{affine_info['scale_y']:.8f}",
    )

    print(
        "Determinant:",
        f"{affine_info['determinant']:.8f}",
    )

    print(
        "Axis dot:",
        f"{affine_info['axis_dot']:.8f}",
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        OUT_DIR
        / "pair004_consensus_inliers.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "source_x",
                "source_y",
                "reference_x",
                "reference_y",
                "support_runs",
                "support_ids",
                "source_spread_median_px",
                "source_spread_max_px",
                "reference_spread_median_px",
                "reference_spread_max_px",
                "ransac_residual_px",
            ]
        )

        for (
            candidate,
            src,
            ref,
            residual,
        ) in zip(
            consensus_inliers,
            src_in,
            ref_in,
            residuals,
        ):

            writer.writerow(
                [
                    float(src[0]),
                    float(src[1]),
                    float(ref[0]),
                    float(ref[1]),

                    int(
                        candidate[
                            "support"
                        ]
                    ),

                    ";".join(
                        candidate[
                            "support_ids"
                        ]
                    ),

                    float(
                        candidate[
                            "source_spread_median_px"
                        ]
                    ),

                    float(
                        candidate[
                            "source_spread_max_px"
                        ]
                    ),

                    float(
                        candidate[
                            "reference_spread_median_px"
                        ]
                    ),

                    float(
                        candidate[
                            "reference_spread_max_px"
                        ]
                    ),

                    float(
                        residual
                    ),
                ]
            )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    metrics = {
        "pair_id":
            "pair_004",

        "method":
            "loftr_gradient_perturbation_consensus",

        "number_of_runs":
            len(runs),

        "minimum_run_support":
            MIN_RUN_SUPPORT,

        "source_consensus_tolerance_px":
            SOURCE_CONSENSUS_TOL_PX,

        "reference_consensus_tolerance_px":
            REFERENCE_CONSENSUS_TOL_PX,

        "perturbation_runs": [
            {
                "id":
                    run["id"],

                "raw_matches":
                    run[
                        "raw_matches"
                    ],

                "canonical_matches":
                    run[
                        "canonical_matches"
                    ],
            }
            for run in runs
        ],

        "consensus_before_dedup":
            int(len(candidates)),

        "consensus_after_dedup":
            int(len(consensus)),

        "support_distribution":
            {
                str(k): int(v)
                for k, v
                in support_distribution.items()
            },

        "ransac_inliers":
            int(len(src_in)),

        "ransac_inlier_ratio":
            float(
                len(src_in)
                / len(consensus)
            ),

        "ransac_reprojection_rmse_px":
            rmse,

        "median_residual_px":
            median,

        "mean_residual_px":
            mean,

        "coverage_cells":
            int(coverage_n),

        "valid_coverage_cells":
            int(coverage_d),

        "coverage_ratio":
            float(
                coverage_ratio
            ),

        "median_source_run_spread_px":
            src_spread_median,

        "median_reference_run_spread_px":
            ref_spread_median,

        "affine":
            matrix.tolist(),

        "affine_parameters":
            affine_info,

        "transform_sane":
            sane,

        "note": (
            "Correspondences survive only if "
            "they recur after independent known "
            "perturbations of the LRO NAC source. "
            "RANSAC reprojection RMSE remains an "
            "internal/self-consistency residual, "
            "not independent real-pair ground truth."
        ),
    }

    json_path = (
        OUT_DIR
        / "pair004_consensus_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    source_gradient = (
        gradient_uint8(
            source_intensity,
            matcher_valid,
        )
    )

    source_gradient[
        ~matcher_valid
    ] = 0

    visualization_path = (
        OUT_DIR
        / "pair004_consensus_matches.png"
    )

    save_matches(
        source_gradient,
        reference_gradient,
        src_in,
        ref_in,
        visualization_path,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n")
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)

    print(json_path)
    print(csv_path)
    print(
        visualization_path
    )

    print("\nIMPORTANT:")
    print(
        "This experiment tests correspondence "
        "stability under deliberate source "
        "perturbation."
    )

    print(
        "The RANSAC residual is still NOT "
        "independent real-pair ground-truth "
        "registration accuracy."
    )

    print(
        "\nPAIR 004 PERTURBATION "
        "CONSENSUS COMPLETE."
    )


if __name__ == "__main__":
    main()