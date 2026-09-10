import math

import numpy as np

from src.evaluation.metrics import reprojection_errors
from src.geometry.ransac_filter import filter_matches_ransac


def decompose_affine(matrix):
    """
    One shared decomposition of a 2x3 affine matrix into physically
    meaningful parameters. v1 had ~4 near-duplicate copies of this
    logic scattered across scripts; v2 uses this single version.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    linear = matrix[:, :2]

    col0 = linear[:, 0]
    col1 = linear[:, 1]

    scale_x = float(np.linalg.norm(col0))
    scale_y = float(np.linalg.norm(col1))

    determinant = float(np.linalg.det(linear))

    if scale_x > 0 and scale_y > 0:
        axis_dot = float(np.dot(col0, col1) / (scale_x * scale_y))
    else:
        axis_dot = float("nan")

    rotation_deg = float(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))

    translation_x = float(matrix[0, 2])
    translation_y = float(matrix[1, 2])

    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "rotation_deg": rotation_deg,
        "translation_x_px": translation_x,
        "translation_y_px": translation_y,
        "translation_magnitude_px": float(math.hypot(translation_x, translation_y)),
        "determinant": determinant,
        "axis_dot": axis_dot,
    }


def compose_affine(outer, inner):
    """
    Compose two 2x3 affine matrices: apply `inner` first, then `outer`
    (matches function-composition order, outer(inner(x))). Needed for
    perturbation validation: expected_affine = compose(baseline_affine,
    inverse(perturbation)).
    """

    def to_3x3(m):
        full = np.eye(3, dtype=np.float64)
        full[:2, :] = np.asarray(m, dtype=np.float64)
        return full

    result = to_3x3(outer) @ to_3x3(inner)

    return result[:2, :]


def affine_sanity(params, config):
    """
    Physical plausibility check on a decomposed affine. Deliberately has
    NO translation-magnitude cap -- v1 learned the hard way (Pair006/007)
    that a large translation is not automatically an insane transform;
    only scale/rotation/skew/orientation are checked.
    """

    return bool(
        params["determinant"] > 0
        and config["scale_min"] <= params["scale_x"] <= config["scale_max"]
        and config["scale_min"] <= params["scale_y"] <= config["scale_max"]
        and abs(params["rotation_deg"]) <= config["max_rotation_deg"]
        and abs(params["axis_dot"]) <= config["max_axis_dot"]
    )


def fit_constrained_affine(source_points, reference_points, config):
    """
    Robust affine fit through a point set, via v1's existing
    `filter_matches_ransac` (unchanged, already generic). Returns None
    if there aren't enough points to fit.
    """

    source_points = np.asarray(source_points, dtype=np.float32)
    reference_points = np.asarray(reference_points, dtype=np.float32)

    if len(source_points) < 3:
        return None

    matrix, inlier_mask, inlier_source, inlier_reference = filter_matches_ransac(
        source_points,
        reference_points,
        reprojection_threshold=config["ransac_reprojection_threshold_px"],
    )

    params = decompose_affine(matrix)
    sane = affine_sanity(params, config)

    return {
        "matrix": matrix,
        "params": params,
        "sane": sane,
        "inlier_count": int(inlier_mask.sum()),
        "candidate_count": int(len(source_points)),
        "inlier_ratio": float(inlier_mask.sum()) / float(len(source_points)),
        "inlier_source_points": inlier_source,
        "inlier_reference_points": inlier_reference,
    }


def residual_consistency(source_points, reference_points, matrix, max_residual_px):
    """
    Reprojection residual of every point against the fitted affine
    (reusing `src/evaluation/metrics.py::reprojection_errors`, already
    generic). Points beyond `max_residual_px` are flagged as fit
    outliers -- reported, never silently dropped.
    """

    source_points = np.asarray(source_points, dtype=np.float32)
    reference_points = np.asarray(reference_points, dtype=np.float32)

    residuals = reprojection_errors(source_points, reference_points, matrix)

    is_outlier = residuals > max_residual_px

    return residuals, is_outlier
