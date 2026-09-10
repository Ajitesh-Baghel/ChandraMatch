import cv2
import numpy as np
from scipy import ndimage


def compute_interior_roi_mask(reference_mask, source_mask, baseline_affine, margin_px):
    """
    The literal fix for the ChatGPT-handoff's "Lesson 8": v1's
    perturbation tests let the common support shrink as pixels warp off
    the fixed canvas (e.g. Pair007's common support dropped from 815,047
    to 235,080 pixels after just the baseline ~903px warp), so a
    perturbation test could fail from boundary support loss rather than
    real matcher instability.

    Computes the REAL common overlap once source is registered by the
    baseline affine (reference_mask AND baseline-warped source_mask --
    not reference_mask alone), then erodes it by `margin_px` so that
    every one of the small test perturbations (bounded by config's
    `perturbations` list) keeps every ROI pixel's source correspondence
    inside the original valid canvas. A FAIL measured only inside this
    ROI is about real instability, not a boundary artifact.
    """

    height, width = reference_mask.shape

    warped_source_mask = (
        cv2.warpAffine(
            source_mask.astype(np.uint8),
            np.asarray(baseline_affine, dtype=np.float64),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )

    common_after_baseline = reference_mask & warped_source_mask

    if margin_px <= 0:
        return common_after_baseline

    return ndimage.binary_erosion(
        common_after_baseline,
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(round(margin_px)),
        border_value=0,
    )


def find_max_safe_margin(
    reference_mask,
    source_mask,
    baseline_affine,
    min_roi_pixels,
    search_max_px=200,
    step_px=5,
):
    """
    Largest erosion margin (searched top-down from `search_max_px`) whose
    resulting interior ROI still has at least `min_roi_pixels`. Returns
    0 if even no erosion at all can't reach `min_roi_pixels` (post-
    baseline common overlap itself is too small).

    Used when the pair's real post-registration overlap is too thin to
    fit the standard perturbation set's safety margin at all (e.g.
    Pair007) -- rather than silently skipping validation, this finds
    the largest margin the pair's own geometry can actually support, so
    perturbations can be scaled down to fit it as an explicitly
    lower-confidence supplementary check (see
    `stage2_validation.run_reduced_scale_perturbation_validation`).
    """

    for margin in range(search_max_px, -1, -step_px):
        roi = compute_interior_roi_mask(
            reference_mask, source_mask, baseline_affine, margin
        )

        if np.count_nonzero(roi) >= min_roi_pixels:
            return margin

    return 0


def perturbation_margin_px(perturbations, buffer_px, canvas_shape=None):
    """
    Erosion margin: the largest translation magnitude among the
    configured test perturbations, plus a fixed safety buffer, plus (if
    `canvas_shape` is given) an explicit bound on rotation/scale-induced
    corner displacement.

    That displacement is NOT always negligible relative to the fixed
    buffer alone: rotation/scale in `create_affine_transform` pivot
    around the canvas's own center, so a point at the canvas's half-
    diagonal distance R moves by roughly R*|theta_rad| (rotation) plus
    R*|scale-1| (scale) beyond the pure translation -- for a large
    canvas this can exceed a small fixed buffer even for the
    sub-degree, near-unit-scale perturbations this project uses.
    Without `canvas_shape`, falls back to the fixed-buffer-only
    approximation (safe for small/medium canvases, e.g. the reduced-
    scale supplementary test's own re-scaled perturbations).
    """

    max_translation = max(
        float(np.hypot(p["tx"], p["ty"])) for p in perturbations
    )

    rotation_scale_margin = 0.0

    if canvas_shape is not None:
        height, width = canvas_shape
        half_diagonal = 0.5 * float(np.hypot(height, width))

        for p in perturbations:
            theta_rad = abs(np.radians(p.get("rotation_deg", 0.0)))
            scale_delta = abs(p.get("scale", 1.0) - 1.0)
            rotation_scale_margin = max(
                rotation_scale_margin, half_diagonal * (theta_rad + scale_delta)
            )

    return max_translation + rotation_scale_margin + buffer_px
