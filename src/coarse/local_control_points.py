import numpy as np

from src.coarse.masked_correlation import (
    ambiguity_ratio,
    channel_summed_surface,
    extract_topk_peaks,
    subpixel_shift_correction,
    surface_index_to_shift,
)
from src.coarse.verification import mask_bbox


def grid_points(bbox, spacing_px):
    """
    Uniform point grid over a (y0, y1, x0, x1) bounding box, spaced
    `spacing_px` apart and inset by half a spacing from the box edges so
    every point has room for a local window around it.
    """

    y0, y1, x0, x1 = bbox

    ys = range(y0 + spacing_px // 2, y1, spacing_px)
    xs = range(x0 + spacing_px // 2, x1, spacing_px)

    return [(y, x) for y in ys for x in xs]


def refine_point(
    source_stack,
    source_mask,
    reference_stack,
    reference_mask,
    center_y,
    center_x,
    global_dy,
    global_dx,
    window_px,
    min_valid_fraction,
    min_peak_score,
    min_ambiguity_ratio,
    max_residual_px,
    overlap_ratio,
):
    """
    Independently refine one control point's local displacement around
    the Stage 0 global anchor, and gate it.

    Same trick as Stage 0's local_cell_agreement: the source window is
    cropped pre-shifted by the global anchor, so the local masked-FFT
    correlation only has to resolve the small residual around (0, 0),
    which is then sub-pixel-corrected via parabolic interpolation.

    Returns a dict describing the point whether accepted or rejected --
    `accepted` is False and `reject_reasons` lists why for anything that
    didn't clear every gate (insufficient local support, weak/ambiguous
    local correlation, or an implausibly large residual).
    """

    height, width = reference_mask.shape
    half = window_px // 2

    ry0 = max(0, center_y - half)
    ry1 = min(height, center_y + half)
    rx0 = max(0, center_x - half)
    rx1 = min(width, center_x + half)

    base = {
        "reference_y": center_y,
        "reference_x": center_x,
        "source_y": None,
        "source_x": None,
        "residual_dy": None,
        "residual_dx": None,
        "residual_magnitude_px": None,
        "peak_score": None,
        "ambiguity_ratio": None,
        "valid_fraction": None,
        "accepted": False,
        "reject_reasons": [],
    }

    ref_window_mask = reference_mask[ry0:ry1, rx0:rx1]

    if ref_window_mask.size == 0:
        base["reject_reasons"] = ["empty reference window"]
        return base

    valid_fraction = float(np.count_nonzero(ref_window_mask)) / float(
        ref_window_mask.size
    )
    base["valid_fraction"] = valid_fraction

    if valid_fraction < min_valid_fraction:
        base["reject_reasons"] = [
            f"valid_fraction={valid_fraction:.3f} below min_valid_fraction="
            f"{min_valid_fraction}"
        ]
        return base

    sy0 = ry0 - global_dy
    sy1 = ry1 - global_dy
    sx0 = rx0 - global_dx
    sx1 = rx1 - global_dx

    if sy0 < 0 or sx0 < 0 or sy1 > height or sx1 > width:
        base["reject_reasons"] = [
            "source window falls outside canvas after global pre-shift"
        ]
        return base

    src_window_mask = source_mask[sy0:sy1, sx0:sx1]

    if src_window_mask.shape != ref_window_mask.shape:
        base["reject_reasons"] = ["source/reference window shape mismatch"]
        return base

    src_window_stack = source_stack[:, sy0:sy1, sx0:sx1]
    ref_window_stack = reference_stack[:, ry0:ry1, rx0:rx1]

    try:
        surface, ref_shape = channel_summed_surface(
            src_window_stack,
            src_window_mask,
            ref_window_stack,
            ref_window_mask,
            overlap_ratio=overlap_ratio,
        )
    except Exception as exc:
        base["reject_reasons"] = [f"local correlation failed: {exc!r}"]
        return base

    peaks = extract_topk_peaks(surface, top_k=2, nms_radius=5)

    if not peaks:
        base["reject_reasons"] = ["no local correlation peak found"]
        return base

    peak_score = peaks[0]["score"]
    ratio = ambiguity_ratio(peaks)

    residual_dy, residual_dx = surface_index_to_shift(peaks[0]["index"], ref_shape)
    sub_dy, sub_dx = subpixel_shift_correction(surface, peaks[0]["index"])

    residual_dy += sub_dy
    residual_dx += sub_dx

    residual_magnitude = float(np.hypot(residual_dy, residual_dx))

    total_dy = global_dy + residual_dy
    total_dx = global_dx + residual_dx

    base.update(
        {
            "source_y": center_y - total_dy,
            "source_x": center_x - total_dx,
            "residual_dy": residual_dy,
            "residual_dx": residual_dx,
            "residual_magnitude_px": residual_magnitude,
            "peak_score": peak_score,
            "ambiguity_ratio": ratio,
        }
    )

    reasons = []

    if peak_score < min_peak_score:
        reasons.append(
            f"peak_score={peak_score:.4f} below min_peak_score={min_peak_score}"
        )

    if ratio is None or ratio < min_ambiguity_ratio:
        reasons.append(
            f"ambiguity_ratio={ratio} below min_ambiguity_ratio={min_ambiguity_ratio}"
        )

    if residual_magnitude > max_residual_px:
        reasons.append(
            f"residual_magnitude_px={residual_magnitude:.2f} exceeds "
            f"max_residual_px={max_residual_px}"
        )

    base["accepted"] = len(reasons) == 0
    base["reject_reasons"] = reasons

    return base


def points_from_keypoint_fallback(keypoint_fallback_result):
    """
    Stage 1 for pairs whose Stage 0 anchor came from
    `keypoint_fallback.run_keypoint_fallback` rather than dense
    correlation: the dense oriented-gradient local search shares the
    same descriptor Stage 0's dense search uses, so if that descriptor
    doesn't carry signal for this pair (confirmed on Pair002/OHRC --
    Stage 1's dense path gets 0/72 accepted there), running it again at
    a finer grid won't help either. Reuse the fallback matcher's own
    RANSAC-inlier correspondences directly instead of re-deriving points
    a broken descriptor can't find.

    These points are already independently RANSAC-verified against a
    global affine (that's what made them inliers), so they're
    `accepted=True` unconditionally here -- there's no separate local
    ambiguity/peak-score signal to gate on for this path.
    """

    source_points = keypoint_fallback_result["inlier_source_points"]
    reference_points = keypoint_fallback_result["inlier_reference_points"]

    points = []

    for (sx, sy), (rx, ry) in zip(source_points, reference_points):
        points.append(
            {
                "reference_y": ry,
                "reference_x": rx,
                "source_y": sy,
                "source_x": sx,
                "residual_dy": None,
                "residual_dx": None,
                "residual_magnitude_px": None,
                "peak_score": None,
                "ambiguity_ratio": None,
                "valid_fraction": None,
                "accepted": True,
                "reject_reasons": [],
                "source": f"keypoint_fallback_{keypoint_fallback_result['method']}",
            }
        )

    summary = {
        "grid_points": len(points),
        "accepted_count": len(points),
        "rejected_count": 0,
        "residual_mean_px": None,
        "residual_median_px": None,
        "residual_max_px": None,
        "source": "keypoint_fallback",
    }

    return {"points": points, "summary": summary, "config": None}


def run_stage1(
    source_stack,
    source_mask,
    reference_stack,
    reference_mask,
    global_dy,
    global_dx,
    config,
):
    """
    Distribute control points uniformly across the valid overlap and
    independently refine + gate each one against the Stage 0 global
    anchor. `source_stack`/`reference_stack` are the same (K, H, W)
    oriented-gradient stacks (and matching eroded masks) Stage 0 already
    built -- Stage 1 reuses them at a finer per-point grid rather than
    rebuilding the descriptor.
    """

    bbox = mask_bbox(reference_mask)

    if bbox is None:
        return {
            "points": [],
            "summary": {
                "grid_points": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "residual_mean_px": None,
                "residual_median_px": None,
                "residual_max_px": None,
            },
            "config": config,
        }

    points = []

    for center_y, center_x in grid_points(bbox, config["grid_spacing_px"]):
        points.append(
            refine_point(
                source_stack,
                source_mask,
                reference_stack,
                reference_mask,
                center_y,
                center_x,
                global_dy,
                global_dx,
                window_px=config["window_px"],
                min_valid_fraction=config["min_valid_fraction"],
                min_peak_score=config["min_peak_score"],
                min_ambiguity_ratio=config["min_ambiguity_ratio"],
                max_residual_px=config["max_residual_px"],
                overlap_ratio=config["overlap_ratio_fft"],
            )
        )

    accepted = [p for p in points if p["accepted"]]
    rejected = [p for p in points if not p["accepted"]]

    residuals = [
        p["residual_magnitude_px"]
        for p in accepted
        if p["residual_magnitude_px"] is not None
    ]

    summary = {
        "grid_points": len(points),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "residual_mean_px": float(np.mean(residuals)) if residuals else None,
        "residual_median_px": float(np.median(residuals)) if residuals else None,
        "residual_max_px": float(np.max(residuals)) if residuals else None,
    }

    return {"points": points, "summary": summary, "config": config}
