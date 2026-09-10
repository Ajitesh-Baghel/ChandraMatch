import numpy as np

from src.coarse.masked_correlation import (
    channel_summed_surface,
    extract_topk_peaks,
    surface_index_to_shift,
)


def shift_array(array, dy, dx, fill_value=0):
    """
    Return `array` shifted so that shifted(y, x) = array(y - dy, x - dx),
    the same convention used throughout this package (matches
    cv2.warpAffine's default translation semantics). Out-of-bounds pixels
    are filled with `fill_value`.
    """

    height, width = array.shape[:2]

    shifted = np.full_like(array, fill_value)

    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)

    if src_y0 >= src_y1 or src_x0 >= src_x1:
        return shifted

    dst_y0 = src_y0 + dy
    dst_y1 = src_y1 + dy
    dst_x0 = src_x0 + dx
    dst_x1 = src_x1 + dx

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]

    return shifted


def valid_overlap_fraction(source_mask, reference_mask, dy, dx):
    """
    Fraction of the smaller mask's area that lands on valid reference
    pixels after shifting the source mask by (dy, dx).
    """

    warped_source_mask = shift_array(source_mask.astype(np.uint8), dy, dx) > 0

    overlap = np.count_nonzero(warped_source_mask & reference_mask)

    denom = min(
        int(np.count_nonzero(source_mask)),
        int(np.count_nonzero(reference_mask)),
    )

    if denom == 0:
        return 0.0

    return float(overlap) / float(denom)


def direct_ncc(a, b, mask):
    """
    Direct-domain (non-FFT) normalized cross-correlation over mask-valid
    pixels. Used to independently re-check a hypothesis without reusing
    the FFT search surface's own score.
    """

    valid = mask & np.isfinite(a) & np.isfinite(b)

    if np.count_nonzero(valid) < 100:
        return None

    x = a[valid].astype(np.float64)
    y = b[valid].astype(np.float64)

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))

    if denom <= 1e-12:
        return None

    return float(np.sum(x * y) / denom)


def structural_ncc_before_after(
    source_representation,
    source_mask,
    reference_representation,
    reference_mask,
    dy,
    dx,
):
    """
    Direct-domain structural NCC (summed oriented-gradient magnitude
    collapsed to a single channel, e.g. via `.sum(axis=0)` before calling
    this) evaluated before any shift and after warping the source by the
    winning hypothesis. Independent of the FFT surface score by
    construction.
    """

    common_before = source_mask & reference_mask

    ncc_before = direct_ncc(
        source_representation,
        reference_representation,
        common_before,
    )

    warped_source = shift_array(source_representation, dy, dx, fill_value=0.0)
    warped_source_mask = shift_array(source_mask.astype(np.uint8), dy, dx) > 0

    common_after = warped_source_mask & reference_mask

    ncc_after = direct_ncc(
        warped_source,
        reference_representation,
        common_after,
    )

    return ncc_before, ncc_after


def mask_bbox(mask):
    """
    Returns (y0, y1, x0, x1) of the valid-pixel bounding box (y1/x1
    exclusive), or None if `mask` has no valid pixels.
    """

    valid_rows, valid_cols = np.where(mask)

    if valid_rows.size == 0:
        return None

    return (
        int(valid_rows.min()),
        int(valid_rows.max()) + 1,
        int(valid_cols.min()),
        int(valid_cols.max()) + 1,
    )


def mask_bbox_fill_ratio(mask):
    """
    Valid-pixel count divided by the area of its own bounding box.

    A compact/blob-shaped footprint fills most of its bounding box
    (ratio close to 1). A narrow, non-axis-aligned swath crossing a much
    larger canvas fills only a small fraction of its own bounding box
    even though the swath itself may be "full" of valid pixels -- e.g.
    Pair004's diagonal LRO NAC swath has a bounding box that spans nearly
    the entire canonical canvas, giving a fill ratio of ~0.24, versus
    Pair006/007's wide-but-roughly-rectangular overlaps at ~0.52/~0.40.
    This is the discriminator for the aperture-problem failure mode:
    whole-frame dense correlation is only well-conditioned when the
    valid support is reasonably compact, because a narrow elongated
    support lets many translations along its long axis stay
    self-overlapping, and homogeneous cratered terrain doesn't have
    enough large-scale distinctiveness to break that tie.

    Returns None if `mask` has no valid pixels.
    """

    bbox = mask_bbox(mask)

    if bbox is None:
        return None

    y0, y1, x0, x1 = bbox

    bbox_area = (y1 - y0) * (x1 - x0)

    if bbox_area == 0:
        return None

    return float(np.count_nonzero(mask)) / float(bbox_area)


def local_cell_agreement(
    source_stack,
    source_mask,
    reference_stack,
    reference_mask,
    global_dy,
    global_dx,
    grid_rows=4,
    grid_cols=4,
    window_px=256,
    min_valid_fraction=0.3,
    tolerance_px=8.0,
    overlap_ratio=0.15,
):
    """
    Independent per-cell check: does the same real terrain support the
    global hypothesis everywhere, or only in the region a sparse matcher
    happened to sample?

    An interior N x N grid is laid out over the *valid-pixel bounding box*
    of the reference mask, not the full canonical canvas -- these
    canonical rasters are frequently a narrow sensor swath occupying a
    small fraction of a much larger co-registration canvas (e.g. Pair004:
    ~24% valid, Pair006: ~11% valid), and a grid over the full canvas
    would waste most cells on empty area, leaving too few samples for
    "does the terrain agree everywhere" to mean anything. For each cell
    with enough valid pixels, a small window centered on that cell is
    cropped from both source (already conceptually shifted by the global
    hypothesis, so a *local* correlation search only has to find the
    residual) and reference, and a local displacement is independently
    re-estimated via the same masked FFT correlation machinery used for
    the global search, restricted to a small neighborhood around zero
    residual. The cell "agrees" if its local displacement is within
    `tolerance_px` of (0, 0) residual, i.e. within tolerance of the
    global hypothesis.

    Returns a dict with `cells` (per-cell detail) and
    `agreement_fraction` (float, or None if no cell had enough support).
    """

    height, width = reference_mask.shape

    bbox = mask_bbox(reference_mask)

    if bbox is None:
        return {"cells": [], "agreement_fraction": None}

    bbox_y0, bbox_y1, bbox_x0, bbox_x1 = bbox

    half = window_px // 2

    cells = []

    for row in range(grid_rows):
        cell_y0 = bbox_y0 + int(round(row * (bbox_y1 - bbox_y0) / grid_rows))
        cell_y1 = bbox_y0 + int(round((row + 1) * (bbox_y1 - bbox_y0) / grid_rows))
        center_y = (cell_y0 + cell_y1) // 2

        for col in range(grid_cols):
            cell_x0 = bbox_x0 + int(round(col * (bbox_x1 - bbox_x0) / grid_cols))
            cell_x1 = bbox_x0 + int(
                round((col + 1) * (bbox_x1 - bbox_x0) / grid_cols)
            )
            center_x = (cell_x0 + cell_x1) // 2

            ry0 = max(0, center_y - half)
            ry1 = min(height, center_y + half)
            rx0 = max(0, center_x - half)
            rx1 = min(width, center_x + half)

            ref_window_mask = reference_mask[ry0:ry1, rx0:rx1]

            if ref_window_mask.size == 0:
                continue

            valid_fraction = float(np.count_nonzero(ref_window_mask)) / float(
                ref_window_mask.size
            )

            if valid_fraction < min_valid_fraction:
                continue

            # Source window pre-shifted by the global hypothesis so the
            # local search only needs to resolve the residual around 0.
            sy0 = ry0 - global_dy
            sy1 = ry1 - global_dy
            sx0 = rx0 - global_dx
            sx1 = rx1 - global_dx

            if sy0 < 0 or sx0 < 0 or sy1 > height or sx1 > width:
                continue

            src_window_mask = source_mask[sy0:sy1, sx0:sx1]

            if src_window_mask.shape != ref_window_mask.shape:
                continue

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
            except Exception:
                continue

            peaks = extract_topk_peaks(surface, top_k=1, nms_radius=5)

            if not peaks:
                continue

            residual_dy, residual_dx = surface_index_to_shift(
                peaks[0]["index"], ref_shape
            )

            residual_magnitude = float(np.hypot(residual_dy, residual_dx))

            agrees = residual_magnitude <= tolerance_px

            cells.append(
                {
                    "row": row,
                    "col": col,
                    "valid_fraction": valid_fraction,
                    "residual_dy": residual_dy,
                    "residual_dx": residual_dx,
                    "residual_magnitude_px": residual_magnitude,
                    "score": peaks[0]["score"],
                    "agrees": agrees,
                }
            )

    if not cells:
        return {"cells": cells, "agreement_fraction": None}

    agreeing = sum(1 for cell in cells if cell["agrees"])

    return {
        "cells": cells,
        "agreement_fraction": float(agreeing) / float(len(cells)),
    }
