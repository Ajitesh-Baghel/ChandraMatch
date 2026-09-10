import numpy as np
from skimage.registration._masked_phase_cross_correlation import (
    cross_correlate_masked,
)


def channel_summed_surface(stack_a, mask_a, stack_b, mask_b, overlap_ratio=0.15):
    """
    Masked normalized cross-correlation (Padfield) summed across the K
    descriptor channels, evaluated at every possible integer displacement
    in one batched FFT pass.

    `stack_a`/`stack_b` are (K, H, W) oriented-gradient stacks (see
    oriented_gradients.build_oriented_gradient_stack); `mask_a`/`mask_b`
    are the shared (H, W) validity masks broadcast to every channel.
    Summing per-channel masked-NCC surfaces is equivalent to correlating
    the concatenated per-pixel descriptor vector, which is the correct
    formulation for dense multi-channel structural matching.

    Returns:
        surface: (2H-1, 2W-1) float64 array of summed correlation scores.
        ref_shape: (H, W) of `stack_b`, needed by surface_index_to_shift.
    """

    k, h, w = stack_a.shape

    mask_a_full = np.broadcast_to(mask_a, stack_a.shape)
    mask_b_full = np.broadcast_to(mask_b, stack_b.shape)

    per_channel = cross_correlate_masked(
        stack_a,
        stack_b,
        mask_a_full,
        mask_b_full,
        mode="full",
        axes=(-2, -1),
        overlap_ratio=overlap_ratio,
    )

    surface = np.sum(np.real(per_channel), axis=0)

    return surface, stack_b.shape[1:]


def surface_index_to_shift(index, ref_shape):
    """
    Convert a (row, col) index into the 'full'-mode correlation surface
    into a pixel displacement (dy, dx).

    Convention (verified empirically against a synthetic known shift):
    if warping `source` by translation (tx, ty) via
        warped(y, x) = source(y - ty, x - tx)
    aligns it onto `reference`, and the surface was built as
        channel_summed_surface(source_stack, source_mask,
                                reference_stack, reference_mask),
    then (ty, tx) = surface_index_to_shift(argmax_index, reference.shape).
    """

    ref_h, ref_w = ref_shape

    dy = (ref_h - 1) - int(index[0])
    dx = (ref_w - 1) - int(index[1])

    return dy, dx


def extract_topk_peaks(surface, top_k=5, nms_radius=15):
    """
    Greedy non-max-suppression peak picking on a 2D correlation surface.

    Returns a list of dicts (highest score first), each
        {"index": (row, col), "score": float}
    with fewer than `top_k` entries if the surface is smaller than the
    suppression neighborhood allows.
    """

    working = surface.astype(np.float64).copy()

    height, width = working.shape

    peaks = []

    for _ in range(top_k):
        flat_index = int(np.argmax(working))

        row, col = np.unravel_index(flat_index, working.shape)

        score = float(working[row, col])

        if not np.isfinite(score) or score <= -np.inf:
            break

        peaks.append({"index": (int(row), int(col)), "score": score})

        r0 = max(0, row - nms_radius)
        r1 = min(height, row + nms_radius + 1)
        c0 = max(0, col - nms_radius)
        c1 = min(width, col + nms_radius + 1)

        working[r0:r1, c0:c1] = -np.inf

    return peaks


def subpixel_shift_correction(surface, index):
    """
    Sub-pixel correction to add to the integer (dy, dx) from
    `surface_index_to_shift(index, ...)`, via 1D parabolic interpolation
    of the 3 surface samples straddling the integer peak along each axis
    independently. Returns (0.0, 0.0) if the peak sits on the surface
    edge (no interpolation possible) or the local samples are degenerate
    (flat/inverted, i.e. not really a peak).

    Standard formula: for samples (left, center, right) around the
    integer peak, the fractional offset (in samples, positive toward
    `right`) is 0.5*(left-right)/(left-2*center+right). Since increasing
    the surface row/col index *decreases* dy/dx (see
    `surface_index_to_shift`), the sign is flipped when converting the
    row/col fractional offsets into a (dy, dx) correction.
    """

    row, col = index
    height, width = surface.shape

    def parabolic_offset(left, center, right):
        denom = left - 2.0 * center + right

        if denom == 0:
            return 0.0

        offset = 0.5 * (left - right) / denom

        if not np.isfinite(offset):
            return 0.0

        return float(np.clip(offset, -0.5, 0.5))

    row_offset = 0.0
    col_offset = 0.0

    if 0 < row < height - 1:
        row_offset = parabolic_offset(
            surface[row - 1, col], surface[row, col], surface[row + 1, col]
        )

    if 0 < col < width - 1:
        col_offset = parabolic_offset(
            surface[row, col - 1], surface[row, col], surface[row, col + 1]
        )

    return -row_offset, -col_offset


def ambiguity_ratio(peaks):
    """
    peak1_score / peak2_score, using a small positive floor on the
    denominator so a near-zero or negative second peak doesn't blow the
    ratio up to a meaningless value. Returns None if fewer than 2 peaks.
    """

    if len(peaks) < 2:
        return None

    peak1 = peaks[0]["score"]
    peak2 = peaks[1]["score"]

    denom = peak2 if peak2 > 1e-6 else 1e-6

    return float(peak1 / denom)
