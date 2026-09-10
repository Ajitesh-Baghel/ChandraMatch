"""
Honest `sub_pixel_achieved` determination.

Found necessary by direct testing: on Pair003, `sub_pixel_achieved: True`
with an exact-zero `rmse_px` was an artifact, not genuine sub-pixel
precision -- LightGlue's fixed LIGHTGLUE_MAX_DIM downsample cap (see
src/coarse/keypoint_fallback.py) quantizes matched-point coordinates to
roughly (canvas_max_dim / LIGHTGLUE_MAX_DIM) native pixels on an oversized
canonical canvas (Pair003's was ~150M px, 22041px wide -> ~10.8px effective
resolution), and re-fitting RANSAC a second time over that already-coarse,
already-anchor-defining point set can reproduce a suspiciously tight residual
that has nothing to do with true registration accuracy.

This module is a metric-honesty fix ONLY -- it does not change any matcher's
behavior (LightGlue's downsampling stays exactly as it was). It answers one
question: is whatever produced `residuals` actually an INDEPENDENT
local-refinement measurement taken at native resolution, or is it a re-fit
of the same coarse anchor points (dense_correlation's Stage 1 is the only
case where it's the former)?
"""

from src.coarse.keypoint_fallback import LIGHTGLUE_MAX_DIM

SUB_PIXEL_THRESHOLD_PX = 1.0


def effective_anchor_resolution_px(anchor_source, canonical_shape):
    """
    Native pixels per unit actually resolved by whichever method produced
    the anchor. 1.0 means "measured at native resolution" (dense
    correlation's Stage 1 runs directly on native-resolution descriptor
    stacks with no resize anywhere in that path; SIFT's fallback likewise
    never resizes, per src/matching/sift_matcher.py). Only LightGlue's
    fallback path can exceed native resolution, and only when the canonical
    canvas's longer side exceeds LIGHTGLUE_MAX_DIM.
    """

    if anchor_source == "dense_correlation":
        return 1.0

    if "lightglue" in anchor_source:
        longest_side = max(canonical_shape)
        return max(1.0, float(longest_side) / float(LIGHTGLUE_MAX_DIM))

    # sift_intensity or any other non-resizing fallback method.
    return 1.0


def compute_rmse_px(residuals):
    if residuals is None or len(residuals) == 0:
        return None

    import numpy as np

    return float(np.sqrt(np.mean(np.asarray(residuals, dtype=np.float64) ** 2)))


def compute_sub_pixel_status(anchor_source, fit, residuals, canonical_shape):
    """
    Returns (sub_pixel_achieved: bool, sub_pixel_reason: str|None,
    rmse_px: float|None, effective_anchor_resolution_px: float).

    `sub_pixel_achieved` is True ONLY when:
      1. a robust fit actually exists (Stage 2 ran), AND
      2. there are residuals to measure (enough accepted points), AND
      3. the anchor came from dense_correlation -- the ONLY path where
         Stage 1 is a genuine independent local-refinement measurement
         (masked-FFT correlation + parabolic sub-pixel correction) rather
         than a reuse of the same points that already defined the coarse
         anchor. A keypoint-fallback anchor's "Stage 1" is that reuse by
         design (see src/coarse/local_control_points.py::
         points_from_keypoint_fallback) -- refitting RANSAC on those same
         points a second time in Stage 2 is not an independent finer-scale
         measurement, so it can never support a sub-pixel claim regardless
         of how tight the resulting residual looks, AND
      4. the RMSE of those residuals is below one native pixel.
    """

    rmse_px = compute_rmse_px(residuals)
    resolution_px = effective_anchor_resolution_px(anchor_source, canonical_shape)

    if fit is None:
        return False, "stage2_did_not_run", rmse_px, resolution_px

    if residuals is None or len(residuals) == 0:
        return False, "insufficient_accepted_points", rmse_px, resolution_px

    if anchor_source != "dense_correlation":
        if resolution_px > 1.0:
            return False, "anchor_downsample_exceeds_native_resolution", rmse_px, resolution_px

        return False, "stage2_is_coarse_anchor_reuse", rmse_px, resolution_px

    if rmse_px is not None and rmse_px < SUB_PIXEL_THRESHOLD_PX:
        return True, None, rmse_px, resolution_px

    return False, "residual_rmse_exceeds_one_native_pixel", rmse_px, resolution_px
