import cv2
import numpy as np

"""
Coarse rotation/scale hint via Fourier-Mellin log-polar phase
correlation.

NOTE: this is the "optional" piece of Stage 0. It has only been checked
against a synthetic rotation+scale pair, not yet against real
Pair004/006/007 data (all three already show <0.2 degree rotation and
~1.0 scale in v1's benchmarks, so it is gated off by default in
configs/stage0_coarse_search.json). Validate independently before
flipping `enable_fourier_mellin` on for a real pair.
"""


def _log_polar_magnitude_spectrum(image):
    height, width = image.shape

    window = np.outer(
        np.hanning(height),
        np.hanning(width),
    ).astype(np.float32)

    spectrum = np.fft.fftshift(np.fft.fft2(image.astype(np.float32) * window))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)

    center = (width / 2.0, height / 2.0)
    max_radius = min(height, width) / 2.0

    log_polar = cv2.warpPolar(
        magnitude,
        (width, height),
        center,
        max_radius,
        cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR,
    )

    return log_polar, max_radius


def estimate_rotation_scale(image_a, image_b):
    """
    Returns (rotation_deg, scale) such that rotating/scaling `image_b` by
    this amount roughly aligns its dominant structure orientation/size
    with `image_a`. Coarse only -- meant as a pre-step before the
    translation search, not a final estimate.
    """

    log_polar_a, max_radius = _log_polar_magnitude_spectrum(image_a)
    log_polar_b, _ = _log_polar_magnitude_spectrum(image_b)

    height, width = image_a.shape

    (shift_x, shift_y), _response = cv2.phaseCorrelate(
        log_polar_a.astype(np.float32),
        log_polar_b.astype(np.float32),
    )

    rotation_deg = shift_y * 360.0 / height
    scale = float(np.exp(shift_x * np.log(max_radius) / width))

    return float(rotation_deg), scale
