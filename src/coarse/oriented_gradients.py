import cv2
import numpy as np
from scipy import ndimage


def robust_unit_stretch(image, mask, low_percentile=2.0, high_percentile=98.0):
    """
    Percentile-stretch a raster to float32 [0, 1] using only mask-valid
    pixels for the stretch statistics. Pixels outside the mask are set to 0.
    """

    output = np.zeros(image.shape, dtype=np.float32)

    values = image[mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return output

    low, high = np.percentile(values, [low_percentile, high_percentile])

    if high <= low:
        high = low + 1.0

    stretched = (image.astype(np.float32) - float(low)) / float(high - low)
    stretched = np.clip(stretched, 0.0, 1.0)

    output[mask] = stretched[mask]

    return output


def erode_mask(mask, erosion_px):
    if erosion_px <= 0:
        return mask.copy()

    return ndimage.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(erosion_px),
        border_value=0,
    )


def build_oriented_gradient_stack(
    image,
    mask,
    num_orientations=8,
    gaussian_sigma=1.0,
    mask_erosion_px=5,
):
    """
    CFOG/HOPC-style structural descriptor.

    `image` may be a raw single-band raster (any dtype/range); it is
    percentile-stretched internally over `mask`. Returns a (K, H, W)
    float32 stack where channel k is
        |cos(theta_k) * gx + sin(theta_k) * gy|
    Gaussian-smoothed, for theta_k = k * pi / K, k = 0..K-1, plus the
    eroded mask actually used for the descriptor (see below) -- callers
    must use this returned mask, not the input `mask`, for every
    downstream step (correlation, verification), so it stays consistent
    with what the descriptor values actually represent.

    Orientations span [0, pi) rather than [0, 2*pi) because the absolute
    value makes each channel invariant to a 180 degree flip of the
    gradient direction -- this is what lets the descriptor survive a
    sun-angle/illumination reversal instead of just a sensor difference.

    `mask_erosion_px`: `robust_unit_stretch` zero-fills pixels outside
    `mask` before the Sobel pass, which manufactures an artificial edge
    exactly along the mask boundary. On a real footprint mask (e.g. a
    narrow diagonal sensor swath crossing a much larger canonical
    canvas), that boundary can be long relative to the interior area, and
    the Sobel+Gaussian-blur kernel leaks this artificial edge a few
    pixels into the interior of otherwise-valid data -- which then
    correlates against the *mask shape itself* rather than real surface
    texture, and can out-score genuine structural correlation entirely
    (observed on Pair004's diagonal NAC swath). Eroding the mask by a
    margin at least as large as the Sobel radius plus the blur radius
    removes that contaminated band from both the descriptor values and
    the mask handed to the correlation surface. Matches the precedent in
    scripts/benchmark_pair007_global.py's MASK_EROSION_PIXELS.
    """

    stretched = robust_unit_stretch(image, mask)

    gx = cv2.Sobel(stretched, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(stretched, cv2.CV_32F, 0, 1, ksize=3)

    correlation_mask = erode_mask(mask, mask_erosion_px)

    channels = np.empty((num_orientations,) + image.shape, dtype=np.float32)

    for k in range(num_orientations):
        theta = k * np.pi / num_orientations

        projected = np.abs(np.cos(theta) * gx + np.sin(theta) * gy)

        smoothed = cv2.GaussianBlur(
            projected,
            (0, 0),
            sigmaX=gaussian_sigma,
            sigmaY=gaussian_sigma,
        )

        smoothed[~correlation_mask] = 0.0

        channels[k] = smoothed

    return channels, correlation_mask
