import cv2
import numpy as np


def clahe_normalization(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Local contrast normalization.
    Useful when different regions are illuminated differently.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(image)


def local_contrast_normalization(
    image,
    sigma=25.0
):
    """
    Remove slowly varying illumination while preserving
    local lunar surface structure.
    """

    image_float = image.astype(
        np.float32
    )

    background = cv2.GaussianBlur(
        image_float,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    local = (
        image_float
        - background
    )

    valid = image > 0

    output = np.zeros_like(
        image,
        dtype=np.uint8
    )

    values = local[valid]

    if values.size == 0:
        return output

    low, high = np.percentile(
        values,
        [1, 99]
    )

    if high <= low:
        return output

    local = np.clip(
        local,
        low,
        high
    )

    local = (
        (local - low)
        / (high - low)
        * 255.0
    )

    output[valid] = (
        local[valid]
        .astype(np.uint8)
    )

    return output


def gradient_representation(
    image
):
    """
    Structural representation that depends less on absolute
    brightness and more on surface boundaries.
    """

    image_float = image.astype(
        np.float32
    )

    gx = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    valid = image > 0

    output = np.zeros_like(
        image,
        dtype=np.uint8
    )

    values = magnitude[
        valid
    ]

    if values.size == 0:
        return output

    high = np.percentile(
        values,
        99
    )

    if high <= 0:
        return output

    magnitude = np.clip(
        magnitude,
        0,
        high
    )

    magnitude = (
        magnitude
        / high
        * 255.0
    )

    output[valid] = (
        magnitude[valid]
        .astype(np.uint8)
    )

    return output


def get_illumination_variant(
    image,
    variant
):
    """
    Standard entry point used by benchmarks and later by
    ChandraMatch's preprocessing stage.
    """

    if variant == "baseline":
        return image.copy()

    if variant == "clahe":
        return clahe_normalization(
            image
        )

    if variant == "local_contrast":
        return local_contrast_normalization(
            image
        )

    if variant == "gradient":
        return gradient_representation(
            image
        )

    raise ValueError(
        f"Unknown illumination preprocessing: {variant}"
    )