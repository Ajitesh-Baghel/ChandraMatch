import numpy as np
import cv2
import rasterio


def read_tmc2(path):
    """
    Read a TMC-2 GeoTIFF and return:
    image: uint16 array
    profile: rasterio metadata
    """

    with rasterio.open(path) as src:
        image = src.read(1)
        profile = src.profile.copy()

    return image, profile


def normalize_tmc2(image):
    """
    Convert TMC-2 uint16 imagery to uint8 while ignoring
    zero/background pixels.
    """

    valid = image > 0

    output = np.zeros_like(image, dtype=np.uint8)

    values = image[valid]

    if values.size == 0:
        return output

    low, high = np.percentile(values, [1, 99])

    if high <= low:
        return output

    normalized = np.clip(
        image.astype(np.float32),
        low,
        high
    )

    normalized = (
        (normalized - low)
        / (high - low)
        * 255.0
    )

    output[valid] = normalized[valid].astype(np.uint8)

    return output


def preprocess_tmc2(image, use_clahe=False):
    """
    Complete preprocessing for TMC-2 imagery.
    """

    image = normalize_tmc2(image)

    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        image = clahe.apply(image)

    return image