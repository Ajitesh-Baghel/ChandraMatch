import numpy as np
import rasterio
import cv2


def read_ohrc(path):
    """
    Read OHRC raster while retaining geospatial metadata.
    """

    with rasterio.open(path) as src:

        image = src.read(1)

        profile = src.profile.copy()

    return image, profile


def normalize_ohrc(image):
    """
    Robust percentile normalization of OHRC imagery.
    """

    valid = image > 0

    output = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    values = image[valid]

    if values.size == 0:
        return output

    low, high = np.percentile(
        values,
        [1, 99]
    )

    if high <= low:
        return output

    work = np.clip(
        image.astype(np.float32),
        low,
        high
    )

    work = (
        (work - low)
        / (high - low)
        * 255.0
    )

    output[valid] = (
        work[valid]
        .astype(np.uint8)
    )

    return output


def preprocess_ohrc(
    image,
    use_clahe=False
):

    image = normalize_ohrc(
        image
    )

    if use_clahe:

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        image = clahe.apply(
            image
        )

    return image