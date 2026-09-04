import cv2
import numpy as np


def create_affine_transform(
    image_shape,
    rotation_deg=0.0,
    scale=1.0,
    translation_x=0.0,
    translation_y=0.0
):
    """
    Create a known affine transformation.

    Returns:
        reference_to_source
        source_to_reference_ground_truth
    """

    height, width = image_shape

    center = (
        width / 2.0,
        height / 2.0
    )

    reference_to_source = (
        cv2.getRotationMatrix2D(
            center,
            rotation_deg,
            scale
        )
    )

    reference_to_source[
        0, 2
    ] += translation_x

    reference_to_source[
        1, 2
    ] += translation_y


    source_to_reference = (
        cv2.invertAffineTransform(
            reference_to_source
        )
    )

    return (
        reference_to_source,
        source_to_reference
    )


def apply_affine(
    image,
    transformation
):
    height, width = image.shape[:2]

    return cv2.warpAffine(
        image,
        transformation,
        (
            width,
            height
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def transform_points(
    points,
    transformation
):
    """
    Apply affine transform to Nx2 points.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    result = cv2.transform(
        points.reshape(-1, 1, 2),
        transformation
    )

    return result.reshape(-1, 2)