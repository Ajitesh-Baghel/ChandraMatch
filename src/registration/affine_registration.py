import cv2
import numpy as np


def register_affine(
    source,
    transformation,
    output_shape
):
    """
    Warp source image into reference coordinate system.
    """

    height, width = output_shape

    registered = cv2.warpAffine(
        source,
        transformation,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return registered


def create_overlay(
    registered,
    reference
):
    """
    RGB overlay:
        Red   = reference
        Green = registered source

    Perfectly aligned structures tend toward yellow/neutral overlap.
    """

    overlay = np.zeros(
        (
            reference.shape[0],
            reference.shape[1],
            3
        ),
        dtype=np.uint8
    )

    overlay[:, :, 2] = reference
    overlay[:, :, 1] = registered

    return overlay


def create_difference(
    registered,
    reference
):
    return cv2.absdiff(
        registered,
        reference
    )