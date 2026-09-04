import cv2
import numpy as np


def filter_matches_ransac(
    source_points,
    reference_points,
    reprojection_threshold=3.0
):
    """
    Estimate an affine transformation using RANSAC.

    Returns:
        transformation
        inlier_mask
        inlier_source_points
        inlier_reference_points
    """

    if len(source_points) < 3:
        raise RuntimeError(
            "Not enough matches for affine estimation."
        )

    transformation, inlier_mask = cv2.estimateAffine2D(
        source_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
        maxIters=5000,
        confidence=0.999,
        refineIters=10
    )

    if transformation is None:
        raise RuntimeError(
            "RANSAC could not estimate transformation."
        )

    inlier_mask = inlier_mask.ravel().astype(bool)

    inlier_source = source_points[inlier_mask]
    inlier_reference = reference_points[inlier_mask]

    return (
        transformation,
        inlier_mask,
        inlier_source,
        inlier_reference
    )