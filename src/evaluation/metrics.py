import cv2
import numpy as np


def identity_errors(
    source_points,
    reference_points
):
    """
    Raw displacement between corresponding points before
    applying the estimated registration transformation.

    This measures initial misalignment, NOT final registration error.
    """

    differences = reference_points - source_points

    return np.linalg.norm(
        differences,
        axis=1
    )


def reprojection_errors(
    source_points,
    reference_points,
    transformation
):
    """
    Apply the estimated affine transformation to source points
    and measure their residual distance from corresponding
    reference points.
    """

    if len(source_points) == 0:
        return np.array([])

    transformed_source = cv2.transform(
        source_points.reshape(-1, 1, 2),
        transformation
    ).reshape(-1, 2)

    differences = (
        reference_points - transformed_source
    )

    return np.linalg.norm(
        differences,
        axis=1
    )


def calculate_metrics(
    candidate_count,
    inlier_source,
    inlier_reference,
    transformation
):
    inlier_count = len(inlier_source)

    if candidate_count > 0:
        inlier_ratio = (
            inlier_count / candidate_count
        )
    else:
        inlier_ratio = 0.0

    # Before registration
    pre_errors = identity_errors(
        inlier_source,
        inlier_reference
    )

    # After estimated registration
    post_errors = reprojection_errors(
        inlier_source,
        inlier_reference,
        transformation
    )

    def summarize(errors):

        if len(errors) == 0:
            return {
                "rmse": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "max": float("nan")
            }

        return {
            "rmse": float(
                np.sqrt(np.mean(errors ** 2))
            ),

            "mean": float(
                np.mean(errors)
            ),

            "median": float(
                np.median(errors)
            ),

            "max": float(
                np.max(errors)
            )
        }

    pre = summarize(pre_errors)
    post = summarize(post_errors)

    return {
        "candidate_matches": candidate_count,

        "inlier_matches": inlier_count,

        "inlier_ratio": float(inlier_ratio),

        "pre_registration_rmse_px":
            pre["rmse"],

        "pre_registration_mean_error_px":
            pre["mean"],

        "pre_registration_median_error_px":
            pre["median"],

        "ransac_reprojection_rmse_px":
            post["rmse"],

        "ransac_reprojection_mean_error_px":
            post["mean"],

        "ransac_reprojection_median_error_px":
            post["median"],

        "ransac_reprojection_max_error_px":
            post["max"]
    }