import cv2
import numpy as np


def transform_points(
    points,
    transformation
):
    """
    Apply affine transformation to Nx2 points.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    if len(points) == 0:
        return np.empty(
            (0, 2),
            dtype=np.float32
        )

    transformed = cv2.transform(
        points.reshape(-1, 1, 2),
        transformation
    )

    return transformed.reshape(-1, 2)


def correspondence_ground_truth_errors(
    source_points,
    predicted_reference_points,
    ground_truth_transform
):
    """
    Ground-truth location for every source point is known
    because we created the synthetic affine transformation.

    Compare matcher prediction against exact GT location.
    """

    gt_reference_points = transform_points(
        source_points,
        ground_truth_transform
    )

    errors = np.linalg.norm(
        predicted_reference_points
        - gt_reference_points,
        axis=1
    )

    return errors


def calculate_error_summary(errors):

    if len(errors) == 0:

        return {
            "rmse": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan")
        }

    return {

        "rmse": float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
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


def transformation_ground_truth_error(
    estimated_transform,
    ground_truth_transform,
    source_mask,
    reference_mask,
    grid_step=64
):
    """
    Evaluate estimated SOURCE → REFERENCE affine transform
    against known ground truth using a uniformly distributed
    grid of points.

    This gives independent registration accuracy rather than
    RANSAC self-consistency.
    """

    height, width = source_mask.shape

    points = []

    for y in range(
        grid_step // 2,
        height,
        grid_step
    ):

        for x in range(
            grid_step // 2,
            width,
            grid_step
        ):

            if source_mask[y, x]:
                points.append(
                    [x, y]
                )

    if len(points) == 0:

        return {
            "rmse": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "evaluated_points": 0
        }

    points = np.array(
        points,
        dtype=np.float32
    )

    gt_points = transform_points(
        points,
        ground_truth_transform
    )

    estimated_points = transform_points(
        points,
        estimated_transform
    )

    # Only evaluate locations whose GT destination actually
    # lies inside the valid reference image.

    gt_x = np.round(
        gt_points[:, 0]
    ).astype(int)

    gt_y = np.round(
        gt_points[:, 1]
    ).astype(int)

    inside = (
        (gt_x >= 0)
        &
        (gt_x < width)
        &
        (gt_y >= 0)
        &
        (gt_y < height)
    )

    points = points[inside]
    gt_points = gt_points[inside]
    estimated_points = estimated_points[inside]

    gt_x = gt_x[inside]
    gt_y = gt_y[inside]

    valid_destination = reference_mask[
        gt_y,
        gt_x
    ]

    gt_points = gt_points[
        valid_destination
    ]

    estimated_points = estimated_points[
        valid_destination
    ]

    errors = np.linalg.norm(
        estimated_points
        - gt_points,
        axis=1
    )

    summary = calculate_error_summary(
        errors
    )

    summary["evaluated_points"] = int(
        len(errors)
    )

    return summary