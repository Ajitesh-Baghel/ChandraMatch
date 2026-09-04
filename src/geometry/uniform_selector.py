import cv2
import numpy as np


def transform_points(
    points,
    transformation
):
    """
    Apply 2x3 affine transformation to Nx2 points.
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


def select_uniform_correspondences(
    source_points,
    reference_points,
    confidence,
    transformation,
    image_shape,
    grid_rows=8,
    grid_cols=8,
    max_per_cell=5
):
    """
    Select geometrically strong correspondences while ensuring
    spatial distribution across the image.

    Priority:
        1. low geometric reprojection error
        2. high matcher confidence
    """

    source_points = np.asarray(
        source_points,
        dtype=np.float32
    )

    reference_points = np.asarray(
        reference_points,
        dtype=np.float32
    )

    if confidence is None:
        confidence = np.ones(
            len(source_points),
            dtype=np.float32
        )

    confidence = np.asarray(
        confidence,
        dtype=np.float32
    )


    predicted_reference = transform_points(
        source_points,
        transformation
    )


    residuals = np.linalg.norm(
        reference_points
        - predicted_reference,
        axis=1
    )


    height, width = image_shape[:2]


    selected_indices = []


    for row in range(grid_rows):

        y_min = (
            row
            * height
            / grid_rows
        )

        y_max = (
            (row + 1)
            * height
            / grid_rows
        )


        for col in range(grid_cols):

            x_min = (
                col
                * width
                / grid_cols
            )

            x_max = (
                (col + 1)
                * width
                / grid_cols
            )


            inside = np.where(

                (source_points[:, 0] >= x_min)
                &
                (source_points[:, 0] < x_max)
                &
                (source_points[:, 1] >= y_min)
                &
                (source_points[:, 1] < y_max)

            )[0]


            if len(inside) == 0:
                continue


            # Primary priority:
            # lower geometric residual
            #
            # Secondary priority:
            # higher matcher confidence

            order = np.lexsort(
                (
                    -confidence[inside],
                    residuals[inside]
                )
            )


            best = inside[
                order[
                    :max_per_cell
                ]
            ]


            selected_indices.extend(
                best.tolist()
            )


    selected_indices = np.array(
        selected_indices,
        dtype=int
    )


    return {

        "source_points":
            source_points[
                selected_indices
            ],

        "reference_points":
            reference_points[
                selected_indices
            ],

        "confidence":
            confidence[
                selected_indices
            ],

        "geometric_residual":
            residuals[
                selected_indices
            ],

        "indices":
            selected_indices
    }