import numpy as np


def calculate_spatial_coverage(
    points,
    image_shape,
    grid_rows=8,
    grid_cols=8
):
    """
    Percentage of grid cells containing at least one correspondence.
    """

    height, width = image_shape

    occupied = np.zeros(
        (grid_rows, grid_cols),
        dtype=bool
    )

    for x, y in points:

        col = min(
            int(x / width * grid_cols),
            grid_cols - 1
        )

        row = min(
            int(y / height * grid_rows),
            grid_rows - 1
        )

        if row >= 0 and col >= 0:
            occupied[row, col] = True

    occupied_cells = int(occupied.sum())
    total_cells = grid_rows * grid_cols

    coverage = occupied_cells / total_cells

    return {
        "coverage_ratio": float(coverage),
        "occupied_cells": occupied_cells,
        "total_cells": total_cells,
        "grid": occupied
    }