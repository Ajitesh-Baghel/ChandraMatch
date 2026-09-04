import cv2
import numpy as np


def draw_matches(
    source,
    reference,
    source_points,
    reference_points,
    output_path,
    max_matches=150
):

    source_rgb = cv2.cvtColor(
        source,
        cv2.COLOR_GRAY2BGR
    )

    reference_rgb = cv2.cvtColor(
        reference,
        cv2.COLOR_GRAY2BGR
    )

    height = max(
        source_rgb.shape[0],
        reference_rgb.shape[0]
    )

    width = (
        source_rgb.shape[1]
        + reference_rgb.shape[1]
    )

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    canvas[
        :source_rgb.shape[0],
        :source_rgb.shape[1]
    ] = source_rgb

    offset = source_rgb.shape[1]

    canvas[
        :reference_rgb.shape[0],
        offset:offset + reference_rgb.shape[1]
    ] = reference_rgb

    count = min(
        len(source_points),
        max_matches
    )

    if len(source_points) > count:

        indices = np.linspace(
            0,
            len(source_points) - 1,
            count
        ).astype(int)

    else:
        indices = np.arange(
            len(source_points)
        )

    for i in indices:

        x1, y1 = source_points[i]
        x2, y2 = reference_points[i]

        p1 = (
            int(round(x1)),
            int(round(y1))
        )

        p2 = (
            int(round(x2)) + offset,
            int(round(y2))
        )

        cv2.circle(
            canvas,
            p1,
            4,
            (0, 255, 0),
            1
        )

        cv2.circle(
            canvas,
            p2,
            4,
            (0, 255, 0),
            1
        )

        cv2.line(
            canvas,
            p1,
            p2,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(
        str(output_path),
        canvas
    )