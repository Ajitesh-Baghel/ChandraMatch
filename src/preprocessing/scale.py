import cv2


def calculate_scale_ratio(
    source_resolution,
    reference_resolution
):
    """
    Resolution should be in the same physical/angular units.
    """

    if reference_resolution <= 0:
        raise ValueError(
            "Reference resolution must be positive."
        )

    return (
        source_resolution
        / reference_resolution
    )


def resize_to_resolution(
    image,
    current_resolution,
    target_resolution
):
    """
    Resample image so its effective resolution approaches
    target_resolution.

    Example:

        OHRC 0.25 m/px
             ↓
        target = 5 m/px

    implies roughly 20x downsampling.
    """

    if (
        current_resolution <= 0
        or
        target_resolution <= 0
    ):
        raise ValueError(
            "Resolution values must be positive."
        )

    scale = (
        current_resolution
        / target_resolution
    )

    new_width = max(
        1,
        int(
            round(
                image.shape[1]
                * scale
            )
        )
    )

    new_height = max(
        1,
        int(
            round(
                image.shape[0]
                * scale
            )
        )
    )


    if scale < 1.0:

        interpolation = (
            cv2.INTER_AREA
        )

    else:

        interpolation = (
            cv2.INTER_LINEAR
        )


    resized = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=interpolation
    )


    return resized, scale