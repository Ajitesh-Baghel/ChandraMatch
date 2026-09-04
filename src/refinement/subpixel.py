import cv2
import numpy as np


def transform_points(
    points,
    transformation
):
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


def gradient_image(image):
    """
    Structural representation that is less dependent on
    absolute brightness.
    """

    image = image.astype(
        np.float32
    )

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    return magnitude


def normalize_patch(patch):

    patch = patch.astype(
        np.float32
    )

    mean = np.mean(patch)
    std = np.std(patch)

    if std < 1e-6:
        return None

    return (
        patch - mean
    ) / std


def refine_correspondences_phase(
    source,
    reference,
    source_points,
    transformation,
    patch_size=64,
    max_shift=3.0,
    min_response=0.05,
    use_gradient=True
):
    """
    Sub-pixel local refinement.

    Workflow:

        SOURCE
          ↓
        global affine registration
          ↓
        approximately aligned source
          ↓
        local phase correlation
          ↓
        sub-pixel residual displacement
          ↓
        refined reference coordinate

    Returns only points for which refinement was considered
    reliable.
    """

    height, width = reference.shape[:2]


    # --------------------------------------------------------
    # Warp source into reference coordinate system
    # --------------------------------------------------------

    registered_source = cv2.warpAffine(

        source,

        transformation,

        (
            width,
            height
        ),

        flags=cv2.INTER_LINEAR,

        borderMode=cv2.BORDER_CONSTANT,

        borderValue=0
    )


    # --------------------------------------------------------
    # Structural representation
    # --------------------------------------------------------

    if use_gradient:

        registered_work = gradient_image(
            registered_source
        )

        reference_work = gradient_image(
            reference
        )

    else:

        registered_work = (
            registered_source
            .astype(np.float32)
        )

        reference_work = (
            reference
            .astype(np.float32)
        )


    # --------------------------------------------------------
    # Global prediction of correspondence positions
    # --------------------------------------------------------

    predicted_reference = transform_points(

        source_points,

        transformation
    )


    half = patch_size // 2


    hanning = cv2.createHanningWindow(

        (
            patch_size,
            patch_size
        ),

        cv2.CV_32F
    )


    refined_source = []
    refined_reference = []
    geometric_reference = []
    responses = []
    local_shifts = []
    kept_indices = []


    # --------------------------------------------------------
    # Refine every selected correspondence
    # --------------------------------------------------------

    for index, point in enumerate(
        predicted_reference
    ):

        x = float(point[0])
        y = float(point[1])


        # Avoid border patches.

        margin = (
            half
            + int(np.ceil(max_shift))
            + 2
        )


        if (
            x < margin
            or
            y < margin
            or
            x >= width - margin
            or
            y >= height - margin
        ):
            continue


        registered_patch = cv2.getRectSubPix(

            registered_work,

            (
                patch_size,
                patch_size
            ),

            (
                x,
                y
            )
        )


        reference_patch = cv2.getRectSubPix(

            reference_work,

            (
                patch_size,
                patch_size
            ),

            (
                x,
                y
            )
        )


        registered_patch = normalize_patch(
            registered_patch
        )

        reference_patch = normalize_patch(
            reference_patch
        )


        if (
            registered_patch is None
            or
            reference_patch is None
        ):
            continue


        # ----------------------------------------------------
        # Sub-pixel phase correlation
        #
        # This estimates the residual translation from the
        # globally registered source patch to reference patch.
        # ----------------------------------------------------

        shift, response = cv2.phaseCorrelate(

            registered_patch,

            reference_patch,

            hanning
        )


        dx = float(
            shift[0]
        )

        dy = float(
            shift[1]
        )

        response = float(
            response
        )


        if not (
            np.isfinite(dx)
            and
            np.isfinite(dy)
            and
            np.isfinite(response)
        ):
            continue


        if response < min_response:
            continue


        if (
            abs(dx) > max_shift
            or
            abs(dy) > max_shift
        ):
            continue


        refined_point = np.array(

            [
                x + dx,
                y + dy
            ],

            dtype=np.float32
        )


        refined_source.append(
            source_points[index]
        )

        refined_reference.append(
            refined_point
        )

        geometric_reference.append(
            point
        )

        responses.append(
            response
        )

        local_shifts.append(
            [dx, dy]
        )

        kept_indices.append(
            index
        )


    return {

        "source_points":
            np.asarray(
                refined_source,
                dtype=np.float32
            ),

        "reference_points":
            np.asarray(
                refined_reference,
                dtype=np.float32
            ),

        "geometric_reference_points":
            np.asarray(
                geometric_reference,
                dtype=np.float32
            ),

        "responses":
            np.asarray(
                responses,
                dtype=np.float32
            ),

        "local_shifts":
            np.asarray(
                local_shifts,
                dtype=np.float32
            ),

        "kept_indices":
            np.asarray(
                kept_indices,
                dtype=int
            ),

        "registered_source":
            registered_source
    }