import cv2
import numpy as np


def match_sift(
    source,
    reference,
    mask=None,
    ratio_threshold=0.75,
    max_features=10000
):
    """
    Detect SIFT features and perform descriptor matching.

    Returns:
        source_points
        reference_points
        good_matches
        source_keypoints
        reference_keypoints
    """

    sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=0.02,
        edgeThreshold=10,
        sigma=1.6
    )

    if mask is not None:
        mask_uint8 = (
            mask.astype(np.uint8) * 255
        )
    else:
        mask_uint8 = None

    source_kp, source_desc = sift.detectAndCompute(
        source,
        mask_uint8
    )

    reference_kp, reference_desc = sift.detectAndCompute(
        reference,
        mask_uint8
    )

    print("Source keypoints:", len(source_kp))
    print("Reference keypoints:", len(reference_kp))

    if source_desc is None or reference_desc is None:
        raise RuntimeError(
            "SIFT could not generate descriptors."
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )

    matches = matcher.knnMatch(
        source_desc,
        reference_desc,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    source_points = np.float32(
        [
            source_kp[m.queryIdx].pt
            for m in good_matches
        ]
    )

    reference_points = np.float32(
        [
            reference_kp[m.trainIdx].pt
            for m in good_matches
        ]
    )

    return (
        source_points,
        reference_points,
        good_matches,
        source_kp,
        reference_kp
    )