from src.preprocessing.illumination import (
    get_illumination_variant
)


def preprocess_for_matcher(
    image,
    matcher_name,
    illumination_robust=True
):
    """
    Select preprocessing based on experimental results.

    Current policy derived from real TMC-2 Sun-angle benchmark.
    """

    matcher_name = matcher_name.lower()

    if not illumination_robust:
        return image.copy()

    if matcher_name == "sift":
        return get_illumination_variant(
            image,
            "gradient"
        )

    if matcher_name in {
        "lightglue",
        "superpoint_lightglue"
    }:
        return get_illumination_variant(
            image,
            "baseline"
        )

    if matcher_name == "loftr":
        return get_illumination_variant(
            image,
            "baseline"
        )

    raise ValueError(
        f"Unknown matcher: {matcher_name}"
    )