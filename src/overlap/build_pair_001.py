import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import reproject


SOURCE_PATH = (
    r"E:\SIH26166\data\raw\tmc2\2019_12_12"
    r"\data\derived\20191212"
    r"\ch2_tmc_ndn_20191212T0423276513_d_oth_d18.tif"
)

REFERENCE_PATH = (
    r"E:\SIH26166\data\raw\tmc2\2020_01_08"
    r"\data\derived\20200108"
    r"\ch2_tmc_ndn_20200108T1153531115_d_oth_d18.tif"
)

OUT_DIR = Path(
    r"E:\SIH26166\data\processed\pairs\pair_001"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 2048
SEARCH_STRIDE = 512
PROBE_SIZE = 256


def normalize_preview(image, valid_mask):
    output = np.zeros(image.shape, dtype=np.uint8)

    values = image[valid_mask]

    if values.size == 0:
        return output

    low, high = np.percentile(values, (1, 99))

    if high <= low:
        return output

    clipped = np.clip(image.astype(np.float32), low, high)

    normalized = (clipped - low) / (high - low)
    normalized = normalized * 255

    output[valid_mask] = normalized[valid_mask].astype(np.uint8)

    return output


def read_probe(src, bounds):
    left, bottom, right, top = bounds

    window = window_from_bounds(
        left,
        bottom,
        right,
        top,
        transform=src.transform
    )

    image = src.read(
        1,
        window=window,
        out_shape=(PROBE_SIZE, PROBE_SIZE),
        boundless=True,
        fill_value=0,
        resampling=Resampling.nearest
    )

    return image


with rasterio.open(SOURCE_PATH) as source, \
     rasterio.open(REFERENCE_PATH) as reference:

    # ---------------------------------------------------------
    # 1. Basic compatibility check
    # ---------------------------------------------------------

    if source.crs != reference.crs:
        raise RuntimeError(
            "Pair 001 currently expects the same CRS."
        )

    resolution_x = abs(source.res[0])
    resolution_y = abs(source.res[1])

    print("Source resolution:", source.res)
    print("Reference resolution:", reference.res)

    # ---------------------------------------------------------
    # 2. Geographic bounding-box intersection
    # ---------------------------------------------------------

    intersection_left = max(
        source.bounds.left,
        reference.bounds.left
    )

    intersection_right = min(
        source.bounds.right,
        reference.bounds.right
    )

    intersection_bottom = max(
        source.bounds.bottom,
        reference.bounds.bottom
    )

    intersection_top = min(
        source.bounds.top,
        reference.bounds.top
    )

    if (
        intersection_left >= intersection_right
        or intersection_bottom >= intersection_top
    ):
        raise RuntimeError("No geographic overlap.")

    print("\nIntersection:")
    print(
        intersection_left,
        intersection_bottom,
        intersection_right,
        intersection_top
    )

    # ---------------------------------------------------------
    # 3. Dimensions of one 2048x2048 patch in lunar coordinates
    # ---------------------------------------------------------

    patch_world_width = PATCH_SIZE * resolution_x
    patch_world_height = PATCH_SIZE * resolution_y

    intersection_width = (
        intersection_right - intersection_left
    )

    intersection_height = (
        intersection_top - intersection_bottom
    )

    if (
        patch_world_width > intersection_width
        or patch_world_height > intersection_height
    ):
        raise RuntimeError(
            "Intersection is too small for 2048x2048 patch."
        )

    # ---------------------------------------------------------
    # 4. Search for patch having maximum valid pixels in BOTH
    # ---------------------------------------------------------

    stride_x = SEARCH_STRIDE * resolution_x
    stride_y = SEARCH_STRIDE * resolution_y

    best_ratio = -1
    best_bounds = None

    y_top = intersection_top

    checked = 0

    while y_top - patch_world_height >= intersection_bottom:

        x_left = intersection_left

        while x_left + patch_world_width <= intersection_right:

            x_right = x_left + patch_world_width
            y_bottom = y_top - patch_world_height

            candidate_bounds = (
                x_left,
                y_bottom,
                x_right,
                y_top
            )

            source_probe = read_probe(
                source,
                candidate_bounds
            )

            reference_probe = read_probe(
                reference,
                candidate_bounds
            )

            source_valid = source_probe > 0
            reference_valid = reference_probe > 0

            common_valid = (
                source_valid
                & reference_valid
            )

            ratio = common_valid.mean()

            checked += 1

            if ratio > best_ratio:
                best_ratio = ratio
                best_bounds = candidate_bounds

            x_left += stride_x

        y_top -= stride_y

    print("\nCandidate patches checked:", checked)
    print("Best common-valid ratio:", best_ratio)
    print("Best bounds:", best_bounds)

    if best_bounds is None:
        raise RuntimeError("Could not find valid patch.")

    # ---------------------------------------------------------
    # 5. Build one shared target grid
    # ---------------------------------------------------------

    left, bottom, right, top = best_bounds

    target_transform = from_bounds(
        left,
        bottom,
        right,
        top,
        PATCH_SIZE,
        PATCH_SIZE
    )

    source_patch = np.zeros(
        (PATCH_SIZE, PATCH_SIZE),
        dtype=np.uint16
    )

    reference_patch = np.zeros(
        (PATCH_SIZE, PATCH_SIZE),
        dtype=np.uint16
    )

    # ---------------------------------------------------------
    # 6. Reproject SOURCE onto common grid
    # ---------------------------------------------------------

    reproject(
        source=rasterio.band(source, 1),
        destination=source_patch,

        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=0,

        dst_transform=target_transform,
        dst_crs=source.crs,
        dst_nodata=0,

        resampling=Resampling.bilinear
    )

    # ---------------------------------------------------------
    # 7. Reproject REFERENCE onto same grid
    # ---------------------------------------------------------

    reproject(
        source=rasterio.band(reference, 1),
        destination=reference_patch,

        src_transform=reference.transform,
        src_crs=reference.crs,
        src_nodata=0,

        dst_transform=target_transform,
        dst_crs=reference.crs,
        dst_nodata=0,

        resampling=Resampling.bilinear
    )

    # ---------------------------------------------------------
    # 8. Common valid mask
    # ---------------------------------------------------------

    source_valid = source_patch > 0
    reference_valid = reference_patch > 0

    common_valid = (
        source_valid
        & reference_valid
    )

    final_valid_ratio = common_valid.mean()

    print("\nFinal common-valid ratio:")
    print(final_valid_ratio)

    # ---------------------------------------------------------
    # 9. Save georeferenced TIFFs
    # ---------------------------------------------------------

    profile = source.profile.copy()

    profile.update(
        width=PATCH_SIZE,
        height=PATCH_SIZE,
        count=1,
        dtype="uint16",
        transform=target_transform,
        crs=source.crs,
        nodata=0,
        compress="deflate"
    )

    source_tif = OUT_DIR / "source.tif"

    with rasterio.open(
        source_tif,
        "w",
        **profile
    ) as dst:
        dst.write(source_patch, 1)

    reference_tif = OUT_DIR / "reference.tif"

    with rasterio.open(
        reference_tif,
        "w",
        **profile
    ) as dst:
        dst.write(reference_patch, 1)

    # ---------------------------------------------------------
    # 10. Preview images
    # ---------------------------------------------------------

    source_preview = normalize_preview(
        source_patch,
        common_valid
    )

    reference_preview = normalize_preview(
        reference_patch,
        common_valid
    )

    plt.imsave(
        OUT_DIR / "source.png",
        source_preview,
        cmap="gray"
    )

    plt.imsave(
        OUT_DIR / "reference.png",
        reference_preview,
        cmap="gray"
    )

    plt.imsave(
        OUT_DIR / "valid_mask.png",
        common_valid.astype(np.uint8),
        cmap="gray"
    )

    # ---------------------------------------------------------
    # 11. Side-by-side image
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    axes[0].imshow(
        source_preview,
        cmap="gray"
    )

    axes[0].set_title(
        "SOURCE - TMC-2 2019-12-12"
    )

    axes[0].axis("off")

    axes[1].imshow(
        reference_preview,
        cmap="gray"
    )

    axes[1].set_title(
        "REFERENCE - TMC-2 2020-01-08"
    )

    axes[1].axis("off")

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "pair_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # 12. Save metadata
    # ---------------------------------------------------------

    metadata = {
        "pair_id": "pair_001",

        "source": {
            "sensor": "TMC-2",
            "date": "2019-12-12",
            "product_id":
                "ch2_tmc_ndn_20191212T0423276513_d_oth_d18",
            "original_path": SOURCE_PATH
        },

        "reference": {
            "sensor": "TMC-2",
            "date": "2020-01-08",
            "product_id":
                "ch2_tmc_ndn_20200108T1153531115_d_oth_d18",
            "original_path": REFERENCE_PATH
        },

        "patch_size": PATCH_SIZE,

        "bounds": {
            "left": left,
            "bottom": bottom,
            "right": right,
            "top": top
        },

        "resolution": {
            "x": resolution_x,
            "y": resolution_y
        },

        "common_valid_ratio": float(
            final_valid_ratio
        ),

        "crs": str(source.crs),

        "source_file": "source.tif",
        "reference_file": "reference.tif"
    }

    with open(
        OUT_DIR / "pair_metadata.json",
        "w"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )


print("\nPAIR 001 CREATED SUCCESSFULLY")

print("\nFiles:")
for path in OUT_DIR.iterdir():
    print("-", path.name)