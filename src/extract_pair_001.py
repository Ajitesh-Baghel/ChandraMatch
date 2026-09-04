import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PATH_2019 = r"E:\SIH26166\data\raw\tmc2\2019_12_12\data\derived\20191212\ch2_tmc_ndn_20191212T0423276513_d_oth_d18.tif"

PATH_2020 = r"E:\SIH26166\data\raw\tmc2\2020_01_08\data\derived\20200108\ch2_tmc_ndn_20200108T1153531115_d_oth_d18.tif"

OUT_DIR = Path(r"E:\SIH26166\data\processed\pairs\pair_001")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_for_preview(img):
    valid = img[img > 0]

    if valid.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    low, high = np.percentile(valid, (1, 99))

    img = np.clip(img, low, high)

    img = (img - low) / max(high - low, 1)
    img = img * 255

    return img.astype(np.uint8)


with rasterio.open(PATH_2019) as src19, rasterio.open(PATH_2020) as src20:

    left = max(src19.bounds.left, src20.bounds.left)
    right = min(src19.bounds.right, src20.bounds.right)
    bottom = max(src19.bounds.bottom, src20.bounds.bottom)
    top = min(src19.bounds.top, src20.bounds.top)

    print("Overlap bounds:")
    print(left, bottom, right, top)

    if left >= right or bottom >= top:
        raise RuntimeError("No overlap found.")

    win19 = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=src19.transform
    )

    win20 = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=src20.transform
    )

    out_width = 2048
    aspect = (top - bottom) / (right - left)
    out_height = int(out_width * aspect)

    print("Output size:", out_width, "x", out_height)

    img19 = src19.read(
        1,
        window=win19,
        out_shape=(out_height, out_width),
        resampling=Resampling.bilinear
    )

    img20 = src20.read(
        1,
        window=win20,
        out_shape=(out_height, out_width),
        resampling=Resampling.bilinear
    )


mask19 = img19 > 0
mask20 = img20 > 0
common = mask19 & mask20

print("2019 valid ratio:", mask19.mean())
print("2020 valid ratio:", mask20.mean())
print("Common valid ratio:", common.mean())

img19[~common] = 0
img20[~common] = 0

preview19 = normalize_for_preview(img19)
preview20 = normalize_for_preview(img20)

plt.imsave(
    OUT_DIR / "tmc2_2019_preview.png",
    preview19,
    cmap="gray"
)

plt.imsave(
    OUT_DIR / "tmc2_2020_preview.png",
    preview20,
    cmap="gray"
)

plt.imsave(
    OUT_DIR / "common_mask.png",
    common.astype(np.uint8),
    cmap="gray"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 10))

axes[0].imshow(preview19, cmap="gray")
axes[0].set_title("TMC-2 2019-12-12")
axes[0].axis("off")

axes[1].imshow(preview20, cmap="gray")
axes[1].set_title("TMC-2 2020-01-08")
axes[1].axis("off")

plt.tight_layout()

plt.savefig(
    OUT_DIR / "pair_001_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved outputs to:")
print(OUT_DIR)