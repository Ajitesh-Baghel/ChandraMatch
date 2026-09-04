import sys
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.ingestion.ohrc_product import (
    open_ohrc_image,
    make_quicklook,
    read_ohrc_window
)


PRODUCT_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "ohrc"
    / "2021_12_28"
)


DATA_DIR = (
    PRODUCT_ROOT
    / "data"
    / "calibrated"
    / "20211228"
)


IMG_PATH = next(
    DATA_DIR.glob(
        "*.img"
    )
)


XML_PATH = next(
    DATA_DIR.glob(
        "*.xml"
    )
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "ohrc"
    / "2021_12_28"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)

print(
    "CHANDRAMATCH - OHRC NATIVE READER TEST"
)

print("=" * 70)


print(
    "IMG:"
)

print(
    IMG_PATH
)


print(
    "\nXML:"
)

print(
    XML_PATH
)


# ============================================================
# OPEN IMAGE
# ============================================================

image, metadata = (
    open_ohrc_image(
        IMG_PATH,
        XML_PATH
    )
)


print(
    "\nNative image shape:",
    image.shape
)

print(
    "dtype:",
    image.dtype
)

print(
    "Memory mapped:",
    isinstance(
        image,
        np.memmap
    )
)


print("\nMetadata:")

for key, value in (
    metadata.items()
):

    print(
        f"{key}: {value}"
    )


# ============================================================
# QUICKLOOK
# ============================================================

quicklook = make_quicklook(

    image,

    max_width=1200
)


quicklook_path = (
    OUTPUT_DIR
    / "native_quicklook.png"
)


cv2.imwrite(
    str(
        quicklook_path
    ),
    quicklook
)


print(
    "\nQuicklook shape:",
    quicklook.shape
)

print(
    "Quicklook saved:"
)

print(
    quicklook_path
)


# ============================================================
# CENTRAL NATIVE PATCH
# ============================================================

height, width = (
    image.shape
)


patch_size = 2048


x = (
    width // 2
    -
    patch_size // 2
)

y = (
    height // 2
    -
    patch_size // 2
)


patch = read_ohrc_window(

    image,

    x,
    y,

    patch_size,
    patch_size
)


patch_path = (
    OUTPUT_DIR
    / "native_center_2048.png"
)


cv2.imwrite(
    str(
        patch_path
    ),
    patch
)


print(
    "\nCentral patch shape:",
    patch.shape
)

print(
    "Central native patch saved:"
)

print(
    patch_path
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = (
    OUTPUT_DIR
    / "metadata.json"
)


with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print("\n")
print("=" * 70)

print(
    "OHRC READER TEST COMPLETE"
)

print("=" * 70)