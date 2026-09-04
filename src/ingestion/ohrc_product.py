from pathlib import Path
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np


# ============================================================
# XML HELPERS
# ============================================================

def clean_tag(tag):
    if "}" in tag:
        tag = tag.split("}", 1)[1]

    return tag


def normalize_tag(tag):
    return re.sub(
        r"[^a-z0-9]",
        "",
        clean_tag(tag).lower()
    )


def find_first_text(
    root,
    names
):
    names = {
        re.sub(
            r"[^a-z0-9]",
            "",
            name.lower()
        )
        for name in names
    }

    for element in root.iter():

        tag = normalize_tag(
            element.tag
        )

        if (
            tag in names
            and
            element.text
        ):

            return (
                element.text.strip()
            )

    return None


def find_first_float(
    root,
    names
):
    value = find_first_text(
        root,
        names
    )

    if value is None:
        return None

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        value
    )

    if not match:
        return None

    return float(
        match.group()
    )


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

def read_axis_dimensions(root):
    """
    Read PDS4 Axis_Array definitions.

    Expected OHRC structure:
        Line   -> 79796
        Sample -> 12000
    """

    axes = {}

    for element in root.iter():

        if (
            normalize_tag(
                element.tag
            )
            != "axisarray"
        ):
            continue

        axis_name = None
        elements = None

        for child in element.iter():

            tag = normalize_tag(
                child.tag
            )

            if (
                tag == "axisname"
                and child.text
            ):
                axis_name = (
                    child.text
                    .strip()
                    .lower()
                )

            elif (
                tag == "elements"
                and child.text
            ):

                elements = int(
                    child.text.strip()
                )

        if (
            axis_name is not None
            and elements is not None
        ):
            axes[
                axis_name
            ] = elements


    if (
        "line" not in axes
        or
        "sample" not in axes
    ):

        raise RuntimeError(
            "Could not determine OHRC "
            "Line/Sample dimensions."
        )


    return (
        axes["line"],
        axes["sample"]
    )


# ============================================================
# DATA TYPE
# ============================================================

def pds_dtype_to_numpy(
    data_type
):

    mapping = {

        "UnsignedByte":
            np.uint8,

        "SignedByte":
            np.int8,

        "UnsignedMSB2":
            ">u2",

        "UnsignedLSB2":
            "<u2",

        "SignedMSB2":
            ">i2",

        "SignedLSB2":
            "<i2",

        "IEEE754MSBSingle":
            ">f4",

        "IEEE754LSBSingle":
            "<f4",
    }


    if data_type not in mapping:

        raise ValueError(
            f"Unsupported PDS data type: "
            f"{data_type}"
        )


    return np.dtype(
        mapping[data_type]
    )


# ============================================================
# READ LABEL
# ============================================================

def read_ohrc_label(
    xml_path
):

    xml_path = Path(
        xml_path
    )


    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()


    lines, samples = (
        read_axis_dimensions(
            root
        )
    )


    data_type = find_first_text(
        root,
        ["data_type"]
    )


    if data_type is None:
        raise RuntimeError(
            "OHRC data_type missing "
            "from XML."
        )


    offset_value = find_first_float(
        root,
        ["offset"]
    )


    if offset_value is None:
        offset_value = 0


    file_name = find_first_text(
        root,
        ["file_name"]
    )


    metadata = {

        "file_name":
            file_name,

        "lines":
            lines,

        "samples":
            samples,

        "data_type":
            data_type,

        "numpy_dtype":
            str(
                pds_dtype_to_numpy(
                    data_type
                )
            ),

        "offset_bytes":
            int(offset_value),

        "pixel_resolution_m":
            find_first_float(
                root,
                ["pixel_resolution"]
            ),

        "spacecraft_altitude_km":
            find_first_float(
                root,
                ["spacecraft_altitude"]
            ),

        "sun_azimuth_deg":
            find_first_float(
                root,
                ["sun_azimuth"]
            ),

        "sun_elevation_deg":
            find_first_float(
                root,
                ["sun_elevation"]
            ),

        "solar_incidence_deg":
            find_first_float(
                root,
                ["solar_incidence"]
            ),

        "upper_left_latitude":
            find_first_float(
                root,
                ["upper_left_latitude"]
            ),

        "upper_left_longitude":
            find_first_float(
                root,
                ["upper_left_longitude"]
            ),

        "upper_right_latitude":
            find_first_float(
                root,
                ["upper_right_latitude"]
            ),

        "upper_right_longitude":
            find_first_float(
                root,
                ["upper_right_longitude"]
            ),

        "lower_left_latitude":
            find_first_float(
                root,
                ["lower_left_latitude"]
            ),

        "lower_left_longitude":
            find_first_float(
                root,
                ["lower_left_longitude"]
            ),

        "lower_right_latitude":
            find_first_float(
                root,
                ["lower_right_latitude"]
            ),

        "lower_right_longitude":
            find_first_float(
                root,
                ["lower_right_longitude"]
            )
    }


    return metadata


# ============================================================
# MEMORY-MAPPED IMAGE
# ============================================================

def open_ohrc_image(
    img_path,
    xml_path
):
    """
    Open native OHRC IMG without loading ~1 GB into RAM.
    """

    img_path = Path(
        img_path
    )


    metadata = read_ohrc_label(
        xml_path
    )


    dtype = pds_dtype_to_numpy(
        metadata[
            "data_type"
        ]
    )


    expected_bytes = (

        metadata["lines"]
        *
        metadata["samples"]
        *
        dtype.itemsize

    )


    actual_bytes = (
        img_path.stat().st_size
        -
        metadata["offset_bytes"]
    )


    if expected_bytes != actual_bytes:

        raise RuntimeError(

            "IMG size does not match XML metadata.\n"
            f"Expected: {expected_bytes} bytes\n"
            f"Actual:   {actual_bytes} bytes"
        )


    image = np.memmap(

        img_path,

        dtype=dtype,

        mode="r",

        offset=
            metadata[
                "offset_bytes"
            ],

        shape=(
            metadata["lines"],
            metadata["samples"]
        ),

        order="C"
    )


    return (
        image,
        metadata
    )


# ============================================================
# READ A WINDOW
# ============================================================

def read_ohrc_window(
    image,
    x,
    y,
    width,
    height
):
    """
    Read only a rectangular native-resolution OHRC region.
    """

    image_height, image_width = (
        image.shape
    )


    x0 = max(
        0,
        int(x)
    )

    y0 = max(
        0,
        int(y)
    )

    x1 = min(
        image_width,
        x0 + int(width)
    )

    y1 = min(
        image_height,
        y0 + int(height)
    )


    return np.asarray(
        image[
            y0:y1,
            x0:x1
        ]
    ).copy()


# ============================================================
# FAST QUICKLOOK
# ============================================================

def make_quicklook(
    image,
    max_width=1200
):
    """
    Efficiently downsample the memory-mapped image.

    Avoids loading the entire ~1 GB image into RAM.
    """

    height, width = (
        image.shape
    )


    stride = max(
        1,
        int(
            np.ceil(
                width
                / max_width
            )
        )
    )


    sampled = np.asarray(
        image[
            ::stride,
            ::stride
        ]
    )


    if sampled.dtype == np.uint8:

        return sampled.copy()


    sampled_float = (
        sampled.astype(
            np.float32
        )
    )


    low, high = np.percentile(
        sampled_float,
        [1, 99]
    )


    sampled_float = np.clip(
        sampled_float,
        low,
        high
    )


    if high > low:

        sampled_float = (

            (
                sampled_float
                - low
            )

            /
            (
                high - low
            )

            * 255.0
        )


    return sampled_float.astype(
        np.uint8
    )