import re
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import rasterio


# ============================================================
# XML HELPERS
# ============================================================

def clean_tag(tag):
    """
    Remove XML namespace and normalize tag name.
    """

    if "}" in tag:
        tag = tag.split("}", 1)[1]

    return tag.strip()


def normalized_tag(tag):
    """
    Simplified form used for matching different XML naming
    conventions.
    """

    tag = clean_tag(tag).lower()

    return re.sub(
        r"[^a-z0-9]",
        "",
        tag
    )


def try_float(value):

    if value is None:
        return None

    value = str(value).strip()

    try:
        return float(value)
    except ValueError:
        pass

    # Sometimes metadata can contain:
    # "34.52 deg"

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        value
    )

    if match:

        try:
            return float(
                match.group()
            )
        except ValueError:
            return None

    return None


# ============================================================
# ACQUISITION DATE
# ============================================================

def acquisition_from_product_id(product_id):

    # Example:
    #
    # ch2_tmc_ndn_20191212T0423276513_d_oth_d18

    match = re.search(
        r"(\d{8})T",
        product_id
    )

    if not match:
        return None

    try:

        date = datetime.strptime(
            match.group(1),
            "%Y%m%d"
        )

        return date.strftime(
            "%Y-%m-%d"
        )

    except ValueError:
        return None


# ============================================================
# XML METADATA EXTRACTION
# ============================================================

def read_xml_fields(xml_path):

    if xml_path is None:
        return []

    xml_path = Path(
        xml_path
    )

    if not xml_path.exists():
        return []

    try:

        tree = ET.parse(
            xml_path
        )

    except Exception as error:

        print(
            f"Could not parse XML {xml_path}: {error}"
        )

        return []


    root = tree.getroot()

    fields = []


    for element in root.iter():

        tag = clean_tag(
            element.tag
        )

        text = (
            element.text.strip()
            if element.text
            else ""
        )


        if text:

            fields.append({

                "tag": tag,

                "normalized_tag":
                    normalized_tag(tag),

                "value": text
            })


        # Some XML schemas store values in attributes.

        for attribute_name, attribute_value in (
            element.attrib.items()
        ):

            fields.append({

                "tag":
                    f"{tag}@{attribute_name}",

                "normalized_tag":
                    normalized_tag(
                        attribute_name
                    ),

                "value":
                    str(attribute_value)
            })


    return fields


# ============================================================
# FIND BEST XML FIELD
# ============================================================

def find_numeric_field(
    fields,
    candidate_names
):

    normalized_candidates = [

        re.sub(
            r"[^a-z0-9]",
            "",
            name.lower()
        )

        for name in candidate_names
    ]


    # First try exact normalized matches.

    for candidate in normalized_candidates:

        for field in fields:

            if (
                field["normalized_tag"]
                == candidate
            ):

                number = try_float(
                    field["value"]
                )

                if number is not None:
                    return (
                        number,
                        field["tag"]
                    )


    # Then allow partial matches.

    for candidate in normalized_candidates:

        for field in fields:

            if (
                candidate
                in field["normalized_tag"]
            ):

                number = try_float(
                    field["value"]
                )

                if number is not None:

                    return (
                        number,
                        field["tag"]
                    )


    return (
        None,
        None
    )


# ============================================================
# ILLUMINATION METADATA
# ============================================================

def extract_illumination_metadata(
    fields
):

    sun_azimuth, sun_azimuth_tag = (
        find_numeric_field(

            fields,

            [
                "sun_azimuth",
                "sun_azimuth_angle",
                "solar_azimuth",
                "solar_azimuth_angle"
            ]
        )
    )


    sun_elevation, sun_elevation_tag = (
        find_numeric_field(

            fields,

            [
                "sun_elevation",
                "sun_elevation_angle",
                "solar_elevation",
                "solar_elevation_angle"
            ]
        )
    )


    incidence, incidence_tag = (
        find_numeric_field(

            fields,

            [
                "incidence_angle",
                "solar_incidence_angle",
                "sun_incidence_angle"
            ]
        )
    )


    emission, emission_tag = (
        find_numeric_field(

            fields,

            [
                "emission_angle",
                "sensor_emission_angle"
            ]
        )
    )


    phase, phase_tag = (
        find_numeric_field(

            fields,

            [
                "phase_angle",
                "solar_phase_angle"
            ]
        )
    )


    return {

        "sun_azimuth_deg":
            sun_azimuth,

        "sun_azimuth_tag":
            sun_azimuth_tag,

        "sun_elevation_deg":
            sun_elevation,

        "sun_elevation_tag":
            sun_elevation_tag,

        "incidence_angle_deg":
            incidence,

        "incidence_angle_tag":
            incidence_tag,

        "emission_angle_deg":
            emission,

        "emission_angle_tag":
            emission_tag,

        "phase_angle_deg":
            phase,

        "phase_angle_tag":
            phase_tag
    }


# ============================================================
# FIND MATCHING XML
# ============================================================

def find_xml_for_tif(
    tif_path
):

    tif_path = Path(
        tif_path
    )

    direct = tif_path.with_suffix(
        ".xml"
    )

    if direct.exists():
        return direct


    # Fallback:
    # search product directory

    product_stem = (
        tif_path.stem
    )


    for xml_path in (
        tif_path.parent.rglob(
            "*.xml"
        )
    ):

        if (
            product_stem
            in xml_path.stem
        ):

            return xml_path


    return None


# ============================================================
# BUILD ONE PRODUCT RECORD
# ============================================================

def inspect_tmc2_product(
    tif_path
):

    tif_path = Path(
        tif_path
    )


    product_id = tif_path.stem


    xml_path = find_xml_for_tif(
        tif_path
    )


    xml_fields = read_xml_fields(
        xml_path
    )


    illumination = (
        extract_illumination_metadata(
            xml_fields
        )
    )


    with rasterio.open(
        tif_path
    ) as src:

        record = {

            "product_id":
                product_id,

            "acquisition_date":
                acquisition_from_product_id(
                    product_id
                ),

            "tif_path":
                str(
                    tif_path.resolve()
                ),

            "xml_path":
                (
                    str(
                        xml_path.resolve()
                    )
                    if xml_path
                    else None
                ),

            "width":
                src.width,

            "height":
                src.height,

            "resolution_x":
                abs(
                    src.res[0]
                ),

            "resolution_y":
                abs(
                    src.res[1]
                ),

            "left":
                src.bounds.left,

            "bottom":
                src.bounds.bottom,

            "right":
                src.bounds.right,

            "top":
                src.bounds.top,

            "crs":
                str(src.crs)
        }


    record.update(
        illumination
    )


    return (
        record,
        xml_fields
    )


# ============================================================
# ILLUMINATION-RELATED XML FIELDS
# ============================================================

def illumination_related_fields(
    fields
):

    keywords = [

        "sun",
        "solar",
        "illumination",
        "incidence",
        "emission",
        "phase",
        "azimuth",
        "elevation"
    ]


    result = []


    for field in fields:

        normalized = field[
            "normalized_tag"
        ]

        if any(
            keyword in normalized
            for keyword in keywords
        ):

            result.append(
                field
            )


    return result