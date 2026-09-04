import sys
import csv
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


GEOMETRY_DIR = (
    ROOT
    / "data"
    / "raw"
    / "ohrc"
    / "2021_12_28"
    / "geometry"
    / "calibrated"
    / "20211228"
)


CSV_PATH = next(
    GEOMETRY_DIR.glob(
        "*.csv"
    )
)


XML_PATH = next(
    GEOMETRY_DIR.glob(
        "*.xml"
    )
)


def clean_tag(tag):

    if "}" in tag:
        tag = tag.split(
            "}",
            1
        )[1]

    return tag


print("=" * 80)

print(
    "CHANDRAMATCH - OHRC GEOMETRY INSPECTOR"
)

print("=" * 80)


print(
    "\nCSV:"
)

print(
    CSV_PATH
)


print(
    "\nXML:"
)

print(
    XML_PATH
)


# ============================================================
# RAW CSV LINES
# ============================================================

print("\n")
print("=" * 80)

print(
    "FIRST 10 RAW CSV LINES"
)

print("=" * 80)


with open(
    CSV_PATH,
    "r",
    encoding="utf-8",
    errors="replace"
) as file:

    for index in range(10):

        line = file.readline()

        if not line:
            break

        print(
            f"{index + 1:02d}: "
            f"{line.rstrip()}"
        )


# ============================================================
# CSV STRUCTURE
# ============================================================

print("\n")
print("=" * 80)

print(
    "PANDAS CSV INSPECTION"
)

print("=" * 80)


try:

    df = pd.read_csv(
        CSV_PATH
    )


    print(
        "Shape:",
        df.shape
    )

    print(
        "Columns:"
    )

    for column in df.columns:

        print(
            repr(column)
        )


    print(
        "\nFirst rows:"
    )

    print(
        df.head(
            10
        ).to_string()
    )


except Exception as error:

    print(
        "Normal CSV parsing failed:"
    )

    print(
        error
    )


    print(
        "\nTrying header=None..."
    )


    df = pd.read_csv(

        CSV_PATH,

        header=None
    )


    print(
        "Shape:",
        df.shape
    )


    print(
        df.head(
            10
        ).to_string(
            header=False,
            index=False
        )
    )


# ============================================================
# XML FIELD DEFINITIONS
# ============================================================

print("\n")
print("=" * 80)

print(
    "XML FIELD DEFINITIONS"
)

print("=" * 80)


tree = ET.parse(
    XML_PATH
)

root = tree.getroot()


field_number = 0


for element in root.iter():

    tag = clean_tag(
        element.tag
    )


    if tag.lower() not in {
        "field_delimited",
        "field_fixed_length"
    }:
        continue


    field_number += 1


    print(
        f"\nFIELD {field_number}"
    )

    print(
        "-" * 40
    )


    for child in element.iter():

        child_tag = clean_tag(
            child.tag
        )

        text = (
            child.text.strip()
            if child.text
            else ""
        )


        if text:

            print(
                f"{child_tag}: {text}"
            )


# ============================================================
# ALL NAME / DESCRIPTION TAGS
# ============================================================

print("\n")
print("=" * 80)

print(
    "POSSIBLE GEOMETRY FIELD NAMES"
)

print("=" * 80)


for element in root.iter():

    tag = clean_tag(
        element.tag
    ).lower()


    if (
        "name" in tag
        or
        "description" in tag
    ):

        text = (
            element.text.strip()
            if element.text
            else ""
        )


        if text:

            print(
                f"{clean_tag(element.tag)}: "
                f"{text}"
            )


print("\n")
print("=" * 80)

print(
    "GEOMETRY INSPECTION COMPLETE"
)

print("=" * 80)