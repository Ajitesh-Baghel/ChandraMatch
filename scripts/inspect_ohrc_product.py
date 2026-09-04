import sys
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OHRC_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "ohrc"
    / "2021_12_28"
)

OUTPUT_DIR = (
    ROOT
    / "metadata"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "ohrc_20211228_inspection.json"
)


# ============================================================
# HELPERS
# ============================================================

def clean_tag(tag):
    """
    Remove XML namespaces.
    """

    if "}" in tag:
        tag = tag.split("}", 1)[1]

    return tag


def normalize_tag(tag):

    tag = clean_tag(tag).lower()

    return re.sub(
        r"[^a-z0-9]",
        "",
        tag
    )


def human_size(size_bytes):

    size = float(size_bytes)

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB"
    ]:

        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


# ============================================================
# XML PARSER
# ============================================================

def inspect_xml(xml_path):

    result = {
        "path": str(xml_path),
        "fields": []
    }

    try:

        tree = ET.parse(
            xml_path
        )

        root = tree.getroot()

    except Exception as error:

        result["parse_error"] = str(error)

        return result


    interesting_keywords = [

        # Image dimensions
        "line",
        "lines",
        "sample",
        "samples",
        "axis",
        "elements",

        # Pixel format
        "datatype",
        "bit",
        "byte",
        "offset",
        "scaling",
        "valueoffset",

        # Geography
        "latitude",
        "longitude",
        "corner",
        "center",

        # Illumination
        "sun",
        "solar",
        "azimuth",
        "elevation",
        "incidence",
        "emission",
        "phase",

        # Resolution / geometry
        "resolution",
        "pixel",
        "altitude",
        "spacecraft",

        # File references
        "filename",
        "file_name",

        # Product metadata
        "product",
        "starttime",
        "stoptime",
        "observation"
    ]


    for element in root.iter():

        tag = clean_tag(
            element.tag
        )

        normalized = normalize_tag(
            tag
        )

        text = (
            element.text.strip()
            if element.text
            else ""
        )


        interesting = any(
            keyword in normalized
            for keyword in interesting_keywords
        )


        if interesting and text:

            result["fields"].append({

                "tag": tag,
                "value": text
            })


        for attribute_name, attribute_value in (
            element.attrib.items()
        ):

            normalized_attribute = (
                normalize_tag(
                    attribute_name
                )
            )

            interesting_attribute = any(
                keyword in normalized_attribute
                for keyword in interesting_keywords
            )

            if interesting_attribute:

                result["fields"].append({

                    "tag":
                        f"{tag}@{attribute_name}",

                    "value":
                        str(attribute_value)
                })


    return result


# ============================================================
# FIND ALL FILES
# ============================================================

if not OHRC_ROOT.exists():

    raise RuntimeError(
        f"OHRC directory does not exist: {OHRC_ROOT}"
    )


all_files = sorted(

    path

    for path in OHRC_ROOT.rglob("*")

    if path.is_file()
)


print("=" * 80)

print(
    "CHANDRAMATCH - OHRC PRODUCT INSPECTOR"
)

print("=" * 80)


print(
    "\nProduct directory:"
)

print(
    OHRC_ROOT
)


print(
    "\nTotal files:",
    len(all_files)
)


# ============================================================
# FILE LISTING
# ============================================================

file_records = []


print("\n")
print("=" * 80)

print(
    "FILES"
)

print("=" * 80)


for index, path in enumerate(
    all_files,
    start=1
):

    relative = path.relative_to(
        OHRC_ROOT
    )

    size = path.stat().st_size

    suffix = (
        path.suffix.lower()
        if path.suffix
        else "<none>"
    )


    record = {

        "relative_path":
            str(relative),

        "absolute_path":
            str(path.resolve()),

        "extension":
            suffix,

        "size_bytes":
            size,

        "size":
            human_size(size)
    }


    file_records.append(
        record
    )


    print(
        f"[{index:02d}] "
        f"{relative}"
    )

    print(
        f"     Type: {suffix}"
    )

    print(
        f"     Size: {human_size(size)}"
    )


# ============================================================
# GROUP BY EXTENSION
# ============================================================

extensions = {}


for record in file_records:

    extension = record[
        "extension"
    ]

    extensions.setdefault(
        extension,
        0
    )

    extensions[
        extension
    ] += 1


print("\n")
print("=" * 80)

print(
    "FILE TYPES"
)

print("=" * 80)


for extension, count in sorted(
    extensions.items()
):

    print(
        f"{extension:12s}: {count}"
    )


# ============================================================
# XML INSPECTION
# ============================================================

xml_files = [

    path

    for path in all_files

    if path.suffix.lower() == ".xml"
]


xml_results = []


print("\n")
print("=" * 80)

print(
    "XML LABEL INSPECTION"
)

print("=" * 80)


if not xml_files:

    print(
        "No XML files found."
    )


for xml_path in xml_files:

    print("\n")
    print("-" * 80)

    print(
        xml_path.relative_to(
            OHRC_ROOT
        )
    )

    print("-" * 80)


    result = inspect_xml(
        xml_path
    )


    xml_results.append(
        result
    )


    if "parse_error" in result:

        print(
            "XML parse error:",
            result[
                "parse_error"
            ]
        )

        continue


    if not result["fields"]:

        print(
            "No selected metadata fields found."
        )

        continue


    for field in result[
        "fields"
    ]:

        print(
            f"{field['tag']}: "
            f"{field['value']}"
        )


# ============================================================
# BROWSE IMAGE INSPECTION
# ============================================================

browse_extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


browse_results = []


print("\n")
print("=" * 80)

print(
    "BROWSE / STANDARD IMAGE FILES"
)

print("=" * 80)


for path in all_files:

    if (
        path.suffix.lower()
        not in browse_extensions
    ):
        continue


    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )


    record = {

        "path":
            str(path.resolve()),

        "readable_by_opencv":
            image is not None
    }


    if image is not None:

        record[
            "shape"
        ] = list(
            image.shape
        )

        record[
            "dtype"
        ] = str(
            image.dtype
        )


        print("\n")
        print(
            path.relative_to(
                OHRC_ROOT
            )
        )

        print(
            "Shape:",
            image.shape
        )

        print(
            "dtype:",
            image.dtype
        )


    browse_results.append(
        record
    )


# ============================================================
# LARGE BINARY FILES
# ============================================================

binary_extensions = {
    ".img",
    ".dat",
    ".bin"
}


binary_results = []


print("\n")
print("=" * 80)

print(
    "BINARY IMAGE CANDIDATES"
)

print("=" * 80)


for path in all_files:

    if (
        path.suffix.lower()
        not in binary_extensions
    ):
        continue


    size = path.stat().st_size


    record = {

        "path":
            str(path.resolve()),

        "size_bytes":
            size,

        "size":
            human_size(size)
    }


    binary_results.append(
        record
    )


    print("\n")

    print(
        path.relative_to(
            OHRC_ROOT
        )
    )

    print(
        "Size:",
        human_size(size)
    )


# ============================================================
# README
# ============================================================

readme_files = [

    path

    for path in all_files

    if (
        "readme"
        in path.name.lower()
    )
]


readme_contents = []


print("\n")
print("=" * 80)

print(
    "README"
)

print("=" * 80)


for readme_path in readme_files:

    print("\n")

    print(
        readme_path.relative_to(
            OHRC_ROOT
        )
    )

    print("-" * 80)


    try:

        text = readme_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

    except Exception as error:

        print(
            "Could not read:",
            error
        )

        continue


    # Avoid dumping enormous text files.
    preview = text[
        :12000
    ]


    print(
        preview
    )


    readme_contents.append({

        "path":
            str(
                readme_path.resolve()
            ),

        "preview":
            preview
    })


# ============================================================
# SAVE STRUCTURED INSPECTION
# ============================================================

report = {

    "product_root":
        str(
            OHRC_ROOT.resolve()
        ),

    "total_files":
        len(all_files),

    "file_types":
        extensions,

    "files":
        file_records,

    "xml":
        xml_results,

    "browse_images":
        browse_results,

    "binary_candidates":
        binary_results,

    "readme":
        readme_contents
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n")
print("=" * 80)

print(
    "OHRC INSPECTION COMPLETE"
)

print("=" * 80)


print(
    "\nSaved:"
)

print(
    OUTPUT_JSON
)