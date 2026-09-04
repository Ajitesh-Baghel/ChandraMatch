from pathlib import Path
import json
import re


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NAC_DIR = (
    ROOT
    / "data"
    / "raw"
    / "lro_nac"
    / "2012_02_29"
)

PRODUCT_ID = "M185196277LE"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "pair_004"
    / "inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LABEL_TXT = (
    OUTPUT_DIR
    / "M185196277LE_label.txt"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "M185196277LE_summary.json"
)


# ============================================================
# FIND PRODUCT
# ============================================================

def find_product():

    candidates = []

    for path in NAC_DIR.iterdir():

        if (
            path.is_file()
            and
            path.stem.upper() == PRODUCT_ID
            and
            path.suffix.lower() in [".img", ".imq"]
        ):

            candidates.append(path)

    if not candidates:

        # Allow filenames with additional suffixes.
        for path in NAC_DIR.iterdir():

            if (
                path.is_file()
                and
                PRODUCT_ID.lower()
                in path.name.lower()
            ):

                candidates.append(path)

    if not candidates:

        raise FileNotFoundError(
            f"Could not find {PRODUCT_ID} under:\n"
            f"{NAC_DIR}"
        )

    candidates.sort(
        key=lambda p: p.stat().st_size,
        reverse=True
    )

    return candidates[0]


# ============================================================
# READ PDS LABEL
# ============================================================

def read_pds_label(path):

    # PDS3 EDR label is stored at the beginning of the IMG.
    # Read enough of the file to reach END.

    max_read = min(
        path.stat().st_size,
        4 * 1024 * 1024
    )

    with path.open("rb") as file:

        raw = file.read(max_read)

    text = raw.decode(
        "ascii",
        errors="ignore"
    )

    # Find standalone END marking the end of the PDS label.

    match = re.search(
        r"(?m)^\s*END\s*$",
        text
    )

    if match:

        text = text[
            :match.end()
        ]

    return text


# ============================================================
# SIMPLE PDS VALUE PARSER
# ============================================================

def clean_value(value):

    value = value.strip()

    # Remove comments after value where possible.

    if "/*" in value:

        value = value.split(
            "/*",
            1
        )[0].strip()

    # Remove quotes.

    if (
        len(value) >= 2
        and
        value[0] == '"'
        and
        value[-1] == '"'
    ):

        value = value[
            1:-1
        ]

    return value


def get_value(
    label,
    key
):

    # Match:
    # KEY = something

    pattern = (
        rf"(?m)^\s*"
        rf"{re.escape(key)}"
        rf"\s*=\s*(.+?)\s*$"
    )

    match = re.search(
        pattern,
        label
    )

    if not match:

        return None

    return clean_value(
        match.group(1)
    )


def get_number(
    label,
    key
):

    value = get_value(
        label,
        key
    )

    if value is None:

        return None

    # Remove PDS units:
    #
    # 123 <BYTES>
    # 0.92 <METER/PIXEL>

    value = re.sub(
        r"<[^>]+>",
        "",
        value
    ).strip()

    # Handle something like:
    # 123
    # 123.4

    try:

        if any(
            token in value.lower()
            for token in [
                ".",
                "e"
            ]
        ):

            return float(value)

        return int(value)

    except ValueError:

        match = re.search(
            r"[-+]?"
            r"(?:\d+\.\d+|\d+|\.\d+)"
            r"(?:[Ee][-+]?\d+)?",
            value
        )

        if match:

            try:

                return float(
                    match.group(0)
                )

            except ValueError:

                pass

    return None


# ============================================================
# IMAGE POINTER
# ============================================================

def parse_image_pointer(
    label
):

    pointer = get_value(
        label,
        "^IMAGE"
    )

    if pointer is None:

        return None

    return pointer


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - INSPECT LRO NAC EDR"
    )

    print("=" * 80)

    print(
        "\nLooking under:"
    )

    print(
        NAC_DIR
    )


    # ========================================================
    # PRODUCT
    # ========================================================

    product_path = find_product()

    file_size = (
        product_path.stat().st_size
    )


    print(
        "\nProduct:"
    )

    print(
        product_path
    )

    print(
        "\nFile size:"
    )

    print(
        f"{file_size:,} bytes"
    )

    print(
        f"{file_size / (1024 ** 2):.2f} MiB"
    )

    print(
        f"{file_size / (1024 ** 3):.3f} GiB"
    )


    # ========================================================
    # LABEL
    # ========================================================

    label = read_pds_label(
        product_path
    )

    LABEL_TXT.write_text(
        label,
        encoding="utf-8"
    )


    print("\n")
    print("=" * 80)

    print(
        "PDS LABEL"
    )

    print("=" * 80)


    # ========================================================
    # BASIC METADATA
    # ========================================================

    string_keys = [

        "PDS_VERSION_ID",
        "DATA_SET_ID",

        "PRODUCT_ID",
        "PRODUCT_VERSION_ID",

        "MISSION_NAME",
        "MISSION_PHASE_NAME",

        "INSTRUMENT_HOST_NAME",
        "INSTRUMENT_NAME",
        "INSTRUMENT_ID",

        "TARGET_NAME",

        "START_TIME",
        "STOP_TIME",

        "SPACECRAFT_CLOCK_START_COUNT",
        "SPACECRAFT_CLOCK_STOP_COUNT",

        "NAC_FRAME",

        "SAMPLE_TYPE",

        "COMPRESSION_TYPE",
        "RATIONALE_DESC"
    ]


    numeric_keys = [

        "ORBIT_NUMBER",

        "LINES",
        "LINE_SAMPLES",

        "SAMPLE_BITS",
        "SAMPLE_BIT_MASK",

        "LINE_PREFIX_BYTES",
        "LINE_SUFFIX_BYTES",

        "RECORD_BYTES",
        "FILE_RECORDS",

        "IMAGE_LINES",

        "NAC_LINE_EXPOSURE_DURATION",

        "SLEW_ANGLE"
    ]


    summary = {

        "product_path":
            str(
                product_path
            ),

        "file_size_bytes":
            int(
                file_size
            ),

        "image_pointer":
            parse_image_pointer(
                label
            ),

        "metadata":
            {}
    }


    for key in string_keys:

        value = get_value(
            label,
            key
        )

        if value is not None:

            summary[
                "metadata"
            ][
                key
            ] = value


    for key in numeric_keys:

        value = get_number(
            label,
            key
        )

        if value is not None:

            summary[
                "metadata"
            ][
                key
            ] = value


    # ========================================================
    # PRINT IMPORTANT VALUES
    # ========================================================

    important = [

        "PDS_VERSION_ID",
        "DATA_SET_ID",

        "PRODUCT_ID",
        "PRODUCT_VERSION_ID",

        "START_TIME",
        "STOP_TIME",

        "ORBIT_NUMBER",

        "INSTRUMENT_NAME",
        "INSTRUMENT_ID",

        "NAC_FRAME",

        "LINES",
        "LINE_SAMPLES",

        "SAMPLE_BITS",
        "SAMPLE_TYPE",

        "LINE_PREFIX_BYTES",
        "LINE_SUFFIX_BYTES",

        "RECORD_BYTES",
        "FILE_RECORDS",

        "NAC_LINE_EXPOSURE_DURATION",

        "SLEW_ANGLE"
    ]


    for key in important:

        value = summary[
            "metadata"
        ].get(
            key
        )

        if value is not None:

            print(
                f"{key:<32}: {value}"
            )


    print(
        f"{'^IMAGE':<32}: "
        f"{summary['image_pointer']}"
    )


    # ========================================================
    # ESTIMATE RAW IMAGE PAYLOAD
    # ========================================================

    lines = summary[
        "metadata"
    ].get(
        "LINES"
    )

    samples = summary[
        "metadata"
    ].get(
        "LINE_SAMPLES"
    )

    bits = summary[
        "metadata"
    ].get(
        "SAMPLE_BITS"
    )


    if (
        lines is not None
        and
        samples is not None
        and
        bits is not None
    ):

        bytes_per_sample = (
            float(bits)
            /
            8.0
        )


        estimated_image_bytes = (
            float(lines)
            *
            float(samples)
            *
            bytes_per_sample
        )


        summary[
            "estimated_uncompressed_image_bytes"
        ] = int(
            estimated_image_bytes
        )


        summary[
            "estimated_uncompressed_image_mib"
        ] = float(
            estimated_image_bytes
            /
            (
                1024
                **
                2
            )
        )


        print(
            "\nEstimated uncompressed image payload:"
        )

        print(
            f"{estimated_image_bytes:,.0f} bytes"
        )

        print(
            f"{estimated_image_bytes / (1024 ** 2):.2f} MiB"
        )


    # ========================================================
    # SEARCH GEOMETRY-LIKE LABEL VALUES
    # ========================================================

    geometry_keywords = [

        "CENTER_LATITUDE",
        "CENTER_LONGITUDE",

        "UPPER_LEFT_LATITUDE",
        "UPPER_LEFT_LONGITUDE",

        "UPPER_RIGHT_LATITUDE",
        "UPPER_RIGHT_LONGITUDE",

        "LOWER_LEFT_LATITUDE",
        "LOWER_LEFT_LONGITUDE",

        "LOWER_RIGHT_LATITUDE",
        "LOWER_RIGHT_LONGITUDE",

        "INCIDENCE_ANGLE",
        "EMISSION_ANGLE",
        "PHASE_ANGLE",

        "SOLAR_AZIMUTH_ANGLE",

        "SUB_SOLAR_LATITUDE",
        "SUB_SOLAR_LONGITUDE",

        "SPACECRAFT_ALTITUDE",

        "MAP_RESOLUTION",
        "MAP_SCALE"
    ]


    geometry = {}


    for key in geometry_keywords:

        value = get_value(
            label,
            key
        )

        if value is not None:

            geometry[
                key
            ] = value


    summary[
        "geometry_fields_found"
    ] = geometry


    print("\n")
    print("=" * 80)

    print(
        "GEOMETRY VALUES PRESENT IN EDR LABEL"
    )

    print("=" * 80)


    if geometry:

        for key, value in geometry.items():

            print(
                f"{key:<32}: {value}"
            )

    else:

        print(
            "No map/footprint geometry found in "
            "the raw EDR label."
        )

        print(
            "This is normal for a raw NAC EDR; "
            "SPICE camera geometry will be used."
        )


    # ========================================================
    # FILE SIGNATURE CHECK
    # ========================================================

    with product_path.open(
        "rb"
    ) as file:

        first_bytes = file.read(
            64
        )


    summary[
        "first_64_bytes_ascii"
    ] = first_bytes.decode(
        "ascii",
        errors="replace"
    )


    # ========================================================
    # SAVE
    # ========================================================

    SUMMARY_JSON.write_text(

        json.dumps(
            summary,
            indent=2
        ),

        encoding="utf-8"
    )


    print("\n")
    print("=" * 80)

    print(
        "LRO NAC INSPECTION COMPLETE"
    )

    print("=" * 80)


    print(
        "\nSaved label:"
    )

    print(
        LABEL_TXT
    )


    print(
        "\nSaved summary:"
    )

    print(
        SUMMARY_JSON
    )


if __name__ == "__main__":

    main()