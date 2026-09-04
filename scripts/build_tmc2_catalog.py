import sys
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.ingestion.tmc2_metadata import (
    inspect_tmc2_product,
    illumination_related_fields
)


# ============================================================
# PATHS
# ============================================================

TMC_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "tmc2"
)

METADATA_DIR = (
    ROOT
    / "metadata"
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CATALOG_PATH = (
    METADATA_DIR
    / "tmc2_products.csv"
)

SOLAR_DEBUG_PATH = (
    METADATA_DIR
    / "tmc2_illumination_fields.json"
)


# ============================================================
# SEARCH TIFF FILES
# ============================================================

tif_files = sorted(
    TMC_ROOT.rglob(
        "*.tif"
    )
)


if not tif_files:

    raise RuntimeError(
        f"No TMC-2 TIFF files found under {TMC_ROOT}"
    )


print("=" * 70)

print(
    "CHANDRAMATCH - TMC-2 PRODUCT CATALOGUE"
)

print("=" * 70)

print(
    "TMC-2 products found:",
    len(tif_files)
)


# ============================================================
# INSPECT PRODUCTS
# ============================================================

records = []

debug_metadata = {}


for index, tif_path in enumerate(
    tif_files,
    start=1
):

    print("\n")
    print(
        f"[{index}/{len(tif_files)}]"
    )

    print(
        tif_path
    )


    try:

        (
            record,
            xml_fields

        ) = inspect_tmc2_product(
            tif_path
        )


    except Exception as error:

        print(
            "FAILED:",
            error
        )

        continue


    records.append(
        record
    )


    relevant_fields = (
        illumination_related_fields(
            xml_fields
        )
    )


    debug_metadata[
        record["product_id"]
    ] = relevant_fields


    print(
        "Date:",
        record[
            "acquisition_date"
        ]
    )

    print(
        "Sun azimuth:",
        record[
            "sun_azimuth_deg"
        ],
        "tag:",
        record[
            "sun_azimuth_tag"
        ]
    )

    print(
        "Sun elevation:",
        record[
            "sun_elevation_deg"
        ],
        "tag:",
        record[
            "sun_elevation_tag"
        ]
    )

    print(
        "Incidence:",
        record[
            "incidence_angle_deg"
        ],
        "tag:",
        record[
            "incidence_angle_tag"
        ]
    )


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(
    records
)


df = df.sort_values(
    by="acquisition_date",
    na_position="last"
)


df.to_csv(
    CATALOG_PATH,
    index=False
)


# ============================================================
# SAVE DEBUG XML FIELDS
# ============================================================

with open(
    SOLAR_DEBUG_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        debug_metadata,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)

print(
    "TMC-2 CATALOGUE COMPLETE"
)

print("=" * 70)


columns = [

    "product_id",
    "acquisition_date",
    "sun_azimuth_deg",
    "sun_elevation_deg",
    "incidence_angle_deg",
    "left",
    "bottom",
    "right",
    "top"
]


print(
    df[
        columns
    ].to_string(
        index=False
    )
)


print(
    "\nSaved catalogue:"
)

print(
    CATALOG_PATH
)


print(
    "\nSaved illumination XML fields:"
)

print(
    SOLAR_DEBUG_PATH
)