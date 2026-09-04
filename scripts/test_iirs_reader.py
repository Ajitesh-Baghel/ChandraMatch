from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "data"
    / "raw"
    / "iirs"
    / "2024_01_15"
    / "data"
    / "derived"
    / "20240115"
)

PREFIX = "ch2_iir_ndi_20240115T2100076733"


RFL_HDR = (
    DATA_DIR
    / f"{PREFIX}_d_rfl_d18_srd.hdr"
)

RFL_QUB = (
    DATA_DIR
    / f"{PREFIX}_d_rfl_d18_srd.qub"
)


LOC_HDR = (
    DATA_DIR
    / f"{PREFIX}_d_loc_d18_ard.hdr"
)

LOC_IMG = (
    DATA_DIR
    / f"{PREFIX}_d_loc_d18_ard.img"
)


OBS_HDR = (
    DATA_DIR
    / f"{PREFIX}_d_obs_d18_ard.hdr"
)

OBS_IMG = (
    DATA_DIR
    / f"{PREFIX}_d_obs_d18_ard.img"
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "iirs"
    / "2024_01_15"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ENVI HEADER PARSER
# ============================================================

def parse_envi_header(path):

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    metadata = {}

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if (
            not line
            or
            line.upper() == "ENVI"
            or
            "=" not in line
        ):
            i += 1
            continue

        key, value = line.split(
            "=",
            1
        )

        key = key.strip().lower()

        value = value.strip()


        # Multi-line { ... } value
        if value.startswith("{"):

            pieces = [value]

            while (
                "}" not in pieces[-1]
                and
                i + 1 < len(lines)
            ):

                i += 1

                pieces.append(
                    lines[i].strip()
                )

            value = " ".join(
                pieces
            )


        if (
            value.startswith("{")
            and
            value.endswith("}")
        ):

            value = value[1:-1]


        metadata[key] = (
            value.strip()
        )

        i += 1


    return metadata


# ============================================================
# LIST PARSER
# ============================================================

def parse_comma_list(value):

    if value is None:
        return []

    return [

        item.strip()

        for item
        in value.split(",")

        if item.strip()
    ]


# ============================================================
# ENVI DTYPE
# ============================================================

def envi_dtype(
    metadata
):

    data_type = int(
        metadata["data type"]
    )

    byte_order = int(
        metadata.get(
            "byte order",
            "0"
        )
    )


    # ENVI datatype 4 = IEEE float32

    if data_type != 4:

        raise RuntimeError(
            f"Unsupported ENVI data type: "
            f"{data_type}"
        )


    if byte_order == 0:

        return np.dtype(
            "<f4"
        )

    elif byte_order == 1:

        return np.dtype(
            ">f4"
        )

    else:

        raise RuntimeError(
            f"Unsupported byte order: "
            f"{byte_order}"
        )


# ============================================================
# MEMMAP ENVI BSQ
# ============================================================

def open_envi_bsq(
    data_path,
    header_path
):

    metadata = parse_envi_header(
        header_path
    )


    samples = int(
        metadata["samples"]
    )

    lines = int(
        metadata["lines"]
    )

    bands = int(
        metadata["bands"]
    )


    interleave = (
        metadata["interleave"]
        .strip()
        .lower()
    )


    if interleave != "bsq":

        raise RuntimeError(
            "This reader currently expects "
            f"BSQ, got {interleave}"
        )


    dtype = envi_dtype(
        metadata
    )


    expected_bytes = (

        bands
        *
        lines
        *
        samples
        *
        dtype.itemsize
    )


    actual_bytes = (
        data_path.stat().st_size
    )


    if actual_bytes != expected_bytes:

        raise RuntimeError(

            "\nFile-size mismatch:\n"
            f"File: {data_path}\n"
            f"Expected: {expected_bytes}\n"
            f"Actual:   {actual_bytes}"
        )


    data = np.memmap(

        data_path,

        dtype=dtype,

        mode="r",

        shape=(
            bands,
            lines,
            samples
        ),

        order="C"
    )


    return (
        data,
        metadata
    )


# ============================================================
# IMAGE STRETCH
# ============================================================

def percentile_stretch(
    image,
    valid_mask,
    low_percentile=1,
    high_percentile=99
):

    output = np.zeros(
        image.shape,
        dtype=np.uint8
    )


    values = image[
        valid_mask
    ]


    values = values[
        np.isfinite(
            values
        )
    ]


    if values.size == 0:

        return output, None, None


    low, high = np.percentile(

        values,

        [
            low_percentile,
            high_percentile
        ]
    )


    if high <= low:

        return (
            output,
            float(low),
            float(high)
        )


    normalized = (

        image
        -
        low

    ) / (

        high
        -
        low
    )


    normalized = np.clip(
        normalized,
        0,
        1
    )


    output[
        valid_mask
    ] = (

        normalized[
            valid_mask
        ]

        *
        255.0

    ).astype(
        np.uint8
    )


    return (
        output,
        float(low),
        float(high)
    )


# ============================================================
# SAFE STATISTICS
# ============================================================

def safe_stats(
    array,
    mask
):

    values = array[
        mask
    ]


    values = values[
        np.isfinite(
            values
        )
    ]


    if values.size == 0:

        return None


    p = np.percentile(

        values,

        [
            0,
            1,
            50,
            99,
            100
        ]
    )


    return {

        "minimum":
            float(p[0]),

        "p01":
            float(p[1]),

        "median":
            float(p[2]),

        "p99":
            float(p[3]),

        "maximum":
            float(p[4]),

        "mean":
            float(
                np.mean(
                    values
                )
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "CHANDRAMATCH - IIRS NATIVE PRODUCT INSPECTION"
    )

    print("=" * 80)


    # ========================================================
    # CHECK FILES
    # ========================================================

    paths = [

        RFL_HDR,
        RFL_QUB,

        LOC_HDR,
        LOC_IMG,

        OBS_HDR,
        OBS_IMG
    ]


    for path in paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing:\n{path}"
            )


    # ========================================================
    # OPEN PRODUCTS
    # ========================================================

    rfl, rfl_meta = open_envi_bsq(

        RFL_QUB,
        RFL_HDR
    )


    loc, loc_meta = open_envi_bsq(

        LOC_IMG,
        LOC_HDR
    )


    obs, obs_meta = open_envi_bsq(

        OBS_IMG,
        OBS_HDR
    )


    print("\nReflectance cube:")

    print(
        "Shape:",
        rfl.shape
    )

    print(
        "Dtype:",
        rfl.dtype
    )

    print(
        "Memory mapped:",
        isinstance(
            rfl,
            np.memmap
        )
    )

    print(
        "File size:",
        RFL_QUB.stat().st_size
    )


    print("\nLocation cube:")

    print(
        "Shape:",
        loc.shape
    )

    print(
        "Dtype:",
        loc.dtype
    )

    print(
        "File size:",
        LOC_IMG.stat().st_size
    )


    print("\nObservation cube:")

    print(
        "Shape:",
        obs.shape
    )

    print(
        "Dtype:",
        obs.dtype
    )

    print(
        "File size:",
        OBS_IMG.stat().st_size
    )


    # ========================================================
    # WAVELENGTHS
    # ========================================================

    wavelengths = np.array(

        [
            float(value)

            for value in parse_comma_list(
                rfl_meta.get(
                    "wavelength"
                )
            )
        ],

        dtype=np.float64
    )


    if wavelengths.size != rfl.shape[0]:

        raise RuntimeError(

            "Wavelength count does not "
            "match number of bands."
        )


    print(
        "\nWavelength range:"
    )

    print(
        float(
            wavelengths.min()
        ),
        "nm ->",
        float(
            wavelengths.max()
        ),
        "nm"
    )


    # ========================================================
    # LOCATION DATA
    # ========================================================

    longitude = np.asarray(
        loc[0],
        dtype=np.float32
    )

    latitude = np.asarray(
        loc[1],
        dtype=np.float32
    )

    radius = np.asarray(
        loc[2],
        dtype=np.float32
    )

    height = np.asarray(
        loc[3],
        dtype=np.float32
    )


    geometry_valid = (

        np.isfinite(
            longitude
        )

        &

        np.isfinite(
            latitude
        )

        &

        np.isfinite(
            radius
        )

        &

        (
            latitude
            >=
            -90.0
        )

        &

        (
            latitude
            <=
            90.0
        )

        &

        (
            np.abs(
                longitude
            )
            <=
            360.0
        )

        &

        (
            radius
            >
            1_000_000.0
        )
    )


    geometry_valid_count = int(
        np.count_nonzero(
            geometry_valid
        )
    )


    geometry_valid_ratio = (

        geometry_valid_count
        /
        geometry_valid.size
    )


    print("\n")
    print("=" * 80)
    print("PER-PIXEL GEOLOCATION")
    print("=" * 80)


    print(
        "Spatial pixels:",
        geometry_valid.size
    )

    print(
        "Valid geolocation pixels:",
        geometry_valid_count
    )

    print(
        "Valid geolocation ratio:",
        geometry_valid_ratio
    )


    if geometry_valid_count == 0:

        raise RuntimeError(
            "No valid IIRS geolocation found."
        )


    longitude_180 = (

        (
            longitude
            +
            180.0
        )

        %
        360.0

    ) - 180.0


    print(
        "\nLatitude range:"
    )

    print(

        float(
            np.min(
                latitude[
                    geometry_valid
                ]
            )
        ),

        "->",

        float(
            np.max(
                latitude[
                    geometry_valid
                ]
            )
        )
    )


    print(
        "\nLongitude range "
        "(-180..180):"
    )

    print(

        float(
            np.min(
                longitude_180[
                    geometry_valid
                ]
            )
        ),

        "->",

        float(
            np.max(
                longitude_180[
                    geometry_valid
                ]
            )
        )
    )


    print(
        "\nRadius statistics:"
    )

    print(
        safe_stats(
            radius,
            geometry_valid
        )
    )


    print(
        "\nHeight statistics:"
    )

    print(
        safe_stats(
            height,
            geometry_valid
        )
    )


    # ========================================================
    # BUILD TMC-COMPARABLE IIRS REPRESENTATION
    #
    # Average IIRS spectral bands between
    # roughly 700 and 850 nm.
    # ========================================================

    selected_indices = np.where(

        (
            wavelengths
            >=
            700.0
        )

        &

        (
            wavelengths
            <=
            850.0
        )

    )[0]


    print("\n")
    print("=" * 80)
    print("OPTICAL IIRS REPRESENTATION")
    print("=" * 80)


    print(
        "Selected band indices (1-based):",
        (
            selected_indices
            +
            1
        ).tolist()
    )


    print(
        "Selected wavelengths:"
    )

    print(
        wavelengths[
            selected_indices
        ].tolist()
    )


    lines = rfl.shape[1]

    samples = rfl.shape[2]


    spectral_sum = np.zeros(

        (
            lines,
            samples
        ),

        dtype=np.float32
    )


    spectral_count = np.zeros(

        (
            lines,
            samples
        ),

        dtype=np.uint8
    )


    for band_index in selected_indices:

        print(

            "Reading band",
            int(
                band_index
                +
                1
            ),
            "-",
            float(
                wavelengths[
                    band_index
                ]
            ),
            "nm"
        )


        band = np.asarray(
            rfl[
                band_index
            ],
            dtype=np.float32
        )


        good = (

            geometry_valid

            &

            np.isfinite(
                band
            )

            &

            (
                np.abs(
                    band
                )
                <
                1_000_000.0
            )
        )


        spectral_sum[
            good
        ] += band[
            good
        ]


        spectral_count[
            good
        ] += 1


    optical_valid = (

        geometry_valid

        &

        (
            spectral_count
            >
            0
        )
    )


    optical_image = np.full(

        (
            lines,
            samples
        ),

        np.nan,

        dtype=np.float32
    )


    optical_image[
        optical_valid
    ] = (

        spectral_sum[
            optical_valid
        ]

        /

        spectral_count[
            optical_valid
        ]
    )


    print(
        "\nOptical valid pixels:",
        int(
            np.count_nonzero(
                optical_valid
            )
        )
    )


    print(
        "Optical valid ratio:",
        float(
            optical_valid.mean()
        )
    )


    optical_stats = safe_stats(

        optical_image,
        optical_valid
    )


    print(
        "\nOptical reflectance statistics:"
    )

    print(
        optical_stats
    )


    # ========================================================
    # QUICKLOOK
    # ========================================================

    quicklook, stretch_low, stretch_high = (
        percentile_stretch(

            optical_image,

            optical_valid,

            low_percentile=1,

            high_percentile=99
        )
    )


    full_quicklook_path = (

        OUTPUT_DIR
        /
        "iirs_optical_700_850nm_full.png"
    )


    cv2.imwrite(

        str(
            full_quicklook_path
        ),

        quicklook
    )


    # ========================================================
    # STRETCHED OVERVIEW
    #
    # IIRS is an extremely long narrow strip,
    # so horizontally stretch it for visual inspection.
    # This image is VISUALIZATION ONLY.
    # ========================================================

    overview_height = 3000

    overview_width = 700


    overview = cv2.resize(

        quicklook,

        (
            overview_width,
            overview_height
        ),

        interpolation=
            cv2.INTER_AREA
    )


    overview_path = (

        OUTPUT_DIR
        /
        "iirs_optical_700_850nm_stretched.png"
    )


    cv2.imwrite(

        str(
            overview_path
        ),

        overview
    )


    # ========================================================
    # CENTER STRIP
    # ========================================================

    crop_height = min(
        2048,
        lines
    )


    center_line = (
        lines
        //
        2
    )


    crop_start = max(

        0,

        center_line
        -
        crop_height // 2
    )


    crop_end = min(

        lines,

        crop_start
        +
        crop_height
    )


    center_crop = quicklook[
        crop_start:crop_end,
        :
    ]


    center_crop_stretched = cv2.resize(

        center_crop,

        (
            1000,
            center_crop.shape[0]
        ),

        interpolation=
            cv2.INTER_CUBIC
    )


    center_crop_path = (

        OUTPUT_DIR
        /
        "iirs_optical_center_2048_stretched.png"
    )


    cv2.imwrite(

        str(
            center_crop_path
        ),

        center_crop_stretched
    )


    # ========================================================
    # LOCATION MASK
    # ========================================================

    location_mask = (

        geometry_valid.astype(
            np.uint8
        )

        *
        255
    )


    location_mask_path = (

        OUTPUT_DIR
        /
        "iirs_geolocation_valid_mask.png"
    )


    cv2.imwrite(

        str(
            location_mask_path
        ),

        location_mask
    )


    # ========================================================
    # OBSERVATION GEOMETRY
    # ========================================================

    observation_band_names = (

        parse_comma_list(
            obs_meta.get(
                "band names"
            )
        )
    )


    observation_summary = {}


    print("\n")
    print("=" * 80)
    print("OBSERVATION GEOMETRY")
    print("=" * 80)


    for band_index in range(
        min(
            5,
            obs.shape[0]
        )
    ):

        values = np.asarray(

            obs[
                band_index
            ],

            dtype=np.float32
        )


        valid = (

            geometry_valid

            &

            np.isfinite(
                values
            )

            &

            (
                np.abs(
                    values
                )
                <
                1_000_000.0
            )
        )


        stats = safe_stats(
            values,
            valid
        )


        if (
            band_index
            <
            len(
                observation_band_names
            )
        ):

            name = (
                observation_band_names[
                    band_index
                ]
            )

        else:

            name = (
                f"band_{band_index + 1}"
            )


        observation_summary[
            name
        ] = stats


        print(
            f"{name}:"
        )

        print(
            stats
        )


    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {

        "product":
            PREFIX,

        "reflectance_cube": {

            "shape": [
                int(v)
                for v in rfl.shape
            ],

            "dtype":
                str(
                    rfl.dtype
                ),

            "bytes":
                int(
                    RFL_QUB.stat().st_size
                ),

            "memory_mapped":
                bool(
                    isinstance(
                        rfl,
                        np.memmap
                    )
                )
        },

        "location_cube": {

            "shape": [
                int(v)
                for v in loc.shape
            ],

            "band_names":
                parse_comma_list(
                    loc_meta.get(
                        "band names"
                    )
                )
        },

        "observation_cube": {

            "shape": [
                int(v)
                for v in obs.shape
            ],

            "band_names":
                observation_band_names
        },

        "geolocation": {

            "valid_pixels":
                geometry_valid_count,

            "valid_ratio":
                float(
                    geometry_valid_ratio
                ),

            "latitude_min":
                float(
                    np.min(
                        latitude[
                            geometry_valid
                        ]
                    )
                ),

            "latitude_max":
                float(
                    np.max(
                        latitude[
                            geometry_valid
                        ]
                    )
                ),

            "longitude_180_min":
                float(
                    np.min(
                        longitude_180[
                            geometry_valid
                        ]
                    )
                ),

            "longitude_180_max":
                float(
                    np.max(
                        longitude_180[
                            geometry_valid
                        ]
                    )
                )
        },

        "optical_representation": {

            "wavelength_min_nm":
                700.0,

            "wavelength_max_nm":
                850.0,

            "bands_1_based":
                (
                    selected_indices
                    +
                    1
                ).tolist(),

            "wavelengths_nm":
                wavelengths[
                    selected_indices
                ].tolist(),

            "valid_pixels":
                int(
                    np.count_nonzero(
                        optical_valid
                    )
                ),

            "valid_ratio":
                float(
                    optical_valid.mean()
                ),

            "stretch_p01":
                stretch_low,

            "stretch_p99":
                stretch_high,

            "statistics":
                optical_stats
        },

        "observation_geometry":
            observation_summary
    }


    summary_path = (

        OUTPUT_DIR
        /
        "iirs_native_summary.json"
    )


    summary_path.write_text(

        json.dumps(
            summary,
            indent=2
        ),

        encoding="utf-8"
    )


    # ========================================================
    # DONE
    # ========================================================

    print("\n")
    print("=" * 80)

    print(
        "IIRS NATIVE INSPECTION COMPLETE"
    )

    print("=" * 80)


    print(
        "\nSummary:"
    )

    print(
        summary_path
    )


    print(
        "\nFull optical quicklook:"
    )

    print(
        full_quicklook_path
    )


    print(
        "\nStretched overview:"
    )

    print(
        overview_path
    )


    print(
        "\nCenter strip:"
    )

    print(
        center_crop_path
    )


    print(
        "\nGeolocation mask:"
    )

    print(
        location_mask_path
    )


if __name__ == "__main__":

    main()