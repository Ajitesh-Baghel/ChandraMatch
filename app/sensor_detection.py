import re

import rasterio

# Filename patterns matching the real product-ID conventions already seen
# across this project's actual data (metadata.json files under
# data/processed/*, docs referenced throughout the R&D history).
FILENAME_PATTERNS = [
    ("Chandrayaan-2 TMC-2", re.compile(r"ch2_tmc", re.I)),
    ("Chandrayaan-2 OHRC", re.compile(r"ch2_ohr", re.I)),
    ("Chandrayaan-2 IIRS", re.compile(r"ch2_iir", re.I)),
    ("LRO NAC", re.compile(r"^m\d{9}[lr][ec]", re.I)),
    ("SELENE/Kaguya TC", re.compile(r"dtmtco|_tco_|kaguya", re.I)),
]

# Native resolution ranges observed in this project's real canonical data
# (pair metadata across pairs 001-007) -- a secondary, weaker signal used
# only when the filename doesn't match anything, since resolution alone
# is ambiguous (a resampled/cropped file may not carry its native scale).
RESOLUTION_HINTS_M = [
    ("Chandrayaan-2 OHRC", 0.15, 0.45),
    ("LRO NAC", 0.4, 1.6),
    ("Chandrayaan-2 TMC-2", 4.0, 6.0),
    ("Chandrayaan-2 IIRS", 50.0, 80.0),
    ("SELENE/Kaguya TC", 8.0, 12.0),
]


def detect_from_filename(filename):
    for label, pattern in FILENAME_PATTERNS:
        if pattern.search(filename):
            return label

    return None


def detect_from_resolution(path):
    try:
        with rasterio.open(path) as ds:
            resolution_m = float(abs(ds.transform.a))
    except Exception:
        return None, None

    for label, low, high in RESOLUTION_HINTS_M:
        if low <= resolution_m <= high:
            return label, resolution_m

    return None, resolution_m


def detect_sensor(path, filename):
    """
    Returns {"label": str, "confidence": "high"|"low"|"unknown",
    "method": str, "native_resolution_m": float|None}.

    Never silently forces an unrecognized file into a guessed sensor --
    "unknown" is a legitimate result the caller (the UI) must surface
    and let the user confirm manually, per the project's own
    never-force-a-match discipline applied here to ingestion instead of
    registration.
    """

    label = detect_from_filename(filename)

    if label is not None:
        _, resolution_m = detect_from_resolution(path)

        return {
            "label": label,
            "confidence": "high",
            "method": "filename_pattern",
            "native_resolution_m": resolution_m,
        }

    resolution_label, resolution_m = detect_from_resolution(path)

    if resolution_label is not None:
        return {
            "label": resolution_label,
            "confidence": "low",
            "method": "resolution_hint",
            "native_resolution_m": resolution_m,
        }

    return {
        "label": "unknown",
        "confidence": "unknown",
        "method": "none",
        "native_resolution_m": resolution_m,
    }
