"""
Stage -1: sensor auto-detection for an arbitrary uploaded file.

Generalizes app/sensor_detection.py (kept as a thin re-export there for
backward compatibility -- see the bottom of this file) with the label/header
inspection step §4 of the unification pass asks for: a PDS3 file (LRO NAC's
raw .IMG) carries a plain-ASCII label at byte 0 we can regex directly; a PDS4
product (OHRC's raw .img) instead has a detached same-stem .xml label; IIRS's
ARD .img/.qub products instead ship an ENVI .hdr sidecar with no reliable
instrument tag, so for those (and any other case) we fall through to the
filename convention, then to a resolution-range heuristic, exactly as before.

Never silently forces an unrecognized file into a guessed sensor -- "unknown"
is a legitimate result the caller must surface and let the user confirm
manually, per this project's own never-force-a-match discipline.
"""

import re
from pathlib import Path

import rasterio

# Filename patterns matching the real product-ID conventions seen across this
# project's actual data (data/raw/**, metadata.json files under
# data/processed/*).
FILENAME_PATTERNS = [
    ("Chandrayaan-2 TMC-2", re.compile(r"ch2_tmc", re.I)),
    ("Chandrayaan-2 OHRC", re.compile(r"ch2_ohr", re.I)),
    ("Chandrayaan-2 IIRS", re.compile(r"ch2_iir", re.I)),
    ("LRO NAC", re.compile(r"^m\d{9}[lr][ec]", re.I)),
    ("SELENE/Kaguya TC", re.compile(r"dtmtco|_tco_|kaguya", re.I)),
]

# Native resolution ranges observed in this project's real canonical data
# (pair metadata across pairs 001-007) -- a weaker signal, used only when
# neither a label nor the filename matched anything, since resolution alone
# is ambiguous (a resampled/cropped file may not carry its native scale).
RESOLUTION_HINTS_M = [
    ("Chandrayaan-2 OHRC", 0.15, 0.45),
    ("LRO NAC", 0.4, 1.6),
    ("Chandrayaan-2 TMC-2", 4.0, 6.0),
    ("Chandrayaan-2 IIRS", 50.0, 80.0),
    ("SELENE/Kaguya TC", 8.0, 12.0),
]

# PDS3 label INSTRUMENT_ID / DATA_SET_ID values -> our sensor label. LRO NAC's
# raw .IMG confirmed (by direct inspection, not assumed) to carry a plain
# ASCII PDS3 label starting at byte 0 with these exact keys.
PDS3_INSTRUMENT_MAP = [
    (re.compile(r"INSTRUMENT_ID\s*=\s*LROC", re.I), "LRO NAC"),
    (re.compile(r"INSTRUMENT_NAME\s*=\s*\"LUNAR RECONNAISSANCE ORBITER CAMERA\"", re.I), "LRO NAC"),
]

# PDS4 XML label tags -> our sensor label, for a same-stem sidecar .xml
# (OHRC's raw .img ships exactly this pattern -- confirmed by direct
# inspection). Matched loosely against the raw XML text rather than parsed,
# since the exact namespace/schema varies across product types (the same
# reason src/ingestion/ohrc_product.py does fuzzy tag matching rather than a
# strict XML schema parse).
PDS4_XML_PATTERNS = [
    ("Chandrayaan-2 OHRC", re.compile(r"ohrc|orbiter high resolution camera", re.I)),
    ("Chandrayaan-2 TMC-2", re.compile(r"\btmc\b|terrain mapping camera", re.I)),
    ("Chandrayaan-2 IIRS", re.compile(r"\biirs\b|imaging.{0,10}infra.?red spectrometer", re.I)),
]


def _read_pds3_label(path, max_bytes=4096):
    try:
        with open(path, "rb") as f:
            head = f.read(max_bytes)
    except OSError:
        return None

    try:
        text = head.decode("ascii", errors="ignore")
    except Exception:
        return None

    if "PDS_VERSION_ID" not in text:
        return None

    return text


def detect_from_pds3_label(path):
    """
    LRO NAC's raw .IMG carries its PDS3 label as plain ASCII at the very
    start of the file -- no sidecar file needed. Returns None for anything
    else (including PDS4 products, which use a detached XML label instead).
    """

    if Path(path).suffix.lower() != ".img":
        return None

    text = _read_pds3_label(path)

    if text is None:
        return None

    for pattern, label in PDS3_INSTRUMENT_MAP:
        if pattern.search(text):
            return label

    return None


def detect_from_pds4_sidecar(path):
    """
    A same-stem .xml next to a raw .img/.qub is this project's PDS4
    convention (confirmed on OHRC's real 2021-12-28 product). IIRS's ARD
    products instead ship a .hdr (ENVI header, no instrument tag), so this
    intentionally returns None for those rather than guessing from a header
    format that doesn't carry the information.
    """

    path = Path(path)
    sidecar = path.with_suffix(".xml")

    if not sidecar.exists():
        return None

    try:
        text = sidecar.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    for label, pattern in PDS4_XML_PATTERNS:
        if pattern.search(text):
            return label

    return None


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


def native_format_of(path):
    return "raw_pds3_img" if Path(path).suffix.lower() == ".img" else "geotiff"


def detect_sensor(path, filename=None):
    """
    Returns a plain dict (not the SensorInfo dataclass -- kept dict-shaped so
    this remains a drop-in replacement for app/sensor_detection.py's existing
    callers, which already consume this exact shape):
    {"label": str, "confidence": "high"|"medium"|"low"|"unknown",
     "method": str, "native_format": str, "native_resolution_m": float|None}

    Detection order, per the project's own convention (strongest evidence
    first): PDS3 embedded label -> PDS4 sidecar label -> filename pattern ->
    resolution heuristic -> unknown.
    """

    path = Path(path)
    filename = filename or path.name
    native_format = native_format_of(path)

    label = detect_from_pds3_label(path)

    if label is not None:
        _, resolution_m = detect_from_resolution(path)
        return {
            "label": label,
            "confidence": "high",
            "method": "pds3_label",
            "native_format": native_format,
            "native_resolution_m": resolution_m,
        }

    label = detect_from_pds4_sidecar(path)

    if label is not None:
        _, resolution_m = detect_from_resolution(path)
        return {
            "label": label,
            "confidence": "high",
            "method": "pds4_label",
            "native_format": native_format,
            "native_resolution_m": resolution_m,
        }

    label = detect_from_filename(filename)

    if label is not None:
        _, resolution_m = detect_from_resolution(path)
        return {
            "label": label,
            "confidence": "high",
            "method": "filename_pattern",
            "native_format": native_format,
            "native_resolution_m": resolution_m,
        }

    resolution_label, resolution_m = detect_from_resolution(path)

    if resolution_label is not None:
        return {
            "label": resolution_label,
            "confidence": "low",
            "method": "resolution_hint",
            "native_format": native_format,
            "native_resolution_m": resolution_m,
        }

    return {
        "label": "unknown",
        "confidence": "unknown",
        "method": "none",
        "native_format": native_format,
        "native_resolution_m": resolution_m,
    }
