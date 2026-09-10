"""
Shared schema for the unified ChandraMatch pipeline (src/pipeline/run.py::register()).

Design choice, stated explicitly: these dataclasses define the CONTRACT at the
boundary between pipeline stages and between register() and its callers (CLI,
web app, regression scripts). They deliberately do NOT replace the existing,
already-validated dict-shaped return values inside src/coarse/*.py (Stage 0's
hypothesis list, Stage 1's per-point dicts, Stage 2's perturbation-control
dicts, etc.) -- those are embedded verbatim as fields below. Rewriting every
internal function to return a dataclass would touch dozens of already-proven
call sites for no behavioral benefit, and would make the regression diff this
refactor depends on (old script output vs new register() output) much harder
to trust, since it would no longer be comparing like-for-like internals.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SensorInfo:
    """Result of sensor_id.py's detection for ONE uploaded file."""

    label: str  # e.g. "Chandrayaan-2 TMC-2", "LRO NAC", or "unknown"
    confidence: str  # "high" | "medium" | "low" | "unknown"
    method: str  # "pds3_label" | "pds4_label" | "filename_pattern" | "resolution_hint" | "none"
    native_format: str  # "geotiff" | "raw_pds3_img" | "unknown"
    native_resolution_m: float | None = None
    product_id: str | None = None


@dataclass
class RasterPair:
    """Paths + sensor identity for one source/reference pair, pre-canonicalization."""

    source_path: Path
    reference_path: Path
    source_sensor: SensorInfo
    reference_sensor: SensorInfo


@dataclass
class CanonicalPair:
    """Output of canonicalize.py -- arrays already loaded into memory."""

    source: Any  # np.ndarray, float32
    reference: Any  # np.ndarray
    matcher_mask: Any  # np.ndarray[bool]
    science_mask: Any  # np.ndarray[bool]
    source_nodata: float | None
    reference_profile: dict
    canonical_dir: Path
    metadata: dict  # build_canonical_pair's own metadata dict, unchanged


@dataclass
class AnchorResult:
    """
    Output of anchor.py::resolve_anchor(). Wraps whichever of Stage 0's two
    existing result shapes (dense or keypoint-fallback) actually produced the
    anchor -- `raw` keeps that original dict intact (every field downstream
    code / the console log already relies on), the fields below are the
    subset every caller actually branches on.
    """

    verdict: str  # "PASS" | "AMBIGUOUS" | "FAIL"
    verdict_reasons: list
    anchor_source: str  # "dense_correlation" | "keypoint_fallback_sift_intensity" | ...
    dy: int | None
    dx: int | None
    confidence: float | None  # anchor-confidence model probability, when the model ran
    raw: dict  # full run_stage0_with_fallback() result, unmodified


@dataclass
class ControlPoints:
    """Output of refine.py::run_stage1_generic(). Wraps Stage 1's existing point-list shape."""

    points: list
    accepted_points: list
    summary: dict


@dataclass
class RegistrationResult:
    """
    THE public return type of register(). Every field a caller (CLI, app,
    regression harness) needs is here directly; `metrics` carries the full
    §6-compliant bundle that also gets written to metrics.json.
    """

    verdict: str
    verdict_reasons: list
    anchor_source: str
    sensor_source: str
    sensor_reference: str
    translation_px: tuple
    rotation_deg: float | None
    scale: tuple | None
    inlier_count: int | None
    inlier_ratio: float | None
    rmse_px: float | None
    sub_pixel_achieved: bool
    match_points: list  # every Stage 1 point, dense or fallback-derived
    perturbation_summary: dict | None
    metrics: dict  # the full metrics.json content
    out_dir: Path
    files: dict = field(default_factory=dict)  # name -> Path, for every §6 artifact written
