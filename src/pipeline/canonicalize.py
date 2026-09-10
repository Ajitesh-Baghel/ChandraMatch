"""
Stage 0b: reproject/resample source + reference onto one common grid with
paired science/matcher masks, then load the result back into memory as plain
arrays for the rest of the pipeline.

This is a thin wrapper: all the actual reprojection/masking logic already
lives in src/canonicalization/build_canonical_pair.py (generic across every
sensor pair, validated against Pair006's ground truth). What's new here is
`load_canonical_pair`, generalized from app/pipeline_runner.py's
function of the same name so the CLI/regression harness and the web app load
canonical output identically -- previously that loader only existed inside
the app.
"""

from pathlib import Path

import numpy as np
import rasterio

from src.canonicalization.build_canonical_pair import build_canonical_pair
from src.pipeline.types import CanonicalPair


def read_band(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds.nodata, ds.profile.copy()


def load_canonical_pair(canonical_dir):
    """
    Reads back one canonicalize() output directory into arrays.

    `matcher_mask` is the feature-extraction mask (eroded, positive-support
    only -- what Stage 0/1's oriented-gradient descriptor uses).
    `science_mask` is the true observed-footprint mask (real zero-DN shadow
    stays valid) -- kept separate per this project's standing rule that a
    zero pixel is never silently treated as "not really observed."
    """

    canonical_dir = Path(canonical_dir)

    source, source_nodata, _ = read_band(canonical_dir / "source.tif")
    reference, _reference_nodata, reference_profile = read_band(canonical_dir / "reference.tif")
    matcher_mask_raw, _, _ = read_band(canonical_dir / "matcher_mask.tif")
    science_mask_raw, _, _ = read_band(canonical_dir / "science_mask.tif")

    if source.shape != reference.shape or source.shape != matcher_mask_raw.shape:
        raise RuntimeError(
            f"canonical source/reference/mask shapes differ: "
            f"{source.shape} / {reference.shape} / {matcher_mask_raw.shape}"
        )

    matcher_mask = matcher_mask_raw > 0
    matcher_mask &= np.isfinite(source)
    matcher_mask &= np.isfinite(reference)

    if source_nodata is not None:
        matcher_mask &= source != source_nodata

    matcher_mask &= reference > 0

    science_mask = science_mask_raw > 0

    return source, reference, matcher_mask, science_mask, source_nodata, reference_profile


def canonicalize(source_path, reference_path, out_dir, source_sensor, reference_sensor):
    """
    Stage 0b entry point used by run.py::register(). `source_sensor` /
    `reference_sensor` are the SensorInfo.label strings, passed through into
    the canonical metadata for traceability only -- canonicalization itself
    is sensor-agnostic (any two single-band georeferenced rasters).
    """

    out_dir = Path(out_dir)

    _paths, metadata = build_canonical_pair(
        source_path=source_path,
        reference_path=reference_path,
        output_dir=out_dir,
        pair_id=out_dir.name,
        source_sensor=source_sensor,
        reference_sensor=reference_sensor,
    )

    source, reference, matcher_mask, science_mask, source_nodata, reference_profile = (
        load_canonical_pair(out_dir)
    )

    return CanonicalPair(
        source=source,
        reference=reference,
        matcher_mask=matcher_mask,
        science_mask=science_mask,
        source_nodata=source_nodata,
        reference_profile=reference_profile,
        canonical_dir=out_dir,
        metadata=metadata,
    )
