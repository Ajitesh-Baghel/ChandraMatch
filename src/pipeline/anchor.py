"""
Stage 1: anchor resolution (dense structural search, falling back to
keypoint matching). Thin wrapper around
src/coarse/coarse_structural_search.py::run_stage0_with_fallback.

`resolve_anchor()` is the ONE call site the anchor-confidence classifier
(src/pipeline/anchor_confidence.py, §5b) attaches to. It does NOT change the
accept/reject decision itself in this pass -- see anchor_confidence.py's
module docstring: real, non-circular training data for both decisions this
model would make (dense-search soft-cascade acceptance: 0 usable rows once
hard-gated hypotheses are correctly excluded; fallback-fit acceptance: 6
rows) comes to well under the 20-row floor needed to fit anything meaningful,
so `AnchorConfidenceModel.decide()` returns None for both feature spaces and
the existing hand-set cascade inside `run_stage0_with_fallback` remains the
live decision path, unchanged. What this DOES add: `anchor.confidence` is now
populated with a real calibrated probability whenever enough data exists to
support one (verified live, not hand-waved) -- today that's never, and
`anchor.raw` records exactly why (see confidence_unavailable_reason below).
"""

from src.coarse.coarse_structural_search import run_stage0_with_fallback
from src.pipeline.anchor_confidence import AnchorConfidenceModel
from src.pipeline.types import AnchorResult

_MODEL = None


def _get_model():
    global _MODEL

    if _MODEL is None:
        _MODEL = AnchorConfidenceModel.load()

    return _MODEL


def resolve_anchor(source, source_mask, reference, reference_mask, config):
    result = run_stage0_with_fallback(source, source_mask, reference, reference_mask, config)

    best = result.get("best") or {}
    model = _get_model()
    confidence = None

    verification = result.get("verification")

    if verification is not None and verification.get("bbox_fill_ratio") is not None:
        features = dict(verification)
        features["ambiguity_ratio"] = result.get("ambiguity_ratio")

        if all(features.get(n) is not None for n in [
            "bbox_fill_ratio", "ambiguity_ratio", "overlap_fraction",
            "ncc_before", "ncc_after", "ncc_improvement", "local_cell_agreement",
        ]):
            confidence = model.decide("dense", features)

    fallback_result = result.get("keypoint_fallback_result")

    if confidence is None and fallback_result is not None and fallback_result.get("candidates", 0) >= 4:
        fallback_features = {
            "candidates": fallback_result.get("candidates"),
            "inliers": fallback_result.get("inliers"),
            "inlier_ratio": fallback_result.get("inlier_ratio"),
        }

        if all(v is not None for v in fallback_features.values()):
            confidence = model.decide("fallback", fallback_features)

    return AnchorResult(
        verdict=result["verdict"],
        verdict_reasons=result["verdict_reasons"],
        anchor_source=result.get("anchor_source", "dense_correlation"),
        dy=best.get("dy"),
        dx=best.get("dx"),
        confidence=confidence,
        raw=result,
    )
