"""
THE single public entry point for the unified ChandraMatch pipeline.

register() runs sensor ID -> ingest -> canonicalize -> anchor resolution ->
local refinement -> fit + validate -> export, in that fixed order, for ANY
two lunar rasters. It never branches on which pair this is and never reads a
pair name/ID from the caller -- the only inputs are file paths and config.
Every other code path (the CLI regression wrappers under scripts/, and the
FastAPI app) calls this and nothing else; no stage-skipping branch is visible
above this function.
"""

from pathlib import Path

from src.pipeline import ingest, sensor_id
from src.pipeline.anchor import resolve_anchor
from src.pipeline.canonicalize import canonicalize
from src.pipeline.export import export_bundle
from src.pipeline.fit_validate import fit_and_validate
from src.pipeline.refine import run_refinement
from src.pipeline.types import RegistrationResult


def _empty_fit_result():
    return {
        "fit": None,
        "final_points": [],
        "outlier_count": 0,
        "outlier_rate": 1.0,
        "validation": None,
        "reduced_scale_validation": None,
        "residuals": None,
    }


def _early_exit(out_dir, verdict, reasons, sensor_source, sensor_reference, log, anchor=None):
    """
    Produces a full (if mostly empty) §6 bundle even when the pipeline stops
    before Stage 2 -- every run gets a metrics.json/console.log explaining
    the honest verdict, never a bare exception or a silently missing file.
    """

    from src.pipeline.types import AnchorResult

    if anchor is None:
        anchor = AnchorResult(
            verdict=verdict,
            verdict_reasons=reasons,
            anchor_source="none",
            dy=None,
            dx=None,
            confidence=None,
            raw={},
        )

    metrics, files = export_bundle(
        out_dir=out_dir,
        source=None,
        reference=None,
        mask=None,
        source_nodata=None,
        reference_profile=None,
        all_points=[],
        fit_result=_empty_fit_result(),
        anchor=anchor,
        verdict=verdict,
        verdict_reasons=reasons,
        perturbation_validation_informational_only=False,
        sensor_source=sensor_source,
        sensor_reference=sensor_reference,
        canonical_metadata=None,
        console_lines=log,
    )

    return RegistrationResult(
        verdict=verdict,
        verdict_reasons=reasons,
        anchor_source=anchor.anchor_source,
        sensor_source=sensor_source,
        sensor_reference=sensor_reference,
        translation_px=(None, None),
        rotation_deg=None,
        scale=None,
        inlier_count=None,
        inlier_ratio=None,
        rmse_px=None,
        sub_pixel_achieved=False,
        match_points=[],
        perturbation_summary=metrics["perturbation_summary"],
        metrics=metrics,
        out_dir=Path(out_dir),
        files=files,
    )


def register(source_path, reference_path, config, out_dir):
    """
    Runs the full pipeline for one arbitrary source/reference lunar raster
    pair. `config` is a PipelineConfig (src/pipeline/config.py). Returns a
    RegistrationResult and writes the full §6 output bundle into `out_dir`
    regardless of where the pipeline stops.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = [f"register(source={source_path}, reference={reference_path})"]

    # ------------------------------------------------------------
    # Stage -1: sensor identification
    # ------------------------------------------------------------

    if config.source_sensor_override:
        source_info = {
            "label": config.source_sensor_override,
            "confidence": "override",
            "method": "caller_override",
            "native_format": sensor_id.native_format_of(source_path),
            "native_resolution_m": None,
        }
    else:
        source_info = sensor_id.detect_sensor(source_path)

    if config.reference_sensor_override:
        reference_info = {
            "label": config.reference_sensor_override,
            "confidence": "override",
            "method": "caller_override",
            "native_format": sensor_id.native_format_of(reference_path),
            "native_resolution_m": None,
        }
    else:
        reference_info = sensor_id.detect_sensor(reference_path)

    log.append(f"source sensor: {source_info}")
    log.append(f"reference sensor: {reference_info}")

    if source_info["label"] == "unknown" or reference_info["label"] == "unknown":
        reasons = []

        if source_info["label"] == "unknown":
            reasons.append(
                "source sensor could not be identified with confidence -- confirm "
                "manually (pass source_sensor_override) and retry"
            )

        if reference_info["label"] == "unknown":
            reasons.append(
                "reference sensor could not be identified with confidence -- confirm "
                "manually (pass reference_sensor_override) and retry"
            )

        return _early_exit(
            out_dir, "AMBIGUOUS", reasons, source_info["label"], reference_info["label"], log
        )

    # ------------------------------------------------------------
    # Raw-format gate: an honest FAIL for any raw .img sensor without an
    # automated ingestion path, rather than a silent wrong guess.
    # ------------------------------------------------------------

    for role, path, info in (
        ("source", source_path, source_info),
        ("reference", reference_path, reference_info),
    ):
        if ingest.is_unsupported_raw_format(path, info["label"]):
            return _early_exit(
                out_dir,
                "FAIL",
                [
                    f"{role} is a raw .img file for sensor '{info['label']}', which doesn't "
                    "have automated ingestion built yet (only LRO NAC does) -- upload an "
                    "already-processed .tif for this sensor instead"
                ],
                source_info["label"],
                reference_info["label"],
                log,
            )

    # ------------------------------------------------------------
    # Stage 0a: ingest (ISIS for a supported raw .img, pass-through otherwise)
    # ------------------------------------------------------------

    ingest_dir = out_dir / "ingest"

    resolved_source_path = ingest.ingest(source_path, source_info["label"], ingest_dir / "source")
    resolved_reference_path = ingest.ingest(
        reference_path, reference_info["label"], ingest_dir / "reference"
    )

    log.append(f"resolved source path: {resolved_source_path}")
    log.append(f"resolved reference path: {resolved_reference_path}")

    # ------------------------------------------------------------
    # Stage 0b: canonicalize
    # ------------------------------------------------------------

    canonical_dir = out_dir / "canonical"

    canonical = canonicalize(
        resolved_source_path,
        resolved_reference_path,
        canonical_dir,
        source_info["label"],
        reference_info["label"],
    )

    log.append(f"canonical shape: {canonical.source.shape}")
    log.append(f"canonical metadata: {canonical.metadata}")

    # ------------------------------------------------------------
    # Stage 1: anchor resolution (dense search + fallback chain)
    # ------------------------------------------------------------

    anchor = resolve_anchor(
        canonical.source,
        canonical.matcher_mask,
        canonical.reference,
        canonical.matcher_mask,
        config.stage0,
    )

    log.append(f"anchor verdict: {anchor.verdict} ({anchor.anchor_source})")
    log.append(f"anchor reasons: {anchor.verdict_reasons}")

    # Persisted unconditionally (regardless of verdict) so the raw Stage 0
    # verification metrics -- bbox_fill_ratio, ambiguity_ratio, NCC, local-
    # cell-agreement, plus every fallback attempt's own candidates/inliers --
    # are always available as an inspectable artifact, not just a summary
    # line in console.log. This is what src/pipeline/anchor_confidence.py's
    # training-table assembly reads from for future re-fitting.
    import json as _json

    with open(out_dir / "stage0_raw.json", "w", encoding="utf-8") as f:
        _json.dump(anchor.raw, f, indent=2, default=str)

    if anchor.verdict != "PASS":
        return _early_exit(
            out_dir,
            anchor.verdict,
            anchor.verdict_reasons,
            source_info["label"],
            reference_info["label"],
            log,
            anchor=anchor,
        )

    # ------------------------------------------------------------
    # Stage 2: local refinement
    # ------------------------------------------------------------

    control_points, source_stack, source_corr_mask, reference_stack, reference_corr_mask = (
        run_refinement(
            canonical.source,
            canonical.matcher_mask,
            canonical.reference,
            anchor,
            config.stage0,
            config.stage1,
        )
    )

    log.append(
        f"stage1: {len(control_points.points)} total, "
        f"{len(control_points.accepted_points)} accepted"
    )

    if len(control_points.accepted_points) < 3:
        return _early_exit(
            out_dir,
            "FAIL",
            [f"only {len(control_points.accepted_points)} accepted Stage 1 points -- cannot fit an affine"],
            source_info["label"],
            reference_info["label"],
            log,
            anchor=anchor,
        )

    # ------------------------------------------------------------
    # Stage 3: fit + validate
    # ------------------------------------------------------------

    fit_result = fit_and_validate(
        canonical.source,
        canonical.matcher_mask,
        canonical.reference,
        control_points.accepted_points,
        anchor.anchor_source,
        source_corr_mask,
        reference_stack,
        reference_corr_mask,
        config.stage0,
        config.stage1,
        config.stage2,
    )

    from src.pipeline.fit_validate import decide_final_verdict

    verdict, reasons, perturbation_informational = decide_final_verdict(
        fit_result, anchor.anchor_source, config.stage0
    )

    log.append(f"final verdict: {verdict} reasons={reasons}")

    # ------------------------------------------------------------
    # Stage 4: export (always, regardless of verdict)
    # ------------------------------------------------------------

    metrics, files = export_bundle(
        out_dir=out_dir,
        source=canonical.source,
        reference=canonical.reference,
        mask=canonical.matcher_mask,
        source_nodata=canonical.source_nodata,
        reference_profile=canonical.reference_profile,
        all_points=control_points.points,
        fit_result=fit_result,
        anchor=anchor,
        verdict=verdict,
        verdict_reasons=reasons,
        perturbation_validation_informational_only=perturbation_informational,
        sensor_source=source_info["label"],
        sensor_reference=reference_info["label"],
        canonical_metadata=canonical.metadata,
        console_lines=log,
    )

    return RegistrationResult(
        verdict=verdict,
        verdict_reasons=reasons,
        anchor_source=anchor.anchor_source,
        sensor_source=source_info["label"],
        sensor_reference=reference_info["label"],
        translation_px=metrics["translation_px"],
        rotation_deg=metrics["rotation_deg"],
        scale=metrics["scale"],
        inlier_count=metrics["inlier_count"],
        inlier_ratio=metrics["inlier_ratio"],
        rmse_px=metrics["rmse_px"],
        sub_pixel_achieved=metrics["sub_pixel_achieved"],
        match_points=control_points.points,
        perturbation_summary=metrics["perturbation_summary"],
        metrics=metrics,
        out_dir=out_dir,
        files=files,
    )
