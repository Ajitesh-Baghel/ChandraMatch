"""
Stage 4: writes the exact PS-required output bundle for every run, regardless
of anchor method or sensor -- same filenames, same metrics.json schema, every
time. This is the concrete answer to SIH26166's Expected Solution text
("Software and registered product with corresponding match points" +
"Evaluation metric (e.g. RMSE, inlier match count, inlier ratio, etc.)").

Reuses src/coarse/export.py's already-proven raster/visualization writers
unchanged; what's new here is the uniform bundle assembly (match_points.csv's
PS-facing schema, the always-attempted perturbation summary, the explicit
sub_pixel_achieved flag, and console.log).
"""

import csv
import json
from pathlib import Path

import numpy as np

from src.coarse.export import (
    register_source,
    robust_uint8,
    save_checkerboard,
    save_overlay,
    write_registered_geotiff,
)
from src.pipeline.sub_pixel import compute_sub_pixel_status
from src.pipeline.verification_tier import compute_perturbation_verification


def write_match_points_csv(points, path):
    """
    PS-facing schema: source_x, source_y, reference_x, reference_y,
    sub_pixel_residual, method. `sub_pixel_residual` is Stage 2's fit
    residual (the only residual computed identically for every anchor type,
    dense or fallback-derived) -- Stage 1's own per-point residual is
    dense-search-only and would be blank for every keypoint-fallback pair,
    which is not "uniform distribution across the images" evidence the PS
    can compare across sensors.
    """

    fields = ["source_x", "source_y", "reference_x", "reference_y", "sub_pixel_residual", "method"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for p in points:
            writer.writerow(
                {
                    "source_x": p.get("source_x"),
                    "source_y": p.get("source_y"),
                    "reference_x": p.get("reference_x"),
                    "reference_y": p.get("reference_y"),
                    "sub_pixel_residual": p.get("fit_residual_px"),
                    "method": p.get("source", "dense_structural"),
                }
            )


def build_perturbation_summary(fit_result):
    """
    Always attempted for every run (per §6 -- "always attempted, not just
    for the pairs that historically had it"); honestly reports why it
    couldn't run when it couldn't, rather than omitting the field.
    """

    validation = fit_result.get("validation")
    reduced = fit_result.get("reduced_scale_validation")

    if validation is None:
        return {
            "attempted": False,
            "reason": "no robust fit was produced (insufficient accepted control points)",
        }

    return {
        "attempted": True,
        "all_pass": validation["all_pass"],
        "mean_response_rmse_px": validation["mean_response_rmse_px"],
        "worst_response_rmse_px": validation["worst_response_rmse_px"],
        "controls": validation["controls"],
        "reduced_scale": reduced,
        "note": validation["note"],
    }


def export_bundle(
    out_dir,
    source,
    reference,
    mask,
    source_nodata,
    reference_profile,
    all_points,
    fit_result,
    anchor,
    verdict,
    verdict_reasons,
    perturbation_validation_informational_only,
    sensor_source,
    sensor_reference,
    canonical_metadata,
    console_lines,
):
    """
    Writes every §6 artifact into `out_dir` and returns (metrics_dict,
    files_dict). `console_lines` is a list of strings already printed/logged
    by the caller -- written verbatim to console.log so the full run,
    including the anchor-confidence model's feature values once that's
    wired in, is always reconstructable from disk.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit = fit_result["fit"]
    files = {}

    canonical_shape = mask.shape if mask is not None else (1, 1)

    sub_pixel_achieved, sub_pixel_reason, rmse_px, effective_anchor_resolution_px = (
        compute_sub_pixel_status(
            anchor.anchor_source, fit, fit_result.get("residuals"), canonical_shape
        )
    )

    perturbation_verified, perturbation_reason, verification_tier = (
        compute_perturbation_verification(anchor.anchor_source, fit_result)
    )

    if fit is not None:
        source_valid = mask & np.isfinite(source)
        reference_valid = mask & np.isfinite(reference)

        registered, registered_valid = register_source(
            source, source_valid, reference_valid, fit["matrix"]
        )

        registered_path = out_dir / "registered_product.tif"
        nodata = write_registered_geotiff(
            registered_path, registered, registered_valid, reference_profile, source_nodata
        )

        visual_valid = registered_valid & reference_valid
        registered_preview = robust_uint8(registered, visual_valid)
        reference_preview = robust_uint8(reference, visual_valid)

        overlay_path = out_dir / "overlay.png"
        checkerboard_path = out_dir / "checkerboard.png"

        save_overlay(registered_preview, reference_preview, visual_valid, overlay_path)
        save_checkerboard(registered_preview, reference_preview, visual_valid, checkerboard_path)

        files["registered_product.tif"] = registered_path
        files["overlay.png"] = overlay_path
        files["checkerboard.png"] = checkerboard_path

        fit_summary = {
            "affine": fit["matrix"].tolist(),
            "affine_parameters": fit["params"],
            "sane": fit["sane"],
            "ransac_inliers": fit["inlier_count"],
            "ransac_candidates": fit["candidate_count"],
            "ransac_inlier_ratio": fit["inlier_ratio"],
        }
        registered_product_meta = {
            "filename": registered_path.name,
            "valid_pixels": int(registered_valid.sum()),
            "nodata_value": nodata,
        }
    else:
        fit_summary = None
        registered_product_meta = None

    match_points_path = out_dir / "match_points.csv"
    write_match_points_csv(fit_result["final_points"] or all_points, match_points_path)
    files["match_points.csv"] = match_points_path

    perturbation_summary = build_perturbation_summary(fit_result)

    translation_px = (None, None)
    rotation_deg = None
    scale = None

    if fit is not None:
        params = fit["params"]
        translation_px = (params["translation_x_px"], params["translation_y_px"])
        rotation_deg = params["rotation_deg"]
        scale = (params["scale_x"], params["scale_y"])

    metrics = {
        "sensor_source": sensor_source,
        "sensor_reference": sensor_reference,
        "canonical_metadata": canonical_metadata,
        "anchor_source": anchor.anchor_source,
        "anchor_confidence": anchor.confidence,
        "stage1_points_total": len(all_points),
        "stage1_points_accepted": sum(1 for p in all_points if p.get("stage1_accepted")),
        "fit": fit_summary,
        "rmse_px": rmse_px,
        "inlier_count": fit["inlier_count"] if fit is not None else None,
        "inlier_ratio": fit["inlier_ratio"] if fit is not None else None,
        "translation_px": translation_px,
        "rotation_deg": rotation_deg,
        "scale": scale,
        "sub_pixel_achieved": sub_pixel_achieved,
        "sub_pixel_reason": sub_pixel_reason,
        "effective_anchor_resolution_px": effective_anchor_resolution_px,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "perturbation_validation_informational_only": perturbation_validation_informational_only,
        "perturbation_verified": perturbation_verified,
        "perturbation_reason": perturbation_reason,
        "verification_tier": verification_tier,
        "perturbation_summary": perturbation_summary,
        "registered_product": registered_product_meta,
        "interpretation_notes": [
            "RANSAC reprojection statistics (rmse_px, inlier_count, inlier_ratio) are "
            "internal self-consistency, not independent ground-truth accuracy.",
            "Interior-ROI perturbation response measures motion-following robustness "
            "inside a region guaranteed to stay valid under every tested perturbation; "
            "it is not independent absolute accuracy for the unperturbed pair.",
            "sub_pixel_achieved is True only for a dense_correlation anchor whose Stage 2 "
            "residuals (an independent native-resolution local-refinement measurement) are "
            "below one pixel -- a keypoint-fallback anchor's Stage 1 reuses the same points "
            "that already defined the coarse anchor, so it can never support a sub-pixel "
            "claim regardless of how tight the resulting residual looks; see sub_pixel_reason.",
            "verification_tier distinguishes HOW a PASS/AMBIGUOUS verdict is supported: "
            "'perturbation_confirmed' is the only tier backed by an independent stress test; "
            "'perturbation_partial' has some but incomplete independent support; "
            "'ransac_and_prior_ground_truth_only' means the anchor came from a keypoint matcher "
            "and the perturbation test cannot arbitrate this anchor class at all (a keypoint "
            "matcher re-verifying its own anchor under large injected motion is a documented, "
            "repeatedly-reproduced blind spot -- see perturbation_reason) -- its PASS rests on "
            "RANSAC coarse-fit quality and, where available, separately-established prior "
            "ground truth, not on this pipeline's own perturbation test.",
        ],
    }

    metrics_path = out_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    files["metrics.json"] = metrics_path

    console_path = out_dir / "console.log"
    console_path.write_text("\n".join(console_lines) + "\n", encoding="utf-8")
    files["console.log"] = console_path

    return metrics, files
