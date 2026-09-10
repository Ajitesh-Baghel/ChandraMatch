"""
Stage 3: robust affine fit through Stage 2's accepted control points, plus
the interior-ROI synthetic-perturbation stress test (and its reduced-scale
supplementary variant when the standard test is untestable).

`decide_final_verdict` extracts the final PASS/AMBIGUOUS/FAIL logic that was
previously duplicated, keystroke-for-keystroke, between
app/pipeline_runner.py and scripts/run_stage2_registration.py -- both had
this exact reasoning inline. This is now the one place it lives; both
callers are updated to call it instead of re-deriving it (see run.py and the
updated CLI scripts).
"""

import numpy as np

from src.coarse.oriented_gradients import build_oriented_gradient_stack
from src.coarse.robust_fit import fit_constrained_affine, residual_consistency
from src.coarse.stage2_validation import (
    run_interior_roi_perturbation_validation,
    run_reduced_scale_perturbation_validation,
)


def build_descriptor_stacks_if_missing(
    source, mask, reference, source_corr_mask, reference_stack, reference_corr_mask, stage0_config
):
    """
    Perturbation validation always needs the oriented-gradient descriptor
    stacks regardless of anchor type (a keypoint-anchored pair still gets the
    dense-method's descriptor built here, purely as validation input -- see
    stage2_validation.run_single_perturbation_keypoint, which needs the raw
    arrays, not the stack, but the dense branch below does). Rebuilt only if
    refine.py's caller didn't already have them in memory (the fallback-
    anchor branch never builds them).
    """

    if reference_stack is None:
        reference_stack, reference_corr_mask = build_oriented_gradient_stack(
            reference,
            mask,
            stage0_config["num_orientations"],
            stage0_config["gaussian_sigma"],
            stage0_config.get("mask_erosion_px", 5),
        )

    if source_corr_mask is None:
        _, source_corr_mask = build_oriented_gradient_stack(
            source,
            mask,
            stage0_config["num_orientations"],
            stage0_config["gaussian_sigma"],
            stage0_config.get("mask_erosion_px", 5),
        )

    return source_corr_mask, reference_stack, reference_corr_mask


def fit_and_validate(
    source,
    mask,
    reference,
    accepted_points,
    anchor_source,
    source_corr_mask,
    reference_stack,
    reference_corr_mask,
    stage0_config,
    stage1_config,
    stage2_config,
):
    """
    Returns a dict with `fit`, `residuals`/`is_outlier` applied onto
    `accepted_points` in place (adds fit_residual_px/fit_outlier/
    final_accepted, matching the existing CSV schema), `validation`,
    `reduced_scale_validation`, and `final_points`. Returns
    {"fit": None, ...} if there aren't enough points to fit -- callers must
    check this before proceeding, same as before.
    """

    if len(accepted_points) < 3:
        return {
            "fit": None,
            "final_points": [],
            "outlier_count": 0,
            "outlier_rate": 1.0,
            "validation": None,
            "reduced_scale_validation": None,
            "residuals": None,
        }

    source_xy = np.asarray(
        [[p["source_x"], p["source_y"]] for p in accepted_points], dtype=np.float32
    )
    reference_xy = np.asarray(
        [[p["reference_x"], p["reference_y"]] for p in accepted_points], dtype=np.float32
    )

    fit = fit_constrained_affine(source_xy, reference_xy, stage2_config)

    if fit is None:
        return {
            "fit": None,
            "final_points": [],
            "outlier_count": 0,
            "outlier_rate": 1.0,
            "validation": None,
            "reduced_scale_validation": None,
            "residuals": None,
        }

    residuals, is_outlier = residual_consistency(
        source_xy, reference_xy, fit["matrix"], stage2_config["max_fit_residual_px"]
    )

    for p, residual, outlier in zip(accepted_points, residuals, is_outlier):
        p["fit_residual_px"] = float(residual)
        p["fit_outlier"] = bool(outlier)
        p["final_accepted"] = bool(not outlier)

    final_points = [p for p in accepted_points if p["final_accepted"]]
    outlier_count = int(np.sum(is_outlier))
    outlier_rate = outlier_count / len(accepted_points) if accepted_points else 1.0

    source_corr_mask, reference_stack, reference_corr_mask = build_descriptor_stacks_if_missing(
        source, mask, reference, source_corr_mask, reference_stack, reference_corr_mask, stage0_config
    )

    validation = run_interior_roi_perturbation_validation(
        source,
        source_corr_mask,
        reference_stack,
        reference_corr_mask,
        fit["matrix"],
        stage0_config,
        stage1_config,
        stage2_config,
        anchor_source=anchor_source,
        reference_raw=reference,
    )

    reduced_scale_validation = None

    all_untestable = bool(validation["controls"]) and all(
        c["interior_roi_pixels"] == 0 for c in validation["controls"]
    )

    if all_untestable:
        reduced_scale_validation = run_reduced_scale_perturbation_validation(
            source,
            source_corr_mask,
            reference_stack,
            reference_corr_mask,
            fit["matrix"],
            stage0_config,
            stage1_config,
            stage2_config,
        )

    return {
        "fit": fit,
        "final_points": final_points,
        "outlier_count": outlier_count,
        "outlier_rate": outlier_rate,
        "validation": validation,
        "reduced_scale_validation": reduced_scale_validation,
        "residuals": residuals,
    }


def decide_final_verdict(fit_result, anchor_source, stage0_config, final_points_min=5):
    """
    Single shared verdict logic -- previously duplicated identically between
    app/pipeline_runner.py and scripts/run_stage2_registration.py. Returns
    (verdict, reasons, perturbation_validation_informational_only).
    """

    fit = fit_result["fit"]
    reasons = []
    perturbation_validation_informational_only = False

    if fit is None:
        return "FAIL", ["robust affine fit failed"], False

    if not fit["sane"]:
        reasons.append("fitted affine failed shape-sanity check")

    if fit_result["outlier_rate"] > 0.3:
        reasons.append(f"fit-residual outlier rate {fit_result['outlier_rate']:.2f} exceeds 0.30")

    validation = fit_result["validation"]
    reduced_scale_validation = fit_result["reduced_scale_validation"]

    if anchor_source != "dense_correlation":
        # Confirmed by direct testing, not assumed: re-verifying a
        # keypoint-fallback anchor with ANOTHER keypoint matcher under large
        # injected perturbation reproduces the documented "identity bias"
        # failure mode of learned matchers -- the same matcher family can't
        # independently check itself. Falls back to judging coarse-fit
        # quality alone.
        perturbation_validation_informational_only = True

        fallback_inliers = stage0_config.get("min_fallback_inliers", 15)
        fallback_ratio = stage0_config.get("min_fallback_inlier_ratio", 0.08)

        if fit["inlier_count"] < fallback_inliers:
            reasons.append(
                f"only {fit['inlier_count']} RANSAC inliers on the coarse fit, "
                f"below {fallback_inliers}"
            )

        if fit["inlier_ratio"] < fallback_ratio:
            reasons.append(
                f"inlier_ratio={fit['inlier_ratio']:.3f} on the coarse fit, "
                f"below min_fallback_inlier_ratio={fallback_ratio}"
            )
    elif validation is not None and not validation["all_pass"]:
        untestable = [
            c["perturbation_id"]
            for c in validation["controls"]
            if not c["pass"] and c["interior_roi_pixels"] == 0
        ]
        failed = [
            c["perturbation_id"]
            for c in validation["controls"]
            if not c["pass"] and c["interior_roi_pixels"] != 0
        ]

        if untestable and not failed:
            reasons.append(
                f"interior-ROI perturbation validation UNTESTABLE for {untestable} -- "
                "this pair's post-registration common overlap has no region safely "
                "interior to the perturbation margins; this is a limit of what can be "
                "validated, not a failed test"
            )

            if reduced_scale_validation is not None:
                if not reduced_scale_validation["possible"]:
                    reasons.append(
                        "reduced-scale supplementary test also not possible: "
                        f"{reduced_scale_validation['reason']}"
                    )
                elif reduced_scale_validation["all_pass"]:
                    reasons.append(
                        "reduced-scale supplementary test PASSES "
                        f"(scale={reduced_scale_validation['scale_factor']:.3f}x standard "
                        "magnitude) -- supplementary evidence only, not equivalent to the "
                        "full-strength standard test"
                    )
                else:
                    reasons.append("reduced-scale supplementary test also did not pass")
        elif failed:
            reasons.append(
                f"interior-ROI perturbation validation FAILED for {failed} "
                "(response exceeded threshold or affine was insane)"
            )
            if untestable:
                reasons.append(f"also untestable (empty interior ROI) for {untestable}")

    verdict = "PASS" if not reasons else ("FAIL" if len(fit_result["final_points"]) < final_points_min else "AMBIGUOUS")

    return verdict, reasons, perturbation_validation_informational_only
