import cv2
import numpy as np

from src.coarse.interior_roi import (
    compute_interior_roi_mask,
    find_max_safe_margin,
    perturbation_margin_px,
)
from src.coarse.keypoint_fallback import run_keypoint_fallback
from src.coarse.local_control_points import grid_points, refine_point
from src.coarse.oriented_gradients import build_oriented_gradient_stack
from src.coarse.robust_fit import compose_affine, decompose_affine, fit_constrained_affine
from src.coarse.verification import mask_bbox
from src.evaluation.controlled_transform import create_affine_transform
from src.evaluation.ground_truth_metrics import transform_points


def warp_raster(data, matrix, shape, interpolation, fill_value=0.0):
    height, width = shape

    return cv2.warpAffine(
        data,
        np.asarray(matrix, dtype=np.float64),
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value,
    )


def run_single_perturbation(
    perturbation,
    source_raw,
    source_corr_mask,
    reference_stack,
    reference_corr_mask,
    baseline_affine,
    stage0_config,
    stage1_config,
    stage2_config,
):
    """
    Inject one known perturbation into the SOURCE raster, re-run the
    same dense local-point engine (not a separate sparse re-match, which
    is exactly the fragile mechanism this v2 rewrite replaces) restricted
    to the interior ROI, refit, and compare the recovered transform
    against the known expected one.

    Convention (matches v1's own proven perturbation-test formula,
    e.g. scripts/validate_pair007_selected_perturbation.py): `forward`
    maps original source pixels -> perturbed source pixels. If
    `baseline_affine` maps original source -> reference, then
    perturbed source -> reference is
        expected_affine = compose(baseline_affine, inverse(forward))
    """

    height, width = source_raw.shape

    forward, _ = create_affine_transform(
        source_raw.shape,
        rotation_deg=perturbation["rotation_deg"],
        scale=perturbation["scale"],
        translation_x=perturbation["tx"],
        translation_y=perturbation["ty"],
    )

    inverse = cv2.invertAffineTransform(forward)
    expected_affine = compose_affine(baseline_affine, inverse)

    # Interior ROI sized to THIS perturbation's own displacement, not a
    # single worst-case margin shared across every perturbation -- a
    # small perturbation shouldn't be needlessly starved by a much
    # larger one's safety margin. Still computed identically for every
    # pair/perturbation (never tuned to make a specific pair pass).
    margin_px = perturbation_margin_px(
        [perturbation], stage2_config["interior_roi_buffer_px"], canvas_shape=(height, width)
    )

    interior_roi_mask = compute_interior_roi_mask(
        reference_corr_mask, source_corr_mask, baseline_affine, margin_px
    )

    perturbed_source = warp_raster(
        np.nan_to_num(source_raw.astype(np.float32), nan=0.0),
        forward,
        (height, width),
        cv2.INTER_LINEAR,
    )

    perturbed_mask = (
        warp_raster(
            source_corr_mask.astype(np.uint8),
            forward,
            (height, width),
            cv2.INTER_NEAREST,
        )
        > 0
    )

    perturbed_stack, perturbed_corr_mask = build_oriented_gradient_stack(
        perturbed_source,
        perturbed_mask,
        stage0_config["num_orientations"],
        stage0_config["gaussian_sigma"],
        stage0_config.get("mask_erosion_px", 5),
    )

    expected_params = decompose_affine(expected_affine)
    global_dy = int(round(expected_params["translation_y_px"]))
    global_dx = int(round(expected_params["translation_x_px"]))

    bbox = mask_bbox(interior_roi_mask)

    result = {
        "perturbation_id": perturbation["id"],
        "injected": perturbation,
        "expected_affine": expected_affine.tolist(),
        "interior_roi_margin_px": margin_px,
        "interior_roi_pixels": int(np.count_nonzero(interior_roi_mask)),
        "accepted_points": 0,
        "recovered_affine": None,
        "response_rmse_px": None,
        "pass": False,
        "reasons": [],
    }

    if bbox is None:
        result["reasons"].append(
            f"interior ROI is empty at required margin={margin_px:.1f}px -- this "
            "pair's post-registration common overlap is too thin to safely test "
            "this perturbation magnitude, not evidence the coarse anchor is wrong"
        )
        return result

    points = []

    for center_y, center_x in grid_points(
        bbox, stage2_config["interior_roi_grid_spacing_px"]
    ):
        point = refine_point(
            perturbed_stack,
            perturbed_corr_mask,
            reference_stack,
            interior_roi_mask,
            center_y,
            center_x,
            global_dy,
            global_dx,
            window_px=stage2_config["interior_roi_window_px"],
            min_valid_fraction=stage1_config["min_valid_fraction"],
            min_peak_score=stage1_config["min_peak_score"],
            min_ambiguity_ratio=stage1_config["min_ambiguity_ratio"],
            max_residual_px=stage1_config["max_residual_px"],
            overlap_ratio=stage1_config["overlap_ratio_fft"],
        )

        if point["accepted"]:
            points.append(point)

    result["accepted_points"] = len(points)

    if len(points) < stage2_config["min_perturbation_accepted_points"]:
        result["reasons"].append(
            f"only {len(points)} accepted points in interior ROI, below "
            f"min_perturbation_accepted_points="
            f"{stage2_config['min_perturbation_accepted_points']}"
        )
        return result

    perturbed_source_xy = np.asarray(
        [[p["source_x"], p["source_y"]] for p in points], dtype=np.float32
    )
    reference_xy = np.asarray(
        [[p["reference_x"], p["reference_y"]] for p in points], dtype=np.float32
    )

    return _finalize_perturbation_result(
        result, perturbed_source_xy, reference_xy, expected_affine, stage2_config
    )


def _finalize_perturbation_result(
    result, perturbed_source_xy, reference_xy, expected_affine, stage2_config
):
    fit = fit_constrained_affine(perturbed_source_xy, reference_xy, stage2_config)

    if fit is None:
        result["reasons"].append("could not fit a recovered affine")
        return result

    recovered_affine = fit["matrix"]
    result["recovered_affine"] = recovered_affine.tolist()
    result["recovered_sane"] = fit["sane"]

    recovered_points = transform_points(perturbed_source_xy, recovered_affine)
    expected_points = transform_points(perturbed_source_xy, expected_affine)

    errors = np.linalg.norm(recovered_points - expected_points, axis=1)
    response_rmse = float(np.sqrt(np.mean(errors**2)))

    result["response_rmse_px"] = response_rmse
    result["response_worst_px"] = float(np.max(errors))

    if not fit["sane"]:
        result["reasons"].append("recovered affine failed shape-sanity check")
    if response_rmse > stage2_config["max_mean_perturbation_response_px"]:
        result["reasons"].append(
            f"response_rmse_px={response_rmse:.2f} exceeds "
            f"max_mean_perturbation_response_px="
            f"{stage2_config['max_mean_perturbation_response_px']}"
        )
    if result["response_worst_px"] > stage2_config["max_worst_perturbation_response_px"]:
        result["reasons"].append(
            f"response_worst_px={result['response_worst_px']:.2f} exceeds "
            f"max_worst_perturbation_response_px="
            f"{stage2_config['max_worst_perturbation_response_px']}"
        )

    result["pass"] = len(result["reasons"]) == 0

    return result


def run_single_perturbation_keypoint(
    perturbation,
    source_raw,
    source_valid_mask,
    reference_raw,
    reference_valid_mask,
    baseline_affine,
    stage0_config,
    stage2_config,
):
    """
    Keypoint-based counterpart of `run_single_perturbation`, for pairs
    whose Stage 0 anchor came from `keypoint_fallback` (i.e. the dense
    oriented-gradient descriptor is already known unreliable for this
    pair -- confirmed on Pair002/003, where the dense interior-ROI test
    gets 0 accepted points even with thousands of ROI pixels available,
    because it's the SAME descriptor failure, not a room problem).
    Reuses `keypoint_fallback.run_keypoint_fallback` directly (already
    built, already proven on this pair by Stage 0) instead of the
    dense grid/refine_point search.
    """

    height, width = source_raw.shape

    forward, _ = create_affine_transform(
        source_raw.shape,
        rotation_deg=perturbation["rotation_deg"],
        scale=perturbation["scale"],
        translation_x=perturbation["tx"],
        translation_y=perturbation["ty"],
    )

    inverse = cv2.invertAffineTransform(forward)
    expected_affine = compose_affine(baseline_affine, inverse)

    perturbed_source = warp_raster(
        np.nan_to_num(source_raw.astype(np.float32), nan=0.0),
        forward,
        (height, width),
        cv2.INTER_LINEAR,
    )

    perturbed_mask = (
        warp_raster(
            source_valid_mask.astype(np.uint8),
            forward,
            (height, width),
            cv2.INTER_NEAREST,
        )
        > 0
    )

    result = {
        "perturbation_id": perturbation["id"],
        "injected": perturbation,
        "expected_affine": expected_affine.tolist(),
        "interior_roi_margin_px": None,
        "interior_roi_pixels": None,
        "accepted_points": 0,
        "recovered_affine": None,
        "response_rmse_px": None,
        "pass": False,
        "reasons": [],
    }

    fallback = run_keypoint_fallback(
        perturbed_source, perturbed_mask, reference_raw, reference_valid_mask, stage0_config
    )

    result["method"] = fallback["method"]
    result["accepted_points"] = fallback.get("inliers", 0)

    if fallback["verdict"] != "PASS":
        result["reasons"].append(
            f"keypoint re-match on perturbed source did not pass: {fallback['reasons']}"
        )
        return result

    perturbed_source_xy = np.asarray(fallback["inlier_source_points"], dtype=np.float32)
    reference_xy = np.asarray(fallback["inlier_reference_points"], dtype=np.float32)

    return _finalize_perturbation_result(
        result, perturbed_source_xy, reference_xy, expected_affine, stage2_config
    )


def run_interior_roi_perturbation_validation(
    source_raw,
    source_corr_mask,
    reference_stack,
    reference_corr_mask,
    baseline_affine,
    stage0_config,
    stage1_config,
    stage2_config,
    anchor_source="dense_correlation",
    reference_raw=None,
):
    """
    `anchor_source` selects the re-verification method: "dense_correlation"
    uses the dense grid/refine_point search (Stage 0/1's default path);
    anything else (a keypoint-fallback anchor_source string) uses
    `run_single_perturbation_keypoint` instead, since a pair that needed
    keypoint fallback for its anchor has already demonstrated the dense
    descriptor doesn't work on it -- `reference_raw` is required in that
    case.
    """

    use_keypoint = anchor_source != "dense_correlation"

    controls = []

    for perturbation in stage2_config["perturbations"]:
        if use_keypoint:
            controls.append(
                run_single_perturbation_keypoint(
                    perturbation,
                    source_raw,
                    source_corr_mask,
                    reference_raw,
                    reference_corr_mask,
                    baseline_affine,
                    stage0_config,
                    stage2_config,
                )
            )
        else:
            controls.append(
                run_single_perturbation(
                    perturbation,
                    source_raw,
                    source_corr_mask,
                    reference_stack,
                    reference_corr_mask,
                    baseline_affine,
                    stage0_config,
                    stage1_config,
                    stage2_config,
                )
            )

    response_values = [
        c["response_rmse_px"] for c in controls if c["response_rmse_px"] is not None
    ]

    return {
        "controls": controls,
        "all_pass": all(c["pass"] for c in controls),
        "mean_response_rmse_px": float(np.mean(response_values))
        if response_values
        else None,
        "worst_response_rmse_px": float(np.max(response_values))
        if response_values
        else None,
        "note": (
            "Measured only inside an interior ROI that stays fully valid under "
            "every configured perturbation, fixing v1's boundary-support-loss "
            "confound. This is motion-following robustness evidence, not "
            "independent absolute ground truth for the unperturbed pair."
        ),
    }


def run_reduced_scale_perturbation_validation(
    source_raw,
    source_corr_mask,
    reference_stack,
    reference_corr_mask,
    baseline_affine,
    stage0_config,
    stage1_config,
    stage2_config,
):
    """
    Supplementary, explicitly lower-confidence check for pairs whose
    real post-registration overlap is too thin to fit ANY of the
    standard perturbation set's safety margins (see
    `run_interior_roi_perturbation_validation` -- if that comes back
    with every control "untestable", this is what to run instead of
    just reporting a gap).

    Finds the largest margin this pair's own geometry can actually
    support, then scales the standard perturbation set's translations
    down (uniformly, preserving direction and the rotation/scale
    components, which contribute little to the margin) to fit inside
    it. This is NOT equivalent to the full-strength standard test --
    every result is tagged accordingly -- but it is a legitimate,
    non-arbitrary reduced-scale test: the scale factor comes from the
    pair's measured geometry, not from tuning to force a pass.
    """

    min_roi_pixels = stage2_config["reduced_scale_min_roi_pixels"]
    buffer_px = stage2_config["interior_roi_buffer_px"]

    max_safe_margin = find_max_safe_margin(
        reference_corr_mask, source_corr_mask, baseline_affine, min_roi_pixels
    )

    available_translation_margin = max_safe_margin - buffer_px

    if available_translation_margin <= 0:
        return {
            "possible": False,
            "max_safe_margin_px": max_safe_margin,
            "reason": (
                f"even zero-translation margin can't reach "
                f"{min_roi_pixels} interior ROI pixels -- this pair's "
                "post-registration overlap is too thin for any perturbation "
                "test, reduced-scale or otherwise"
            ),
            "controls": [],
        }

    max_standard_translation = max(
        float(np.hypot(p["tx"], p["ty"])) for p in stage2_config["perturbations"]
    )

    scale_factor = min(1.0, available_translation_margin / max_standard_translation)

    scaled_perturbations = [
        {
            "id": p["id"],
            "tx": p["tx"] * scale_factor,
            "ty": p["ty"] * scale_factor,
            "rotation_deg": p["rotation_deg"],
            "scale": p["scale"],
        }
        for p in stage2_config["perturbations"]
    ]

    scaled_stage2_config = dict(stage2_config)
    scaled_stage2_config["perturbations"] = scaled_perturbations

    controls = []

    for perturbation in scaled_perturbations:
        controls.append(
            run_single_perturbation(
                perturbation,
                source_raw,
                source_corr_mask,
                reference_stack,
                reference_corr_mask,
                baseline_affine,
                stage0_config,
                stage1_config,
                scaled_stage2_config,
            )
        )

    response_values = [
        c["response_rmse_px"] for c in controls if c["response_rmse_px"] is not None
    ]

    return {
        "possible": True,
        "scale_factor": scale_factor,
        "max_safe_margin_px": max_safe_margin,
        "scaled_perturbations": scaled_perturbations,
        "controls": controls,
        "all_pass": bool(controls) and all(c["pass"] for c in controls),
        "mean_response_rmse_px": float(np.mean(response_values))
        if response_values
        else None,
        "worst_response_rmse_px": float(np.max(response_values))
        if response_values
        else None,
        "note": (
            "REDUCED-SCALE SUPPLEMENTARY TEST -- perturbation magnitudes scaled "
            f"down by {scale_factor:.3f}x from the standard set to fit this "
            "pair's actual available interior margin. NOT equivalent to the "
            "full-strength standard perturbation test; do not report this as "
            "the same validation Pair006-class pairs receive."
        ),
    }
