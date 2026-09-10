import json
import math
from pathlib import Path

import numpy as np

from src.coarse.keypoint_fallback import run_keypoint_fallback
from src.coarse.oriented_gradients import build_oriented_gradient_stack
from src.coarse.masked_correlation import (
    ambiguity_ratio,
    channel_summed_surface,
    extract_topk_peaks,
    surface_index_to_shift,
)
from src.coarse.verification import (
    local_cell_agreement,
    mask_bbox_fill_ratio,
    structural_ncc_before_after,
    valid_overlap_fraction,
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.pop("notes", None)

    return config


def _decide_verdict(
    ambiguity_ratio_value,
    overlap_fraction,
    ncc_before,
    ncc_after,
    local_agreement,
    bbox_fill_ratio,
    config,
):
    reasons = []

    if bbox_fill_ratio is None or bbox_fill_ratio < config["min_bbox_fill_ratio"]:
        reasons.append(
            f"bbox_fill_ratio={bbox_fill_ratio} below "
            f"min_bbox_fill_ratio={config['min_bbox_fill_ratio']} -- footprint is too "
            "elongated/sparse within its own bounding box for whole-frame translation "
            "search to be well-conditioned (known aperture-problem blind spot, see "
            "Pair004)"
        )
        return "FAIL", reasons

    if overlap_fraction is None or overlap_fraction < config["min_overlap_fraction"]:
        reasons.append(
            f"overlap_fraction={overlap_fraction} below "
            f"min_overlap_fraction={config['min_overlap_fraction']}"
        )
        return "FAIL", reasons

    if ncc_before is None or ncc_after is None:
        reasons.append(
            "structural NCC could not be computed (insufficient common "
            "valid pixels before or after warp)"
        )
        return "FAIL", reasons

    ncc_improvement = ncc_after - ncc_before

    if ncc_after < 0 or ncc_improvement < 0:
        reasons.append(
            f"structural NCC did not improve after warp "
            f"(before={ncc_before:.4f}, after={ncc_after:.4f})"
        )
        return "FAIL", reasons

    soft_failures = []

    if (
        ambiguity_ratio_value is None
        or ambiguity_ratio_value < config["ambiguity_ratio_min"]
    ):
        soft_failures.append(
            f"ambiguity_ratio={ambiguity_ratio_value} below "
            f"ambiguity_ratio_min={config['ambiguity_ratio_min']}"
        )

    if ncc_after < config["min_ncc_after"]:
        soft_failures.append(
            f"ncc_after={ncc_after:.4f} below "
            f"min_ncc_after={config['min_ncc_after']}"
        )

    if ncc_improvement < config["min_ncc_improvement"]:
        soft_failures.append(
            f"ncc_improvement={ncc_improvement:.4f} below "
            f"min_ncc_improvement={config['min_ncc_improvement']}"
        )

    if (
        local_agreement is None
        or local_agreement < config["min_local_cell_agreement"]
    ):
        soft_failures.append(
            f"local_cell_agreement={local_agreement} below "
            f"min_local_cell_agreement={config['min_local_cell_agreement']}"
        )

    if soft_failures:
        return "AMBIGUOUS", soft_failures

    return "PASS", []


def run_stage0(source_image, source_mask, reference_image, reference_mask, config):
    """
    Dense structural coarse search (Stage 0).

    `source_image`/`reference_image` are raw single-band rasters (any
    dtype/range); `source_mask`/`reference_mask` are boolean arrays, all
    of the same (H, W) shape. `config` is the dict loaded from
    configs/stage0_coarse_search.json -- the same config must be used for
    every pair, never a per-pair copy.

    Returns a dict with ranked `hypotheses`, `verification` metrics for
    the best hypothesis, and a `verdict` of "PASS" / "AMBIGUOUS" / "FAIL".
    """

    num_orientations = config["num_orientations"]
    gaussian_sigma = config["gaussian_sigma"]
    mask_erosion_px = config.get("mask_erosion_px", 5)

    # Memory-safety check, found necessary by real testing: a raw LRO
    # NAC framelet canonicalized against a full raw mission-scale TMC-2
    # scene produced a 13827x6058 canvas, and the 'full'-mode masked FFT
    # correlation needs padded arrays proportional to canvas area --
    # that specific case needed ~20GB and crashed with a MemoryError.
    # Every pair validated so far (up to Pair006's 6735x1886, ~12.7M
    # px) works fine; this threshold sits comfortably above that and
    # safely below the case that crashed. Skipping straight to FAIL here
    # (rather than crashing) lets run_stage0_with_fallback's keypoint
    # path take over, which doesn't have this scaling problem.
    max_dense_search_pixels = config.get("max_dense_search_pixels", 30_000_000)
    canvas_pixels = source_image.shape[0] * source_image.shape[1]

    if canvas_pixels > max_dense_search_pixels:
        return {
            "hypotheses": [],
            "ambiguity_ratio": None,
            "best": None,
            "verification": None,
            "verdict": "FAIL",
            "verdict_reasons": [
                f"canonical canvas ({source_image.shape[0]}x{source_image.shape[1]} = "
                f"{canvas_pixels:,} px) exceeds max_dense_search_pixels="
                f"{max_dense_search_pixels:,} -- the 'full'-mode masked FFT correlation "
                "needs memory proportional to canvas area and would exceed available "
                "RAM; skipping straight to keypoint fallback rather than crashing"
            ],
            "anchor_source": "dense_correlation",
            "config": config,
        }

    source_stack, source_corr_mask = build_oriented_gradient_stack(
        source_image, source_mask, num_orientations, gaussian_sigma, mask_erosion_px
    )

    reference_stack, reference_corr_mask = build_oriented_gradient_stack(
        reference_image,
        reference_mask,
        num_orientations,
        gaussian_sigma,
        mask_erosion_px,
    )

    surface, ref_shape = channel_summed_surface(
        source_stack,
        source_corr_mask,
        reference_stack,
        reference_corr_mask,
        overlap_ratio=config["overlap_ratio_fft"],
    )

    peaks = extract_topk_peaks(
        surface,
        top_k=config["top_k_peaks"],
        nms_radius=config["nms_radius_px"],
    )

    hypotheses = []

    for peak in peaks:
        dy, dx = surface_index_to_shift(peak["index"], ref_shape)

        hypotheses.append(
            {
                "dy": dy,
                "dx": dx,
                "translation_magnitude_px": float(math.hypot(dy, dx)),
                "score": peak["score"],
            }
        )

    ratio = ambiguity_ratio(peaks)

    if not hypotheses:
        return {
            "hypotheses": [],
            "ambiguity_ratio": None,
            "best": None,
            "verification": None,
            "verdict": "FAIL",
            "verdict_reasons": ["no correlation peaks found"],
            "config": config,
        }

    best = hypotheses[0]
    dy, dx = best["dy"], best["dx"]

    overlap_fraction = valid_overlap_fraction(
        source_corr_mask, reference_corr_mask, dy, dx
    )

    bbox_fill_ratio = mask_bbox_fill_ratio(reference_corr_mask)

    source_structural = source_stack.sum(axis=0)
    reference_structural = reference_stack.sum(axis=0)

    ncc_before, ncc_after = structural_ncc_before_after(
        source_structural,
        source_corr_mask,
        reference_structural,
        reference_corr_mask,
        dy,
        dx,
    )

    local = local_cell_agreement(
        source_stack,
        source_corr_mask,
        reference_stack,
        reference_corr_mask,
        dy,
        dx,
        grid_rows=config["local_grid_rows"],
        grid_cols=config["local_grid_cols"],
        window_px=config["local_window_px"],
        min_valid_fraction=config["local_min_valid_fraction"],
        tolerance_px=config["local_agreement_tolerance_px"],
        overlap_ratio=config["overlap_ratio_fft"],
    )

    verdict, reasons = _decide_verdict(
        ratio,
        overlap_fraction,
        ncc_before,
        ncc_after,
        local["agreement_fraction"],
        bbox_fill_ratio,
        config,
    )

    ncc_improvement = (
        (ncc_after - ncc_before)
        if (ncc_before is not None and ncc_after is not None)
        else None
    )

    return {
        "hypotheses": hypotheses,
        "ambiguity_ratio": ratio,
        "best": best,
        "verification": {
            "bbox_fill_ratio": bbox_fill_ratio,
            "overlap_fraction": overlap_fraction,
            "ncc_before": ncc_before,
            "ncc_after": ncc_after,
            "ncc_improvement": ncc_improvement,
            "local_cell_agreement": local["agreement_fraction"],
            "local_cells": local["cells"],
        },
        "verdict": verdict,
        "verdict_reasons": reasons,
        "anchor_source": "dense_correlation",
        "config": config,
    }


def _jsonify_attempt(attempt):
    cleaned = {}

    for key, value in attempt.items():
        if isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        else:
            cleaned[key] = value

    return cleaned


def _clean_fallback_for_json(fallback):
    cleaned = _jsonify_attempt(fallback)

    if "attempts" in cleaned:
        cleaned["attempts"] = [_jsonify_attempt(attempt) for attempt in cleaned["attempts"]]

    return cleaned


def run_stage0_with_fallback(
    source_image, source_mask, reference_image, reference_mask, config
):
    """
    Runs the dense structural search first. If it doesn't PASS and
    `enable_keypoint_fallback` is set, attempts v1's proven SIFT matcher
    as a fallback anchor-finder (see keypoint_fallback.py for why this
    is architecturally safe -- Stage 1/2 still independently verify
    whatever anchor comes out of here). The dense-search result is
    always preserved under `dense_search_result` for transparency, even
    when the fallback is what actually produces the final anchor.
    """

    dense_result = run_stage0(source_image, source_mask, reference_image, reference_mask, config)

    if dense_result["verdict"] == "PASS" or not config.get("enable_keypoint_fallback", False):
        return dense_result

    fallback = run_keypoint_fallback(
        source_image, source_mask, reference_image, reference_mask, config
    )

    if fallback["verdict"] != "PASS":
        result = dict(dense_result)
        result["dense_search_result"] = None
        result["keypoint_fallback_result"] = _clean_fallback_for_json(fallback)
        result["verdict_reasons"] = dense_result["verdict_reasons"] + [
            f"keypoint fallback also did not pass: {fallback['reasons']}"
        ]
        return result

    dy = int(round(fallback["params"]["translation_y_px"]))
    dx = int(round(fallback["params"]["translation_x_px"]))

    # Recompute verification AT THE FALLBACK'S OWN ANCHOR -- dense_result's
    # verification was computed at the dense search's (different, rejected)
    # anchor and would be silently wrong here otherwise.
    num_orientations = config["num_orientations"]
    gaussian_sigma = config["gaussian_sigma"]
    mask_erosion_px = config.get("mask_erosion_px", 5)

    source_stack, source_corr_mask = build_oriented_gradient_stack(
        source_image, source_mask, num_orientations, gaussian_sigma, mask_erosion_px
    )
    reference_stack, reference_corr_mask = build_oriented_gradient_stack(
        reference_image, reference_mask, num_orientations, gaussian_sigma, mask_erosion_px
    )

    overlap_fraction = valid_overlap_fraction(source_corr_mask, reference_corr_mask, dy, dx)
    bbox_fill_ratio = mask_bbox_fill_ratio(reference_corr_mask)

    ncc_before, ncc_after = structural_ncc_before_after(
        source_stack.sum(axis=0),
        source_corr_mask,
        reference_stack.sum(axis=0),
        reference_corr_mask,
        dy,
        dx,
    )

    local = local_cell_agreement(
        source_stack,
        source_corr_mask,
        reference_stack,
        reference_corr_mask,
        dy,
        dx,
        grid_rows=config["local_grid_rows"],
        grid_cols=config["local_grid_cols"],
        window_px=config["local_window_px"],
        min_valid_fraction=config["local_min_valid_fraction"],
        tolerance_px=config["local_agreement_tolerance_px"],
        overlap_ratio=config["overlap_ratio_fft"],
    )

    ncc_improvement = (
        (ncc_after - ncc_before)
        if (ncc_before is not None and ncc_after is not None)
        else None
    )

    return {
        "hypotheses": dense_result["hypotheses"],
        "ambiguity_ratio": dense_result["ambiguity_ratio"],
        "best": {
            "dy": dy,
            "dx": dx,
            "translation_magnitude_px": fallback["params"]["translation_magnitude_px"],
            "score": None,
        },
        "verification": {
            "overlap_fraction": overlap_fraction,
            "bbox_fill_ratio": bbox_fill_ratio,
            "ncc_before": ncc_before,
            "ncc_after": ncc_after,
            "ncc_improvement": ncc_improvement,
            "local_cell_agreement": local["agreement_fraction"],
            "local_cells": local["cells"],
        },
        "verdict": "PASS",
        "verdict_reasons": [
            "anchor found via keypoint fallback, not dense correlation -- "
            "verification above is recomputed at this anchor but the "
            "ambiguity_ratio/hypotheses fields still reflect the dense "
            "search's (rejected) result, see dense_search_result"
        ],
        "anchor_source": f"keypoint_fallback_{fallback['method']}",
        "dense_search_result": {
            "verdict": dense_result["verdict"],
            "verdict_reasons": dense_result["verdict_reasons"],
        },
        "keypoint_fallback_result": _clean_fallback_for_json(fallback),
        "config": config,
    }
