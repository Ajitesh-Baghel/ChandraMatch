"""
Stage 2: uniform local control points across the full valid overlap.

Extracted from the branch that was duplicated between app/pipeline_runner.py
and scripts/run_stage1_local_control_points.py -- both built the same
oriented-gradient descriptor stack and called the same two functions
(`run_stage1` for a dense anchor, `points_from_keypoint_fallback` for a
fallback anchor); this is now the one place that decision is made.
"""

from src.coarse.local_control_points import points_from_keypoint_fallback, run_stage1
from src.coarse.oriented_gradients import build_oriented_gradient_stack
from src.pipeline.types import ControlPoints


def run_refinement(source, mask, reference, anchor, stage0_config, stage1_config):
    """
    `anchor` is an AnchorResult (src/pipeline/anchor.py). Returns
    (ControlPoints, source_stack, source_corr_mask, reference_stack,
    reference_corr_mask) -- the descriptor stacks are None for a
    keypoint-fallback anchor (never built, since Stage 1 doesn't need them in
    that branch) and are returned here (rather than rebuilt again in
    fit_validate.py) purely as a performance carry-over matching what
    app/pipeline_runner.py already did in-process.
    """

    source_stack = source_corr_mask = None
    reference_stack = reference_corr_mask = None

    if anchor.anchor_source != "dense_correlation":
        stage1_result = points_from_keypoint_fallback(anchor.raw["keypoint_fallback_result"])
    else:
        source_stack, source_corr_mask = build_oriented_gradient_stack(
            source,
            mask,
            stage0_config["num_orientations"],
            stage0_config["gaussian_sigma"],
            stage0_config.get("mask_erosion_px", 5),
        )

        reference_stack, reference_corr_mask = build_oriented_gradient_stack(
            reference,
            mask,
            stage0_config["num_orientations"],
            stage0_config["gaussian_sigma"],
            stage0_config.get("mask_erosion_px", 5),
        )

        stage1_result = run_stage1(
            source_stack,
            source_corr_mask,
            reference_stack,
            reference_corr_mask,
            anchor.dy,
            anchor.dx,
            stage1_config,
        )

    all_points = stage1_result["points"]

    # write_correspondence_csv (src/coarse/export.py) uses "stage1_accepted"
    # as the column name (to disambiguate from Stage 2's "final_accepted");
    # Stage 1's own point dicts use "accepted" natively -- alias here once,
    # in the one place both the app and the CLI now go through.
    for p in all_points:
        p["stage1_accepted"] = p["accepted"]

    accepted_points = [p for p in all_points if p["accepted"]]

    control_points = ControlPoints(
        points=all_points,
        accepted_points=accepted_points,
        summary=stage1_result["summary"],
    )

    return control_points, source_stack, source_corr_mask, reference_stack, reference_corr_mask
