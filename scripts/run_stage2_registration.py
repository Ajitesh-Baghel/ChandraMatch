import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_stage0_coarse_search import PAIR_FILES, load_pair
from src.coarse.export import (
    robust_uint8,
    register_source,
    save_checkerboard,
    save_overlay,
    write_correspondence_csv,
    write_registered_geotiff,
)
from src.coarse.oriented_gradients import build_oriented_gradient_stack
from src.coarse.robust_fit import residual_consistency, fit_constrained_affine
from src.coarse.stage2_validation import (
    run_interior_roi_perturbation_validation,
    run_reduced_scale_perturbation_validation,
)

import rasterio


STAGE0_CONFIG_PATH = ROOT / "configs" / "stage0_coarse_search.json"
STAGE1_CONFIG_PATH = ROOT / "configs" / "stage1_local_control_points.json"
STAGE2_CONFIG_PATH = ROOT / "configs" / "stage2_registration.json"


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.pop("notes", None)

    return config


def parse_bool(value):
    return str(value).strip().lower() == "true"


def load_stage1_points(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    points = []

    for row in rows:
        points.append(
            {
                "reference_x": float(row["reference_x"]),
                "reference_y": float(row["reference_y"]),
                "source_x": float(row["source_x"]) if row["source_x"] else None,
                "source_y": float(row["source_y"]) if row["source_y"] else None,
                "residual_dy": float(row["residual_dy"]) if row["residual_dy"] else None,
                "residual_dx": float(row["residual_dx"]) if row["residual_dx"] else None,
                "residual_magnitude_px": float(row["residual_magnitude_px"])
                if row["residual_magnitude_px"]
                else None,
                "peak_score": float(row["peak_score"]) if row["peak_score"] else None,
                "ambiguity_ratio": float(row["ambiguity_ratio"])
                if row["ambiguity_ratio"]
                else None,
                "valid_fraction": float(row["valid_fraction"])
                if row["valid_fraction"]
                else None,
                "stage1_accepted": parse_bool(row["accepted"]),
            }
        )

    return points


def find_grid_cells(points, width, height, rows=8, cols=8):
    cells = set()

    for p in points:
        gx = min(cols - 1, max(0, int(p["reference_x"] / width * cols)))
        gy = min(rows - 1, max(0, int(p["reference_y"] / height * rows)))
        cells.add((gy, gx))

    return cells


def valid_grid_cells(mask, rows=8, cols=8):
    height, width = mask.shape
    cells = set()

    for gy in range(rows):
        y0 = int(round(gy * height / rows))
        y1 = int(round((gy + 1) * height / rows))

        for gx in range(cols):
            x0 = int(round(gx * width / cols))
            x1 = int(round((gx + 1) * width / cols))

            if np.any(mask[y0:y1, x0:x1]):
                cells.add((gy, gx))

    return cells


def main():
    parser = argparse.ArgumentParser(description="Stage 2 registration + export")
    parser.add_argument("--pair", required=True, choices=sorted(PAIR_FILES.keys()))
    args = parser.parse_args()

    pair_id = args.pair

    print("=" * 78)
    print(f"STAGE 2 -- REGISTRATION + EXPORT -- {pair_id.upper()}")
    print("=" * 78)

    stage1_csv_path = (
        ROOT
        / "results"
        / pair_id
        / "stage1_local_control_points"
        / f"{pair_id}_stage1_correspondences.csv"
    )

    if not stage1_csv_path.exists():
        raise SystemExit(
            f"No Stage 1 correspondences found at {stage1_csv_path}. "
            "Run scripts/run_stage1_local_control_points.py first."
        )

    stage0_result_path = (
        ROOT
        / "results"
        / pair_id
        / "stage0_coarse_search"
        / f"{pair_id}_stage0_metrics.json"
    )

    with open(stage0_result_path, "r", encoding="utf-8") as f:
        anchor_source = json.load(f).get("anchor_source", "dense_correlation")

    print(f"\nAnchor source (from Stage 0): {anchor_source}")

    stage0_config = load_json_config(STAGE0_CONFIG_PATH)
    stage1_config = load_json_config(STAGE1_CONFIG_PATH)
    stage2_config = load_json_config(STAGE2_CONFIG_PATH)

    print("\nStage 2 config:", STAGE2_CONFIG_PATH)
    print(json.dumps(stage2_config, indent=2))

    all_points = load_stage1_points(stage1_csv_path)
    accepted_points = [p for p in all_points if p["stage1_accepted"]]

    print(f"\nStage 1 points: {len(all_points)} total, {len(accepted_points)} accepted")

    if len(accepted_points) < 3:
        raise SystemExit(
            f"{pair_id}: only {len(accepted_points)} accepted Stage 1 points -- "
            "cannot fit an affine."
        )

    source_xy = np.asarray(
        [[p["source_x"], p["source_y"]] for p in accepted_points], dtype=np.float32
    )
    reference_xy = np.asarray(
        [[p["reference_x"], p["reference_y"]] for p in accepted_points], dtype=np.float32
    )

    # --------------------------------------------------------
    # Robust constrained affine fit
    # --------------------------------------------------------

    fit = fit_constrained_affine(source_xy, reference_xy, stage2_config)

    if fit is None:
        raise SystemExit(f"{pair_id}: robust affine fit failed.")

    print("\nFitted affine:")
    print(fit["matrix"])
    print("Params:", json.dumps(fit["params"], indent=2))
    print(f"RANSAC inliers: {fit['inlier_count']}/{fit['candidate_count']}")
    print("Shape-sane:", fit["sane"])

    # --------------------------------------------------------
    # Local residual consistency (against ALL accepted points, not
    # just the RANSAC inlier subset -- reports every point's fit)
    # --------------------------------------------------------

    residuals, is_outlier = residual_consistency(
        source_xy, reference_xy, fit["matrix"], stage2_config["max_fit_residual_px"]
    )

    for p, residual, outlier in zip(accepted_points, residuals, is_outlier):
        p["fit_residual_px"] = float(residual)
        p["fit_outlier"] = bool(outlier)
        p["final_accepted"] = bool(not outlier)

    for p in all_points:
        if not p["stage1_accepted"]:
            p["fit_residual_px"] = None
            p["fit_outlier"] = None
            p["final_accepted"] = False

    final_points = [p for p in accepted_points if p["final_accepted"]]
    outlier_count = int(np.sum(is_outlier))

    print(
        f"\nFit-residual outliers: {outlier_count}/{len(accepted_points)} "
        f"(cap {stage2_config['max_fit_residual_px']}px)"
    )
    print(f"Final accepted points: {len(final_points)}")

    # --------------------------------------------------------
    # Load raw pair + build descriptor stacks (needed for perturbation
    # validation's local refinement, same descriptor as Stage 0/1)
    # --------------------------------------------------------

    source, reference, mask = load_pair(pair_id)

    reference_stack, reference_corr_mask = build_oriented_gradient_stack(
        reference,
        mask,
        stage0_config["num_orientations"],
        stage0_config["gaussian_sigma"],
        stage0_config.get("mask_erosion_px", 5),
    )

    _, source_corr_mask = build_oriented_gradient_stack(
        source,
        mask,
        stage0_config["num_orientations"],
        stage0_config["gaussian_sigma"],
        stage0_config.get("mask_erosion_px", 5),
    )

    # --------------------------------------------------------
    # Interior-ROI perturbation validation
    # --------------------------------------------------------

    print("\nRunning interior-ROI perturbation validation...")

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

    for c in validation["controls"]:
        margin_str = (
            f"{c['interior_roi_margin_px']:.1f}px"
            if c["interior_roi_margin_px"] is not None
            else "n/a"
        )
        pixels_str = (
            f"{c['interior_roi_pixels']:,}" if c["interior_roi_pixels"] is not None else "n/a"
        )
        print(
            f"  {c['perturbation_id']:<18} "
            f"roi_margin={margin_str} "
            f"roi_pixels={pixels_str} "
            f"accepted={c['accepted_points']:<4} "
            f"response_rmse={c['response_rmse_px']} pass={c['pass']} "
            f"reasons={c['reasons']}"
        )

    print(f"\nAll perturbations pass: {validation['all_pass']}")

    # --------------------------------------------------------
    # Reduced-scale supplementary test, only if EVERY standard control
    # was untestable (empty interior ROI) -- not run when the standard
    # test could run at all, and never a substitute for it.
    # --------------------------------------------------------

    reduced_scale_validation = None

    all_untestable = bool(validation["controls"]) and all(
        c["interior_roi_pixels"] == 0 for c in validation["controls"]
    )

    if all_untestable:
        print(
            "\nStandard perturbation set is untestable on this pair's geometry -- "
            "running reduced-scale supplementary test..."
        )

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

        if not reduced_scale_validation["possible"]:
            print("  Not possible:", reduced_scale_validation["reason"])
        else:
            print(
                f"  scale_factor={reduced_scale_validation['scale_factor']:.3f} "
                f"max_safe_margin={reduced_scale_validation['max_safe_margin_px']}px"
            )

            for c in reduced_scale_validation["controls"]:
                print(
                    f"  {c['perturbation_id']:<18} "
                    f"roi_pixels={c['interior_roi_pixels']:,} "
                    f"accepted={c['accepted_points']:<4} "
                    f"response_rmse={c['response_rmse_px']} pass={c['pass']} "
                    f"reasons={c['reasons']}"
                )

            print(f"  Reduced-scale all pass: {reduced_scale_validation['all_pass']}")

    # --------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------

    reasons = []
    perturbation_validation_informational_only = False

    if not fit["sane"]:
        reasons.append("fitted affine failed shape-sanity check")

    outlier_rate = outlier_count / len(accepted_points) if accepted_points else 1.0

    if outlier_rate > 0.3:
        reasons.append(f"fit-residual outlier rate {outlier_rate:.2f} exceeds 0.30")

    if anchor_source != "dense_correlation":
        # Confirmed by direct testing (not assumed): re-verifying a
        # keypoint-fallback anchor with ANOTHER keypoint matcher under
        # large injected perturbation reproduces the documented
        # "identity bias" failure mode of learned matchers -- recovered
        # translation stayed near-zero (0.08-1.28px) across every tested
        # injected magnitude from 8px to 80px, i.e. it doesn't track the
        # perturbation AT ALL rather than tracking it badly. That's not
        # evidence of instability, it's evidence this specific
        # re-verification method can't be trusted for this anchor type
        # (the same matcher family can't independently check itself).
        # Falls back to judging the coarse fit quality alone, same as
        # v1 did before any perturbation test existed.
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
    else:
        if not validation["all_pass"]:
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
                    f"interior-ROI perturbation validation UNTESTABLE for "
                    f"{untestable} -- this pair's post-registration common overlap "
                    "has no region safely interior to the perturbation margins; this "
                    "is a limit of what can be validated, not a failed test"
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
                            f"(scale={reduced_scale_validation['scale_factor']:.3f}x "
                            "standard magnitude) -- supplementary evidence only, not "
                            "equivalent to the full-strength standard test"
                        )
                    else:
                        reasons.append(
                            "reduced-scale supplementary test also did not pass"
                        )
            elif failed:
                reasons.append(
                    f"interior-ROI perturbation validation FAILED for {failed} "
                    f"(response exceeded threshold or affine was insane)"
                )
                if untestable:
                    reasons.append(
                        f"also untestable (empty interior ROI) for {untestable}"
                    )

    verdict = "PASS" if not reasons else ("FAIL" if len(final_points) < 5 else "AMBIGUOUS")

    print("\nFINAL VERDICT:", verdict)

    if perturbation_validation_informational_only:
        print(
            "  (perturbation-robustness NOT independently validated for this "
            "anchor type -- keypoint-vs-keypoint re-verification exhibits "
            "identity bias under large injected motion, confirmed by direct "
            "testing; verdict is based on coarse-fit quality only)"
        )

    for r in reasons:
        print("  -", r)

    # --------------------------------------------------------
    # Export: registered product, previews, correspondence CSV, metrics
    # --------------------------------------------------------

    out_dir = ROOT / "results" / pair_id / "stage2_registration"
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(PAIR_FILES[pair_id]["dir"] / PAIR_FILES[pair_id]["reference"]) as ds:
        reference_profile = ds.profile.copy()
        source_nodata = None

    with rasterio.open(PAIR_FILES[pair_id]["dir"] / PAIR_FILES[pair_id]["source"]) as ds:
        source_nodata = ds.nodata

    source_valid = mask & np.isfinite(source)
    reference_valid = mask & np.isfinite(reference)

    registered, registered_valid = register_source(
        source, source_valid, reference_valid, fit["matrix"]
    )

    registered_path = out_dir / f"{pair_id}_registered.tif"
    nodata = write_registered_geotiff(
        registered_path, registered, registered_valid, reference_profile, source_nodata
    )

    visual_valid = registered_valid & reference_valid
    registered_preview = robust_uint8(registered, visual_valid)
    reference_preview = robust_uint8(reference, visual_valid)

    overlay_path = out_dir / f"{pair_id}_overlay.png"
    checkerboard_path = out_dir / f"{pair_id}_checkerboard.png"

    save_overlay(registered_preview, reference_preview, visual_valid, overlay_path)
    save_checkerboard(
        registered_preview, reference_preview, visual_valid, checkerboard_path
    )

    csv_path = out_dir / f"{pair_id}_stage2_correspondences.csv"
    write_correspondence_csv(all_points, csv_path)

    occupied_cells = find_grid_cells(final_points, mask.shape[1], mask.shape[0])
    valid_cells = valid_grid_cells(mask)

    metrics = {
        "pair_id": pair_id,
        "stage1_points_total": len(all_points),
        "stage1_points_accepted": len(accepted_points),
        "fit": {
            "affine": fit["matrix"].tolist(),
            "affine_parameters": fit["params"],
            "sane": fit["sane"],
            "ransac_inliers": fit["inlier_count"],
            "ransac_candidates": fit["candidate_count"],
            "ransac_inlier_ratio": fit["inlier_ratio"],
        },
        "residual_consistency": {
            "max_fit_residual_px": stage2_config["max_fit_residual_px"],
            "outlier_count": outlier_count,
            "outlier_rate": outlier_rate,
            "final_accepted_points": len(final_points),
        },
        "spatial_coverage": {
            "occupied_cells": len(occupied_cells),
            "valid_cells": len(valid_cells),
            "coverage_ratio": len(occupied_cells) / len(valid_cells)
            if valid_cells
            else 0.0,
        },
        "anchor_source": anchor_source,
        "interior_roi_perturbation_validation": validation,
        "reduced_scale_perturbation_validation": reduced_scale_validation,
        "perturbation_validation_informational_only": perturbation_validation_informational_only,
        "registered_product": {
            "filename": registered_path.name,
            "valid_pixels": int(registered_valid.sum()),
            "nodata_value": nodata,
        },
        "verdict": verdict,
        "verdict_reasons": reasons,
        "interpretation_notes": [
            "RANSAC reprojection statistics are internal self-consistency, not "
            "independent ground-truth accuracy.",
            "Interior-ROI perturbation response measures motion-following "
            "robustness inside a region guaranteed to stay valid under every "
            "tested perturbation; it is not independent absolute accuracy for "
            "the unperturbed pair.",
        ],
    }

    metrics_path = out_dir / f"{pair_id}_stage2_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nOutputs:")
    print(" ", registered_path)
    print(" ", overlay_path)
    print(" ", checkerboard_path)
    print(" ", csv_path)
    print(" ", metrics_path)

    print(f"\nSTAGE 2 COMPLETE -- {pair_id.upper()}")


if __name__ == "__main__":
    main()
