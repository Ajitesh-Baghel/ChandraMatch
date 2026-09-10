import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_stage0_coarse_search import PAIR_FILES, load_pair
from src.coarse.local_control_points import points_from_keypoint_fallback, run_stage1
from src.coarse.oriented_gradients import build_oriented_gradient_stack


STAGE0_CONFIG_PATH = ROOT / "configs" / "stage0_coarse_search.json"
STAGE1_CONFIG_PATH = ROOT / "configs" / "stage1_local_control_points.json"


def load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.pop("notes", None)

    return config


def write_correspondences_csv(points, path):
    fields = [
        "reference_x",
        "reference_y",
        "source_x",
        "source_y",
        "residual_dy",
        "residual_dx",
        "residual_magnitude_px",
        "peak_score",
        "ambiguity_ratio",
        "valid_fraction",
        "accepted",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for p in points:
            writer.writerow({field: p.get(field) for field in fields})


def preview_png(reference_mask, points, out_path):
    height, width = reference_mask.shape

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :, 2] = reference_mask.astype(np.uint8) * 120

    for p in points:
        color = (0, 255, 0) if p["accepted"] else (0, 0, 255)

        cv2.circle(
            canvas,
            (int(round(p["reference_x"])), int(round(p["reference_y"]))),
            4,
            color,
            -1,
        )

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 uniform local control points"
    )
    parser.add_argument("--pair", required=True, choices=sorted(PAIR_FILES.keys()))
    args = parser.parse_args()

    pair_id = args.pair

    print("=" * 78)
    print(f"STAGE 1 -- UNIFORM LOCAL CONTROL POINTS -- {pair_id.upper()}")
    print("=" * 78)

    stage0_result_path = (
        ROOT
        / "results"
        / pair_id
        / "stage0_coarse_search"
        / f"{pair_id}_stage0_metrics.json"
    )

    if not stage0_result_path.exists():
        raise SystemExit(
            f"No Stage 0 result found at {stage0_result_path}. "
            "Run scripts/run_stage0_coarse_search.py first."
        )

    with open(stage0_result_path, "r", encoding="utf-8") as f:
        stage0_result = json.load(f)

    if stage0_result["verdict"] != "PASS":
        raise SystemExit(
            f"{pair_id}: Stage 0 verdict is {stage0_result['verdict']!r}, not PASS "
            "-- Stage 1 only runs on a trustworthy global anchor. Reasons: "
            f"{stage0_result['verdict_reasons']}"
        )

    global_dy = stage0_result["best"]["dy"]
    global_dx = stage0_result["best"]["dx"]

    print(f"\nStage 0 anchor: dy={global_dy} dx={global_dx} (verdict PASS)")

    stage0_config = load_json_config(STAGE0_CONFIG_PATH)
    stage1_config = load_json_config(STAGE1_CONFIG_PATH)

    print("\nStage 1 config:", STAGE1_CONFIG_PATH)
    print(json.dumps(stage1_config, indent=2))

    source, reference, mask = load_pair(pair_id)

    anchor_source = stage0_result.get("anchor_source", "dense_correlation")

    if anchor_source != "dense_correlation":
        print(
            f"\nStage 0 anchor came from {anchor_source!r}, not dense correlation -- "
            "the dense oriented-gradient descriptor is known unreliable for this pair "
            "(that's why Stage 0 needed a fallback in the first place), so Stage 1 "
            "reuses the fallback matcher's own RANSAC-inlier correspondences instead "
            "of re-running a dense local search that would also fail."
        )

        result = points_from_keypoint_fallback(stage0_result["keypoint_fallback_result"])
        reference_corr_mask = mask
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

        result = run_stage1(
            source_stack,
            source_corr_mask,
            reference_stack,
            reference_corr_mask,
            global_dy,
            global_dx,
            stage1_config,
        )

    summary = result["summary"]

    print("\nSummary:")
    print(f"  grid points     : {summary['grid_points']}")
    print(f"  accepted        : {summary['accepted_count']}")
    print(f"  rejected        : {summary['rejected_count']}")
    print(f"  residual mean   : {summary['residual_mean_px']}")
    print(f"  residual median : {summary['residual_median_px']}")
    print(f"  residual max    : {summary['residual_max_px']}")

    rejection_counts = {}

    for p in result["points"]:
        if p["accepted"]:
            continue

        for reason in p["reject_reasons"]:
            key = reason.split("=")[0]
            rejection_counts[key] = rejection_counts.get(key, 0) + 1

    print("\nRejection reasons (by category):")

    for key, count in sorted(rejection_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<20} {count}")

    out_dir = ROOT / "results" / pair_id / "stage1_local_control_points"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"{pair_id}_stage1_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    csv_path = out_dir / f"{pair_id}_stage1_correspondences.csv"
    write_correspondences_csv(result["points"], csv_path)

    preview_path = out_dir / f"{pair_id}_stage1_preview.png"
    preview_png(reference_corr_mask, result["points"], preview_path)

    print("\nOutputs:")
    print(" ", metrics_path)
    print(" ", csv_path)
    print(" ", preview_path)

    print(f"\nSTAGE 1 COMPLETE -- {pair_id.upper()}")


if __name__ == "__main__":
    main()
