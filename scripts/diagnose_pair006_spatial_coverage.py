from pathlib import Path
import csv
import json

import cv2
import numpy as np
import pandas as pd
import rasterio


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pair_006"
    / "canonical"
)

MASK_PATH = (
    PAIR_DIR
    / "matcher_mask.tif"
)

INLIER_CSV = (
    ROOT
    / "results"
    / "pair_006"
    / "tiled_lightglue"
    / "pair006_tiled_lightglue_inliers.csv"
)

METRICS_PATH = (
    ROOT
    / "results"
    / "pair_006"
    / "tiled_lightglue"
    / "pair006_tiled_lightglue_metrics.json"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_006"
    / "spatial_diagnosis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Same grid definition used by frozen matcher.
GRID_ROWS = 8
GRID_COLS = 8

# More detailed along-track diagnosis.
ALONG_TRACK_BINS = 12


# ============================================================
# HELPERS
# ============================================================


def load_mask():
    with rasterio.open(MASK_PATH) as ds:
        mask = ds.read(1) > 0
        transform = ds.transform

    return mask, transform


def load_inliers():
    df = pd.read_csv(INLIER_CSV)

    required = [
        "source_x",
        "source_y",
        "reference_x",
        "reference_y",
        "ransac_residual_px",
    ]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"Missing required CSV column: {col}"
            )

    return df


def cell_index(value, size, cells):
    idx = int(
        np.floor(
            float(value)
            / float(size)
            * cells
        )
    )

    return max(
        0,
        min(
            cells - 1,
            idx,
        ),
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 80)
    print("PAIR 006 — SPATIAL COVERAGE DIAGNOSIS")
    print("=" * 80)

    mask, transform = load_mask()
    df = load_inliers()

    height, width = mask.shape

    print("\nCanonical shape:")
    print(f"  {height} x {width}")

    print("\nMatcher support:")
    print(
        f"  {int(mask.sum()):,} pixels "
        f"({mask.mean():.6f})"
    )

    print("\nSaved RANSAC inliers:")
    print(f"  {len(df):,}")

    # --------------------------------------------------------
    # MATCHER SUPPORT BOUNDING BOX
    # --------------------------------------------------------

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError(
            "Matcher mask contains no valid pixels."
        )

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())

    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1

    print("\nMatcher-support bounding box:")
    print(
        f"  X: {min_x} -> {max_x} "
        f"({bbox_width} px)"
    )

    print(
        f"  Y: {min_y} -> {max_y} "
        f"({bbox_height} px)"
    )

    print(
        f"  Physical width : "
        f"{bbox_width * 5 / 1000:.3f} km"
    )

    print(
        f"  Physical height: "
        f"{bbox_height * 5 / 1000:.3f} km"
    )

    # ========================================================
    # 1. ALONG-TRACK BIN DIAGNOSIS
    # ========================================================

    print("\n" + "=" * 80)
    print("ALONG-TRACK DISTRIBUTION")
    print("=" * 80)

    edges = np.linspace(
        min_x,
        max_x + 1,
        ALONG_TRACK_BINS + 1,
    )

    bin_rows = []

    for i in range(ALONG_TRACK_BINS):

        x0 = int(np.floor(edges[i]))
        x1 = int(np.ceil(edges[i + 1]))

        if i == ALONG_TRACK_BINS - 1:
            x1 = max_x + 1

        support = mask[:, x0:x1]

        support_pixels = int(
            support.sum()
        )

        src_sel = (
            (df["source_x"] >= x0)
            &
            (df["source_x"] < x1)
        )

        ref_sel = (
            (df["reference_x"] >= x0)
            &
            (df["reference_x"] < x1)
        )

        src_count = int(
            src_sel.sum()
        )

        ref_count = int(
            ref_sel.sum()
        )

        if support_pixels > 0:
            src_density = (
                src_count
                / support_pixels
                * 100000.0
            )
        else:
            src_density = 0.0

        if src_count > 0:
            median_residual = float(
                df.loc[
                    src_sel,
                    "ransac_residual_px"
                ].median()
            )
        else:
            median_residual = None

        row = {
            "bin": i + 1,
            "x_start": x0,
            "x_end": x1 - 1,
            "support_pixels": support_pixels,
            "source_inliers": src_count,
            "reference_inliers": ref_count,
            "source_inliers_per_100k_support_px":
                src_density,
            "median_ransac_residual_px":
                median_residual,
        }

        bin_rows.append(row)

        print(
            f"Bin {i + 1:02d} "
            f"x={x0:4d}->{x1 - 1:4d} | "
            f"support={support_pixels:8,d} | "
            f"src inliers={src_count:4d} | "
            f"ref inliers={ref_count:4d} | "
            f"density={src_density:8.3f}"
        )

    pd.DataFrame(
        bin_rows
    ).to_csv(
        OUT_DIR
        / "pair006_along_track_bins.csv",
        index=False,
    )

    # ========================================================
    # 2. EXACT 8 × 8 COVERAGE GRID DIAGNOSIS
    # ========================================================

    print("\n" + "=" * 80)
    print("8 x 8 GRID DISTRIBUTION")
    print("=" * 80)

    cell_h = height / GRID_ROWS
    cell_w = width / GRID_COLS

    cell_rows = []

    valid_cells = 0
    occupied_cells = 0

    for gy in range(GRID_ROWS):

        y0 = int(
            np.floor(
                gy * cell_h
            )
        )

        y1 = int(
            np.floor(
                (gy + 1) * cell_h
            )
        )

        if gy == GRID_ROWS - 1:
            y1 = height

        for gx in range(GRID_COLS):

            x0 = int(
                np.floor(
                    gx * cell_w
                )
            )

            x1 = int(
                np.floor(
                    (gx + 1) * cell_w
                )
            )

            if gx == GRID_COLS - 1:
                x1 = width

            support_pixels = int(
                mask[
                    y0:y1,
                    x0:x1,
                ].sum()
            )

            valid = (
                support_pixels > 0
            )

            if valid:
                valid_cells += 1

            inlier_selection = (
                (df["source_x"] >= x0)
                &
                (df["source_x"] < x1)
                &
                (df["source_y"] >= y0)
                &
                (df["source_y"] < y1)
            )

            inlier_count = int(
                inlier_selection.sum()
            )

            occupied = (
                valid
                and
                inlier_count > 0
            )

            if occupied:
                occupied_cells += 1

            if support_pixels > 0:
                density = (
                    inlier_count
                    / support_pixels
                    * 100000.0
                )
            else:
                density = 0.0

            cell_rows.append(
                {
                    "grid_row": gy,
                    "grid_col": gx,
                    "x_start": x0,
                    "x_end": x1 - 1,
                    "y_start": y0,
                    "y_end": y1 - 1,
                    "support_pixels":
                        support_pixels,
                    "valid_cell":
                        bool(valid),
                    "inlier_count":
                        inlier_count,
                    "occupied":
                        bool(occupied),
                    "inliers_per_100k_support_px":
                        density,
                }
            )

            if valid:
                state = (
                    "MATCHED"
                    if occupied
                    else "EMPTY"
                )

                print(
                    f"Cell ({gy},{gx}) "
                    f"support={support_pixels:8,d} "
                    f"inliers={inlier_count:4d} "
                    f"{state}"
                )

    coverage_ratio = (
        occupied_cells
        / valid_cells
        if valid_cells
        else 0
    )

    print("\nCoverage recomputed from source inliers:")
    print(
        f"  occupied cells : "
        f"{occupied_cells}"
    )

    print(
        f"  valid cells    : "
        f"{valid_cells}"
    )

    print(
        f"  coverage       : "
        f"{occupied_cells}/{valid_cells} "
        f"= {coverage_ratio:.6f}"
    )

    pd.DataFrame(
        cell_rows
    ).to_csv(
        OUT_DIR
        / "pair006_8x8_coverage_cells.csv",
        index=False,
    )

    # ========================================================
    # 3. MATCH EXTENT
    # ========================================================

    print("\n" + "=" * 80)
    print("INLIER EXTENT")
    print("=" * 80)

    src_min_x = float(
        df["source_x"].min()
    )

    src_max_x = float(
        df["source_x"].max()
    )

    src_min_y = float(
        df["source_y"].min()
    )

    src_max_y = float(
        df["source_y"].max()
    )

    print(
        f"Source X: "
        f"{src_min_x:.2f} -> "
        f"{src_max_x:.2f}"
    )

    print(
        f"Source Y: "
        f"{src_min_y:.2f} -> "
        f"{src_max_y:.2f}"
    )

    support_span = (
        max_x - min_x
    )

    inlier_span = (
        src_max_x - src_min_x
    )

    span_fraction = (
        inlier_span
        / support_span
        if support_span > 0
        else 0
    )

    print(
        f"\nAlong-track support span covered "
        f"by inliers: "
        f"{span_fraction:.6f}"
    )

    # ========================================================
    # 4. VISUAL DIAGNOSTIC
    # ========================================================

    preview = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # Matcher support = dark gray.
    preview[mask] = (
        70,
        70,
        70,
    )

    # Draw along-track divisions.
    for edge in edges:

        x = int(
            round(edge)
        )

        x = max(
            0,
            min(
                width - 1,
                x,
            ),
        )

        cv2.line(
            preview,
            (x, 0),
            (x, height - 1),
            (255, 255, 255),
            1,
        )

    # Draw source inliers.
    for _, row in df.iterrows():

        x = int(
            round(
                row["source_x"]
            )
        )

        y = int(
            round(
                row["source_y"]
            )
        )

        if (
            0 <= x < width
            and
            0 <= y < height
        ):
            cv2.circle(
                preview,
                (x, y),
                3,
                (0, 255, 0),
                -1,
            )

    diagnostic_path = (
        OUT_DIR
        / "pair006_spatial_diagnosis.png"
    )

    cv2.imwrite(
        str(diagnostic_path),
        preview,
    )

    # ========================================================
    # 5. SUMMARY JSON
    # ========================================================

    summary = {
        "pair_id": "pair_006",

        "canonical_width":
            width,

        "canonical_height":
            height,

        "matcher_support_pixels":
            int(mask.sum()),

        "ransac_inliers":
            int(len(df)),

        "matcher_bbox": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "width": bbox_width,
            "height": bbox_height,
        },

        "source_inlier_extent": {
            "min_x": src_min_x,
            "max_x": src_max_x,
            "min_y": src_min_y,
            "max_y": src_max_y,
        },

        "along_track_support_span_fraction":
            span_fraction,

        "grid": {
            "rows": GRID_ROWS,
            "cols": GRID_COLS,
            "valid_cells": valid_cells,
            "occupied_cells":
                occupied_cells,
            "coverage_ratio":
                coverage_ratio,
        },

        "along_track_bins":
            bin_rows,

        "interpretation_note":
            (
                "This script diagnoses spatial "
                "distribution of the already-frozen "
                "Pair006 result. It does not rerun "
                "or modify the matcher."
            ),
    }

    with open(
        OUT_DIR
        / "pair006_spatial_diagnosis.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("OUTPUTS")
    print("=" * 80)

    print(
        OUT_DIR
        / "pair006_along_track_bins.csv"
    )

    print(
        OUT_DIR
        / "pair006_8x8_coverage_cells.csv"
    )

    print(
        OUT_DIR
        / "pair006_spatial_diagnosis.json"
    )

    print(
        diagnostic_path
    )

    print(
        "\nNo matcher parameters were changed."
    )

    print(
        "PAIR 006 spatial diagnosis complete."
    )


if __name__ == "__main__":
    main()