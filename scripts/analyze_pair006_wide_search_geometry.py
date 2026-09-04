from pathlib import Path
import json
import math

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

CANDIDATE_CSV = (
    ROOT
    / "results"
    / "pair_006"
    / "wide_search_diagnostic"
    / "pair006_wide_search_distinct_candidates.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_006"
    / "wide_search_diagnostic"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Split the actual matcher-support span along-track.
ALONG_TRACK_BINS = 12

# Same RANSAC threshold as frozen real-pair matcher.
RANSAC_THRESHOLD_PX = 3.0


# ============================================================
# AFFINE HELPERS
# ============================================================

def affine_points(matrix, points):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    ones = np.ones(
        (len(points), 1),
        dtype=np.float64,
    )

    homogeneous = np.hstack(
        [points, ones]
    )

    return homogeneous @ matrix.T


def decompose_affine(matrix):
    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    tx = float(matrix[0, 2])

    c = float(matrix[1, 0])
    d = float(matrix[1, 1])
    ty = float(matrix[1, 2])

    scale_x = math.sqrt(
        a * a + c * c
    )

    scale_y = math.sqrt(
        b * b + d * d
    )

    rotation_deg = math.degrees(
        math.atan2(c, a)
    )

    determinant = (
        a * d
        - b * c
    )

    axis_dot = (
        a * b
        + c * d
    )

    return {
        "translation_x_px": tx,
        "translation_y_px": ty,
        "translation_magnitude_px":
            math.hypot(tx, ty),
        "rotation_deg": rotation_deg,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "determinant": determinant,
        "axis_dot": axis_dot,
    }


def fit_local_affine(
    source,
    reference,
):
    if len(source) < 4:
        return None

    matrix, inlier_mask = cv2.estimateAffine2D(
        source.astype(np.float32),
        reference.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX,
        maxIters=20000,
        confidence=0.999,
        refineIters=100,
    )

    if (
        matrix is None
        or inlier_mask is None
    ):
        return None

    inlier_mask = (
        inlier_mask.ravel() > 0
    )

    if inlier_mask.sum() < 4:
        return None

    source_inliers = (
        source[inlier_mask]
    )

    reference_inliers = (
        reference[inlier_mask]
    )

    predicted = affine_points(
        matrix,
        source_inliers,
    )

    residuals = np.linalg.norm(
        predicted - reference_inliers,
        axis=1,
    )

    return {
        "matrix": matrix,
        "inliers": int(
            inlier_mask.sum()
        ),
        "inlier_ratio": float(
            inlier_mask.mean()
        ),
        "rmse_px": float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        ),
        "median_residual_px": float(
            np.median(residuals)
        ),
        "mean_residual_px": float(
            np.mean(residuals)
        ),
        "parameters":
            decompose_affine(matrix),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 88)
    print(
        "PAIR 006 — WIDE-SEARCH CANDIDATE GEOMETRY DIAGNOSIS"
    )
    print("=" * 88)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        CANDIDATE_CSV
    )

    required_columns = [
        "source_x",
        "source_y",
        "reference_x",
        "reference_y",
        "dx_px",
        "dy_px",
        "displacement_px",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise RuntimeError(
                f"Missing required column: {column}"
            )

    with rasterio.open(
        MASK_PATH
    ) as ds:
        matcher_mask = (
            ds.read(1) > 0
        )

    height, width = (
        matcher_mask.shape
    )

    ys, xs = np.where(
        matcher_mask
    )

    if len(xs) == 0:
        raise RuntimeError(
            "Matcher mask is empty."
        )

    min_x = int(xs.min())
    max_x = int(xs.max())

    print()
    print(
        f"Canonical shape          : "
        f"{height} x {width}"
    )

    print(
        f"Matcher support X        : "
        f"{min_x} -> {max_x}"
    )

    print(
        f"Matcher support span     : "
        f"{max_x - min_x + 1} px"
    )

    print(
        f"Distinct pre-RANSAC cand.: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Global displacement statistics
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("GLOBAL PRE-RANSAC DISPLACEMENT")
    print("=" * 88)

    print(
        f"Median dx               : "
        f"{df['dx_px'].median():.3f} px"
    )

    print(
        f"Median dy               : "
        f"{df['dy_px'].median():.3f} px"
    )

    print(
        f"Median displacement     : "
        f"{df['displacement_px'].median():.3f} px"
    )

    print(
        f"90th percentile disp.   : "
        f"{df['displacement_px'].quantile(0.90):.3f} px"
    )

    print(
        f"Maximum displacement    : "
        f"{df['displacement_px'].max():.3f} px"
    )

    # --------------------------------------------------------
    # Along-track bins
    # --------------------------------------------------------

    edges = np.linspace(
        min_x,
        max_x + 1,
        ALONG_TRACK_BINS + 1,
    )

    bin_results = []

    print()
    print("=" * 88)
    print("LOCAL AFFINE MODELS BY ALONG-TRACK REGION")
    print("=" * 88)

    for i in range(
        ALONG_TRACK_BINS
    ):
        x0 = int(
            np.floor(edges[i])
        )

        x1 = int(
            np.ceil(edges[i + 1])
        )

        if i == ALONG_TRACK_BINS - 1:
            x1 = max_x + 1

        selection = (
            (df["source_x"] >= x0)
            &
            (df["source_x"] < x1)
        )

        sub = (
            df.loc[selection]
            .copy()
        )

        candidate_count = len(sub)

        support_pixels = int(
            matcher_mask[
                :,
                x0:x1,
            ].sum()
        )

        if support_pixels > 0:
            candidate_density = (
                candidate_count
                / support_pixels
                * 100000.0
            )
        else:
            candidate_density = 0.0

        row = {
            "bin": i + 1,
            "x_start": x0,
            "x_end": x1 - 1,
            "support_pixels":
                support_pixels,
            "candidate_count":
                candidate_count,
            "candidate_density_per_100k_support_px":
                candidate_density,
        }

        if candidate_count > 0:
            row.update(
                {
                    "median_dx_px": float(
                        sub["dx_px"].median()
                    ),
                    "median_dy_px": float(
                        sub["dy_px"].median()
                    ),
                    "median_displacement_px": float(
                        sub[
                            "displacement_px"
                        ].median()
                    ),
                }
            )
        else:
            row.update(
                {
                    "median_dx_px": None,
                    "median_dy_px": None,
                    "median_displacement_px":
                        None,
                }
            )

        # ----------------------------------------------------
        # Local affine RANSAC
        # ----------------------------------------------------

        if candidate_count >= 4:
            source_points = (
                sub[
                    [
                        "source_x",
                        "source_y",
                    ]
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            reference_points = (
                sub[
                    [
                        "reference_x",
                        "reference_y",
                    ]
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            local = fit_local_affine(
                source_points,
                reference_points,
            )
        else:
            local = None

        if local is not None:
            p = local["parameters"]

            row.update(
                {
                    "local_ransac_inliers":
                        local["inliers"],
                    "local_inlier_ratio":
                        local["inlier_ratio"],
                    "local_rmse_px":
                        local["rmse_px"],
                    "local_median_residual_px":
                        local[
                            "median_residual_px"
                        ],
                    "local_mean_residual_px":
                        local[
                            "mean_residual_px"
                        ],
                    "translation_x_px":
                        p[
                            "translation_x_px"
                        ],
                    "translation_y_px":
                        p[
                            "translation_y_px"
                        ],
                    "translation_magnitude_px":
                        p[
                            "translation_magnitude_px"
                        ],
                    "rotation_deg":
                        p["rotation_deg"],
                    "scale_x":
                        p["scale_x"],
                    "scale_y":
                        p["scale_y"],
                    "determinant":
                        p["determinant"],
                    "axis_dot":
                        p["axis_dot"],
                }
            )

            print(
                f"Bin {i + 1:02d} "
                f"x={x0:4d}->{x1 - 1:4d} | "
                f"support={support_pixels:8,d} | "
                f"cand={candidate_count:4d} | "
                f"local={local['inliers']:4d} "
                f"({local['inlier_ratio']:.3f}) | "
                f"tx={p['translation_x_px']:8.2f} "
                f"ty={p['translation_y_px']:8.2f} | "
                f"rot={p['rotation_deg']:7.3f}° | "
                f"sx={p['scale_x']:.5f} "
                f"sy={p['scale_y']:.5f} | "
                f"rmse={local['rmse_px']:6.3f}"
            )

        else:
            row.update(
                {
                    "local_ransac_inliers": 0,
                    "local_inlier_ratio": 0.0,
                    "local_rmse_px": None,
                    "local_median_residual_px":
                        None,
                    "local_mean_residual_px":
                        None,
                    "translation_x_px": None,
                    "translation_y_px": None,
                    "translation_magnitude_px":
                        None,
                    "rotation_deg": None,
                    "scale_x": None,
                    "scale_y": None,
                    "determinant": None,
                    "axis_dot": None,
                }
            )

            print(
                f"Bin {i + 1:02d} "
                f"x={x0:4d}->{x1 - 1:4d} | "
                f"support={support_pixels:8,d} | "
                f"cand={candidate_count:4d} | "
                f"NO LOCAL MODEL"
            )

        bin_results.append(row)

    # --------------------------------------------------------
    # Save bin table
    # --------------------------------------------------------

    bin_df = pd.DataFrame(
        bin_results
    )

    bin_csv_path = (
        OUT_DIR
        / "pair006_wide_search_geometry_bins.csv"
    )

    bin_df.to_csv(
        bin_csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # Diagnostic image
    # --------------------------------------------------------

    image = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    # Gray = matcher-valid terrain
    image[matcher_mask] = (
        55,
        55,
        55,
    )

    # Along-track bin boundaries
    for edge in edges:
        x = int(round(edge))

        x = max(
            0,
            min(
                width - 1,
                x,
            ),
        )

        cv2.line(
            image,
            (x, 0),
            (x, height - 1),
            (255, 255, 255),
            1,
        )

    # All distinct pre-RANSAC candidates.
    #
    # Yellow = source candidate location
    # Green line = displacement source -> reference
    for _, row in df.iterrows():
        sx = int(
            round(row["source_x"])
        )

        sy = int(
            round(row["source_y"])
        )

        rx = int(
            round(row["reference_x"])
        )

        ry = int(
            round(row["reference_y"])
        )

        if (
            0 <= sx < width
            and
            0 <= sy < height
        ):
            cv2.circle(
                image,
                (sx, sy),
                2,
                (0, 255, 255),
                -1,
            )

        if (
            0 <= sx < width
            and
            0 <= sy < height
            and
            0 <= rx < width
            and
            0 <= ry < height
        ):
            cv2.line(
                image,
                (sx, sy),
                (rx, ry),
                (0, 180, 0),
                1,
            )

    image_path = (
        OUT_DIR
        / "pair006_wide_search_geometry.png"
    )

    cv2.imwrite(
        str(image_path),
        image,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    bins_with_candidates = sum(
        row["candidate_count"] > 0
        for row in bin_results
    )

    bins_with_local_models = sum(
        row["local_ransac_inliers"] >= 4
        for row in bin_results
    )

    total_local_inliers = sum(
        row["local_ransac_inliers"]
        for row in bin_results
    )

    summary = {
        "pair_id":
            "pair_006",

        "analysis":
            "wide_search_pre_ransac_candidate_geometry",

        "distinct_candidates":
            int(len(df)),

        "canonical_shape": {
            "height": height,
            "width": width,
        },

        "matcher_support_x": {
            "min_x": min_x,
            "max_x": max_x,
            "span_px":
                max_x - min_x + 1,
        },

        "global_candidate_displacement": {
            "median_dx_px": float(
                df["dx_px"].median()
            ),
            "median_dy_px": float(
                df["dy_px"].median()
            ),
            "median_magnitude_px": float(
                df[
                    "displacement_px"
                ].median()
            ),
            "p90_magnitude_px": float(
                df[
                    "displacement_px"
                ].quantile(0.90)
            ),
        },

        "along_track_bins":
            ALONG_TRACK_BINS,

        "bins_with_candidates":
            int(
                bins_with_candidates
            ),

        "bins_with_local_affine":
            int(
                bins_with_local_models
            ),

        "sum_of_local_ransac_inliers":
            int(
                total_local_inliers
            ),

        "bins":
            bin_results,

        "note":
            (
                "Diagnostic analysis of the "
                "1004 frozen Pair006 "
                "pre-RANSAC correspondences. "
                "Local RANSAC residuals are "
                "self-consistency metrics and "
                "are not independent registration "
                "ground truth."
            ),
    }

    json_path = (
        OUT_DIR
        / "pair006_wide_search_geometry.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)

    print(
        "Bins containing candidates :",
        f"{bins_with_candidates}/"
        f"{ALONG_TRACK_BINS}",
    )

    print(
        "Bins with local affine     :",
        f"{bins_with_local_models}/"
        f"{ALONG_TRACK_BINS}",
    )

    print(
        "Sum local RANSAC inliers   :",
        total_local_inliers,
    )

    print()
    print("Outputs:")
    print(bin_csv_path)
    print(json_path)
    print(image_path)

    print()
    print(
        "PAIR 006 wide-search geometry diagnosis complete."
    )


if __name__ == "__main__":
    main()
