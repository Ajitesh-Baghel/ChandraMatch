import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "pair_001"

OUT_DIR = ROOT / "results" / "benchmarks"
FIGURE_DIR = ROOT / "results" / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND EXPERIMENT METRICS
# ============================================================

records = []

for metrics_file in RESULTS_DIR.glob("*/metrics.json"):

    with open(metrics_file, "r") as f:
        data = json.load(f)

    record = {
        "experiment": metrics_file.parent.name,
        "pair": data.get("pair"),
        "matcher": data.get("matcher"),

        "candidate_matches":
            data.get("candidate_matches"),

        "inlier_matches":
            data.get("inlier_matches"),

        "inlier_ratio":
            data.get("inlier_ratio"),

        "reprojection_rmse_px":
            data.get(
                "ransac_reprojection_rmse_px"
            ),

        "reprojection_median_px":
            data.get(
                "ransac_reprojection_median_error_px"
            ),

        "spatial_coverage":
            data.get("spatial_coverage"),

        "runtime_seconds":
            data.get("runtime_seconds"),
    }

    records.append(record)


if not records:
    raise RuntimeError(
        "No metrics.json files found."
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# Put matchers in predictable order

preferred_order = [
    "SIFT",
    "SuperPoint + LightGlue",
    "LoFTR"
]


df["_order"] = df["matcher"].apply(
    lambda x: (
        preferred_order.index(x)
        if x in preferred_order
        else 999
    )
)

df = (
    df
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)


# ============================================================
# SAVE MASTER CSV
# ============================================================

csv_path = (
    OUT_DIR
    / "all_experiments.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SAVE MARKDOWN TABLE
# ============================================================

markdown_path = (
    OUT_DIR
    / "all_experiments.md"
)

with open(
    markdown_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "# ChandraMatch Benchmark Summary\n\n"
    )

    f.write(
        df.to_markdown(
            index=False
        )
    )

    f.write("\n")


# ============================================================
# MATCH COUNT CHART
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    df["matcher"],
    df["inlier_matches"]
)

plt.ylabel(
    "RANSAC Inlier Matches"
)

plt.xlabel(
    "Matcher"
)

plt.title(
    "ChandraMatch Pair 001 — Valid Correspondences"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "pair001_inlier_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RMSE CHART
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    df["matcher"],
    df["reprojection_rmse_px"]
)

plt.ylabel(
    "RANSAC Reprojection RMSE (px)"
)

plt.xlabel(
    "Matcher"
)

plt.title(
    "ChandraMatch Pair 001 — Geometric Consistency"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "pair001_rmse_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RUNTIME CHART
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    df["matcher"],
    df["runtime_seconds"]
)

plt.ylabel(
    "Runtime (seconds)"
)

plt.xlabel(
    "Matcher"
)

plt.title(
    "ChandraMatch Pair 001 — Runtime"
)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "pair001_runtime_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("=" * 60)
print("CHANDRAMATCH BENCHMARK SUMMARY")
print("=" * 60)

print(
    df.to_string(index=False)
)

print("\nSaved:")
print(csv_path)
print(markdown_path)

print("\nFigures:")

for path in sorted(
    FIGURE_DIR.glob("pair001_*.png")
):
    print("-", path)