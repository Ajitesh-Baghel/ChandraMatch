"""
Regression harness for the unified pipeline (src/pipeline/run.py::register()),
per the SIH26166 unification pass, §8.5.

This is a NEW script, not a rewrite of any existing scripts/pair00X_*.py --
those are frozen v1 experiment evidence (tiled-LightGlue benchmarks etc.) and
are left untouched per the "never overwrite frozen evidence" rule. It is also
deliberately separate from scripts/run_stage{0,1,2}_coarse_search.py (kept
as-is for now) so this pass's diff is against a real, unmodified baseline.

Input-data note (read before trusting numbers from this script): the OLD
per-pair scripts load pre-built, pair-specific "canonical" files that were
never produced by a single generic function -- three different, mutually
inconsistent naming/masking conventions exist on disk today (pair_002/003's
`valid_mask.tif`-only scheme, pair_004's un-eroded `common_valid_mask.tif`,
and pair_006/007's pre-eroded `science_mask.tif`+`matcher_mask.tif` split).
register() always canonicalizes through build_canonical_pair(), which
standardizes on the pair_006/007 (science/matcher split, always-eroded)
convention -- so for a fair regression comparison this script feeds register()
the best available NEAR-RAW single-band input for each pair (an ISIS-mapped
LRO NAC cube/tif for 004/006/007, or the existing sensor-specific derived
GeoTIFF for 002/003, since neither OHRC nor IIRS raw-format ingestion is
built yet) rather than the old scripts' own already-pair-canonicalized files.
Where this changes Pair004's result specifically, that is EXPECTED and
already documented in project memory (build_canonical_pair's erosion differs
from Pair004's original un-eroded mask) -- reported explicitly below, not
silently accepted.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.config import load_pipeline_config
from src.pipeline.run import register

REGRESSION_OUT_ROOT = ROOT / "results" / "unified_regression"

# (source_path, reference_path, source_sensor_override, reference_sensor_override)
# Overrides are supplied where the near-raw file's name/format wouldn't
# auto-detect confidently on its own (e.g. an ISIS .cub with no PDS label,
# or an already-cropped derived .tif whose filename lacks the ch2_* prefix).
PAIR_INPUTS = {
    "pair_004": (
        ROOT / "data" / "processed" / "pair_004" / "lro_nac_isis" / "M185196277LE_map_5m.cub",
        ROOT / "data" / "raw" / "tmc2" / "2023_10_25" / "data" / "derived" / "20231025"
        / "ch2_tmc_ndn_20231025T1956513800_d_oth_d18.tif",
        "LRO NAC",
        None,
    ),
    "pair_006": (
        ROOT / "data" / "processed" / "pair_006" / "lro_nac_isis" / "M141704260RE_map_5m.tif",
        ROOT / "data" / "raw" / "tmc2" / "2023_10_25" / "data" / "derived" / "20231025"
        / "ch2_tmc_ndn_20231025T1956513800_d_oth_d18.tif",
        "LRO NAC",
        None,
    ),
    "pair_007": (
        ROOT / "data" / "processed" / "pair_007" / "lro_nac_isis" / "M185210693RE_map_5m.tif",
        ROOT / "data" / "raw" / "tmc2" / "2023_10_25" / "data" / "derived" / "20231025"
        / "ch2_tmc_ndn_20231025T1956513800_d_oth_d18.tif",
        "LRO NAC",
        None,
    ),
    "pair_002": (
        ROOT / "data" / "processed" / "pairs" / "pair_002" / "source_ohrc_on_tmc_grid.tif",
        ROOT / "data" / "raw" / "tmc2" / "2023_10_25" / "data" / "derived" / "20231025"
        / "ch2_tmc_ndn_20231025T1956513800_d_oth_d18.tif",
        "Chandrayaan-2 OHRC",
        None,
    ),
    "pair_003": (
        ROOT / "data" / "processed" / "pairs" / "pair_003" / "source_iirs_on_common_grid.tif",
        ROOT / "data" / "raw" / "tmc2" / "2023_10_25" / "data" / "derived" / "20231025"
        / "ch2_tmc_ndn_20231025T1956513800_d_oth_d18.tif",
        "Chandrayaan-2 IIRS",
        None,
    ),
    # SELENE/Kaguya TC raw .img has no automated ingestion path (only LRO NAC
    # does -- see src/pipeline/ingest.py's SUPPORTED_SENSORS), so, exactly
    # like OHRC/IIRS above, this feeds register() the existing legacy-
    # processed (already map-projected) GeoTIFF rather than the raw archive
    # file, which register()'s own raw-format gate would otherwise reject.
    "pair_005": (
        ROOT / "data" / "processed" / "pair_005" / "canonical" / "kaguya_tc_source.tif",
        ROOT / "data" / "raw" / "tmc2" / "2023_10_25" / "data" / "derived" / "20231025"
        / "ch2_tmc_ndn_20231025T1956513800_d_oth_d18.tif",
        "SELENE/Kaguya TC",
        None,
    ),
}

OLD_STAGE0_METRICS = {
    pid: ROOT / "results" / pid / "stage0_coarse_search" / f"{pid}_stage0_metrics.json"
    for pid in PAIR_INPUTS
}
OLD_STAGE2_METRICS = {
    pid: ROOT / "results" / pid / "stage2_registration" / f"{pid}_stage2_metrics.json"
    for pid in PAIR_INPUTS
}


def load_json(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(pair_id, result):
    old_stage0 = load_json(OLD_STAGE0_METRICS[pair_id])
    old_stage2 = load_json(OLD_STAGE2_METRICS[pair_id])

    print(f"\n--- {pair_id}: OLD vs NEW ---")
    print(f"  OLD stage0 verdict : {old_stage0['verdict'] if old_stage0 else 'N/A'}")
    print(f"  NEW verdict        : {result.verdict}")

    if old_stage2 is not None:
        old_params = old_stage2["fit"]["affine_parameters"]
        print(
            f"  OLD translation_px : ({old_params['translation_x_px']:.2f}, "
            f"{old_params['translation_y_px']:.2f})  rotation={old_params['rotation_deg']:.4f} "
            f"inliers={old_stage2['fit']['ransac_inliers']}/{old_stage2['fit']['ransac_candidates']}"
        )
    else:
        print("  OLD stage2         : N/A (Stage 0/1 never reached PASS in the old path)")

    if result.translation_px[0] is not None:
        print(
            f"  NEW translation_px : ({result.translation_px[0]:.2f}, "
            f"{result.translation_px[1]:.2f})  rotation={result.rotation_deg:.4f} "
            f"inliers={result.inlier_count}/{result.metrics['fit']['ransac_candidates'] if result.metrics.get('fit') else 'N/A'}"
        )
    else:
        print("  NEW translation_px : N/A")

    print(f"  NEW anchor_source  : {result.anchor_source}")
    print(f"  NEW rmse_px        : {result.rmse_px}")
    print(f"  NEW sub_pixel      : {result.sub_pixel_achieved}")


def main():
    parser = argparse.ArgumentParser(description="Regression-test register() against old results")
    parser.add_argument("--pair", choices=sorted(PAIR_INPUTS.keys()), required=True)
    args = parser.parse_args()

    pair_id = args.pair
    source_path, reference_path, source_override, reference_override = PAIR_INPUTS[pair_id]

    if not source_path.exists():
        raise SystemExit(f"{pair_id}: source input not found: {source_path}")

    if not reference_path.exists():
        raise SystemExit(f"{pair_id}: reference input not found: {reference_path}")

    config = load_pipeline_config(
        source_sensor_override=source_override, reference_sensor_override=reference_override
    )

    out_dir = REGRESSION_OUT_ROOT / pair_id

    print("=" * 78)
    print(f"UNIFIED register() REGRESSION -- {pair_id.upper()}")
    print("=" * 78)
    print(f"source:    {source_path}")
    print(f"reference: {reference_path}")

    result = register(source_path, reference_path, config, out_dir)

    compare(pair_id, result)

    print(f"\nOutputs written under: {out_dir}")


if __name__ == "__main__":
    main()
