import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coarse.coarse_structural_search import load_config, run_stage0_with_fallback
from src.coarse.oriented_gradients import robust_unit_stretch
from src.coarse.verification import shift_array
from src.registration.affine_registration import create_difference, create_overlay


CONFIG_PATH = ROOT / "configs" / "stage0_coarse_search.json"

# I/O plumbing only: the two canonical naming conventions already on disk.
# No algorithmic thresholds live here -- those come entirely from
# CONFIG_PATH and are identical for every pair.
PAIR_FILES = {
    "pair_002": {
        "dir": ROOT / "data" / "processed" / "pairs" / "pair_002",
        "source": "source_ohrc_on_tmc_grid.tif",
        "reference": "reference_tmc2.tif",
        "mask": "valid_mask.tif",
    },
    "pair_003": {
        "dir": ROOT / "data" / "processed" / "pairs" / "pair_003",
        "source": "source_iirs_on_common_grid.tif",
        "reference": "reference_tmc2_on_common_grid.tif",
        "mask": "valid_mask.tif",
    },
    "pair_004": {
        "dir": ROOT / "data" / "processed" / "pair_004" / "canonical",
        "source": "lro_nac_source.tif",
        "reference": "tmc2_reference.tif",
        "mask": "common_valid_mask.tif",
    },
    "pair_006": {
        "dir": ROOT / "data" / "processed" / "pair_006" / "canonical",
        "source": "source_lro_nac.tif",
        "reference": "reference_tmc2.tif",
        "mask": "matcher_mask.tif",
    },
    "pair_007": {
        "dir": ROOT / "data" / "processed" / "pair_007" / "canonical",
        "source": "source_lro_nac.tif",
        "reference": "reference_tmc2.tif",
        "mask": "matcher_mask.tif",
    },
}


def read_band(path):
    with rasterio.open(path) as ds:
        data = ds.read(1)
        nodata = ds.nodata

        return data, nodata


def load_pair(pair_id):
    spec = PAIR_FILES[pair_id]
    pair_dir = spec["dir"]

    source, source_nodata = read_band(pair_dir / spec["source"])
    reference, _reference_nodata = read_band(pair_dir / spec["reference"])
    mask_raw, _mask_nodata = read_band(pair_dir / spec["mask"])

    if source.shape != reference.shape or source.shape != mask_raw.shape:
        raise RuntimeError(
            f"{pair_id}: source/reference/mask shapes differ: "
            f"{source.shape} / {reference.shape} / {mask_raw.shape}"
        )

    valid = mask_raw > 0
    valid &= np.isfinite(source)
    valid &= np.isfinite(reference)

    if source_nodata is not None:
        valid &= source != source_nodata

    # Matches v1's matcher-mask convention (benchmark_pair007_global.py):
    # zero-DN reference pixels carry no usable structure for correlation.
    valid &= reference > 0

    return source, reference, valid


def surface_preview_png(source_mask, reference_mask, best, out_path):
    """
    Quick visual sanity check: overlay the two masks with the best
    hypothesis's displacement marked, since the correlation surface
    itself can be tens of thousands of pixels wide and isn't a useful
    image on its own without the shift context.
    """

    height, width = reference_mask.shape

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :, 2] = reference_mask.astype(np.uint8) * 180

    if best is not None:
        warped_source_mask = shift_array(
            source_mask.astype(np.uint8), best["dy"], best["dx"]
        )
        canvas[:, :, 1] = warped_source_mask * 180

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Stage 0 dense structural coarse search")
    parser.add_argument("--pair", required=True, choices=sorted(PAIR_FILES.keys()))
    args = parser.parse_args()

    pair_id = args.pair

    print("=" * 78)
    print(f"STAGE 0 -- DENSE STRUCTURAL COARSE SEARCH -- {pair_id.upper()}")
    print("=" * 78)

    config = load_config(CONFIG_PATH)

    print("\nConfig:", CONFIG_PATH)
    print(json.dumps(config, indent=2))

    source, reference, mask = load_pair(pair_id)

    print("\nShape:", source.shape)
    print("Valid pixels:", f"{int(mask.sum()):,}", "/", f"{mask.size:,}")

    result = run_stage0_with_fallback(source, mask, reference, mask, config)

    print("\nAnchor source:", result["anchor_source"])

    if result.get("keypoint_fallback_result") is not None:
        print("Keypoint fallback result:", json.dumps(result["keypoint_fallback_result"], indent=2, default=str))

    print("\nHypotheses (ranked):")

    for i, hyp in enumerate(result["hypotheses"], start=1):
        print(
            f"  {i}. dy={hyp['dy']:>6}  dx={hyp['dx']:>6}  "
            f"|t|={hyp['translation_magnitude_px']:>8.2f}px  "
            f"score={hyp['score']:.4f}"
        )

    print("\nAmbiguity ratio (peak1/peak2):", result["ambiguity_ratio"])

    if result["verification"] is not None:
        v = result["verification"]

        print("\nIndependent verification of best hypothesis:")
        print("  overlap_fraction     :", v["overlap_fraction"])
        print("  structural NCC before:", v["ncc_before"])
        print("  structural NCC after :", v["ncc_after"])
        print("  NCC improvement      :", v["ncc_improvement"])
        print("  local_cell_agreement :", v["local_cell_agreement"])

        agreeing = sum(1 for c in v["local_cells"] if c["agrees"])
        print(
            f"  local cells agreeing : {agreeing}/{len(v['local_cells'])}"
        )

    print("\nVERDICT:", result["verdict"])

    for reason in result["verdict_reasons"]:
        print("  -", reason)

    out_dir = ROOT / "results" / pair_id / "stage0_coarse_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"{pair_id}_stage0_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    surface_preview_png(
        mask, mask, result["best"], out_dir / f"{pair_id}_stage0_mask_overlay.png"
    )

    if result["best"] is not None:
        dy, dx = result["best"]["dy"], result["best"]["dx"]

        source_u8 = (robust_unit_stretch(source, mask) * 255.0).astype(np.uint8)
        reference_u8 = (robust_unit_stretch(reference, mask) * 255.0).astype(np.uint8)

        warped_source_u8 = shift_array(source_u8, dy, dx, fill_value=0)

        overlay = create_overlay(warped_source_u8, reference_u8)
        difference = create_difference(warped_source_u8, reference_u8)

        cv2.imwrite(str(out_dir / f"{pair_id}_stage0_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{pair_id}_stage0_difference.png"), difference)

    print("\nOutputs:")
    print(" ", metrics_path)
    print(" ", out_dir)

    print(f"\nSTAGE 0 COMPLETE -- {pair_id.upper()}")


if __name__ == "__main__":
    main()
