from pathlib import Path
import json
import re

import numpy as np
import rasterio
import kornia.feature as KF

import validate_pair005_fine_perturbation as proto


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pair_007"
    / "canonical"
)

SOURCE_PATH = (
    CANONICAL_DIR
    / "source_lro_nac.tif"
)

REFERENCE_PATH = (
    CANONICAL_DIR
    / "reference_tmc2.tif"
)

MATCHER_MASK_PATH = (
    CANONICAL_DIR
    / "matcher_mask.tif"
)

BENCHMARK_SCRIPT = (
    ROOT
    / "scripts"
    / "benchmark_pair007_global.py"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_007"
    / "loftr_intensity_perturbation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_JSON = (
    OUT_DIR
    / "pair007_loftr_intensity_perturbation.json"
)


# ============================================================
# READ FROZEN GLOBAL MATCHING DIMENSION
# ============================================================

def read_max_dim():
    text = BENCHMARK_SCRIPT.read_text(
        encoding="utf-8-sig"
    )

    m = re.search(
        r"(?m)^\s*LOFTR_MAX_DIM\s*=\s*(\d+)\s*(?:#.*)?$",
        text,
    )

    if not m:
        raise RuntimeError(
            "LOFTR_MAX_DIM was not found inside "
            "benchmark_pair007_global.py"
        )

    return (
        "LOFTR_MAX_DIM",
        int(m.group(1)),
    )

# ============================================================
# JSON CONVERTER
# ============================================================

def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, dict):
        return {
            str(k): jsonable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            jsonable(v)
            for v in obj
        ]

    return obj


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print(
        "PAIR 007 — LOFTR INTENSITY "
        "CONTROLLED PERTURBATION VALIDATION"
    )
    print("=" * 78)

    constant_name, max_dim = (
        read_max_dim()
    )

    # Use Pair007 global benchmark matching resolution.
    proto.MAX_DIM = max_dim

    # Preserve validated Pair005 perturbation protocol.
    # Its RANSAC threshold is already 3 px.
    print()
    print(
        "Inherited Pair007 benchmark:",
        f"{constant_name}={max_dim}",
    )

    print(
        "RANSAC threshold:",
        proto.RANSAC_THRESHOLD_PX,
        "px",
    )

    print(
        "Perturbation mask erosion radius:",
        proto.MASK_EROSION_RADIUS,
    )

    print(
        "Device:",
        proto.DEVICE,
    )

    print()

    # --------------------------------------------------------
    # LOAD PAIR
    # --------------------------------------------------------

    with rasterio.open(
        SOURCE_PATH
    ) as ds:
        source = (
            ds.read(1)
            .astype(np.float32)
        )

    with rasterio.open(
        REFERENCE_PATH
    ) as ds:
        reference = (
            ds.read(1)
            .astype(np.float32)
        )

    with rasterio.open(
        MATCHER_MASK_PATH
    ) as ds:
        matcher_mask = (
            ds.read(1)
            > 0
        )

    if (
        source.shape
        != reference.shape
        or source.shape
        != matcher_mask.shape
    ):
        raise RuntimeError(
            "Pair007 source/reference/mask "
            "shape mismatch."
        )

    print(
        "Shape:",
        source.shape,
    )

    print(
        "Matcher-valid pixels:",
        f"{int(matcher_mask.sum()):,}",
    )

    print(
        "Matcher-valid ratio:",
        f"{matcher_mask.mean():.6f}",
    )

    # For Pair007 the canonical matcher mask already
    # represents common valid NAC/TMC support.
    base_common = (
        matcher_mask.copy()
    )

    reference_matcher = (
        matcher_mask.copy()
    )

    # --------------------------------------------------------
    # LOFTR
    # --------------------------------------------------------

    print()
    print(
        "Loading LoFTR outdoor..."
    )

    loftr = (
        KF.LoFTR(
            pretrained="outdoor"
        )
        .eval()
        .to(
            proto.DEVICE
        )
    )

    def loftr_runner(
        a,
        b,
    ):
        return proto.run_loftr(
            a,
            b,
            loftr,
        )

    # --------------------------------------------------------
    # RUN ORIGINAL PAIR + CONTROLLED PERTURBATIONS
    # --------------------------------------------------------

    result = (
        proto.validate_matcher(
            "LoFTR",
            "intensity",
            source,
            reference,
            base_common,
            reference_matcher,
            loftr_runner,
        )
    )

    payload = {
        "pair_id":
            "pair_007",

        "source":
            "LRO NAC M185210693RE",

        "reference":
            "Chandrayaan-2 TMC-2",

        "canonical_resolution_m":
            5.0,

        "matcher":
            "LoFTR",

        "representation":
            "intensity",

        "pretrained":
            "outdoor",

        "max_matching_dimension":
            max_dim,

        "purpose":
            (
                "Held-out test of whether the "
                "Pair007 LoFTR large-displacement "
                "branch follows known injected "
                "source motion."
            ),

        "result":
            result,

        "interpretation_note":
            (
                "Perturbation response measures "
                "motion-following robustness. "
                "It is not independent absolute "
                "ground-truth accuracy for the "
                "unperturbed NAC/TMC registration."
            ),
    }

    payload = jsonable(
        payload
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print(
        "PAIR 007 LOFTR PERTURBATION RESULT"
    )
    print("=" * 78)

    baseline = result[
        "baseline"
    ]

    print(
        "Baseline candidates:",
        baseline.get(
            "candidates"
        ),
    )

    print(
        "Baseline inliers:",
        baseline.get(
            "inliers"
        ),
    )

    print(
        "Baseline self RMSE:",
        baseline.get(
            "self_rmse_px"
        ),
    )

    print(
        "Baseline coverage:",
        baseline.get(
            "coverage"
        ),
    )

    print()

    for test in result[
        "tests"
    ]:
        print(
            json.dumps(
                jsonable(test),
                indent=2,
            )
        )

    print()
    print(
        "All tests pass:",
        result.get(
            "all_tests_pass"
        ),
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_JSON
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Before interpreting perturbation "
        "performance, verify that the LoFTR "
        "baseline itself recovered the same "
        "~900 px Pair007 displacement basin."
    )


if __name__ == "__main__":
    main()


