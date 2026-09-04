from pathlib import Path
import csv
import json
import math
import re
import shutil
import subprocess
import sys


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GLOBAL_RESULTS_PATH = (
    ROOT
    / "results"
    / "pair_006"
    / "global_benchmark"
    / "pair006_global_benchmark.json"
)

FROZEN_MATCHER_SCRIPT = (
    ROOT
    / "scripts"
    / "pair006_tiled_lightglue.py"
)

OUT_DIR = (
    ROOT
    / "results"
    / "pair_006"
    / "auto_hypothesis_selection"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUNNER_DIR = (
    ROOT
    / "scripts"
)

SUMMARY_CSV = (
    OUT_DIR
    / "pair006_auto_hypothesis_summary.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "pair006_auto_hypothesis_summary.json"
)

SELECTED_PRIOR_JSON = (
    OUT_DIR
    / "pair006_selected_prior.json"
)

SELECTED_DIR = (
    OUT_DIR
    / "selected"
)


# ============================================================
# HELPERS
# ============================================================

def safe_name(text):
    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def affine_parameters(matrix):
    a = float(matrix[0][0])
    b = float(matrix[0][1])
    tx = float(matrix[0][2])

    c = float(matrix[1][0])
    d = float(matrix[1][1])
    ty = float(matrix[1][2])

    scale_x = math.sqrt(
        a * a + c * c
    )

    scale_y = math.sqrt(
        b * b + d * d
    )

    rotation_deg = math.degrees(
        math.atan2(
            c,
            a,
        )
    )

    determinant = (
        a * d
        - b * c
    )

    axis_dot = (
        a * b
        + c * d
    )

    translation = math.hypot(
        tx,
        ty,
    )

    return {
        "translation_x_px": tx,
        "translation_y_px": ty,
        "translation_magnitude_px":
            translation,
        "rotation_deg":
            rotation_deg,
        "scale_x":
            scale_x,
        "scale_y":
            scale_y,
        "determinant":
            determinant,
        "axis_dot":
            axis_dot,
    }


def shape_sane(info):
    """
    Geometry sanity test deliberately ignores absolute translation.

    Pair006 demonstrated that a large real coarse displacement can
    still have a sensible affine shape.
    """

    return (
        0.85
        <= info["scale_x"]
        <= 1.15

        and

        0.85
        <= info["scale_y"]
        <= 1.15

        and

        abs(
            info["rotation_deg"]
        )
        <= 5.0

        and

        0.70
        <= info["determinant"]
        <= 1.30

        and

        abs(
            info["axis_dot"]
        )
        <= 0.20
    )


def replace_assignment(
    text,
    variable,
    replacement,
):
    pattern = (
        rf"{variable}\s*=\s*\("
        rf".*?"
        rf"\)"
    )

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.DOTALL,
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one "
            f"{variable} block, "
            f"found {len(matches)}."
        )

    return re.sub(
        pattern,
        lambda match: replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )


def extract_hypotheses():
    with open(
        GLOBAL_RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        results = json.load(f)

    hypotheses = []

    for index, result in enumerate(results):
        affine = result.get(
            "affine"
        )

        if affine is None:
            continue

        method = result.get(
            "method",
            f"method_{index}",
        )

        representation = result.get(
            "representation",
            "unknown",
        )

        candidate_id = (
            f"{safe_name(method)}_"
            f"{safe_name(representation)}"
        )

        # Avoid duplicate names.
        original_id = candidate_id
        suffix = 2

        existing_ids = {
            item["candidate_id"]
            for item in hypotheses
        }

        while (
            candidate_id
            in existing_ids
        ):
            candidate_id = (
                f"{original_id}_{suffix}"
            )
            suffix += 1

        hypotheses.append(
            {
                "candidate_id":
                    candidate_id,

                "method":
                    method,

                "representation":
                    representation,

                "affine":
                    affine,

                "global_result":
                    result,
            }
        )

    if not hypotheses:
        raise RuntimeError(
            "No global affine hypotheses found."
        )

    return hypotheses


def create_candidate_prior(
    hypothesis,
    candidate_dir,
):
    path = (
        candidate_dir
        / "candidate_prior.json"
    )

    # pair006_tiled_lightglue.py specifically looks for:
    # method=lightglue and representation=gradient.
    #
    # We therefore expose each candidate affine through that
    # expected interface without modifying the frozen matcher.
    payload = [
        {
            "method":
                "lightglue",

            "representation":
                "gradient",

            "affine":
                hypothesis[
                    "affine"
                ],

            "diagnostic_source_method":
                hypothesis[
                    "method"
                ],

            "diagnostic_source_representation":
                hypothesis[
                    "representation"
                ],

            "candidate_id":
                hypothesis[
                    "candidate_id"
                ],
        }
    ]

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    return path


def create_runner(
    hypothesis,
    prior_path,
    candidate_dir,
):
    with open(
        FROZEN_MATCHER_SCRIPT,
        "r",
        encoding="utf-8-sig",
    ) as f:
        text = f.read()

    global_replacement = (
        "GLOBAL_RESULTS_PATH = "
        f"Path(r\"{prior_path}\")"
    )

    text = replace_assignment(
        text,
        "GLOBAL_RESULTS_PATH",
        global_replacement,
    )

    out_replacement = (
        "OUT_DIR = "
        f"Path(r\"{candidate_dir}\")"
    )

    text = replace_assignment(
        text,
        "OUT_DIR",
        out_replacement,
    )

    candidate_id = (
        hypothesis[
            "candidate_id"
        ]
    )

    runner_path = (
        RUNNER_DIR
        / (
            "__pair006_auto_"
            f"{candidate_id}.py"
        )
    )

    with open(
        runner_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text)

    return runner_path


def run_candidate(
    hypothesis,
):
    candidate_id = (
        hypothesis[
            "candidate_id"
        ]
    )

    candidate_dir = (
        OUT_DIR
        / "candidates"
        / candidate_id
    )

    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prior_path = (
        create_candidate_prior(
            hypothesis,
            candidate_dir,
        )
    )

    runner_path = (
        create_runner(
            hypothesis,
            prior_path,
            candidate_dir,
        )
    )

    log_path = (
        candidate_dir
        / "console.log"
    )

    print()
    print("=" * 90)

    print(
        "VERIFYING HYPOTHESIS:",
        candidate_id,
    )

    initial_info = (
        affine_parameters(
            hypothesis[
                "affine"
            ]
        )
    )

    print(
        "Source:",
        hypothesis["method"],
        "/",
        hypothesis["representation"],
    )

    print(
        "Initial translation:",
        f"{initial_info['translation_magnitude_px']:.3f}",
        "px",
    )

    print(
        "Initial tx / ty:",
        f"{initial_info['translation_x_px']:.3f}",
        "/",
        f"{initial_info['translation_y_px']:.3f}",
    )

    print("=" * 90)

    try:
        process = subprocess.run(
            [
                sys.executable,
                str(runner_path),
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )

        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                process.stdout
            )

            if process.stderr:
                f.write(
                    "\n\nSTDERR\n"
                )

                f.write(
                    process.stderr
                )

        if process.returncode != 0:
            print(
                "FAILED:",
                candidate_id,
            )

            print(
                "See:",
                log_path,
            )

            return {
                "candidate_id":
                    candidate_id,

                "method":
                    hypothesis[
                        "method"
                    ],

                "representation":
                    hypothesis[
                        "representation"
                    ],

                "status":
                    "failed",

                "error":
                    (
                        process.stderr[-2000:]
                        if process.stderr
                        else
                        "unknown"
                    ),
            }

        metrics_path = (
            candidate_dir
            / "pair006_tiled_lightglue_metrics.json"
        )

        if not metrics_path.exists():
            raise RuntimeError(
                "Matcher finished but metrics "
                "JSON was not created."
            )

        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as f:
            metrics = json.load(f)

        affine = metrics[
            "affine"
        ]

        recovered_info = (
            affine_parameters(
                affine
            )
        )

        sane_shape = (
            shape_sane(
                recovered_info
            )
        )

        inliers = int(
            metrics.get(
                "ransac_inliers",
                0,
            )
        )

        inlier_ratio = float(
            metrics.get(
                "ransac_inlier_ratio",
                0.0,
            )
        )

        coverage = float(
            metrics.get(
                "coverage_ratio",
                0.0,
            )
        )

        rmse = float(
            metrics.get(
                "ransac_reprojection_rmse_px",
                999999.0,
            )
        )

        # ----------------------------------------------------
        # Verification score
        #
        # Rewards:
        #   - many RANSAC inliers
        #   - high inlier ratio
        #   - broad spatial coverage
        #
        # Penalizes:
        #   - poor self-consistency
        #
        # Translation magnitude is intentionally NOT penalized.
        # ----------------------------------------------------

        if sane_shape:
            score = (
                inliers
                * inlier_ratio
                * coverage
                / max(
                    rmse,
                    0.5,
                )
            )
        else:
            score = 0.0

        print(
            "Recovered:",
            f"inliers={inliers}",
            f"ratio={inlier_ratio:.4f}",
            f"coverage={coverage:.4f}",
            f"selfRMSE={rmse:.4f}",
        )

        print(
            "Recovered translation:",
            f"{recovered_info['translation_magnitude_px']:.3f}",
            "px",
        )

        print(
            "Shape sane:",
            sane_shape,
        )

        print(
            "Verification score:",
            f"{score:.3f}",
        )

        return {
            "candidate_id":
                candidate_id,

            "method":
                hypothesis[
                    "method"
                ],

            "representation":
                hypothesis[
                    "representation"
                ],

            "status":
                "success",

            "initial_affine":
                hypothesis[
                    "affine"
                ],

            "initial_translation_x_px":
                initial_info[
                    "translation_x_px"
                ],

            "initial_translation_y_px":
                initial_info[
                    "translation_y_px"
                ],

            "initial_translation_magnitude_px":
                initial_info[
                    "translation_magnitude_px"
                ],

            "distinct_candidates":
                int(
                    metrics.get(
                        "distinct_candidates",
                        0,
                    )
                ),

            "ransac_inliers":
                inliers,

            "ransac_inlier_ratio":
                inlier_ratio,

            "coverage_cells":
                int(
                    metrics.get(
                        "coverage_cells",
                        0,
                    )
                ),

            "valid_coverage_cells":
                int(
                    metrics.get(
                        "valid_coverage_cells",
                        0,
                    )
                ),

            "coverage_ratio":
                coverage,

            "ransac_self_rmse_px":
                rmse,

            "recovered_affine":
                affine,

            "recovered_translation_x_px":
                recovered_info[
                    "translation_x_px"
                ],

            "recovered_translation_y_px":
                recovered_info[
                    "translation_y_px"
                ],

            "recovered_translation_magnitude_px":
                recovered_info[
                    "translation_magnitude_px"
                ],

            "recovered_rotation_deg":
                recovered_info[
                    "rotation_deg"
                ],

            "recovered_scale_x":
                recovered_info[
                    "scale_x"
                ],

            "recovered_scale_y":
                recovered_info[
                    "scale_y"
                ],

            "recovered_determinant":
                recovered_info[
                    "determinant"
                ],

            "recovered_axis_dot":
                recovered_info[
                    "axis_dot"
                ],

            "shape_sane":
                sane_shape,

            "verification_score":
                float(score),

            "candidate_directory":
                str(candidate_dir),

            "console_log":
                str(log_path),
        }

    finally:
        # Temporary runner is not part of experiment output.
        if runner_path.exists():
            runner_path.unlink()


def save_summary(
    hypotheses,
    results,
):
    successful = [
        result
        for result in results
        if (
            result.get(
                "status"
            )
            == "success"
        )
    ]

    if not successful:
        raise RuntimeError(
            "Every coarse hypothesis failed."
        )

    ranked = sorted(
        successful,
        key=lambda item: (
            item[
                "verification_score"
            ],
            item[
                "coverage_ratio"
            ],
            item[
                "ransac_inliers"
            ],
        ),
        reverse=True,
    )

    best = ranked[0]

    # --------------------------------------------------------
    # JSON summary
    # --------------------------------------------------------

    summary = {
        "pair_id":
            "pair_006",

        "method":
            "automatic_multi_hypothesis_coarse_prior_selection",

        "hypothesis_source":
            str(
                GLOBAL_RESULTS_PATH
            ),

        "verification_matcher":
            str(
                FROZEN_MATCHER_SCRIPT
            ),

        "number_of_hypotheses":
            len(hypotheses),

        "number_successful":
            len(successful),

        "selection_rule":
            (
                "shape-sane candidates ranked by "
                "(RANSAC inliers × inlier ratio × "
                "spatial coverage) / self-consistency RMSE; "
                "absolute translation magnitude is not penalized"
            ),

        "selected_candidate_id":
            best[
                "candidate_id"
            ],

        "selected_source_method":
            best[
                "method"
            ],

        "selected_source_representation":
            best[
                "representation"
            ],

        "selected_recovered_affine":
            best[
                "recovered_affine"
            ],

        "selected_verification_score":
            best[
                "verification_score"
            ],

        "ranking":
            ranked,

        "all_results":
            results,

        "note":
            (
                "RANSAC reprojection RMSE is internal "
                "self-consistency only. This selection "
                "does not establish independent absolute "
                "registration ground truth."
            ),
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # CSV summary
    # --------------------------------------------------------

    columns = [
        "rank",
        "candidate_id",
        "method",
        "representation",
        "initial_translation_magnitude_px",
        "recovered_translation_magnitude_px",
        "distinct_candidates",
        "ransac_inliers",
        "ransac_inlier_ratio",
        "coverage_cells",
        "valid_coverage_cells",
        "coverage_ratio",
        "ransac_self_rmse_px",
        "shape_sane",
        "verification_score",
    ]

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        for rank, result in enumerate(
            ranked,
            start=1,
        ):
            writer.writerow(
                {
                    "rank":
                        rank,

                    "candidate_id":
                        result[
                            "candidate_id"
                        ],

                    "method":
                        result[
                            "method"
                        ],

                    "representation":
                        result[
                            "representation"
                        ],

                    "initial_translation_magnitude_px":
                        result[
                            "initial_translation_magnitude_px"
                        ],

                    "recovered_translation_magnitude_px":
                        result[
                            "recovered_translation_magnitude_px"
                        ],

                    "distinct_candidates":
                        result[
                            "distinct_candidates"
                        ],

                    "ransac_inliers":
                        result[
                            "ransac_inliers"
                        ],

                    "ransac_inlier_ratio":
                        result[
                            "ransac_inlier_ratio"
                        ],

                    "coverage_cells":
                        result[
                            "coverage_cells"
                        ],

                    "valid_coverage_cells":
                        result[
                            "valid_coverage_cells"
                        ],

                    "coverage_ratio":
                        result[
                            "coverage_ratio"
                        ],

                    "ransac_self_rmse_px":
                        result[
                            "ransac_self_rmse_px"
                        ],

                    "shape_sane":
                        result[
                            "shape_sane"
                        ],

                    "verification_score":
                        result[
                            "verification_score"
                        ],
                }
            )

    # --------------------------------------------------------
    # Save selected prior in same interface expected by
    # tiled-LightGlue.
    # --------------------------------------------------------

    selected_prior = [
        {
            "method":
                "lightglue",

            "representation":
                "gradient",

            "affine":
                best[
                    "recovered_affine"
                ],

            "selected_from":
                best[
                    "candidate_id"
                ],

            "verification_score":
                best[
                    "verification_score"
                ],
        }
    ]

    with open(
        SELECTED_PRIOR_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            selected_prior,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Copy best candidate artifacts into selected/
    # --------------------------------------------------------

    if SELECTED_DIR.exists():
        shutil.rmtree(
            SELECTED_DIR
        )

    SELECTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_candidate_dir = Path(
        best[
            "candidate_directory"
        ]
    )

    for name in [
        "pair006_tiled_lightglue_metrics.json",
        "pair006_tiled_lightglue_inliers.csv",
        "pair006_tiled_lightglue_matches.png",
        "console.log",
        "candidate_prior.json",
    ]:
        src = (
            source_candidate_dir
            / name
        )

        if src.exists():
            shutil.copy2(
                src,
                SELECTED_DIR
                / name,
            )

    return ranked, best


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print(
        "PAIR 006 — AUTOMATIC COARSE-HYPOTHESIS SELECTION"
    )
    print("=" * 90)

    print()
    print(
        "Global hypothesis source:"
    )

    print(
        GLOBAL_RESULTS_PATH
    )

    hypotheses = (
        extract_hypotheses()
    )

    print()
    print(
        "Global affine hypotheses found:",
        len(hypotheses),
    )

    for index, hypothesis in enumerate(
        hypotheses,
        start=1,
    ):
        info = affine_parameters(
            hypothesis[
                "affine"
            ]
        )

        print(
            f"{index:02d}. "
            f"{hypothesis['candidate_id']:<25} "
            f"translation="
            f"{info['translation_magnitude_px']:.2f}px "
            f"tx={info['translation_x_px']:.2f} "
            f"ty={info['translation_y_px']:.2f}"
        )

    results = []

    for hypothesis in hypotheses:
        result = run_candidate(
            hypothesis
        )

        results.append(
            result
        )

    ranked, best = (
        save_summary(
            hypotheses,
            results,
        )
    )

    print()
    print("=" * 90)
    print("AUTOMATIC HYPOTHESIS RANKING")
    print("=" * 90)

    for rank, result in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"{result['candidate_id']:<25} "
            f"inliers={result['ransac_inliers']:4d} "
            f"ratio={result['ransac_inlier_ratio']:.4f} "
            f"coverage={result['coverage_ratio']:.4f} "
            f"RMSE={result['ransac_self_rmse_px']:.3f} "
            f"tx={result['recovered_translation_x_px']:.1f} "
            f"ty={result['recovered_translation_y_px']:.1f} "
            f"score={result['verification_score']:.2f}"
        )

    print()
    print("=" * 90)
    print("SELECTED COARSE HYPOTHESIS")
    print("=" * 90)

    print(
        "Candidate:",
        best[
            "candidate_id"
        ],
    )

    print(
        "Origin:",
        best[
            "method"
        ],
        "/",
        best[
            "representation"
        ],
    )

    print(
        "RANSAC inliers:",
        best[
            "ransac_inliers"
        ],
    )

    print(
        "Spatial coverage:",
        f"{best['coverage_cells']}/"
        f"{best['valid_coverage_cells']}",
    )

    print(
        "Recovered translation:",
        f"{best['recovered_translation_magnitude_px']:.3f}",
        "px",
    )

    print(
        "Recovered tx / ty:",
        f"{best['recovered_translation_x_px']:.3f}",
        "/",
        f"{best['recovered_translation_y_px']:.3f}",
    )

    print(
        "Verification score:",
        f"{best['verification_score']:.3f}",
    )

    print()
    print("Outputs:")

    print(
        SUMMARY_CSV
    )

    print(
        SUMMARY_JSON
    )

    print(
        SELECTED_PRIOR_JSON
    )

    print(
        SELECTED_DIR
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This automatically selects a coarse prior "
        "using cross-sensor match consistency and "
        "spatial coverage."
    )

    print(
        "RANSAC residual remains an internal "
        "self-consistency metric, not independent "
        "absolute ground-truth accuracy."
    )

    print()
    print(
        "PAIR 006 AUTOMATIC COARSE-HYPOTHESIS "
        "SELECTION COMPLETE."
    )


if __name__ == "__main__":
    main()
