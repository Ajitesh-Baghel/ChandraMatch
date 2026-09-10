"""
The anchor-confidence classifier (unification pass, §5b).

IMPORTANT SCOPE NOTE, found by actually assembling the training table rather
than assuming a row count: this pass does NOT ship a fitted model. Real,
honestly-usable, non-circular training data for the two decisions this model
would need to make comes to well under the ~20-row floor the original spec
itself set as the point below which fitting is unsafe -- see
`build_training_table()`'s docstring for the exact count and why. Per that
spec's own explicit contingency, the correct outcome is to say so plainly and
keep the existing hand-set threshold cascade in
src/coarse/coarse_structural_search.py::_decide_verdict as the live decision
path -- NOT to fit an unstable model to single digits of rows and present it
as calibrated. `AnchorConfidenceModel.decide()` returns None in that case,
which `anchor.py` treats as "use the existing cascade, unchanged."

This module still implements the full fit/save/load/decide machinery (using
scikit-learn's LogisticRegression, already a project dependency -- no new
install), so that once more real pairs are validated and this file is
re-generated, switching over requires no architecture change, just re-running
`main()` here.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(__file__).resolve().parent / "models" / "anchor_confidence_v1.json"

# Two separate feature spaces / two separate decisions, because they are
# different questions asked at different points in the fallback chain, with
# almost no feature overlap: "should this dense-correlation hypothesis (which
# already cleared the hard physical gates) be trusted" vs "should this
# fallback matcher's RANSAC fit be trusted." Pooling them into one table
# would mean padding each row with NaN placeholders for the other space's
# features, which does not make either decision more data-rich -- it just
# hides the fact that each individually has too few rows.
DENSE_FEATURE_NAMES = [
    "bbox_fill_ratio",
    "ambiguity_ratio",
    "overlap_fraction",
    "ncc_before",
    "ncc_after",
    "ncc_improvement",
    "local_cell_agreement",
]
FALLBACK_FEATURE_NAMES = ["candidates", "inliers", "inlier_ratio"]

# verification_tier is NOT a feature anywhere in this file -- it is
# leakage-from-the-future if used as one. It's computed in Stage 4, from
# Stage 2/3's perturbation test, which only runs AFTER an anchor has already
# been accepted at Stage 0/1. A live acceptance model decides before that
# test exists, so it can never actually receive this value at the moment it
# needs to decide, regardless of encoding -- a prior version of this module
# one-hot-encoded it as a per-row feature, which was wrong for exactly this
# reason (also visibly symptomatic: the fallback feature space's tier column
# was constant across every row, since a fallback anchor always gets the
# same tier by construction -- a column that can't vary can't predict
# anything, which was the tell that it didn't belong as a feature).
#
# What verification_tier legitimately IS: evidence about how much to TRUST a
# training row's label, decided only at fit time, never seen by decide().
# perturbation_confirmed rows have their PASS/FAIL label backed by an
# independent stress test -- the strongest ground truth this project has.
# perturbation_partial rows have some but incomplete independent support.
# Every other tier (ransac_and_prior_ground_truth_only, unverified_no_fit,
# perturbation_untestable, perturbation_failed) rests on RANSAC fit quality
# and/or separately-established prior ground truth only -- real evidence,
# just not independently perturbation-confirmed, so down-weighted rather
# than excluded. Applied via scikit-learn's `sample_weight` in `_try_fit`
# (LogisticRegression.fit(..., sample_weight=...) is a first-class,
# documented parameter -- this is not a workaround).
VERIFICATION_TIER_CATEGORIES = [
    "perturbation_confirmed",
    "perturbation_partial",
    "perturbation_untestable",
    "perturbation_failed",
    "ransac_and_prior_ground_truth_only",
    "unverified_no_fit",
]
TIER_TRAINING_WEIGHTS = {
    "perturbation_confirmed": 1.0,
    "perturbation_partial": 0.7,
    "ransac_and_prior_ground_truth_only": 0.4,
    "unverified_no_fit": 0.4,
    "perturbation_untestable": 0.4,
    "perturbation_failed": 0.4,
}

MIN_ROWS_TO_FIT = 20

# Every pair's final metrics.json, where verification_tier is actually
# computed and written (src/pipeline/export.py, Stage 4) -- NOT available in
# the Stage 0 stage0_raw.json this module otherwise reads rows from.
METRICS_JSON_PATHS = {
    pid: ROOT / "results" / "unified_regression" / pid / "metrics.json"
    for pid in ("pair_002", "pair_003", "pair_004", "pair_005", "pair_006", "pair_007")
}


# The 6 real, archive-sourced pairs this project has ever validated. Every
# training row below comes from one of these -- no synthetic data anywhere.
#
# Prefers each pair's freshly-recomputed `stage0_raw.json` under
# results/unified_regression/ (written by src/pipeline/run.py, or by the
# one-off recomputation this module's checkpoint pass ran directly against
# already-canonicalized data) over the STALE frozen file from
# scripts/run_stage0_coarse_search.py -- the frozen files predate the fix
# that exposes `bbox_fill_ratio` in run_stage0's own verification dict (it
# was computed but silently discarded before), so every one of them fails
# this module's completeness check and contributes 0 dense rows. Falls back
# to the frozen file only if the fresh one doesn't exist yet.
def _stage0_metrics_path(pair_id):
    fresh = ROOT / "results" / "unified_regression" / pair_id / "stage0_raw.json"

    if fresh.exists():
        return fresh

    return ROOT / "results" / pair_id / "stage0_coarse_search" / f"{pair_id}_stage0_metrics.json"


STAGE0_METRICS_PATHS = {
    pid: _stage0_metrics_path(pid)
    for pid in ("pair_002", "pair_003", "pair_004", "pair_005", "pair_006", "pair_007")
}

# Pair004's ORIGINAL frozen stage0 JSON recorded its lightglue_gradient
# fallback attempt against the un-eroded common_valid_mask.tif (4 inliers,
# rejected) -- the unified pipeline's corrected run (23/23 inliers, accepted,
# see the checkpoint report) is a materially different, more-trustworthy
# example of the SAME decision and belongs in the table too, sourced from the
# regression run rather than the stale frozen file.
PAIR004_CORRECTED_METRICS_PATH = (
    ROOT / "results" / "unified_regression" / "pair_004" / "metrics.json"
)


def _load_json(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_training_table():
    """
    Assembles every real, labeled training row this project currently has.

    Returns (dense_rows, fallback_rows), each a list of dicts:
    {"pair_id", "method", "features": {...}, "label": 0/1, "source": path}.

    Label convention: 1 = this hypothesis/fit was ACCEPTED (PASS) by the
    existing hand-set logic, 0 = REJECTED (FAIL or AMBIGUOUS). Pair004's
    dense-correlation attempt is deliberately EXCLUDED from `dense_rows`: it
    never reaches the soft-failure cascade this classifier would replace --
    it's rejected at the hard bbox_fill_ratio physical gate first (0.21 <
    0.30), which stays a non-negotiable geometric sanity check regardless of
    any learned model, not a data point about the soft cascade's threshold
    quality. Including it would train the model to associate "otherwise
    healthy-looking dense-search features" with a FAIL it never actually
    reached on those grounds -- a mislabeled row, not a real example.
    """

    dense_rows = []
    fallback_rows = []

    for pair_id, path in STAGE0_METRICS_PATHS.items():
        data = _load_json(path)

        if data is None:
            continue

        pair_metrics = _load_json(METRICS_JSON_PATHS[pair_id])
        verification_tier = pair_metrics.get("verification_tier") if pair_metrics is not None else None

        verification = data.get("verification")

        # Only a row whose `anchor_source` is (still) "dense_correlation"
        # describes ONE consistent hypothesis end-to-end -- when the fallback
        # chain succeeds, run_stage0_with_fallback's returned "verification"
        # is recomputed AT THE FALLBACK'S anchor, but its top-level
        # "ambiguity_ratio" is left over from the original, REJECTED dense
        # hypothesis (see that function's own docstring caveat). Merging
        # those into one feature row would combine two different anchors'
        # numbers into a single, meaningless vector -- excluded here, not
        # just for the fallback anchor's sake but for this model's honesty.
        # This does NOT skip the pair entirely -- its fallback attempts (if
        # any) are still extracted below.
        is_pure_dense_result = data.get("anchor_source") == "dense_correlation"

        # A hard-gated rejection (bbox_fill_ratio or overlap_fraction) never
        # reaches full verification -- coarse_structural_search.py's
        # `_decide_verdict` returns before computing NCC/local-cell-agreement
        # in that case, OR (pre-fix) never recorded bbox_fill_ratio at all.
        # Only rows with a complete verification dict reflect a hypothesis
        # that reached the soft-failure cascade this model is meant to
        # replace. ambiguity_ratio lives at the top level of the raw result,
        # not inside `verification` -- merged in here.
        if is_pure_dense_result and verification is not None:
            features = dict(verification)
            features["ambiguity_ratio"] = data.get("ambiguity_ratio")

            if all(features.get(name) is not None for name in DENSE_FEATURE_NAMES) and verification_tier is not None:
                dense_rows.append(
                    {
                        "pair_id": pair_id,
                        "method": "dense_correlation",
                        "verification_tier": verification_tier,
                        "training_weight": TIER_TRAINING_WEIGHTS.get(verification_tier, 0.4),
                        "features": {name: features[name] for name in DENSE_FEATURE_NAMES},
                        "label": 1 if data["verdict"] == "PASS" else 0,
                        "source": str(path),
                    }
                )

        fallback = data.get("keypoint_fallback_result")

        if fallback:
            for attempt in fallback.get("attempts", []):
                if attempt.get("candidates", 0) < 4:
                    # _fit_and_verdict's own hard floor (need >=4 points to
                    # even attempt a RANSAC fit) -- no inlier_ratio/inliers
                    # were computed for these, not a usable feature row.
                    continue

                if verification_tier is None:
                    # This pair's final metrics.json wasn't found/didn't run
                    # far enough to have a tier -- exclude rather than guess.
                    continue

                # verification_tier here is the PAIR's final evidence tier
                # (computed once, downstream, for whichever anchor actually
                # won) -- attached to every attempt within that pair,
                # including a rejected sift_intensity attempt from the same
                # pair, as a TRAINING WEIGHT only (see TIER_TRAINING_WEIGHTS
                # above), never as a feature. It describes "how much to
                # trust this pair's eventual label," not "was this specific
                # attempt itself independently verified" -- there is no
                # per-attempt perturbation check, only one per pipeline run.
                fallback_rows.append(
                    {
                        "pair_id": pair_id,
                        "method": attempt.get("method"),
                        "verification_tier": verification_tier,
                        "training_weight": TIER_TRAINING_WEIGHTS.get(verification_tier, 0.4),
                        "features": {
                            "candidates": attempt.get("candidates"),
                            "inliers": attempt.get("inliers"),
                            "inlier_ratio": attempt.get("inlier_ratio"),
                        },
                        "label": 1 if attempt.get("verdict") == "PASS" else 0,
                        "source": str(path),
                    }
                )

    # Supersede pair_004's stale lightglue_gradient row (4 inliers, rejected,
    # from the un-eroded original mask) with the corrected one from the
    # unified pipeline's own validated regression run (23 inliers, accepted).
    corrected = _load_json(PAIR004_CORRECTED_METRICS_PATH)

    if corrected is not None and corrected.get("fit") is not None:
        fallback_rows = [
            r for r in fallback_rows if not (r["pair_id"] == "pair_004" and r["method"] == "lightglue_gradient")
        ]
        corrected_tier = corrected.get("verification_tier")

        if corrected_tier is not None:
            fallback_rows.append(
                {
                    "pair_id": "pair_004",
                    "method": "lightglue_gradient",
                    "verification_tier": corrected_tier,
                    "training_weight": TIER_TRAINING_WEIGHTS.get(corrected_tier, 0.4),
                    "features": {
                        "candidates": corrected["fit"]["ransac_candidates"],
                        "inliers": corrected["fit"]["ransac_inliers"],
                        "inlier_ratio": corrected["fit"]["ransac_inlier_ratio"],
                    },
                    "label": 1,
                    "source": str(PAIR004_CORRECTED_METRICS_PATH),
                }
            )

    return dense_rows, fallback_rows


@dataclass
class FitOutcome:
    feature_space: str
    status: str  # "fitted" | "insufficient_training_data"
    row_count: int
    rows: list
    model: dict | None = None  # {"feature_names","coefficients","intercept","mean","scale"}


def _try_fit(feature_names, rows):
    if len(rows) < MIN_ROWS_TO_FIT:
        return None

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.array([[r["features"][name] for name in feature_names] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    # verification_tier-derived weight, NOT a feature column -- see
    # TIER_TRAINING_WEIGHTS. Trusts a perturbation_confirmed row's label
    # more than a ransac_and_prior_ground_truth_only row's during fitting;
    # never seen by decide() at live inference.
    sample_weight = np.array([r["training_weight"] for r in rows], dtype=np.float64)

    if len(set(y.tolist())) < 2:
        return None  # can't fit/calibrate a decision boundary from one class

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    clf = LogisticRegression(max_iter=1000).fit(X_scaled, y, sample_weight=sample_weight)

    return {
        "feature_names": feature_names,
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def fit_or_document(dense_rows, fallback_rows):
    outcomes = []

    for feature_space, feature_names, rows in (
        ("dense", DENSE_FEATURE_NAMES, dense_rows),
        ("fallback", FALLBACK_FEATURE_NAMES, fallback_rows),
    ):
        fitted = _try_fit(feature_names, rows)

        outcomes.append(
            FitOutcome(
                feature_space=feature_space,
                status="fitted" if fitted is not None else "insufficient_training_data",
                row_count=len(rows),
                rows=[{k: v for k, v in r.items() if k != "features"} | {"features": r["features"]} for r in rows],
                model=fitted,
            )
        )

    return outcomes


def save_model(outcomes, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "min_rows_to_fit": MIN_ROWS_TO_FIT,
        "spaces": {},
    }

    for outcome in outcomes:
        document["spaces"][outcome.feature_space] = {
            "status": outcome.status,
            "row_count": outcome.row_count,
            "rows": outcome.rows,
            "model": outcome.model,
            "fallback": (
                None
                if outcome.status == "fitted"
                else "existing hand-set thresholds in src/coarse/coarse_structural_search.py "
                "(_decide_verdict) / src/coarse/keypoint_fallback.py (_fit_and_verdict) -- "
                f"only {outcome.row_count} real, non-circular labeled rows exist for this "
                f"decision, below the {MIN_ROWS_TO_FIT}-row floor this project set for fitting "
                "a classifier meaningfully; re-run src/pipeline/anchor_confidence.py once more "
                "validated pairs exist"
            ),
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)

    return path


class AnchorConfidenceModel:
    """
    Loaded once by anchor.py. `decide(feature_space, features)` returns a
    calibrated probability in [0, 1] if a model was actually fitted for that
    feature space, or None if this decision should fall back to the existing
    hand-set cascade (the current, honest state for both spaces -- see
    MODEL_PATH's `fallback` field for why).

    verification_tier never appears in `feature_names` here -- it is used
    only as a training-row sample weight in `_try_fit` (see
    TIER_TRAINING_WEIGHTS), never as an input `decide()` consumes. This
    keeps `feature_names` limited to exactly what Stage 0/1 can actually
    supply live, so a fitted model is always deployable at anchor.py's call
    site with no circularity. The `all(name in features ...)` guard below is
    kept anyway as general defensive practice (a genuinely missing/None
    upstream value should defer to the cascade, not crash), independent of
    the tier question.
    """

    def __init__(self, document):
        self.document = document

    @classmethod
    def load(cls, path=MODEL_PATH):
        path = Path(path)

        if not path.exists():
            return cls({"spaces": {}})

        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def decide(self, feature_space, features):
        space = self.document.get("spaces", {}).get(feature_space)

        if space is None or space["status"] != "fitted" or space["model"] is None:
            return None

        model = space["model"]

        if not all(name in features for name in model["feature_names"]):
            # General defensive guard: if the live caller can't supply every
            # feature the fitted model needs, defer to the hand-set cascade
            # rather than guess or crash. (feature_names is always a subset
            # of DENSE_FEATURE_NAMES/FALLBACK_FEATURE_NAMES -- both fully
            # available at Stage 0/1 -- so this should never actually fire
            # for a missing-by-design reason like verification_tier; it's a
            # safety net for genuinely incomplete upstream data instead.)
            return None

        x = np.array([features[name] for name in model["feature_names"]], dtype=np.float64)
        mean = np.array(model["scaler_mean"], dtype=np.float64)
        scale = np.array(model["scaler_scale"], dtype=np.float64)
        x_scaled = (x - mean) / scale

        z = float(np.dot(x_scaled, model["coefficients"]) + model["intercept"])

        return 1.0 / (1.0 + np.exp(-z))


def main():
    dense_rows, fallback_rows = build_training_table()

    print(f"dense feature-space rows: {len(dense_rows)}")
    for r in dense_rows:
        print(
            f"  {r['pair_id']:<10} label={r['label']}  "
            f"tier={r['verification_tier']:<30} weight={r['training_weight']}  {r['features']}"
        )

    print(f"\nfallback feature-space rows: {len(fallback_rows)}")
    for r in fallback_rows:
        print(
            f"  {r['pair_id']:<10} {r['method']:<18} label={r['label']}  "
            f"tier={r['verification_tier']:<30} weight={r['training_weight']}  {r['features']}"
        )

    outcomes = fit_or_document(dense_rows, fallback_rows)

    for outcome in outcomes:
        print(
            f"\n{outcome.feature_space}: {outcome.row_count} rows -> {outcome.status} "
            f"(need >= {MIN_ROWS_TO_FIT} to fit)"
        )

    path = save_model(outcomes)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
