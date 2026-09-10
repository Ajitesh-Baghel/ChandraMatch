# Unified Pipeline — Checkpoint Log

Durable record of the `src/pipeline/` unification pass (SIH26166 final-product
refactor), kept up to date as checkpoints complete. Supersedes nothing in
`docs/AGENT_STATE.md` — that file is the earlier autonomous-R&D plan; this
log is the actual outcome.

## Regression status: 2 of 5 true regressions, 3 of 5 explained divergences

**Do not describe this as "5/5 regression passed" anywhere.** Two different
claims are being made and must stay separate:

- **Category A — true regression (identical inputs, output must match):**
  Pair006, Pair007. Ran the OLD scripts and the NEW `register()` on the
  literal same canonical files; translation/rotation/inlier counts and
  verdict/reasons text matched exactly.
- **Category B — verified against harder, more-realistic inputs; divergence
  root-caused, not a like-for-like check:** Pair002, Pair003, Pair004. Fed
  `register()` a near-raw source plus the FULL raw TMC-2 reference scene
  (not the old pair-specific pre-crop) so canonicalization runs for real.
  Every divergence traces to one deliberate change — standardizing on
  `build_canonical_pair`'s science/matcher mask convention instead of each
  pair's old bespoke mask file — confirmed by direct mask-overlap
  measurement (Pair002: only 61% pixel overlap between old and new masks),
  not by assumption.

## Verification tiers (all 5 pairs, backfilled)

| Pair | anchor_source | verdict | `perturbation_verified` | `verification_tier` |
|---|---|---|---|---|
| 006 | dense_correlation | PASS | **True** | `perturbation_confirmed` |
| 007 | dense_correlation | AMBIGUOUS | False | `perturbation_partial` (standard test untestable — thin post-registration overlap; reduced-scale supplementary test: 1/3 perturbations pass cleanly, 2/3 have too few interior points to judge) |
| 002 | keypoint_fallback_lightglue_gradient | PASS | False | `ransac_and_prior_ground_truth_only` |
| 003 | keypoint_fallback_lightglue_gradient | PASS | False | `ransac_and_prior_ground_truth_only` |
| 004 | keypoint_fallback_lightglue_gradient | PASS | False | `ransac_and_prior_ground_truth_only` |

**Plain statement: 3 of 5 currently-validated pairs (002/003/004) have no
independent perturbation check at all.** This is a known, previously-disclosed
limitation (identity bias: a keypoint matcher re-verifying its own
keypoint-derived anchor under large injected motion cannot distinguish "the
anchor is right, nothing moved" from "I can't see what moved" — first
documented on Pair005/Pair002/Pair003), not a new finding. What IS new this
pass: running the standard 3-axis stress test against Pair004's corrected
anchor reproduced the *exact* signature a fourth time — two of three
perturbations recovered a perfectly identity affine (zero response) against
real 80px/100px injected motion. Full numbers are in
`results/unified_regression/pair_004/metrics.json`'s `perturbation_summary`.
Per instruction, this does not revert Pair004's verdict (the failure mode is
already proven uninformative for this anchor class, not new evidence against
the anchor) — it's recorded as `perturbation_verified: False`,
`perturbation_reason: reverification_method_shares_anchor_method_class_known_blind_spot`.

**Roadmap item, explicitly NOT in scope for this pass:** a perturbation oracle
based on dense structural correlation rather than keypoint re-matching, that
doesn't share the keypoint matcher's blind spot and remains usable even on
pairs where dense correlation was rejected as the primary anchor (i.e., an
independent second opinion instead of a same-family re-check). No design work
has started on this.

## `sub_pixel_achieved` — fixed, all 5 pairs audited

Old behavior (bug): derived from whatever residual happened to be attached to
the fit, regardless of anchor type — produced `sub_pixel_achieved: True` with
an exact-zero `rmse_px` on Pair003 that was actually a downsampling
quantization artifact (LightGlue's fixed 2048px cap on a ~150M-pixel canvas →
~10.8px effective resolution), not real precision.

New rule (`src/pipeline/sub_pixel.py`): `True` only for a `dense_correlation`
anchor (the only case where Stage 1 independently re-measures at native
resolution rather than reusing the anchor's own points) with residual RMSE
under one native pixel.

| Pair | canvas (px) | `sub_pixel_achieved` | `sub_pixel_reason` | effective resolution |
|---|---|---|---|---|
| 006 | 5040×1886-ish | True | — | 1.0 px |
| 007 | similar | True | — | 1.0 px |
| 002 | 990×832 | False | `stage2_is_coarse_anchor_reuse` | **1.0 px** (audited explicitly — no downsampling occurred; disqualified by architecture, not resolution loss) |
| 004 | 5040×1825 | False | `anchor_downsample_exceeds_native_resolution` | 2.46 px |
| 003 | 22041×6845 | False | `anchor_downsample_exceeds_native_resolution` | 10.76 px |

## Anchor-confidence classifier (§5b): not fitted — insufficient real data

Built the full training-table assembly, fit/save/load machinery
(`src/pipeline/anchor_confidence.py`, scikit-learn `LogisticRegression`,
already a project dependency). Ran it against the real, honestly-usable
labeled rows this project actually has:

- **Dense soft-cascade decision:** 2 usable rows (Pair006, Pair007 — both
  label PASS). Pairs 002/003/004/005 do NOT contribute dense rows: each
  one's dense hypothesis was rejected and a fallback anchor took over, and
  `run_stage0_with_fallback`'s returned `verification` in that case is
  recomputed at the FALLBACK's anchor while `ambiguity_ratio` is left over
  from the original, different, rejected dense hypothesis — merging those
  would combine two different anchors' numbers into one meaningless feature
  vector, so they're correctly excluded rather than included as noise.
  Pair004's dense attempt is separately excluded because it's rejected at
  the hard `bbox_fill_ratio` physical gate before reaching the soft cascade
  at all. 2 rows, both one class — could not support a decision boundary
  even before the 20-row floor is considered.
- **Fallback RANSAC-fit decision:** 7 rows (3 rejected SIFT attempts across
  pairs 002/003/004, plus 4 accepted attempts — 3 LightGlue across
  002/003/004 and 1 SIFT for pair_005, whose own SIFT attempt was the
  winning anchor rather than a rejected one).

Both are far below the 20-row floor the original spec itself set as the point
below which fitting is unsafe. Per that spec's own explicit contingency:
**no model was fit.** `src/pipeline/models/anchor_confidence_v1.json` records
`status: "insufficient_training_data"` for both feature spaces, the exact
rows, and the fallback (existing hand-set thresholds, unchanged). This is
re-run automatically once enough validated pairs exist —
`python -m src.pipeline.anchor_confidence`.

### `verification_tier` is a training-row weight, never a live feature — a bug, caught and fixed

An earlier pass of this module one-hot-encoded `verification_tier` directly
into each row's feature vector (`verification_tier__<category>` × 6 columns)
for both feature spaces. **This was leakage from the future and has been
removed.** `verification_tier` is computed in Stage 4, from Stage 2/3's
perturbation test — a test that only runs *after* an anchor has already been
accepted at Stage 0/1. A live acceptance model decides *before* that test
exists, so it can never actually receive this value at the moment it needs to
decide, no matter how it's encoded. The bug was self-evident in its own
output: the fallback feature space's tier column was constant across every
row (every fallback anchor gets `ransac_and_prior_ground_truth_only` by
construction — the tier logic gives every non-dense anchor the same category),
and a column that cannot vary within a feature space cannot predict anything
in it — a tell that it never belonged there as a feature.

What `verification_tier` legitimately is: evidence about how much to *trust a
training row's label*, applied only when fitting a model, never seen by
`decide()` at live inference. `perturbation_confirmed` rows (label backed by
an independent stress test) get `sample_weight = 1.0`; `perturbation_partial`
rows get `0.7`; every other tier (`ransac_and_prior_ground_truth_only`,
`unverified_no_fit`, `perturbation_untestable`, `perturbation_failed` — real
evidence, just not independently perturbation-confirmed) gets `0.4`. Applied
via scikit-learn's native `LogisticRegression.fit(..., sample_weight=...)`,
not a workaround. `AnchorConfidenceModel.decide()`'s feature list is back to
exactly `DENSE_FEATURE_NAMES`/`FALLBACK_FEATURE_NAMES` — the same 7/3 numeric
Stage 0/1 values `anchor.py` can actually supply live — so a model fitted
under this scheme is always deployable with no circularity, unlike the
one-hot version. Re-running `python -m src.pipeline.anchor_confidence` after
the fix reproduces the same 2/6 row counts and the same
`insufficient_training_data` outcome — the fix changes the classifier's
internal design, not any pair's live registration output (confirmed via the
Pair006 smoke test, unchanged).

Also renamed the tier value itself, `ransac_and_v1_ground_truth_only` →
`ransac_and_prior_ground_truth_only`, everywhere it appeared in code
(`src/pipeline/verification_tier.py`, `src/pipeline/anchor_confidence.py`,
`src/pipeline/export.py`'s interpretation notes) and in all 5 pairs'
`metrics.json`, for consistency with this project's "one continuous project,
no v1/v2 language" framing — grepped the full repo afterward to confirm zero
remaining occurrences of the old string.

### Path to activation

The **dense** feature space is unlikely to reach a usable row count without
many more dense-correlation-anchored pairs being validated — only pairs whose
anchor survives the hard physical gates *and* the soft cascade as a
dense-correlation hypothesis contribute here, and this project has validated
exactly 2 so far. The **fallback** feature space could reasonably activate
sooner, once enough additional keypoint-fallback pairs are processed — it
already has 7 rows across 2 classes (Pair005's migration added one), just
over a third of the 20-row floor.
Note explicitly: the fallback space's tier-derived sample weight is currently
uniform (every fallback row gets weight 0.4, since every fallback anchor gets
the same tier by construction) — this dimension of the weighting scheme has
no effect on fallback-space fitting until some future architectural change
lets fallback anchors receive differentiated perturbation evidence (the
roadmap item above, a dense-correlation-based perturbation oracle usable even
for fallback-anchored pairs, would be exactly such a change).

## Web app wired to register()

`app/pipeline_runner.py` and `app/main.py` now call `src.pipeline.sensor_id`
and `src.pipeline.run.register()` directly, superseding the older
`app/sensor_detection.py` and `app/isis_ingestion.py` modules — both are
intentionally left in place as orphaned (zero remaining importers, confirmed
by grep), not accidentally-forgotten dead code.
