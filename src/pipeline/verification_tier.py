"""
Evidence-tier labeling for the synthetic-perturbation stress test.

`perturbation_validation_informational_only` already existed to keep a
keypoint-fallback anchor's (uninformative, identity-bias-prone) perturbation
result out of the verdict computation. This module makes that same
distinction explicit and inspectable in metrics.json, and adds the one
gradation that distinction was previously missing: a dense-correlation
anchor's perturbation test can itself land in three different evidentiary
states (fully confirmed, genuinely failed, or untestable-with-partial-
supplementary-support), and those are not equivalent even though only one of
them ("fully confirmed") should ever be reported as independent proof.

Verification-tier assignment is DATA-DRIVEN (computed from each run's own
validation result), never a hardcoded pair-name lookup table -- so it stays
correct if a future pair's actual evidence doesn't match what a similar pair
produced before.
"""


def compute_perturbation_verification(anchor_source, fit_result):
    """
    Returns (perturbation_verified: bool, perturbation_reason: str|None,
    verification_tier: str).

    verification_tier is one of:
      "perturbation_confirmed"  -- dense anchor, full-strength standard
                                    perturbation set passed cleanly (the only
                                    tier that counts as independent proof).
      "perturbation_partial"    -- dense anchor, standard test untestable
                                    (overlap too thin for the safety margin),
                                    but the reduced-scale supplementary test
                                    found at least one perturbation that
                                    passed -- real but weaker evidence than
                                    "confirmed", never described as
                                    equivalent to it.
      "perturbation_untestable" -- dense anchor, standard test untestable and
                                    no reduced-scale support either.
      "perturbation_failed"     -- dense anchor, a perturbation WITH real
                                    interior-ROI pixels available still
                                    exceeded the response threshold -- a
                                    genuine motion-tracking failure, not a
                                    thin-overlap limitation.
      "ransac_and_prior_ground_truth_only" -- keypoint-fallback anchor; the
                                    perturbation test cannot arbitrate this
                                    anchor class at all (identity bias: the
                                    same matcher family re-verifying its own
                                    anchor under large injected motion is a
                                    documented, reproduced blind spot, not
                                    evidence for or against correctness).
      "unverified_no_fit"       -- Stage 2 never produced a fit to validate.
    """

    if anchor_source == "none":
        # register() stopped before any anchor was even resolved (unknown
        # sensor, unsupported raw format, or Stage 0 FAIL/AMBIGUOUS) --
        # distinct from a keypoint-fallback anchor's known blind spot.
        return False, "pipeline_stopped_before_anchor_resolution", "unverified_no_fit"

    if anchor_source != "dense_correlation":
        return (
            False,
            "reverification_method_shares_anchor_method_class_known_blind_spot",
            "ransac_and_prior_ground_truth_only",
        )

    validation = fit_result.get("validation")

    if validation is None:
        return False, "stage2_did_not_run", "unverified_no_fit"

    if validation["all_pass"]:
        return True, None, "perturbation_confirmed"

    controls = validation["controls"]
    all_untestable = bool(controls) and all(c["interior_roi_pixels"] == 0 for c in controls)

    if not all_untestable:
        return False, "perturbation_response_exceeded_threshold", "perturbation_failed"

    reduced = fit_result.get("reduced_scale_validation")

    if reduced is not None and reduced.get("possible") and any(
        c["pass"] for c in reduced.get("controls", [])
    ):
        return (
            False,
            "untestable_standard_scale_partial_reduced_scale_support",
            "perturbation_partial",
        )

    return False, "untestable_thin_overlap_no_supplementary_pass", "perturbation_untestable"
