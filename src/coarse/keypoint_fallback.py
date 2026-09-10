import cv2
import numpy as np

from src.coarse.export import robust_uint8
from src.coarse.robust_fit import affine_sanity, decompose_affine
from src.geometry.ransac_filter import filter_matches_ransac
from src.matching.sift_matcher import match_sift

try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    LIGHTGLUE_AVAILABLE = True
except Exception:
    LIGHTGLUE_AVAILABLE = False


# Named (not just a default parameter value) so callers outside this module
# -- specifically src/pipeline/export.py's sub-pixel honesty check -- can
# compute the same effective-resolution-loss ratio this function's own
# `resize()` step introduces, without duplicating the number or importing
# private internals.
LIGHTGLUE_MAX_DIM = 2048


def gradient_uint8(image_u8, mask):
    """
    Sobel gradient-magnitude representation, matching the one v1's
    Pair002 global benchmark used to actually find a correct anchor
    (see results/pair_002/benchmarks/loftr_gradient/metrics.json) --
    plain intensity is more exposed to the sun-angle difference than
    gradient magnitude is.
    """

    blurred = cv2.GaussianBlur(image_u8.astype(np.float32) / 255.0, (5, 5), 0.8)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    out = np.zeros(image_u8.shape, dtype=np.uint8)
    values = magnitude[mask]

    if values.size == 0:
        return out

    high = np.percentile(values, 99.0)

    if high <= 0:
        return out

    out[mask] = np.round(np.clip(magnitude[mask] / high, 0.0, 1.0) * 255.0).astype(
        np.uint8
    )

    return out


def run_lightglue_on_arrays(
    source_u8, reference_u8, source_mask=None, reference_mask=None, max_dim=LIGHTGLUE_MAX_DIM
):
    """
    SuperPoint + LightGlue directly on in-memory uint8 arrays (no file
    round-trip), mirroring v1's scripts/benchmark_pair007_global.py
    pattern rather than src/matching/lightglue_matcher.py's file-path
    based class, since this pipeline already has the arrays loaded.

    `source_mask`/`reference_mask`, if given, filter OUT any match whose
    point falls outside the valid region -- unlike `match_sift` (which
    restricts detection itself via OpenCV's mask parameter), SuperPoint
    has no equivalent, so without this a large injected perturbation's
    zero-fill border can produce many spurious matches on the
    border/background shape rather than real terrain (confirmed: on
    Pair002 this produced MORE inliers, 506, than the unperturbed match
    (386), all converging on a near-zero recovered translation that
    ignored the actual injected shift entirely).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resize(image):
        h, w = image.shape
        current_max = max(h, w)

        if current_max <= max_dim:
            return image.copy(), 1.0

        scale = max_dim / current_max
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    src_small, scale = resize(source_u8)
    ref_small, ref_scale = resize(reference_u8)

    if abs(scale - ref_scale) > 1e-6:
        raise RuntimeError("Source/reference resize scales differ.")

    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    def to_tensor(image):
        return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).to(device)

    with torch.inference_mode():
        feats0 = extractor.extract(to_tensor(src_small))
        feats1 = extractor.extract(to_tensor(ref_small))
        pred = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, pred = rbd(feats0), rbd(feats1), rbd(pred)

    keypoints0 = feats0["keypoints"].detach().cpu().numpy()
    keypoints1 = feats1["keypoints"].detach().cpu().numpy()
    matches = pred["matches"].detach().cpu().numpy()

    del extractor, matcher

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if matches.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    source_points = (keypoints0[matches[:, 0]] / scale).astype(np.float32)
    reference_points = (keypoints1[matches[:, 1]] / scale).astype(np.float32)

    def points_inside(points, mask):
        if mask is None:
            return np.ones(len(points), dtype=bool)

        height, width = mask.shape
        x = np.rint(points[:, 0]).astype(np.int64)
        y = np.rint(points[:, 1]).astype(np.int64)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)

        result = np.zeros(len(points), dtype=bool)
        result[inside] = mask[y[inside], x[inside]]

        return result

    keep = points_inside(source_points, source_mask) & points_inside(
        reference_points, reference_mask
    )

    return source_points[keep], reference_points[keep]


def _fit_and_verdict(method_name, source_points, reference_points, config):
    candidate_count = len(source_points)

    if candidate_count < 4:
        return {
            "method": method_name,
            "candidates": candidate_count,
            "verdict": "FAIL",
            "reasons": [f"only {candidate_count} raw candidates, need at least 4"],
        }

    matrix, inlier_mask, inlier_source, inlier_reference = filter_matches_ransac(
        source_points,
        reference_points,
        reprojection_threshold=config["ransac_reprojection_threshold_px"],
    )

    params = decompose_affine(matrix)
    sane = affine_sanity(params, config)

    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count) / float(candidate_count)

    reasons = []

    if not sane:
        reasons.append("affine failed shape-sanity check")

    if inlier_count < config["min_fallback_inliers"]:
        reasons.append(
            f"only {inlier_count} inliers, below "
            f"min_fallback_inliers={config['min_fallback_inliers']}"
        )

    if inlier_ratio < config["min_fallback_inlier_ratio"]:
        reasons.append(
            f"inlier_ratio={inlier_ratio:.3f} below "
            f"min_fallback_inlier_ratio={config['min_fallback_inlier_ratio']}"
        )

    verdict = "PASS" if not reasons else "FAIL"

    return {
        "method": method_name,
        "matrix": matrix,
        "params": params,
        "sane": sane,
        "candidates": candidate_count,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "inlier_source_points": inlier_source,
        "inlier_reference_points": inlier_reference,
        "verdict": verdict,
        "reasons": reasons,
    }


def run_keypoint_fallback(source_image, source_mask, reference_image, reference_mask, config):
    """
    Coarse-anchor fallback for pairs where Stage 0's dense structural
    search can't find a confident peak at all -- confirmed (not
    assumed) on Pair002 (OHRC vs TMC-2): the oriented-gradient
    structural correlation score at the true, independently-known-good
    offset is actually NEGATIVE there (anti-correlated), not just weak,
    and stays negative under both heavier smoothing and local-contrast
    normalization. OHRC's near-grazing sun elevation (-0.04 deg) versus
    TMC's +7 deg creates an illumination difference severe enough that
    simple oriented-gradient magnitude isn't the right descriptor for
    this pair -- a real capability gap in the descriptor, not a
    tunable parameter.

    Tries, in order, v1's already-proven matchers: SIFT on intensity
    (cheap, no GPU, but confirmed on Pair002 to find only 11 raw
    candidates -- SIFT's descriptor isn't illumination-robust enough for
    this gap either), then SuperPoint+LightGlue on GRADIENT
    representation (confirmed on Pair002, matching
    results/pair_002/benchmarks/loftr_gradient/metrics.json, to find the
    correct near-identity anchor with 22 sane RANSAC inliers -- gradient
    magnitude and a learned descriptor together are what actually works
    here). Stops at the first method that passes.

    Important: this does NOT bypass the project's core discipline. An
    anchor found this way still has to survive Stage 1's independent
    dense local-point re-verification and Stage 2's interior-ROI
    perturbation validation exactly like a dense-correlation anchor
    would -- this function only replaces how the INITIAL anchor is
    found, not how it gets confirmed. That distinction is what stops
    this from repeating v1's original mistake on Pair007 (trusting a
    keypoint-RANSAC hypothesis without independent dense verification).
    Thresholds here are deliberately looser than Stage 2's -- this
    function's only job is producing a plausible coarse anchor, the
    same role v1's raw global benchmark step played before prior-guided
    refinement, not the final word on whether the pair registers.
    """

    common_mask = source_mask & reference_mask

    source_u8 = robust_uint8(source_image, common_mask)
    reference_u8 = robust_uint8(reference_image, common_mask)

    attempts = []

    try:
        source_points, reference_points, _good_matches, _, _ = match_sift(
            source_u8,
            reference_u8,
            mask=common_mask,
            ratio_threshold=config["sift_ratio_threshold"],
            max_features=config["sift_max_features"],
        )

        result = _fit_and_verdict("sift_intensity", source_points, reference_points, config)
    except RuntimeError as exc:
        result = {
            "method": "sift_intensity",
            "candidates": 0,
            "verdict": "FAIL",
            "reasons": [f"SIFT matching failed: {exc!r}"],
        }

    attempts.append(result)

    if result["verdict"] == "PASS":
        result = dict(result)
        result["attempts"] = attempts
        return result

    if LIGHTGLUE_AVAILABLE:
        source_grad = gradient_uint8(source_u8, common_mask)
        reference_grad = gradient_uint8(reference_u8, common_mask)

        try:
            source_points, reference_points = run_lightglue_on_arrays(
                source_grad, reference_grad, source_mask=common_mask, reference_mask=common_mask
            )

            result = _fit_and_verdict(
                "lightglue_gradient", source_points, reference_points, config
            )
        except (RuntimeError, ValueError) as exc:
            # Deliberately narrow: covers genuine matcher-level failures
            # (explicit RuntimeErrors, CUDA OOM -- a RuntimeError subclass
            # in torch) as an honest fallback-failed result. A bare
            # `except Exception` here would also silently reclassify real
            # programming bugs (e.g. AttributeError from a typo) as an
            # ordinary matcher failure, hiding them from debugging -- let
            # those propagate instead.
            result = {
                "method": "lightglue_gradient",
                "candidates": 0,
                "verdict": "FAIL",
                "reasons": [f"LightGlue matching failed: {exc!r}"],
            }

        attempts.append(result)

    best = dict(next((a for a in attempts if a["verdict"] == "PASS"), attempts[-1]))
    best["attempts"] = attempts

    return best
