# ChandraMatch v2 Research Decision 001

## Scope

This note records the literature review and implementation decision made before changing the registration pipeline. It is intentionally conservative: Pair004 and Pair006 remain frozen evidence, Pair007 remains a held-out failure of the frozen v1 end-to-end matcher, and no pair-specific parameters or manual offsets are permitted.

## Problem framing

The current evidence says that the dominant failure mode is not simply geometric estimation after a good match set. Pair007 reaches a plausible coarse translation basin (~900 px) but the learned fine-match branches fail controlled perturbation robustness. The v2 coarse stage therefore needs to rely on cross-modal structural evidence rather than raw intensity or a single learned correspondence model.

The initial v2 replay should also be translation-first. Pair004 is near identity, Pair006 is approximately a 692 px translation magnitude, and Pair007 is approximately a 904 px translation magnitude. Rotation/scale-capable methods remain valuable as later fallbacks, but adding those degrees of freedom to the first coarse search would increase ambiguity and implementation complexity without evidence that they are currently required.

## Literature reviewed

### CFOG + FFT matching

Y. Ye, L. Bruzzone, J. Shan, F. Bovolo, and Q. Zhu, "Fast and Robust Matching for Multimodal Remote Sensing Image Registration," IEEE Transactions on Geoscience and Remote Sensing, 57(11), 9059-9070, 2019. DOI: 10.1109/TGRS.2019.2924684.

Paper: https://doi.org/10.1109/TGRS.2019.2924684

Key points relevant to ChandraMatch:
- forms a pixel-wise structural representation with Channel Features of Oriented Gradients (CFOG),
- performs matching in the frequency domain using FFT,
- is explicitly designed for nonlinear radiometric differences in multimodal remote-sensing imagery,
- separates robust structural representation from efficient search.

This is the closest match to the immediate ChandraMatch need: a translation-dominant coarse hypothesis search that is not tied to raw intensity.

### SFOC / Fast-NCCSFOC

B. Zhu, J. Zhang, T. Tang, and Y. Ye, "SFOC: A Novel Multi-Directional and Multi-Scale Structural Descriptor for Multimodal Remote Sensing Image Matching," ISPRS Archives, XLIII-B2-2022, 113-120, 2022. DOI: 10.5194/isprs-archives-XLIII-B2-2022-113-2022.

Paper: https://doi.org/10.5194/isprs-archives-XLIII-B2-2022-113-2022

Related full registration formulation: Y. Ye et al., "A Robust Multimodal Remote Sensing Image Registration Method and System Using Steerable Filters with First- and Second-order Gradients," 2022 preprint: https://arxiv.org/abs/2202.13347

Key points:
- SFOC combines first- and second-order steerable-filter channels,
- uses multiple directions and scales to describe cross-modal structure,
- Fast-NCCSFOC combines FFT-based correlation with integral-image normalization,
- its extra second-order information is attractive for lunar terrain, where edges, ridges, and crater curvature can remain structurally meaningful despite illumination/sensor differences.

SFOC is the preferred direction after the first reproducible CFOG-style baseline because it is more discriminative, but implementing it faithfully requires additional algorithmic detail and careful validation rather than an ad-hoc approximation.

### RIFT / RIFT2

J. Li, Q. Hu, and M. Ai, "RIFT: Multi-modal Image Matching Based on Radiation-variation Insensitive Feature Transform," IEEE Transactions on Image Processing, 29, 3296-3310, 2020.

Paper: https://arxiv.org/abs/1804.09493
Official code: https://github.com/LJY-RS/RIFT-multimodal-image-matching

RIFT uses phase congruency for feature detection and a Maximum Index Map built from log-Gabor responses for description. Its strengths are strong nonlinear-radiometric robustness and feature-level matching across very different modalities. RIFT2 (https://arxiv.org/abs/2303.00319) reduces the rotation-invariance cost.

Decision for ChandraMatch: keep RIFT/RIFT2 as a local control-point fallback/corroboration stage after the coarse basin is established. Do not make it the first global search, because global dense structural correlation is simpler to score for ambiguity and overlap and better matches the current translation-dominant evidence.

### Radiation-Invariant Phase Correlation

T. Peng, L. Zhou, G. Lei, P. Yang, and Y. Ye, "Robust Multimodal Image Matching Based on Radiation Invariant Phase Correlation," ISPRS Annals X-3-2024, 309-316, 2024. DOI: 10.5194/isprs-annals-X-3-2024-309-2024.

Paper: https://doi.org/10.5194/isprs-annals-X-3-2024-309-2024

This work combines a radiation-insensitive descriptor, log-polar handling of rotation/scale, and phase correlation. It is a strong candidate if replay evidence shows that translation-only structural correlation is insufficient or if future scenes contain meaningful global rotation/scale residuals.

### Other structural multimodal work

S. Cui, M. Xu, A. Ma, and Y. Zhong, "Modality-Free Feature Detector and Descriptor for Multimodal Remote Sensing Image Registration," Remote Sensing 12(18), 2937, 2020. DOI: 10.3390/rs12182937.

Paper: https://doi.org/10.3390/rs12182937

This work reinforces the general design principle used here: modality-robust registration benefits from mapping images into a structural representation rather than directly comparing raw intensity/ordinary gradients.

A more recent 2025 example, Cof-SIFT / Cof-SIFT_HOG, also reports gains from suppressing modality-specific texture while retaining edges/contours and using HOG-like structural matching (Remote Sensing 17(13), 2246; https://doi.org/10.3390/rs17132246). It is useful corroborating evidence but does not displace CFOG/SFOC as the first dense coarse-search baseline.

## Candidate comparison

| Candidate | Global coarse translation search | NRD robustness | Rotation/scale | Computational profile | Decision |
|---|---|---|---|---|---|
| raw phase correlation | excellent | weak-to-moderate | translation only | very fast | diagnostic only |
| CFOG + FFT | excellent | strong structural robustness | translation-first | fast | **implement first** |
| SFOC + Fast-NCC | excellent | stronger structural representation | translation-first in chosen use | moderate | **next upgrade after baseline** |
| RIFT/RIFT2 | local sparse matching | very strong | strong | moderate/high | local fallback/corroboration |
| radiation-invariant phase correlation | strong | strong | rotation + scale + translation | moderate | future global fallback if needed |
| learned LightGlue/LoFTR | correspondence-driven | data/model dependent | implicit | GPU-efficient but failure can be brittle | corroboration/fallback, not sole evidence |

## v2 implementation decision

### Stage 1: reproducible CFOG-style structural coarse baseline

Implement a dependency-free (relative to the already-installed environment) structural representation using NumPy/OpenCV only:
1. robust grayscale normalization restricted to valid pixels,
2. first-order Sobel gradients,
3. unsigned orientation-channel soft assignment,
4. local Gaussian smoothing of each channel,
5. pyramid downsampling for a global search,
6. FFT/correlation scoring over structural channels,
7. top-K translation hypotheses rather than a single argmax,
8. score margin / peak-ratio / overlap diagnostics for ambiguity.

The implementation must be labelled **CFOG-style** unless it exactly reproduces the paper formulation. This avoids overstating methodological fidelity.

### Stage 2: SFOC upgrade

After the CFOG-style baseline is replayed identically on Pair004/006/007, add a faithful first+second-order steerable-filter representation and compare it under the same search/decision framework. A SFOC change is global: all three replay scenes must be rerun with identical parameters.

### Stage 3: local structural control points

Once a global basin is selected, estimate local translations on a uniform interior grid using structural templates. Reject points with low peak margin, poor overlap, or locally inconsistent displacement. RIFT/RIFT2 can later be evaluated as a fallback/corroboration branch if its implementation is already available locally or if the user authorizes any required download.

## Scoring and scientific constraints

The coarse stage must return more than an offset. At minimum it should record:
- selected dx/dy,
- top-K hypotheses and scores,
- best/second-best score margin or ratio,
- estimated overlap fraction,
- structural similarity before/after the selected warp,
- boundary-contact flag (to identify truncated searches),
- pyramid level / scale,
- deterministic parameter set.

The following are explicitly prohibited as correctness claims:
- calling RANSAC reprojection RMSE independent registration accuracy,
- calling perturbation response absolute ground truth,
- changing thresholds/offsets for Pair007 alone,
- converting Pair007's historical v1 failure into a v1 pass.

Pair007 may pass a future **v2** validator only if the v2 method and thresholds are declared globally in advance and replayed unchanged on Pair004, Pair006, and Pair007. Its v1 result remains a recorded failure regardless.

## No-download constraint

No new package, model weight, external repository, dataset, or binary is required for the first CFOG-style baseline. The current runner health check confirms the existing local environment already contains NumPy, OpenCV, Rasterio, PyTorch/CUDA, Kornia, WSL, and ISIS. Any later proposal that requires an external implementation or pretrained weight must be paused until explicit user permission is obtained.

## Acceptance gate for the next implementation step

Before local fine matching is added, the coarse module must:
1. execute with one frozen parameter configuration across Pair004/006/007,
2. produce deterministic machine-readable top-K hypotheses,
3. recover a basin consistent with the previously observed regimes without using those expected offsets as inputs,
4. expose ambiguity rather than forcing a PASS,
5. preserve Pair007's v1 failure status separately from any v2 result.
