# ChandraMatch

A self-auditing correspondence pipeline for Chandrayaan-2 optical imagery against lunar reference data (LRO NAC, SELENE/Kaguya) — sub-pixel where the evidence supports it, uniformly distributed across the frame, with disclosed PASS / AMBIGUOUS / FAIL verdicts instead of a silent best-effort guess.

## Links

- **Problem Statement:** SIH26166 (ISRO, Software, Space Technology)
- **Full report:** [REPORT LINK -- ADD ONCE UPLOADED]
- **Video walkthrough:** [YOUTUBE LINK -- ADD ONCE UPLOADED]
- **Live demo:** [LIVE DEMO -- see "Running locally" below if not yet deployed]

## The problem

Lunar images taken by different instruments, at different times, rarely look alike. Sun-angle differences change shadow direction and length across the same terrain; native resolutions between sensors in this project differ by as much as 25x; and much of the lunar surface is visually repetitive cratered terrain with few unambiguous landmarks. Standard correspondence matchers — SIFT, LoFTR, LightGlue — are not designed to refuse an answer: given a hard scene, they can converge on a confident, RANSAC-clean, geometrically consistent match that is simply wrong, with nothing in the output to distinguish it from a correct one. ChandraMatch is built around the opposite default: every run ends in an explicit, evidence-backed verdict, and a run that can't be trusted says so instead of shipping a number.

## Architecture

Every pair — regardless of sensor combination — goes through one public entry point, `register(source_path, reference_path, config, out_dir)`, used identically by the CLI, the regression harness, and the web app. It runs a fixed seven-stage sequence and always writes a full result bundle, even when it stops early:

1. **Sensor ID** — identifies each input (PDS3/PDS4 label inspection first, then filename convention, then a resolution heuristic), never silently guessing "unknown."
2. **Ingest** — runs ISIS calibration/map-projection for raw LRO NAC `.IMG` (via WSL); passes GeoTIFF inputs through unchanged.
3. **Canonicalize** — reprojects source and reference onto a common grid and builds a science-validity mask and a separate, eroded matcher mask.
4. **Stage 0 — coarse search** — dense oriented-gradient structural correlation, with a disclosed fallback to keypoint matching (SIFT, then LightGlue) when the dense hypothesis is hard-gated out or ambiguous. An anchor-confidence classifier gate exists but is currently inert (see Known limitations) — the hand-set threshold cascade is the live decision path.
5. **Stage 1 — local control points** — independent local re-verification of the coarse anchor at native resolution, when the anchor allows it.
6. **Stage 2 — registration + validation** — robust affine fit, then a synthetic-perturbation stress test that measures whether the fit actually tracks injected motion, not just whether it looks self-consistent.
7. **Export** — writes the registered product, match points, metrics, and visualizations for every run, regardless of outcome.

See `docs/UNIFIED_PIPELINE_CHECKPOINT_LOG.md` for the implementation-level detail and the full report (above) for the underlying methodology.

## Results

| Pair | Sensors | Anchor method | Verdict | Translation (px) | Inliers |
|---|---|---|---|---|---|
| 002 | OHRC ↔ TMC-2 | Keypoint fallback (LightGlue) | PASS | 0.77 / 0.70 | 333 / 355 |
| 003 | IIRS ↔ TMC-2 | Keypoint fallback (LightGlue) | PASS | ~0 / 0 | 16 / 16 |
| 004 | LRO NAC ↔ TMC-2 | Keypoint fallback (dense search gated out) | PASS | 0.00 / 0.00 | 23 / 23 |
| 005 | Kaguya TC ↔ TMC-2 | Keypoint fallback (SIFT) | PASS | 846.67 / -5.66 | 558 / 564 |
| 006 | LRO NAC ↔ TMC-2 | Dense structural correlation | PASS | 692.13 / -11.70 | 104 / 104 |
| 007 | LRO NAC ↔ TMC-2 | Dense structural correlation | AMBIGUOUS | 905.35 / -2.51 | 39 / 39 |

Pair 007's AMBIGUOUS verdict is a disclosed, honest limit — its post-registration overlap is too thin for the standard perturbation test to run at full strength — not a failure being hidden.

## Running locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Run a single pair through the pipeline:**

```bash
python scripts/regression_register_pair.py --pair pair_006
```

(`--pair` accepts `pair_002`, `pair_003`, `pair_004`, `pair_005`, `pair_006`, or `pair_007`; output is written to `results/unified_regression/<pair_id>/`.)

**Start the web app:**

```bash
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000** and upload a source and reference image.

Raw `.IMG` ingestion currently requires a WSL environment with ISIS3 installed and is LRO-NAC-only (see `src/pipeline/ingest.py`). GeoTIFF inputs work directly, with no WSL/ISIS dependency, for every supported sensor.

## Repository structure

```
src/pipeline/   the unified pipeline -- register() and every stage it calls
src/coarse/     the underlying dense-search / keypoint-fallback / validation algorithms
app/            FastAPI web app (upload -> register() -> download results)
scripts/        CLI entry points, incl. the regression harness used above
results/        per-pair output bundles (metrics.json, registered product, visualizations)
docs/           project status and the detailed pipeline checkpoint log
```

## Data provenance

This repository's **code** is licensed under the MIT License (see `LICENSE`). The **imagery** it processes is not owned by this project:

- LRO NAC data is public-domain NASA archive data from the Planetary Data System (PDS).
- Chandrayaan-2 data (TMC-2, OHRC, IIRS) is from ISRO's public ISSDC archive.
- SELENE/Kaguya Terrain Camera data is from JAXA's public lunar data archive.

Nothing here claims ownership of, or exclusive rights to, the source imagery — only the registration pipeline code itself.

## Known limitations

- **Raw ingestion is LRO-NAC-only.** OHRC, IIRS, and Kaguya raw archive formats have no automated ingestion path yet; they're processed here from already-mapped GeoTIFF inputs.
- **Sub-pixel precision is disclosed per pair, never assumed.** `sub_pixel_achieved` is `True` only when the anchor came from dense correlation and an independent native-resolution local refinement measured residuals under one pixel — a keypoint-fallback anchor's local refinement reuses the same points that defined the anchor, so it can never support a sub-pixel claim, no matter how tight the resulting number looks.
- **Four of the six validated pairs (002, 003, 004, 005) rely on RANSAC fit quality and separately-established ground truth, not this pipeline's own independent perturbation test.** Their anchors come from keypoint matching, and a keypoint matcher re-verifying its own anchor under injected motion is a documented blind spot — it can recover a false "no motion detected" result regardless of how much motion was actually injected, so that test cannot arbitrate this anchor class at all.
- **The anchor-confidence ML classifier is built and wired in, but currently inert.** Both its feature spaces (dense-search and fallback-fit) have too few labeled rows to fit safely — well under the 20-row floor this project set for fitting responsibly — so it reports `insufficient_training_data` and the existing hand-set threshold cascade remains the live decision path.
