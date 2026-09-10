# ChandraMatch

### Multi-Modal, Sun-Angle and Scale-Invariant Lunar Image Correspondence

ChandraMatch is a lunar image correspondence and registration framework developed for **Smart India Hackathon 2026 – Problem Statement 26166**, proposed by the Indian Space Research Organisation (ISRO).

The objective is to identify reliable and spatially distributed correspondence points between lunar images acquired at different times, scales, viewpoints and sensor modalities, followed by geometric registration and quantitative evaluation.

---

## Problem

Lunar image registration becomes difficult when images differ because of:

- illumination and Sun-angle variation
- viewpoint variation
- large spatial-resolution differences
- different imaging sensors/modalities
- partial or irregular observation footprints

The system is designed for imagery including:

- Chandrayaan-2 TMC-2
- Chandrayaan-2 OHRC
- Chandrayaan-2 IIRS
- external lunar reference imagery such as LRO/SELENE

---

## Proposed Pipeline

```text
Source + Reference Lunar Images
            ↓
Metadata / Product Ingestion
            ↓
True Observation Overlap Detection
            ↓
Sensor-Aware Preprocessing
            ↓
Multi-Scale Representation
            ↓
Feature Correspondence
    ┌────────┼────────┐
   SIFT   LightGlue   LoFTR
    └────────┼────────┘
             ↓
Geometric Verification
           RANSAC
             ↓
Sub-Pixel Refinement
             ↓
Spatial Coverage Optimization
             ↓
Image Registration
             ↓
Correspondence Points + Registered Product
             ↓
RMSE / Inliers / Coverage / Runtime         

Current Prototype Status
Dataset preparation
Chandrayaan-2 TMC-2 GeoTIFF ingestion
geographic overlap detection
real valid-pixel overlap verification
automatic high-validity patch selection
georeferenced canonical pair generation
Correspondence engines
SIFT
SuperPoint + LightGlue
LoFTR
Geometry and evaluation
RANSAC affine verification
correspondence export
reprojection-error evaluation
spatial coverage evaluation
image registration
overlay and difference visualization
Initial TMC-2 Benchmark

Real Chandrayaan-2 TMC-2 observations:

Source: 12 December 2019
Reference: 08 January 2020
Patch resolution: 2048 × 2048
Common valid pixels: ~99%
Matcher	Candidate Matches	RANSAC Inliers	Inlier Ratio	Reprojection RMSE	Spatial Coverage	Runtime
SIFT	1024	980	95.70%	0.611 px	100%	0.86 s
SuperPoint + LightGlue	1714	1329	77.54%	1.483 px	100%	4.48 s
LoFTR	9696	9388	96.82%	0.975 px	100%	1.11 s

These values represent geometric consistency on the current same-sensor TMC-2 benchmark. Controlled ground-truth and cross-sensor experiments are under development.

Planned Validation
Controlled rotation / translation / scale benchmarks
Sub-pixel correspondence refinement
Lunar illumination variation experiments
OHRC ↔ TMC-2 cross-scale correspondence
TMC-2 ↔ IIRS multimodal correspondence
Generic ChandraMatch application and visualization interface
Installation

Python 3.11 is recommended.

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Repository Structure
src/          reusable ChandraMatch modules
experiments/  reproducible benchmark experiments
results/      quantitative results and presentation figures
docs/         project documentation and architecture
examples/     lightweight example imagery
Outputs

Each experiment produces:

correspondence visualization
RANSAC-verified correspondence visualization
registered source image
source/reference overlay
difference image
correspondences.csv
metrics.json

## Data Provenance & License

This repository's **code** is licensed under the MIT License (see `LICENSE`).

The **imagery** it processes is not owned by this project and is not
redistributed in bulk here:

- Chandrayaan-2 (TMC-2, OHRC, IIRS) data is from ISRO's public
  [Indian Space Science Data Centre (ISSDC)](https://www.issdc.gov.in/) archive.
- LRO NAC data is public-domain NASA archive data from the
  [Planetary Data System (PDS)](https://pds.nasa.gov/), operated by NASA/USGS.
- SELENE/Kaguya Terrain Camera data is from JAXA's public lunar data archive.

Nothing in this repository claims ownership of, or exclusive rights to, any
of the above source imagery — only the registration pipeline code itself.