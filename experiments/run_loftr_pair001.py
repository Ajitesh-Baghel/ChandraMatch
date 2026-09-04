import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import kornia.feature as KF


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


# ============================================================
# EXISTING CHANDRAMATCH MODULES
# ============================================================

from src.preprocessing.tmc2 import (
    read_tmc2,
    preprocess_tmc2
)

from src.geometry.ransac_filter import (
    filter_matches_ransac
)

from src.geometry.coverage_filter import (
    calculate_spatial_coverage
)

from src.evaluation.metrics import (
    calculate_metrics
)

from src.utils.visualize_matches import (
    draw_matches
)

from src.registration.affine_registration import (
    register_affine,
    create_overlay,
    create_difference
)


# ============================================================
# CONFIGURATION
# ============================================================

PAIR_DIR = (
    ROOT
    / "data"
    / "processed"
    / "pairs"
    / "pair_001"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "pair_001"
    / "loftr"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SOURCE_PATH = (
    PAIR_DIR / "source.tif"
)

REFERENCE_PATH = (
    PAIR_DIR / "reference.tif"
)


# LoFTR working resolution.
# 840 is a reasonable starting point for RTX 4070 Laptop 8 GB.
MAX_DIMENSION = 840

# Remove very weak LoFTR correspondences.
CONFIDENCE_THRESHOLD = 0.2


# ============================================================
# LoFTR MATCHER
# ============================================================

class LoFTRMatcher:

    def __init__(
        self,
        max_dimension=840,
        confidence_threshold=0.2,
        device=None
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.max_dimension = max_dimension

        self.confidence_threshold = (
            confidence_threshold
        )


        print(
            "LoFTR device:",
            self.device
        )

        if self.device.type == "cuda":

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

            memory_gb = (
                torch.cuda.get_device_properties(0)
                .total_memory
                / 1024**3
            )

            print(
                "GPU VRAM:",
                round(memory_gb, 2),
                "GB"
            )


        print(
            "Loading pretrained LoFTR outdoor model..."
        )

        self.matcher = KF.LoFTR(
            pretrained="outdoor"
        )

        self.matcher = (
            self.matcher
            .eval()
            .to(self.device)
        )


    def prepare_image(
        self,
        image
    ):
        """
        Resize grayscale uint8 image for LoFTR while keeping
        coordinate scale factors so matches can be mapped
        back to the original 2048x2048 coordinates.
        """

        original_height = image.shape[0]
        original_width = image.shape[1]


        scale = min(
            1.0,
            self.max_dimension
            / max(
                original_height,
                original_width
            )
        )


        new_width = int(
            round(
                original_width * scale
            )
        )

        new_height = int(
            round(
                original_height * scale
            )
        )


        resized = cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )


        # Convert:
        #
        # H x W uint8
        #
        # into:
        #
        # 1 x 1 x H x W float32
        #
        # expected by LoFTR.

        tensor = torch.from_numpy(
            resized
        )

        tensor = tensor.float() / 255.0

        tensor = (
            tensor
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )


        # Used later to transform LoFTR coordinates
        # back to 2048x2048 image coordinates.

        scale_x = (
            original_width
            / new_width
        )

        scale_y = (
            original_height
            / new_height
        )


        return (
            tensor,
            scale_x,
            scale_y
        )


    @torch.inference_mode()
    def match(
        self,
        source,
        reference
    ):

        (
            source_tensor,
            source_scale_x,
            source_scale_y

        ) = self.prepare_image(
            source
        )


        (
            reference_tensor,
            reference_scale_x,
            reference_scale_y

        ) = self.prepare_image(
            reference
        )


        print(
            "\nLoFTR source tensor:",
            tuple(
                source_tensor.shape
            )
        )

        print(
            "LoFTR reference tensor:",
            tuple(
                reference_tensor.shape
            )
        )


        input_dict = {

            "image0":
                source_tensor,

            "image1":
                reference_tensor
        }


        correspondences = (
            self.matcher(
                input_dict
            )
        )


        source_points = (
            correspondences[
                "keypoints0"
            ]
            .detach()
            .cpu()
            .numpy()
        )


        reference_points = (
            correspondences[
                "keypoints1"
            ]
            .detach()
            .cpu()
            .numpy()
        )


        confidence = (
            correspondences[
                "confidence"
            ]
            .detach()
            .cpu()
            .numpy()
        )


        print(
            "Raw LoFTR matches:",
            len(source_points)
        )


        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        keep = (
            confidence
            >= self.confidence_threshold
        )


        source_points = (
            source_points[keep]
        )

        reference_points = (
            reference_points[keep]
        )

        confidence = (
            confidence[keep]
        )


        print(
            "Matches after confidence filtering:",
            len(source_points)
        )


        # ----------------------------------------------------
        # Restore coordinates to original 2048x2048 resolution
        # ----------------------------------------------------

        source_points[:, 0] *= (
            source_scale_x
        )

        source_points[:, 1] *= (
            source_scale_y
        )


        reference_points[:, 0] *= (
            reference_scale_x
        )

        reference_points[:, 1] *= (
            reference_scale_y
        )


        return (

            source_points.astype(
                np.float32
            ),

            reference_points.astype(
                np.float32
            ),

            confidence.astype(
                np.float32
            )
        )


# ============================================================
# START EXPERIMENT
# ============================================================

print("=" * 60)

print(
    "CHANDRAMATCH - LoFTR BASELINE"
)

print("=" * 60)


# ============================================================
# LOAD TMC-2 PAIR
# ============================================================

source_raw, _ = read_tmc2(
    SOURCE_PATH
)

reference_raw, _ = read_tmc2(
    REFERENCE_PATH
)


print(
    "Source shape:",
    source_raw.shape
)

print(
    "Reference shape:",
    reference_raw.shape
)


# ============================================================
# COMMON VALID MASK
# ============================================================

valid_mask = (
    (source_raw > 0)
    &
    (reference_raw > 0)
)


print(
    "Common valid ratio:",
    valid_mask.mean()
)


# ============================================================
# SENSOR-SPECIFIC PREPROCESSING
# ============================================================

source = preprocess_tmc2(
    source_raw,
    use_clahe=False
)

reference = preprocess_tmc2(
    reference_raw,
    use_clahe=False
)


cv2.imwrite(
    str(
        RESULT_DIR
        / "source_preprocessed.png"
    ),
    source
)

cv2.imwrite(
    str(
        RESULT_DIR
        / "reference_preprocessed.png"
    ),
    reference
)


# ============================================================
# INITIALISE LoFTR
#
# Model loading happens before timing so weight loading does
# not distort the matcher runtime.
# ============================================================

loftr = LoFTRMatcher(

    max_dimension=
        MAX_DIMENSION,

    confidence_threshold=
        CONFIDENCE_THRESHOLD
)


# ============================================================
# RUN LoFTR
# ============================================================

if torch.cuda.is_available():

    torch.cuda.synchronize()


start_time = (
    time.perf_counter()
)


(
    source_points,
    reference_points,
    confidence

) = loftr.match(
    source,
    reference
)


if torch.cuda.is_available():

    torch.cuda.synchronize()


runtime = (
    time.perf_counter()
    - start_time
)


candidate_count = len(
    source_points
)


print(
    "\nCandidate matches:",
    candidate_count
)


if candidate_count < 3:

    raise RuntimeError(
        "LoFTR returned fewer than 3 usable matches."
    )


# ============================================================
# REMOVE MATCHES OUTSIDE COMMON VALID REGION
#
# LoFTR works on the image globally, so ensure returned points
# correspond to pixels that are genuinely valid in BOTH lunar
# observations.
# ============================================================

source_x = np.round(
    source_points[:, 0]
).astype(int)

source_y = np.round(
    source_points[:, 1]
).astype(int)


reference_x = np.round(
    reference_points[:, 0]
).astype(int)

reference_y = np.round(
    reference_points[:, 1]
).astype(int)


height, width = source.shape


inside = (

    (source_x >= 0)
    &
    (source_x < width)

    &

    (source_y >= 0)
    &
    (source_y < height)

    &

    (reference_x >= 0)
    &
    (reference_x < width)

    &

    (reference_y >= 0)
    &
    (reference_y < height)
)


source_points = (
    source_points[inside]
)

reference_points = (
    reference_points[inside]
)

confidence = (
    confidence[inside]
)


source_x = source_x[inside]
source_y = source_y[inside]

reference_x = reference_x[inside]
reference_y = reference_y[inside]


valid_correspondence = (

    valid_mask[
        source_y,
        source_x
    ]

    &

    valid_mask[
        reference_y,
        reference_x
    ]
)


source_points = (
    source_points[
        valid_correspondence
    ]
)

reference_points = (
    reference_points[
        valid_correspondence
    ]
)

confidence = (
    confidence[
        valid_correspondence
    ]
)


candidate_count = len(
    source_points
)


print(
    "Matches inside valid lunar area:",
    candidate_count
)


if candidate_count < 3:

    raise RuntimeError(
        "Too few valid LoFTR matches after mask filtering."
    )


# ============================================================
# RANSAC GEOMETRIC VERIFICATION
# ============================================================

(
    transformation,
    inlier_mask,
    inlier_source,
    inlier_reference

) = filter_matches_ransac(

    source_points,
    reference_points,

    reprojection_threshold=3.0
)


inlier_confidence = (
    confidence[
        inlier_mask
    ]
)


print(
    "RANSAC inliers:",
    len(
        inlier_source
    )
)


# ============================================================
# METRICS
# ============================================================

metrics = calculate_metrics(

    candidate_count,

    inlier_source,

    inlier_reference,

    transformation
)


# ============================================================
# SPATIAL COVERAGE
# ============================================================

coverage = (
    calculate_spatial_coverage(

        inlier_source,

        source.shape,

        grid_rows=8,

        grid_cols=8
    )
)


metrics[
    "spatial_coverage"
] = coverage[
    "coverage_ratio"
]


metrics[
    "occupied_grid_cells"
] = coverage[
    "occupied_cells"
]


metrics[
    "total_grid_cells"
] = coverage[
    "total_cells"
]


# ============================================================
# OTHER METADATA
# ============================================================

metrics[
    "runtime_seconds"
] = float(
    runtime
)


metrics[
    "matcher"
] = "LoFTR"


metrics[
    "pair"
] = "pair_001"


metrics[
    "loftr_max_dimension"
] = MAX_DIMENSION


metrics[
    "confidence_threshold"
] = CONFIDENCE_THRESHOLD


metrics[
    "transformation"
] = transformation.tolist()


# ============================================================
# SAVE CORRESPONDENCE POINTS
# ============================================================

correspondence_df = (
    pd.DataFrame({

        "source_x":
            inlier_source[:, 0],

        "source_y":
            inlier_source[:, 1],

        "reference_x":
            inlier_reference[:, 0],

        "reference_y":
            inlier_reference[:, 1],

        "confidence":
            inlier_confidence
    })
)


correspondence_df.to_csv(

    RESULT_DIR
    / "correspondences.csv",

    index=False
)


# ============================================================
# REGISTER SOURCE IMAGE
# ============================================================

registered = register_affine(

    source,

    transformation,

    reference.shape
)


# ============================================================
# CREATE OVERLAY
# ============================================================

overlay = create_overlay(

    registered,

    reference
)


# ============================================================
# CREATE DIFFERENCE IMAGE
# ============================================================

difference = create_difference(

    registered,

    reference
)


# ============================================================
# SAVE REGISTRATION PRODUCTS
# ============================================================

cv2.imwrite(

    str(
        RESULT_DIR
        / "registered_source.png"
    ),

    registered
)


cv2.imwrite(

    str(
        RESULT_DIR
        / "overlay.png"
    ),

    overlay
)


cv2.imwrite(

    str(
        RESULT_DIR
        / "difference.png"
    ),

    difference
)


# ============================================================
# VISUALISE RAW LoFTR MATCHES
# ============================================================

draw_matches(

    source,

    reference,

    source_points,

    reference_points,

    RESULT_DIR
    / "candidate_matches.png",

    max_matches=150
)


# ============================================================
# VISUALISE RANSAC INLIERS
# ============================================================

draw_matches(

    source,

    reference,

    inlier_source,

    inlier_reference,

    RESULT_DIR
    / "ransac_inliers.png",

    max_matches=150
)


# ============================================================
# SAVE METRICS
# ============================================================

with open(

    RESULT_DIR
    / "metrics.json",

    "w"

) as file:

    json.dump(

        metrics,

        file,

        indent=4
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\nTransformation:")

print(
    transformation
)


print("\nMetrics:")


for key, value in metrics.items():

    print(
        f"{key}: {value}"
    )


print(
    "\nResults saved to:"
)

print(
    RESULT_DIR
)


print("\nGenerated files:")

for file in sorted(
    RESULT_DIR.iterdir()
):

    print(
        "-",
        file.name
    )


print(
    "\nLoFTR experiment complete."
)