from pathlib import Path

import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


class SuperPointLightGlueMatcher:

    def __init__(
        self,
        max_keypoints=4096,
        device=None
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print(
            "LightGlue device:",
            self.device
        )

        if self.device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        # SuperPoint feature extractor
        self.extractor = SuperPoint(
            max_num_keypoints=max_keypoints
        ).eval().to(self.device)

        # LightGlue matcher
        self.matcher = LightGlue(
            features="superpoint"
        ).eval().to(self.device)


    @torch.inference_mode()
    def match(
        self,
        source_path,
        reference_path
    ):
        """
        Match two preprocessed images.

        Returns:
            source_points      Nx2 float32
            reference_points   Nx2 float32
            confidence         N float32
        """

        source_path = Path(source_path)
        reference_path = Path(reference_path)

        # Load images using LightGlue's own loader
        image0 = load_image(
            source_path
        ).to(self.device)

        image1 = load_image(
            reference_path
        ).to(self.device)

        # Extract SuperPoint features
        features0 = self.extractor.extract(
            image0
        )

        features1 = self.extractor.extract(
            image1
        )

        # Match features
        matches01 = self.matcher({
            "image0": features0,
            "image1": features1
        })

        # Remove batch dimension
        features0 = rbd(features0)
        features1 = rbd(features1)
        matches01 = rbd(matches01)

        matches = matches01["matches"]

        if len(matches) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32)
            )

        keypoints0 = features0["keypoints"]
        keypoints1 = features1["keypoints"]

        matched0 = keypoints0[
            matches[:, 0]
        ]

        matched1 = keypoints1[
            matches[:, 1]
        ]

        # LightGlue normally provides confidence scores
        if "scores" in matches01:
            confidence = matches01[
                "scores"
            ]
        else:
            confidence = torch.ones(
                len(matches),
                device=self.device
            )

        source_points = (
            matched0
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        reference_points = (
            matched1
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        confidence = (
            confidence
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return (
            source_points,
            reference_points,
            confidence
        )