import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF


class LoFTRMatcher:

    def __init__(
        self,
        device=None,
        max_dimension=840,
        confidence_threshold=0.3
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

        print("LoFTR device:", self.device)

        if self.device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        self.matcher = KF.LoFTR(
            pretrained="outdoor"
        ).eval().to(self.device)


    def _prepare_image(self, image):
        """
        Resize image for LoFTR while retaining scale factors
        required to return coordinates to original resolution.
        """

        original_height, original_width = (
            image.shape[:2]
        )

        scale = min(
            1.0,
            self.max_dimension
            / max(
                original_height,
                original_width
            )
        )

        new_width = int(
            round(original_width * scale)
        )

        new_height = int(
            round(original_height * scale)
        )

        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        tensor = torch.from_numpy(
            resized
        ).float()

        tensor = tensor / 255.0

        tensor = (
            tensor
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        scale_x = (
            original_width / new_width
        )

        scale_y = (
            original_height / new_height
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

        ) = self._prepare_image(
            source
        )

        (
            reference_tensor,
            reference_scale_x,
            reference_scale_y

        ) = self._prepare_image(
            reference
        )


        input_dict = {
            "image0": source_tensor,
            "image1": reference_tensor
        }


        correspondences = self.matcher(
            input_dict
        )


        source_points = (
            correspondences["keypoints0"]
            .detach()
            .cpu()
            .numpy()
        )

        reference_points = (
            correspondences["keypoints1"]
            .detach()
            .cpu()
            .numpy()
        )

        confidence = (
            correspondences["confidence"]
            .detach()
            .cpu()
            .numpy()
        )


        # Remove low-confidence matches
        keep = (
            confidence
            >= self.confidence_threshold
        )

        source_points = source_points[keep]
        reference_points = (
            reference_points[keep]
        )
        confidence = confidence[keep]


        # Return coordinates to native
        # 2048x2048 image coordinates

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
            source_points.astype(np.float32),
            reference_points.astype(np.float32),
            confidence.astype(np.float32)
        )