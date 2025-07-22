# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
from typing import Union, List, Optional
from PIL import Image
import warnings

from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
)


@preprocessor_registry("subject")
class SubjectPreprocessor(BasePreprocessor):
    def __init__(
        self,
        mode: str = "salientmasktrack",
        use_aug: bool = False,
        use_crop: bool = False,
        roi_only: bool = False,
        return_mask: bool = True,
        inpainting_config: dict = None,
        **kwargs,
    ):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)

        self.mode = mode
        self.use_aug = use_aug
        self.use_crop = use_crop
        self.roi_only = roi_only
        self.return_mask = return_mask

        # Note: Inpainting and mask augmentation dependencies are not available
        # This is a simplified implementation
        if inpainting_config is not None:
            warnings.warn(
                "Inpainting functionality requires additional dependencies and is not fully implemented."
            )

        if self.use_aug:
            warnings.warn(
                "Mask augmentation functionality requires additional dependencies and is disabled."
            )
            self.use_aug = False

        assert self.mode in [
            "plain",
            "salient",
            "mask",
            "bbox",
            "salientmasktrack",
            "salientbboxtrack",
            "masktrack",
            "bboxtrack",
            "label",
            "caption",
            "all",
        ]

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str] = None,
        mode: Optional[str] = None,
        return_mask: Optional[bool] = None,
        mask_cfg: Optional[dict] = None,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        bbox: Optional[List[float]] = None,
        label: Optional[str] = None,
        caption: Optional[str] = None,
    ):

        return_mask = return_mask if return_mask is not None else self.return_mask
        mode = mode if mode is not None else self.mode

        if mode == "plain":
            image = self._load_image(image)
            image_array = np.array(image)
            return {"image": image_array, "mask": None} if return_mask else image_array

        # For other modes, we need inpainting functionality which is not available
        # This is a simplified implementation that works with provided masks
        image = self._load_image(image)
        image_array = np.array(image)

        if mask is not None:
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2).astype(np.uint8)
        else:
            # Create a dummy mask if none provided
            mask = (
                np.ones((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
                * 255
            )

        # Process the mask
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        if (
            binary_mask is None
            or binary_mask.size == 0
            or cv2.countNonZero(binary_mask) == 0
        ):
            x, y, w, h = 0, 0, binary_mask.shape[1], binary_mask.shape[0]
        else:
            x, y, w, h = cv2.boundingRect(binary_mask)

        ret_mask = mask.copy()
        ret_image = image_array.copy()

        if self.roi_only:
            ret_image[mask == 0] = 255

        if self.use_crop:
            ret_image = ret_image[y : y + h, x : x + w]
            ret_mask = ret_mask[y : y + h, x : x + w]

        if return_mask:
            return {"image": ret_image, "mask": ret_mask}
        else:
            return ret_image

    def __str__(self):
        return f"SubjectPreprocessor(mode={self.mode}, use_crop={self.use_crop}, roi_only={self.roi_only})"

    def __repr__(self):
        return self.__str__()
