# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
from typing import Union, List, Optional, Literal
from PIL import Image
from src.preprocess.inpainting import InpaintingPreprocessor
from src.preprocess.canvas.mask_aug import MaskAugAnnotator
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
    BaseOutput,
)

subject_mode = Literal[
    "salient",
    "mask",
    "bbox",
    "label",
    "caption",
    "plain",
    "salientmasktrack",
    "salientbboxtrack",
    "masktrack",
    "bboxtrack",
]


class SubjectOutput(BaseOutput):
    image: np.ndarray
    mask: np.ndarray
    src_image: np.ndarray


@preprocessor_registry("subject")
class SubjectPreprocessor(BasePreprocessor):
    def __init__(
        self,
        mode: subject_mode = "plain",
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
            self.inpainting_preprocessor = InpaintingPreprocessor(inpainting_config)
        else:
            self.inpainting_preprocessor = InpaintingPreprocessor()

        if self.use_aug:
            self.mask_aug_annotator = MaskAugAnnotator()
            self.use_aug = True
        else:
            self.mask_aug_annotator = None

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str] = None,
        mode: Optional[subject_mode] = "salient",
        return_mask: Optional[bool] = True,
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
            return SubjectOutput(image=image_array, mask=None)

        image = self._load_image(image)
        image_array = np.array(image)

        inp_res = self.inpainting_preprocessor(
            image,
            mask=mask,
            bbox=bbox,
            label=label,
            caption=caption,
            mode=mode,
            return_mask=True,
            return_source=True,
        )
        image_array = inp_res.src_image
        mask = inp_res.mask

        if self.use_aug:
            mask = self.mask_aug_annotator(mask, mask_cfg)

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
            return SubjectOutput(image=ret_image, mask=ret_mask)
        else:
            return SubjectOutput(image=ret_image)

    def __str__(self):
        return f"SubjectPreprocessor(mode={self.mode}, use_crop={self.use_crop}, roi_only={self.roi_only})"

    def __repr__(self):
        return self.__str__()
