# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import random
from typing import Union, List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
    BaseOutput,
)


class OutpaintingOutput(BaseOutput):
    image: Image.Image
    mask: Image.Image | None = None
    src_image: Image.Image | None = None
    
class OutpaintingVideoOutput(BaseOutput):
    frames: List[Image.Image]
    masks: List[Image.Image] | None = None
    src_frames: List[Image.Image] | None = None
    

def get_mask_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Get bounding box of non-zero regions in mask."""
    if mask.max() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (cmin, rmin, cmax + 1, rmax + 1)


@preprocessor_registry("outpainting")
class OutpaintingPreprocessor(BasePreprocessor):
    def __init__(
        self,
        mask_blur=0,
        random_cfg=None,
        return_mask=True,
        return_source=True,
        keep_padding_ratio=8,
        mask_color=0,
        **kwargs,
    ):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        self.mask_blur = mask_blur
        self.random_cfg = random_cfg
        self.return_mask = return_mask
        self.return_source = return_source
        self.keep_padding_ratio = keep_padding_ratio
        self.mask_color = mask_color

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        expand_ratio: float = 0.3,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        direction: List[str] = None,
        return_mask: Optional[bool] = None,
        return_source: Optional[bool] = None,
        mask_color: Optional[int] = None,
    ):

        if direction is None:
            direction = ["left", "right", "up", "down"]

        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = (
            return_source if return_source is not None else self.return_source
        )
        mask_color = mask_color if mask_color is not None else self.mask_color

        image = self._load_image(image)

        if self.random_cfg:
            direction_range = self.random_cfg.get(
                "DIRECTION_RANGE", ["left", "right", "up", "down"]
            )
            ratio_range = self.random_cfg.get("RATIO_RANGE", [0.0, 1.0])
            direction = random.sample(
                direction_range, random.choice(list(range(1, len(direction_range) + 1)))
            )
            expand_ratio = random.uniform(ratio_range[0], ratio_range[1])

        if mask is None:
            init_image = image
            src_width, src_height = init_image.width, init_image.height
            left = int(expand_ratio * src_width) if "left" in direction else 0
            right = int(expand_ratio * src_width) if "right" in direction else 0
            up = int(expand_ratio * src_height) if "up" in direction else 0
            down = int(expand_ratio * src_height) if "down" in direction else 0
            tar_width = (
                math.ceil((src_width + left + right) / self.keep_padding_ratio)
                * self.keep_padding_ratio
            )
            tar_height = (
                math.ceil((src_height + up + down) / self.keep_padding_ratio)
                * self.keep_padding_ratio
            )
            if left > 0:
                left = left * (tar_width - src_width) // (left + right)
            if right > 0:
                right = tar_width - src_width - left
            if up > 0:
                up = up * (tar_height - src_height) // (up + down)
            if down > 0:
                down = tar_height - src_height - up
            if mask_color is not None:
                img = Image.new("RGB", (tar_width, tar_height), color=mask_color)
            else:
                img = Image.new("RGB", (tar_width, tar_height))
            img.paste(init_image, (left, up))
            mask_img = Image.new("L", (img.width, img.height), "white")
            draw = ImageDraw.Draw(mask_img)

            draw.rectangle(
                (
                    left + (self.mask_blur * 2 if left > 0 else 0),
                    up + (self.mask_blur * 2 if up > 0 else 0),
                    mask_img.width
                    - right
                    - (self.mask_blur * 2 if right > 0 else 0)
                    - 1,
                    mask_img.height
                    - down
                    - (self.mask_blur * 2 if down > 0 else 0)
                    - 1,
                ),
                fill="black",
            )
        else:
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2).astype(np.uint8)

            bbox = get_mask_box(mask)

            if bbox is None:
                img = image
                mask_img = Image.fromarray(mask, mode="L")
                init_image = image
            else:
                mask_img = Image.new("L", (image.width, image.height), "white")
                mask_zero = Image.new(
                    "L", (bbox[2] - bbox[0], bbox[3] - bbox[1]), "black"
                )
                mask_img.paste(mask_zero, (bbox[0], bbox[1]))
                crop_image = image.crop(bbox)
                init_image = Image.new("RGB", (image.width, image.height), "black")
                init_image.paste(crop_image, (bbox[0], bbox[1]))
                img = image

        if return_mask:
            if return_source:
                return OutpaintingOutput(src_image=init_image, image=img, mask=mask_img)
            else:
                return OutpaintingOutput(image=img, mask=mask_img)
        else:
            if return_source:
                return OutpaintingOutput(src_image=init_image, image=img)
            else:
                return OutpaintingOutput(image=img)

    def __str__(self):
        return f"OutpaintingPreprocessor(return_mask={self.return_mask}, return_source={self.return_source})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("outpainting.inner")
class OutpaintingInnerPreprocessor(BasePreprocessor):
    def __init__(
        self,
        mask_blur=0,
        random_cfg=None,
        return_mask=True,
        return_source=True,
        keep_padding_ratio=8,
        mask_color=0,
        **kwargs,
    ):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        self.mask_blur = mask_blur
        self.random_cfg = random_cfg
        self.return_mask = return_mask
        self.return_source = return_source
        self.keep_padding_ratio = keep_padding_ratio
        self.mask_color = mask_color

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        expand_ratio: float = 0.3,
        direction: List[str] = None,
        return_mask: Optional[bool] = None,
        return_source: Optional[bool] = None,
        mask_color: Optional[int] = None,
    ):

        if direction is None:
            direction = ["left", "right", "up", "down"]

        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = (
            return_source if return_source is not None else self.return_source
        )
        mask_color = mask_color if mask_color is not None else self.mask_color

        image = self._load_image(image)

        if self.random_cfg:
            direction_range = self.random_cfg.get(
                "DIRECTION_RANGE", ["left", "right", "up", "down"]
            )
            ratio_range = self.random_cfg.get("RATIO_RANGE", [0.0, 1.0])
            direction = random.sample(
                direction_range, random.choice(list(range(1, len(direction_range) + 1)))
            )
            expand_ratio = random.uniform(ratio_range[0], ratio_range[1])

        init_image = image
        src_width, src_height = init_image.width, init_image.height
        left = int(expand_ratio * src_width) if "left" in direction else 0
        right = int(expand_ratio * src_width) if "right" in direction else 0
        up = int(expand_ratio * src_height) if "up" in direction else 0
        down = int(expand_ratio * src_height) if "down" in direction else 0

        crop_left = left
        crop_right = src_width - right
        crop_up = up
        crop_down = src_height - down
        crop_box = (crop_left, crop_up, crop_right, crop_down)
        cropped_image = init_image.crop(crop_box)

        if mask_color is not None:
            img = Image.new("RGB", (src_width, src_height), color=mask_color)
        else:
            img = Image.new("RGB", (src_width, src_height))

        paste_x = left
        paste_y = up
        img.paste(cropped_image, (paste_x, paste_y))

        mask_img = Image.new("L", (img.width, img.height), "white")
        draw = ImageDraw.Draw(mask_img)

        x0 = paste_x + (self.mask_blur * 2 if left > 0 else 0)
        y0 = paste_y + (self.mask_blur * 2 if up > 0 else 0)
        x1 = paste_x + cropped_image.width - (self.mask_blur * 2 if right > 0 else 0)
        y1 = paste_y + cropped_image.height - (self.mask_blur * 2 if down > 0 else 0)
        draw.rectangle((x0, y0, x1, y1), fill="black")

        if return_mask:
            if return_source:
                ret_data = OutpaintingOutput(src_image=init_image, image=img, mask=mask_img)
            else:
                ret_data = OutpaintingOutput(image=img, mask=mask_img)
        else:
            if return_source:
                ret_data = OutpaintingOutput(src_image=init_image, image=img)
            else:
                ret_data = OutpaintingOutput(image=img)
        return ret_data

    def __str__(self):
        return f"OutpaintingInnerPreprocessor(return_mask={self.return_mask}, return_source={self.return_source})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("outpainting.video")
class OutpaintingVideoPreprocessor(OutpaintingPreprocessor, BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor_type = PreprocessorType.VIDEO
        self.key_map = {"src_image": "src_images", "image": "frames", "mask": "masks"}

    def __call__(
        self,
        frames: Union[List[Image.Image], List[str], str],
        expand_ratio: float = 0.3,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        direction: List[str] = None,
        return_mask: Optional[bool] = True,
        return_source: Optional[bool] = True,
        mask_color: Optional[int] = None,
    ):

        frames = self._load_video(frames)
        ret_frames = {"src_images": [], "frames": [], "masks": []}

        for frame in frames:
            anno_frame = super().__call__(
                frame,
                expand_ratio=expand_ratio,
                mask=mask,
                direction=direction,
                return_mask=return_mask,
                return_source=return_source,
                mask_color=mask_color,
            )
            if anno_frame.image is not None:
                ret_frames["frames"].append(anno_frame.image)
            if anno_frame.mask is not None:
                ret_frames["masks"].append(anno_frame.mask)
            if anno_frame.src_image is not None:
                ret_frames["src_images"].append(anno_frame.src_image)
        
        return OutpaintingVideoOutput(frames=ret_frames["frames"], masks=ret_frames["masks"] if return_mask else None, src_frames=ret_frames["src_images"] if return_source else None)

    def __str__(self):
        return f"OutpaintingVideoPreprocessor(return_mask={self.return_mask}, return_source={self.return_source})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("outpainting.inner.video")
class OutpaintingInnerVideoPreprocessor(OutpaintingInnerPreprocessor, BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor_type = PreprocessorType.VIDEO
        self.key_map = {"src_image": "src_images", "image": "frames", "mask": "masks"}

    def __call__(
        self,
        frames: Union[List[Image.Image], List[str], str],
        expand_ratio: float = 0.3,
        direction: List[str] = None,
        return_mask: Optional[bool] = True,
        return_source: Optional[bool] = True,
        mask_color: Optional[int] = None,
    ):

        frames = self._load_video(frames)
        ret_frames = {"src_images": [], "frames": [], "masks": []}

        for frame in frames:
            anno_frame = super().__call__(
                frame,
                expand_ratio=expand_ratio,
                direction=direction,
                return_mask=return_mask,
                return_source=return_source,
                mask_color=mask_color,
            )
            if anno_frame.src_image is not None:
                ret_frames["src_images"].append(anno_frame.src_image)
            if anno_frame.image is not None:
                ret_frames["frames"].append(anno_frame.image)
            if anno_frame.mask is not None:
                ret_frames["masks"].append(anno_frame.mask)
        return OutpaintingVideoOutput(frames=ret_frames["frames"], masks=ret_frames["masks"] if return_mask else None)

    def __str__(self):
        return f"OutpaintingInnerVideoPreprocessor(return_mask={self.return_mask}, return_source={self.return_source})"

    def __repr__(self):
        return self.__str__()
