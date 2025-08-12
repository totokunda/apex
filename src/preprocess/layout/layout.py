# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
from src.preprocess.base import (
    BaseOutput,
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
)
from typing import Union, List, Optional, Tuple
from PIL import Image
import warnings
from src.utils.preprocessors import RAM_TAG_COLOR_PATH
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
from src.preprocess.inpainting import InpaintingVideoPreprocessor


class LayoutBboxOutput(BaseOutput):
    frames: List[np.ndarray]


class LayoutMaskOutput(BaseOutput):
    frames: List[np.ndarray]


class LayoutTrackOutput(BaseOutput):
    frames: List[np.ndarray]


@preprocessor_registry("layout.bbox")
class LayoutBboxPreprocessor(BasePreprocessor):
    def __init__(
        self,
        bg_color=None,
        box_color=None,
        frame_size=None,
        num_frames=81,
        ram_tag_color_path=None,
        **kwargs,
    ):
        if ram_tag_color_path is None:
            ram_tag_color_path = RAM_TAG_COLOR_PATH

        self.ram_tag_color_path = self._download(
            ram_tag_color_path, DEFAULT_PREPROCESSOR_SAVE_PATH
        )
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        self.bg_color = bg_color if bg_color is not None else [255, 255, 255]
        self.box_color = box_color if box_color is not None else [0, 0, 0]
        self.frame_size = frame_size if frame_size is not None else [720, 1280]
        self.num_frames = num_frames
        self.color_dict = {"default": tuple(self.box_color)}

        if self.ram_tag_color_path is not None:
            try:
                lines = [
                    id_name_color.strip().split("#;#")
                    for id_name_color in open(self.ram_tag_color_path).readlines()
                ]
                self.color_dict.update(
                    {
                        id_name_color[1]: tuple(eval(id_name_color[2]))
                        for id_name_color in lines
                    }
                )
            except Exception as e:
                warnings.warn(
                    f"Could not load color mappings from {ram_tag_color_path}: {e}"
                )

    def _convert_bbox(self, bbox: List[float], shape) -> List[float]:
        if bbox[0] < 1:
            bbox[0] = bbox[0] * shape[1]
        if bbox[1] < 1:
            bbox[1] = bbox[1] * shape[0]
        if bbox[2] > 1:
            bbox[2] = bbox[2] * shape[1]
        if bbox[3] > 1:
            bbox[3] = bbox[3] * shape[0]
        return bbox

    def __call__(
        self,
        bbox: List[List[float]],
        frame_size: Optional[List[int]] = None,
        num_frames: Optional[int] = None,
        label: Optional[Union[str, List[str]]] = None,
        color: Optional[Tuple[int, int, int]] = None,
    ):
        frame_size = frame_size if frame_size is not None else self.frame_size
        num_frames = num_frames if num_frames is not None else self.num_frames
        assert (
            len(bbox) == 2
        ), "bbox should be a list of two elements (start_bbox & end_bbox)"

        label = label[0] if label is not None and isinstance(label, list) else label
        if label is not None and label in self.color_dict:
            box_color = self.color_dict[label]
        elif color is not None:
            box_color = color
        else:
            box_color = self.color_dict["default"]

        start_bbox, end_bbox = [
            self._convert_bbox(bbox[0], frame_size),
            self._convert_bbox(bbox[1], frame_size),
        ]
        start_bbox = [
            start_bbox[0],
            start_bbox[1],
            start_bbox[2] - start_bbox[0],
            start_bbox[3] - start_bbox[1],
        ]
        start_bbox = np.array(start_bbox, dtype=np.float32)
        end_bbox = [
            end_bbox[0],
            end_bbox[1],
            end_bbox[2] - end_bbox[0],
            end_bbox[3] - end_bbox[1],
        ]

        end_bbox = np.array(end_bbox, dtype=np.float32)
        bbox_increment = (end_bbox - start_bbox) / num_frames

        ret_frames = []
        for frame_idx in range(num_frames):
            frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
            frame[:] = self.bg_color
            current_bbox = start_bbox + bbox_increment * frame_idx
            current_bbox = current_bbox.astype(int)
            x, y, w, h = current_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            ret_frames.append(frame[..., ::-1])  # Convert BGR to RGB
        return LayoutBboxOutput(frames=ret_frames)

    def __str__(self):
        return f"LayoutBboxPreprocessor(frame_size={self.frame_size}, num_frames={self.num_frames})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("layout.mask")
class LayoutMaskPreprocessor(BasePreprocessor):
    def __init__(
        self,
        use_aug=False,
        bg_color=None,
        box_color=None,
        ram_tag_color_path=None,
        **kwargs,
    ):
        if ram_tag_color_path is None:
            ram_tag_color_path = RAM_TAG_COLOR_PATH

        self.ram_tag_color_path = self._download(
            ram_tag_color_path, DEFAULT_PREPROCESSOR_SAVE_PATH
        )
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        self.use_aug = use_aug
        self.bg_color = bg_color if bg_color is not None else [255, 255, 255]
        self.box_color = box_color if box_color is not None else [0, 0, 0]
        self.color_dict = {"default": tuple(self.box_color)}

        if self.ram_tag_color_path is not None:
            try:
                lines = [
                    id_name_color.strip().split("#;#")
                    for id_name_color in open(self.ram_tag_color_path).readlines()
                ]
                self.color_dict.update(
                    {
                        id_name_color[1]: tuple(eval(id_name_color[2]))
                        for id_name_color in lines
                    }
                )
            except Exception as e:
                warnings.warn(
                    f"Could not load color mappings from {self.ram_tag_color_path}: {e}"
                )

        if self.use_aug:
            warnings.warn(
                "Mask augmentation is not available. Proceeding without augmentation."
            )
            self.use_aug = False

    def find_contours(self, mask):
        contours, hier = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def draw_contours(self, canvas, contour, color):
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        canvas = cv2.drawContours(canvas, contour, -1, color, thickness=3)
        return canvas

    def __call__(
        self,
        mask: Union[Image.Image, np.ndarray, str, List],
        color: Optional[Tuple[int, int, int]] = None,
        label: Optional[str] = None,
        mask_cfg: Optional[dict] = None,
    ):
        if not isinstance(mask, list):
            mask = [mask]

        if label is not None and label in self.color_dict:
            color = self.color_dict[label]
        elif color is not None:
            color = color
        else:
            color = self.color_dict["default"]

        ret_data = []
        for sub_mask in mask:
            # Convert mask to numpy array
            if isinstance(sub_mask, (str, Image.Image)):
                sub_mask = self._load_image(sub_mask)
                sub_mask = np.array(sub_mask)
            elif not isinstance(sub_mask, np.ndarray):
                sub_mask = np.array(sub_mask)

            # Convert to grayscale if needed
            if len(sub_mask.shape) == 3:
                sub_mask = cv2.cvtColor(sub_mask, cv2.COLOR_RGB2GRAY)

            canvas = np.ones((sub_mask.shape[0], sub_mask.shape[1], 3)) * 255
            contour = self.find_contours(sub_mask)
            frame = self.draw_contours(canvas, contour, color)
            ret_data.append(frame)

        return LayoutMaskOutput(frames=ret_data)

    def __str__(self):
        return f"LayoutMaskPreprocessor(use_aug={self.use_aug})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("layout.track")
class LayoutTrackPreprocessor(BasePreprocessor):
    def __init__(
        self,
        use_aug=False,
        bg_color=None,
        box_color=None,
        ram_tag_color_path=None,
        inpainting_config=None,
        **kwargs,
    ):
        if ram_tag_color_path is None:
            ram_tag_color_path = RAM_TAG_COLOR_PATH

        self.ram_tag_color_path = self._download(
            ram_tag_color_path, DEFAULT_PREPROCESSOR_SAVE_PATH
        )
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)
        self.use_aug = use_aug
        self.bg_color = bg_color if bg_color is not None else [255, 255, 255]
        self.box_color = box_color if box_color is not None else [0, 0, 0]
        self.color_dict = {"default": tuple(self.box_color)}

        if self.ram_tag_color_path is not None:
            try:
                lines = [
                    id_name_color.strip().split("#;#")
                    for id_name_color in open(self.ram_tag_color_path).readlines()
                ]
                self.color_dict.update(
                    {
                        id_name_color[1]: tuple(eval(id_name_color[2]))
                        for id_name_color in lines
                    }
                )
            except Exception as e:
                warnings.warn(
                    f"Could not load color mappings from {self.ram_tag_color_path}: {e}"
                )

        if self.use_aug:
            warnings.warn(
                "Mask augmentation is not available. Proceeding without augmentation."
            )
            self.use_aug = False

        self.inpainting_ins = InpaintingVideoPreprocessor(
            **(inpainting_config if inpainting_config is not None else {})
        )

    def find_contours(self, mask):
        contours, hier = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def draw_contours(self, canvas, contour, color):
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        canvas = cv2.drawContours(canvas, contour, -1, color, thickness=3)
        return canvas

    def __call__(
        self,
        frames: Union[List[Image.Image], List[str], str] = None,
        mask: Optional[Union[Image.Image, np.ndarray, List]] = None,
        bbox: Optional[List] = None,
        label: Optional[str] = None,
        caption: Optional[str] = None,
        mode: str = "salient",
        color: Optional[Tuple[int, int, int]] = None,
        mask_cfg: Optional[dict] = None,
    ):

        if frames is not None:
            frames = self._load_video(frames)

        inp_data = self.inpainting_ins(
            frames=frames,
            mode=mode,
            mask=mask,
            bbox=bbox,
            label=label,
            caption=caption,
        )

        inp_masks = inp_data.masks

        label = label[0] if label is not None and isinstance(label, list) else label
        if label is not None and label in self.color_dict:
            color = self.color_dict[label]
        elif color is not None:
            color = color
        else:
            color = self.color_dict["default"]

        num_frames = len(inp_masks)
        ret_data = []
        for i in range(num_frames):
            sub_mask = inp_masks[i]
            canvas = np.ones((sub_mask.shape[0], sub_mask.shape[1], 3)) * 255
            contour = self.find_contours(sub_mask)
            frame = self.draw_contours(canvas, contour, color)
            ret_data.append(frame)

        return LayoutTrackOutput(frames=ret_data)

    def __str__(self):
        return f"LayoutTrackPreprocessor(use_aug={self.use_aug})"

    def __repr__(self):
        return self.__str__()
