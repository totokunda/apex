# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
from typing import Union, List, Optional, Dict, Tuple
from PIL import Image
import warnings
from collections import defaultdict
from src.preprocess.sam2 import SAM2VideoOutput
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
    BaseOutput,
)
from src.preprocess.canvas.mask_aug import MaskAugAnnotator
import pycocotools.mask as mask_utils


class InpaintingOutput(BaseOutput):
    image: np.ndarray
    mask: np.ndarray
    src_image: np.ndarray


class InpaintingVideoOutput(BaseOutput):
    frames: List[np.ndarray]
    masks: List[np.ndarray]
    src_frames: List[np.ndarray]


def single_mask_to_rle(mask):
    rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def single_rle_to_mask(rle):
    mask = np.array(mask_utils.decode(rle)).astype(np.uint8)
    return mask


def get_mask_box(mask, threshold=255):
    """Get bounding box from mask"""
    locs = np.where(mask >= threshold)
    if len(locs) < 1 or locs[0].shape[0] < 1 or locs[1].shape[0] < 1:
        return None
    left, right = np.min(locs[1]), np.max(locs[1])
    top, bottom = np.min(locs[0]), np.max(locs[0])
    return [left, top, right, bottom]


def read_video_one_frame(video_path, use_type="cv2", is_rgb=True):
    """Read the first frame from a video"""
    image_first = None
    if use_type == "cv2":
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                if is_rgb:
                    image_first = frame[..., ::-1]
                else:
                    image_first = frame
            cap.release()
        except Exception as e:
            print(f"OpenCV read error: {e}")
            return None
    else:
        raise ValueError(f"Unknown video type {use_type}")
    return image_first


@preprocessor_registry("inpainting")
class InpaintingPreprocessor(BasePreprocessor):
    def __init__(
        self,
        mode: str = "mask",
        use_aug: bool = True,
        return_mask: bool = True,
        return_source: bool = True,
        mask_color: int = 128,
        salient_config: Optional[Dict] = None,
        sam2_config: Optional[Dict] = None,
        gdino_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)

        self.mode = mode
        self.use_aug = use_aug
        self.return_mask = return_mask
        self.return_source = return_source
        self.mask_color = mask_color

        assert self.mode in [
            "salient",
            "mask",
            "bbox",
            "salientmasktrack",
            "salientbboxtrack",
            "maskpointtrack",
            "maskbboxtrack",
            "masktrack",
            "bboxtrack",
            "label",
            "caption",
        ]

        # Initialize dependent preprocessors based on mode
        self.salient_model = None
        self.sam2_model = None
        self.gdino_model = None
        self.maskaug_anno = None
        self.salient_config = salient_config
        self.sam2_config = sam2_config
        self.gdino_config = gdino_config

        if self.mode in [
            "salient",
            "salienttrack",
            "salientmasktrack",
            "salientbboxtrack",
        ]:
            self.load_model("salient", salient_config)

        if self.mode in [
            "masktrack",
            "bboxtrack",
            "salienttrack",
            "salientmasktrack",
            "salientbboxtrack",
            "maskpointtrack",
            "maskbboxtrack",
            "label",
            "caption",
        ]:
            self.load_model("sam2", sam2_config)

        if self.mode in ["label", "caption"]:
            self.load_model("gdino", gdino_config)

        if self.use_aug:
            self.mask_aug_annotator = MaskAugAnnotator()
            self.use_aug = True
        else:
            self.mask_aug_annotator = None
            self.use_aug = False

    def load_model(self, model_name: str, config: Dict | None = None):
        if config is None:
            config = {}
        if model_name == "salient" and self.salient_model is None:
            from src.preprocess.salient import SalientPreprocessor

            self.salient_model = SalientPreprocessor(**config)
        elif model_name == "sam2" and self.sam2_model is None:
            from src.preprocess.sam2 import SAM2Preprocessor

            self.sam2_model = SAM2Preprocessor(**config)
        elif model_name == "gdino" and self.gdino_model is None:
            from src.preprocess.gdino import GDINOPreprocessor

            self.gdino_model = GDINOPreprocessor(**config)

    def apply_plain_mask(self, image: np.ndarray, mask: np.ndarray, mask_color: int):
        """Apply a plain mask to an image"""
        bool_mask = mask > 0
        out_image = image.copy()
        out_image[bool_mask] = mask_color
        out_mask = np.where(bool_mask, 255, 0).astype(np.uint8)
        return out_image, out_mask

    def apply_seg_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        mask_color: int,
        mask_cfg: Optional[Dict] = None,
    ):
        """Apply a segmentation mask to an image"""
        out_mask = (mask * 255).astype("uint8")
        if self.use_aug and mask_cfg is not None:
            out_mask = self.mask_aug_annotator(out_mask, mask_cfg)
        bool_mask = out_mask > 0
        out_image = image.copy()
        out_image[bool_mask] = mask_color
        return out_image, out_mask

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str] = None,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        bbox: Optional[List[float]] = None,
        label: Optional[str] = None,
        caption: Optional[str] = None,
        mode: Optional[str] = None,
        return_mask: Optional[bool] = None,
        return_source: Optional[bool] = None,
        mask_color: Optional[int] = None,
        mask_cfg: Optional[Dict] = None,
    ):

        mode = mode if mode is not None else self.mode
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = (
            return_source if return_source is not None else self.return_source
        )
        mask_color = mask_color if mask_color is not None else self.mask_color

        # Load and convert image
        image = self._load_image(image)
        image_array = np.array(image)
        out_image, out_mask = None, None

        if mode == "salient":
            if self.salient_model is None:
                self.load_model("salient", self.salient_config)
            salient_output = self.salient_model(image)

            out_image, out_mask = self.apply_plain_mask(
                image_array, salient_output.mask, mask_color
            )

        elif mode == "mask":
            if mask is None:
                raise ValueError("Mask is required for 'mask' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2).astype(np.uint8)

            mask_h, mask_w = mask.shape[:2]
            h, w = image_array.shape[:2]
            if (mask_h != h) or (mask_w != w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            out_image, out_mask = self.apply_plain_mask(image_array, mask, mask_color)

        elif mode == "bbox":
            if bbox is None:
                raise ValueError("Bbox is required for 'bbox' mode")
            x1, y1, x2, y2 = self.preprocess_bbox(bbox, image_array.shape)

            h, w = image_array.shape[:2]
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w, x2)), int(min(h, y2))
            out_image = image_array.copy()
            out_image[y1:y2, x1:x2] = mask_color
            out_mask = np.zeros((h, w), dtype=np.uint8)
            out_mask[y1:y2, x1:x2] = 255

        elif mode == "salientmasktrack":
            if self.salient_model is None or self.sam2_model is None:
                self.load_model("salient", self.salient_config)
                self.load_model("sam2", self.sam2_config)
            mask_result = self.salient_model(image)
            resize_mask = cv2.resize(
                mask_result.mask, (256, 256), interpolation=cv2.INTER_NEAREST
            )
            out_mask = self.sam2_model(
                image=image, mask=resize_mask, task_type="mask", return_mask=True
            )
            out_image, out_mask = self.apply_seg_mask(
                image_array, out_mask.masks, mask_color, mask_cfg
            )

        elif mode == "salientbboxtrack":
            if self.salient_model is None or self.sam2_model is None:
                self.load_model("salient", self.salient_config)
                self.load_model("sam2", self.sam2_config)
            mask_result = self.salient_model(image)
            bbox = get_mask_box(mask_result.mask, threshold=1)
            if bbox is None:
                raise ValueError("Could not extract bounding box from salient mask")
            out_mask = self.sam2_model(
                image=image, input_box=bbox, task_type="input_box", return_mask=True
            )
            out_image, out_mask = self.apply_seg_mask(
                image_array, out_mask.masks, mask_color, mask_cfg
            )

        elif mode == "maskpointtrack":
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            if mask is None:
                raise ValueError("Mask is required for 'maskpointtrack' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            out_mask = self.sam2_model(
                image=image, mask=mask, task_type="mask_point", return_mask=True
            )
            out_image, out_mask = self.apply_seg_mask(
                image_array, out_mask.masks, mask_color, mask_cfg
            )

        elif mode == "maskbboxtrack":
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            if mask is None:
                raise ValueError("Mask is required for 'maskbboxtrack' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            out_mask = self.sam2_model(
                image=image, mask=mask, task_type="mask_box", return_mask=True
            )
            out_image, out_mask = self.apply_seg_mask(
                image_array, out_mask.masks, mask_color, mask_cfg
            )

        elif mode == "masktrack":
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            if mask is None:
                raise ValueError("Mask is required for 'masktrack' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            resize_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            out_mask = self.sam2_model(
                image=image, mask=resize_mask, task_type="mask", return_mask=True
            )
            out_image, out_mask = self.apply_seg_mask(
                image_array, out_mask.masks, mask_color, mask_cfg
            )

        elif mode == "bboxtrack":
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            if bbox is None:
                raise ValueError("Bbox is required for 'bboxtrack' mode")
            x1, y1, x2, y2 = self.preprocess_bbox(bbox, image_array.shape)
            bbox = [x1, y1, x2, y2]
            out_mask = self.sam2_model(
                image=image, input_box=bbox, task_type="input_box", return_mask=True
            )
            out_image, out_mask = self.apply_seg_mask(
                image_array, out_mask.masks, mask_color, mask_cfg
            )

        elif mode == "label":
            if self.gdino_model is None or self.sam2_model is None:
                self.load_model("gdino", self.gdino_config)
                self.load_model("sam2", self.sam2_config)
            if label is None:
                raise ValueError("Label is required for 'label' mode")
            gdino_res = self.gdino_model(image, classes=label)
            if gdino_res.boxes is not None and len(gdino_res.boxes) > 0:
                bboxes = gdino_res.boxes[0]
            else:
                raise ValueError(
                    f"Unable to find the corresponding boxes of label: {label}"
                )
            out_mask = self.sam2_model(
                image=image, input_box=bboxes, task_type="input_box", return_mask=True
            )
            out_image, out_mask = self.apply_seg_mask(
                image_array, out_mask.masks, mask_color, mask_cfg
            )

        elif mode == "caption":
            if self.gdino_model is None or self.sam2_model is None:
                self.load_model("gdino", self.gdino_config)
                self.load_model("sam2", self.sam2_config)
            if caption is None:
                raise ValueError("Caption is required for 'caption' mode")
            gdino_res = self.gdino_model(image, caption=caption)
            if gdino_res.boxes is not None and len(gdino_res.boxes) > 0:
                bboxes = gdino_res.boxes[0]
            else:
                raise ValueError(
                    f"Unable to find the corresponding boxes of caption: {caption}"
                )
            out_mask = self.sam2_model(
                image=image, input_box=bboxes, task_type="input_box", return_mask=True
            )
            out_image, out_mask = self.apply_seg_mask(
                image_array, out_mask.masks, mask_color, mask_cfg
            )

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        output = InpaintingOutput(image=out_image)

        if return_mask:
            output.mask = out_mask
        if return_source:
            output.src_image = image_array

        return output

    def __str__(self):
        return (
            f"InpaintingPreprocessor(mode={self.mode}, return_mask={self.return_mask})"
        )

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("inpainting.video")
class InpaintingVideoPreprocessor(BasePreprocessor):
    def __init__(
        self,
        mode: str = "mask",
        use_aug: bool = True,
        return_frame: bool = True,
        return_mask: bool = True,
        return_source: bool = True,
        mask_color: int = 128,
        salient_config: Optional[Dict] = None,
        sam2_config: Optional[Dict] = None,
        gdino_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)

        self.mode = mode
        self.use_aug = use_aug
        self.return_frame = return_frame
        self.return_mask = return_mask
        self.return_source = return_source
        self.mask_color = mask_color
        self.salient_config = salient_config
        self.sam2_config = sam2_config
        self.gdino_config = gdino_config

        assert self.mode in [
            "salient",
            "mask",
            "bbox",
            "salientmasktrack",
            "salientbboxtrack",
            "maskpointtrack",
            "maskbboxtrack",
            "masktrack",
            "bboxtrack",
            "label",
            "caption",
            "all",
        ]

        # Initialize dependent preprocessors based on mode
        self.salient_model = None
        self.sam2_model = None
        self.gdino_model = None
        self.maskaug_anno = None

        if self.mode in [
            "salient",
            "salienttrack",
            "salientmasktrack",
            "salientbboxtrack",
            "all",
        ]:
            self.load_model("salient", salient_config)

        if self.mode in [
            "masktrack",
            "bboxtrack",
            "salienttrack",
            "salientmasktrack",
            "salientbboxtrack",
            "maskpointtrack",
            "maskbboxtrack",
            "label",
            "caption",
            "all",
        ]:
            self.load_model("sam2", sam2_config)

        if self.mode in ["label", "caption", "all"]:
            self.load_model("gdino", gdino_config)

        if self.use_aug:
            self.mask_aug_annotator = MaskAugAnnotator()
            self.use_aug = True
        else:
            self.mask_aug_annotator = None
            self.use_aug = False

    def load_model(self, model_name: str, config: Dict | None = None):
        if config is None:
            config = {}
        if model_name == "salient" and self.salient_model is None:
            from src.preprocess.salient import SalientPreprocessor

            self.salient_model = SalientPreprocessor(**config)
        elif model_name == "sam2" and self.sam2_model is None:
            from src.preprocess.sam2 import SAM2VideoPreprocessor

            self.sam2_model = SAM2VideoPreprocessor(**config)
        elif model_name == "gdino" and self.gdino_model is None:
            from src.preprocess.gdino import GDINOPreprocessor

            self.gdino_model = GDINOPreprocessor(**config)

    def apply_plain_mask(
        self,
        frames: List[np.ndarray],
        mask: np.ndarray,
        mask_color: int,
        return_frame: bool = True,
    ):
        """Apply a plain mask to video frames"""
        out_frames = []
        num_frames = len(frames)
        bool_mask = mask > 0
        out_masks = [np.where(bool_mask, 255, 0).astype(np.uint8)] * num_frames

        if not return_frame:
            return None, out_masks

        for i in range(num_frames):
            masked_frame = frames[i].copy()
            masked_frame[bool_mask] = mask_color
            out_frames.append(masked_frame)
        return out_frames, out_masks

    def apply_seg_mask(
        self,
        mask_data: SAM2VideoOutput,
        frames: List[np.ndarray],
        mask_color: int,
        mask_cfg: Optional[Dict] = None,
        return_frame: bool = True,
    ):
        """Apply segmentation masks to video frames"""
        out_frames = []
        out_masks = [
            (single_rle_to_mask(val[0]["mask"]) * 255).astype("uint8")
            for key, val in mask_data["annotations"].items()
        ]

        if not return_frame:
            return None, out_masks

        num_frames = min(len(out_masks), len(frames))
        for i in range(num_frames):
            sub_mask = out_masks[i]
            if self.use_aug and mask_cfg is not None:
                sub_mask = self.mask_aug_annotator(sub_mask, mask_cfg)
            bool_mask = sub_mask > 0
            masked_frame = frames[i].copy()
            masked_frame[bool_mask] = mask_color
            out_frames.append(masked_frame)
        out_masks = out_masks[:num_frames]
        return out_frames, out_masks

    def __call__(
        self,
        frames: Optional[Union[List[Image.Image], List[str], str]] = None,
        fps: Optional[int] = None,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        bbox: Optional[List[float]] = None,
        label: Optional[str] = None,
        caption: Optional[str] = None,
        mode: Optional[str] = None,
        return_frame: Optional[bool] = None,
        return_mask: Optional[bool] = None,
        return_source: Optional[bool] = None,
        mask_color: Optional[int] = None,
        mask_cfg: Optional[Dict] = None,
    ):
        mode = mode if mode is not None else self.mode
        return_frame = return_frame if return_frame is not None else self.return_frame
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = (
            return_source if return_source is not None else self.return_source
        )
        mask_color = mask_color if mask_color is not None else self.mask_color

        frames, fps = self._load_video(frames, fps=fps, return_fps=True)

        # Convert frames to numpy arrays
        frames = [np.array(frame) for frame in frames]
        out_frames, out_masks = [], []

        if mode == "salient":
            if self.salient_model is None:
                self.load_model("salient", self.salient_config)
            first_frame = frames[0]
            output = self.salient_model(first_frame)
            out_frames, out_masks = self.apply_plain_mask(
                frames, output.mask, mask_color, return_frame
            )

        elif mode == "mask":
            if mask is None:
                raise ValueError("Mask is required for 'mask' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2).astype(np.uint8)

            first_frame = frames[0]
            mask_h, mask_w = mask.shape[:2]
            h, w = first_frame.shape[:2]
            if (mask_h != h) or (mask_w != w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            out_frames, out_masks = self.apply_plain_mask(
                frames, mask, mask_color, return_frame
            )

        elif mode == "bbox":
            if bbox is None:
                raise ValueError("Bbox is required for 'bbox' mode")
            first_frame = frames[0]
            num_frames = len(frames)
            x1, y1, x2, y2 = bbox
            if isinstance(x1, float) and x1 < 1:
                x1 = x1 * first_frame.shape[1]
            if isinstance(y1, float) and y1 < 1:
                y1 = y1 * first_frame.shape[0]
            if isinstance(x2, float) and x2 < 1:
                x2 = x2 * first_frame.shape[1]
            if isinstance(y2, float) and y2 < 1:
                y2 = y2 * first_frame.shape[0]
            h, w = first_frame.shape[:2]
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w, x2)), int(min(h, y2))
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            out_masks = [mask] * num_frames

            if not return_frame:
                out_frames = None
            else:
                for i in range(num_frames):
                    masked_frame = frames[i].copy()
                    masked_frame[y1:y2, x1:x2] = mask_color
                    out_frames.append(masked_frame)

        elif mode == "salientmasktrack":
            if self.salient_model is None:
                self.load_model("salient", self.salient_config)
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            first_frame = frames[0]
            salient_mask = self.salient_model(first_frame)
            sam2_output = self.sam2_model(
                video=frames, mask=salient_mask.mask, task_type="mask", fps=fps
            )
            out_frames, out_masks = self.apply_seg_mask(
                sam2_output, frames, mask_color, mask_cfg, return_frame
            )

        elif mode == "salientbboxtrack":
            if self.salient_model is None or self.sam2_model is None:
                self.load_model("salient", self.salient_config)
                self.load_model("sam2", self.sam2_config)
            first_frame = frames[0]
            salient_mask = self.salient_model(first_frame)
            bbox = get_mask_box(np.array(salient_mask.mask), threshold=1)
            if bbox is None:
                raise ValueError("Could not extract bounding box from salient mask")
            mask_data = self.sam2_model(
                video=frames, input_box=bbox, task_type="input_box", fps=fps
            )
            out_frames, out_masks = self.apply_seg_mask(
                mask_data, frames, mask_color, mask_cfg, return_frame
            )
        elif mode == "maskpointtrack":
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            mask_data = self.sam2_model(
                video=frames, mask=mask, task_type="mask_point", fps=fps
            )
            out_frames, out_masks = self.apply_seg_mask(
                mask_data, frames, mask_color, mask_cfg, return_frame
            )
        elif mode == "maskbboxtrack":
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            mask_data = self.sam2_model(
                video=frames, mask=mask, task_type="mask_box", fps=fps
            )
            out_frames, out_masks = self.apply_seg_mask(
                mask_data, frames, mask_color, mask_cfg, return_frame
            )
        elif mode == "masktrack":
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            mask_data = self.sam2_model(video=frames, mask=mask, task_type="mask")
            out_frames, out_masks = self.apply_seg_mask(
                mask_data, frames, mask_color, mask_cfg, return_frame
            )
        elif mode == "bboxtrack":
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            mask_data = self.sam2_model(
                video=frames, input_box=bbox, task_type="input_box", fps=fps
            )
            out_frames, out_masks = self.apply_seg_mask(
                mask_data, frames, mask_color, mask_cfg, return_frame
            )
        elif mode == "label":
            if self.gdino_model is None:
                self.load_model("gdino", self.gdino_config)
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            first_frame = frames[0]
            gdino_res = self.gdino_model(first_frame, classes=label)
            if gdino_res.boxes is not None and len(gdino_res.boxes) > 0:
                bboxes = gdino_res.boxes[0]
            else:
                raise ValueError(
                    f"Unable to find the corresponding boxes of label: {label}"
                )
            mask_data = self.sam2_model(
                video=frames, input_box=bboxes, task_type="input_box"
            )
            out_frames, out_masks = self.apply_seg_mask(
                mask_data, frames, mask_color, mask_cfg, return_frame
            )
        elif mode == "caption":
            if self.gdino_model is None:
                self.load_model("gdino", self.gdino_config)
            if self.sam2_model is None:
                self.load_model("sam2", self.sam2_config)
            first_frame = frames[0]
            gdino_res = self.gdino_model(first_frame, caption=caption)
            if gdino_res.boxes is not None and len(gdino_res.boxes) > 0:
                bboxes = gdino_res.boxes[0]
            else:
                raise ValueError(
                    f"Unable to find the corresponding boxes of caption: {caption}"
                )
            mask_data = self.sam2_model(
                video=frames, input_box=bboxes, task_type="input_box", fps=fps
            )
            out_frames, out_masks = self.apply_seg_mask(
                mask_data, frames, mask_color, mask_cfg, return_frame
            )

        ret_data = defaultdict(list)

        if return_frame:
            ret_data["frames"].extend(out_frames)
        if return_mask:
            ret_data["masks"].extend(out_masks)
        if return_source:
            ret_data["src_frames"].extend(frames)

        return InpaintingVideoOutput(
            frames=ret_data["frames"],
            masks=ret_data["masks"],
            src_frames=ret_data["src_frames"],
        )

    def __str__(self):
        return f"InpaintingVideoPreprocessor(mode={self.mode}, return_frame={self.return_frame})"

    def __repr__(self):
        return self.__str__()
