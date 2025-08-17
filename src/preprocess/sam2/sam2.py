# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import numpy as np
import torch
from scipy import ndimage
from typing import Union, List, Optional, Dict
from PIL import Image
import warnings
from src.utils.preprocessors import MODEL_WEIGHTS, MODEL_CONFIGS
from src.utils.defaults import DEFAULT_DEVICE
from diffusers.utils import export_to_video
import tempfile
from src.utils.defaults import DEFAULT_CACHE_PATH

from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
    BaseOutput,
)
from src.preprocess.salient import SalientPreprocessor
from src.preprocess.gdino import GDINOPreprocessor
import pycocotools.mask as mask_utils


def single_mask_to_rle(mask):
    rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def single_mask_to_xyxy(mask):
    """Convert single mask to bounding box format"""
    if mask.max() == 0:
        return [0, 0, 0, 0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax + 1, rmax + 1]


class SAM2Output(BaseOutput):
    masks: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    logits: Optional[np.ndarray] = None


class SAM2VideoOutput(BaseOutput):
    annotations: Optional[Dict] = None


@preprocessor_registry("sam2")
class SAM2Preprocessor(BasePreprocessor):
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        task_type: str = "input_box",
        return_mask: bool = False,
        device: str = DEFAULT_DEVICE,
        **kwargs,
    ):
        if model_path is None:
            model_path = MODEL_WEIGHTS["sam2"]
        if config_path is None:
            config_path = MODEL_CONFIGS["sam2"]
        super().__init__(
            model_path=model_path,
            config_path=config_path,
            preprocessor_type=PreprocessorType.IMAGE,
            **kwargs,
        )

        self.task_type = task_type
        self.return_mask = return_mask
        self.device = device

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            warnings.warn("please pip install sam2 package")
            raise ImportError("SAM2 package not available")

        # The SAM2 package expects a Hydra config name like "configs/sam2.1/sam2.1_hiera_l.yaml",
        # not a filesystem path. Convert any local/URL path to the expected relative config name.
        def _to_sam2_config_name(path_like: str) -> str:
            if path_like is None:
                return None
            normalized = str(path_like).replace("\\", "/")
            # If it's a URL, extract the path portion
            if normalized.startswith("http://") or normalized.startswith("https://"):
                from urllib.parse import urlparse

                normalized = urlparse(normalized).path.lstrip("/")
            # Try to find the packaged config root
            marker = "configs/"
            idx = normalized.find(marker)
            return normalized[idx:] if idx != -1 else normalized

        config_name = _to_sam2_config_name(self.config_path)
        sam2_model = build_sam2(
            config_name, ckpt_path=self.model_path, device=self.device
        )
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.predictor.fill_hole_area = 0

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        input_box: Optional[Union[List[float], np.ndarray]] = None,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        task_type: Optional[str] = None,
        return_mask: Optional[bool] = None,
    ):

        task_type = task_type if task_type is not None else self.task_type
        return_mask = return_mask if return_mask is not None else self.return_mask

        image = self._load_image(image)
        image_array = np.array(image)

        if mask is not None:
            mask = self._load_image(mask)
            mask = mask.convert("L")
            if task_type == "mask":
                mask = mask.resize((256, 256))
            mask = np.array(mask)

        if task_type == "mask_point":
            if mask is None:
                raise ValueError("Mask is required for 'mask_point' task type")
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)  # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble > 0)
            centers = ndimage.center_of_mass(
                scribble, labeled_array, range(1, num_features + 1)
            )
            point_coords = np.array(centers)
            point_labels = np.array([1] * len(centers))
            sample = {"point_coords": point_coords, "point_labels": point_labels}
        elif task_type == "mask_box":
            if mask is None:
                raise ValueError("Mask is required for 'mask_box' task type")
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)  # (H, W) -> (W, H)

            labeled_array, num_features = ndimage.label(scribble > 0)
            centers = ndimage.center_of_mass(
                scribble, labeled_array, range(1, num_features + 1)
            )
            centers = np.array(centers)
            # (x1, y1, x2, y2)
            x_min = centers[:, 0].min()
            x_max = centers[:, 0].max()
            y_min = centers[:, 1].min()
            y_max = centers[:, 1].max()
            bbox = np.array([x_min, y_min, x_max, y_max])
            sample = {"box": bbox}
        elif task_type == "input_box":
            if input_box is None:
                raise ValueError("input_box is required for 'input_box' task type")
            if isinstance(input_box, list):
                input_box = self.preprocess_bbox(input_box, np.array(image).shape)
                input_box = np.array(input_box)
            sample = {"box": input_box}
        elif task_type == "mask":
            if mask is None:
                raise ValueError("Mask is required for 'mask' task type")
            sample = {"mask_input": mask[None, :, :]}
        else:
            raise NotImplementedError(f"Task type '{task_type}' is not implemented")

        self.predictor.set_image(image_array)
        masks, scores, logits = self.predictor.predict(multimask_output=False, **sample)
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        if return_mask:
            return SAM2Output(masks=masks[0])
        else:
            return SAM2Output(masks=masks, scores=scores, logits=logits)

    def __str__(self):
        return f"SAM2Preprocessor(task_type={self.task_type}, return_mask={self.return_mask})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("sam2.video")
class SAM2VideoPreprocessor(BasePreprocessor):
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        task_type: str = "input_box",
        device: str | torch.device = DEFAULT_DEVICE,
        **kwargs,
    ):
        if model_path is None:
            model_path = MODEL_WEIGHTS["sam2"]
        if config_path is None:
            config_path = MODEL_CONFIGS["sam2"]

        super().__init__(
            model_path=model_path,
            config_path=config_path,
            preprocessor_type=PreprocessorType.VIDEO,
            **kwargs,
        )

        self.task_type = task_type
        self.device = device
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            warnings.warn("please pip install sam2 package")
            raise ImportError("SAM2 package not available")

        def _to_sam2_config_name(path_like: str) -> str:
            if path_like is None:
                return None
            normalized = str(path_like).replace("\\", "/")
            if normalized.startswith("http://") or normalized.startswith("https://"):
                from urllib.parse import urlparse

                normalized = urlparse(normalized).path.lstrip("/")
            marker = "configs/"
            idx = normalized.find(marker)
            return normalized[idx:] if idx != -1 else normalized

        config_name = _to_sam2_config_name(self.config_path)
        self.video_predictor = build_sam2_video_predictor(
            config_name, ckpt_path=self.model_path, device=self.device
        )
        self.video_predictor.fill_hole_area = 0

    def __call__(
        self,
        video: str | List[Image.Image] | List[np.ndarray],
        fps: int = 24,
        input_box: Optional[Union[List[float], np.ndarray]] = None,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        task_type: Optional[str] = None,
    ):

        task_type = task_type if task_type is not None else self.task_type

        video_path = None
        tmp_video = None
        
        if isinstance(video, str):
            video_path = video
        elif isinstance(video, list):
            tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", dir=DEFAULT_CACHE_PATH, delete=False)
            export_to_video(video, tmp_video.name, fps=fps)
            video_path = tmp_video.name
        else:
            raise ValueError(f"Unsupported video type: {type(video)}")

        if mask is not None:
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2).astype(np.uint8)

        if task_type == "mask_point":
            if mask is None:
                raise ValueError("Mask is required for 'mask_point' task type")
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)  # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(
                scribble, labeled_array, range(1, num_features + 1)
            )
            point_coords = np.array(centers)
            point_labels = np.array([1] * len(centers))
            sample = {"points": point_coords, "labels": point_labels}
        elif task_type == "mask_box":
            if mask is None:
                raise ValueError("Mask is required for 'mask_box' task type")
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)  # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(
                scribble, labeled_array, range(1, num_features + 1)
            )
            centers = np.array(centers)
            # (x1, y1, x2, y2)
            x_min = centers[:, 0].min()
            x_max = centers[:, 0].max()
            y_min = centers[:, 1].min()
            y_max = centers[:, 1].max()
            bbox = np.array([x_min, y_min, x_max, y_max])
            sample = {"box": bbox}
        elif task_type == "input_box":
            if input_box is None:
                raise ValueError("input_box is required for 'input_box' task type")
            if isinstance(input_box, list):
                input_box = self.preprocess_bbox(
                    input_box, np.array(self._load_video(video_path)[0]).shape
                )
                input_box = np.array(input_box)
            sample = {"box": input_box}
        elif task_type == "mask":
            if mask is None:
                raise ValueError("Mask is required for 'mask' task type")
            sample = {"mask": mask}
        else:
            raise NotImplementedError(f"Task type '{task_type}' is not implemented")

        ann_frame_idx = 0
        object_id = 0

        with torch.inference_mode(), torch.autocast(
            self.device.type if isinstance(self.device, torch.device) else self.device,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        ):
            inference_state = self.video_predictor.init_state(video_path=video_path)

            if task_type in ["mask_point", "mask_box", "input_box"]:
                _, out_obj_ids, out_mask_logits = (
                    self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=object_id,
                        **sample,
                    )
                )
            elif task_type in ["mask"]:
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    **sample,
                )
            else:
                raise NotImplementedError(f"Task type '{task_type}' is not implemented")

            video_segments = (
                {}
            )  # video_segments contains the per-frame segmentation results
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.video_predictor.propagate_in_video(inference_state):
                frame_segments = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
                    frame_segments[out_obj_id] = {
                        "mask": single_mask_to_rle(mask),
                        "mask_area": int(mask.sum()),
                        "mask_box": single_mask_to_xyxy(mask),
                    }
                video_segments[out_frame_idx] = frame_segments
        if tmp_video is not None:
            tmp_video.close()
            os.unlink(tmp_video.name)
        return SAM2VideoOutput(annotations=video_segments)

    def __str__(self):
        return f"SAM2VideoPreprocessor(task_type={self.task_type})"

    def __repr__(self):
        return self.__str__()


class SAM2SalientVideoPreprocessor(BasePreprocessor):
    def __init__(self, sam2_config: Dict = None, salient_config: Dict = None, **kwargs):
        super().__init__(**kwargs)

        self.sam2_preprocessor = SAM2VideoPreprocessor(
            **(sam2_config if sam2_config is not None else {})
        )
        self.salient_preprocessor = SalientPreprocessor(
            **(salient_config if salient_config is not None else {})
        )

    def __call__(self, video: str | Image.Image | np.ndarray, **kwargs):
        loaded_video = self._load_video(video)
        image = loaded_video[0]

        salient_output = self.salient_preprocessor(image, **kwargs)
        sam_2_output = self.sam2_preprocessor(
            video, mask=salient_output.mask, task_type="mask"
        )

        return sam_2_output


class SAM2GDINOVideoPreprocessor(BasePreprocessor):
    def __init__(self, sam2_config: Dict = None, gdino_config: Dict = None, **kwargs):
        super().__init__(**kwargs)

        self.sam2_preprocessor = SAM2VideoPreprocessor(
            **(sam2_config if sam2_config is not None else {})
        )
        self.gdino_preprocessor = GDINOPreprocessor(
            **(gdino_config if gdino_config is not None else {})
        )

    def __call__(
        self,
        video: str | Image.Image | np.ndarray,
        classes: List[str] = None,
        caption: str = None,
        **kwargs,
    ):
        loaded_video = self._load_video(video)
        image = loaded_video[0]

        if classes is not None:
            gdino_output = self.gdino_preprocessor(image, classes=classes, **kwargs)
        else:
            gdino_output = self.gdino_preprocessor(image, caption=caption, **kwargs)
        if gdino_output.boxes is not None and len(gdino_output.boxes) > 0:
            bboxes = gdino_output.boxes[0]
        else:
            raise ValueError("Unable to find the corresponding boxes")
        sam2_output = self.sam2_preprocessor(
            video=video, input_box=bboxes, task_type="input_box"
        )
        return sam2_output
