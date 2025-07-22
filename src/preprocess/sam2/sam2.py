# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import numpy as np
import torch
from scipy import ndimage
from typing import Union, List, Optional, Dict, Any
from PIL import Image
import warnings

from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
)


def single_mask_to_rle(mask):
    """Convert single mask to RLE format - placeholder function"""
    # This would typically use the proper RLE encoding from SAM2
    return {"size": mask.shape, "counts": "placeholder"}


def single_mask_to_xyxy(mask):
    """Convert single mask to bounding box format"""
    if mask.max() == 0:
        return [0, 0, 0, 0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax + 1, rmax + 1]


@preprocessor_registry("sam2")
class SAM2Preprocessor(BasePreprocessor):
    def __init__(
        self,
        config_path: str,
        model_path: str,
        task_type: str = "input_box",
        return_mask: bool = False,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path, preprocessor_type=PreprocessorType.IMAGE, **kwargs
        )

        self.task_type = task_type
        self.return_mask = return_mask

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            warnings.warn(
                "please pip install sam2 package, or you can refer to models/VACE-Annotators/sam2/SAM_2-1.0-cp310-cp310-linux_x86_64.whl"
            )
            raise ImportError("SAM2 package not available")

        # Handle config path
        local_config_path = os.path.join(*config_path.rsplit("/")[-3:])
        if not os.path.exists(local_config_path):
            os.makedirs(os.path.dirname(local_config_path), exist_ok=True)
            shutil.copy(config_path, local_config_path)

        sam2_model = build_sam2(local_config_path, self.model_path)
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
            sample = {"point_coords": point_coords, "point_labels": point_labels}
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
            return masks[0]
        else:
            ret_data = {"masks": masks, "scores": scores, "logits": logits}
            return ret_data

    def __str__(self):
        return f"SAM2Preprocessor(task_type={self.task_type}, return_mask={self.return_mask})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("sam2.video")
class SAM2VideoPreprocessor(BasePreprocessor):
    def __init__(
        self,
        config_path: str,
        model_path: str,
        task_type: str = "input_box",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path, preprocessor_type=PreprocessorType.VIDEO, **kwargs
        )

        self.task_type = task_type

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            warnings.warn(
                "please pip install sam2 package, or you can refer to models/VACE-Annotators/sam2/SAM_2-1.0-cp310-cp310-linux_x86_64.whl"
            )
            raise ImportError("SAM2 package not available")

        # Handle config path
        local_config_path = os.path.join(*config_path.rsplit("/")[-3:])
        if not os.path.exists(local_config_path):
            os.makedirs(os.path.dirname(local_config_path), exist_ok=True)
            shutil.copy(config_path, local_config_path)

        self.video_predictor = build_sam2_video_predictor(
            local_config_path, self.model_path
        )
        self.video_predictor.fill_hole_area = 0

    def __call__(
        self,
        video: str,
        input_box: Optional[Union[List[float], np.ndarray]] = None,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        task_type: Optional[str] = None,
    ):

        task_type = task_type if task_type is not None else self.task_type

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

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.video_predictor.init_state(video_path=video)

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

        ret_data = {"annotations": video_segments}
        return ret_data

    def __str__(self):
        return f"SAM2VideoPreprocessor(task_type={self.task_type})"

    def __repr__(self):
        return self.__str__()


# Placeholder for combined preprocessors that would depend on other preprocessors
# These would need to be implemented when the dependencies are available
warnings.warn(
    "SAM2SalientVideoPreprocessor and SAM2GDINOVideoPreprocessor require additional dependencies and are not implemented yet."
)
