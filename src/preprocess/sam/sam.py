# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from scipy import ndimage
from typing import Union, List, Optional, Dict, Any
from PIL import Image

from src.preprocess.base import BasePreprocessor, preprocessor_registry, PreprocessorType


@preprocessor_registry("sam")
class SAMPreprocessor(BasePreprocessor):
    def __init__(self, model_path: str, model_name: str = 'vit_b', task_type: str = 'input_box', 
                 return_mask: bool = False, device: str = 'cuda', **kwargs):
        super().__init__(model_path=model_path, preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            from segment_anything.utils.transforms import ResizeLongestSide
        except ImportError:
            import warnings
            warnings.warn("please pip install sam package, or you can refer to models/VACE-Annotators/sam/segment_anything-1.0-py3-none-any.whl")
            raise ImportError("SAM package not available")
        
        self.task_type = task_type
        self.return_mask = return_mask
        self.transform = ResizeLongestSide(1024)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == 'cuda' else torch.device(device)
        
        seg_model = sam_model_registry[model_name](checkpoint=self.model_path).eval().to(self.device)
        self.predictor = SamPredictor(seg_model)

    def __call__(self,
                image: Union[Image.Image, np.ndarray, str],
                input_box: Optional[Union[List[float], np.ndarray]] = None,
                mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
                task_type: Optional[str] = None,
                return_mask: Optional[bool] = None):
        
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

        if task_type == 'mask_point':
            if mask is None:
                raise ValueError("Mask is required for 'mask_point' task type")
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)   # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            point_coords = np.array(centers)
            point_labels = np.array([1] * len(centers))
            sample = {
                'point_coords': point_coords,
                'point_labels': point_labels
            }
        elif task_type == 'mask_box':
            if mask is None:
                raise ValueError("Mask is required for 'mask_box' task type")
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)  # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            centers = np.array(centers)
            # (x1, y1, x2, y2)
            x_min = centers[:, 0].min()
            x_max = centers[:, 0].max()
            y_min = centers[:, 1].min()
            y_max = centers[:, 1].max()
            bbox = np.array([x_min, y_min, x_max, y_max])
            sample = {'box': bbox}
        elif task_type == 'input_box':
            if input_box is None:
                raise ValueError("input_box is required for 'input_box' task type")
            if isinstance(input_box, list):
                input_box = np.array(input_box)
            sample = {'box': input_box}
        elif task_type == 'mask':
            if mask is None:
                raise ValueError("Mask is required for 'mask' task type")
            sample = {'mask_input': mask[None, :, :]}
        else:
            raise NotImplementedError(f"Task type '{task_type}' is not implemented")

        self.predictor.set_image(image_array)
        masks, scores, logits = self.predictor.predict(
            multimask_output=False,
            **sample
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        
        if return_mask:
            return masks[0]
        else:
            ret_data = {
                "masks": masks,
                "scores": scores,
                "logits": logits
            }
            return ret_data

    def __str__(self):
        return f"SAMPreprocessor(task_type={self.task_type}, return_mask={self.return_mask})"

    def __repr__(self):
        return self.__str__() 