# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from scipy import ndimage
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
)
from typing import Union, Optional, List
from PIL import Image


@preprocessor_registry("mask.draw")
class MaskDrawPreprocessor(BasePreprocessor):
    def __init__(self, mode="maskpoint", return_dict=True, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        self.mode = mode
        self.return_dict = return_dict
        assert self.mode in ["maskpoint", "maskbbox", "mask", "bbox"]

    def __call__(
        self,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        image: Optional[Union[Image.Image, np.ndarray, str]] = None,
        bbox: Optional[Union[List[float], np.ndarray]] = None,
        mode: Optional[str] = None,
        return_dict: Optional[bool] = None,
    ):
        mode = mode if mode is not None else self.mode
        return_dict = return_dict if return_dict is not None else self.return_dict

        # Load and convert inputs to numpy arrays
        if mask is not None:
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2).astype(np.uint8)

        if image is not None:
            if isinstance(image, (str, Image.Image)):
                image = self._load_image(image)
            image = np.array(image)

        if mask is not None:
            mask_shape = mask.shape
        elif image is not None:
            if len(image.shape) == 3:
                mask_shape = image.shape[:2]
            else:
                mask_shape = image.shape
        else:
            raise ValueError(
                "Either mask or image must be provided to determine output shape"
            )

        out_image = None

        if mode == "maskpoint":
            if mask is None:
                raise ValueError("Mask is required for 'maskpoint' mode")
            scribble = mask.transpose(1, 0)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(
                scribble, labeled_array, range(1, num_features + 1)
            )
            centers = np.array(centers)
            out_mask = np.zeros(mask_shape, dtype=np.uint8)
            if len(centers) >= 3:  # Need at least 3 points for ConvexHull
                hull = ConvexHull(centers)
                hull_vertices = centers[hull.vertices]
                rr, cc = polygon(hull_vertices[:, 1], hull_vertices[:, 0], mask_shape)
                out_mask[rr, cc] = 255
            elif len(centers) > 0:
                # If less than 3 points, just mark the center points
                for center in centers:
                    y, x = int(center[1]), int(center[0])
                    if 0 <= y < mask_shape[0] and 0 <= x < mask_shape[1]:
                        out_mask[y, x] = 255

        elif mode == "maskbbox":
            if mask is None:
                raise ValueError("Mask is required for 'maskbbox' mode")
            scribble = mask.transpose(1, 0)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(
                scribble, labeled_array, range(1, num_features + 1)
            )
            centers = np.array(centers)
            if len(centers) > 0:
                # (x1, y1, x2, y2)
                x_min = centers[:, 0].min()
                x_max = centers[:, 0].max()
                y_min = centers[:, 1].min()
                y_max = centers[:, 1].max()
                out_mask = np.zeros(mask_shape, dtype=np.uint8)
                out_mask[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1] = 255
                if image is not None:
                    out_image = image[
                        int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1
                    ]
            else:
                out_mask = np.zeros(mask_shape, dtype=np.uint8)

        elif mode == "bbox":
            if bbox is None:
                raise ValueError("Bbox is required for 'bbox' mode")
            if isinstance(bbox, list):
                bbox = np.array(bbox)
            x_min, y_min, x_max, y_max = bbox
            out_mask = np.zeros(mask_shape, dtype=np.uint8)
            out_mask[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1] = 255
            if image is not None:
                out_image = image[
                    int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1
                ]

        elif mode == "mask":
            if mask is None:
                raise ValueError("Mask is required for 'mask' mode")
            out_mask = mask

        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented")

        if return_dict:
            result = {"mask": out_mask}
            if out_image is not None:
                result["image"] = out_image
            return result
        else:
            if out_image is not None:
                return out_image, out_mask
            else:
                return out_mask

    def __str__(self):
        return f"MaskDrawPreprocessor(mode={self.mode}, return_dict={self.return_dict})"

    def __repr__(self):
        return self.__str__()
