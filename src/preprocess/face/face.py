# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from src.preprocess.base import BasePreprocessor, preprocessor_registry, PreprocessorType
from insightface.app import FaceAnalysis
from typing import List
import torch
from PIL import Image

@preprocessor_registry("face")
class FacePreprocessor(BasePreprocessor):
    def __init__(self, model_name: str=None, save_path: str=None, device: str = 'cuda', return_raw: bool = True, return_mask: bool = False, return_dict: bool = False, multi_face: bool = True):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE)
        self.model = FaceAnalysis(name=model_name, root=save_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=device, det_size=(640, 640))
        self.return_raw = return_raw
        self.return_mask = return_mask
        self.return_dict = return_dict
        self.multi_face = multi_face

    def __call__(self, image: str | List[str] | np.ndarray | torch.Tensor | Image.Image, return_mask: bool = False, return_dict: bool = False, multi_face: bool = False, return_raw: bool = False):
        image = self._load_image(image)
        image = np.array(image)
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_dict = return_dict if return_dict is not None else self.return_dict
        # [dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])]
        faces = self.model.get(image)
        if self.return_raw:
            return faces
        else:
            crop_face_list, mask_list = [], []
            if len(faces) > 0:
                if not self.multi_face:
                    faces = faces[:1]
                for face in faces:
                    x_min, y_min, x_max, y_max = face['bbox'].tolist()
                    crop_face = image[int(y_min): int(y_max) + 1, int(x_min): int(x_max) + 1]
                    crop_face_list.append(crop_face)
                    mask = np.zeros_like(image[:, :, 0])
                    mask[int(y_min): int(y_max) + 1, int(x_min): int(x_max) + 1] = 255
                    mask_list.append(mask)
                if not self.multi_face:
                    crop_face_list = crop_face_list[0]
                    mask_list = mask_list[0]
                if return_mask:
                    if return_dict:
                        return {'image': crop_face_list, 'mask': mask_list}
                    else:
                        return crop_face_list, mask_list
                else:
                    return crop_face_list
            else:
                return None
        