# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
    BaseOutput,
)
from insightface.app import FaceAnalysis, DEFAULT_MP_NAME
from typing import List
import torch
from PIL import Image
from src.utils.defaults import  DEFAULT_PREPROCESSOR_SAVE_PATH

class FaceOutput(BaseOutput):
    bbox: List[float]
    kps: List[float]
    det_score: float
    landmark_3d_68: List[float]
    pose: List[float]
    landmark_2d_106: List[float]
    gender: List[int]
    images: List[Image.Image] | None = None
    masks: List[Image.Image] | None = None

@preprocessor_registry("face")
class FacePreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_name: str = DEFAULT_MP_NAME,
        save_path: str = DEFAULT_PREPROCESSOR_SAVE_PATH,
        device_id: int = 0,
        return_raw: bool = True,
        return_mask: bool = False,
        return_dict: bool = False,
        multi_face: bool = True,
    ):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE)
        self.model = FaceAnalysis(
            name=model_name,
            root=save_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.model.prepare(ctx_id=device_id, det_size=(640, 640))
        self.return_raw = return_raw
        self.return_mask = return_mask
        self.return_dict = return_dict
        self.multi_face = multi_face
        

    def __call__(
        self,
        image: str | List[str] | np.ndarray | torch.Tensor | Image.Image,
        return_mask: bool = True,
        multi_face: bool = True,
    ):
        image = self._load_image(image)
        image = np.array(image)
        return_mask = return_mask if return_mask is not None else self.return_mask
        # [dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])]
        faces = self.model.get(image)
        
        crop_face_list, mask_list = [], []
        if len(faces) > 0:
            if not self.multi_face:
                faces = faces[:1]
            for face in faces:
                x_min, y_min, x_max, y_max = face["bbox"].tolist()
                crop_face = image[
                    int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1
                ]
                crop_face_list.append(crop_face)
                mask = np.zeros_like(image[:, :, 0])
                mask[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1] = 255
                mask_list.append(mask)
            return FaceOutput(
                images=[Image.fromarray(crop_face) for crop_face in crop_face_list],
                masks=[Image.fromarray(mask) for mask in mask_list] if return_mask else None,
                bbox=[face["bbox"].tolist() for face in faces],
                kps=[face["kps"].tolist() for face in faces],
                det_score=[face["det_score"] for face in faces],
                landmark_3d_68=[face["landmark_3d_68"].tolist() for face in faces],
                pose=[face["pose"].tolist() for face in faces],
                landmark_2d_106=[face["landmark_2d_106"].tolist() for face in faces],
                gender=[face["gender"] for face in faces],
            )
        else:
            return None
