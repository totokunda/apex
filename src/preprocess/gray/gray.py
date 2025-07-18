# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np
from src.preprocess.base import BasePreprocessor, preprocessor_registry, PreprocessorType
from typing import Union, List
from PIL import Image

@preprocessor_registry("gray.image")
class GrayPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)

    def __call__(self, image: Union[Image.Image, str, np.ndarray]):
        image = self._load_image(image)
        image_array = np.array(image)
        gray_map = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        return gray_map[..., None].repeat(3, axis=2)

    def __str__(self):
        return "GrayPreprocessor()"

    def __repr__(self):
        return self.__str__()

@preprocessor_registry("gray.video")
class GrayVideoPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)

    def __call__(self, frames: Union[List[Image.Image], List[str], List[np.ndarray], str]):
        frames = self._load_video(frames)
        ret_frames = []
        for frame in frames:
            frame_array = np.array(frame)
            gray_map = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
            anno_frame = gray_map[..., None].repeat(3, axis=2)
            ret_frames.append(anno_frame)
        return ret_frames

    def __str__(self):
        return "GrayVideoPreprocessor()"

    def __repr__(self):
        return self.__str__()
