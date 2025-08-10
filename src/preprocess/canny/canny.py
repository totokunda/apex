# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import numpy as np
import cv2
from src.preprocess.base import BasePreprocessor, preprocessor_registry, BaseOutput, PreprocessorType
from typing import List, Literal, Union
import torch
from PIL import Image
from tqdm import tqdm

class CannyOutput(BaseOutput):
    frame: Image.Image

class CannyVideoOutput(BaseOutput):
    frames: List[Image.Image]

@preprocessor_registry("canny")
class CannyPreprocessor(BasePreprocessor):
    def __init__(self, threshold1: int = 100, threshold2: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        
    def __call__(self, image: Union[Image.Image, np.ndarray, str], threshold1: int = None, threshold2: int = None):
        threshold1 = threshold1 if threshold1 is not None else self.threshold1
        threshold2 = threshold2 if threshold2 is not None else self.threshold2
        
        image = self._load_image(image)
        # use cv2 to convert image to numpy array
        array_np = np.array(image)
        # convert to grayscale
        grayscale_image = cv2.cvtColor(array_np, cv2.COLOR_BGR2GRAY)
        canny_image = cv2.Canny(grayscale_image, threshold1, threshold2)
        return CannyOutput(frame=Image.fromarray(canny_image))
    
    def __str__(self):
        return f"CannyPreprocessor(threshold1={self.threshold1}, threshold2={self.threshold2})"

    def __repr__(self):
        return self.__str__()

@preprocessor_registry("canny.video")
class CannyVideoPreprocessor(CannyPreprocessor, BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor_type = PreprocessorType.VIDEO

    def __call__(self, frames: Union[List[Image.Image], List[str], str]):
        frames = self._load_video(frames)
        ret_frames = []
        for frame in tqdm(frames):
            anno_frame = super().__call__(frame)
            ret_frames.append(anno_frame.frame)
        return CannyVideoOutput(frames=ret_frames)

    def __str__(self):
        return f"CannyVideoPreprocessor(threshold1={self.threshold1}, threshold2={self.threshold2})"

    def __repr__(self):
        return self.__str__()
