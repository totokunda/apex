from src.preprocess.openpose import OpenposeDetector
from src.preprocess.base import BasePreprocessor, preprocessor_registry, PreprocessorType, BaseOutput
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH, DEFAULT_DEVICE
from typing import Union
from PIL import Image
import numpy as np
from typing import List
import torch
from tqdm import tqdm

class OpenposeOutput(BaseOutput):
    detected_map: Image.Image

class OpenposeVideoOutput(BaseOutput):
    outputs: List[Image.Image]

@preprocessor_registry("openpose")
class OpenposePreprocessor(BasePreprocessor):
    def __init__(self, save_path=DEFAULT_PREPROCESSOR_SAVE_PATH, device=DEFAULT_DEVICE):
        self.openpose_detector = OpenposeDetector.from_pretrained(save_path=save_path)
        self.openpose_detector.to(device)

    @torch.inference_mode()
    def __call__(self, image: Union[Image.Image, np.ndarray, str], **kwargs):
        image = self._load_image(image)
        output = self.openpose_detector(image, **kwargs)
        return OpenposeOutput(detected_map=output)
    
    def __str__(self):
        return "OpenposePreprocessor"
    
    def __repr__(self):
        return f"OpenposePreprocessor(save_path={self.save_path})"
    
@preprocessor_registry("openpose.video")
class OpenposeVideoPreprocessor(OpenposePreprocessor):
    def __init__(self, save_path=DEFAULT_PREPROCESSOR_SAVE_PATH, device=DEFAULT_DEVICE):
        super().__init__(save_path=save_path, device=device)

    def __call__(self, video: Union[Image.Image, np.ndarray, str], **kwargs):
        frames = self._load_video(video)
        outputs = []
        for frame in tqdm(frames, desc="Processing video"):
            output = super().__call__(frame, **kwargs)
            outputs.append(output.detected_map)
        return OpenposeVideoOutput(outputs=outputs)
    
    def __str__(self):
        return f"OpenposeVideoPreprocessor(save_path={self.save_path})"
    
    def __repr__(self):
        return f"OpenposeVideoPreprocessor(save_path={self.save_path})"