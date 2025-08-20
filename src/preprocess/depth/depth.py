from PIL import Image
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
)
from typing import Union
import numpy as np
import torch
from typing import Union, List, Optional
from .midas import MiDaSInference
from .dpt_da import DepthAnythingV2
from einops import rearrange
import cv2
from src.preprocess.base import BaseOutput
from src.utils.preprocessors import MODEL_WEIGHTS
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH, DEFAULT_DEVICE
from tqdm import tqdm


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
    )
    return img, k


def resize_image_ori(h, w, image, k):
    img = cv2.resize(
        image, (w, h), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    )
    return img


class DepthOutput(BaseOutput):
    depth: Image.Image


class VideoDepthOutput(BaseOutput):
    depth: List[Image.Image]


@preprocessor_registry("depth.midas")
class MidasDepthPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path: Optional[str] = None,
        save_path: Optional[str] = None,
        model_type: str = "dpt_hybrid",
        device: str = DEFAULT_DEVICE,
        **kwargs,
    ):
        if model_path is None:
            model_path = MODEL_WEIGHTS["depth.midas"]
        if save_path is None:
            save_path = DEFAULT_PREPROCESSOR_SAVE_PATH
        super().__init__(
            model_path, save_path, preprocessor_type=PreprocessorType.IMAGE, **kwargs
        )
        self.midas_inference = MiDaSInference(
            model_type=model_type, model_path=self.model_path
        )
        self.midas_inference.to(device)
        self.device = device
        self.model_type = model_type

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(
        self,
        image: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
    ):
        loaded_image = self._load_image(image)
        np_image = np.array(loaded_image)
        image_depth = np_image
        h, w, c = image_depth.shape
        image_depth, k = resize_image(
            image_depth, 1024 if min(h, w) > 1024 else min(h, w)
        )
        image_depth = torch.from_numpy(image_depth).float().to(self.device)
        image_depth = image_depth / 127.5 - 1.0
        image_depth = rearrange(image_depth, "h w c -> 1 c h w")
        depth = self.midas_inference(image_depth)[0]

        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)
        depth_pt = depth_pt.cpu().numpy()
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)
        depth_image = depth_image[..., None].repeat(3, 2)

        depth_image = resize_image_ori(h, w, depth_image, k)

        return DepthOutput(depth=Image.fromarray(depth_image))

    def __str__(self):
        return f"MidasDepthPreprocessor(model_path={self.model_path}, save_path={self.save_path}, model_type={self.model_type})"


@preprocessor_registry("depth.anything_v2")
class DepthAnythingV2Preprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path: Optional[str] = None,
        save_path: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        encoder: str = "vitl",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        **kwargs,
    ):
        if model_path is None:
            model_path = MODEL_WEIGHTS["depth.anything_v2"]
        if save_path is None:
            save_path = DEFAULT_PREPROCESSOR_SAVE_PATH
        super().__init__(
            model_path, save_path, preprocessor_type=PreprocessorType.IMAGE, **kwargs
        )
        self.model = DepthAnythingV2(
            encoder=encoder, features=features, out_channels=out_channels
        ).to(device)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=device, mmap=True)
        )
        self.model.eval()

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(
        self,
        image: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
    ):
        loaded_image = self._load_image(image)
        np_image = np.array(loaded_image)
        depth = self.model.infer_image(np_image)
        depth_pt = depth.copy()
        depth_pt -= np.min(depth_pt)
        depth_pt /= np.max(depth_pt)
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)
        depth_image = depth_image[..., np.newaxis]
        depth_image = np.repeat(depth_image, 3, axis=2)
        return DepthOutput(depth=Image.fromarray(depth_image))


@preprocessor_registry("depth.video.anything_v2")
class VideoDepthAnythingV2Preprocessor(DepthAnythingV2Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        video: Union[
            str, List[str], List[Image.Image], List[np.ndarray], List[torch.Tensor]
        ],
    ):
        frames = self._load_video(video)
        depths = []
        for frame in tqdm(frames):
            depth = super().__call__(frame)
            depths.append(depth.depth)
        return VideoDepthOutput(depth=depths)


@preprocessor_registry("depth.video.midas")
class VideoMidasDepthPreprocessor(MidasDepthPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        video: Union[
            str, List[str], List[Image.Image], List[np.ndarray], List[torch.Tensor]
        ],
    ):
        frames = self._load_video(video)
        depths = []
        for frame in tqdm(frames):
            depth = super().__call__(frame)
            depths.append(depth.depth)
        return VideoDepthOutput(depth=depths)
