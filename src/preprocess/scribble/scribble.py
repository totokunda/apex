# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from typing import Union, List, Optional
from PIL import Image
from tqdm import tqdm
from src.utils.defaults import DEFAULT_DEVICE
from src.utils.preprocessors import MODEL_WEIGHTS

from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
    BaseOutput,
)

norm_layer = nn.InstanceNorm2d


class ScribbleOutput(BaseOutput):
    image: Image.Image


class ScribbleVideoOutput(BaseOutput):
    frames: List[Image.Image]


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ContourInference(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(ContourInference, self).__init__()

        # Initial convolution block
        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True),
        ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


@preprocessor_registry("scribble")
class ScribblePreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path: str | None = None,
        input_nc: int = 3,
        output_nc: int = 1,
        n_residual_blocks: int = 3,
        sigmoid: bool = True,
        device: str = DEFAULT_DEVICE,
        **kwargs
    ):
        if model_path is None:
            model_path = MODEL_WEIGHTS["scribble"]
        super().__init__(
            model_path=model_path, preprocessor_type=PreprocessorType.IMAGE, **kwargs
        )

        self.device = device
        self.model = ContourInference(input_nc, output_nc, n_residual_blocks, sigmoid)
        self.model.load_state_dict(
            torch.load(self.model_path, weights_only=True, map_location=self.device)
        )
        self.model = self.model.eval().requires_grad_(False).to(self.device)

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, image: Union[Image.Image, np.ndarray, str, List]):
        # Handle batch vs single image
        is_batch = False
        if isinstance(image, list):
            is_batch = True
            images = [self._load_image(img) for img in image]
            image_arrays = [np.array(img) for img in images]
        else:
            image = self._load_image(image)
            image_arrays = [np.array(image)]
            is_batch = len(image_arrays[0].shape) == 4  # Check if batched

        # Convert to torch tensor
        if not is_batch and len(image_arrays[0].shape) == 3:
            # Single image case
            image_tensor = (
                torch.from_numpy(image_arrays[0]).permute(2, 0, 1).unsqueeze(0)
            )  # H,W,C -> 1,C,H,W
        else:
            # Batch case
            if len(image_arrays[0].shape) == 3:
                # List of single images
                image_tensor = torch.stack(
                    [torch.from_numpy(img).permute(2, 0, 1) for img in image_arrays]
                )
            else:
                # Already batched
                image_tensor = torch.from_numpy(image_arrays[0]).permute(
                    0, 3, 1, 2
                )  # B,H,W,C -> B,C,H,W
            is_batch = True

        image_tensor = image_tensor.float().div(255).to(self.device)
        contour_map = self.model(image_tensor)
        contour_map = (
            (contour_map.squeeze(dim=1) * 255.0)
            .clip(0, 255)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        contour_map = contour_map[..., None].repeat(3, -1)

        if not is_batch:
            contour_map = contour_map.squeeze()

        return ScribbleOutput(image=Image.fromarray(contour_map))

    def __str__(self):
        return "ScribblePreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("scribble.video")
class ScribbleVideoPreprocessor(ScribblePreprocessor, BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor_type = PreprocessorType.VIDEO

    def __call__(self, frames: Union[List[Image.Image], List[str], str]):
        frames = self._load_video(frames)
        ret_frames = []
        for frame in tqdm(frames):
            anno_frame = super().__call__(frame)
            ret_frames.append(anno_frame.image)
        return ScribbleVideoOutput(frames=ret_frames)

    def __str__(self):
        return "ScribbleVideoPreprocessor()"

    def __repr__(self):
        return self.__str__()
