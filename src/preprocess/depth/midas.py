# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from .midas_blocks import (
    FeatureFusionBlock,
    Interpolate,
    _make_encoder,
    FeatureFusionBlock_custom,
)

from torchvision import transforms as T
import cv2
from .transforms import NormalizeImage, PrepareForNet, Resize
from .dpt_midas import DPTDepthModel


def load_midas_transform(model_type):
    # https://github.com/isl-org/MiDaS/blob/master/run.py
    # load transform only
    if model_type == "dpt_large":  # DPT-Large
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21":
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small":
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type large"

    transform = T.Compose(
        [
            T.Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return transform


def load_model(model_type, model_path):
    # https://github.com/isl-org/MiDaS/blob/master/run.py
    # load network
    # model_path = ISL_PATHS[model_type]
    if model_type == "dpt_large":  # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "midas_v21_small":
        model = MidasNet_small(
            model_path,
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={"expand": True},
        )
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return model.eval(), transform


class MidasNet(nn.Module):
    """Network for monocular depth estimation."""

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(
            backbone="resnext101_wsl", features=features, use_pretrained=use_pretrained
        )

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)


class MidasNet_small(nn.Module):
    """Network for monocular depth estimation."""

    def __init__(
        self,
        path=None,
        features=64,
        backbone="efficientnet_lite3",
        non_negative=True,
        exportable=True,
        channels_last=False,
        align_corners=True,
        blocks={"expand": True},
    ):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet_small, self).__init__()

        use_pretrained = False if path else True

        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks["expand"] is True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        self.pretrained, self.scratch = _make_encoder(
            self.backbone,
            features,
            use_pretrained,
            groups=self.groups,
            expand=self.expand,
            exportable=exportable,
        )

        self.scratch.activation = nn.ReLU(False)

        self.scratch.refinenet4 = FeatureFusionBlock_custom(
            features4,
            self.scratch.activation,
            deconv=False,
            bn=False,
            expand=self.expand,
            align_corners=align_corners,
        )
        self.scratch.refinenet3 = FeatureFusionBlock_custom(
            features3,
            self.scratch.activation,
            deconv=False,
            bn=False,
            expand=self.expand,
            align_corners=align_corners,
        )
        self.scratch.refinenet2 = FeatureFusionBlock_custom(
            features2,
            self.scratch.activation,
            deconv=False,
            bn=False,
            expand=self.expand,
            align_corners=align_corners,
        )
        self.scratch.refinenet1 = FeatureFusionBlock_custom(
            features1,
            self.scratch.activation,
            deconv=False,
            bn=False,
            align_corners=align_corners,
        )

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(
                features,
                features // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.groups,
            ),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        if self.channels_last is True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)


def fuse_model(m):
    prev_previous_type = nn.Identity()
    prev_previous_name = ""
    previous_type = nn.Identity()
    previous_name = ""
    for name, module in m.named_modules():
        if (
            prev_previous_type == nn.Conv2d
            and previous_type == nn.BatchNorm2d
            and type(module) == nn.ReLU
        ):
            # print("FUSED ", prev_previous_name, previous_name, name)
            torch.quantization.fuse_modules(
                m, [prev_previous_name, previous_name, name], inplace=True
            )
        elif prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d:
            # print("FUSED ", prev_previous_name, previous_name)
            torch.quantization.fuse_modules(
                m, [prev_previous_name, previous_name], inplace=True
            )
        # elif previous_type == nn.Conv2d and type(module) == nn.ReLU:
        #    print("FUSED ", previous_name, name)
        #    torch.quantization.fuse_modules(m, [previous_name, name], inplace=True)

        prev_previous_type = previous_type
        prev_previous_name = previous_name
        previous_type = type(module)
        previous_name = name


class MiDaSInference(nn.Module):
    MODEL_TYPES_TORCH_HUB = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    MODEL_TYPES_ISL = [
        "dpt_large",
        "dpt_hybrid",
        "midas_v21",
        "midas_v21_small",
    ]

    def __init__(self, model_type, model_path):
        super().__init__()
        assert model_type in self.MODEL_TYPES_ISL
        model, _ = load_model(model_type, model_path)
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            prediction = self.model(x)
        return prediction
