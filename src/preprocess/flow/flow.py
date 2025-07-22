# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import numpy as np
import argparse
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
)
from raft import RAFT
from raft.utils.utils import InputPadder
from raft.utils import flow_viz
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH


@preprocessor_registry("flow")
class FlowPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path: str = None,
        save_path: str = DEFAULT_PREPROCESSOR_SAVE_PATH,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_path, save_path)
        params = {"small": False, "mixed_precision": False, "alternate_corr": False}
        params = argparse.Namespace(**params)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.model = RAFT(params)
        self.model.load_state_dict(
            {
                k.replace("module.", ""): v
                for k, v in torch.load(
                    self.model_path, map_location="cpu", weights_only=True
                ).items()
            }
        )
        self.model = self.model.to(self.device).eval()
        self.InputPadder = InputPadder
        self.flow_viz = flow_viz

    def __call__(self, frames, *args, **kwargs):
        # frames / RGB
        frames = self._load_video(frames)
        frames = [
            np.array(frame)
            .astype(np.uint8)
            .transpose(2, 0, 1)
            .float()[None]
            .to(self.device)
            for frame in frames
        ]
        flow_up_list, flow_up_vis_list = [], []
        with torch.no_grad():
            for i, (image1, image2) in enumerate(zip(frames[:-1], frames[1:])):
                padder = self.InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
                flow_up_vis = self.flow_viz.flow_to_image(flow_up)
                flow_up_list.append(flow_up)
                flow_up_vis_list.append(flow_up_vis)
        return flow_up_list, flow_up_vis_list  # RGB
