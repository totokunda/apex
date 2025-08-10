# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import numpy as np
import cv2
from src.preprocess.base import BasePreprocessor, preprocessor_registry, BaseOutput
from typing import List, Literal, Union
import torch
from PIL import Image

class FrameOutput(BaseOutput):
    frames: List[np.ndarray]
    masks: List[np.ndarray]



def align_frames(first_frame, last_frame):
    h1, w1 = first_frame.shape[:2]
    h2, w2 = last_frame.shape[:2]
    if (h1, w1) == (h2, w2):
        return last_frame
    ratio = min(w1 / w2, h1 / h2)
    new_w = int(w2 * ratio)
    new_h = int(h2 * ratio)
    resized = cv2.resize(last_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    aligned = np.ones((h1, w1, 3), dtype=np.uint8) * 255
    x_offset = (w1 - new_w) // 2
    y_offset = (h1 - new_h) // 2
    aligned[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return aligned


@preprocessor_registry("frameref.extract")
class FrameRefExtractPreprocessor(BasePreprocessor):
    def __init__(
        self,
        ref_cfg=None,
        ref_num=1,
        ref_color:int=128,
        return_mask=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # first / last / firstlast / random
        self.ref_cfg = (
            ref_cfg
            if ref_cfg is not None
            else [
                {"mode": "first", "proba": 0.1},
                {"mode": "last", "proba": 0.1},
                {"mode": "firstlast", "proba": 0.1},
                {"mode": "random", "proba": 0.1},
            ]
        )
        self.ref_num = ref_num
        self.ref_color = ref_color
        self.return_mask = return_mask

    def __call__(self, frames: Union[List[str], List[Image.Image], str, Image.Image], ref_cfg=None, ref_num=None, return_mask=True):
        frames = self._load_video(frames)
        frames = [np.array(frame) for frame in frames]

        return_mask = return_mask if return_mask is not None else self.return_mask
        ref_cfg = ref_cfg if ref_cfg is not None else self.ref_cfg
        ref_cfg = [ref_cfg] if not isinstance(ref_cfg, list) else ref_cfg
        probas = [
            item["proba"] if "proba" in item else 1.0 / len(ref_cfg) for item in ref_cfg
        ]
        sel_ref_cfg = random.choices(ref_cfg, weights=probas, k=1)[0]
        mode = sel_ref_cfg["mode"] if "mode" in sel_ref_cfg else "first"
        ref_num = int(ref_num) if ref_num is not None else self.ref_num
        
        frame_num = len(frames)
        frame_num_range = list(range(frame_num))
        if mode == "first":
            sel_idx = frame_num_range[:ref_num]
        elif mode == "last":
            sel_idx = frame_num_range[-ref_num:]
        elif mode == "firstlast":
            sel_idx = frame_num_range[:ref_num] + frame_num_range[-ref_num:]
        elif mode == "random":
            sel_idx = random.sample(frame_num_range, ref_num)
        else:
            raise NotImplementedError

        out_frames, out_masks = [], []
        for i in range(frame_num):
            if i in sel_idx:
                out_frame = frames[i]
                out_mask = np.zeros_like(frames[i][:, :, 0])
            else:
                out_frame = np.ones_like(frames[i]) * int(self.ref_color)
                out_mask = np.ones_like(frames[i][:, :, 0]) * 255
            out_frames.append(out_frame)
            out_masks.append(out_mask)

        return FrameOutput(frames=out_frames, masks=out_masks if return_mask else None)

    def __str__(self):
        return f"FrameRefExtractPreprocessor(ref_cfg={self.ref_cfg}, ref_num={self.ref_num}, ref_color={self.ref_color})"

    def __repr__(self):
        return self.__str__()

expand_mode = Literal["firstframe", "lastframe", "firstlastframe", "firstclip", "lastclip", "firstlastclip", "all"]

@preprocessor_registry("frameref.expand")
class FrameRefExpandPreprocessor(BasePreprocessor):
    def __init__(
        self,
        ref_color:int=128,
        return_mask=True,
        mode: expand_mode = "firstframe",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # first / last / firstlast
        self.ref_color = ref_color
        self.return_mask = return_mask
        self.mode = mode
        assert self.mode in [
            "firstframe",
            "lastframe",
            "firstlastframe",
            "firstclip",
            "lastclip",
            "firstlastclip",
            "all",
        ]

    def __call__(
        self,
        image: str | List[str] | np.ndarray | torch.Tensor | Image.Image = None,
        image_2: str | List[str] | np.ndarray | torch.Tensor | Image.Image = None,
        frames: List[str] | np.ndarray | torch.Tensor | List[Image.Image] = None,
        frames_2:  List[str] | np.ndarray | torch.Tensor | List[Image.Image] = None,
        mode: expand_mode = "firstframe",
        expand_num: int = 24,
        return_mask: bool = True,
        resize: bool = False,
    ):
        mode = mode if mode is not None else self.mode
        return_mask = return_mask if return_mask is not None else self.return_mask

        if "frame" in mode:
            frames = (
                [image]
                if image is not None and not isinstance(frames, list)
                else frames
            )
            frames_2 = (
                [image_2]
                if image_2 is not None and not isinstance(image_2, list)
                else frames_2
            )

        frames = self._load_video(frames)
        frames = [np.array(frame) for frame in frames]
        frames_2 = self._load_video(frames_2) if frames_2 is not None else []
        frames_2 = [np.array(frame) for frame in frames_2]
        
        expand_frames = [np.ones_like(frames[0]) * self.ref_color] * expand_num
        expand_masks = [np.ones_like(frames[0][:, :, 0]) * 255] * expand_num
        source_frames = frames
        source_masks = [np.zeros_like(frames[0][:, :, 0])] * len(frames)
        
        if resize and frames_2:
            # resize frames_2 to frames[0]
            frames_2 = [cv2.resize(f2, (frames[0].shape[1], frames[0].shape[0]), interpolation=cv2.INTER_AREA) for f2 in frames_2]

        if mode in ["firstframe", "firstclip"]:
            out_frames = source_frames + expand_frames
            out_masks = source_masks + expand_masks
        elif mode in ["lastframe", "lastclip"]:
            out_frames = expand_frames + source_frames
            out_masks = expand_masks + source_masks
        elif mode in ["firstlastframe", "firstlastclip"]:
            source_frames_2 = [align_frames(source_frames[0], f2) for f2 in frames_2]
            source_masks_2 = [np.zeros_like(source_frames_2[0][:, :, 0])] * len(
                frames_2
            )
            out_frames = source_frames + expand_frames + source_frames_2
            out_masks = source_masks + expand_masks + source_masks_2
        else:
            raise NotImplementedError

        return FrameOutput(frames=out_frames, masks=out_masks if return_mask else None)

    def __str__(self):
        return (
            f"FrameRefExpandPreprocessor(mode={self.mode}, ref_color={self.ref_color})"
        )

    def __repr__(self):
        return self.__str__()
