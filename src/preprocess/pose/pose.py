# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import cv2
import torch
import numpy as np
from typing import Union, List, Optional
from PIL import Image
from tqdm import tqdm
from src.utils.preprocessors import MODEL_WEIGHTS
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH, DEFAULT_DEVICE
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
    BaseOutput,
)
from ..dwpose import util
from ..dwpose.wholebody import Wholebody, HWC3, resize_image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PoseOutput(BaseOutput):
    detected_map_body: np.ndarray | None = None
    detected_map_face: np.ndarray | None = None
    detected_map_bodyface: np.ndarray | None = None
    detected_map_handbodyface: np.ndarray | None = None
    det_result: np.ndarray | None = None


class PoseBodyFaceOutput(BaseOutput):
    detected_map_bodyface: np.ndarray | None = None


class PoseBodyOutput(BaseOutput):
    detected_map_body: np.ndarray | None = None


class PoseVideoOutput(BaseOutput):
    frames: List[PoseOutput]


class PoseBodyFaceVideoOutput(BaseOutput):
    frames: List[PoseBodyFaceOutput]


class PoseBodyVideoOutput(BaseOutput):
    frames: List[PoseBodyOutput]


def draw_pose(pose, H, W, use_hand=False, use_body=False, use_face=False):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if use_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if use_hand:
        canvas = util.draw_handpose(canvas, hands)
    if use_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


@preprocessor_registry("pose")
class PosePreprocessor(BasePreprocessor):
    def __init__(
        self,
        detection_model: Optional[str] = None,
        pose_model: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        resize_size: int = 1024,
        use_body: bool = True,
        use_face: bool = True,
        use_hand: bool = True,
        save_path: str = DEFAULT_PREPROCESSOR_SAVE_PATH,
        **kwargs,
    ):
        if detection_model is None:
            detection_model = MODEL_WEIGHTS["pose_detection"]
        if pose_model is None:
            pose_model = MODEL_WEIGHTS["pose"]
        super().__init__(preprocessor_type=PreprocessorType.POSE, **kwargs)
        self.device = device
        detection_model = self._download(detection_model, save_path=save_path)
        pose_model = self._download(pose_model, save_path=save_path)
        self.pose_estimation = Wholebody(
            detection_model, pose_model, device=str(self.device)
        )
        self.resize_size = resize_size
        self.use_body = use_body
        self.use_face = use_face
        self.use_hand = use_hand

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, image: Union[Image.Image, np.ndarray, str]):
        image = self._load_image(image)
        image_array = np.array(image)
        input_image = HWC3(image_array[..., ::-1])  # RGB to BGR
        return self.process(
            resize_image(input_image, self.resize_size), image_array.shape[:2]
        )

    def process(self, ori_img, ori_shape):
        ori_h, ori_w = ori_shape
        ori_img = ori_img.copy()
        H, W, C = ori_img.shape

        with torch.no_grad():
            candidate, subset, det_result = self.pose_estimation(ori_img)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]

            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]
            faces = candidate[:, 24:92]
            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            ret_data = {}
            if self.use_body:
                detected_map_body = draw_pose(pose, H, W, use_body=True)
                detected_map_body = cv2.resize(
                    detected_map_body[..., ::-1],
                    (ori_w, ori_h),
                    interpolation=(
                        cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA
                    ),
                )
                ret_data["detected_map_body"] = detected_map_body

            if self.use_face:
                detected_map_face = draw_pose(pose, H, W, use_face=True)
                detected_map_face = cv2.resize(
                    detected_map_face[..., ::-1],
                    (ori_w, ori_h),
                    interpolation=(
                        cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA
                    ),
                )
                ret_data["detected_map_face"] = detected_map_face

            if self.use_body and self.use_face:
                detected_map_bodyface = draw_pose(
                    pose, H, W, use_body=True, use_face=True
                )
                detected_map_bodyface = cv2.resize(
                    detected_map_bodyface[..., ::-1],
                    (ori_w, ori_h),
                    interpolation=(
                        cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA
                    ),
                )
                ret_data["detected_map_bodyface"] = detected_map_bodyface

            if self.use_hand and self.use_body and self.use_face:
                detected_map_handbodyface = draw_pose(
                    pose, H, W, use_hand=True, use_body=True, use_face=True
                )
                detected_map_handbodyface = cv2.resize(
                    detected_map_handbodyface[..., ::-1],
                    (ori_w, ori_h),
                    interpolation=(
                        cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA
                    ),
                )
                ret_data["detected_map_handbodyface"] = detected_map_handbodyface

            # convert_size
            if det_result.shape[0] > 0:
                w_ratio, h_ratio = ori_w / W, ori_h / H
                det_result[..., ::2] *= h_ratio
                det_result[..., 1::2] *= w_ratio
                det_result = det_result.astype(np.int32)

            return PoseOutput(**ret_data, det_result=det_result)

    def __str__(self):
        return f"PosePreprocessor(use_body={self.use_body}, use_face={self.use_face}, use_hand={self.use_hand})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("pose.bodyface")
class PoseBodyFacePreprocessor(PosePreprocessor, BasePreprocessor):
    def __init__(
        self,
        detection_model: Optional[str] = None,
        pose_model: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        resize_size: int = 1024,
        **kwargs,
    ):
        super().__init__(
            detection_model,
            pose_model,
            device,
            resize_size,
            use_body=True,
            use_face=True,
            use_hand=False,
            **kwargs,
        )

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, image: Union[Image.Image, np.ndarray, str]):
        ret_data = super().__call__(image)
        return PoseBodyFaceOutput(detected_map_bodyface=ret_data.detected_map_bodyface)

    def __str__(self):
        return "PoseBodyFacePreprocessor(use_body=True, use_face=True, use_hand=False)"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("pose.bodyface.video")
class PoseBodyFaceVideoPreprocessor(PoseBodyFacePreprocessor, BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor_type = PreprocessorType.VIDEO

    def __call__(self, frames: Union[List[Image.Image], List[str], str]):
        frames = self._load_video(frames)
        ret_frames = []
        for frame in tqdm(frames):
            anno_frame = super().__call__(frame)
            ret_frames.append(anno_frame)
        return PoseBodyFaceVideoOutput(frames=ret_frames)

    def __str__(self):
        return "PoseBodyFaceVideoPreprocessor(use_body=True, use_face=True, use_hand=False)"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("pose.body")
class PoseBodyPreprocessor(PosePreprocessor, BasePreprocessor):
    def __init__(
        self,
        detection_model: Optional[str] = None,
        pose_model: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        resize_size: int = 1024,
        **kwargs,
    ):
        super().__init__(
            detection_model,
            pose_model,
            device,
            resize_size,
            use_body=True,
            use_face=False,
            use_hand=False,
            **kwargs,
        )

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, image: Union[Image.Image, np.ndarray, str]):
        ret_data = super().__call__(image)
        return PoseBodyOutput(detected_map_body=ret_data.detected_map_body)

    def __str__(self):
        return "PoseBodyPreprocessor(use_body=True, use_face=False, use_hand=False)"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("pose.body.video")
class PoseBodyVideoPreprocessor(PoseBodyPreprocessor, BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor_type = PreprocessorType.VIDEO

    def __call__(self, frames: Union[List[Image.Image], List[str], str]):
        frames = self._load_video(frames)
        ret_frames = []
        for frame in tqdm(frames):
            anno_frame = super().__call__(frame)
            ret_frames.append(anno_frame)
        return PoseBodyVideoOutput(frames=ret_frames)

    def __str__(self):
        return (
            "PoseBodyVideoPreprocessor(use_body=True, use_face=False, use_hand=False)"
        )

    def __repr__(self):
        return self.__str__()
