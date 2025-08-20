# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from typing import Union, List, Optional, Dict, Any
from PIL import Image
from src.preprocess.subject import SubjectPreprocessor, subject_mode
from collections import defaultdict
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
    BaseOutput,
)
from tqdm import tqdm


class CompositionOutput(BaseOutput):
    video: List[Image.Image]
    mask: List[Image.Image] | None = None


class ReferenceAnythingOutput(BaseOutput):
    images: List[Image.Image]
    masks: List[Image.Image] | None = None


class AnimateAnythingOutput(BaseOutput):
    frames: List[Image.Image]
    images: List[Image.Image] | None = None


class SwapAnythingOutput(BaseOutput):
    video: List[Image.Image]
    mask: List[Image.Image] | None = None
    images: List[Image.Image] | None = None


class ExpandAnythingOutput(BaseOutput):
    frames: List[Image.Image]
    masks: List[Image.Image] | None = None
    images: List[Image.Image] | None = None


class MoveAnythingOutput(BaseOutput):
    frames: List[Image.Image]
    masks: List[Image.Image] | None = None


@preprocessor_registry("composition")
class CompositionPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)

        self.process_types = ["repaint", "extension", "control"]
        self.process_map = {
            "repaint": "repaint",
            "extension": "extension",
            "control": "control",
            "inpainting": "repaint",
            "outpainting": "repaint",
            "frameref": "extension",
            "clipref": "extension",
            "depth": "control",
            "flow": "control",
            "gray": "control",
            "pose": "control",
            "scribble": "control",
            "layout": "control",
        }

    def __call__(
        self,
        process_type_1: str,
        process_type_2: str,
        video_1: Union[List[Union[Image.Image, np.ndarray]], str],
        video_2: Union[List[Union[Image.Image, np.ndarray]], str],
        mask_1: Union[List[Union[Image.Image, np.ndarray]], str],
        mask_2: Union[List[Union[Image.Image, np.ndarray]], str],
        height: Optional[int] = None,
        width: Optional[int] = None,
        resample: Optional[Image.Resampling] = Image.Resampling.BICUBIC,
    ) -> CompositionOutput:

        # Convert inputs to numpy arrays
        frames_1 = self._load_video(video_1)
        frames_2 = self._load_video(video_2)
        masks_1 = self._load_video(mask_1, convert_method=lambda x: x.convert("L"))
        masks_2 = self._load_video(mask_2, convert_method=lambda x: x.convert("L"))

        if height is None or width is None:
            height = frames_1[0].size[1]
            width = frames_1[0].size[0]

        frames_1 = [
            frame.resize((width, height), resample=resample) for frame in frames_1
        ]
        frames_2 = [
            frame.resize((width, height), resample=resample) for frame in frames_2
        ]
        masks_1 = [mask.resize((width, height), resample=resample) for mask in masks_1]
        masks_2 = [mask.resize((width, height), resample=resample) for mask in masks_2]

        # Convert masks to normalized float arrays [0, 1]
        masks_1_norm = []
        masks_2_norm = []

        for mask in masks_1:
            if not isinstance(mask, np.ndarray):
                mask = np.array(self._load_image(mask))
            # Convert to grayscale if needed and normalize to [0, 1]
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2)
            mask = mask.astype(np.float32) / 255.0
            masks_1_norm.append(mask)

        for mask in masks_2:
            if not isinstance(mask, np.ndarray):
                mask = np.array(self._load_image(mask))
            # Convert to grayscale if needed and normalize to [0, 1]
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2)
            mask = mask.astype(np.float32) / 255.0
            masks_2_norm.append(mask)

        total_frames = min(
            len(frames_1), len(frames_2), len(masks_1_norm), len(masks_2_norm)
        )

        combine_type = (
            self.process_map[process_type_1],
            self.process_map[process_type_2],
        )

        output_video = []
        output_mask = []

        if combine_type in [
            ("extension", "repaint"),
            ("extension", "control"),
            ("extension", "extension"),
        ]:
            for i in range(total_frames):
                # Apply mask_1 to blend frames
                mask_1_expanded = masks_1_norm[i][
                    ..., np.newaxis
                ]  # Add channel dimension
                frame_out = frames_2[i] * mask_1_expanded + frames_1[i] * (
                    1 - mask_1_expanded
                )
                mask_out = masks_1_norm[i] * masks_2_norm[i] * 255
                output_video.append(frame_out.astype(np.uint8))
                output_mask.append(mask_out.astype(np.uint8))

        elif combine_type in [
            ("repaint", "extension"),
            ("control", "extension"),
            ("repaint", "repaint"),
        ]:
            for i in range(total_frames):
                # Apply mask_2 to blend frames
                mask_2_expanded = masks_2_norm[i][
                    ..., np.newaxis
                ]  # Add channel dimension
                frame_out = (
                    frames_1[i] * (1 - mask_2_expanded) + frames_2[i] * mask_2_expanded
                )
                mask_out = (
                    masks_1_norm[i] * (1 - masks_2_norm[i])
                    + masks_2_norm[i] * masks_2_norm[i]
                ) * 255
                output_video.append(frame_out.astype(np.uint8))
                output_mask.append(mask_out.astype(np.uint8))

        elif combine_type in [("repaint", "control"), ("control", "repaint")]:
            if combine_type == ("control", "repaint"):
                frames_1, frames_2, masks_1_norm, masks_2_norm = (
                    frames_2,
                    frames_1,
                    masks_2_norm,
                    masks_1_norm,
                )
            for i in range(total_frames):
                # Apply mask_1 to blend frames
                mask_1_expanded = masks_1_norm[i][
                    ..., np.newaxis
                ]  # Add channel dimension
                frame_out = (
                    frames_1[i] * (1 - mask_1_expanded) + frames_2[i] * mask_1_expanded
                )
                mask_out = masks_1_norm[i] * 255
                output_video.append(frame_out.astype(np.uint8))
                output_mask.append(mask_out.astype(np.uint8))

        elif combine_type == ("control", "control"):  # apply masks_2
            for i in range(total_frames):
                # Apply mask_2 to blend frames
                mask_2_expanded = masks_2_norm[i][
                    ..., np.newaxis
                ]  # Add channel dimension
                frame_out = (
                    frames_1[i] * (1 - mask_2_expanded) + frames_2[i] * mask_2_expanded
                )
                mask_out = (
                    masks_1_norm[i] * (1 - masks_2_norm[i])
                    + masks_2_norm[i] * masks_2_norm[i]
                ) * 255
                output_video.append(frame_out.astype(np.uint8))
                output_mask.append(mask_out.astype(np.uint8))
        else:
            raise ValueError(f"Unknown combine type: {combine_type}")

        return CompositionOutput(
            video=[Image.fromarray(frame) for frame in output_video],
            mask=[Image.fromarray(mask) for mask in output_mask],
        )

    def __str__(self):
        return "CompositionPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.reference_anything")
class ReferenceAnythingPreprocessor(BasePreprocessor):
    def __init__(self, subject_config: Dict = None, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)

        self.sbjref_ins = SubjectPreprocessor(
            **(
                subject_config
                if subject_config is not None
                else {
                    "use_aug": True,
                    "use_crop": True,
                    "roi_only": True,
                }
            )
        )
        self.key_map = {"image": "images", "mask": "masks"}

    def __call__(
        self,
        images: Union[List[Union[Image.Image, np.ndarray, str]], str],
        mask: Optional[Union[List[Union[Image.Image, np.ndarray, str]], str]] = None,
        mode: Optional[subject_mode] = "salient",
        return_mask: Optional[bool] = None,
        mask_cfg: Optional[Dict] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        bbox: Optional[List[float | int] | List[List[float | int]]] = None,
    ):

        images = self._load_video(images)
        ret_data = defaultdict(list)
        if mask is not None:
            masks = self._load_video(mask, convert_method=lambda x: x.convert("L"))
            if len(masks) > len(images):
                masks = masks[: len(images)]
            elif len(masks) < len(images):
                masks = (
                    masks * (len(images) // len(masks))
                    + masks[: len(images) % len(masks)]
                )
        else:
            masks = None

        if isinstance(bbox, list) and isinstance(bbox[0], (float, int)):
            bbox = [bbox] * len(images)
        elif isinstance(bbox, list) and isinstance(bbox[0], list):
            if len(bbox) > len(images):
                bbox = bbox[: len(images)]
            elif len(bbox) < len(images):
                bbox = (
                    bbox * (len(images) // len(bbox)) + bbox[: len(images) % len(bbox)]
                )

        for i, image in tqdm(
            enumerate(images), total=len(images), desc="Processing images"
        ):
            ret_one_data = self.sbjref_ins(
                image=image,
                mode=mode,
                return_mask=return_mask,
                mask_cfg=mask_cfg,
                caption=caption,
                label=label,
                bbox=bbox[i] if bbox is not None else None,
                mask=masks[i] if masks is not None else None,
            )

            ret_data["images"].append(ret_one_data.image)
            if return_mask:
                ret_data["masks"].append(ret_one_data.mask)

        return ReferenceAnythingOutput(
            images=[Image.fromarray(image) for image in ret_data["images"]],
            mask=(
                [Image.fromarray(mask) for mask in ret_data["masks"]]
                if ret_data["masks"] is not None
                else None
            ),
        )

    def __str__(self):
        return "ReferenceAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.animate_anything")
class AnimateAnythingPreprocessor(BasePreprocessor):
    def __init__(
        self, pose_config: Dict = None, reference_config: Dict = None, **kwargs
    ):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)

        from src.preprocess.pose import PoseBodyFaceVideoPreprocessor

        self.pose_ins = PoseBodyFaceVideoPreprocessor(
            **(pose_config if pose_config is not None else {})
        )
        self.ref_ins = ReferenceAnythingPreprocessor(
            **(
                reference_config
                if reference_config is not None
                else {
                    "subject_config": {
                        "use_aug": True,
                        "use_crop": True,
                        "roi_only": True,
                    }
                }
            )
        )

    def __call__(
        self,
        frames: Optional[Union[List[Image.Image], List[str], str]] = None,
        images: Optional[List[Union[Image.Image, np.ndarray, str]]] = None,
        ref_mode: Optional[subject_mode] = "salient",
        return_mask: Optional[bool] = None,
        mask_cfg: Optional[Dict] = None,
    ) -> AnimateAnythingOutput:

        frames = self._load_video(frames)

        ret_data = defaultdict(list)

        # Process pose from frames
        ret_pose_data = self.pose_ins(frames=frames)
        ret_data.update(
            {
                "frames": [
                    Image.fromarray(frame.detected_map_bodyface)
                    for frame in ret_pose_data.frames
                ]
            }
        )

        # Process reference images
        if images is not None:
            images = [self._load_image(image) for image in images]
            ret_ref_data = self.ref_ins(
                images=images, mode=ref_mode, return_mask=return_mask, mask_cfg=mask_cfg
            )
            ret_data.update({"images": ret_ref_data.video})

        return AnimateAnythingOutput(
            frames=ret_data["frames"], images=ret_data["images"]
        )

    def __str__(self):
        return "AnimateAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.swap_anything")
class SwapAnythingPreprocessor(BasePreprocessor):
    def __init__(
        self,
        inpainting_config: Optional[Dict] = None,
        reference_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)

        from src.preprocess.inpainting import InpaintingVideoPreprocessor

        self.inp_ins = InpaintingVideoPreprocessor(
            **(inpainting_config if inpainting_config is not None else {})
        )
        self.ref_ins = ReferenceAnythingPreprocessor(
            **(
                reference_config
                if reference_config is not None
                else {
                    "subject_config": {
                        "use_aug": True,
                        "use_crop": True,
                        "roi_only": True,
                    }
                }
            )
        )

    def __call__(
        self,
        frames: Optional[Union[List[Image.Image], List[str], str]] = None,
        images: Optional[List[Union[Image.Image, np.ndarray, str]]] = None,
        mode: Optional[str] = None,
        mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
        bbox: Optional[List[float]] = None,
        label: Optional[str] = None,
        caption: Optional[str] = None,
        return_mask: Optional[bool] = None,
        mask_cfg: Optional[Dict] = None,
    ) -> SwapAnythingOutput:

        frames, fps = self._load_video(frames, return_fps=True)

        ret_data = {}

        # Split mode if comma-separated
        if mode and "," in mode:
            mode_list = mode.split(",")
        else:
            mode_list = [mode, mode] if mode else ["mask", "mask"]

        # Process inpainting
        ret_inp_data = self.inp_ins(
            frames=frames,
            mode=mode_list[0],
            mask=mask,
            bbox=bbox,
            label=label,
            caption=caption,
            mask_cfg=mask_cfg,
            fps=fps,
        )
        ret_data.update(
            {
                "video": [Image.fromarray(frame) for frame in ret_inp_data.frames],
                "mask": [
                    Image.fromarray(mask).convert("L") for mask in ret_inp_data.masks
                ],
            }
        )

        # Process reference images
        if images is not None:
            images = [self._load_image(image) for image in images]
            ret_ref_data = self.ref_ins(
                frames=images,
                mask=mask,
                bbox=bbox,
                label=label,
                caption=caption,
                mode=mode_list[1],
                return_mask=return_mask,
                mask_cfg=mask_cfg,
            )
            ret_data.update({"images": ret_ref_data.video})

        return SwapAnythingOutput(
            video=ret_data["video"], mask=ret_data["mask"], images=ret_data["images"]
        )

    def __str__(self):
        return "SwapAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.expand_anything")
class ExpandAnythingPreprocessor(BasePreprocessor):
    def __init__(
        self,
        reference_config: Optional[Dict] = None,
        frameref_config: Optional[Dict] = None,
        outpainting_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)

        from src.preprocess.outpainting import OutpaintingPreprocessor
        from src.preprocess.frameref import FrameRefExpandPreprocessor

        self.ref_ins = ReferenceAnythingPreprocessor(
            **(
                reference_config
                if reference_config is not None
                else {
                    "subject_config": {
                        "use_aug": True,
                        "use_crop": True,
                        "roi_only": True,
                    }
                }
            )
        )
        self.frameref_ins = FrameRefExpandPreprocessor(
            **(frameref_config if frameref_config is not None else {})
        )
        self.outpainting_ins = OutpaintingPreprocessor(
            **(
                outpainting_config
                if outpainting_config is not None
                else {
                    "keep_padding_ratio": 1,
                    "mask_color": 1,
                    "return_mask": True,
                }
            )
        )

    def __call__(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        mode: Optional[str] = None,
        return_mask: Optional[bool] = None,
        mask_cfg: Optional[Dict] = None,
        direction: Optional[str] = None,
        expand_ratio: float = 0.3,
        expand_num: int = 24,
    ) -> ExpandAnythingOutput:

        ret_data = {}
        expand_image, reference_image = images[0], images[1:]
        expand_image = self._load_image(expand_image)
        reference_image = [self._load_image(image) for image in reference_image]

        # Split mode if comma-separated
        if mode and "," in mode:
            mode_list = mode.split(",")
        else:
            mode_list = ["firstframe", mode] if mode else ["firstframe", "mask"]

        # Process outpainting
        outpainting_data = self.outpainting_ins(
            expand_image,
            expand_ratio=expand_ratio,
            direction=direction,
            return_mask=True,
        )

        outpainting_image, outpainting_mask = (
            outpainting_data.image,
            outpainting_data.mask,
        )

        # Process frame reference expansion
        frameref_data = self.frameref_ins(
            outpainting_image,
            mode=mode_list[0],
            expand_num=expand_num,
            return_mask=return_mask,
        )

        frames, masks = frameref_data.frames, frameref_data.masks
        masks[0] = np.array(outpainting_mask)
        ret_data.update({"frames": frames, "masks": masks})

        ret_ref_data = self.ref_ins(
            frames=reference_image,
            mode=mode_list[1],
            return_mask=return_mask,
            mask_cfg=mask_cfg,
        )
        ret_data.update({"images": ret_ref_data.video})

        return ExpandAnythingOutput(
            frames=[
                Image.fromarray(frame.astype(np.uint8)) for frame in ret_data["frames"]
            ],
            masks=(
                [Image.fromarray(mask.astype(np.uint8)) for mask in ret_data["masks"]]
                if ret_data["masks"] is not None
                else None
            ),
            images=ret_data["images"],
        )

    def __str__(self):
        return "ExpandAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.move_anything")
class MoveAnythingPreprocessor(BasePreprocessor):
    def __init__(self, layout_bbox_config: Dict = None, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)

        from src.preprocess.layout import LayoutBboxPreprocessor

        self.layout_bbox_ins = LayoutBboxPreprocessor(
            **(layout_bbox_config if layout_bbox_config is not None else {})
        )

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str],
        start_bbox: List[float],
        end_bbox: List[float],
        label: Optional[str] = None,
        expand_num: Optional[int] = None,
    ) -> MoveAnythingOutput:

        # Load and convert image
        image = self._load_image(image)
        image_array = np.array(image)
        frame_size = image_array.shape[:2]  # [H, W]

        # Process layout bbox
        ret_layout_data = self.layout_bbox_ins(
            [start_bbox, end_bbox],
            frame_size=frame_size,
            num_frames=expand_num,
            label=label,
        ).frames

        # Create output frames and masks
        out_frames = [image_array] + ret_layout_data
        out_mask = [np.zeros(frame_size, dtype=np.uint8)] + [
            np.ones(frame_size, dtype=np.uint8) * 255
        ] * len(ret_layout_data)

        return MoveAnythingOutput(
            frames=[Image.fromarray(frame.astype(np.uint8)) for frame in out_frames],
            masks=[Image.fromarray(mask.astype(np.uint8)) for mask in out_mask],
        )

    def __str__(self):
        return "MoveAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__()
