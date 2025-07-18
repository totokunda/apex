# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
from PIL import Image
import warnings

from src.preprocess.base import BasePreprocessor, preprocessor_registry, PreprocessorType


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
            "layout": "control"
        }

    def __call__(self, 
                process_type_1: str, 
                process_type_2: str, 
                frames_1: List[Union[Image.Image, np.ndarray]], 
                frames_2: List[Union[Image.Image, np.ndarray]], 
                masks_1: List[Union[Image.Image, np.ndarray]], 
                masks_2: List[Union[Image.Image, np.ndarray]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        # Convert inputs to numpy arrays
        frames_1 = self._load_video(frames_1)
        frames_2 = self._load_video(frames_2)
        
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
        
        total_frames = min(len(frames_1), len(frames_2), len(masks_1_norm), len(masks_2_norm))
        combine_type = (self.process_map[process_type_1], self.process_map[process_type_2])
        
        output_video = []
        output_mask = []
        
        if combine_type in [("extension", "repaint"), ("extension", "control"), ("extension", "extension")]:
            for i in range(total_frames):
                # Apply mask_1 to blend frames
                mask_1_expanded = masks_1_norm[i][..., np.newaxis]  # Add channel dimension
                frame_out = frames_2[i] * mask_1_expanded + frames_1[i] * (1 - mask_1_expanded)
                mask_out = masks_1_norm[i] * masks_2_norm[i] * 255
                output_video.append(frame_out.astype(np.uint8))
                output_mask.append(mask_out.astype(np.uint8))
                
        elif combine_type in [("repaint", "extension"), ("control", "extension"), ("repaint", "repaint")]:
            for i in range(total_frames):
                # Apply mask_2 to blend frames
                mask_2_expanded = masks_2_norm[i][..., np.newaxis]  # Add channel dimension
                frame_out = frames_1[i] * (1 - mask_2_expanded) + frames_2[i] * mask_2_expanded
                mask_out = (masks_1_norm[i] * (1 - masks_2_norm[i]) + masks_2_norm[i] * masks_2_norm[i]) * 255
                output_video.append(frame_out.astype(np.uint8))
                output_mask.append(mask_out.astype(np.uint8))
                
        elif combine_type in [("repaint", "control"), ("control", "repaint")]:
            if combine_type == ("control", "repaint"):
                frames_1, frames_2, masks_1_norm, masks_2_norm = frames_2, frames_1, masks_2_norm, masks_1_norm
            for i in range(total_frames):
                # Apply mask_1 to blend frames
                mask_1_expanded = masks_1_norm[i][..., np.newaxis]  # Add channel dimension
                frame_out = frames_1[i] * (1 - mask_1_expanded) + frames_2[i] * mask_1_expanded
                mask_out = masks_1_norm[i] * 255
                output_video.append(frame_out.astype(np.uint8))
                output_mask.append(mask_out.astype(np.uint8))
                
        elif combine_type == ("control", "control"):  # apply masks_2
            for i in range(total_frames):
                # Apply mask_2 to blend frames
                mask_2_expanded = masks_2_norm[i][..., np.newaxis]  # Add channel dimension
                frame_out = frames_1[i] * (1 - mask_2_expanded) + frames_2[i] * mask_2_expanded
                mask_out = (masks_1_norm[i] * (1 - masks_2_norm[i]) + masks_2_norm[i] * masks_2_norm[i]) * 255
                output_video.append(frame_out.astype(np.uint8))
                output_mask.append(mask_out.astype(np.uint8))
        else:
            raise ValueError(f"Unknown combine type: {combine_type}")
        
        return output_video, output_mask

    def __str__(self):
        return "CompositionPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.reference_anything")
class ReferenceAnythingPreprocessor(BasePreprocessor):
    def __init__(self, subject_config: Dict, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        
        from src.preprocess.subject import SubjectPreprocessor
        self.sbjref_ins = SubjectPreprocessor(**subject_config)
        self.key_map = {
            "image": "images",
            "mask": "masks"
        }

    def __call__(self, 
                images: List[Union[Image.Image, np.ndarray, str]] | str, 
                mode: Optional[str] = None, 
                return_mask: Optional[bool] = None, 
                mask_cfg: Optional[Dict] = None) -> Dict[str, List]:
        
        images = self._load_video(images)
        ret_data = {}
        for image in images:
            ret_one_data = self.sbjref_ins(image=image, mode=mode, return_mask=return_mask, mask_cfg=mask_cfg)
            if isinstance(ret_one_data, dict):
                for key, val in ret_one_data.items():
                    if key in self.key_map:
                        new_key = self.key_map[key]
                    else:
                        continue
                    if new_key in ret_data:
                        ret_data[new_key].append(val)
                    else:
                        ret_data[new_key] = [val]
            else:
                if 'images' in ret_data:
                    ret_data['images'].append(ret_one_data)
                else:
                    ret_data['images'] = [ret_one_data]
        return ret_data

    def __str__(self):
        return "ReferenceAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.animate_anything")
class AnimateAnythingPreprocessor(BasePreprocessor):
    def __init__(self, pose_config: Dict, reference_config: Dict, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)
        
        from src.preprocess.pose import PoseBodyFaceVideoPreprocessor
        self.pose_ins = PoseBodyFaceVideoPreprocessor(**pose_config)
        self.ref_ins = ReferenceAnythingPreprocessor(**reference_config)

    def __call__(self, 
                frames: Optional[Union[List[Image.Image], List[str], str]] = None, 
                images: Optional[List[Union[Image.Image, np.ndarray, str]]] = None, 
                mode: Optional[str] = None, 
                return_mask: Optional[bool] = None, 
                mask_cfg: Optional[Dict] = None) -> Dict[str, Any]:
        
        frames = self._load_video(frames)
        
        ret_data = {}
        
        # Process pose from frames
        ret_pose_data = self.pose_ins(frames=frames)
        ret_data.update({"frames": ret_pose_data})

        # Process reference images
        if images is not None:
            images = self._load_video(images)
            ret_ref_data = self.ref_ins(images=images, mode=mode, return_mask=return_mask, mask_cfg=mask_cfg)
            ret_data.update({"images": ret_ref_data['images']})

        return ret_data

    def __str__(self):
        return "AnimateAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.swap_anything")
class SwapAnythingPreprocessor(BasePreprocessor):
    def __init__(self, inpainting_config: Dict, reference_config: Dict, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)
        
        from src.preprocess.inpainting import InpaintingVideoPreprocessor
        self.inp_ins = InpaintingVideoPreprocessor(**inpainting_config)
        self.ref_ins = ReferenceAnythingPreprocessor(**reference_config)

    def __call__(self, 
                video: Optional[str] = None, 
                frames: Optional[Union[List[Image.Image], List[str], str]] = None, 
                images: Optional[List[Union[Image.Image, np.ndarray, str]]] = None, 
                mode: Optional[str] = None, 
                mask: Optional[Union[Image.Image, np.ndarray, str]] = None, 
                bbox: Optional[List[float]] = None, 
                label: Optional[str] = None, 
                caption: Optional[str] = None, 
                return_mask: Optional[bool] = None, 
                mask_cfg: Optional[Dict] = None) -> Dict[str, Any]:
        
        frames = self._load_video(frames)
        
        ret_data = {}
        
        # Split mode if comma-separated
        if mode and ',' in mode:
            mode_list = mode.split(',')
        else:
            mode_list = [mode, mode] if mode else ["mask", "mask"]

        # Process inpainting
        ret_inp_data = self.inp_ins(video=video, frames=frames, mode=mode_list[0], 
                                   mask=mask, bbox=bbox, label=label, caption=caption, mask_cfg=mask_cfg)
        ret_data.update(ret_inp_data)

        # Process reference images
        if images is not None:
            images = self._load_video(images)
            ret_ref_data = self.ref_ins(images=images, mode=mode_list[1], return_mask=return_mask, mask_cfg=mask_cfg)
            ret_data.update({"images": ret_ref_data['images']})

        return ret_data

    def __str__(self):
        return "SwapAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.expand_anything")
class ExpandAnythingPreprocessor(BasePreprocessor):
    def __init__(self, reference_config: Dict, frameref_config: Dict, outpainting_config: Dict, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        
        from src.preprocess.outpainting import OutpaintingPreprocessor
        from src.preprocess.frameref import FrameRefExpandPreprocessor
        
        self.ref_ins = ReferenceAnythingPreprocessor(**reference_config)
        self.frameref_ins = FrameRefExpandPreprocessor(**frameref_config)
        self.outpainting_ins = OutpaintingPreprocessor(**outpainting_config)

    def __call__(self, 
                images: List[Union[Image.Image, np.ndarray, str]], 
                mode: Optional[str] = None, 
                return_mask: Optional[bool] = None, 
                mask_cfg: Optional[Dict] = None, 
                direction: Optional[str] = None, 
                expand_ratio: Optional[float] = None, 
                expand_num: Optional[int] = None) -> Dict[str, Any]:
        
        ret_data = {}
        expand_image, reference_image = images[0], images[1:]
        
        # Split mode if comma-separated
        if mode and ',' in mode:
            mode_list = mode.split(',')
        else:
            mode_list = ['firstframe', mode] if mode else ['firstframe', 'mask']

        # Process outpainting
        outpainting_data = self.outpainting_ins(expand_image, expand_ratio=expand_ratio, direction=direction)
        outpainting_image, outpainting_mask = outpainting_data['image'], outpainting_data['mask']

        # Process frame reference expansion
        frameref_data = self.frameref_ins(outpainting_image, mode=mode_list[0], expand_num=expand_num)
        frames, masks = frameref_data['frames'], frameref_data['masks']
        masks[0] = outpainting_mask
        ret_data.update({"frames": frames, "masks": masks})

        # Process reference images
        if reference_image:
            ret_ref_data = self.ref_ins(images=reference_image, mode=mode_list[1], return_mask=return_mask, mask_cfg=mask_cfg)
            ret_data.update({"images": ret_ref_data['images']})

        return ret_data

    def __str__(self):
        return "ExpandAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("composition.move_anything")
class MoveAnythingPreprocessor(BasePreprocessor):
    def __init__(self, layout_bbox_config: Dict, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)
        
        from src.preprocess.layout import LayoutBboxPreprocessor
        self.layout_bbox_ins = LayoutBboxPreprocessor(**layout_bbox_config)

    def __call__(self, 
                image: Union[Image.Image, np.ndarray, str], 
                bbox: List[float], 
                label: Optional[str] = None, 
                expand_num: Optional[int] = None) -> Dict[str, Any]:
        
        # Load and convert image
        image = self._load_image(image)
        image_array = np.array(image)
        frame_size = image_array.shape[:2]   # [H, W]
        
        # Process layout bbox
        ret_layout_data = self.layout_bbox_ins(bbox, frame_size=frame_size, num_frames=expand_num, label=label)

        # Create output frames and masks
        out_frames = [image_array] + ret_layout_data
        out_mask = [np.zeros(frame_size, dtype=np.uint8)] + [np.ones(frame_size, dtype=np.uint8) * 255] * len(ret_layout_data)

        ret_data = {
            "frames": out_frames,
            "masks": out_mask
        }
        return ret_data

    def __str__(self):
        return "MoveAnythingPreprocessor()"

    def __repr__(self):
        return self.__str__() 