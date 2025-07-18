# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
from src.preprocess.base import BasePreprocessor, preprocessor_registry, PreprocessorType
from typing import Union, List, Optional, Tuple
from PIL import Image
import warnings

@preprocessor_registry("layout.bbox")
class LayoutBboxPreprocessor(BasePreprocessor):
    def __init__(self, bg_color=None, box_color=None, frame_size=None, num_frames=81, ram_tag_color_path=None, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)
        self.bg_color = bg_color if bg_color is not None else [255, 255, 255]
        self.box_color = box_color if box_color is not None else [0, 0, 0]
        self.frame_size = frame_size if frame_size is not None else [720, 1280]  # [H, W]
        self.num_frames = num_frames
        self.color_dict = {'default': tuple(self.box_color)}
        
        if ram_tag_color_path is not None:
            try:
                lines = [id_name_color.strip().split('#;#') for id_name_color in open(ram_tag_color_path).readlines()]
                self.color_dict.update({id_name_color[1]: tuple(eval(id_name_color[2])) for id_name_color in lines})
            except Exception as e:
                warnings.warn(f"Could not load color mappings from {ram_tag_color_path}: {e}")

    def __call__(self, bbox: List[List[float]], frame_size: Optional[List[int]] = None, num_frames: Optional[int] = None, 
                 label: Optional[Union[str, List[str]]] = None, color: Optional[Tuple[int, int, int]] = None):
        frame_size = frame_size if frame_size is not None else self.frame_size
        num_frames = num_frames if num_frames is not None else self.num_frames
        assert len(bbox) == 2, 'bbox should be a list of two elements (start_bbox & end_bbox)'
        
        # frame_size = [H, W]
        # bbox = [x1, y1, x2, y2]
        label = label[0] if label is not None and isinstance(label, list) else label
        if label is not None and label in self.color_dict:
            box_color = self.color_dict[label]
        elif color is not None:
            box_color = color
        else:
            box_color = self.color_dict['default']
        
        start_bbox, end_bbox = bbox
        start_bbox = [start_bbox[0], start_bbox[1], start_bbox[2] - start_bbox[0], start_bbox[3] - start_bbox[1]]
        start_bbox = np.array(start_bbox, dtype=np.float32)
        end_bbox = [end_bbox[0], end_bbox[1], end_bbox[2] - end_bbox[0], end_bbox[3] - end_bbox[1]]
        end_bbox = np.array(end_bbox, dtype=np.float32)
        bbox_increment = (end_bbox - start_bbox) / num_frames
        
        ret_frames = []
        for frame_idx in range(num_frames):
            frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
            frame[:] = self.bg_color
            current_bbox = start_bbox + bbox_increment * frame_idx
            current_bbox = current_bbox.astype(int)
            x, y, w, h = current_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            ret_frames.append(frame[..., ::-1])  # Convert BGR to RGB
        return ret_frames

    def __str__(self):
        return f"LayoutBboxPreprocessor(frame_size={self.frame_size}, num_frames={self.num_frames})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("layout.mask")
class LayoutMaskPreprocessor(BasePreprocessor):
    def __init__(self, use_aug=False, bg_color=None, box_color=None, ram_tag_color_path=None, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        self.use_aug = use_aug
        self.bg_color = bg_color if bg_color is not None else [255, 255, 255]
        self.box_color = box_color if box_color is not None else [0, 0, 0]
        self.color_dict = {'default': tuple(self.box_color)}
        
        if ram_tag_color_path is not None:
            try:
                lines = [id_name_color.strip().split('#;#') for id_name_color in open(ram_tag_color_path).readlines()]
                self.color_dict.update({id_name_color[1]: tuple(eval(id_name_color[2])) for id_name_color in lines})
            except Exception as e:
                warnings.warn(f"Could not load color mappings from {ram_tag_color_path}: {e}")
        
        if self.use_aug:
            warnings.warn("Mask augmentation is not available. Proceeding without augmentation.")
            self.use_aug = False

    def find_contours(self, mask):
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, canvas, contour, color):
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        canvas = cv2.drawContours(canvas, contour, -1, color, thickness=3)
        return canvas

    def __call__(self, mask: Union[Image.Image, np.ndarray, str, List], color: Optional[Tuple[int, int, int]] = None, 
                 label: Optional[str] = None, mask_cfg: Optional[dict] = None):
        if not isinstance(mask, list):
            is_batch = False
            mask = [mask]
        else:
            is_batch = True

        if label is not None and label in self.color_dict:
            color = self.color_dict[label]
        elif color is not None:
            color = color
        else:
            color = self.color_dict['default']

        ret_data = []
        for sub_mask in mask:
            # Convert mask to numpy array
            if isinstance(sub_mask, (str, Image.Image)):
                sub_mask = self._load_image(sub_mask)
                sub_mask = np.array(sub_mask)
            elif not isinstance(sub_mask, np.ndarray):
                sub_mask = np.array(sub_mask)
            
            # Convert to grayscale if needed
            if len(sub_mask.shape) == 3:
                sub_mask = cv2.cvtColor(sub_mask, cv2.COLOR_RGB2GRAY)
            
            canvas = np.ones((sub_mask.shape[0], sub_mask.shape[1], 3)) * 255
            contour = self.find_contours(sub_mask)
            frame = self.draw_contours(canvas, contour, color)
            ret_data.append(frame)

        if is_batch:
            return ret_data
        else:
            return ret_data[0]

    def __str__(self):
        return f"LayoutMaskPreprocessor(use_aug={self.use_aug})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("layout.track")
class LayoutTrackPreprocessor(BasePreprocessor):
    def __init__(self, use_aug=False, bg_color=None, box_color=None, ram_tag_color_path=None, inpainting_config=None, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)
        self.use_aug = use_aug
        self.bg_color = bg_color if bg_color is not None else [255, 255, 255]
        self.box_color = box_color if box_color is not None else [0, 0, 0]
        self.color_dict = {'default': tuple(self.box_color)}
        
        if ram_tag_color_path is not None:
            try:
                lines = [id_name_color.strip().split('#;#') for id_name_color in open(ram_tag_color_path).readlines()]
                self.color_dict.update({id_name_color[1]: tuple(eval(id_name_color[2])) for id_name_color in lines})
            except Exception as e:
                warnings.warn(f"Could not load color mappings from {ram_tag_color_path}: {e}")
        
        if self.use_aug:
            warnings.warn("Mask augmentation is not available. Proceeding without augmentation.")
            self.use_aug = False
            
        # Note: InpaintingVideoAnnotator is not available, so we'll provide a simplified implementation
        warnings.warn("InpaintingVideoAnnotator is not available. Using simplified tracking.")

    def find_contours(self, mask):
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, canvas, contour, color):
        canvas = np.ascontiguousarray(canvas, dtype=np.uint8)
        canvas = cv2.drawContours(canvas, contour, -1, color, thickness=3)
        return canvas

    def __call__(self, frames: Union[List[Image.Image], List[str], str] = None, video: Optional[str] = None, 
                 mask: Optional[Union[Image.Image, np.ndarray, List]] = None, bbox: Optional[List] = None, 
                 label: Optional[str] = None, caption: Optional[str] = None, mode: Optional[str] = None, 
                 color: Optional[Tuple[int, int, int]] = None, mask_cfg: Optional[dict] = None):
        
        # Load frames
        if frames is not None:
            frames = self._load_video(frames)
        elif video is not None:
            frames = self._load_video(video)
        else:
            raise ValueError("Either frames or video must be provided")
        
        # Simplified tracking: if mask is provided, use it for all frames
        # In a full implementation, this would use the inpainting annotator
        if mask is not None:
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
                mask = np.array(mask)
            elif not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            # Resize mask to match frame size
            frame_height, frame_width = np.array(frames[0]).shape[:2]
            mask = cv2.resize(mask, (frame_width, frame_height))
            
            inp_masks = [mask] * len(frames)
        else:
            # If no mask provided, create a dummy mask
            frame_height, frame_width = np.array(frames[0]).shape[:2]
            dummy_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            inp_masks = [dummy_mask] * len(frames)

        label = label[0] if label is not None and isinstance(label, list) else label
        if label is not None and label in self.color_dict:
            color = self.color_dict[label]
        elif color is not None:
            color = color
        else:
            color = self.color_dict['default']

        num_frames = len(inp_masks)
        ret_data = []
        for i in range(num_frames):
            sub_mask = inp_masks[i]
            canvas = np.ones((sub_mask.shape[0], sub_mask.shape[1], 3)) * 255
            contour = self.find_contours(sub_mask)
            frame = self.draw_contours(canvas, contour, color)
            ret_data.append(frame)

        return ret_data

    def __str__(self):
        return f"LayoutTrackPreprocessor(use_aug={self.use_aug})"

    def __repr__(self):
        return self.__str__()


