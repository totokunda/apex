# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
from typing import Union, List, Optional, Dict, Any
from PIL import Image
import warnings

from src.preprocess.base import BasePreprocessor, preprocessor_registry, PreprocessorType


def single_rle_to_mask(rle):
    """Convert RLE format to mask - placeholder function"""
    try:
        import pycocotools.mask as mask_utils
        mask = np.array(mask_utils.decode(rle)).astype(np.uint8)
        return mask
    except ImportError:
        warnings.warn("pycocotools not available, returning dummy mask")
        return np.zeros((224, 224), dtype=np.uint8)


def single_mask_to_rle(mask):
    """Convert mask to RLE format - placeholder function"""
    try:
        import pycocotools.mask as mask_utils
        rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    except ImportError:
        warnings.warn("pycocotools not available, returning dummy RLE")
        return {"size": mask.shape, "counts": "placeholder"}


def get_mask_box(mask, threshold=255):
    """Get bounding box from mask"""
    locs = np.where(mask >= threshold)
    if len(locs) < 1 or locs[0].shape[0] < 1 or locs[1].shape[0] < 1:
        return None
    left, right = np.min(locs[1]), np.max(locs[1])
    top, bottom = np.min(locs[0]), np.max(locs[0])
    return [left, top, right, bottom]


def read_video_one_frame(video_path, use_type='cv2', is_rgb=True):
    """Read the first frame from a video"""
    image_first = None
    if use_type == "cv2":
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                if is_rgb:
                    image_first = frame[..., ::-1]
                else:
                    image_first = frame
            cap.release()
        except Exception as e:
            print(f"OpenCV read error: {e}")
            return None
    else:
        raise ValueError(f"Unknown video type {use_type}")
    return image_first


@preprocessor_registry("inpainting")
class InpaintingPreprocessor(BasePreprocessor):
    def __init__(self, mode: str = "mask", use_aug: bool = True, return_mask: bool = True, 
                 return_source: bool = True, mask_color: int = 128, 
                 salient_config: Optional[Dict] = None, sam2_config: Optional[Dict] = None, 
                 gdino_config: Optional[Dict] = None, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.IMAGE, **kwargs)
        
        self.mode = mode
        self.use_aug = use_aug
        self.return_mask = return_mask
        self.return_source = return_source
        self.mask_color = mask_color
        
        assert self.mode in ["salient", "mask", "bbox", "salientmasktrack", "salientbboxtrack", 
                           "maskpointtrack", "maskbboxtrack", "masktrack", "bboxtrack", 
                           "label", "caption", "all"]
        
        # Initialize dependent preprocessors based on mode
        self.salient_model = None
        self.sam2_model = None
        self.gdino_model = None
        self.maskaug_anno = None
        
        if self.mode in ["salient", "salienttrack", "salientmasktrack", "salientbboxtrack", "all"]:
            if salient_config:
                from src.preprocess.salient import SalientPreprocessor
                self.salient_model = SalientPreprocessor(**salient_config)
            else:
                warnings.warn("Salient mode selected but no salient_config provided")
        
        if self.mode in ['masktrack', 'bboxtrack', 'salienttrack', 'salientmasktrack', 
                        'salientbboxtrack', 'maskpointtrack', 'maskbboxtrack', 'label', 'caption', 'all']:
            if sam2_config:
                from src.preprocess.sam2 import SAM2Preprocessor
                self.sam2_model = SAM2Preprocessor(**sam2_config)
            else:
                warnings.warn("SAM2 mode selected but no sam2_config provided")
        
        if self.mode in ['label', 'caption', 'all']:
            if gdino_config:
                from src.preprocess.gdino import GDINOPreprocessor
                self.gdino_model = GDINOPreprocessor(**gdino_config)
            else:
                warnings.warn("GDINO mode selected but no gdino_config provided")
        
        if self.use_aug:
            warnings.warn("Mask augmentation functionality requires additional dependencies and is disabled.")
            self.use_aug = False

    def apply_plain_mask(self, image: np.ndarray, mask: np.ndarray, mask_color: int):
        """Apply a plain mask to an image"""
        bool_mask = mask > 0
        out_image = image.copy()
        out_image[bool_mask] = mask_color
        out_mask = np.where(bool_mask, 255, 0).astype(np.uint8)
        return out_image, out_mask
        
    def apply_seg_mask(self, image: np.ndarray, mask: np.ndarray, mask_color: int, mask_cfg: Optional[Dict] = None):
        """Apply a segmentation mask to an image"""
        out_mask = (mask * 255).astype('uint8')
        if self.use_aug and mask_cfg is not None:
            warnings.warn("Mask augmentation not implemented")
        bool_mask = out_mask > 0
        out_image = image.copy()
        out_image[bool_mask] = mask_color
        return out_image, out_mask

    def __call__(self, 
                image: Union[Image.Image, np.ndarray, str] = None,
                mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
                bbox: Optional[List[float]] = None,
                label: Optional[str] = None,
                caption: Optional[str] = None,
                mode: Optional[str] = None,
                return_mask: Optional[bool] = None,
                return_source: Optional[bool] = None,
                mask_color: Optional[int] = None,
                mask_cfg: Optional[Dict] = None):
        
        mode = mode if mode is not None else self.mode
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = return_source if return_source is not None else self.return_source
        mask_color = mask_color if mask_color is not None else self.mask_color

        # Load and convert image
        image = self._load_image(image)
        image_array = np.array(image)
        out_image, out_mask = None, None
        
        if mode == 'salient':
            if self.salient_model is None:
                raise ValueError("Salient model not initialized")
            mask_result = self.salient_model(image)
            out_image, out_mask = self.apply_plain_mask(image_array, mask_result, mask_color)
            
        elif mode == 'mask':
            if mask is None:
                raise ValueError("Mask is required for 'mask' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2).astype(np.uint8)
            
            mask_h, mask_w = mask.shape[:2]
            h, w = image_array.shape[:2]
            if (mask_h != h) or (mask_w != w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            out_image, out_mask = self.apply_plain_mask(image_array, mask, mask_color)
            
        elif mode == 'bbox':
            if bbox is None:
                raise ValueError("Bbox is required for 'bbox' mode")
            x1, y1, x2, y2 = bbox
            h, w = image_array.shape[:2]
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w, x2)), int(min(h, y2))
            out_image = image_array.copy()
            out_image[y1:y2, x1:x2] = mask_color
            out_mask = np.zeros((h, w), dtype=np.uint8)
            out_mask[y1:y2, x1:x2] = 255
            
        elif mode == 'salientmasktrack':
            if self.salient_model is None or self.sam2_model is None:
                raise ValueError("Salient and SAM2 models not initialized")
            mask_result = self.salient_model(image)
            resize_mask = cv2.resize(mask_result, (256, 256), interpolation=cv2.INTER_NEAREST)
            out_mask = self.sam2_model(image=image, mask=resize_mask, task_type='mask', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image_array, out_mask, mask_color, mask_cfg)
            
        elif mode == 'salientbboxtrack':
            if self.salient_model is None or self.sam2_model is None:
                raise ValueError("Salient and SAM2 models not initialized")
            mask_result = self.salient_model(image)
            bbox = get_mask_box(np.array(mask_result), threshold=1)
            if bbox is None:
                raise ValueError("Could not extract bounding box from salient mask")
            out_mask = self.sam2_model(image=image, input_box=bbox, task_type='input_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image_array, out_mask, mask_color, mask_cfg)
            
        elif mode == 'maskpointtrack':
            if self.sam2_model is None:
                raise ValueError("SAM2 model not initialized")
            if mask is None:
                raise ValueError("Mask is required for 'maskpointtrack' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            out_mask = self.sam2_model(image=image, mask=mask, task_type='mask_point', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image_array, out_mask, mask_color, mask_cfg)
            
        elif mode == 'maskbboxtrack':
            if self.sam2_model is None:
                raise ValueError("SAM2 model not initialized")
            if mask is None:
                raise ValueError("Mask is required for 'maskbboxtrack' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            out_mask = self.sam2_model(image=image, mask=mask, task_type='mask_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image_array, out_mask, mask_color, mask_cfg)
            
        elif mode == 'masktrack':
            if self.sam2_model is None:
                raise ValueError("SAM2 model not initialized")
            if mask is None:
                raise ValueError("Mask is required for 'masktrack' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            resize_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            out_mask = self.sam2_model(image=image, mask=resize_mask, task_type='mask', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image_array, out_mask, mask_color, mask_cfg)
            
        elif mode == 'bboxtrack':
            if self.sam2_model is None:
                raise ValueError("SAM2 model not initialized")
            if bbox is None:
                raise ValueError("Bbox is required for 'bboxtrack' mode")
            out_mask = self.sam2_model(image=image, input_box=bbox, task_type='input_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image_array, out_mask, mask_color, mask_cfg)
            
        elif mode == 'label':
            if self.gdino_model is None or self.sam2_model is None:
                raise ValueError("GDINO and SAM2 models not initialized")
            if label is None:
                raise ValueError("Label is required for 'label' mode")
            gdino_res = self.gdino_model(image, classes=label)
            if 'boxes' in gdino_res and len(gdino_res['boxes']) > 0:
                bboxes = gdino_res['boxes'][0]
            else:
                raise ValueError(f"Unable to find the corresponding boxes of label: {label}")
            out_mask = self.sam2_model(image=image, input_box=bboxes, task_type='input_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image_array, out_mask, mask_color, mask_cfg)
            
        elif mode == 'caption':
            if self.gdino_model is None or self.sam2_model is None:
                raise ValueError("GDINO and SAM2 models not initialized")
            if caption is None:
                raise ValueError("Caption is required for 'caption' mode")
            gdino_res = self.gdino_model(image, caption=caption)
            if 'boxes' in gdino_res and len(gdino_res['boxes']) > 0:
                bboxes = gdino_res['boxes'][0]
            else:
                raise ValueError(f"Unable to find the corresponding boxes of caption: {caption}")
            out_mask = self.sam2_model(image=image, input_box=bboxes, task_type='input_box', return_mask=True)
            out_image, out_mask = self.apply_seg_mask(image_array, out_mask, mask_color, mask_cfg)
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        ret_data = {"image": out_image}
        if return_mask:
            ret_data["mask"] = out_mask
        if return_source:
            ret_data["src_image"] = image_array
        return ret_data

    def __str__(self):
        return f"InpaintingPreprocessor(mode={self.mode}, return_mask={self.return_mask})"

    def __repr__(self):
        return self.__str__()


@preprocessor_registry("inpainting.video")
class InpaintingVideoPreprocessor(BasePreprocessor):
    def __init__(self, mode: str = "mask", use_aug: bool = True, return_frame: bool = True,
                 return_mask: bool = True, return_source: bool = True, mask_color: int = 128,
                 salient_config: Optional[Dict] = None, sam2_config: Optional[Dict] = None, 
                 gdino_config: Optional[Dict] = None, **kwargs):
        super().__init__(preprocessor_type=PreprocessorType.VIDEO, **kwargs)
        
        self.mode = mode
        self.use_aug = use_aug
        self.return_frame = return_frame
        self.return_mask = return_mask
        self.return_source = return_source
        self.mask_color = mask_color
        
        assert self.mode in ["salient", "mask", "bbox", "salientmasktrack", "salientbboxtrack", 
                           "maskpointtrack", "maskbboxtrack", "masktrack", "bboxtrack", 
                           "label", "caption", "all"]
        
        # Initialize dependent preprocessors based on mode
        self.salient_model = None
        self.sam2_model = None
        self.gdino_model = None
        self.maskaug_anno = None
        
        if self.mode in ["salient", "salienttrack", "salientmasktrack", "salientbboxtrack", "all"]:
            if salient_config:
                from src.preprocess.salient import SalientPreprocessor
                self.salient_model = SalientPreprocessor(**salient_config)
            else:
                warnings.warn("Salient mode selected but no salient_config provided")
        
        if self.mode in ['masktrack', 'bboxtrack', 'salienttrack', 'salientmasktrack', 
                        'salientbboxtrack', 'maskpointtrack', 'maskbboxtrack', 'label', 'caption', 'all']:
            if sam2_config:
                from src.preprocess.sam2 import SAM2VideoPreprocessor
                self.sam2_model = SAM2VideoPreprocessor(**sam2_config)
            else:
                warnings.warn("SAM2 mode selected but no sam2_config provided")
        
        if self.mode in ['label', 'caption', 'all']:
            if gdino_config:
                from src.preprocess.gdino import GDINOPreprocessor
                self.gdino_model = GDINOPreprocessor(**gdino_config)
            else:
                warnings.warn("GDINO mode selected but no gdino_config provided")
        
        if self.use_aug:
            warnings.warn("Mask augmentation functionality requires additional dependencies and is disabled.")
            self.use_aug = False

    def apply_plain_mask(self, frames: List[np.ndarray], mask: np.ndarray, mask_color: int, return_frame: bool = True):
        """Apply a plain mask to video frames"""
        out_frames = []
        num_frames = len(frames)
        bool_mask = mask > 0
        out_masks = [np.where(bool_mask, 255, 0).astype(np.uint8)] * num_frames
        
        if not return_frame:
            return None, out_masks
            
        for i in range(num_frames):
            masked_frame = frames[i].copy()
            masked_frame[bool_mask] = mask_color
            out_frames.append(masked_frame)
        return out_frames, out_masks

    def apply_seg_mask(self, mask_data: Dict, frames: List[np.ndarray], mask_color: int, 
                      mask_cfg: Optional[Dict] = None, return_frame: bool = True):
        """Apply segmentation masks to video frames"""
        out_frames = []
        out_masks = [(single_rle_to_mask(val[0]["mask"]) * 255).astype('uint8') 
                     for key, val in mask_data['annotations'].items()]
        
        if not return_frame:
            return None, out_masks
            
        num_frames = min(len(out_masks), len(frames))
        for i in range(num_frames):
            sub_mask = out_masks[i]
            if self.use_aug and mask_cfg is not None:
                warnings.warn("Mask augmentation not implemented")
            bool_mask = sub_mask > 0
            masked_frame = frames[i].copy()
            masked_frame[bool_mask] = mask_color
            out_frames.append(masked_frame)
        out_masks = out_masks[:num_frames]
        return out_frames, out_masks

    def __call__(self, 
                frames: Optional[Union[List[Image.Image], List[str], str]] = None,
                video: Optional[str] = None,
                mask: Optional[Union[Image.Image, np.ndarray, str]] = None,
                bbox: Optional[List[float]] = None,
                label: Optional[str] = None,
                caption: Optional[str] = None,
                mode: Optional[str] = None,
                return_frame: Optional[bool] = None,
                return_mask: Optional[bool] = None,
                return_source: Optional[bool] = None,
                mask_color: Optional[int] = None,
                mask_cfg: Optional[Dict] = None):
        
        mode = mode if mode is not None else self.mode
        return_frame = return_frame if return_frame is not None else self.return_frame
        return_mask = return_mask if return_mask is not None else self.return_mask
        return_source = return_source if return_source is not None else self.return_source
        mask_color = mask_color if mask_color is not None else self.mask_color

        # Load frames if not provided
        if frames is None and video is not None:
            frames = self._load_video(video)
        elif frames is not None:
            frames = self._load_video(frames)
        else:
            raise ValueError("Either frames or video must be provided")
        
        # Convert frames to numpy arrays
        frames = [np.array(frame) for frame in frames]
        out_frames, out_masks = [], []
        
        if mode == 'salient':
            if self.salient_model is None:
                raise ValueError("Salient model not initialized")
            first_frame = frames[0] if frames else read_video_one_frame(video)
            mask = self.salient_model(first_frame)
            out_frames, out_masks = self.apply_plain_mask(frames, mask, mask_color, return_frame)
            
        elif mode == 'mask':
            if mask is None:
                raise ValueError("Mask is required for 'mask' mode")
            if isinstance(mask, (str, Image.Image)):
                mask = self._load_image(mask)
            mask = np.array(mask)
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2).astype(np.uint8)
            
            first_frame = frames[0] if frames else read_video_one_frame(video)
            mask_h, mask_w = mask.shape[:2]
            h, w = first_frame.shape[:2]
            if (mask_h != h) or (mask_w != w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            out_frames, out_masks = self.apply_plain_mask(frames, mask, mask_color, return_frame)
            
        elif mode == 'bbox':
            if bbox is None:
                raise ValueError("Bbox is required for 'bbox' mode")
            first_frame = frames[0] if frames else read_video_one_frame(video)
            num_frames = len(frames)
            x1, y1, x2, y2 = bbox
            h, w = first_frame.shape[:2]
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(w, x2)), int(min(h, y2))
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            out_masks = [mask] * num_frames
            
            if not return_frame:
                out_frames = None
            else:
                for i in range(num_frames):
                    masked_frame = frames[i].copy()
                    masked_frame[y1:y2, x1:x2] = mask_color
                    out_frames.append(masked_frame)
                    
        elif mode == 'salientmasktrack':
            if self.salient_model is None or self.sam2_model is None:
                raise ValueError("Salient and SAM2 models not initialized")
            first_frame = frames[0] if frames else read_video_one_frame(video)
            salient_mask = self.salient_model(first_frame)
            mask_data = self.sam2_model(video=video, mask=salient_mask, task_type='mask')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)
            
        elif mode == 'salientbboxtrack':
            if self.salient_model is None or self.sam2_model is None:
                raise ValueError("Salient and SAM2 models not initialized")
            first_frame = frames[0] if frames else read_video_one_frame(video)
            salient_mask = self.salient_model(first_frame)
            bbox = get_mask_box(np.array(salient_mask), threshold=1)
            if bbox is None:
                raise ValueError("Could not extract bounding box from salient mask")
            mask_data = self.sam2_model(video=video, input_box=bbox, task_type='input_box')
            out_frames, out_masks = self.apply_seg_mask(mask_data, frames, mask_color, mask_cfg, return_frame)
            
        # Add other modes as needed...
        else:
            raise ValueError(f"Mode '{mode}' not fully implemented for video yet")

        ret_data = {}
        if return_frame:
            ret_data["frames"] = out_frames
        if return_mask:
            ret_data["masks"] = out_masks
        if return_source:
            ret_data["src_frames"] = frames
        return ret_data

    def __str__(self):
        return f"InpaintingVideoPreprocessor(mode={self.mode}, return_frame={self.return_frame})"

    def __repr__(self):
        return self.__str__() 