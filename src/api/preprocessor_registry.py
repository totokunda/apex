"""
Enhanced preprocessor registry with detailed parameter information
"""
from typing import Dict, Any, List
import os
from pathlib import Path
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
import importlib

# Enhanced registry with parameter definitions for all preprocessors
PREPROCESSOR_REGISTRY = {
    "anime_face_segment": {
        "module": "src.auxillary.anime_face_segment",
        "class": "AnimeFaceSegmentor",
        "name": "Anime Face Segmentation",
        "description": "Segment anime faces from images",
        "category": "Segmentation",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "remove_background", "type": "bool", "default": True, "description": "Remove background from result"}
        ]
    },
    "binary": {
        "module": "src.auxillary.binary",
        "class": "BinaryDetector",
        "name": "Binary Threshold",
        "description": "Convert image to binary (black and white)",
        "supports_video": True,
        "supports_image": True,
        "category": "Line",
        "parameters": [
            {"name": "bin_threshold", "type": "int", "default": 0, "description": "Binary threshold (0 for Otsu auto)"},
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "canny": {
        "module": "src.auxillary.canny",
        "class": "CannyDetector",
        "name": "Canny Edge Detection",
        "description": "Classic Canny edge detection",
        "supports_video": True,
        "supports_image": True,
        "category": "Line",
        "parameters": [
            {"name": "low_threshold", "type": "int", "default": 100, "description": "Lower threshold for edge detection"},
            {"name": "high_threshold", "type": "int", "default": 200, "description": "Upper threshold for edge detection"},
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "color": {
        "category": "Color",
        "module": "src.auxillary.color",
        "class": "ColorDetector",
        "name": "Color Palette",
        "description": "Extract color palette from image",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"}
        ]
    },
    "densepose": {
        "category": "Face and Pose",
        "module": "src.auxillary.densepose",
        "class": "DenseposeDetector",
        "name": "DensePose",
        "description": "Dense human pose estimation",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "cmap", "type": "str", "default": "viridis", "description": "Colormap for visualization"}
        ]
    },
    "depth_anything": {
        "category": "Depth and Normal",
        "module": "src.auxillary.depth_anything.transformers",
        "class": "DepthAnythingDetector",
        "name": "Depth Anything",
        "description": "Monocular depth estimation using Depth Anything",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "depth_anything_v2": {
        "category": "Depth and Normal",
        "module": "src.auxillary.depth_anything_v2",
        "class": "DepthAnythingV2Detector",
        "name": "Depth Anything V2",
        "description": "Improved monocular depth estimation",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "encoder", "type": "str", "default": "vits", "options": ["vits", "vitb", "vitl", "vitg"], "description": "Encoder model size"}
        ]
    },
    "diffusion_edge": {
        "category": "Line",
        "module": "src.auxillary.diffusion_edge",
        "class": "DiffusionEdgeDetector",
        "name": "Diffusion Edge",
        "description": "Edge detection using diffusion models",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "dsine": {
        "category": "Depth and Normal",
        "module": "src.auxillary.dsine",
        "class": "DsineDetector",
        "name": "DSINE Normal Estimation",
        "description": "Surface normal estimation using DSINE",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "dwpose": {
        "category": "Face and Pose",
        "module": "src.auxillary.dwpose",
        "class": "DwposeDetector",
        "name": "DWPose",
        "description": "Whole body pose estimation including face and hands",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "include_body", "type": "bool", "default": True, "description": "Include body keypoints"},
            {"name": "include_hand", "type": "bool", "default": False, "description": "Include hand keypoints"},
            {"name": "include_face", "type": "bool", "default": False, "description": "Include face keypoints"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "xinsr_stick_scaling", "type": "bool", "default": False, "description": "XinSR stick scaling"}
        ]
    },
    "animalpose": {
        "category": "Face and Pose",
        "module": "src.auxillary.dwpose",
        "class": "AnimalPoseDetector",
        "name": "Animal Pose",
        "description": "Animal pose estimation using RTMPose AP10k model",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "hed": {
        "category": "Line",
        "module": "src.auxillary.hed",
        "class": "HEDdetector",
        "name": "HED Edge Detection",
        "description": "Holistically-nested edge detection",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "safe", "type": "bool", "default": False, "description": "Enable safe mode"},
            {"name": "scribble", "type": "bool", "default": False, "description": "Generate scribble-style output"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "leres": {
        "category": "Depth and Normal",
        "module": "src.auxillary.leres",
        "class": "LeresDetector",
        "name": "LeReS Depth",
        "description": "Depth estimation using LeReS",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "thr_a", "type": "float", "default": 0.0, "description": "Threshold parameter A"},
            {"name": "thr_b", "type": "float", "default": 0.0, "description": "Threshold parameter B"},
            {"name": "boost", "type": "bool", "default": False, "description": "Enable boost mode"}
        ]
    },
    "lineart": {
        "category": "Line",
        "module": "src.auxillary.lineart",
        "class": "LineartDetector",
        "name": "Line Art",
        "description": "Extract line art from images",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "coarse", "type": "bool", "default": False, "description": "Use coarse line extraction"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "lineart_anime": {
        "category": "Line",
        "module": "src.auxillary.lineart_anime",
        "class": "LineartAnimeDetector",
        "name": "Line Art Anime",
        "description": "Extract line art optimized for anime",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "lineart_standard": {
        "category": "Line",
        "module": "src.auxillary.lineart_standard",
        "class": "LineartStandardDetector",
        "name": "Line Art Standard",
        "description": "Standard line art extraction",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "guassian_sigma", "type": "float", "default": 2.0, "description": "Gaussian blur sigma"}
        ]
    },
    "manga_line": {
        "category": "Line",
        "module": "src.auxillary.manga_line",
        "class": "LineartMangaDetector",
        "name": "Manga Line Art",
        "description": "Extract line art optimized for manga",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "mediapipe_face": {
        "category": "Face and Pose",
        "module": "src.auxillary.mediapipe_face",
        "class": "MediapipeFaceDetector",
        "name": "MediaPipe Face",
        "description": "Face mesh detection using MediaPipe",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "max_faces", "type": "int", "default": 1, "description": "Maximum number of faces to detect"},
            {"name": "min_confidence", "type": "float", "default": 0.5, "description": "Minimum detection confidence"},
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "mesh_graphormer": {
        "category": "Face and Pose",
        "module": "src.auxillary.mesh_graphormer",
        "class": "MeshGraphormerDetector",
        "name": "Mesh Graphormer",
        "description": "3D hand mesh reconstruction",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "mask_bbox_padding", "type": "int", "default": 30, "description": "Padding for hand bounding box"}
        ]
    },
    "metric3d": {
        "category": "Depth and Normal",
        "module": "src.auxillary.metric3d",
        "class": "Metric3DDetector",
        "name": "Metric3D",
        "description": "Metric depth and normal estimation",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "fx", "type": "float", "default": 1000.0, "description": "Focal length X"},
            {"name": "fy", "type": "float", "default": 1000.0, "description": "Focal length Y"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "output_type", "type": "str", "default": "Depth and Normal", "options": ["Depth and Normal", "normal"], "description": "Output type"}
        ]
    },
    "midas": {
        "category": "Depth and Normal",
        "module": "src.auxillary.midas.transformers",
        "class": "MidasDetector",
        "name": "MiDaS Depth",
        "description": "Monocular depth estimation using MiDaS",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "mlsd": {
        "category": "Line",
        "module": "src.auxillary.mlsd",
        "class": "MLSDdetector",
        "name": "M-LSD Line Detection",
        "description": "Mobile line segment detection",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "thr_v", "type": "float", "default": 0.1, "description": "Line segment threshold V"},
            {"name": "thr_d", "type": "float", "default": 0.1, "description": "Line segment threshold D"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "normalbae": {
        "category": "Depth and Normal",
        "module": "src.auxillary.normalbae",
        "class": "NormalBaeDetector",
        "name": "Normal BAE",
        "description": "Surface normal estimation using BAE",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "oneformer": {
        "category": "Segmentation",
        "module": "src.auxillary.oneformer.transformers",
        "class": "OneformerSegmentor",
        "name": "OneFormer Segmentation",
        "description": "Universal image segmentation",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "open_pose": {
        "category": "Face and Pose",
        "module": "src.auxillary.open_pose",
        "class": "OpenposeDetector",
        "name": "OpenPose",
        "description": "Human pose estimation with body, hands, and face",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "include_body", "type": "bool", "default": True, "description": "Include body keypoints"},
            {"name": "include_hand", "type": "bool", "default": True, "description": "Include hand keypoints"},
            {"name": "include_face", "type": "bool", "default": True, "description": "Include face keypoints"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "xinsr_stick_scaling", "type": "bool", "default": False, "description": "XinSR stick scaling"}
        ]
    },
    "pidi": {
        "category": "Line",
        "module": "src.auxillary.pidi",
        "class": "PidiNetDetector",
        "name": "PiDiNet Edge Detection",
        "description": "Pixel difference network for edge detection",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "safe", "type": "bool", "default": False, "description": "Enable safe mode"},
            {"name": "scribble", "type": "bool", "default": False, "description": "Generate scribble-style output"},
            {"name": "apply_filter", "type": "bool", "default": False, "description": "Apply filter"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "ptlflow": {
        "category": "Optical Flow",
        "module": "src.auxillary.ptlflow",
        "class": "PTLFlowDetector",
        "name": "PTLFlow Optical Flow",
        "description": "Optical flow estimation for videos",
        "supports_video": True,
        "supports_image": False,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "output_type", "type": "str", "default": "vis", "options": ["vis", "flow"], "description": "Output type: visualization or raw flow"}
        ]
    },
    "pyracanny": {
        "category": "Line",
        "module": "src.auxillary.pyracanny",
        "class": "PyraCannyDetector",
        "name": "Pyramid Canny",
        "description": "Multi-scale pyramid Canny edge detection",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "low_threshold", "type": "int", "default": 100, "description": "Lower threshold for edge detection"},
            {"name": "high_threshold", "type": "int", "default": 200, "description": "Upper threshold for edge detection"},
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "rembg": {
        "category": "Background Removal",
        "module": "src.auxillary.rembg",
        "class": "RembgDetector",
        "name": "Background Removal",
        "description": "Remove background from images and videos",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 0, "description": "Resolution for detection (0 = original)"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "alpha_matting", "type": "bool", "default": False, "description": "Enable alpha matting for better edges"},
            {"name": "alpha_matting_foreground_threshold", "type": "int", "default": 240, "description": "Foreground threshold"},
            {"name": "alpha_matting_background_threshold", "type": "int", "default": 10, "description": "Background threshold"},
            {"name": "alpha_matting_erode_size", "type": "int", "default": 10, "description": "Erosion size"},
            {"name": "post_process_mask", "type": "bool", "default": False, "description": "Post-process mask"}
        ]
    },
    "recolor": {
        "category": "Color",
        "module": "src.auxillary.recolor",
        "class": "Recolorizer",
        "name": "Recolor",
        "description": "Recolor image based on luminance or intensity",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "mode", "type": "str", "default": "luminance", "options": ["luminance", "intensity"], "description": "Recolor mode"},
            {"name": "gamma_correction", "type": "float", "default": 1.0, "description": "Gamma correction factor"},
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "scribble": {
        "category": "Line",
        "module": "src.auxillary.scribble",
        "class": "ScribbleDetector",
        "name": "Scribble",
        "description": "Generate scribble-style edges",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_AREA", "description": "Image resizing method"}
        ]
    },
    "scribble_xdog": {
        "category": "Line",
        "module": "src.auxillary.scribble",
        "class": "ScribbleXDogDetector",
        "name": "Scribble XDoG",
        "description": "Extended Difference of Gaussians scribble",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "thr_a", "type": "int", "default": 32, "description": "Threshold parameter"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "scribble_anime": {
        "category": "Line",
        "module": "src.auxillary.scribble_anime",
        "class": "ScribbleAnimeDetector",
        "name": "Scribble Anime",
        "description": "Generate anime-style scribble edges using neural network",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "shuffle": {
        "category": "Color",
        "module": "src.auxillary.shuffle",
        "class": "ContentShuffleDetector",
        "name": "Content Shuffle",
        "description": "Shuffle image content spatially",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "h", "type": "int", "default": None, "description": "Height for noise generation"},
            {"name": "w", "type": "int", "default": None, "description": "Width for noise generation"},
            {"name": "f", "type": "int", "default": 256, "description": "Frequency parameter"},
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "seed", "type": "int", "default": -1, "description": "Random seed (-1 for random)"}
        ]
    },
    "teed": {
        "category": "Line",
        "module": "src.auxillary.teed",
        "class": "TEDDetector",
        "name": "TEED Edge Detection",
        "description": "Tiny and efficient edge detector",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "safe_steps", "type": "int", "default": 2, "description": "Number of safe steps"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "tile": {
        "category": "Color",
        "module": "src.auxillary.tile",
        "class": "TileDetector",
        "name": "Tile Resample",
        "description": "Pyramid downsampling and upsampling",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "pyrUp_iters", "type": "int", "default": 3, "description": "Number of pyramid up iterations"},
            {"name": "upscale_method", "type": "str", "default": "INTER_AREA", "description": "Image resizing method"}
        ]
    },
    "tile_gf": {
        "category": "Color",
        "module": "src.auxillary.tile",
        "class": "TTPlanet_Tile_Detector_GF",
        "name": "Tile Guided Filter",
        "description": "Tile with guided filter preprocessing",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "scale_factor", "type": "float", "default": 2.0, "description": "Downscale factor"},
            {"name": "blur_strength", "type": "float", "default": 1.0, "description": "Gaussian blur strength"},
            {"name": "radius", "type": "int", "default": 3, "description": "Guided filter radius"},
            {"name": "eps", "type": "float", "default": 0.01, "description": "Guided filter epsilon"}
        ]
    },
    "tile_simple": {
        "category": "Color",
        "module": "src.auxillary.tile",
        "class": "TTPLanet_Tile_Detector_Simple",
        "name": "Tile Simple",
        "description": "Simple tile preprocessing with blur",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "scale_factor", "type": "float", "default": 2.0, "description": "Downscale factor"},
            {"name": "blur_strength", "type": "float", "default": 1.0, "description": "Gaussian blur strength"}
        ]
    },
    "uniformer": {
        "category": "Segmentation",
        "module": "src.auxillary.uniformer",
        "class": "UniformerSegmentor",
        "name": "Uniformer Segmentation",
        "description": "Semantic segmentation using Uniformer",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "unimatch": {
        "category": "Optical Flow",
        "module": "src.auxillary.unimatch",
        "class": "UnimatchDetector",
        "name": "UniMatch Optical Flow",
        "description": "Optical flow estimation using UniMatch",
        "supports_video": True,
        "supports_image": False,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"},
            {"name": "pred_bwd_flow", "type": "bool", "default": False, "description": "Predict backward flow"},
            {"name": "pred_bidir_flow", "type": "bool", "default": False, "description": "Predict bidirectional flow"}
        ]
    },
    "zoe": {
        "category": "Depth and Normal",
        "module": "src.auxillary.zoe.transformers",
        "class": "ZoeDetector",
        "name": "ZoeDepth",
        "description": "Metric depth estimation using ZoeDepth",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
    "zoe_depth_anything": {
        "category": "Depth and Normal",
        "module": "src.auxillary.zoe.transformers",
        "class": "ZoeDepthAnythingDetector",
        "name": "ZoeDepth Anything",
        "description": "ZoeDepth with Depth Anything features",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "detect_resolution", "type": "int", "default": 512, "description": "Resolution for detection"},
            {"name": "upscale_method", "type": "str", "default": "INTER_CUBIC", "description": "Image resizing method"}
        ]
    },
}


def get_preprocessor_info(preprocessor_name: str) -> Dict[str, Any]:
    """
    Get preprocessor module and class info.
    
    Args:
        preprocessor_name: Name of the preprocessor
        
    Returns:
        Dictionary with preprocessor information
    """
    if preprocessor_name not in PREPROCESSOR_REGISTRY:
        raise ValueError(f"Preprocessor {preprocessor_name} not found. Available: {list(PREPROCESSOR_REGISTRY.keys())}")
    return PREPROCESSOR_REGISTRY[preprocessor_name]


def list_preprocessors(check_downloaded: bool = False) -> List[Dict[str, Any]]:
    """
    List all available preprocessors with their metadata.
    
    Args:
        check_downloaded: If True, check download status for each preprocessor (slower)
    
    Returns:
        List of preprocessor information dictionaries
    """
    result = []
    for name, info in PREPROCESSOR_REGISTRY.items():
        preprocessor_info = {
            "id": name,
            "name": info.get("name", name),
            "category": info.get("category", ""),
            "description": info.get("description", ""),
            "supports_video": info.get("supports_video", True),
            "supports_image": info.get("supports_image", True),
            "parameters": info.get("parameters", [])
        }
        
        # Optionally check download status
        if check_downloaded:
            preprocessor_info["is_downloaded"] = check_preprocessor_downloaded(name)
        
        result.append(preprocessor_info)
    
    return sorted(result, key=lambda x: x["name"])


def initialize_download_tracking():
    """
    Initialize the download tracking file with preprocessors that don't require downloads.
    This should be called on app startup.
    """
    from src.auxillary.base_preprocessor import BasePreprocessor
    
    # Preprocessors that don't require downloads
    NO_DOWNLOAD_REQUIRED = [
        "binary", "canny", "color", "pyracanny", "recolor", 
        "scribble", "scribble_xdog", "shuffle", "tile", 
        "tile_gf", "tile_simple"
    ]
    
    for preprocessor_name in NO_DOWNLOAD_REQUIRED:
        BasePreprocessor._mark_as_downloaded(preprocessor_name)


def check_preprocessor_downloaded(preprocessor_name: str) -> bool:
    """
    Check if a preprocessor's model files are downloaded.
    
    Args:
        preprocessor_name: Name of the preprocessor
        
    Returns:
        True if downloaded/ready, False otherwise
    """
    # Preprocessors that don't require downloads
    NO_DOWNLOAD_REQUIRED = {
        "binary", "canny", "color", "pyracanny", "recolor", 
        "scribble", "scribble_xdog", "shuffle", "tile", 
        "tile_gf", "tile_simple"
    }
    
    if preprocessor_name in NO_DOWNLOAD_REQUIRED:
        return True
    
    # Check the downloaded preprocessors tracking file
    from src.auxillary.base_preprocessor import BasePreprocessor
    return BasePreprocessor._is_downloaded(preprocessor_name)


def get_preprocessor_details(preprocessor_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific preprocessor.
    
    Args:
        preprocessor_name: Name of the preprocessor
        
    Returns:
        Dictionary with detailed preprocessor information including parameters
    """
    if preprocessor_name not in PREPROCESSOR_REGISTRY:
        raise ValueError(f"Preprocessor {preprocessor_name} not found. Available: {list(PREPROCESSOR_REGISTRY.keys())}")
    
    info = PREPROCESSOR_REGISTRY[preprocessor_name]
    is_downloaded = check_preprocessor_downloaded(preprocessor_name)
    
    return {
        "id": preprocessor_name,
        "name": info.get("name", preprocessor_name),
        "category": info.get("category", ""),
        "description": info.get("description", ""),
        "module": info["module"],
        "class": info["class"],
        "supports_video": info.get("supports_video", True),
        "supports_image": info.get("supports_image", True),
        "parameters": info.get("parameters", []),
        "is_downloaded": is_downloaded
    }

