"""
Enhanced preprocessor registry with detailed parameter information
"""
from typing import Dict, Any, List
import os
from pathlib import Path
from functools import lru_cache
from src.utils.yaml import load_yaml as load_yaml_file
PREPROCESSOR_PATH = Path(__file__).parent.parent.parent / 'manifest' / 'preprocessor'
PREPROCESSOR_PATH = Path(__file__).parent.parent.parent / 'manifest' / 'preprocessor'

@lru_cache(maxsize=None)
def _available_preprocessor_names() -> List[str]:
    if not PREPROCESSOR_PATH.exists():
        return []
    names: List[str] = []
    for entry in PREPROCESSOR_PATH.iterdir():
        if entry.is_file() and entry.suffix in {'.yml', '.yaml'} and not entry.name.startswith('shared'):
            names.append(entry.stem)
    return sorted(names)

@lru_cache(maxsize=None)
def _load_preprocessor_yaml(preprocessor_name: str) -> Dict[str, Any]:
    file_path_yml = PREPROCESSOR_PATH / f"{preprocessor_name}.yml"
    file_path_yaml = PREPROCESSOR_PATH / f"{preprocessor_name}.yaml"
    file_path = file_path_yml if file_path_yml.exists() else (file_path_yaml if file_path_yaml.exists() else None)
    if file_path is None:
        available = _available_preprocessor_names()
        raise ValueError(f"Preprocessor {preprocessor_name} not found. Available: {available}")
    data = load_yaml_file(file_path)
    if not isinstance(data, dict):
        data = {}
    data.setdefault('name', preprocessor_name)
    data.setdefault('category', '')
    data.setdefault('description', '')
    data.setdefault('module', '')
    data.setdefault('class', '')
    data.setdefault('supports_video', True)
    data.setdefault('supports_image', True)
    data.setdefault('parameters', [])
    return data



detect_resolution_parameter = {
    "name": "detect_resolution",
    "display_name": "Detection Resolution",
    "type": "category",
    "default": 512,
    "options": [{
       "name": "Standard",
       "value": 512
    }, {
       "name": "High Definition",
       "value": 1024
    }, {"name": "Current Image", "value": 0}],
    "description": "The resolution used for detection and inference. Higher resolutions provide more detail but require more processing time and memory."
}

upscale_method_parameter = {
    "name": "upscale_method",
    "display_name": "Upscale Method",
    "type": "category",
    "default": "INTER_CUBIC",
    "options": [{
       "name": "Nearest Neighbor",
       "value": "INTER_NEAREST"
    }, {"name": "Linear", "value": "INTER_LINEAR"}, 
    {"name": "Cubic", "value": "INTER_CUBIC"}, 
    {"name": "Lanczos", "value": "INTER_LANCZOS4"}],
    "description": "The interpolation method used when resizing images. Bicubic and Lanczos provide smoother results, while Nearest Neighbor preserves sharp edges."
}



# Enhanced registry with parameter definitions for all preprocessors
PREPROCESSOR_REGISTRY = {
    "anime_face_segment": {
        "module": "src.preprocess.anime_face_segment",
        "class": "AnimeFaceSegmentor",
        "name": "Anime Face Segmentation",
        "description": "Detects and segments anime-style faces from images using specialized neural networks trained on anime artwork. Accurately isolates facial regions from anime characters and can optionally remove the background to extract just the face.",
        "category": "Segmentation",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "remove_background", "display_name": "Remove Background", "type": "bool", "default": True, "description": "When enabled, removes the background and keeps only the segmented anime face. When disabled, shows the segmentation mask overlay."}
        ]
    },
    "binary": {
        "module": "src.preprocess.binary",
        "class": "BinaryDetector",
        "name": "Binary Threshold",
        "description": "Converts images to pure black and white using threshold-based segmentation. Pixels above the threshold become white, below become black. Useful for creating high-contrast masks, extracting silhouettes, or preparing images for stylized effects.",
        "supports_video": True,
        "supports_image": True,
        "category": "Line",
        "parameters": [
            {"name": "bin_threshold", "display_name": "Binary Threshold", "type": "int", "default": 0, "description": "The threshold value for binary conversion. Pixels above this value become white, below become black. Set to 0 to use automatic Otsu thresholding.", "min": 0, "max": 255},
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "canny": {
        "module": "src.preprocess.canny",
        "class": "CannyDetector",
        "name": "Canny Edge Detection",
        "description": "Classic multi-stage edge detection algorithm developed by John Canny. Uses gradient analysis to identify edges with precise localization and minimal false positives. Ideal for detecting clear, well-defined edges in photos and architectural images.",
        "supports_video": True,
        "supports_image": True,
        "category": "Line",
        "parameters": [
            {"name": "low_threshold", "display_name": "Low Threshold", "type": "int", "default": 100, "description": "The lower threshold for the Canny edge detector. Edges with gradient values below this threshold are discarded. Lower values detect more edges including weak ones.", "min": 0, "max": 500},
            {"name": "high_threshold", "display_name": "High Threshold", "type": "int", "default": 200, "description": "The upper threshold for the Canny edge detector. Edges with gradient values above this threshold are kept as strong edges. Higher values detect only the most prominent edges.", "min": 0, "max": 500},
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "color": {
        "category": "Color",
        "module": "src.preprocess.color",
        "class": "ColorDetector",
        "name": "Color Palette",
        "description": "Extracts and visualizes the dominant color palette from an image. Analyzes the image to identify the main colors present and creates a simplified color-block representation. Useful for color analysis, style transfer preparation, and creating color reference guides.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter
        ]
    },
    "densepose": {
        "category": "Face and Pose",
        "module": "src.preprocess.densepose",
        "class": "DenseposeDetector",
        "name": "DensePose",
        "description": "Maps all human pixels in an image to a 3D surface model of the body. Provides dense correspondence between 2D image pixels and 3D body surface, enabling detailed body shape understanding. Excellent for human body analysis, pose transfer, and avatar creation.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "cmap", "display_name": "Color Map", "type": "category", "default": "viridis", "options": [{
                "name": "Viridis",
                "value": "viridis"
            }, {
                "name": "Plasma",
                "value": "plasma"
            }, {
                "name": "Inferno", "value": "inferno"
            }, {
                "name": "Magma", "value": "magma"
            }, {
                "name": "Cividis", "value": "cividis"
            },
            ], "description": "The color mapping scheme used to visualize the dense pose estimation. Different colormaps provide different visual representations of the body surface mapping."}
        ]
    },
    "depth_anything": {
        "category": "Depth and Normal",
        "module": "src.preprocess.depth_anything.transformers",
        "class": "DepthAnythingDetector",
        "name": "Depth Anything",
        "description": "State-of-the-art monocular depth estimation that works reliably across diverse scenes and conditions. Trained on massive datasets to handle any image type from indoor scenes to outdoor landscapes. Produces accurate relative depth maps showing distance from camera.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "depth_anything_v2": {
        "category": "Depth and Normal",
        "module": "src.preprocess.depth_anything_v2",
        "class": "DepthAnythingV2Detector",
        "name": "Depth Anything V2",
        "description": "Enhanced version of Depth Anything with improved accuracy and finer detail preservation. Offers multiple model sizes to balance between speed and quality. Better at handling challenging scenes with transparent objects, reflections, and complex depth variations.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "encoder", "display_name": "Encoder Model", "type": "category", "default": "vits", "options": [{
                "name": "VITS",
                "value": "vits"
            }, {
                "name": "ViTB",
                "value": "vitb"
            }, {"name": "ViTL", "value": "vitl"}, {"name": "ViTG", "value": "vitg"}], "description": "The Vision Transformer encoder model size to use. Larger models (ViTL, ViTG) provide better accuracy but require more memory and processing time. Smaller models (VITS, ViTB) are faster but may be less accurate."}
        ]
    },
    "diffusion_edge": {
        "category": "Line",
        "module": "src.preprocess.diffusion_edge",
        "class": "DiffusionEdgeDetector",
        "name": "Diffusion Edge",
        "description": "Advanced edge detection powered by diffusion models for high-quality, semantically-aware edge extraction. Produces cleaner, more perceptually meaningful edges compared to traditional methods. Excellent for artistic applications and complex scene understanding.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "dsine": {
        "category": "Depth and Normal",
        "module": "src.preprocess.dsine",
        "class": "DsineDetector",
        "name": "DSINE Normal Estimation",
        "description": "Estimates surface normal vectors for every pixel, showing the orientation of surfaces in 3D space. Uses deep learning to predict accurate normals even for complex geometries. Essential for relighting, 3D reconstruction, and understanding surface structure.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "dwpose": {
        "category": "Face and Pose",
        "module": "src.preprocess.dwpose",
        "class": "DwposeDetector",
        "name": "DWPose",
        "description": "Comprehensive whole-body pose estimation capturing body, hands, and facial keypoints in a single unified model. Fast and accurate for real-time applications. Ideal for animation reference, motion analysis, and character pose extraction with granular control over which components to detect.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            {"name": "include_body", "display_name": "Include Body", "type": "bool", "default": True, "description": "When enabled, detects and displays body keypoints including torso, arms, and legs. This is typically the main component of pose estimation."},
            {"name": "include_hand", "display_name": "Include Hands", "type": "bool", "default": False, "description": "When enabled, detects and displays detailed hand keypoints for finger and palm positions. Useful for capturing hand gestures and fine hand movements."},
            {"name": "include_face", "display_name": "Include Face", "type": "bool", "default": False, "description": "When enabled, detects and displays facial keypoints including eyes, nose, mouth, and facial landmarks. Useful for capturing facial expressions and head orientation."},
            upscale_method_parameter,
            {"name": "xinsr_stick_scaling", "display_name": "XinSR Stick Scaling", "type": "bool", "default": False, "description": "Applies XinSR stick scaling algorithm to normalize the pose skeleton representation. Useful for maintaining consistent pose proportions across different image sizes."}
        ]
    },
    "animalpose": {
        "category": "Face and Pose",
        "module": "src.preprocess.dwpose",
        "class": "AnimalPoseDetector",
        "name": "Animal Pose",
        "description": "Specialized pose estimation for animals using the RTMPose AP10k model trained on diverse animal species. Detects skeletal keypoints for quadrupeds and other animals. Perfect for wildlife photography analysis, pet videos, and animal animation reference.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "hed": {
        "category": "Line",
        "module": "src.preprocess.hed",
        "class": "HEDdetector",
        "name": "HED Edge Detection",
        "description": "Holistically-Nested Edge Detection using deep learning to identify object boundaries and meaningful edges. Produces thicker, more object-aware edges than traditional methods. Great for converting photos to sketch-like representations and understanding scene structure.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            {"name": "safe", "display_name": "Safe Mode", "type": "bool", "default": False, "description": "When enabled, applies additional filtering to remove noise and produce cleaner edge detection results. Recommended for images with complex backgrounds."},
            {"name": "scribble", "display_name": "Scribble Style", "type": "bool", "default": False, "description": "When enabled, converts the edge detection output to a scribble-style drawing with simplified, artistic lines instead of precise edges."},
            upscale_method_parameter,
        ]
    },
    "leres": {
        "category": "Depth and Normal",
        "module": "src.preprocess.leres",
        "class": "LeresDetector",
        "name": "LeReS Depth",
        "description": "Learning to Recover Shape (LeReS) for monocular depth estimation with optional boost mode for higher accuracy. Excels at recovering fine geometric details and sharp depth boundaries. Particularly effective for architectural scenes and objects with complex geometry.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "thr_a", "display_name": "Threshold A", "type": "float", "default": 0.0, "description": "The first threshold parameter for depth map refinement. Controls the lower bound of depth value adjustments. Set to 0.0 for automatic calculation.", "min": 0.0, "max": 1.0},
            {"name": "thr_b", "display_name": "Threshold B", "type": "float", "default": 0.0, "description": "The second threshold parameter for depth map refinement. Controls the upper bound of depth value adjustments. Set to 0.0 for automatic calculation.", "min": 0.0, "max": 1.0},
            {"name": "boost", "display_name": "Boost Mode", "type": "bool", "default": False, "description": "When enabled, uses a more powerful model variant for improved depth estimation accuracy at the cost of increased processing time and memory usage."}
        ]
    },
    "lineart": {
        "category": "Line",
        "module": "src.preprocess.lineart",
        "class": "LineartDetector",
        "name": "Line Art",
        "description": "Extracts clean line art from photographs and images using neural networks. Converts photos into black and white line drawings suitable for coloring, tracing, or artistic stylization. Offers coarse and fine line extraction modes for different artistic styles.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            {"name": "coarse", "display_name": "Coarse Mode", "type": "bool", "default": False, "description": "When enabled, extracts thicker, more prominent lines with less detail. When disabled, extracts finer lines with more detail and subtlety."},
            upscale_method_parameter,
        ]
    },
    "lineart_anime": {
        "category": "Line",
        "module": "src.preprocess.lineart_anime",
        "class": "LineartAnimeDetector",
        "name": "Line Art Anime",
        "description": "Specialized line art extraction optimized for anime and manga-style images. Trained specifically on anime artwork to produce clean, consistent lines that match anime aesthetic. Perfect for extracting linework from anime screenshots or converting photos to anime-style line art.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "lineart_standard": {
        "category": "Line",
        "module": "src.preprocess.lineart_standard",
        "class": "LineartStandardDetector",
        "name": "Line Art Standard",
        "description": "General-purpose line art extraction with Gaussian smoothing for balanced results across different image types. Adjustable sigma parameter allows fine control over line smoothness. Versatile option for realistic photos, illustrations, and mixed content.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "guassian_sigma", "display_name": "Gaussian Sigma", "type": "float", "default": 2.0, "description": "The standard deviation of the Gaussian blur kernel applied during line extraction. Higher values create smoother, softer lines while lower values preserve more detail and sharpness.", "min": 0.1, "max": 10.0}
        ]
    },
    "manga_line": {
        "category": "Line",
        "module": "src.preprocess.manga_line",
        "class": "LineartMangaDetector",
        "name": "Manga Line Art",
        "description": "Specialized for manga-style line extraction with emphasis on the bold, dynamic linework characteristic of manga art. Produces high-contrast black and white lines matching traditional manga inking. Ideal for manga panels, comic art, and graphic novel preparation.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "mediapipe_face": {
        "category": "Face and Pose",
        "module": "src.preprocess.mediapipe_face",
        "class": "MediapipeFaceDetector",
        "name": "MediaPipe Face",
        "description": "Real-time 3D face mesh detection with 468 facial landmarks using Google's MediaPipe framework. Tracks detailed facial features including eyes, eyebrows, nose, lips, and face contour. Optimized for speed and works well even on mobile devices. Perfect for AR filters, facial animation, and expression analysis.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "max_faces", "display_name": "Maximum Faces", "type": "int", "default": 1, "description": "The maximum number of faces to detect in the image. Detecting more faces requires additional processing time. Range: 1-10 faces.", "min": 1, "max": 10},
            {"name": "min_confidence", "display_name": "Minimum Confidence", "type": "float", "default": 0.5, "description": "The minimum confidence threshold for face detection (0.0 to 1.0). Higher values reduce false positives but may miss less clear faces. Lower values detect more faces but may include false detections.", "min": 0.0, "max": 1.0},
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "mesh_graphormer": {
        "category": "Face and Pose",
        "module": "src.preprocess.mesh_graphormer",
        "class": "MeshGraphormerDetector",
        "name": "Mesh Graphormer",
        "description": "Advanced 3D hand mesh reconstruction from single images using graph convolutional networks. Recovers detailed 3D hand geometry including finger positions and joints. Excellent for hand tracking, gesture recognition, sign language analysis, and creating hand animations.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "mask_bbox_padding", "display_name": "Bounding Box Padding", "type": "int", "default": 30, "description": "The amount of padding (in pixels) added around the detected hand bounding box. More padding captures more context around the hand but may include unwanted background. Less padding focuses tightly on the hand.", "min": 0, "max": 100}
        ]
    },
    "metric3d": {
        "category": "Depth and Normal",
        "module": "src.preprocess.metric3d",
        "class": "Metric3DDetector",
        "name": "Metric3D",
        "description": "Produces metric (absolute scale) depth maps and surface normals from single images. Unlike relative depth, provides real-world distance measurements when camera parameters are known. Can output both depth and normal maps or normals only. Ideal for robotics, AR applications, and precise 3D reconstruction.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            {"name": "fx", "display_name": "Focal Length X", "type": "float", "default": 1000.0, "description": "The horizontal focal length of the camera in pixels. This affects the perspective scaling in the X direction. Higher values represent more zoomed-in/telephoto lenses, lower values represent wider-angle lenses.", "min": 100.0, "max": 5000.0},
            {"name": "fy", "display_name": "Focal Length Y", "type": "float", "default": 1000.0, "description": "The vertical focal length of the camera in pixels. This affects the perspective scaling in the Y direction. Typically similar to focal length X for standard cameras.", "min": 100.0, "max": 5000.0},
            upscale_method_parameter,
            {"name": "output_type", "display_name": "Output Type", "type": "category", "default": "depth", "options": [{"name": "Depth", "value": "depth"}, {"name": "Normal", "value": "normal"}], "description": "The type of output to generate. 'Depth and Normal' produces both depth map and surface normals, while 'Normal Only' produces just the surface normal map."}
        ]
    },
    "midas": {
        "category": "Depth and Normal",
        "module": "src.preprocess.midas.transformers",
        "class": "MidasDetector",
        "name": "MiDaS Depth",
        "description": "Robust monocular depth estimation trained on diverse datasets for generalization across scene types. One of the most widely-used depth estimators with proven reliability. Produces smooth, consistent depth maps for indoor and outdoor scenes. Great general-purpose choice for depth-based effects and 3D scene understanding.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "mlsd": {
        "category": "Line",
        "module": "src.preprocess.mlsd",
        "class": "MLSDdetector",
        "name": "M-LSD Line Detection",
        "description": "Mobile Line Segment Detection optimized for detecting straight lines and architectural features. Lightweight and fast while maintaining accuracy. Excels at finding structural lines in buildings, rooms, and geometric objects. Ideal for architectural visualization, perspective correction, and wireframe extraction.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            {"name": "thr_v", "display_name": "Value Threshold", "type": "float", "default": 0.1, "description": "The value threshold for line segment detection. Controls the minimum confidence required for detecting line segments. Lower values detect more lines including faint ones, higher values detect only strong, clear lines.", "min": 0.0, "max": 1.0},
            {"name": "thr_d", "display_name": "Distance Threshold", "type": "float", "default": 0.1, "description": "The distance threshold for line segment merging. Controls how close line segments need to be to be merged into a single line. Lower values keep line segments separate, higher values merge nearby segments.", "min": 0.0, "max": 1.0},
            upscale_method_parameter,
        ]
    },
    "normalbae": {
        "category": "Depth and Normal",
        "module": "src.preprocess.normalbae",
        "class": "NormalBaeDetector",
        "name": "Normal BAE",
        "description": "Boundary-Aware Estimator for accurate surface normal prediction from single images. Produces high-quality normal maps with precise boundaries and fine geometric details. Particularly effective for objects with complex surface variations. Useful for relighting, material editing, and 3D reconstruction pipelines.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "oneformer": {
        "category": "Segmentation",
        "module": "src.preprocess.oneformer.transformers",
        "class": "OneformerSegmentor",
        "name": "OneFormer Segmentation",
        "description": "Universal image segmentation framework that handles semantic, instance, and panoptic segmentation in one model. Identifies and separates different objects and regions with per-class labeling. Excellent for scene understanding, object isolation, and creating detailed masks for complex scenes.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "open_pose": {
        "category": "Face and Pose",
        "module": "src.preprocess.open_pose",
        "class": "OpenposeDetector",
        "name": "OpenPose",
        "description": "Industry-standard multi-person pose estimation detecting body, hands, and face keypoints simultaneously. The original and most widely-used pose detection system. Robust multi-person tracking even in crowded scenes. Essential for motion capture, fitness analysis, dance videos, and character animation reference.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            {"name": "include_body", "display_name": "Include Body", "type": "bool", "default": True, "description": "When enabled, detects and displays body keypoints including torso, arms, and legs. This is the primary component of full-body pose estimation."},
            {"name": "include_hand", "display_name": "Include Hands", "type": "bool", "default": True, "description": "When enabled, detects and displays detailed hand keypoints for both left and right hands. Captures finger positions and hand gestures."},
            {"name": "include_face", "display_name": "Include Face", "type": "bool", "default": True, "description": "When enabled, detects and displays detailed facial keypoints including eyes, nose, mouth, and facial contours. Useful for capturing facial expressions."},
            upscale_method_parameter,
            {"name": "xinsr_stick_scaling", "display_name": "XinSR Stick Scaling", "type": "bool", "default": False, "description": "Applies XinSR stick scaling algorithm to normalize the pose skeleton representation. Helps maintain consistent proportions across different image resolutions."}
        ]
    },
    "pidi": {
        "category": "Line",
        "module": "src.preprocess.pidi",
        "class": "PidiNetDetector",
        "name": "PiDiNet Edge Detection",
        "description": "Pixel Difference Network for high-quality edge detection with minimal parameters. Efficient and accurate edge detector that works well across various image types. Offers safe mode and scribble output options. Good balance between quality and computational efficiency for edge-based applications.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            {"name": "safe", "display_name": "Safe Mode", "type": "bool", "default": False, "description": "When enabled, applies post-processing to reduce noise and produce cleaner edge maps. Recommended for complex or noisy images."},
            {"name": "scribble", "display_name": "Scribble Style", "type": "bool", "default": False, "description": "When enabled, converts edges to a scribble-style artistic representation with simplified, sketch-like lines."},
            {"name": "apply_filter", "display_name": "Apply Filter", "type": "bool", "default": False, "description": "When enabled, applies additional filtering to refine edge detection and remove artifacts. Can improve edge quality but may reduce detail."},
            upscale_method_parameter,
        ]
    },
    "ptlflow": {
        "category": "Optical Flow",
        "module": "src.preprocess.ptlflow",
        "class": "PTLFlowDetector",
        "name": "PTLFlow Optical Flow",
        "description": "PyTorch Lightning-based optical flow estimation for analyzing motion between video frames. Computes pixel-level motion vectors showing how content moves across frames. Essential for video stabilization, motion analysis, frame interpolation, and understanding dynamic scenes. Outputs visualization or raw flow data.",
        "supports_video": True,
        "supports_image": False,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "pyracanny": {
        "category": "Line",
        "module": "src.preprocess.pyracanny",
        "class": "PyraCannyDetector",
        "name": "Pyramid Canny",
        "description": "Multi-scale Canny edge detection using image pyramids to capture edges at different scales. Detects both fine details and large-scale structures in a single pass. More comprehensive than standard Canny for images with features at multiple scales. Great for complex scenes with varied detail levels.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "low_threshold", "display_name": "Low Threshold", "type": "int", "default": 100, "description": "The lower threshold for the multi-scale Canny edge detector. Edges with gradient values below this are discarded. Lower values detect more edges at multiple scales.", "min": 0, "max": 500},
            {"name": "high_threshold", "display_name": "High Threshold", "type": "int", "default": 200, "description": "The upper threshold for the multi-scale Canny edge detector. Edges with gradient values above this are kept as strong edges at all scales. Higher values detect only the most prominent edges.", "min": 0, "max": 500},
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "rembg": {
        "category": "Background Removal",
        "module": "src.preprocess.rembg",
        "class": "RembgDetector",
        "name": "Background Removal",
        "description": "Automatically removes backgrounds from images and videos using advanced segmentation. Accurately separates subjects from backgrounds with optional alpha matting for fine details like hair and fur. Supports post-processing for cleaner masks. Perfect for product photography, portraits, green screen replacement, and compositing.",
        "supports_video": True,
        "supports_image": True,
        "supports_alpha_channel": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "alpha_matting", "display_name": "Alpha Matting", "type": "bool", "default": False, "description": "When enabled, applies alpha matting for refined edge detection and smoother transparency transitions. Particularly useful for subjects with fine details like hair. Requires more processing time."},
            {"name": "alpha_matting_foreground_threshold", "display_name": "Foreground Threshold", "type": "int", "default": 240, "description": "The threshold value (0-255) for classifying pixels as definite foreground during alpha matting. Higher values are more conservative, classifying fewer pixels as foreground.", "min": 0, "max": 255},
            {"name": "alpha_matting_background_threshold", "display_name": "Background Threshold", "type": "int", "default": 10, "description": "The threshold value (0-255) for classifying pixels as definite background during alpha matting. Lower values are more conservative, classifying fewer pixels as background.", "min": 0, "max": 255},
            {"name": "alpha_matting_erode_size", "display_name": "Erosion Size", "type": "int", "default": 10, "description": "The erosion kernel size in pixels for alpha matting preprocessing. Larger values create a wider transition zone between foreground and background, resulting in smoother edges.", "min": 0, "max": 50},
            {"name": "post_process_mask", "display_name": "Post-Process Mask", "type": "bool", "default": False, "description": "When enabled, applies morphological operations to clean up the segmentation mask by removing small artifacts and smoothing boundaries."}
        ]
    },
    "recolor": {
        "category": "Color",
        "module": "src.preprocess.recolor",
        "class": "Recolorizer",
        "name": "Recolor",
        "supports_alpha_channel": True,
        "description": "Converts images to grayscale while preserving brightness structure using luminance or intensity calculation. Offers gamma correction for brightness adjustment. Useful for preparing images for colorization, creating base layers for artistic effects, or analyzing tonal distribution without color distraction.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "mode", "display_name": "Recolor Mode", "type": "category", "default": "luminance", "options": [{"name": "Luminance", "value": "luminance"}, {"name": "Intensity", "value": "intensity"}], "description": "The method for calculating brightness values. 'Luminance' uses perceptually accurate weights for RGB channels. 'Intensity' uses simple averaging of RGB values."},
            {"name": "gamma_correction", "display_name": "Gamma Correction", "type": "float", "default": 1.0, "description": "The gamma correction factor applied to the recolored output. Values below 1.0 brighten the image, values above 1.0 darken it. 1.0 applies no correction.", "min": 0.1, "max": 3.0},
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "scribble": {
        "category": "Line",
        "module": "src.preprocess.scribble",
        "class": "ScribbleDetector",
        "name": "Scribble",
        "description": "Generates simplified scribble-style edge representations from images. Converts photographs into loose, sketch-like drawings with casual, hand-drawn appearance. Perfect for creating artistic references, coloring book pages, and stylized visual effects that mimic quick sketch drawings.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "scribble_xdog": {
        "category": "Line",
        "module": "src.preprocess.scribble",
        "class": "ScribbleXDogDetector",
        "name": "Scribble XDoG",
        "description": "Extended Difference of Gaussians algorithm for creating artistic scribble effects with adjustable density. Uses advanced edge detection to produce variable-width sketch lines. Threshold control allows balancing between detailed, dense scribbles and sparse, clean lines. Great for non-photorealistic rendering and artistic stylization.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "thr_a", "display_name": "Threshold", "type": "int", "default": 32, "description": "The threshold parameter for the Extended Difference of Gaussians algorithm. Controls the sensitivity of edge detection. Lower values detect more edges and produce denser scribbles, higher values produce sparser, cleaner lines.", "min": 1, "max": 64},
        ]
    },
    "scribble_anime": {
        "category": "Line",
        "module": "src.preprocess.scribble_anime",
        "class": "ScribbleAnimeDetector",
        "name": "Scribble Anime",
        "description": "Neural network-based scribble generation optimized for anime and manga aesthetics. Produces loose, sketchy lines that match anime rough draft style. Trained on anime artwork to understand character features and composition. Ideal for anime sketch effects, animation pre-production references, and manga thumbnails.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "shuffle": {
        "category": "Color",
        "module": "src.preprocess.shuffle",
        "class": "ContentShuffleDetector",
        "name": "Content Shuffle",
        "description": "Spatially shuffles image content using frequency-based noise patterns. Rearranges colors and textures while maintaining overall composition structure. Creates abstract, glitch-art effects. Useful for data augmentation, creative distortion effects, and generating variations while preserving color distribution. Controllable with seed for reproducibility.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "h", "display_name": "Height", "type": "int", "default": None, "description": "The height dimension for noise generation. Leave as default (None) to use the image's original height. Custom values allow for different shuffle patterns.", "min": 64, "max": 2048},
            {"name": "w", "display_name": "Width", "type": "int", "default": None, "description": "The width dimension for noise generation. Leave as default (None) to use the image's original width. Custom values allow for different shuffle patterns.", "min": 64, "max": 2048},
            {"name": "f", "display_name": "Frequency", "type": "int", "default": 256, "description": "The frequency parameter controlling the scale of content shuffling. Higher values create finer, more detailed shuffling patterns, while lower values create broader, blockier patterns.", "min": 1, "max": 512},
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "seed", "display_name": "Random Seed", "type": "int", "default": -1, "description": "The random seed for reproducible shuffle patterns. Set to -1 for a random shuffle each time, or use a specific value to get the same shuffle pattern consistently.", "min": -1, "max": 2147483647}
        ]
    },
    "teed": {
        "category": "Line",
        "module": "src.preprocess.teed",
        "class": "TEDDetector",
        "name": "TEED Edge Detection",
        "description": "Tiny and Efficient Edge Detector designed for speed without sacrificing quality. Lightweight neural network optimized for real-time edge detection. Iterative refinement with safe steps produces progressively cleaner edges. Excellent for mobile devices, video processing, and applications requiring fast edge detection.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "safe_steps", "display_name": "Safe Steps", "type": "int", "default": 2, "description": "The number of refinement iterations applied during edge detection. More steps produce cleaner, more refined edges with reduced noise, but increase processing time. Typical range is 1-5 steps.", "min": 1, "max": 10},
        ]
    },
    "tile": {
        "category": "Color",
        "module": "src.preprocess.tile",
        "class": "TileDetector",
        "name": "Tile Resample",
        "description": "Creates smooth color transitions using pyramid downsampling and upsampling. Repeatedly reduces then enlarges the image to blur fine details while preserving major color regions. Produces painterly, posterized effects with soft color gradients. Great for simplifying images, creating base layers, and stylized color blocking.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "pyrUp_iters", "display_name": "Pyramid Up Iterations", "type": "int", "default": 3, "description": "The number of pyramid upsampling iterations. More iterations create stronger blur and smoother color transitions. Each iteration doubles the scale, so 3 iterations provides 8x downsampling before upsampling back.", "min": 1, "max": 5},
            upscale_method_parameter,
        ]
    },
    "tile_gf": {
        "category": "Color",
        "module": "src.preprocess.tile",
        "class": "TTPlanet_Tile_Detector_GF",
        "name": "Tile Guided Filter",
        "description": "Advanced tile processing using guided filter for edge-preserving smoothing. Combines downsampling with intelligent filtering that maintains important edges while blurring textures. Better edge preservation than simple tile methods. Ideal for stylization that needs to keep subject boundaries sharp while simplifying detail.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "scale_factor", "display_name": "Scale Factor", "type": "float", "default": 2.0, "description": "The downscaling factor before processing. Higher values create more aggressive downsampling and stronger simplification effects. A value of 2.0 reduces dimensions by half.", "min": 1.0, "max": 8.0},
            {"name": "blur_strength", "display_name": "Blur Strength", "type": "float", "default": 1.0, "description": "The strength of Gaussian blur applied during preprocessing. Higher values create smoother, more blurred results. 1.0 is standard blur strength.", "min": 0.1, "max": 5.0},
            {"name": "radius", "display_name": "Filter Radius", "type": "int", "default": 3, "description": "The radius of the guided filter kernel in pixels. Larger radius values preserve more edge structure while smoothing. Typical values are 1-10.", "min": 1, "max": 20},
            {"name": "eps", "display_name": "Filter Epsilon", "type": "float", "default": 0.01, "description": "The regularization parameter for the guided filter. Controls edge preservation versus smoothing. Lower values preserve more edges, higher values create smoother results.", "min": 0.001, "max": 1.0}
        ]
    },
    "tile_simple": {
        "category": "Color",
        "module": "src.preprocess.tile",
        "class": "TTPLanet_Tile_Detector_Simple",
        "name": "Tile Simple",
        "description": "Straightforward tile effect using downscaling and Gaussian blur. Simple approach for creating blocky, posterized color effects. Fast processing with basic color simplification. Good for quick stylization, reducing image complexity, and creating retro pixel-art inspired looks.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            {"name": "scale_factor", "display_name": "Scale Factor", "type": "float", "default": 2.0, "description": "The downscaling factor for image reduction. Higher values create more aggressive downsampling. A value of 2.0 reduces width and height by half. Useful for creating simplified, blocky tile effects.", "min": 1.0, "max": 8.0},
            {"name": "blur_strength", "display_name": "Blur Strength", "type": "float", "default": 1.0, "description": "The intensity of Gaussian blur applied to the image. Higher values create stronger blur and smoother transitions between colors. 1.0 is the standard blur strength.", "min": 0.1, "max": 5.0}
        ]
    },
    "uniformer": {
        "category": "Segmentation",
        "module": "src.preprocess.uniformer",
        "class": "UniformerSegmentor",
        "name": "Uniformer Segmentation",
        "description": "Unified transformer architecture for semantic segmentation with both local and global attention. Accurately segments scenes into semantic categories like sky, road, buildings, vegetation, etc. Excellent balance of accuracy and efficiency. Ideal for scene parsing, autonomous driving visualization, and understanding spatial layout.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "unimatch": {
        "category": "Optical Flow",
        "module": "src.preprocess.unimatch",
        "class": "UnimatchDetector",
        "name": "UniMatch Optical Flow",
        "description": "Unified correspondence matching framework for accurate optical flow with forward, backward, and bidirectional options. State-of-the-art flow estimation handling occlusions and large motions. Supports multiple flow directions for comprehensive motion analysis. Perfect for advanced video effects, motion compensation, and temporal consistency.",
        "supports_video": True,
        "supports_image": False,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
            {"name": "pred_bwd_flow", "display_name": "Predict Backward Flow", "type": "bool", "default": False, "description": "When enabled, computes optical flow from the current frame to the previous frame (backward in time). Useful for motion analysis and tracking objects moving backward through the sequence."},
            {"name": "pred_bidir_flow", "display_name": "Predict Bidirectional Flow", "type": "bool", "default": False, "description": "When enabled, computes optical flow in both forward and backward directions simultaneously. Provides complete motion information but requires more processing time and memory."}
        ]
    },
    "zoe": {
        "category": "Depth and Normal",
        "module": "src.preprocess.zoe.transformers",
        "class": "ZoeDetector",
        "name": "ZoeDepth",
        "description": "Zero-shot metric depth estimation combining relative depth prediction with metric scale recovery. Provides metric depth without requiring camera calibration in many cases. Works across indoor and outdoor scenes with good generalization. Excellent for applications needing real-world depth measurements like robotics and AR.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
        ]
    },
    "zoe_depth_anything": {
        "category": "Depth and Normal",
        "module": "src.preprocess.zoe.transformers",
        "class": "ZoeDepthAnythingDetector",
        "name": "ZoeDepth Anything",
        "description": "Combines ZoeDepth's metric depth capabilities with Depth Anything's robust feature extraction. Best of both worlds: metric scale accuracy with superior generalization across diverse scenes. Handles challenging conditions better than standard ZoeDepth. Ideal when you need both metric accuracy and reliability across varied content.",
        "supports_video": True,
        "supports_image": True,
        "parameters": [
            detect_resolution_parameter,
            upscale_method_parameter,
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
    return _load_preprocessor_yaml(preprocessor_name)


def list_preprocessors(check_downloaded: bool = False) -> List[Dict[str, Any]]:
    """
    List all available preprocessors with their metadata.
    
    Args:
        check_downloaded: If True, check download status for each preprocessor (slower)
    
    Returns:
        List of preprocessor information dictionaries
    """
    result: List[Dict[str, Any]] = []
    for name in _available_preprocessor_names():
        info = _load_preprocessor_yaml(name)
        files = info.get("files", [])
    # Resolve absolute paths if files exist under DEFAULT_PREPROCESSOR_SAVE_PATH
        try:
            from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
            base = Path(DEFAULT_PREPROCESSOR_SAVE_PATH)
            resolved_files: List[Dict[str, Any]] = []
            for f in files:
                rel_path = f.get("path", "")
                abs_path = base / rel_path
                if abs_path.exists():
                    # prefer absolute path when downloaded
                    resolved_files.append({"path": abs_path.__str__(), "size_bytes": f.get("size_bytes")})
                else:
                    resolved_files.append(f)
        except Exception:
            resolved_files = files
        preprocessor_info = {
            "id": name,
            "name": info.get("name", name),
            "category": info.get("category", ""),
            "description": info.get("description", ""),
            "supports_video": bool(info.get("supports_video", True)),
            "supports_image": bool(info.get("supports_image", True)),
            "parameters": info.get("parameters", []),
            "files": resolved_files
        }
        if check_downloaded:
            preprocessor_info["is_downloaded"] = check_preprocessor_downloaded(name)
        result.append(preprocessor_info)
    return sorted(result, key=lambda x: x["name"])


def initialize_download_tracking():
    """
    Initialize the download tracking file with preprocessors that don't require downloads.
    This should be called on app startup.
    """
    from src.preprocess.base_preprocessor import BasePreprocessor
    
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
    from src.preprocess.base_preprocessor import BasePreprocessor
    return BasePreprocessor._is_downloaded(preprocessor_name)


def get_preprocessor_details(preprocessor_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific preprocessor.
    
    Args:
        preprocessor_name: Name of the preprocessor
        
    Returns:
        Dictionary with detailed preprocessor information including parameters
    """
    info = _load_preprocessor_yaml(preprocessor_name)
    is_downloaded = check_preprocessor_downloaded(preprocessor_name)
    files = info.get("files", [])
    # Resolve absolute paths if files exist under DEFAULT_PREPROCESSOR_SAVE_PATH
    try:
        from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
        base = Path(DEFAULT_PREPROCESSOR_SAVE_PATH)
        resolved_files: List[Dict[str, Any]] = []
        for f in files:
            rel_path = f.get("path", "")
            abs_path = base / rel_path
            if abs_path.exists():
                # prefer absolute path when downloaded
                resolved_files.append({"path": abs_path.__str__(), "size_bytes": f.get("size_bytes"), "name": f.get("name")})
            else:
                resolved_files.append(f)
    except Exception:
        resolved_files = files

    return {
        "id": preprocessor_name,
        "name": info.get("name", preprocessor_name),
        "category": info.get("category", ""),
        "description": info.get("description", ""),
        "module": info.get("module", ""),
        "class": info.get("class", ""),
        "supports_video": bool(info.get("supports_video", True)),
        "supports_image": bool(info.get("supports_image", True)),
        "parameters": info.get("parameters", []),
        "is_downloaded": is_downloaded,
        "files": resolved_files
    }

