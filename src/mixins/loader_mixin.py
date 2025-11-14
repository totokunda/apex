from src.utils.defaults import DEFAULT_HEADERS
from urllib.parse import urlparse
import requests
from pathlib import Path
from typing import Dict, Any, Union
import os
import json
import yaml
from diffusers import ModelMixin
from accelerate import init_empty_weights
import torch
from typing import Callable
from logging import Logger
from src.utils.module import find_class_recursive
import importlib
import inspect
from loguru import logger
from src.utils.yaml import LoaderWithInclude
from src.manifest.loader import validate_and_normalize
from PIL import Image
from io import BytesIO
import numpy as np
import PIL
from typing import List
import cv2
import tempfile
from glob import glob
from transformers.modeling_utils import PreTrainedModel
from src.quantize.ggml_layer import patch_model
from src.quantize.load import load_gguf, dequantize_tensor
from src.mixins.download_mixin import DownloadMixin
from src.converters.convert import get_transformer_converter

# Import pretrained config from transformers
from transformers.configuration_utils import PretrainedConfig
from src.utils.safetensors import is_safetensors_file, load_safetensors
import mlx.core as mx
from src.utils.mlx import check_mlx_convolutional_weights
from src.types import InputImage, InputVideo

ACCEPTABLE_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IMAGE_EXTS = [
  'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'ico', 'webp',
]
VIDEO_EXTS = [
  'mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv', 'mpg', 'mpeg', 'm4v',
]

class LoaderMixin(DownloadMixin):
    logger: Logger = logger

    def _load_model(
        self,
        component: Dict[str, Any],
        getter_fn: Callable | None = None,
        module_name: str = "diffusers",
        load_dtype: torch.dtype | mx.Dtype | None = None,
        no_weights: bool = False,
        key_map: Dict[str, str] | None = None,
        extra_kwargs: Dict[str, Any] | None = None,
    ) -> ModelMixin:

        if extra_kwargs is None:
            extra_kwargs = {}

        model_base = component.get("base")
        model_path = component.get("model_path")


        if getter_fn:
            model_class = getter_fn(model_base)
        else:
            model_class = find_class_recursive(
                importlib.import_module(module_name), model_base
            )
        if model_class is None:
            raise ValueError(f"Model class for base '{model_base}' not found")

        config_path = component.get("config_path")
        config = {}
        if config_path:
            config.update(self.fetch_config(config_path))
        if component.get("config"):
            config.update(component.get("config"))

        if (
            not config
            or (
                os.path.isdir(model_path)
                and os.path.exists(os.path.join(model_path, "config.json"))
            )
        ) and not component.get("extra_model_paths"):
            if config:
                # replace the config.json with the config
                config_path = os.path.join(model_path, "config.json")
                self._save_config(config, config_path)

            if (
                hasattr(self, "engine_type")
                and self.engine_type == "mlx"
                and component.get("type") == "transformer"
            ) and load_dtype is not None:
                extra_kwargs["dtype"] = load_dtype
            elif load_dtype is not None:
                extra_kwargs["torch_dtype"] = load_dtype
                
            self.logger.info(f"Loading {model_class} from {model_path}")
            model = model_class.from_pretrained(model_path, **extra_kwargs)
            return model

        with init_empty_weights():
            # Check the constructor signature to determine what it expects
            sig = inspect.signature(model_class.__init__)
            params = list(sig.parameters.values())

            # Skip 'self' parameter
            if params and params[0].name == "self":
                params = params[1:]

            # Check if the first parameter expects a PretrainedConfig object
            expects_pretrained_config = False
            if params:
                first_param = params[0]
                if (
                    first_param.annotation == PretrainedConfig
                    or (
                        hasattr(first_param.annotation, "__name__")
                        and "Config" in first_param.annotation.__name__
                    )
                    or first_param.name in ["config"]
                    and issubclass(model_class, PreTrainedModel)
                ):
                    expects_pretrained_config = True

            if expects_pretrained_config:
                # Use the model's specific config class if available, otherwise fall back to PretrainedConfig
                config_class = getattr(model_class, "config_class", PretrainedConfig)
                conf = config_class(**config)
                if hasattr(model_class, "_from_config"):
                    model = model_class._from_config(conf, **extra_kwargs)
                else:
                    model = model_class.from_config(conf, **extra_kwargs)
            else:
                if hasattr(model_class, "_from_config"):
                    model = model_class._from_config(config, **extra_kwargs)
                else:
                    model = model_class.from_config(config, **extra_kwargs)

        if no_weights:
            return model
    
        if (
            model_path.endswith(".gguf")
            and hasattr(self, "engine_type")
            and self.engine_type == "mlx"
        ):

            self.logger.info(f"Loading GGUF model from {model_path}")
            # Can load gguf directly into mlx model no need to convert
            gguf_weights = mx.load(model_path)
            check_mlx_convolutional_weights(gguf_weights, model)
            model.load_weights(gguf_weights)
        
        elif (
            model_path.endswith(".gguf")
        ):
            logger.info(f"Loading GGUF model from {model_path}")
            gguf_kwargs = component.get("gguf_kwargs", {})
            logger.info(f"\n\n gguf_kwargs: {gguf_kwargs}\n\n")
            state_dict, _ = load_gguf(
                model_path, type=component.get("type"), **gguf_kwargs
            )
            # check if we need to convert the weights
            if component.get("type") == "transformer":
                converter = get_transformer_converter(model_base)
                converter.convert(state_dict)

            # Load GGMLTensors without replacing nn.Parameters by copying data
            patch_model(model)
            model.load_state_dict(state_dict, assign=True)
        else:
            if os.path.isdir(model_path):
                extensions = component.get(
                    "extensions", ["safetensors", "bin", "pt", "ckpt"]
                )
                self.logger.info(f"Loading model from {model_path}")
                files_to_load = []
                for ext in extensions:
                    files_to_load.extend(glob(os.path.join(model_path, f"*.{ext}")))
                if not files_to_load:
                    self.logger.warning(f"No model files found in {model_path}")
            else:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                files_to_load = [model_path]

            extra_model_paths = component.get("extra_model_paths", [])
            if isinstance(extra_model_paths, str):
                extra_model_paths = [extra_model_paths]
            files_to_load.extend(extra_model_paths)

            for file_path in files_to_load:
                self.logger.info(f"Loading weights from {file_path}")
                if is_safetensors_file(file_path):
                    state_dict = load_safetensors(
                        file_path,
                        dtype=load_dtype,
                        framework="np" if self.engine_type == "mlx" else "pt",
                    )
                else:
                    state_dict = torch.load(
                        file_path, map_location="cpu", weights_only=True, mmap=True
                    )
                    if load_dtype:
                        for k, v in state_dict.items():
                            state_dict[k] = v.to(load_dtype)
                # remap keys if key_map is provided replace part of existing key with new key
                if key_map:
                    new_state_dict = {}
                    for k, v in key_map.items():
                        for k2, v2 in state_dict.items():
                            if k in k2:
                                new_state_dict[k2.replace(k, v)] = v2
                            else:
                                new_state_dict[k2] = v2

                    state_dict = new_state_dict

                if hasattr(self, "engine_type") and self.engine_type == "mlx":
                    check_mlx_convolutional_weights(state_dict, model)

                if hasattr(model, "load_state_dict"):
                    model.load_state_dict(
                        state_dict, strict=False, assign=True
                    )  # must be false as we are iteratively loading the state dict
                elif hasattr(model, "load_weights"):
                    model.load_weights(state_dict, strict=False)
                else:
                    raise ValueError(
                        f"Model {model} does not have a load_state_dict or load_weights method"
                    )

        if hasattr(self, "engine_type") and self.engine_type == "torch":
            has_meta_params = False
            for name, param in model.named_parameters():
                if param.device.type == "meta":
                    self.logger.error(f"Parameter {name} is on meta device")
                    has_meta_params = True

            if has_meta_params:
                raise ValueError(
                    "Model has parameters on meta device, this is not supported"
                )

        return model

    def _load_config_file(self, file_path: str | Path):
        try:
            return self._load_json(file_path)
        except json.JSONDecodeError:
            return self._load_yaml(file_path)
        except Exception:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _load_json(self, file_path: str | Path):
        with open(file_path, "r") as f:
            return json.load(f)

    def _load_yaml(self, file_path: str | Path):
        file_path = Path(file_path)
        text = file_path.read_text()

        # --- PASS 1: extract `shared:` (legacy) or `spec.shared` (v1) with a loader that skips !include tags ---
        prelim = yaml.load(text, Loader=yaml.FullLoader)
        # Collect shared entries from both legacy and v1 shapes
        shared_entries = []
        if isinstance(prelim, dict):
            shared_entries.extend(prelim.get("shared", []) or [])
            spec = prelim.get("spec", {}) or {}
            if isinstance(spec, dict):
                shared_entries.extend(spec.get("shared", []) or [])

        # build alias → manifest Path
        shared_manifests = {}
        for entry in shared_entries:
            p = (file_path.parent / entry).resolve()
            # assume e.g. "shared_wan.yml" → alias "wan"
            try:
                alias = p.stem.split("_", 1)[1]
            except Exception:
                alias = p.stem
            shared_manifests[alias] = p

        # attach it to our custom loader
        LoaderWithInclude.shared_manifests = shared_manifests

        # --- PASS 2: real load with !include expansion ---
        loaded = yaml.load(text, Loader=LoaderWithInclude)

        # Validate and normalize if this is a v1 manifest
        try:
            loaded = validate_and_normalize(loaded)
        except Exception as e:
            raise

        return loaded

    def _load_scheduler(self, component: Dict[str, Any]) -> Any:
        component_base = component.get("base")
        if not component_base:
            raise ValueError("Component base not specified.")

        component_split = component_base.split(".")
        if len(component_split) > 1:
            module_name = ".".join(component_split[:-1])
            class_name = component_split[-1]
        else:
            module_name = "diffusers"
            class_name = component_base

        try:
            base_module = importlib.import_module(module_name)
        except ImportError:
            raise ImportError(
                f"Could not import the base module '{module_name}' from component '{component_base}'"
            )

        component_class = find_class_recursive(base_module, class_name)

        if component_class is None:
            raise ValueError(
                f"Could not find scheduler class '{class_name}' in module '{module_name}' or its submodules."
            )

        config_path = component.get("config_path")
        config = component.get("config")
        if config_path and config:
            fetched_config = self.fetch_config(config_path)
            config = {**fetched_config, **config}
        elif config_path:
            config = self.fetch_config(config_path)
        else:
            config = component.get("config", {})

        # Determine which config entries can be passed to the component constructor
        try:
            init_signature = inspect.signature(component_class.__init__)
            init_params = list(init_signature.parameters.values())
            if init_params and init_params[0].name == "self":
                init_params = init_params[1:]
            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params
            )
            init_param_names = {p.name for p in init_params}
        except (TypeError, ValueError):
            # Fallback if signature introspection fails
            accepts_var_kwargs = True
            init_param_names = set()

        if accepts_var_kwargs:
            init_kwargs = dict(config)
            config_to_register = {}
        else:
            init_kwargs = {
                k: v for k, v in (config or {}).items() if k in init_param_names
            }
            config_to_register = {
                k: v for k, v in (config or {}).items() if k not in init_param_names
            }

        component = component_class(**init_kwargs)

        # Register remaining config to the component if supported
        if (
            config_to_register
            and hasattr(component, "register_to_config")
            and callable(getattr(component, "register_to_config"))
        ):
            try:
                component.register_to_config(**config_to_register)
            except Exception as e:
                self.logger.warning(
                    f"Failed to register extra config for {component_class}: {e}"
                )

        return component

    def save_component(
        self,
        component: Any,
        model_path: str,
        component_type: str,
        **save_kwargs: Dict[str, Any],
    ):
        if component_type == "transformer":
            if issubclass(type(component), ModelMixin):
                component.save_pretrained(model_path, **save_kwargs)
            else:
                raise ValueError(f"Unsupported component type: {component_type}")
        elif component_type == "vae":
            if issubclass(type(component), ModelMixin):
                component.save_pretrained(model_path, **save_kwargs)
            else:
                raise ValueError(f"Unsupported component type: {component_type}")
        else:
            raise ValueError(f"Unsupported component type: {component_type}")

    def _load_image(
        self,
        image: "InputImage",
        convert_method: Callable[[Image.Image], Image.Image] | None = None,
    ) -> Image.Image:
        if isinstance(image, Image.Image):
            out_image = image
        elif isinstance(image, str):
            if self._is_url(image):
                out_image = Image.open(
                    BytesIO(
                        requests.get(image, timeout=10, headers=DEFAULT_HEADERS).content
                    )
                )
            else:
                out_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            out_image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            out_image = Image.fromarray(image.numpy())
        else:
            raise ValueError(f"Invalid image type: {type(image)}")

        out_image = PIL.ImageOps.exif_transpose(out_image)
        if convert_method is not None:
            out_image = convert_method(out_image)
        else:
            out_image = out_image.convert("RGB")
        return out_image

    def _load_video(
        self,
        video_input: "InputVideo",
        fps: int | None = None,
        return_fps: bool = False,
        convert_method: Callable[[Image.Image], Image.Image] | None = None,
    ) -> List[Image.Image]:

        if isinstance(video_input, List):
            out_frames = []
            for v in video_input:
                out_frames.append(self._load_image(v, convert_method=convert_method))
            if return_fps:
                return out_frames, fps
            else:
                return out_frames

        if isinstance(video_input, str):
            video_path = video_input
            tmp_file_path = None

            if self._is_url(video_input):
                try:
                    response = requests.get(
                        video_input, timeout=10, headers=DEFAULT_HEADERS
                    )
                    response.raise_for_status()
                    contents = response.content
                    suffix = Path(urlparse(video_input).path).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp_file:
                        tmp_file.write(contents)
                        tmp_file_path = tmp_file.name
                    video_path = tmp_file_path
                except requests.RequestException as e:
                    if tmp_file_path and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                    raise IOError(f"Failed to download video from {video_input}") from e
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file: {video_path}")

                original_fps = cap.get(cv2.CAP_PROP_FPS)
                
                frames = []
                frame_count = 0
                
                if fps is None:
                    # No fps specified, extract all frames
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Check if frame is grayscale or color
                        if len(frame.shape) == 2 or (
                            len(frame.shape) == 3 and frame.shape[2] == 1
                        ):
                            # Grayscale frame, no color conversion needed
                            frame_rgb = frame
                        else:
                            # Color frame, convert from BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(
                            convert_method(Image.fromarray(frame_rgb))
                            if convert_method
                            else Image.fromarray(frame_rgb)
                        )
                        frame_count += 1
                else:
                    # Extract frames at specified fps using time-based sampling
                    frame_interval = original_fps / fps  # frames between each sample
                    next_frame_time = 0.0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Check if current frame should be extracted based on timing
                        if frame_count >= next_frame_time:
                            # Check if frame is grayscale or color
                            if len(frame.shape) == 2 or (
                                len(frame.shape) == 3 and frame.shape[2] == 1
                            ):
                                # Grayscale frame, no color conversion needed
                                frame_rgb = frame
                            else:
                                # Color frame, convert from BGR to RGB
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(
                                convert_method(Image.fromarray(frame_rgb))
                                if convert_method
                                else Image.fromarray(frame_rgb)
                            )
                            next_frame_time += frame_interval
                        
                        frame_count += 1
                if return_fps:
                    return frames, original_fps
                else:
                    return frames
            finally:
                if "cap" in locals() and cap.isOpened():
                    cap.release()
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        if isinstance(video_input, np.ndarray):
            return [
                (
                    convert_method(Image.fromarray(frame))
                    if convert_method
                    else Image.fromarray(frame).convert("RGB")
                )
                for frame in video_input
            ]

        if isinstance(video_input, torch.Tensor):
            tensor = video_input.cpu()
            if tensor.ndim == 5:
                if tensor.shape[1] == 3 or tensor.shape[1] == 1:
                    # Means shape is (B, C, F, H, W)
                    tensor = tensor.permute(0, 2, 1, 3, 4).squeeze(0)
                    frames = []
                elif tensor.shape[2] == 1 or tensor.shape[2] == 3:
                    # Means shape is (B, C, F, H, W)
                    tensor = tensor.squeeze(0)
                    frames = []
                else:
                    raise ValueError(f"Invalid tensor shape: {tensor.shape}")

                for frame in tensor:
                    frame = frame.permute(1, 2, 0).numpy()
                    # check if frame is between 0 and 1
                    if frame.mean() <= 1:
                        frame = (frame * 255).clip(0, 255).astype(np.uint8)
                    # check if frame is grayscale if so then don't convert to RGB
                    if frame.shape[2] == 1:
                        frames.append(
                            convert_method(Image.fromarray(frame.squeeze(2)))
                            if convert_method
                            else Image.fromarray(frame.squeeze(2))
                        )
                    else:
                        frames.append(
                            convert_method(Image.fromarray(frame))
                            if convert_method
                            else Image.fromarray(frame).convert("RGB")
                        )
                if return_fps:
                    return frames, fps
                else:
                    return frames

            if tensor.ndim == 4 and (
                tensor.shape[1] == 3 or tensor.shape[1] == 1
            ):  # NCHW to NHWC
                tensor = tensor.permute(0, 2, 3, 1)

            numpy_array = tensor.numpy()
            if numpy_array.mean() <= 1:
                numpy_array = (numpy_array * 255).clip(0, 255).astype(np.uint8)

            frames = [
                (
                    convert_method(Image.fromarray(frame))
                    if convert_method
                    else (
                        Image.fromarray(frame).convert("RGB")
                        if frame.shape[2] == 3
                        else (
                            convert_method(Image.fromarray(frame))
                            if convert_method
                            else Image.fromarray(frame)
                        )
                    )
                )
                for frame in numpy_array
            ]
            if return_fps:
                return frames, fps
            else:
                return frames
        raise ValueError(f"Invalid video type: {type(video_input)}")

    @staticmethod
    def get_media_type(media_path: str) -> str:
        media_path = media_path.lower()
        if media_path.endswith(tuple(VIDEO_EXTS)):
            return "video"
        elif media_path.endswith(tuple(IMAGE_EXTS)):
            return "image"
        else:
            raise ValueError(f"Invalid media type: {media_path}")