from src.utils.defaults import DEFAULT_HEADERS, DEFAULT_CONFIG_SAVE_PATH
from urllib.parse import urlparse
import requests
from pathlib import Path
from typing import Dict, Any, Union
import os
import json
import yaml
from src.utils.cache import empty_cache
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
from PIL import Image
from io import BytesIO
import numpy as np
import PIL
from typing import List
import cv2
import tempfile
from glob import glob
from transformers.modeling_utils import PreTrainedModel
from src.quantize.ggml_layer import patch_module
from src.quantize.load import load_gguf
from src.mixins.download_mixin import DownloadMixin

# Import pretrained config from transformers
from transformers.configuration_utils import PretrainedConfig
from src.utils.torch import is_safetensors_file, load_safetensors

ACCEPTABLE_DTYPES = [torch.float16, torch.float32, torch.bfloat16]


class LoaderMixin(DownloadMixin):
    logger: Logger = logger

    def _load_model(
        self,
        component: Dict[str, Any],
        getter_fn: Callable | None = None,
        module_name: str = "diffusers",
        load_dtype: torch.dtype | None = None,
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
        if config_path:
            config = self.fetch_config(config_path)
        else:
            config = component.get("config", {}) or {}

        if os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "config.json")
        ):
            if config:
                # replace the config.json with the config
                config_path = os.path.join(model_path, "config.json")
                self._save_config(config, config_path)
            model = model_class.from_pretrained(
                model_path, torch_dtype=load_dtype, **extra_kwargs
            )
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
                model = model_class(conf, **extra_kwargs)
            else:
                model = model_class(**config, **extra_kwargs)

        if no_weights:
            return model

        if model_path.endswith(".gguf"):
            self.logger.info(f"Loading GGUF model from {model_path}")
            gguf_kwargs = component.get("gguf_kwargs", {})
            state_dict, qtype_dict = load_gguf(
                model_path, type=component.get("type"), **gguf_kwargs
            )
            patch_module(model)
            model.load_state_dict(state_dict, assign=True)

        else:
            if os.path.isdir(model_path):
                self.logger.info(f"Loading model from {model_path}")
                file_pattern = component.get("file_pattern", "**/*.safetensors")
                bin_pattern = component.get("bin_pattern", "**/*.bin")
                pt_pattern = component.get("pt_pattern", "**/*.pt")
                files_to_load = glob(
                    os.path.join(model_path, file_pattern), recursive=True
                )
                files_to_load += glob(
                    os.path.join(model_path, bin_pattern), recursive=True
                )
                files_to_load += glob(
                    os.path.join(model_path, pt_pattern), recursive=True
                )
                if not files_to_load:
                    self.logger.warning(f"No model files found in {model_path}")
            else:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                files_to_load = [model_path]
            for file_path in files_to_load:
                self.logger.info(f"Loading weights from {file_path}")
                if is_safetensors_file(file_path):
                    state_dict = load_safetensors(file_path, dtype=load_dtype)
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

                model.load_state_dict(
                    state_dict, strict=False, assign=True
                )  # must be false as we are iteratively loading the state dict
            # Assert no parameters are on meta device

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

        # --- PASS 1: extract your `shared:` list with a loader that skips !include tags ---
        prelim = yaml.load(text, Loader=yaml.FullLoader)
        # prelim.get("shared", [...]) is now a list of file-paths strings.

        # build alias → manifest Path
        shared_manifests = {}
        for entry in prelim.get("shared", []):
            p = (file_path.parent / entry).resolve()
            # assume e.g. "shared_wan.yml" → alias "wan"
            alias = p.stem.split("_", 1)[1]
            shared_manifests[alias] = p

        # attach it to our custom loader
        LoaderWithInclude.shared_manifests = shared_manifests

        # --- PASS 2: real load with !include expansion ---
        return yaml.load(text, Loader=LoaderWithInclude)

    def _load_scheduler(self, component: Dict[str, Any]) -> Any:
        component_base = component.get("base")
        if not component_base:
            raise ValueError("Component base not specified.")

        component_split = component_base.split(".")
        if len(component_split) > 1:
            module_name = component_split[0]
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
        image: Union[Image.Image, str, np.ndarray, torch.Tensor],
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
        video_input: Union[str, List[str], np.ndarray, torch.Tensor, List[Image.Image]],
        fps: int | None = None,
        return_fps: bool = False,
        convert_method: Callable[[Image.Image], Image.Image] | None = None,
    ) -> List[Image.Image]:

        if isinstance(video_input, List):
            if not video_input:
                if return_fps:
                    return video_input, fps
                else:
                    return video_input
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
                frame_skip = 1 if fps is None else max(1, int(original_fps // fps))

                frames = []
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % frame_skip == 0:
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
