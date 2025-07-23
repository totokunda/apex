from src.utils.defaults import DEFAULT_HEADERS, DEFAULT_CONFIG_SAVE_PATH
from urllib.parse import urlparse
import requests
from pathlib import Path
from typing import Dict, Any, Union
import os
import json
import yaml
import hashlib
from src.utils.cache_utils import empty_cache
from diffusers import ModelMixin
from src.utils.load_utils import load_safetensors
from accelerate import init_empty_weights
import torch
from typing import Callable
from logging import Logger
from src.utils.module_utils import find_class_recursive
import importlib
import inspect
from loguru import logger
from src.utils.yaml_utils import LoaderWithInclude
from PIL import Image
from io import BytesIO
import numpy as np
import PIL
from typing import List
import cv2
import tempfile
from glob import glob
import safetensors
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers.modeling_utils import PreTrainedModel

# Import pretrained config from transformers
from transformers.configuration_utils import PretrainedConfig


ACCEPTABLE_DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def is_safetensors_file(file_path: str):
    try:
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
            f.keys()
        return True
    except Exception:
        return False


class LoaderMixin:
    logger: Logger = logger

    def _is_url(self, url: str):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _fetch_url_config(self, url: str):
        response = requests.get(url, timeout=10, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        if url.endswith((".yaml", ".json")):
            return response.json()
        return response.content

    def fetch_config(self, config_path: str):
        if self._is_url(config_path):
            config_path = self._check_config_for_url(config_path)
            if config_path:
                return self._load_config_file(config_path)
        else:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            return self._load_config_file(path)

    def _save_config(self, config: Dict[str, Any], save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".json"):
            with open(save_path, "w") as f:
                json.dump(config, f)
        elif save_path.endswith(".yaml"):
            with open(save_path, "w") as f:
                yaml.dump(config, f)
        else:
            raise ValueError(f"Unsupported config file type: {save_path}")
        return save_path

    def _check_config_for_url(self, url: str):
        if hasattr(self, "config_save_path"):
            config_save_path = self.config_save_path
        else:
            config_save_path = DEFAULT_CONFIG_SAVE_PATH

        if self._is_url(url) and config_save_path:
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            suffix = url.split(".")[-1]
            config_path = os.path.join(config_save_path, f"{url_hash}.{suffix}")
            if os.path.exists(config_path):
                return config_path
            config = self._fetch_url_config(url)
            self._save_config(config, config_path)
            return config_path
        return None

    def _load_model(
        self,
        component: Dict[str, Any],
        getter_fn: Callable | None = None,
        module_name: str = "diffusers",
        load_dtype: torch.dtype | None = None,
        no_weights: bool = False,
    ) -> ModelMixin:
        model_base = component.get("base")
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

        model_path = component.get("model_path")

        if not config:
            # try to load from model_path directly
            model = model_class.from_pretrained(model_path, torch_dtype=load_dtype)
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
                model = model_class(conf)
            else:
                model = model_class(**config)

        if no_weights:
            return model

        if os.path.isdir(model_path):
            self.logger.info(f"Loading model from {model_path}")
            file_pattern = component.get("file_pattern", "*.safetensors")
            files_to_load = glob(os.path.join(model_path, file_pattern))
            if not files_to_load:
                self.logger.warning(
                    f"No model files found in {model_path} with pattern {file_pattern}"
                )
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
            model.load_state_dict(
                state_dict, strict=False, assign=True
            )  # must be false as we are iteratively loading the state dict

        # Assert no parameters are on meta device
        for name, param in model.named_parameters():
            if param.device.type == "meta":
                raise ValueError(f"Parameter {name} is on meta device")

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

    def _load_component(self, component: Dict[str, Any]) -> Any:
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

        component = component_class(**config)

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
    ) -> List[Image.Image]:
        if isinstance(video_input, List):
            if (
                not video_input
                or isinstance(video_input[0], Image.Image)
                or isinstance(video_input[0], str)
            ):
                return video_input
            return [self._load_image(v) for v in video_input]

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

                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                return frames
            finally:
                if "cap" in locals() and cap.isOpened():
                    cap.release()
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        if isinstance(video_input, np.ndarray):
            return [Image.fromarray(frame).convert("RGB") for frame in video_input]

        if isinstance(video_input, torch.Tensor):
            tensor = video_input.cpu()
            if tensor.ndim == 4 and tensor.shape[1] == 3:  # NCHW to NHWC
                tensor = tensor.permute(0, 2, 3, 1)

            numpy_array = tensor.numpy()
            if numpy_array.dtype in [np.float16, np.float32, np.float64]:
                numpy_array = (numpy_array * 255).clip(0, 255).astype(np.uint8)

            return [Image.fromarray(frame).convert("RGB") for frame in numpy_array]

        raise ValueError(f"Invalid video type: {type(video_input)}")
