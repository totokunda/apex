import os
from typing import List, Dict, Any, Optional, Literal
from diffusers.utils.dummy_pt_objects import SchedulerMixin
import torch
from loguru import logger
import urllib3
from diffusers.models.modeling_utils import ModelMixin
import re
from contextlib import contextmanager
from tqdm import tqdm
import math
import shutil
import accelerate   

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.transformer_models.base import TRANSFORMERS_REGISTRY
from src.text_encoder.text_encoder import TextEncoder
from src.vae import get_vae
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.video_processor import VideoProcessor
from src.ui.nodes import UINode
from src.utils.dtype_utils import select_ideal_dtypes
from src.attention import attention_register
from src.utils.cache_utils import empty_cache
from logging import Logger
from src.preprocess import CLIPPreprocessor, CameraPreprocessor
from typing import Callable
from src.utils.defaults import (
    DEFAULT_DEVICE,
    DEFAULT_CONFIG_SAVE_PATH,
    DEFAULT_SAVE_PATH,
)
import tempfile

from src.mixins import DownloadMixin, LoaderMixin, ToMixin, OffloadMixin
from glob import glob
from safetensors import safe_open
from src.converters import (
    get_transformer_keys,
    convert_transformer,
    get_vae_keys,
    convert_vae,
)
import numpy as np
from PIL import Image
from torchvision import transforms as TF
import inspect
from src.preprocess import preprocessor_registry
from src.postprocess import postprocessor_registry

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BaseEngine(DownloadMixin, LoaderMixin, ToMixin, OffloadMixin):
    config: Dict[str, Any]
    scheduler: Any | None = None
    vae: AutoencoderKL | None = None
    text_encoder: TextEncoder | None = None
    transformer: ModelMixin | None = None
    device: torch.device | None = None
    preprocessors: Dict[str, Any] = {}
    postprocessors: Dict[str, Any] = {}
    offload_to_cpu: bool = False
    video_processor: VideoProcessor
    config_save_path: str | None = None
    component_load_dtypes: Dict[str, torch.dtype] | None = None
    component_dtypes: Dict[str, torch.dtype] | None = None
    components_to_load: List[str] | None = None
    preprocessors_to_load: List[str] | None = None
    postprocessors_to_load: List[str] | None = None
    component_init_kwargs: Dict[str, Any] | None = None
    save_path: str | None = None
    logger: Logger
    attention_type: str = "sdpa"
    tag: str | None = None
    check_weights: bool = True
    save_converted_weights: bool = True
    vae_scale_factor_temporal: float = 1.0
    vae_scale_factor_spatial: float = 1.0
    num_channels_latents: int = 4

    def __init__(
        self,
        yaml_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ):
        self.device = device
        self._init_logger()
        self.config = self._load_yaml(yaml_path)

        logger.info(f"Loaded config: {self.config}")

        self.save_path = kwargs.get("save_path", None)

        self.download(self.save_path)

        for key, value in kwargs.items():
            if key not in [
                "save_path",
                "config_kwargs",
                "components_to_load",
                "component_load_dtypes",
                "component_dtypes",
                "component_init_kwargs",
                "preprocessors_to_load",
                "postprocessors_to_load",
                "device",
            ]:
                setattr(self, key, value)

        self._parse_config(
            self.config,
            kwargs.get("config_kwargs", {}),
            kwargs.get("components_to_load", None),
            kwargs.get("component_load_dtypes", None),
            kwargs.get("component_dtypes", None),
            kwargs.get("component_init_kwargs", {}),
            kwargs.get("preprocessors_to_load", None),
            kwargs.get("postprocessors_to_load", None),
        )

        self.attention_type = kwargs.get("attention_type", "sdpa")
        attention_register.set_default(self.attention_type)

    def _init_logger(self):
        self.logger = logger

    def _aspect_ratio_resize(self, image, max_area=720 * 1280, mod_value=16):
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        return image, height, width

    def _center_crop_resize(self, image, height, width):
        # Calculate resize ratio to match first frame dimensions
        resize_ratio = max(width / image.width, height / image.height)

        # Resize the image
        width = round(image.width * resize_ratio)
        height = round(image.height * resize_ratio)
        size = [width, height]
        image = TF.center_crop(image, size)
        return image, height, width

    def _parse_config(
        self,
        config: Dict[str, Any],
        config_kwargs: Dict[str, Any],
        components_to_load: List[str] | None,
        component_load_dtypes: Dict[str, torch.dtype] | None,
        component_dtypes: Dict[str, str] | None,
        component_init_kwargs: Dict[str, Any] | None,
        preprocessors_to_load: List[str] | None,
        postprocessors_to_load: List[str] | None,
    ):
        self.logger.info(f"Loading model {config['name']}")
        self.component_init_kwargs = component_init_kwargs
        ideal_dtypes = select_ideal_dtypes()
        self.component_load_dtypes = component_load_dtypes

        if component_dtypes:
            self.component_dtypes = {}
            for component_type, dtype in component_dtypes.items():
                self.component_dtypes[component_type] = self._parse_dtype(dtype)
        else:
            self.component_dtypes = ideal_dtypes

        if not self.component_load_dtypes:
            self.component_load_dtypes = {}
            for component_type in ideal_dtypes.keys():
                self.component_load_dtypes[component_type] = ideal_dtypes[
                    component_type
                ]

        # check if any component is missing, otherwise use the default dtypes
        for component_type in ideal_dtypes.keys():
            if component_type not in self.component_dtypes:
                self.component_dtypes[component_type] = ideal_dtypes[component_type]
            if component_type not in self.component_load_dtypes:
                self.component_load_dtypes[component_type] = ideal_dtypes[
                    component_type
                ]

        self.config_save_path = config_kwargs.get(
            "config_save_path", DEFAULT_CONFIG_SAVE_PATH
        )
        if self.config_save_path:
            os.makedirs(self.config_save_path, exist_ok=True)
        components = config.get("components", [])
        self.logger.info(f"Loading {len(components_to_load or components)} components")
        self.load_components(components, components_to_load)
        self.load_preprocessors(
            config.get("preprocessors", []) or [], preprocessors_to_load
        )
        self.load_postprocessors(
            config.get("postprocessors", []) or [], postprocessors_to_load
        )

    @contextmanager
    def _progress_bar(self, total: int, desc: str | None = None, **kwargs):
        with tqdm(total=total, desc=desc, **kwargs) as pbar:
            yield pbar

    def _preprocess_kwargs(self, input_nodes: List[UINode] | None = None, **kwargs):
        if input_nodes:
            kwargs.update(self._parse_input_nodes(input_nodes))
        return kwargs

    def _get_default_kwargs(self, func_name: str):
        default_kwargs = {}
        defaults = self.config.get("defaults", {})
        if func_name in defaults:
            default_kwargs.update(defaults[func_name])
        return default_kwargs

    def set_attention_type(self, attention_type: str | None = None):
        if attention_type:
            attention_register.set_default(attention_type)
        self.attention_type = attention_type

    @torch.inference_mode()
    def run(
        self,
        input_nodes: List[UINode] | None = None,
        **kwargs,
    ):
        raise NotImplementedError("Subclasses must implement this method")

    def load_component(self, component: Dict[str, Any], load_dtype: torch.dtype | None, no_weights: bool = False):
        component_type = component.get("type")
        if component_type == "scheduler":
            self.load_scheduler(component)
        elif component_type == "vae":
            self.load_vae(component, load_dtype, no_weights)
        elif component_type == "text_encoder":
            self.load_text_encoder(component, no_weights)
        elif component_type == "transformer":
            self.load_transformer(component, load_dtype, no_weights)
        else:
            raise ValueError(f"Component type {component_type} not supported")
        empty_cache()

    def load_scheduler(self, component: Dict[str, Any]):
        self.scheduler = self._load_component(component)

    def load_vae(self, component: Dict[str, Any], load_dtype: torch.dtype | None):
        component["model_path"], is_converted = self._check_convert_model_path(
            component
        )
        if is_converted:
            component["config_path"] = os.path.join(
                component["model_path"], "config.json"
            )

        if self.check_weights and not self._check_weights(component):
            self.logger.info(f"Found old model weights, converting to diffusers format")
            self.vae = self.convert_vae_weights(component)
            if self.save_converted_weights:
                self.logger.info(
                    f"Saving converted vae weights to {component.get('model_path', None)}"
                )
                model_path = component.get("model_path", None)
                tmp_dir = tempfile.mkdtemp(dir="/mnt/localssd")
                try:
                    self.save_component(
                        self.vae, tmp_dir, "vae", **self.config.get("save_kwargs", {})
                    )
                    if os.path.isdir(model_path):
                        # Safely copy to preserve other files in the directory
                        # add subdirectory called vae
                        vae_dir = os.path.join(model_path, "vae")
                        os.makedirs(vae_dir, exist_ok=True)
                        os.rename(tmp_dir, vae_dir)
                    else:
                        # Atomically replace file with directory
                        shutil.rmtree(model_path, ignore_errors=True)
                        model_dir = os.path.dirname(model_path)
                        vae_path = os.path.join(model_dir, "vae")
                        os.makedirs(vae_path, exist_ok=True)
                        os.rename(tmp_dir, vae_path)
                finally:
                    # Clean up temp dir if it still exists
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                self.logger.info(f"Converted vae weights to diffusers format")
                empty_cache()
        else:
            self.vae = self._load_model(component, get_vae, "VAE", load_dtype, no_weights)
        if self.component_dtypes and "vae" in self.component_dtypes:
            self.to_dtype(self.vae, self.component_dtypes["vae"])
            vae_init_kwargs = self.component_init_kwargs.get("vae", {})
            if (
                "enable_slicing" in vae_init_kwargs
                and vae_init_kwargs["enable_slicing"]
                and hasattr(self.vae, "enable_slicing")
            ):
                self.vae.enable_slicing()
            if (
                "enable_tiling" in vae_init_kwargs
                and vae_init_kwargs["enable_tiling"]
                and hasattr(self.vae, "enable_tiling")
            ):
                self.vae.enable_tiling()
    
    def load_config_by_type(self, component_type: str):
        with accelerate.init_empty_weights():
            for component in self.config.get("components", []):
                if component.get("type") == component_type:
                    self.load_component(
                        component,
                        (
                            self.component_load_dtypes.get(component.get("type"))
                            if self.component_load_dtypes
                            else None
                        ),
                        no_weights=True,
                    )
                    return getattr(getattr(self, component_type), "config", {})
            raise ValueError(f"Component type {component_type} not found")

    def load_text_encoder(self, component: Dict[str, Any], no_weights: bool = False):
        if self._is_url(component.get("config_path")):
            config_path = self._check_config_for_url(component.get("config_path"))
            component["config_path"] = config_path
        self.text_encoder = TextEncoder(component, no_weights)
        if self.component_dtypes and "text_encoder" in self.component_dtypes:
            self.to_dtype(self.text_encoder, self.component_dtypes["text_encoder"])

    def load_transformer(
        self, component: Dict[str, Any], load_dtype: torch.dtype | None, no_weights: bool = False
    ):
        component["model_path"], is_converted = self._check_convert_model_path(
            component
        )
        if is_converted:
            component["config_path"] = os.path.join(
                component["model_path"], "config.json"
            )

        if self.check_weights and not self._check_weights(component):
            self.logger.info(f"Found old model weights, converting to diffusers format")
            self.transformer = self.convert_transformer_weights(component)
            if self.save_converted_weights:
                self.logger.info(
                    f"Saving converted transformer weights to {component.get('model_path', None)}"
                )
                model_path = component.get("model_path", None)
                tmp_dir = tempfile.mkdtemp(dir="/mnt/localssd")
                try:
                    self.save_component(
                        self.transformer,
                        tmp_dir,
                        "transformer",
                        **self.config.get("save_kwargs", {}),
                    )
                    if os.path.isdir(model_path):
                        # Safely copy to preserve other files in the directory
                        # add subdirectory called transformer
                        transformer_dir = os.path.join(model_path, "transformer")
                        os.makedirs(transformer_dir, exist_ok=True)
                        os.rename(tmp_dir, transformer_dir)
                    else:
                        # Atomically replace file with directory
                        shutil.rmtree(model_path, ignore_errors=True)
                        model_dir = os.path.dirname(model_path)
                        transformer_path = os.path.join(model_dir, "transformer")
                        os.makedirs(transformer_path, exist_ok=True)
                        os.rename(tmp_dir, transformer_path)
                finally:
                    # Clean up temp dir if it still exists
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                self.logger.info(f"Converted transformer weights to diffusers format")
                empty_cache()
        else:
            self.transformer = self._load_model(
                component, TRANSFORMERS_REGISTRY.get, "Transformer", load_dtype, no_weights
            )

        if self.component_dtypes and "transformer" in self.component_dtypes:
            self.to_dtype(self.transformer, self.component_dtypes["transformer"])

    def _get_safetensors_keys(self, model_path: str, model_key: str | None = None):
        keys = set()
        with safe_open(model_path, framework="pt", device="cpu") as f:
            if model_key is not None:
                keys.update(f[model_key].keys())
            else:
                if len(f.keys()) < 2:
                    keys.update(f[list(f.keys())[0]].keys())
                else:
                    keys.update(f.keys())
        return keys

    def _get_pt_keys(self, model_path: str, model_key: str | None = None):
        keys = set()
        partial_state = torch.load(
            model_path, map_location="cpu", mmap=True, weights_only=True
        )
        if model_key is not None:
            partial_state = partial_state[model_key]
        else:
            if len(partial_state.keys()) < 2:
                partial_state = partial_state[list(partial_state.keys())[0]]
        keys.update(partial_state.keys())
        return keys

    def _check_convert_model_path(self, component: Dict[str, Any]):
        assert "model_path" in component, "`model_path` is required"
        assert component.get("type") in [
            "transformer",
            "vae",
        ], "Only transformer and vae are supported for now"
        model_path = component["model_path"]
        component_type = component.get("type")
        if os.path.isfile(model_path):
            # check base directory
            if os.path.isdir(os.path.dirname(model_path)):
                if os.path.isdir(
                    os.path.join(os.path.dirname(model_path), component_type)
                ):
                    return (
                        os.path.join(os.path.dirname(model_path), component_type),
                        True,
                    )
        elif os.path.isdir(model_path):
            if os.path.isdir(os.path.join(model_path, component_type)):
                return os.path.join(model_path, component_type), True
        return model_path, False

    def _check_weights(self, component: Dict[str, Any]):
        assert "model_path" in component, "`model_path` is required"
        model_path = component["model_path"]
        base = component.get("base", None)
        component_type = component.get("type")
        assert component_type in [
            "transformer",
            "vae",
        ], "Only transformer and vae are supported for now"
        file_pattern = component.get("file_pattern", None)

        pt_extensions = tuple(["pt", "bin", "pth"])
        model_key = component.get("model_key", None)
        keys = set()

        if os.path.isdir(model_path):
            if file_pattern:
                files = glob(os.path.join(model_path, file_pattern))
            else:
                files = (
                    glob(os.path.join(model_path, "*.safetensors"))
                    + glob(os.path.join(model_path, "*.pt"))
                    + glob(os.path.join(model_path, "*.bin"))
                    + glob(os.path.join(model_path, "*.pth"))
                )

            if len(files) == 0:
                raise ValueError(
                    f"No files found in {model_path} with pattern {file_pattern}"
                )

            for file in files:
                if file.endswith(".safetensors"):
                    keys.update(self._get_safetensors_keys(file, model_key))
                elif file.endswith(pt_extensions):
                    keys.update(self._get_pt_keys(file, model_key))
        else:
            if model_path.endswith(".safetensors"):
                keys.update(self._get_safetensors_keys(model_path, model_key))
            elif model_path.endswith(pt_extensions):
                keys.update(self._get_pt_keys(model_path, model_key))

        if component_type == "transformer":
            _keys = set(
                get_transformer_keys(
                    base,
                    component.get("tag", None),
                    component.get("converter_kwargs", {}),
                )
            )
            iterating_keys = _keys.copy()
        elif component_type == "vae":
            _keys = set(
                get_vae_keys(
                    base,
                    component.get("tag", None),
                    component.get("converter_kwargs", {}),
                )
            )
            iterating_keys = _keys.copy()

        found_keys = set()
        for iterating_key in iterating_keys:
            for key in keys:
                if key == iterating_key:
                    found_keys.add(iterating_key)
                    break

        return len(found_keys) == len(iterating_keys)

    def convert_transformer_weights(self, component: Dict[str, Any]):
        assert "model_path" in component, "`model_path` is required"
        component_type = component.get("type")
        assert component_type == "transformer", "Only transformer is supported for now"
        self.logger.info(f"Converting old model weights to diffusers format")
        return convert_transformer(
            component.get("tag", None),
            component["base"],
            component["model_path"],
            component.get("model_key", None),
            component.get("file_pattern", None),
        )

    def convert_vae_weights(self, component: Dict[str, Any]):
        assert "model_path" in component, "`model_path` is required"
        component_type = component.get("type")
        assert component_type == "vae", "Only vae is supported for now"
        self.logger.info(f"Converting old model weights to diffusers format")
        return convert_vae(
            component.get("tag", None),
            component["base"],
            component["model_path"],
            component.get("model_key", None),
            component.get("file_pattern", None),
            **component.get("converter_kwargs", {}),
        )

    @torch.inference_mode()
    def vae_decode(
        self,
        latents: torch.Tensor,
        offload: bool = False,
        dtype: torch.dtype | None = None,
    ):
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)
        denormalized_latents = self.vae.denormalize_latents(latents).to(
            dtype=self.vae.dtype, device=self.device
        )
        video = self.vae.decode(denormalized_latents, return_dict=False)[0]
        if offload:
            self._offload(self.vae)
        return video.to(dtype=dtype)

    @torch.inference_mode()
    def vae_encode(
        self,
        video: torch.Tensor,
        offload: bool = False,
        sample_mode: str = "mode",
        sample_generator: torch.Generator = None,
        dtype: torch.dtype = None,
        normalize_latents: bool = True,
        normalize_latents_dtype: torch.dtype | None = None,
    ):
        if self.vae is None:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        video = video.to(dtype=self.vae.dtype, device=self.device)

        latents = self.vae.encode(video, return_dict=False)[0]
        if sample_mode == "sample":
            latents = latents.sample(generator=sample_generator)
        elif sample_mode == "mode":
            latents = latents.mode()
        else:
            raise ValueError(f"Invalid sample mode: {sample_mode}")

        if not normalize_latents_dtype:
            normalize_latents_dtype = self.vae.dtype

        if normalize_latents:
            latents = latents.to(dtype=normalize_latents_dtype)
            latents = self.vae.normalize_latents(latents)

        if offload:
            self._offload(self.vae)

        return latents.to(dtype=dtype)

    def load_components(
        self, components: List[Dict[str, Any]], components_to_load: List[str] | None
    ):
        for component in components:
            if components_to_load and component.get("type") in components_to_load:
                self.load_component(
                    component,
                    (
                        self.component_load_dtypes.get(component.get("type"))
                        if self.component_load_dtypes
                        else None
                    ),
                )

    def load_component_by_type(self, component_type: str):
        for component in self.config.get("components", []):
            if component.get("type") == component_type:
                self.load_component(
                    component,
                    (
                        self.component_load_dtypes.get(component.get("type"))
                        if self.component_load_dtypes
                        else None
                    ),
                )
                break

    def set_compiler(self, compiler_name: str):
        pass

    def set_quantizer(self, quantizer_name: str):
        pass

    def load_preprocessor_by_type(self, preprocessor_type: str):
        for preprocessor in self.config.get("preprocessors", []):
            if preprocessor.get("type") == preprocessor_type:
                self.load_preprocessor(preprocessor)
                break

    def load_preprocessor(self, config: Dict[str, Any]):
        preprocessor_type = config.get("type")
        preprocessor_class = preprocessor_registry.get(preprocessor_type)
        if preprocessor_class is None:
            raise ValueError(f"Preprocessor type {preprocessor_type} not supported")
        return preprocessor_class(**config)

    def load_preprocessors(
        self,
        preprocessors: List[Dict[str, Any]],
        preprocessors_to_load: List[str] | None,
    ):
        for preprocessor in preprocessors:
            if (
                preprocessors_to_load is None
                or preprocessor.get("type") in preprocessors_to_load
            ):
                self.preprocessors[preprocessor.get("type")] = self.load_preprocessor(
                    preprocessor
                )

    def load_postprocessor_by_type(self, postprocessor_type: str):
        for postprocessor in self.config.get("postprocessors", []):
            if postprocessor.get("type") == postprocessor_type:
                self.load_postprocessor(postprocessor)
                break

    def load_postprocessor(self, config: Dict[str, Any]):
        postprocessor_type = config.get("type")
        postprocessor_class = postprocessor_registry.get(postprocessor_type)
        if postprocessor_class is None:
            raise ValueError(f"Postprocessor type {postprocessor_type} not supported")
        return postprocessor_class(**config)

    def load_postprocessors(
        self,
        postprocessors: List[Dict[str, Any]],
        postprocessors_to_load: List[str] | None,
    ):
        for postprocessor in postprocessors:
            if (
                postprocessors_to_load is None
                or postprocessor.get("type") in postprocessors_to_load
            ):
                self.postprocessors[postprocessor.get("type")] = (
                    self.load_postprocessor(postprocessor)
                )

    def apply_lora(self, lora_path: str):
        pass

    def _parse_input_nodes(self, input_nodes: List[UINode]):
        kwargs = {}
        for node in input_nodes:
            kwargs.update(node.as_param())
        return kwargs

    def download(
        self,
        save_path: str | None = None,
        components_path: str | None = None,
        preprocessors_path: str | None = None,
        postprocessors_path: str | None = None,
    ):
        if save_path is None:
            save_path = self.save_path
        if save_path is None:
            save_path = DEFAULT_SAVE_PATH
        if components_path is None:
            components_path = os.path.join(save_path, "components")
        if preprocessors_path is None:
            preprocessors_path = os.path.join(save_path, "preprocessors")
        if postprocessors_path is None:
            postprocessors_path = os.path.join(save_path, "postprocessors")

        os.makedirs(save_path, exist_ok=True)
        for component in self.config.get("components", []):
            if model_path := component.get("model_path"):
                downloaded_model_path = self._download(model_path, components_path)
                if downloaded_model_path:
                    component["model_path"] = downloaded_model_path

        preprocessors = self.config.get("preprocessors", []) or []
        if preprocessors:
            self.logger.info(f"Downloading {len(preprocessors)} preprocessors")
        for preprocessor in preprocessors:
            if preprocessor_path := preprocessor.get("model_path"):
                downloaded_preprocessor_path = self._download(
                    preprocessor_path, preprocessors_path
                )
                if downloaded_preprocessor_path:
                    preprocessor["model_path"] = downloaded_preprocessor_path

        postprocessors = self.config.get("postprocessors", []) or []
        if postprocessors:
            self.logger.info(f"Downloading {len(postprocessors)} postprocessors")
        for postprocessor in postprocessors:
            if postprocessor_path := postprocessor.get("model_path"):
                downloaded_postprocessor_path = self._download(
                    postprocessor_path, postprocessors_path
                )
                if downloaded_postprocessor_path:
                    postprocessor["model_path"] = downloaded_postprocessor_path

    def _get_latents(
        self,
        height: int,
        width: int,
        duration: int | str,
        fps: int = 16,
        num_videos: int = 1,
        num_channels_latents: int = None,
        seed: int | None = None,
        dtype: torch.dtype = None,
        layout: torch.layout = None,
        generator: torch.Generator | None = None,
        return_generator: bool = False,
        parse_frames: bool = True,
        order: Literal["BCF", "BFC"] = "BCF",
    ):
        if parse_frames or isinstance(duration, str):
            num_frames = self._parse_num_frames(duration, fps)
            latent_num_frames = math.ceil(
                (num_frames + 3) / self.vae_scale_factor_temporal
            )
        else:
            latent_num_frames = duration

        latent_height = math.ceil(height / self.vae_scale_factor_spatial)
        latent_width = math.ceil(width / self.vae_scale_factor_spatial)

        if seed is not None and generator is not None:
            self.logger.warning(
                f"Both `seed` and `generator` are provided. `seed` will be ignored."
            )

        if generator is None:
            device = self.device
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)
        else:
            device = generator.device

        if order == "BCF":
            shape = (
                num_videos,
                num_channels_latents or self.num_channels_latents,
                latent_num_frames,
                latent_height,
                latent_width,
            )
        elif order == "BFC":
            shape = (
                num_videos,
                latent_num_frames,
                num_channels_latents or self.num_channels_latents,
                latent_height,
                latent_width,
            )
        else:
            raise ValueError(f"Invalid order: {order}")

        noise = torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
            layout=layout or torch.strided,
        ).to(self.device)

        if return_generator:
            return noise, generator
        else:
            return noise

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        video = self.vae_decode(latents)
        rendered_video = self._postprocess(video)
        render_on_step_callback(rendered_video)

    def _postprocess(self, video: torch.Tensor, output_type: str = "np"):
        postprocessed_video = self.video_processor.postprocess_video(
            video, output_type=output_type
        )
        return postprocessed_video

    def _get_timesteps(
        self,
        scheduler: SchedulerMixin | None = None,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        timesteps_as_indices: bool = False,
        **kwargs,
    ):
        scheduler = scheduler or self.scheduler
        device = self.device
        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )

        if timesteps is not None:
            if timesteps_as_indices:
                # This is the logic from the old _get_timesteps
                timestep_ids = torch.tensor(
                    timesteps, dtype=torch.long, device=self.device
                )
                num_train_timesteps = getattr(
                    self.scheduler, "num_train_timesteps", 1000
                )
                timesteps = self.scheduler.timesteps[num_train_timesteps - timestep_ids]
                self.scheduler.timesteps = timesteps
                self.scheduler.sigmas = self.scheduler.timesteps / num_train_timesteps
                timesteps = self.scheduler.timesteps
                num_inference_steps = len(timesteps)
            else:
                # This is the logic from retrieve_timesteps
                accepts_timesteps = "timesteps" in set(
                    inspect.signature(scheduler.set_timesteps).parameters.keys()
                )
                if not accepts_timesteps:
                    raise ValueError(
                        f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                        f" timestep schedules. Please check whether you are using the correct scheduler."
                    )
                scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
                timesteps = scheduler.timesteps
                num_inference_steps = len(timesteps)

        elif sigmas is not None:
            accepts_sigmas = "sigmas" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_sigmas:
                # This is a fallback from retrieve_timesteps
                scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
                timesteps = scheduler.timesteps
            else:
                scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
                timesteps = scheduler.timesteps
                num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps

        return timesteps, num_inference_steps

    def _parse_num_frames(self, duration: int | str, fps: int = 16):
        """Accepts a duration in seconds or a string like "16" or "16s" and returns the number of frames.

        Args:
            duration (int | str): duration in seconds or a string like "16" or "16s"

        Returns:
            int: number of frames
        """

        if isinstance(duration, str):
            if duration.endswith("s"):
                duration = int(duration[:-1]) * fps + 1
            elif duration.endswith("f"):
                duration = int(duration[:-1])
            else:
                duration = int(duration)
        if duration % self.vae_scale_factor_temporal != 1:
            duration = (
                duration
                // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
                + 1
            )
        duration = max(duration, 1)
        return duration

    def _calculate_shift(
        self,
        image_seq_len,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
    ):
        """Calculate shift parameter for timestep scheduling"""
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def __str__(self):
        return f"BaseEngine(config={self.config}, device={self.device})"

    def __repr__(self):
        return self.__str__()
