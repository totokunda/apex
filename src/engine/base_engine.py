import os
from typing import List, Dict, Any, Optional, Literal, Union
from typing import TYPE_CHECKING, overload
from diffusers.utils.dummy_pt_objects import SchedulerMixin
import torch
from loguru import logger
import urllib3
from diffusers.models.modeling_utils import ModelMixin
from contextlib import contextmanager
from tqdm import tqdm
import shutil
import accelerate
from src.utils.defaults import DEFAULT_CACHE_PATH
from src.utils.module import find_class_recursive

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from src.transformer.base import TRANSFORMERS_REGISTRY as TRANSFORMERS_REGISTRY_TORCH
from src.mlx.transformer.base import TRANSFORMERS_REGISTRY as TRANSFORMERS_REGISTRY_MLX
from src.text_encoder.text_encoder import TextEncoder
from src.vae import get_vae
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.video_processor import VideoProcessor
from src.ui.nodes import UINode
from src.utils.dtype import select_ideal_dtypes
from src.attention import attention_register
from src.utils.cache import empty_cache
from logging import Logger
from src.scheduler import SchedulerInterface
from typing import Callable
from src.utils.mlx import convert_dtype_to_torch, convert_dtype_to_mlx
from src.memory_management import MemoryManager, MemoryConfig
import torch.nn as nn
import importlib
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor


from src.utils.defaults import (
    DEFAULT_DEVICE,
    DEFAULT_CONFIG_SAVE_PATH,
    DEFAULT_SAVE_PATH,
    DEFAULT_COMPONENTS_PATH,
    DEFAULT_PREPROCESSOR_SAVE_PATH,
    DEFAULT_POSTPROCESSOR_SAVE_PATH,
    DEFAULT_LORA_SAVE_PATH,
)

import tempfile
from src.mixins import LoaderMixin, ToMixin, OffloadMixin
from glob import glob
from safetensors import safe_open
import mlx.core as mx
import mlx.nn as mx_nn
from src.converters import (
    get_transformer_keys,
    convert_transformer,
    convert_vae,
)

import numpy as np
from PIL import Image
from torchvision import transforms as TF
import inspect
from src.preprocess import preprocessor_registry
from src.postprocess import postprocessor_registry
from src.lora import LoraManager, LoraItem
from src.helpers.helpers import helpers

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AutoLoadingHelperDict(dict):
    """A dictionary wrapper that automatically loads helpers when accessed."""

    def __init__(self, engine_instance):
        super().__init__()
        self._engine = engine_instance

    # ---------------- Static typing for editor IntelliSense ---------------- #
    if TYPE_CHECKING:
        # Import helper classes only for typing to avoid runtime deps/cycles
        from src.helpers.clip import CLIP
        from src.helpers.wan.ati import WanATI
        from src.helpers.wan.recam import WanRecam
        from src.helpers.wan.fun_camera import WanFunCamera
        from src.helpers.wan.multitalk import WanMultiTalk
        from src.helpers.hunyuan.llama import HunyuanLlama
        from src.helpers.hunyuan.avatar import HunyuanAvatar
        from src.helpers.hidream.llama import HidreamLlama
        from src.helpers.stepvideo.text_encoder import StepVideoTextEncoder
        from src.helpers.ltx.patchifier import SymmetricPatchifier

        # Overloads for known helper keys → precise instance types
        @overload
        def __getitem__(self, key: Literal["clip"]) -> "CLIP": ...

        @overload
        def __getitem__(self, key: Literal["wan.ati"]) -> "WanATI": ...

        @overload
        def __getitem__(self, key: Literal["wan.recam"]) -> "WanRecam": ...

        @overload
        def __getitem__(self, key: Literal["wan.fun_camera"]) -> "WanFunCamera": ...

        @overload
        def __getitem__(self, key: Literal["wan.multitalk"]) -> "WanMultiTalk": ...

        @overload
        def __getitem__(self, key: Literal["hunyuan.llama"]) -> "HunyuanLlama": ...

        @overload
        def __getitem__(self, key: Literal["hunyuan.avatar"]) -> "HunyuanAvatar": ...

        @overload
        def __getitem__(self, key: Literal["hidream.llama"]) -> "HidreamLlama": ...

        @overload
        def __getitem__(self, key: Literal["stepvideo.text_encoder"]) -> "StepVideoTextEncoder": ...

        @overload
        def __getitem__(self, key: Literal["ltx.patchifier"]) -> "SymmetricPatchifier": ...

        @overload
        def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key):
        # If helper exists, return it
        if super().__contains__(key):
            return super().__getitem__(key)

        # Try to load helper automatically
        helper = self._engine._auto_load_helper(key)
        if helper is not None:
            self[key] = helper
            return helper

        # If couldn't load, raise KeyError
        raise KeyError(f"Helper '{key}' not found and could not be auto-loaded")

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class BaseEngine(LoaderMixin, ToMixin, OffloadMixin):
    engine_type: Literal["torch", "mlx"] = "torch"
    config: Dict[str, Any]
    scheduler: SchedulerInterface | None = None
    vae: AutoencoderKL | None = None
    text_encoder: TextEncoder | None = None
    transformer: ModelMixin | None = None
    device: torch.device | None = None
    _helpers: AutoLoadingHelperDict
    _preprocessors: Dict[str, Any] = {}
    _postprocessors: Dict[str, Any] = {}
    offload_to_cpu: bool = False
    video_processor: VideoProcessor
    image_processor: VaeImageProcessor
    config_save_path: str | None = None
    component_load_dtypes: Dict[str, torch.dtype] | None = None
    component_dtypes: Dict[str, torch.dtype] | None = None
    components_to_load: List[str] | None = None
    preprocessors_to_load: List[str] | None = None
    postprocessors_to_load: List[str] | None = None
    save_path: str | None = None
    logger: Logger
    attention_type: str = "sdpa"
    check_weights: bool = True
    save_converted_weights: bool = True
    vae_scale_factor_temporal: float = 1.0
    vae_scale_factor_spatial: float = 1.0
    vae_scale_factor: float = 1.0
    num_channels_latents: int = 4
    denoise_type: str | None = None
    vae_tiling: bool = False
    vae_slicing: bool = False
    lora_manager: LoraManager | None = None
    loaded_loras: Dict[str, LoraItem] = {}
    _memory_management_map: Dict[str, MemoryConfig] | None = None
    _component_memory_managers: Dict[str, MemoryManager] = {}
    selected_components: Dict[str, Any] | None = None
    
    def __init__(
        self,
        yaml_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ):
        self.device = device
        self._helpers = AutoLoadingHelperDict(self)
        self._init_logger()
        self.config = self._load_yaml(yaml_path)


        self.save_path = kwargs.get("save_path", None)

        for key, value in kwargs.items():
            if key not in [
                "save_path",
                "config_kwargs",
                "components_to_load",
                "component_load_dtypes",
                "component_dtypes",
                "preprocessors_to_load",
                "postprocessors_to_load",
                "device",
            ]:
                setattr(self, key, value)
        
        self.download(self.save_path)

        # Normalize optional memory management mapping
        self._memory_management_map = self._normalize_memory_management(
            kwargs.get("memory_management", None)
        )

        self._parse_config(
            self.config,
            kwargs.get("config_kwargs", {}),
            kwargs.get("components_to_load", None),
            kwargs.get("component_load_dtypes", None),
            kwargs.get("component_dtypes", None),
            kwargs.get("preprocessors_to_load", None),
            kwargs.get("postprocessors_to_load", None),
        )

        self.attention_type = kwargs.get("attention_type", "sdpa")
        attention_register.set_default(self.attention_type)
        self._init_lora_manager(kwargs.get("lora_save_path", DEFAULT_LORA_SAVE_PATH))
        self._auto_apply_loras(kwargs.get("lora_model_name_or_type", "transformer"))

    def _init_logger(self):
        self.logger = logger

    def _aspect_ratio_resize(
        self,
        image,
        max_area=720 * 1280,
        mod_value=16,
        resize_mode=Image.Resampling.LANCZOS,
    ):
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height), resize_mode)
        return image, height, width

    def _aspect_ratio_to_height_width(
        self,
        aspect_ratio: str,
        resolution: int,
        mod_value: int = 16,
    ) -> tuple[int, int]:
        """Convert an aspect ratio string like "16:9" into (height, width) using the given resolution.

        The resolution is treated as the longer side. The returned dimensions are floored
        to be multiples of mod_value to satisfy common model requirements.
        """
        if not isinstance(resolution, (int, float)) or resolution <= 0:
            resolution = 1024

        if not isinstance(aspect_ratio, str):
            # Fallback to square
            target_w = int(resolution)
            target_h = int(resolution)
        else:
            ar = aspect_ratio.strip().lower().replace("x", ":").replace("/", ":")
            parts = [p for p in ar.split(":") if p]
            try:
                if len(parts) == 2:
                    w_part = float(parts[0])
                    h_part = float(parts[1])
                    if w_part <= 0 or h_part <= 0:
                        raise ValueError
                    ratio = w_part / h_part
                else:
                    # single number implies square
                    ratio = 1.0
            except Exception:
                ratio = 1.0

            # Treat resolution as longer side
            if ratio >= 1.0:
                target_w = int(resolution)
                target_h = int(round(target_w / ratio))
            else:
                target_h = int(resolution)
                target_w = int(round(target_h * ratio))

        # Snap to multiples of mod_value and ensure minimum size
        target_w = max(mod_value, (target_w // mod_value) * mod_value)
        target_h = max(mod_value, (target_h // mod_value) * mod_value)
        return target_h, target_w

    def _resolution_to_height_width(
        self, resolution: int, mod_value: int = 16
    ) -> tuple[int, int]:
        """Return square (height, width) for a given resolution, snapped to mod_value."""
        if not isinstance(resolution, (int, float)) or resolution <= 0:
            resolution = 1024
        r = int(resolution)
        r = max(mod_value, (r // mod_value) * mod_value)
        return r, r

    def _image_to_height_width(self, image, mod_value: int = 16) -> tuple[int, int]:
        """Infer (height, width) from an input image, snapped to mod_value.

        Accepts PIL.Image, numpy array, torch tensor, or path/URL string.
        """
        try:
            pil_image = self._load_image(image)
            width, height = pil_image.size
        except Exception:
            # Fallback to a reasonable default if image cannot be loaded
            return self._resolution_to_height_width(1024, mod_value)

        width = max(mod_value, (int(width) // mod_value) * mod_value)
        height = max(mod_value, (int(height) // mod_value) * mod_value)
        return height, width

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
        preprocessors_to_load: List[str] | None,
        postprocessors_to_load: List[str] | None,
    ):
        self.logger.info(f"Loading model {config['name']}")
        ideal_dtypes = select_ideal_dtypes()
        self.component_load_dtypes = component_load_dtypes
        self.engine_type = config.get("engine_type", "torch")

        if config.get("denoise_type", None):
            self.denoise_type = config.get("denoise_type")

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
        if components_to_load:
            self.logger.info(f"Loading {len(components_to_load)} components")

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

    @torch.no_grad()
    def run(self, *args, **kwargs):
        default_kwargs = self._get_default_kwargs("run")
        merged_kwargs = {**default_kwargs, **kwargs}
        if hasattr(self, "implementation_engine"):
            args, kwargs = self.run_preprocessors(args, merged_kwargs)
            out = self.implementation_engine.run(*args, **kwargs)
            return self.run_postprocessors(out)
        else:
            raise NotImplementedError("Subclasses must implement this method")

    def run_postprocessors(self, out):
        self.load_postprocessors(self.config.get("postprocessors", []))
        for postprocessor in self._postprocessors.values():
            out = postprocessor["postprocessor"](out, **postprocessor["kwargs"])
        return out

    def run_preprocessors(self, args, kwargs):
        # If no preprocessors configured, passthrough
        preprocessors_cfg = self.config.get("preprocessors", []) or []
        if len(preprocessors_cfg) == 0:
            return args, kwargs

        # Ensure preprocessors are loaded
        self.load_preprocessors(preprocessors_cfg)

        # Start with original kwargs intended for the model
        kwargs_out = dict(kwargs)
        
        for key, preprocessor in self._preprocessors.items():

            # Inspect signature and only pass supported kwargs
            try:
                call_sig = inspect.signature(preprocessor.__call__)
                accepted_params = {
                    name
                    for name, p in call_sig.parameters.items()
                    if name != "self"
                    and p.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in call_sig.parameters.values()
                )
            except (TypeError, ValueError):
                accepted_params = set()
                has_var_kw = False

            if has_var_kw:
                call_kwargs = dict(kwargs_out)
            else:
                call_kwargs = {
                    k: v for k, v in kwargs_out.items() if k in accepted_params
                }

            call_kwargs.update(preprocessor["kwargs"])
            # Execute preprocessor
            try:
                result = preprocessor["preprocessor"](**call_kwargs)
            except TypeError:
                # As a last resort, try calling without kwargs if signature is unusual
                result = preprocessor["preprocessor"](**preprocessor["kwargs"])

            if result is None:
                continue

            # Read optional mapped_names from preprocessor config (if provided)
            mapped_names = {}
            try:
                if hasattr(preprocessor, "name_map") and isinstance(
                    getattr(preprocessor, "name_map"), dict
                ):
                    mapped_names = getattr(preprocessor, "name_map") or {}
                elif hasattr(preprocessor, "kwargs") and isinstance(
                    preprocessor.kwargs, dict
                ):
                    maybe_map = preprocessor.kwargs.get("name_map", {})
                    if isinstance(maybe_map, dict):
                        mapped_names = maybe_map
            except Exception:
                mapped_names = {}

            # Merge outputs into kwargs_out with key remapping
            try:
                items_iter = (
                    result.items() if hasattr(result, "items") else dict(result).items()
                )
            except Exception:
                # If not dict-like, skip
                items_iter = []

            for out_key, out_val in items_iter:
                target_key = mapped_names.get(out_key, out_key)
                if out_val is not None:
                    kwargs_out[target_key] = out_val

        # Do not alter positional args
        return args, kwargs_out

    def load_component(
        self,
        component: Dict[str, Any],
        load_dtype: torch.dtype | None = None,
        no_weights: bool = False,
    ):
        component_type = component.get("type")
        component_module = None
        if component_type == "scheduler":
            scheduler = self.load_scheduler(component)
            component_module = scheduler
        elif component_type == "vae":
            vae = self.load_vae(component, load_dtype, no_weights)
            component_module = vae
        elif component_type == "text_encoder":
            text_encoder = self.load_text_encoder(component, no_weights)
            component_module = text_encoder
        elif component_type == "transformer":
            transformer = self.load_transformer(component, load_dtype, no_weights)
            component_module = transformer
        elif component_type == "helper":
            helper = self.load_helper(component)
            component_module = helper
        else:
            raise ValueError(f"Component type {component_type} not supported")
        empty_cache()
        return component_module

    def load_helper(self, component: Dict[str, Any]):

        config = component.copy()  # Don't modify the original
        base = config.pop("base")
        config.pop("type")
        config.pop("name", None)
        module = config.pop("module", None)
        # get the helper class
        try:
            helper_class = helpers.get(base)
        except Exception as e:
            helper_class = find_class_recursive(importlib.import_module(module), base)
        if helper_class is None:
            raise ValueError(f"Helper class {base} not found")

        # create an instance of the helper class
        if hasattr(helper_class, "from_pretrained") and "model_path" in config:
            helper = helper_class.from_pretrained(config["model_path"])
        else:
            helper = helper_class(**config)

        # Store helper with multiple keys for easier access
        helper_name = component.get("name", base)
        self._helpers[base] = helper
        if helper_name != base:
            self._helpers[helper_name] = helper
            # Also store with just the last part of the name (after /)
            if "/" in helper_name:
                short_name = helper_name.split("/")[-1]
                self._helpers[short_name] = helper

        # Move helper to device if possible
        if hasattr(helper, "to") and self.device is not None:
            helper = helper.to(self.device)

        return helper

    def load_helper_by_type(self, helper_type: str):
        for helper in self.config.get("helpers", []):
            if helper.get("type") == helper_type:
                self._helpers[helper_type] = self.load_helper(helper)
                return
        raise ValueError(f"Helper type {helper_type} not found")

    def _auto_load_helper(self, helper_key: str):
        """Automatically load a helper by searching for it in the configuration."""
        # First, check if there's a helper component with matching name or base
        for component in self.config.get("components", []):
            if component.get("type") == "helper":
                component_name = component.get("name", "")
                component_base = component.get("base", "")

                # Match by name or base (with or without namespace prefixes)
                
                if (
                    component_name == helper_key
                    or component_name.endswith(f"/{helper_key}")
                    or component_base == helper_key
                    or component_base.endswith(f".{helper_key}")
                ):

                    try:
                        helper = self.load_helper(component)
                        self.logger.info(
                            f"Auto-loaded helper '{helper_key}' from configuration"
                        )
                        # Move helper to device
                        self.to_device(helper)
                        return helper
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to auto-load helper '{helper_key}': {e}"
                        )
                        return None

        # If not found in current config, check if it's a known helper type that can be loaded
        if helper_key in helpers:
            try:
                helper_class = helpers.get(helper_key)
                helper = helper_class()
                self.logger.info(f"Auto-loaded helper '{helper_key}' from registry")
                # Move helper to device
                if hasattr(helper, "to") and self.device is not None:
                    helper = helper.to(self.device)
                return helper
            except Exception as e:
                self.logger.warning(
                    f"Failed to auto-load helper '{helper_key}' from registry: {e}"
                )
                return None

        return None

    @property
    def helpers(self) -> "AutoLoadingHelperDict":
        return self._helpers

    def load_scheduler(self, component: Dict[str, Any]):
        scheduler = self._load_scheduler(component)

        # Add all SchedulerInterface methods to self.scheduler if not already present
        if (
            not isinstance(scheduler, SchedulerInterface)
            and self.engine_type == "torch"
        ):
            # Create a new class that inherits from both the scheduler's class and SchedulerInterface
            class SchedulerWrapper(scheduler.__class__, SchedulerInterface):
                pass

            # Add all SchedulerInterface methods to the scheduler instance
            scheduler_interface = SchedulerInterface()

            # Add the alphas_cumprod attribute if not present
            if not hasattr(scheduler, "alphas_cumprod"):
                scheduler.alphas_cumprod = getattr(
                    scheduler_interface, "alphas_cumprod", None
                )

            # Add all methods from SchedulerInterface
            for method_name in [
                "add_noise",
                "convert_x0_to_noise",
                "convert_noise_to_x0",
                "convert_velocity_to_x0",
                "convert_flow_pred_to_x0",
                "convert_x0_to_flow_pred",
            ]:
                if not hasattr(scheduler, method_name):
                    method = getattr(scheduler_interface, method_name)
                    setattr(
                        scheduler,
                        method_name,
                        method.__get__(scheduler, type(scheduler)),
                    )

            # Change the class to include SchedulerInterface
            scheduler.__class__ = SchedulerWrapper

        return scheduler

    def load_vae(
        self,
        component: Dict[str, Any],
        load_dtype: torch.dtype | None,
        no_weights: bool = False,
    ):
        vae = self._load_model(
            component,
            get_vae,
            "VAE",
            load_dtype,
            no_weights=no_weights,
            key_map=component.get("key_map", {}),
            extra_kwargs=component.get("extra_kwargs", {}),
        )
        self.vae = vae

        if self.component_dtypes and "vae" in self.component_dtypes:
            self.to_dtype(vae, self.component_dtypes["vae"])
        if self.vae_tiling:
            self.enable_vae_tiling()
        if self.vae_slicing:
            self.enable_vae_slicing()
        vae = vae.eval()

        # Apply memory management if configured
        self._maybe_apply_memory_management(component, vae)
        return vae

    def enable_vae_tiling(self):
        self.vae_tiling = True
        if self.vae is None:
            return
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        else:
            self.logger.warning("VAE does not support tiling")

    def enable_vae_slicing(self):
        self.vae_slicing = True
        if self.vae is None:
            return
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        else:
            self.logger.warning("VAE does not support slicing")

    def load_config_by_type(self, component_type: str):
        with accelerate.init_empty_weights():
            # check if the component is already loaded
            if getattr(self, component_type, None) is not None:
                return getattr(self, component_type).config
            for component in self.config.get("components", []):
                if component.get("type") == component_type:
                    component = self.load_component(
                        component,
                        (
                            self.component_load_dtypes.get(component.get("type"))
                            if self.component_load_dtypes
                            else None
                        ),
                        no_weights=True,
                    )
                    
                    config = getattr(component, "config", {})

                    if config:
                        return config
                    else:
                        return {}
            raise ValueError(f"Component type {component_type} not found")
        
    def load_config_by_name(self, component_name: str):
        with accelerate.init_empty_weights():
            # check if the component is already loaded
            if getattr(self, component_name, None) is not None:
                return getattr(self, component_name).config
            for component in self.config.get("components", []):
                if component.get("name") == component_name:
                    component = self.load_component(
                        component,
                        (
                            self.component_load_dtypes.get(component.get("type"))
                            if self.component_load_dtypes
                            else None
                        ),
                        no_weights=True,
                    )
                    config = getattr(component, "config", {})

                    if config:
                        return config
                    else:
                        return {}
            raise ValueError(f"Component name {component_name} not found")

    def load_text_encoder(self, component: Dict[str, Any], no_weights: bool = False):
        component["load_dtype"] = self.component_load_dtypes.get("text_encoder", None)
        component["dtype"] = self.component_dtypes.get("text_encoder", None)
        text_encoder = TextEncoder(component, no_weights, device=self.device)

        # Lazily wrap its internal model once loaded, if memory management is configured
        mm_config = self._resolve_memory_config_for_component(component)
        if mm_config is not None:
            original_load_model = text_encoder.load_model

            def _patched_load_model(no_weights: bool = False, *args, **kwargs):
                model = original_load_model(no_weights=no_weights, *args, **kwargs)
                # Only wrap once per instance
                if not hasattr(text_encoder, "_mm_wrapped") or not getattr(
                    text_encoder, "_mm_wrapped"
                ):
                    manager = self._get_or_create_memory_manager("text_encoder", mm_config)
                    wrapped_model = manager.wrap_model(model, layer_types=[nn.Linear])
                    text_encoder.model = wrapped_model
                    setattr(text_encoder, "_mm_wrapped", True)
                return text_encoder.model

            text_encoder.load_model = _patched_load_model  # type: ignore

        return text_encoder

    def load_transformer(
        self,
        component: Dict[str, Any],
        load_dtype: torch.dtype | mx.Dtype | None,
        no_weights: bool = False,
    ):

        component["model_path"], is_converted = self._check_convert_model_path(
            component
        )

        if is_converted:
            component.pop("extra_model_paths", None)

        if self.check_weights and not self._check_weights(component):
            self.logger.info(f"Found old model weights, converting to diffusers format")
            transformer = self.convert_transformer_weights(component)

            if self.save_converted_weights:
                self.logger.info(
                    f"Saving converted transformer weights to {component.get('model_path', None)}"
                )

                model_path = component.get("model_path", None)

                tmp_dir = tempfile.mkdtemp(dir=DEFAULT_CACHE_PATH)
                component_name = component.get("name", component.get("type"))
                try:
                    self.save_component(
                        transformer,
                        tmp_dir,
                        component_name,
                        **self.config.get("save_kwargs", {}),
                    )

                    if os.path.isdir(model_path):
                        # Safely copy to preserve other files in the directory
                        # add subdirectory called transformer
                        transformer_dir = os.path.join(model_path, component_name)
                        os.makedirs(transformer_dir, exist_ok=True)
                        os.rename(tmp_dir, transformer_dir)
                    else:
                        # Atomically replace file with directory
                        shutil.rmtree(model_path, ignore_errors=True)
                        model_dir = os.path.dirname(model_path)
                        if component.get("converted_model_path"):
                            transformer_path = component["converted_model_path"]
                            # check if absolute path or relative path
                            if not os.path.isabs(transformer_path):
                                transformer_path = os.path.join(
                                    model_dir, transformer_path
                                )
                        else:
                            transformer_path = os.path.join(model_dir, component_name)
                        os.makedirs(transformer_path, exist_ok=True)
                        os.rename(tmp_dir, transformer_path)
                finally:
                    # Clean up temp dir if it still exists
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                self.logger.info(f"Converted transformer weights to diffusers format")
                empty_cache()
        else:
            base = component.get("base")
            if base.startswith("mlx."):
                registry = TRANSFORMERS_REGISTRY_MLX
                dtype_converter = convert_dtype_to_mlx
                component["base"] = base.replace("mlx.", "")
            else:
                registry = TRANSFORMERS_REGISTRY_TORCH
                dtype_converter = convert_dtype_to_torch
            
            print(component)
            transformer = self._load_model(
                component,
                registry.get,
                "Transformer",
                dtype_converter(load_dtype) if load_dtype else None,
                no_weights,
                key_map=component.get("key_map", {}),
                extra_kwargs=component.get("extra_kwargs", {}),
            )

        if self.component_dtypes and "transformer" in self.component_dtypes:
            if isinstance(transformer, torch.nn.Module):
                self.to_dtype(transformer, self.component_dtypes["transformer"])
            elif isinstance(transformer, mx_nn.Module):
                self.to_mlx_dtype(transformer, self.component_dtypes["transformer"])

        transformer = transformer.eval()

        # Apply memory management if configured
        self._maybe_apply_memory_management(component, transformer)
        return transformer

    def _get_safetensors_keys(
        self, model_path: str, model_key: str | None = None, framework: str = "pt"
    ):
        keys = set()
        with safe_open(model_path, framework=framework, device="cpu") as f:
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
        component_name = component.get("name", component.get("type"))

        model_dir = os.path.dirname(model_path)

        if component.get("converted_model_path"):
            if not os.path.isabs(component["converted_model_path"]):
                component["converted_model_path"] = os.path.join(
                    model_dir, component["converted_model_path"]
                )
            if os.path.isfile(component["converted_model_path"]):
                return component["converted_model_path"], True
            elif os.path.isdir(component["converted_model_path"]):
                return component["converted_model_path"], True

        if os.path.isfile(model_path):
            # check base directory
            if os.path.isdir(os.path.join(os.path.dirname(model_path), component_name)):
                return (
                    os.path.join(os.path.dirname(model_path), component_name),
                    True,
                )
        elif os.path.isdir(model_path):
            if os.path.isdir(os.path.join(model_path, component_name)):
                return os.path.join(model_path, component_name), True
        return model_path, False

    def _check_weights(self, component: Dict[str, Any]):
        assert "model_path" in component, "`model_path` is required"
        model_path = component["model_path"]
        base = component.get("base", None)
        component_type = component.get("type")

        assert component_type in [
            "transformer",
        ], "Only transformer is supported for now"
        file_pattern = component.get("file_pattern", None)

        if model_path.endswith(".gguf"):
            return True  # We don't need to check weights for gguf models

        extensions = tuple(["pt", "bin", "pth", "ckpt", "safetensors"])
        model_key = component.get("model_key", None)
        keys = set()
        extra_model_paths = component.get("extra_model_paths", [])

        config = {}
        config_path = component.get("config_path", None)
        if config_path:
            config = self.fetch_config(config_path)
        if component.get("config", None):
            config.update(component.get("config", {}))
        

        if not config:
            return (
                True  # We can assume that the model is valid is no config is provided
            )

        if os.path.isdir(model_path):
            if file_pattern:
                files = glob(os.path.join(model_path, file_pattern))
            else:
                files = []
                for ext in extensions:
                    files.extend(glob(os.path.join(model_path, f"*.{ext}")))

            if len(files) == 0:
                raise ValueError(
                    f"No files found in {model_path} with pattern {file_pattern}"
                )

            for file in files:
                if file.endswith(".safetensors"):
                    keys.update(
                        self._get_safetensors_keys(
                            file,
                            model_key,
                            framework="np" if self.engine_type == "mlx" else "pt",
                        )
                    )

                elif file.endswith(extensions):
                    keys.update(self._get_pt_keys(file, model_key))
        else:
            if model_path.endswith(".safetensors"):
                keys.update(
                    self._get_safetensors_keys(
                        model_path,
                        model_key,
                        framework="np" if self.engine_type == "mlx" else "pt",
                    )
                )
            elif model_path.endswith(extensions):
                keys.update(self._get_pt_keys(model_path, model_key))

        for extra_model_path in extra_model_paths:
            # check if is dir
            if os.path.isdir(extra_model_path):
                if file_pattern:
                    files = glob(os.path.join(extra_model_path, file_pattern))
                else:
                    files = []
                    for ext in extensions:
                        files.extend(glob(os.path.join(extra_model_path, f"*.{ext}")))
                    for file in files:
                        if file.endswith(".safetensors"):
                            keys.update(
                                self._get_safetensors_keys(
                                    file,
                                    model_key,
                                    framework=(
                                        "np" if self.engine_type == "mlx" else "pt"
                                    ),
                                )
                            )
                        elif file.endswith(extensions):
                            keys.update(self._get_pt_keys(file, model_key))
            else:
                if extra_model_path.endswith(".safetensors"):
                    keys.update(
                        self._get_safetensors_keys(
                            extra_model_path,
                            model_key,
                            framework="np" if self.engine_type == "mlx" else "pt",
                        )
                    )
                elif extra_model_path.endswith(extensions):
                    keys.update(self._get_pt_keys(extra_model_path, model_key))

        if component_type == "transformer":
            _keys = set(
                get_transformer_keys(
                    base,
                    config,
                )
            )
            iterating_keys = _keys.copy()
        found_keys = set()
        for iterating_key in iterating_keys:
            for key in keys:
                if key == iterating_key:
                    found_keys.add(iterating_key)
                    break

        missing_keys = iterating_keys - found_keys
        return len(missing_keys) == 0

    def convert_transformer_weights(self, component: Dict[str, Any]):
        assert "model_path" in component, "`model_path` is required"
        component_type = component.get("type")
        assert component_type == "transformer", "Only transformer is supported for now"
        self.logger.info(f"Converting old model weights to diffusers format")
        model_path = component["model_path"]

        if component.get("extra_model_paths", []):
            extra_model_paths = component.get("extra_model_paths", [])
            if isinstance(extra_model_paths, str):
                extra_model_paths = [extra_model_paths]
            model_path = [model_path] + extra_model_paths

        # try to load config
        config = {}
        config_path = component.get("config_path", None)
        if config_path:
            config = self.fetch_config(config_path)
        if component.get("config", None):
            config.update(component.get("config", {}))

        return convert_transformer(
            config,
            component["base"],
            model_path,
            component.get("model_key", None),
            component.get("file_pattern", None),
        )

    def convert_vae_weights(self, component: Dict[str, Any]):
        assert "model_path" in component, "`model_path` is required"
        component_type = component.get("type")
        assert component_type == "vae", "Only vae is supported for now"
        self.logger.info(f"Converting old model weights to diffusers format")
        config = {}
        config_path = component.get("config_path", None)
        if config_path:
            config = self.fetch_config(config_path)
        if component.get("config", None):
            config.update(component.get("config", {}))

        return convert_vae(
            config,
            component["base"],
            (
                component["model_path"]
                if not component.get("extra_model_paths", [])
                else [component["model_path"]] + component.get("extra_model_paths", [])
            ),
            component.get("model_key", None),
            component.get("file_pattern", None),
            **component.get("converter_kwargs", {}),
        )

    @torch.no_grad()
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

    @torch.no_grad()
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
            if components_to_load and (
                component.get("type") in components_to_load
                or component.get("name") in components_to_load
            ):
                component_module = self.load_component(
                    component,
                    (
                        self.component_load_dtypes.get(component.get("type"))
                        if self.component_load_dtypes
                        else None
                    ),
                )
                # Set for both type and name
                setattr(self, component.get("type"), component_module)
                if component.get("name"):
                    setattr(self, component.get("name"), component_module)

    # -------------------------
    # Memory management helpers
    # -------------------------
    def _normalize_memory_management(
        self, spec: Optional[Dict[str, Union[str, MemoryConfig]]]
    ) -> Optional[Dict[str, MemoryConfig]]:
        if not spec:
            return None

        normalized: Dict[str, MemoryConfig] = {}

        def to_config(v: Union[str, MemoryConfig]) -> MemoryConfig:
            if isinstance(v, MemoryConfig):
                return v
            preset = str(v).strip().lower().replace(" ", "_")
            if preset in {"low", "low_memory", "low-memory"}:
                return MemoryConfig.for_low_memory()
            if preset in {"high", "high_performance", "high-performance"}:
                return MemoryConfig.for_high_performance()
            raise ValueError(
                f"Unknown memory preset '{v}'. Use 'low_memory' or 'high_performance' or pass MemoryConfig."
            )

        for key, value in spec.items():
            try:
                normalized[key] = to_config(value)
            except Exception as e:
                self.logger.warning(f"Invalid memory_management entry for '{key}': {e}")

        return normalized if normalized else None

    def _resolve_memory_config_for_component(
        self, component: Dict[str, Any]
    ) -> Optional[MemoryConfig]:
        if not self._memory_management_map:
            return None
        name = component.get("name")
        ctype = component.get("type")
        # Prefer explicit name mapping, fallback to type mapping, then 'all'
        if name and name in self._memory_management_map:
            return self._memory_management_map[name]
        if ctype and ctype in self._memory_management_map:
            return self._memory_management_map[ctype]
        if "all" in self._memory_management_map:
            return self._memory_management_map["all"]
        return None

    def _get_or_create_memory_manager(
        self, key: str, config: MemoryConfig
    ) -> MemoryManager:
        if key in self._component_memory_managers:
            return self._component_memory_managers[key]
        manager = MemoryManager(config)
        manager.start()
        self._component_memory_managers[key] = manager
        return manager

    def _maybe_apply_memory_management(self, component: Dict[str, Any], module: Any):
        mm_config = self._resolve_memory_config_for_component(component)
        if mm_config is None:
            return
        key = component.get("name") or component.get("type")
        manager = self._get_or_create_memory_manager(key, mm_config)
        try:
            wrapped = manager.wrap_model(module, layer_types=[nn.Linear])
        except Exception as e:
            self.logger.warning(f"Failed to wrap component '{key}' for memory management: {e}")
            return
        # Replace references on engine for known types
        ctype = component.get("type")
        if ctype in {"transformer", "vae", "text_encoder"}:
            setattr(self, ctype, wrapped)

    def load_component_by_type(self, component_type: str):
        for component in self.config.get("components", []):
            if component.get("type") == component_type:
                component_module = self.load_component(
                    component,
                    (
                        self.component_load_dtypes.get(component.get("type"))
                        if self.component_load_dtypes
                        else None
                    ),
                )
                setattr(self, component.get("type"), component_module)
                break

    def load_component_by_name(self, component_name: str):
        for component in self.config.get("components", []):
            if component.get("name") == component_name:
                component_module = self.load_component(
                    component,
                    (
                        self.component_load_dtypes.get(component.get("type"))
                        if self.component_load_dtypes
                        else None
                    ),
                )
                setattr(self, component.get("name"), component_module)
                break

    def load_preprocessor(self, config: Dict[str, Any]):
        preprocessor_type = config.get("type")
        preprocessor_class = preprocessor_registry.get(preprocessor_type)
        if preprocessor_class is None:
            raise ValueError(f"Preprocessor type {preprocessor_type} not supported")
        kwargs = config.get("kwargs", {})
        preprocessor = preprocessor_class(**config)
        self._preprocessors[preprocessor_type] = {
            "preprocessor": preprocessor,
            "kwargs": kwargs,
        }
        return preprocessor

    def load_preprocessors(
        self,
        preprocessors: List[Dict[str, Any]],
        preprocessors_to_load: List[str] | None = None,
    ):
        for preprocessor in preprocessors:
            if (
                preprocessors_to_load is None
                or (preprocessor.get("type") in preprocessors_to_load)
                or (preprocessor.get("name") in preprocessors_to_load)
            ):
                self.load_preprocessor(preprocessor)

    def load_postprocessor(self, config: Dict[str, Any]):
        postprocessor_type = config.get("type")
        if postprocessor_type in self._postprocessors:
            return self._postprocessors[postprocessor_type]
        postprocessor_class = postprocessor_registry.get(postprocessor_type)
        if postprocessor_class is None:
            raise ValueError(f"Postprocessor type {postprocessor_type} not supported")
        # check if engine is part of signature of postprocessor_class
        if "engine" in inspect.signature(postprocessor_class).parameters:
            config["engine"] = self

        kwargs = config.get("kwargs", {})
        postprocessor = postprocessor_class(**config)
        self._postprocessors[postprocessor_type] = {
            "postprocessor": postprocessor,
            "kwargs": kwargs,
        }
        return postprocessor

    def load_postprocessors(
        self,
        postprocessors: List[Dict[str, Any]],
        postprocessors_to_load: List[str] | None = None,
    ):

        for postprocessor in postprocessors:
            if (
                postprocessors_to_load is None
                or (postprocessor.get("type") in postprocessors_to_load)
                or (postprocessor.get("name") in postprocessors_to_load)
            ):
                self.load_postprocessor(postprocessor)

    def apply_lora(self, lora_path: str):
        # Backward-compat shim: allow direct single-path call
        if self.transformer is None:
            self.load_component_by_type("transformer")
        self.apply_loras([lora_path])

    def _init_lora_manager(self, save_dir: str):
        try:
            self.lora_manager = LoraManager(save_dir)
        except Exception as e:
            self.logger.warning(f"Failed to initialize LoraManager: {e}")
            self.lora_manager = None

    def apply_loras(
        self,
        loras: List[Union[str, LoraItem, tuple]],
        adapter_names: List[str] | None = None,
        scales: List[float] | None = None,
        model_name_or_type: str  = "transformer",
    ):
        """
        Apply one or multiple LoRAs to the current transformer using PEFT backend.
        Each entry in `loras` may be a source string, a LoraItem, or (source|LoraItem, scale).
        """
        if model_name_or_type == "transformer":
            if self.transformer is None:
                self.load_component_by_type("transformer")
        else:
            if getattr(self, model_name_or_type) is None:
                self.load_component_by_name(model_name_or_type)
            
        model = getattr(self, model_name_or_type)

        if self.lora_manager is None:
            self._init_lora_manager(DEFAULT_LORA_SAVE_PATH)
        if self.lora_manager is None:
            raise RuntimeError("LoraManager is not available")
        
        resolved = self.lora_manager.load_into(
            model, loras, adapter_names=adapter_names, scales=scales
        )
        # Track by adapter name
        for i, item in enumerate(resolved):
            name = (
                adapter_names[i]
                if adapter_names and i < len(adapter_names)
                else item.name or f"lora_{i}"
            )
            self.loaded_loras[name] = item
        self.logger.info(f"Applied {len(resolved)} LoRA(s) to transformer")

    def _auto_apply_loras(self, model_name_or_type: str = "transformer"):
        """If the YAML config includes a top-level `loras` list, apply them on init.
        Supported formats:
        - ["source1", "source2"]
        - [{"source": "...", "scale": 0.8, "name": "style"}, ...]
        """
        loras_cfg = self.config.get("loras", None)
        if not loras_cfg:
            return
        formatted: List[Union[str, LoraItem, tuple]] = []
        adapter_names: List[str] = []
        for entry in loras_cfg:
            if isinstance(entry, str):
                formatted.append(entry)
                adapter_names.append(None)
            elif isinstance(entry, dict):
                src = entry.get("source") or entry.get("path") or entry.get("url")
                scale = float(entry.get("scale", 1.0))
                name = entry.get("name")
                if name is not None:
                    adapter_names.append(name)
                else:
                    adapter_names.append(None)
                formatted.append((src, scale))
        # remove None names at end so we can pass None overall if all None
        final_names = (
            adapter_names if any(n is not None for n in adapter_names) else None
        )
        try:
            self.apply_loras(formatted, adapter_names=final_names, model_name_or_type=model_name_or_type)
        except Exception as e:
            self.logger.warning(f"Auto-apply LoRAs failed: {e}")

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
            components_path = DEFAULT_COMPONENTS_PATH
        if preprocessors_path is None:
            preprocessors_path = DEFAULT_PREPROCESSOR_SAVE_PATH
        if postprocessors_path is None:
            postprocessors_path = DEFAULT_POSTPROCESSOR_SAVE_PATH

        os.makedirs(save_path, exist_ok=True)
        for i, component in enumerate(self.config.get("components", []) ):
            save_path = component.get("save_path", components_path)
            if config_path := component.get("config_path"):
                downloaded_config_path = self.fetch_config(
                    config_path, return_path=True
                )
                if downloaded_config_path:
                    component["config_path"] = downloaded_config_path
                    
            component_type = component.get("type")
            component_name = component.get("name")
            
            if component_type == "scheduler":
                scheduler_options = component.get("scheduler_options")
                selected_scheduler_option = self.selected_components.get(component_name, self.selected_components.get(component_type, None))
                for scheduler_option in scheduler_options:
                    if selected_scheduler_option['name'] == scheduler_option['name']:
                        component = selected_scheduler_option.copy()
                        component['type'] = 'scheduler'
            else:
                model_path = component.get("model_path")
                if isinstance(model_path, list):
                    selected_model_item = self.selected_components.get(component_name, self.selected_components.get(component_type, None))
                    for model_path_item in model_path:
                        if selected_model_item.get('variant') == model_path_item.get('variant'):
                            component['model_path'] = selected_model_item.get('path')
                            break
                        
                else:
                    downloaded_model_path = self._download(model_path, save_path)
                    if downloaded_model_path:
                        component["model_path"] = downloaded_model_path
                
            if extra_model_paths := component.get("extra_model_paths"):
                for i, extra_model_path in enumerate(extra_model_paths):
                    downloaded_extra_model_path = self._download(
                        extra_model_path, save_path
                    )
                    if downloaded_extra_model_path:
                        component["extra_model_paths"][i] = downloaded_extra_model_path
            self.config["components"][i] = component

        preprocessors = self.config.get("preprocessors", []) or []
        if preprocessors:
            self.logger.info(f"Downloading {len(preprocessors)} preprocessors")
        for preprocessor in preprocessors:
            if preprocessor_path := preprocessor.get("model_path"):
                save_path = preprocessor.get("save_path", preprocessors_path)
                downloaded_preprocessor_path = self._download(
                    preprocessor_path, save_path
                )
                if downloaded_preprocessor_path:
                    preprocessor["model_path"] = downloaded_preprocessor_path
            if config_path := preprocessor.get("config_path"):
                downloaded_config_path = self.fetch_config(
                    config_path, return_path=True
                )
                if downloaded_config_path:
                    preprocessor["config_path"] = downloaded_config_path

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
            if config_path := postprocessor.get("config_path"):
                downloaded_config_path = self.fetch_config(
                    config_path, return_path=True
                )
                if downloaded_config_path:
                    postprocessor["config_path"] = downloaded_config_path

    def _get_latents(
        self,
        height: int,
        width: int,
        duration: int | str,
        fps: int = 16,
        batch_size: int = 1,
        num_channels_latents: int = None,
        vae_scale_factor_spatial: int = None,
        vae_scale_factor_temporal: int = None,
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
            latent_num_frames = (num_frames - 1) // (
                vae_scale_factor_temporal or self.vae_scale_factor_temporal
            ) + 1
        else:
            latent_num_frames = duration

        latent_height = height // (
            vae_scale_factor_spatial or self.vae_scale_factor_spatial
        )
        latent_width = width // (
            vae_scale_factor_spatial or self.vae_scale_factor_spatial
        )

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
                batch_size,
                num_channels_latents or self.num_channels_latents,
                latent_num_frames,
                latent_height,
                latent_width,
            )
        elif order == "BFC":
            shape = (
                batch_size,
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
        rendered_video = self._tensor_to_frames(video)
        render_on_step_callback(rendered_video)

    def _tensor_to_frames(self, video: torch.Tensor, output_type: str = "pil"):
        postprocessed_video = self.video_processor.postprocess_video(
            video, output_type=output_type
        )
        return postprocessed_video

    def _tensor_to_frame(self, image: torch.Tensor, output_type: str = "pil"):
        if hasattr(self, "image_processor"):
            postprocessed_frame = self.image_processor.postprocess(
                image, output_type=output_type
            )
        else:
            postprocessed_frame = self.video_processor.postprocess(
                image, output_type=output_type
            )
        return postprocessed_frame

    def _get_timesteps(
        self,
        scheduler: SchedulerMixin | None = None,
        num_inference_steps: Optional[int] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        timesteps_as_indices: bool = False,
        strength: float = 1.0,
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

        if strength != 1.0:
            init_timestep = min(
                int(num_inference_steps * strength), num_inference_steps
            )
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = timesteps[t_start * self.scheduler.order :]
            num_inference_steps = len(timesteps)

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

    def __str__(self):
        return f"BaseEngine(config={self.config}, device={self.device})"

    def __repr__(self):
        return self.__str__()
    
    
