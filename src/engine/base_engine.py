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
import psutil
from src.utils.defaults import DEFAULT_CACHE_PATH, get_components_path, get_preprocessor_path, get_postprocessor_path, get_offload_path
from src.utils.module import find_class_recursive
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from src.transformer.base import TRANSFORMERS_REGISTRY as TRANSFORMERS_REGISTRY_TORCH
from src.mlx.transformer.base import TRANSFORMERS_REGISTRY as TRANSFORMERS_REGISTRY_MLX
from src.text_encoder.text_encoder import TextEncoder
from src.vae import get_vae
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.video_processor import VideoProcessor
from src.utils.dtype import select_ideal_dtypes
from src.attention import attention_register
from src.utils.cache import empty_cache
from logging import Logger
from src.scheduler import SchedulerInterface
from typing import Callable
from src.utils.mlx import convert_dtype_to_torch, convert_dtype_to_mlx
from src.memory_management import MemoryConfig
import torch.nn as nn
import importlib
from diffusers.utils.torch_utils import randn_tensor
import traceback
from src.utils.defaults import (
    DEFAULT_DEVICE,
    DEFAULT_CONFIG_SAVE_PATH,
    DEFAULT_SAVE_PATH,
    DEFAULT_LORA_SAVE_PATH,
)
from src.utils.compute import validate_compute_requirements, get_compute_capability

import tempfile
from src.transformer import _auto_register_transformers
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
from src.lora import LoraManager, LoraItem
from src.helpers.helpers import helpers
_auto_register_transformers()

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
        from src.helpers.hunyuanvideo.llama import HunyuanLlama
        from src.helpers.hunyuanvideo.avatar import HunyuanAvatar
        from src.helpers.hunyuanvideo15.vision import HunyuanVisionEncoder
        from src.helpers.hunyuanvideo15.text import TextEncoder as HunyuanTextEncoder
        from src.helpers.hidream.llama import HidreamLlama
        from src.helpers.stepvideo.text_encoder import StepVideoTextEncoder
        from src.helpers.ltx.patchifier import SymmetricPatchifier
        from src.helpers.fibo.prompt_gen import PromptGenHelper
        from src.helpers.wan.humo_audio_processor import HuMoAudioProcessor

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
        def __getitem__(self, key: Literal["hunyuanvideo.llama"]) -> "HunyuanLlama": ...

        @overload
        def __getitem__(self, key: Literal["hunyuanvideo.avatar"]) -> "HunyuanAvatar": ...

        @overload
        def __getitem__(self, key: Literal["hunyuanvideo15.vision"]) -> "HunyuanVisionEncoder": ...

        @overload
        def __getitem__(self, key: Literal["hunyuanvideo15.text"]) -> "HunyuanTextEncoder": ...

        @overload
        def __getitem__(self, key: Literal["prompt_gen"]) -> "PromptGenHelper": ...

        @overload
        def __getitem__(self, key: Literal["hidream.llama"]) -> "HidreamLlama": ...

        @overload
        def __getitem__(self, key: Literal["stepvideo.text_encoder"]) -> "StepVideoTextEncoder": ...

        @overload
        def __getitem__(self, key: Literal["ltx.patchifier"]) -> "SymmetricPatchifier": ...

        @overload
        def __getitem__(self, key: Literal["wan.humo_audio_processor"]) -> "HuMoAudioProcessor": ...
        
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
        else:
            return None
    
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
    save_converted_weights: bool = False
    vae_scale_factor_temporal: float = 1.0
    vae_scale_factor_spatial: float = 1.0
    vae_scale_factor: float = 1.0
    num_channels_latents: int = 4
    denoise_type: str | None = None
    vae_tiling: bool =  False
    vae_slicing: bool = False
    lora_manager: LoraManager | None = None
    loaded_loras: Dict[str, LoraItem] = {}
    _memory_management_map: Dict[str, MemoryConfig] | None = None
    selected_components: Dict[str, Any] | None = None
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Wrap subclass __init__ so that `post_init` is called automatically
        # after the subclass' own initialization logic completes.
        if cls is BaseEngine:
            return

        # Avoid double‑wrapping the same subclass
        if getattr(cls, "_apex_post_init_wrapped", False):
            return

        original_init = cls.__init__

        def __init_wrapper(self, *args, **init_kwargs):
            # Run the original __init__ chain (including BaseEngine.__init__)
            original_init(self, *args, **init_kwargs)

            # Ensure post_init runs exactly once per instance, at the very end
            if getattr(self, "_apex_post_init_ran", False):
                return

            post_init = getattr(self, "post_init", None)
            if callable(post_init):
                post_init()
                setattr(self, "_apex_post_init_ran", True)

        cls.__init__ = __init_wrapper
        cls._apex_post_init_wrapped = True

    def __init__(
        self,
        yaml_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ):
        self.device = device
        self._helpers = AutoLoadingHelperDict(self)
        # Keep a copy of the original initialization kwargs so that
        # `post_init` can access them after all subclasses have finished
        # their own initialization.
        self._init_kwargs = dict(kwargs)
        self._init_logger()
        self.config = self._load_yaml(yaml_path)
        
        # Validate compute requirements if specified in config
        self._validate_compute_requirements()

        self.save_path = kwargs.get("save_path", None)
        should_download = kwargs.get("should_download", True)

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
                
        if not hasattr(self, "selected_components") or self.selected_components is None:
            self.selected_components = {}
        
        if should_download:
            self.download(self.save_path)
        

        # Normalize optional memory management mapping.
        # If no explicit mapping is provided, we will try to infer sensible
        # defaults based on the size of the models that will be loaded.
        
        
        self._parse_config(
            self.config,
            kwargs.get("config_kwargs", {}),
            kwargs.get("components_to_load", None),
            kwargs.get("component_load_dtypes", None),
            kwargs.get("component_dtypes", None)
        )

        self.attention_type = kwargs.get("attention_type", "sdpa")
        attention_register.set_default(self.attention_type)
        if should_download:
            self._init_lora_manager(kwargs.get("lora_save_path", DEFAULT_LORA_SAVE_PATH))
        if should_download and kwargs.get("auto_apply_loras", True):
            self._auto_apply_loras(kwargs.get("lora_model_name_or_type", "transformer"))

    def post_init(self):
        """
        Run final setup that depends on fully-initialized subclasses.

        By default this normalizes the optional memory-management mapping
        using the original kwargs passed at construction time.
        """
        init_kwargs = getattr(self, "_init_kwargs", {}) or {}
        memory_spec = init_kwargs.get("memory_management", None)
        self._memory_management_map = self._normalize_memory_management(memory_spec)

    def _init_logger(self):
        self.logger = logger

    def _validate_compute_requirements(self):
        """Validate that the current system meets the compute requirements specified in the config."""
        compute_requirements = self.config.get("compute_requirements")
        if not compute_requirements:
            return
        
        is_valid, error_message = validate_compute_requirements(compute_requirements)
        if not is_valid:
            current_cap = get_compute_capability()
            error_detail = (
                f"\n\nCompute Validation Failed:\n"
                f"  {error_message}\n\n"
                f"Current System:\n"
                f"  Compute Type: {current_cap.compute_type}\n"
            )
            if current_cap.compute_type == "cuda":
                error_detail += f"  CUDA Capability: {current_cap.cuda_compute_capability}\n"
                error_detail += f"  Device: {current_cap.device_name}\n"
            elif current_cap.compute_type == "metal":
                error_detail += f"  Metal Version: {current_cap.metal_version}\n"
            else:
                error_detail += f"  Platform: {current_cap.cpu_info.get('platform', 'unknown')}\n"
            
            error_detail += f"\nRequired:\n"
            if compute_requirements.get("min_cuda_compute_capability"):
                error_detail += f"  Min CUDA Capability: {compute_requirements['min_cuda_compute_capability']}\n"
            if compute_requirements.get("supported_compute_types"):
                error_detail += f"  Supported Types: {', '.join(compute_requirements['supported_compute_types'])}\n"
            
            self.logger.error(error_detail)
            raise RuntimeError(error_detail)

    def _aspect_ratio_resize(
        self,
        image,
        max_area=720 * 1280,
        mod_value=16,
        resize_mode=Image.Resampling.LANCZOS,
    ):
        if max_area is None:
            max_area = 720 * 1280
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
        image = TF.CenterCrop(size)(image)
        return image, height, width

    def _parse_config(
        self,
        config: Dict[str, Any],
        config_kwargs: Dict[str, Any],
        components_to_load: List[str] | None,
        component_load_dtypes: Dict[str, torch.dtype] | None,
        component_dtypes: Dict[str, str] | None,
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
    
    def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

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
            logger.info(f"Loading transformer component: {component}")
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
        except Exception:
            helper_class = find_class_recursive(importlib.import_module(module), base)
        if helper_class is None:
            raise ValueError(f"Helper class {base} not found")

        # create an instance of the helper class
        if hasattr(helper_class, "from_pretrained") and "model_path" in config:
            helper = helper_class.from_pretrained(config["model_path"])
        else:
            # check for config_path
            if "config_path" in config:
                config_path = config.pop("config_path")
                # try download the config file
                try:
                    config_path = self._download(config_path, get_components_path())
                except Exception:
                    pass
                config = self._load_config_file(config_path)
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
                        import traceback
                        self.logger.error(traceback.format_exc())
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
        if self.component_dtypes and "vae" in self.component_dtypes:
            self.to_dtype(vae, self.component_dtypes["vae"])
        if self.vae_tiling:
            self.enable_vae_tiling()
        if self.vae_slicing:
            self.enable_vae_slicing()
        vae = vae.eval()
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
                    config = {}
                    if isinstance(component, TextEncoder):
                        if hasattr(component, "model"):
                            if hasattr(component.model.config, "to_dict"):
                                model_config = component.model.config.to_dict()
                            else:
                                model_config = dict(component.model.config)
                            config.update(model_config)
                    
                    component_config = getattr(component, "config", {})
                    if hasattr(component_config, "to_dict"):
                        component_config = component_config.to_dict()
                    else:
                        component_config = dict(component_config)
                    config.update(component_config)

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
                text_encoder.model = model

                if no_weights:
                    return text_encoder.model

                already_enabled = getattr(text_encoder, "_group_offloading_enabled", False)
                if not already_enabled:
                    self._apply_group_offloading(
                        text_encoder.model,
                        mm_config,
                        module_label=component.get("name") or "text_encoder",
                    )
                    setattr(text_encoder, "_group_offloading_enabled", True)
                return text_encoder.model

            text_encoder.load_model = _patched_load_model  # type: ignore

        if no_weights:
            model = text_encoder.load_model(no_weights=True)
            text_encoder.model = model
        
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
            self.logger.info("Found old model weights, converting to diffusers format")
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
                self.logger.info("Converted transformer weights to diffusers format")
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
        if component["model_path"].endswith(".gguf"):
            return component["model_path"], False
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
                return model_path, False

        elif os.path.isdir(model_path):
            if os.path.isdir(os.path.join(model_path, component_name)):
                return os.path.join(model_path, component_name), True
        return model_path, False

    # -------------------------
    # Model size estimation helpers
    # -------------------------
    def _estimate_component_model_size_bytes(self, component: Dict[str, Any]) -> int:
        """Roughly estimate total on-disk size of the model weights for a component."""
        model_path = component.get("model_path")
        if not model_path:
            return 0
        
        # First, estimate raw size of weight files on disk
        total_size = 0

        # Normalize to list so we can handle main + extra paths uniformly
        paths: List[str] = [model_path]
        extra_model_paths = component.get("extra_model_paths", [])
        if isinstance(extra_model_paths, str):
            extra_model_paths = [extra_model_paths]
        if isinstance(extra_model_paths, list):
            paths.extend(extra_model_paths)

        # Common weight file extensions
        extensions = ("safetensors", "bin", "pt", "pth", "ckpt", "gguf")

        for path in paths:
            if not path:
                continue
            try:
                if os.path.isdir(path):
                    for ext in extensions:
                        for f in glob(os.path.join(path, f"*.{ext}")):
                            try:
                                total_size += os.path.getsize(f)
                            except OSError:
                                continue
                elif os.path.isfile(path):
                    # Only count files that look like weights
                    if path.endswith(extensions):
                        try:
                            total_size += os.path.getsize(path)
                        except OSError:
                            continue
            except Exception:
                # This is a best-effort heuristic; ignore unexpected errors
                continue

        # Adjust the estimate based on the configured in-memory dtype for this component,
        # since weights may be stored as fp32 on disk but used as fp16 / bf16 in memory.
        component_type = component.get("type")
        target_dtype = None
        if getattr(self, "component_dtypes", None) is not None and component_type:
            target_dtype = self.component_dtypes.get(component_type)

        try:
            if target_dtype is not None:
                target_element_size = torch.tensor([], dtype=target_dtype).element_size()
                # Assume on-disk weights are fp32 by default when estimating memory usage.
                source_element_size = torch.tensor([], dtype=torch.float32).element_size()
                if source_element_size > 0:
                    scale = float(target_element_size) / float(source_element_size)
                    return int(total_size * scale)
        except Exception:
            # If anything goes wrong with dtype-based scaling, fall back to the raw size.
            pass

        return total_size

    def _estimate_block_structure(self, component: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
        """
        Load an empty model and estimate the size and count of offloadable blocks.
        
        Returns:
            tuple: (block_size_bytes, num_blocks) or (None, None) if estimation fails
        """
        ctype = component.get("type")
        if ctype not in {"transformer", "vae", "text_encoder"}:
            return None, None
        
        try:
            with accelerate.init_empty_weights():
                # Load component without weights to inspect structure
                temp_module = self.load_component(component, no_weights=True)
                
                # Get the actual module to inspect
                if ctype == "text_encoder" and hasattr(temp_module, "model"):
                    inspect_module = temp_module.model
                else:
                    inspect_module = temp_module
                
                if inspect_module is None:
                    return None, None
                
                # Try to get offloadable_module from config
                offloadable_module_path = component.get("offloadable_module")
                if offloadable_module_path:
                    # Navigate to the offloadable module
                    parts = offloadable_module_path.split(".")
                    offloadable_module = inspect_module
                    for part in parts:
                        if hasattr(offloadable_module, part):
                            offloadable_module = getattr(offloadable_module, part)
                        else:
                            offloadable_module = None
                            break
                else:
                    offloadable_module = inspect_module
                
                if offloadable_module is None:
                    return None, None
                
                # Count blocks - look for common patterns
                blocks = []
                for name in ["blocks", "layers", "transformer_blocks", "down_blocks", "up_blocks"]:
                    if hasattr(offloadable_module, name):
                        attr = getattr(offloadable_module, name)
                        if isinstance(attr, (list, nn.ModuleList)):
                            blocks.extend(list(attr))
                
                if not blocks:
                    return None, None
                
                # Estimate size of first block as representative
                first_block = blocks[0]

                # Count parameters in the representative block
                num_params = sum(
                    p.numel() for p in first_block.parameters() if hasattr(p, "numel")
                )

                # Use the configured component dtype (fp16 / bf16 / etc.) to
                # estimate memory instead of the default (often fp32) dtype
                target_dtype = None
                if getattr(self, "component_dtypes", None) is not None:
                    target_dtype = self.component_dtypes.get(ctype)

                try:
                    if target_dtype is not None:
                        element_size = torch.tensor([], dtype=target_dtype).element_size()
                    else:
                        # Fallback: assume fp16-style 2-byte params if dtype is unknown
                        element_size = 2
                except Exception:
                    # Very defensive: if anything goes wrong, fall back to 2 bytes
                    element_size = 2

                block_size_bytes = num_params * element_size
                
                return block_size_bytes, len(blocks)
                
        except Exception as e:
            self.logger.debug(f"Could not estimate block structure for {ctype}: {e}")
            return None, None

    def _get_system_memory_info(self) -> Dict[str, Optional[float]]:
        """Get GPU and CPU memory information in GB."""
        gpu_total_gb: Optional[float] = None
        gpu_available_gb: Optional[float] = None
        cpu_available_gb: Optional[float] = None
        cpu_total_gb: Optional[float] = None
        
        # GPU memory
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                gpu_total_gb = float(props.total_memory) / 1e9
                # Get available memory
                torch.cuda.synchronize()
                gpu_available_gb = (props.total_memory - torch.cuda.memory_allocated(0)) / 1e9
            except Exception:
                pass
        
        # CPU memory
        try:
            vm = psutil.virtual_memory()
            cpu_available_gb = float(vm.available) / 1e9
            cpu_total_gb = float(vm.total) / 1e9
        except Exception:
            pass
        
        return {
            "gpu_total": gpu_total_gb,
            "gpu_available": gpu_available_gb,
            "cpu_available": cpu_available_gb,
            "cpu_total": cpu_total_gb,
        }

    def _determine_memory_strategy(self, component: Dict[str, Any]) -> Optional[MemoryConfig]:
        """
        Determine the optimal memory management strategy for a component.
        
        This analyzes the component's size, block structure, and available memory to decide:
        - Whether memory management is needed
        - Whether to use leaf-level or block-level offloading
        - Whether to offload to CPU or disk
        
        Returns:
            MemoryConfig if memory management is needed, None otherwise
        """
        ctype = component.get("type")
        if ctype not in {"transformer", "vae", "text_encoder"}:
            return None
        
        # Get system memory info
        mem_info = self._get_system_memory_info()
        gpu_total_gb = mem_info["gpu_total"]
        gpu_available_gb = mem_info["gpu_available"]
        cpu_available_gb = mem_info["cpu_available"]
        
        # Estimate total model size
        total_size_bytes = self._estimate_component_model_size_bytes(component)
        if total_size_bytes <= 0:
            return None
        
        total_size_gb = float(total_size_bytes) / 1e9
        
        # No GPU or model doesn't fit? Need offloading
        needs_offload = False
        if gpu_total_gb is None or gpu_total_gb == 0:
            # No GPU available
            return None
        
        if gpu_available_gb is not None:
            # Use available memory for more accurate decision
            if total_size_gb >= 0.75 * gpu_available_gb:
                needs_offload = True
        else:
            # Fallback to total memory
            if total_size_gb >= 0.75 * gpu_total_gb:
                needs_offload = True
        
        if not needs_offload:
            return None
        
        # Estimate block structure for smarter offloading
        block_size_bytes, num_blocks = self._estimate_block_structure(component)
        
        # Decide on offloading strategy
        config = MemoryConfig.for_block_level()
        
        # Determine if we need disk offload
        # Calculate how many blocks will be in CPU memory at once (rough estimate: 2-3 blocks)
        blocks_in_memory = min(3, num_blocks) if num_blocks else 2
        
        if block_size_bytes and num_blocks and cpu_available_gb and num_blocks * block_size_bytes <= (cpu_available_gb * 1e9) * 0.9:
            config.group_offload_low_cpu_mem_usage = False
        
        if block_size_bytes and num_blocks and cpu_available_gb:
            # Estimate memory needed for offloaded blocks
            estimated_cpu_memory_gb = (block_size_bytes * blocks_in_memory) / 1e9
            
            # Add safety margin (20%) and check against available CPU RAM
            if estimated_cpu_memory_gb * 1.2 >= cpu_available_gb * 0.8:
                # Not enough CPU RAM for safe offloading, use disk
                config.group_offload_disk_path = get_offload_path()
                self.logger.info(
                    f"Component {component.get('name') or ctype}: using disk offload "
                    f"(estimated {estimated_cpu_memory_gb:.2f}GB needed, "
                    f"{cpu_available_gb:.2f}GB available)"
                )
        elif cpu_available_gb and total_size_gb >= 0.85 * cpu_available_gb:
            # Fallback: if we don't have block info, use total size
            config.group_offload_disk_path = get_offload_path()
            self.logger.info(
                f"Component {component.get('name') or ctype}: using disk offload "
                f"(model {total_size_gb:.2f}GB, CPU RAM {cpu_available_gb:.2f}GB available)"
            )
        
        return config

    def _auto_memory_management_from_components(self) -> Optional[Dict[str, MemoryConfig]]:
        """
        Infer memory management configuration for all components.
        
        Uses robust analysis including empty model loading, block size calculation,
        and smart CPU/disk offload decisions.
        """
        components = self.config.get("components", [])
        if not components:
            return None

        auto_map: Dict[str, MemoryConfig] = {}

        for comp in components:
            ctype = comp.get("type")
            if ctype not in {"transformer", "vae", "text_encoder"}:
                continue

            strategy = self._determine_memory_strategy(comp)
            if strategy is not None:
                key = comp.get("name") or ctype
                auto_map[key] = strategy

        return auto_map or None

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
        self.logger.info("Converting old model weights to diffusers format")
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
        self.logger.info("Converting old model weights to diffusers format")
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
        component_name: str = "vae",
    ):
        if getattr(self, component_name, None) is None:
            self.load_component_by_type(component_name)
        self.to_device(getattr(self, component_name))
        denormalized_latents = getattr(self, component_name).denormalize_latents(latents).to(
            dtype=getattr(self, component_name).dtype, device=self.device
        )

        video = getattr(self, component_name).decode(denormalized_latents, return_dict=False)[0]
        if offload:
            self._offload(getattr(self, component_name))
        return video.to(dtype=dtype)

    @torch.no_grad()
    def vae_encode(
        self,
        video: torch.Tensor,
        offload: bool = False,
        sample_mode: str = "mode",
        component_name: str = "vae",
        sample_generator: torch.Generator = None,
        dtype: torch.dtype = None,
        normalize_latents: bool = True,
        normalize_latents_dtype: torch.dtype | None = None,
    ):
        if getattr(self, component_name, None) is None:
            self.load_component_by_type("vae")
        self.to_device(getattr(self, component_name))

        video = video.to(dtype=getattr(self, component_name).dtype, device=self.device)

        latents = getattr(self, component_name).encode(video, return_dict=False)[0]
        if sample_mode == "sample":
            latents = latents.sample(generator=sample_generator)
        elif sample_mode == "mode":
            latents = latents.mode()
        else:
            raise ValueError(f"Invalid sample mode: {sample_mode}")

        if not normalize_latents_dtype:
            normalize_latents_dtype = getattr(self, component_name).dtype

        if normalize_latents:
            latents = latents.to(dtype=normalize_latents_dtype)
            latents = getattr(self, component_name).normalize_latents(latents)

        if offload:
            self._offload(getattr(self, component_name))

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
    
    def _has_memory_management_parameters(self, value:Dict[str, Any]) -> bool:
        # if a dict has any of the keys: group_offload_type, group_offload_num_blocks_per_group, group_offload_use_stream, group_offload_record_stream, group_offload_non_blocking, group_offload_low_cpu_mem_usage, group_offload_offload_device, group_offload_disk_path
        return any(key in value for key in ["group_offload_type", 
                                            "group_offload_num_blocks_per_group", 
                                            "group_offload_use_stream", 
                                            "group_offload_record_stream", 
                                            "group_offload_non_blocking", 
                                            "group_offload_low_cpu_mem_usage", 
                                            "group_offload_offload_device", 
                                            "group_offload_disk_path"])
    def _normalize_memory_management(
        self, spec: Optional[Dict[str, Union[str, MemoryConfig, Dict[str, Any]]]]
    ) -> Optional[Dict[str, MemoryConfig]]:
        # Start with any explicit mapping provided by the caller.
        normalized: Dict[str, MemoryConfig] = {}

        if spec:
            def to_config(v: Union[str, MemoryConfig]) -> MemoryConfig:
                if isinstance(v, MemoryConfig):
                    return v
                if self._has_memory_management_parameters(v):
                    return MemoryConfig(**v)
                return MemoryConfig.for_block_level()

            for key, value in spec.items():
                try:
                    normalized[key] = to_config(value)
                except Exception as e:
                    self.logger.warning(
                        f"Invalid memory_management entry for '{key}': {e}"
                    )

        # If nothing explicit was provided (or everything failed to parse),
        # attempt to infer a sensible default mapping from the configured components.
        auto_map = self._auto_memory_management_from_components()
        if auto_map:
            # Do not overwrite any explicit entries; only fill in missing keys.
            for key, cfg in auto_map.items():
                if key not in normalized:
                    normalized[key] = cfg

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

    def _group_offload_onload_device(self) -> torch.device:
        candidate = getattr(self, "device", None)
        if isinstance(candidate, torch.device):
            return candidate
        if isinstance(candidate, str):
            try:
                return torch.device(candidate)
            except Exception:
                pass
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        return torch.device("cpu")

    def _apply_group_offloading(
        self,
        module: Any,
        config: Optional[MemoryConfig],
        *,
        module_label: str,
    ) -> bool:
        if module is None or config is None:
            return False

        if getattr(module, "_apex_group_offloading_enabled", False):
            return True

        onload_device = self._group_offload_onload_device()

        try:
            kwargs = config.to_group_offload_kwargs(onload_device)
        except Exception as exc:
            self.logger.warning(
                f"Failed to build group offloading options for '{module_label}': {exc}"
            )
            return False

        enable_method = getattr(module, "enable_group_offload", None)
        try:
            if callable(enable_method):
                enable_method(**kwargs)
            else:
                from diffusers.hooks import apply_group_offloading

                apply_group_offloading(module, **kwargs)
        except Exception as exc:
            self.logger.warning(
                f"Failed to enable group offloading for '{module_label}': {exc}"
            )
            return False

        setattr(module, "_apex_group_offloading_enabled", True)
        self.logger.info(
            f"Enabled group offloading for '{module_label}' "
            f"({kwargs.get('offload_type', 'leaf_level')})"
        )
        return True

    def _maybe_apply_memory_management(self, component: Dict[str, Any], module: Any):
        mm_config = self._resolve_memory_config_for_component(component)
        if mm_config is None:
            return
        key = component.get("name") or component.get("type")
        module_label = key or component.get("base") or type(module).__name__
        if not self._apply_group_offloading(module, mm_config, module_label=module_label):
            return
        # Replace references on engine for known types to ensure we reuse the same instance.
        ctype = component.get("type")
        if ctype in {"transformer", "vae", "text_encoder"}:
            setattr(self, ctype, module)

    def load_component_by_type(self, component_type: str):
        for component in self.config.get("components", []):
            if component.get("type") == component_type:
                ignore_load_dtype = component.get("extra_kwargs", {}).get("ignore_load_dtype", False)
                component_module = self.load_component(
                    component,
                    (
                        self.component_load_dtypes.get(component.get("type"))
                        if self.component_load_dtypes and not ignore_load_dtype
                        else None
                    ),
                )
                setattr(self, component.get("type"), component_module)
                break


    def load_component_by_name(self, component_name: str):
        for component in self.config.get("components", []):
            if component.get("name") == component_name:
                ignore_load_dtype = component.get("extra_kwargs", {}).get("ignore_load_dtype", False)
                component_module = self.load_component(
                    component,
                    (
                        self.component_load_dtypes.get(component.get("type"))
                        if self.component_load_dtypes and not ignore_load_dtype
                        else None
                    ),
                )
                setattr(self, component.get("name"), component_module)
                break

    def get_component_by_name(self, component_name: str):
        for component in self.config.get("components", []):
            if component.get("name") == component_name:
                return component
        return None

    def get_component_by_type(self, component_type: str):
        for component in self.config.get("components", []):
            if component.get("type") == component_type:
                return component
        return None
   
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
            traceback.print_exc()
            self.logger.warning(f"Auto-apply LoRAs failed: {e}")

    def download(
        self,
        save_path: str | None = None,
        components_path: str | None = None,
        preprocessors_path: str | None = None,
        postprocessors_path: str | None = None,
    ):
        if save_path is None:
            save_path = DEFAULT_SAVE_PATH
        if components_path is None:
            components_path = get_components_path()
        if preprocessors_path is None:
            preprocessors_path = get_preprocessor_path()
        if postprocessors_path is None:
            postprocessors_path = get_postprocessor_path()
            
        
        os.makedirs(save_path, exist_ok=True)
        for i, component in enumerate(self.config.get("components", []) ):
            if config_path := component.get("config_path"):
                downloaded_config_path = self.fetch_config(
                    config_path, return_path=True, config_save_path=components_path
                )
                if downloaded_config_path:
                    component["config_path"] = downloaded_config_path
            
         
            component_type = component.get("type")
            component_name = component.get("name")

            if component_type == "scheduler":
                scheduler_options = component.get("scheduler_options")
                if not scheduler_options:
                    continue
                selected_scheduler_option = self.selected_components.get(component_name, self.selected_components.get(component_type, None))
                if not selected_scheduler_option:
                    # take the first scheduler option 
                    selected_scheduler_option = scheduler_options[0]
               
                for scheduler_option in scheduler_options:
                    if selected_scheduler_option['name'] == scheduler_option['name']:
                        current_component = component.copy()
                        del current_component['scheduler_options']
                        selected_scheduler_option.update(current_component)
                        component = selected_scheduler_option
                        
                if component.get("config_path"):
                    downloaded_config_path = self.fetch_config(component["config_path"], return_path=True, config_save_path=components_path)
                    if downloaded_config_path:
                        component["config_path"] = downloaded_config_path
            
            else:
                model_path = component.get("model_path")
                if isinstance(model_path, list):
                    selected_model_item = self.selected_components.get(component_name, self.selected_components.get(component_type, None))
                    if not selected_model_item:
                        # take the first model path item
                        selected_model_item = model_path[0]
                    
                    for model_path_item in model_path:
                        if selected_model_item.get('variant') == model_path_item.get('variant'):
                            component['model_path'] = selected_model_item.get('path')
                            if isinstance(model_path_item.get('key_map'), dict):
                                component['key_map'] = model_path_item.get('key_map')
                            if isinstance(model_path_item.get('extra_kwargs'), dict):
                                component['extra_kwargs'] = model_path_item.get('extra_kwargs')
                    
                    path = self.is_downloaded(component['model_path'], components_path)
                    print(path, component['model_path'])
                    if path is None:
                        component['model_path'] = self._download(component['model_path'], components_path)
                    else:
                        component['model_path'] = path
                
                elif isinstance(model_path, str):
                    downloaded_model_path = self._download(model_path, components_path)
                    if downloaded_model_path:
                        component["model_path"] = downloaded_model_path
                
            if extra_model_paths := component.get("extra_model_paths"):
                for m_index, extra_model_path in enumerate(extra_model_paths):
                    if isinstance(extra_model_path, dict):
                        downloaded_extra_model_path = self._download(extra_model_path["path"], components_path)
                    else:
                        downloaded_extra_model_path = self._download(
                            extra_model_path, components_path
                        )
                    if downloaded_extra_model_path:
                        component["extra_model_paths"][m_index] = downloaded_extra_model_path
            

            self.config["components"][i] = component
            
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
                "Both `seed` and `generator` are provided. `seed` will be ignored."
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
        
        noise = randn_tensor(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
            layout=layout or torch.strided,
        )
        
        if return_generator:
            return noise, generator
        else:
            return noise

    def _render_step(self, latents: torch.Tensor, render_on_step_callback: Callable):
        video = self.vae_decode(latents)
        rendered_video = self._tensor_to_frames(video)
        render_on_step_callback(rendered_video[0])

    def _tensor_to_frames(self, video: torch.Tensor, output_type: str = "pil"):
        postprocessed_video = self.video_processor.postprocess_video(
            video, output_type=output_type
        )
        return postprocessed_video

    def _tensor_to_frame(self, image: torch.Tensor, output_type: str = "pil", **kwargs):
        if hasattr(self, "image_processor"):
            postprocessed_frame = self.image_processor.postprocess(
                image, output_type=output_type, **kwargs
            )
        else:
            postprocessed_frame = self.video_processor.postprocess(
                image, output_type=output_type, **kwargs
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

    def denoise(self, *args, **kwargs):
        """
        Dispatch denoising to a type-specific implementation.

        If ``self.denoise_type`` is set, this looks for a method named
        ``\"{denoise_type}_denoise\"`` on ``self`` (where ``denoise_type`` is
        either the string value or, for enums, the ``.value``). If such a
        method exists, it is called with all provided ``*args`` and
        ``**kwargs``.
        """
        denoise_type = getattr(self, "denoise_type", None)
        if denoise_type is None:
            raise ValueError("denoise_type is not set on this engine")

        # Support both plain strings and Enum-like objects with a .value
        if not isinstance(denoise_type, str) and hasattr(denoise_type, "value"):
            denoise_key = str(denoise_type.value)
        else:
            denoise_key = str(denoise_type)

        method_name = f"{denoise_key}_denoise"
        fn = getattr(self, method_name, None)

        # Primary: type-specific denoise method, e.g. "base_denoise"
        if fn is not None and callable(fn):
            return fn(*args, **kwargs)

        # Fallback: locate the first class in the MRO (excluding BaseEngine and
        # object) that defines a `denoise` attribute directly on the class
        # dict. This lets shared mixins override denoising by simply defining
        # a `denoise` method.
        for cls in type(self).mro():
            if cls is BaseEngine or cls is object:
                continue
            if "denoise" in cls.__dict__:
                candidate = cls.__dict__["denoise"]
                if callable(candidate):
                    # Bind the function to this instance and call it
                    return candidate(self, *args, **kwargs)

        raise AttributeError(
            f"No denoise implementation found for type '{denoise_key}' "
            f"(expected method '{method_name}' or a class-defined 'denoise')"
        )
    
    @staticmethod
    def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def _parse_num_frames(self, duration: int | str, fps: int = 16):
        """Accepts a duration in seconds or a string like "16" or "16s" and returns the number of frames.

        Args:
            duration (int | str): duration in seconds or a string like "16" or "16s"

        Returns:
            int: number of frames
        """

        if isinstance(duration, str):
            if duration.endswith("s"):
                duration = int(float(duration[:-1]) * fps) + 1
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

    if TYPE_CHECKING:
        # Hint to type-checkers/IDEs: unknown attributes accessed on engines
        # (e.g. dynamically attached components) are treated as nn.Module by
        # default. This does not affect runtime behaviour.
        def __getattr__(self, name: str) -> nn.Module: ...

    def __str__(self):
        return f"BaseEngine(config={self.config}, device={self.device})"

    def __repr__(self):
        return self.__str__()

    def validate_model_path(self, component: Dict[str, Any]):
        component = self.load_component(component, no_weights=True)
        del component
        empty_cache()
        return True