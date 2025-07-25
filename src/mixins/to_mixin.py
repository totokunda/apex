import re
import torch
from typing import List
from src.utils.cache_utils import empty_cache
from diffusers import ModelMixin


class ToMixin:
    """
    Mixin providing utilities to move Diffusers ModelMixin components
    to specified devices and data types, respecting model-specific settings.
    """

    def _parse_dtype(self, dtype: str | torch.dtype) -> torch.dtype:
        """
        Convert a string or torch.dtype into a torch.dtype.
        """
        if isinstance(dtype, torch.dtype):
            return dtype
        mapping = {
            **{alias: torch.float16 for alias in ("float16", "fp16", "f16")},
            **{alias: torch.bfloat16 for alias in ("bfloat16", "bf16")},
            **{alias: torch.float32 for alias in ("float32", "fp32", "f32")},
            **{alias: torch.float64 for alias in ("float64", "fp64", "f64")},
            **{alias: torch.int8 for alias in ("int8", "i8")},
            **{alias: torch.uint8 for alias in ("uint8", "u8")},
        }
        key = dtype.lower() if isinstance(dtype, str) else dtype
        if key in mapping:
            return mapping[key]
        raise ValueError(f"Unsupported dtype: {dtype}")

    def to_dtype(
        self,
        module: ModelMixin,
        dtype: str | torch.dtype,
        requires_grad: bool = False,
        # ---- new knobs -----------------------------------------------
        layerwise: bool = False,
        storage_dtype: torch.dtype | None = None,
        compute_dtype: torch.dtype | None = None,
        cast_buffers: bool = True,
    ) -> ModelMixin:
        """
        Cast *either* uniformly (like `from_pretrained(torch_dtype=…)`)
        *or* via Diffusers' on-the-fly layer-wise casting.

        Parameters
        ----------
        dtype:
            Target dtype for a uniform cast **or** the *default* storage
            dtype when `layerwise=True`.
        layerwise:
            • False  → blanket cast; ignore `_skip_layerwise_casting_patterns`.
            • True   → call `enable_layerwise_casting`, honouring
                        `_skip_layerwise_casting_patterns`.
        storage_dtype / compute_dtype:
            Only used when `layerwise=True`.
        """
        target_dtype = self._parse_dtype(dtype)

        keep_fp32_patterns = tuple(getattr(module, "_keep_in_fp32_modules", []) or [])
        no_split_patterns = tuple(getattr(module, "_no_split_modules", []) or [])

        # ------------------------------------------------------------------
        # 1. Fast path – use the official API when layer-wise casting is wanted
        # ------------------------------------------------------------------
        if layerwise:
            skip_patterns = tuple(
                getattr(module, "_skip_layerwise_casting_patterns", []) or []
            )
            module.enable_layerwise_casting(
                storage_dtype=storage_dtype or target_dtype,
                compute_dtype=compute_dtype or torch.float32,
                skip_modules_pattern=skip_patterns,
            )  # Diffusers handles the heavy lifting :contentReference[oaicite:3]{index=3}
            return module

        # ------------------------------------------------------------------
        # 2. Uniform cast (like `from_pretrained(torch_dtype=…)`)
        #    – skip-patterns are intentionally *ignored* here
        # ------------------------------------------------------------------
        def _matches(patterns, name):
            return any(re.search(p, name) for p in patterns)

        # a) cast whole sub-modules that must stay together
        frozen_prefixes = []
        for name, submod in module.named_modules():
            if _matches(no_split_patterns, name):
                dtype_mod = (
                    torch.float32
                    if _matches(keep_fp32_patterns, name)
                    else target_dtype
                )
                submod.to(dtype_mod)
                frozen_prefixes.append(name)

        # b) cast individual parameters
        for name, param in module.named_parameters(recurse=True):
            if any(name == p or name.startswith(p + ".") for p in frozen_prefixes):
                param.requires_grad = requires_grad
                continue
            wanted_dtype = (
                torch.float32 if _matches(keep_fp32_patterns, name) else target_dtype
            )
            param.data = param.data.to(wanted_dtype)
            param.requires_grad = requires_grad
            if param.grad is not None:
                param.grad.data = param.grad.data.to(wanted_dtype)

        # c) cast buffers
        if cast_buffers:
            for name, buf in module.named_buffers(recurse=True):
                if any(name == p or name.startswith(p + ".") for p in frozen_prefixes):
                    continue
                wanted_dtype = (
                    torch.float32
                    if _matches(keep_fp32_patterns, name)
                    else target_dtype
                )
                buf.data = buf.data.to(wanted_dtype)

        # d) propagate ignore-lists for state-dict loading
        if hasattr(module, "_keys_to_ignore_on_load_unexpected"):
            module._keys_to_ignore_on_load_unexpected = getattr(
                self, "_keys_to_ignore_on_load_unexpected", []
            )

        return module

    def to_device(
        self,
        *components: torch.nn.Module,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Move specified modules (or defaults) to a device, then clear CUDA cache.

        If no components are provided, tries attributes:
        vae, text_encoder, transformer, scheduler, and self.preprocessors.values().
        """
        # Determine target device
        if device is None:
            device = getattr(self, "device", None) or torch.device("cpu")
        if isinstance(device, str):
            device = torch.device(device)

        # Default components if none specified
        if not components:
            defaults = []
            for attr in ("vae", "text_encoder", "transformer", "scheduler"):
                comp = getattr(self, attr, None)
                if comp is not None:
                    defaults.append(comp)
            extras = getattr(self, "preprocessors", {}).values()
            components = (*defaults, *extras)

        # Move each to device
        for comp in components:
            if hasattr(comp, "to"):
                comp.to(device)

        # Free up any unused CUDA memory
        try:
            empty_cache()
        except Exception:
            pass
