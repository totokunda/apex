from __future__ import annotations

"""model_compiler.py
A unified, extensible interface for compiling PyTorch models into a variety of
runtimes/IRs/back‑ends (Inductor, TorchScript, ONNX, TensorRT, etc.).
Each concrete compiler only needs to implement `compile()`.

Usage
-----
>>> from model_compiler import TorchInductorCompiler, ONNXCompiler
>>> compiled = TorchInductorCompiler({"mode": "max-autotune"})(my_model)
>>> onnx_path = ONNXCompiler({"opset_version": 18}).compile(my_model)

New back‑ends can be added by inheriting from `BaseCompiler` and registering
via `COMPILER_REGISTRY`.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Type
import importlib
import logging

import torch

__all__ = [
    "BaseCompiler",
    "TorchInductorCompiler",
    "TorchScriptCompiler",
    "ONNXCompiler",
    "TensorRTCompiler",
    "get_compiler",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Base class & registry
# ----------------------------------------------------------------------------

COMPILER_REGISTRY: MutableMapping[str, Type["BaseCompiler"]] = {}


def register_compiler(name: str):
    """Decorator that registers a compiler class under *name*."""

    def _decorator(cls: Type["BaseCompiler"]):
        COMPILER_REGISTRY[name.lower()] = cls
        return cls

    return _decorator


class BaseCompiler(ABC):
    """Abstract base class for all compilers.

    Parameters
    ----------
    params : dict, optional
        Hyper‑parameters / kwargs forwarded to the concrete compiler.
    """

    def __init__(self, params: Mapping[str, Any] | None = None):
        self.params: Dict[str, Any] = dict(params or {})

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @abstractmethod
    def compile(
        self, model: torch.nn.Module, *args, **kwargs
    ) -> Any:  # noqa: D401,E501
        """Compile *model* and return the backend‑specific artifact."""

    # Make instances callable for convenience
    def __call__(self, model: torch.nn.Module, *args, **kwargs) -> Any:
        return self.compile(model, *args, **kwargs)

    # ------------------------------------------------------------------
    # Utility helpers (available to subclasses)
    # ------------------------------------------------------------------

    def _ensure_dir(self, key: str, default: str | Path) -> Path:
        """Return an output directory path, creating it if necessary."""
        path = Path(self.params.get(key, default)).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


# ----------------------------------------------------------------------------
# Concrete compilers
# ----------------------------------------------------------------------------


@register_compiler("torch_compile")
class TorchInductorCompiler(BaseCompiler):
    """Wraps `torch.compile` (a.k.a. TorchDynamo + AOTInductor)."""

    def compile(self, model: torch.nn.Module, example_inputs: tuple | None = None):
        mode = self.params.pop("mode", "default")
        fullgraph = self.params.pop("fullgraph", False)
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph, **self.params)
        logger.info("Compiled model with torch.compile (mode=%s)", mode)
        # Run once if example inputs are provided to warm‑up & cache kernels
        if example_inputs is not None:
            with torch.inference_mode():
                compiled(*example_inputs)
        return compiled


@register_compiler("torchscript")
class TorchScriptCompiler(BaseCompiler):
    """Produces a TorchScript `ScriptModule` via `torch.jit.script` or trace."""

    def compile(self, model: torch.nn.Module, example_inputs: tuple | None = None):
        scripted: torch.jit.ScriptModule
        if example_inputs is None:
            scripted = torch.jit.script(model, **self.params)
        else:
            scripted = torch.jit.trace(model, example_inputs, **self.params)
        logger.info(
            "Compiled model to TorchScript (%s)",
            "script" if example_inputs is None else "trace",
        )
        return scripted


@register_compiler("onnx")
class ONNXCompiler(BaseCompiler):
    """Exports the model to an ONNX `.onnx` file path."""

    def compile(
        self,
        model: torch.nn.Module,
        example_inputs: tuple,
        output_path: str | Path | None = None,
    ):
        if example_inputs is None:
            raise ValueError("ONNX export requires example inputs")
        output_dir = self._ensure_dir("output_dir", Path.cwd() / "onnx_exports")
        output_path = Path(
            output_path or output_dir / f"{model.__class__.__name__}.onnx"
        )
        torch.onnx.export(model, example_inputs, output_path, **self.params)
        logger.info("Exported ONNX model → %s", output_path)
        return output_path


@register_compiler("tensorrt")
class TensorRTCompiler(BaseCompiler):
    """Builds a TensorRT engine *via* torch‑tensorrt; falls back to `trtexec` if unavailable."""

    def compile(
        self,
        model: torch.nn.Module,
        example_inputs: tuple,
        output_path: str | Path | None = None,
    ):
        try:
            tensorrt = importlib.import_module("torch_tensorrt")
        except ModuleNotFoundError as e:
            raise RuntimeError("torch_tensorrt is not installed") from e

        dtype = self.params.pop("dtype", torch.float16)
        enabled_precisions = {dtype}
        trt_mod = tensorrt.compile(
            model,
            inputs=[tensorrt.Input(example_inputs[0].shape, dtype=dtype)],
            enabled_precisions=enabled_precisions,
            **self.params,
        )
        logger.info("Compiled TensorRT engine (precision=%s)", dtype)
        return trt_mod


# ----------------------------------------------------------------------------
# Helper: factory getter
# ----------------------------------------------------------------------------


def get_compiler(name: str, params: Mapping[str, Any] | None = None) -> BaseCompiler:
    """Factory that returns an instance of the *name* compiler."""
    try:
        cls = COMPILER_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown compiler '{name}'. Available: {list(COMPILER_REGISTRY)}"
        ) from exc
    return cls(params)
