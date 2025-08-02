"""Module wrapper for automatic memory management of PyTorch modules."""

import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from .config import MemoryConfig
from .offload_strategies import SmartOffloadStrategy


class ParameterMetadata:
    """Metadata for managing offloaded parameters."""

    def __init__(self, name: str, param: torch.Tensor, module_id: str):
        self.name = name
        self.original_device = param.device
        self.shape = param.shape
        self.dtype = param.dtype
        self.requires_grad = param.requires_grad
        self.module_id = module_id
        self.offload_metadata: Optional[Dict[str, Any]] = None
        self.is_offloaded = False
        self.last_access_time = time.time()
        self.access_count = 0


class OffloadableModule(nn.Module):
    """Wrapper that makes any PyTorch module memory-managed with automatic offloading."""

    def __init__(
        self,
        module: nn.Module,
        config: MemoryConfig,
        memory_monitor,
        module_id: Optional[str] = None,
    ):
        super().__init__()

        self.config = config
        self.memory_monitor = memory_monitor
        self.module_id = module_id or f"module_{id(self)}"
        self.offload_strategy = SmartOffloadStrategy(config)

        # Store original module
        self._original_module = module

        # Parameter management
        self._param_metadata: Dict[str, ParameterMetadata] = {}
        self._offloaded_params: Set[str] = set()
        self._lock = threading.RLock()

        # State tracking
        self._is_active = False
        self._last_forward_time = 0
        self._forward_count = 0

        # Setup parameter tracking
        self._setup_parameter_tracking()

        # Register hooks for automatic management
        self._register_hooks()

    def _setup_parameter_tracking(self):
        """Setup tracking for all parameters and buffers."""
        with self._lock:
            # Track parameters
            for name, param in self._original_module.named_parameters():
                if param is not None:
                    self._param_metadata[name] = ParameterMetadata(
                        name=name, param=param, module_id=self.module_id
                    )

            # Track buffers (non-gradient tensors)
            for name, buffer in self._original_module.named_buffers():
                if buffer is not None:
                    self._param_metadata[name] = ParameterMetadata(
                        name=name, param=buffer, module_id=self.module_id
                    )

    def _register_hooks(self):
        """Register forward and backward hooks for automatic management."""
        # Pre-forward hook to load parameters
        self._original_module.register_forward_pre_hook(self._pre_forward_hook)

        # Post-forward hook for potential offloading
        self._original_module.register_forward_hook(self._post_forward_hook)

    def _pre_forward_hook(self, module, input):
        """Hook called before forward pass - ensure parameters are loaded."""
        with self._lock:
            self._is_active = True
            self._last_forward_time = time.time()
            self._forward_count += 1

            # Reload any offloaded parameters
            self._ensure_parameters_loaded()

    def _post_forward_hook(self, module, input, output):
        """Hook called after forward pass - consider offloading if needed."""
        with self._lock:
            self._is_active = False

            # Check if we should offload based on memory pressure
            if self._should_offload():
                self._offload_parameters()

        return output

    def _ensure_parameters_loaded(self):
        """Ensure all parameters are loaded on the correct device."""
        for name in list(self._offloaded_params):
            self._load_parameter(name)

    def _should_offload(self) -> bool:
        """Determine if this module should offload its parameters."""
        # Check memory pressure
        if self.memory_monitor.should_emergency_offload_gpu():
            return True

        if self.memory_monitor.should_offload_from_gpu():
            # Consider offloading if module hasn't been used recently
            time_since_forward = time.time() - self._last_forward_time
            if time_since_forward > 1.0:  # 1 second threshold
                return True

        return False

    def _offload_parameters(self):
        """Offload parameters to CPU/disk."""
        for name, metadata in self._param_metadata.items():
            if not metadata.is_offloaded:
                try:
                    self._offload_parameter(name)
                except Exception as e:
                    print(f"Failed to offload parameter {name}: {e}")

    def _offload_parameter(self, param_name: str):
        """Offload a specific parameter."""
        if param_name in self._offloaded_params:
            return

        metadata = self._param_metadata[param_name]

        # Get current parameter/buffer
        param = self._get_parameter_by_name(param_name)
        if param is None:
            return

        # Offload using strategy
        try:
            offload_metadata = self.offload_strategy.offload(
                param, f"{self.module_id}_{param_name}", self.memory_monitor
            )

            metadata.offload_metadata = offload_metadata
            metadata.is_offloaded = True
            self._offloaded_params.add(param_name)

            # Replace parameter with a placeholder (empty tensor)
            self._replace_parameter_with_placeholder(param_name)

        except Exception as e:
            print(f"Failed to offload {param_name}: {e}")

    def _load_parameter(self, param_name: str):
        """Load a specific parameter back to device."""
        if param_name not in self._offloaded_params:
            return

        metadata = self._param_metadata[param_name]

        if metadata.offload_metadata is None:
            return

        try:
            # Reload tensor
            tensor = self.offload_strategy.reload(metadata.offload_metadata)

            # Always move to original device to ensure correct placement
            tensor = tensor.to(metadata.original_device)

            # Restore parameter
            self._restore_parameter_from_tensor(param_name, tensor)

            # Update metadata
            metadata.is_offloaded = False
            metadata.last_access_time = time.time()
            metadata.access_count += 1
            self._offloaded_params.remove(param_name)

            # Clean up offload storage
            self.offload_strategy.cleanup(metadata.offload_metadata)
            metadata.offload_metadata = None

        except Exception as e:
            print(f"Failed to load {param_name}: {e}")

    def _get_parameter_by_name(self, name: str) -> Optional[torch.Tensor]:
        """Get parameter or buffer by name."""
        # Try parameters first
        for param_name, param in self._original_module.named_parameters():
            if param_name == name:
                return param

        # Try buffers
        for buffer_name, buffer in self._original_module.named_buffers():
            if buffer_name == name:
                return buffer

        return None

    def _replace_parameter_with_placeholder(self, param_name: str):
        """Replace parameter with empty placeholder to free memory."""
        metadata = self._param_metadata[param_name]

        # Create minimal placeholder tensor
        placeholder = torch.empty(
            (1,),
            dtype=metadata.dtype,
            device="cpu",
            requires_grad=metadata.requires_grad,
        )

        # Convert to Parameter if it was originally a parameter
        if metadata.requires_grad:
            placeholder = nn.Parameter(placeholder)

        # Replace in module
        names = param_name.split(".")
        obj = self._original_module

        for name in names[:-1]:
            obj = getattr(obj, name)

        setattr(obj, names[-1], placeholder)

    def _restore_parameter_from_tensor(self, param_name: str, tensor: torch.Tensor):
        """Restore parameter from loaded tensor."""
        metadata = self._param_metadata[param_name]

        # Ensure correct properties
        if metadata.requires_grad and not tensor.requires_grad:
            tensor = tensor.requires_grad_(True)
        elif not metadata.requires_grad and tensor.requires_grad:
            tensor = tensor.detach()

        # Convert to Parameter if it was originally a parameter
        if metadata.requires_grad:
            tensor = nn.Parameter(tensor)

        # Replace in module
        names = param_name.split(".")
        obj = self._original_module

        for name in names[:-1]:
            obj = getattr(obj, name)

        setattr(obj, names[-1], tensor)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        return self._original_module(*args, **kwargs)

    def offload_all(self):
        """Manually offload all parameters."""
        with self._lock:
            self._offload_parameters()

    def load_all(self):
        """Manually load all parameters."""
        with self._lock:
            self._ensure_parameters_loaded()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information for this module."""
        with self._lock:
            total_params = len(self._param_metadata)
            offloaded_params = len(self._offloaded_params)
            loaded_params = total_params - offloaded_params

            # Calculate approximate memory usage
            loaded_memory = 0
            offloaded_memory = 0

            for name, metadata in self._param_metadata.items():
                param_size = 1
                for dim in metadata.shape:
                    param_size *= dim

                # Estimate bytes (assuming float32 = 4 bytes)
                element_size = 4  # Could be more precise based on dtype
                param_bytes = param_size * element_size

                if metadata.is_offloaded:
                    offloaded_memory += param_bytes
                else:
                    loaded_memory += param_bytes

            return {
                "module_id": self.module_id,
                "total_parameters": total_params,
                "loaded_parameters": loaded_params,
                "offloaded_parameters": offloaded_params,
                "offload_ratio": (
                    offloaded_params / total_params if total_params > 0 else 0
                ),
                "loaded_memory_mb": loaded_memory / 1e6,
                "offloaded_memory_mb": offloaded_memory / 1e6,
                "is_active": self._is_active,
                "forward_count": self._forward_count,
                "last_forward_time": self._last_forward_time,
            }

    def cleanup(self):
        """Clean up all offloaded data."""
        with self._lock:
            for name in list(self._offloaded_params):
                metadata = self._param_metadata[name]
                if metadata.offload_metadata:
                    self.offload_strategy.cleanup(metadata.offload_metadata)

            self._offloaded_params.clear()

    def __del__(self):
        """Cleanup when module is deleted."""
        try:
            self.cleanup()
        except:
            pass


def wrap_module(
    module: nn.Module,
    config: MemoryConfig,
    memory_monitor,
    module_id: Optional[str] = None,
) -> OffloadableModule:
    """Convenience function to wrap a module with memory management."""
    return OffloadableModule(module, config, memory_monitor, module_id)


def wrap_model_layers(
    model: nn.Module,
    config: MemoryConfig,
    memory_monitor,
    layer_types: Optional[List[type]] = None,
) -> nn.Module:
    """Wrap specific layer types in a model with memory management.

    Args:
        model: The model to wrap
        config: Memory configuration
        memory_monitor: Memory monitor instance
        layer_types: List of layer types to wrap (default: [nn.Linear])
    """
    if layer_types is None:
        layer_types = [nn.Linear]

    # Track wrapped modules to avoid recursion
    wrapped_count = 0

    def wrap_children(module, prefix=""):
        nonlocal wrapped_count

        for name, child in list(module.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name

            # Check if this child should be wrapped
            if type(child) in layer_types:
                wrapped_module = wrap_module(
                    child, config, memory_monitor, module_id=child_prefix
                )
                setattr(module, name, wrapped_module)
                wrapped_count += 1
            else:
                # Recursively check children
                wrap_children(child, child_prefix)

    wrap_children(model)
    print(f"Wrapped {wrapped_count} layers with memory management")

    return model
