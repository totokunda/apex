"""Module wrapper for automatic memory management of PyTorch modules."""

import threading
import time
import weakref
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
from .config import MemoryConfig
from .offload_strategies import SmartOffloadStrategy


class ParameterMetadata:
    """Metadata for managing offloaded parameters."""

    def __init__(self, name: str, param: torch.Tensor, module_id: str, is_parameter: bool):
        self.name = name
        self.original_device = param.device
        self.shape = param.shape
        self.dtype = param.dtype
        self.requires_grad = param.requires_grad
        self.module_id = module_id
        # Track whether this name was originally registered as a Parameter (vs buffer)
        self.is_parameter = is_parameter
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
        # Debug flag to avoid spamming placeholder diagnostics
        self._debug_logged = False

        # Setup parameter tracking
        self._setup_parameter_tracking()

        # Create lightweight stub Parameters on the wrapper itself for any top-level
        # parameters (e.g. `weight`, `bias`) so that external code can still access
        # attributes like `.weight.shape` and `.weight.dtype` on the OffloadableModule,
        # without keeping a full copy of the real tensor in memory.
        self._create_param_stubs()

        # Register hooks for automatic management
        self._register_hooks()

    def _setup_parameter_tracking(self):
        """Setup tracking for all parameters and buffers."""
        with self._lock:
            # Track parameters
            for name, param in self._original_module.named_parameters():
                if param is not None:
                    self._param_metadata[name] = ParameterMetadata(
                        name=name, param=param, module_id=self.module_id, is_parameter=True
                    )

            # Track buffers (non-gradient tensors)
            for name, buffer in self._original_module.named_buffers():
                if buffer is not None:
                    self._param_metadata[name] = ParameterMetadata(
                        name=name, param=buffer, module_id=self.module_id, is_parameter=False
                    )

    def _create_param_stubs(self):
        """
        Create fake but shape/dtype-accurate Parameters on this wrapper for any
        top-level parameters such as `weight` and `bias`.

        The real tensors live on `self._original_module`, and may be offloaded or
        replaced with placeholders. These stubs are never used for computation;
        they exist solely so that code which expects `module.weight.shape`,
        `module.bias.dtype`, etc. to be present on the OffloadableModule will
        continue to work.
        """
        for name, metadata in self._param_metadata.items():
            # Only expose top-level parameters (no dotted paths) that were originally
            # registered as Parameters, and do not override existing attributes.
            if not metadata.is_parameter:
                continue
            if "." in name:
                continue
            if hasattr(self, name):
                continue

            # Create a minimal CPU stub tensor; we avoid meta tensors so that
            # calling `.to(device)` on the wrapped model continues to work.
            # The stub never participates in computation, so we keep it tiny but
            # let callers rely on its dtype.
            stub_tensor = torch.empty(
                (1,),
                dtype=metadata.dtype,
                device="cpu",
            )

            stub_param = nn.Parameter(stub_tensor, requires_grad=False)
            setattr(self, name, stub_param)

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
            target_device = self._infer_target_device(module, input)
            self._ensure_parameters_loaded(target_device)

    def _post_forward_hook(self, module, input, output):
        """Hook called after forward pass - consider offloading if needed."""
        with self._lock:
            self._is_active = False

            # Check if we should offload based on memory pressure
            if self._should_offload():
                self._offload_parameters()

            # Aggressive mode: offload immediately if over threshold or VRAM cap exceeded
            if self.config.aggressive_post_forward_offload:
                if self.memory_monitor.should_offload_from_gpu() or self.memory_monitor.should_emergency_offload_gpu():
                    self._offload_parameters()

        return output

    def _ensure_parameters_loaded(self, target_device: Optional[torch.device] = None):
        """Ensure all parameters are loaded on the correct device."""
        names_to_load: Set[str] = set(self._offloaded_params)

        for name, metadata in self._param_metadata.items():
            if metadata.is_offloaded:
                names_to_load.add(name)
                continue

            if self._is_placeholder_tensor(name, metadata):
                metadata.is_offloaded = True
                names_to_load.add(name)

        if not names_to_load:
            return

        # Guard inside _load_parameter expects the name to be tracked
        for name in names_to_load:
            self._offloaded_params.add(name)

        for name in list(names_to_load):
            self._load_parameter(name, target_device)

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
        # Optionally free CUDA cache after offloading to release reserved memory
        if self.config.empty_cache_after_offload and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _offload_parameter(self, param_name: str):
        """Offload a specific parameter."""
        if param_name in self._offloaded_params:
            return

        metadata = self._param_metadata[param_name]

        # Optionally keep 1D parameters (biases, LayerNorm scales) resident to avoid shape-mismatch issues
        if (
            not self.config.offload_1d_parameters
            and len(metadata.shape) <= 1
        ):
            return

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
            import traceback
            traceback.print_exc()
            exit()
            print(f"Failed to offload {param_name}: {e}")

    def _load_parameter(self, param_name: str, target_device: Optional[torch.device] = None):
        """Load a specific parameter back to device."""
        metadata = self._param_metadata.get(param_name)
        if metadata is None or metadata.offload_metadata is None:
            return

        try:
            # Reload tensor using the stored strategy (original weights)
            tensor = self.offload_strategy.reload(metadata.offload_metadata)

            # Determine final desired device
            desired_device = target_device or self._get_module_current_device()
            if desired_device is None:
                desired_device = metadata.original_device

            # Move to desired device if needed
            if tensor.device != desired_device:
                tensor = tensor.to(desired_device)

            # Restore parameter/buffer
            self._restore_parameter_from_tensor(param_name, tensor)

            # Update metadata
            metadata.is_offloaded = False
            metadata.last_access_time = time.time()
            metadata.access_count += 1
            # Update original_device to reflect current placement
            metadata.original_device = desired_device
            self._offloaded_params.discard(param_name)

            # Clean up offload storage
            self.offload_strategy.cleanup(metadata.offload_metadata)
            metadata.offload_metadata = None

        except Exception as e:
            
            print(f"Failed to load {param_name}: {e}")

    def _infer_target_device(self, module: nn.Module, input) -> Optional[torch.device]:
        """Infer the device to place parameters for this forward call."""
        # 1) Prefer device of first Tensor in input
        def _find_tensor_device(obj) -> Optional[torch.device]:
            if torch.is_tensor(obj):
                return obj.device
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    dev = _find_tensor_device(item)
                    if dev is not None:
                        return dev
            if isinstance(obj, dict):
                for item in obj.values():
                    dev = _find_tensor_device(item)
                    if dev is not None:
                        return dev
            return None

        input_device = _find_tensor_device(input)
        if input_device is not None:
            return input_device

        # 2) Fallback to device of any existing parameter/buffer in the module
        for p in module.parameters(recurse=False):
            if p is not None:
                return p.device
        for b in module.buffers(recurse=False):
            if b is not None:
                return b.device

        # 3) Unknown
        return None

    def _get_module_current_device(self) -> Optional[torch.device]:
        """Get current device of the original module if determinable."""
        for p in self._original_module.parameters(recurse=False):
            if p is not None:
                return p.device
        for b in self._original_module.buffers(recurse=False):
            if b is not None:
                return b.device
        return None

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

    def _is_placeholder_tensor(
        self, name: str, metadata: ParameterMetadata
    ) -> bool:
        """Detect if the current tensor is a placeholder instead of real data."""
        tensor = self._get_parameter_by_name(name)
        if tensor is None or metadata is None:
            return False

        # Treat 1-element tensors backed by offload metadata or mismatched shapes as placeholders.
        if tensor.numel() != 1:
            return False

        # If we have offload metadata, this name was offloaded at some point – a 1‑element tensor is a placeholder.
        if metadata.offload_metadata is not None:
            return True

        # If the current shape does not match the recorded original shape, it's a placeholder.
        if tensor.shape != metadata.shape:
            return True

        # As a final guard, if the recorded original parameter had >1 elements but we now see a single element,
        # treat it as a placeholder even if shapes somehow match (defensive against bad tracking).
        original_numel = 1
        for dim in metadata.shape:
            original_numel *= dim
        if original_numel > 1:
            return True

        return False

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

        # Convert to Parameter if this name was originally a Parameter (even if requires_grad=False)
        if metadata.is_parameter:
            placeholder = nn.Parameter(placeholder, requires_grad=metadata.requires_grad)

        # Replace in module
        names = param_name.split(".")
        obj = self._original_module

        for name in names[:-1]:
            obj = getattr(obj, name)

        setattr(obj, names[-1], placeholder)

    def _restore_parameter_from_tensor(self, param_name: str, tensor: torch.Tensor):
        """Restore parameter from loaded tensor."""
        metadata = self._param_metadata[param_name]

        # Ensure correct properties and wrap appropriately based on original registration type
        if metadata.is_parameter:
            # Always restore as nn.Parameter, preserving original requires_grad
            if isinstance(tensor, nn.Parameter):
                restored = tensor
                restored.requires_grad = metadata.requires_grad
            else:
                restored = nn.Parameter(tensor, requires_grad=metadata.requires_grad)
            tensor_to_set = restored
        else:
            # Restore as a plain tensor (buffer). Ensure it does not require grad.
            if isinstance(tensor, nn.Parameter):
                tensor = tensor.detach()
            if tensor.requires_grad:
                tensor = tensor.detach()
            tensor_to_set = tensor

        # Replace in module
        names = param_name.split(".")
        obj = self._original_module

        for name in names[:-1]:
            obj = getattr(obj, name)

        setattr(obj, names[-1], tensor_to_set)

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module."""
        # Extra safety: ensure parameters are on the appropriate device inferred from inputs
        with self._lock:
            target_device = self._infer_target_device(self._original_module, args)
            if target_device is None and kwargs:
                target_device = self._infer_target_device(self._original_module, kwargs)
            self._ensure_parameters_loaded(target_device)
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
    wrapped = OffloadableModule(module, config, memory_monitor, module_id)

    if config.eager_offload_on_wrap:
        try:
            wrapped.offload_all()
        except Exception as exc:
            warnings.warn(
                f"Failed to eagerly offload module '{module_id or type(module).__name__}': {exc}",
                RuntimeWarning,
            )

    return wrapped


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

            # If the child is already an OffloadableModule, do not attempt to wrap its
            # internal `_original_module` again. Treat it as an atomic managed unit.
            if isinstance(child, OffloadableModule):
                continue

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
