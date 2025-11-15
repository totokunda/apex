"""Configuration settings for memory management module."""

import dataclasses
from typing import Optional, Union, Dict, Any
import torch


@dataclasses.dataclass
class MemoryConfig:
    """Configuration for memory management system."""

    # GPU memory thresholds (as fraction of total VRAM)
    gpu_offload_threshold: float = 0.85  # Start offloading at 85% usage
    gpu_emergency_threshold: float = 0.95  # Emergency offload at 95%
    gpu_reload_threshold: float = 0.70  # Reload when usage drops below 70%

    # CPU memory thresholds (as fraction of total RAM)
    cpu_offload_threshold: float = 0.90  # Offload to disk at 90% CPU usage
    cpu_emergency_threshold: float = 0.98  # Emergency disk offload at 98%

    # Monitoring settings
    memory_check_interval: float = 0.1  # Check memory every 100ms
    prediction_window: int = 5  # Look ahead 5 operations for prediction

    # Offloading behavior
    enable_cpu_offload: bool = True
    enable_disk_offload: bool = True
    disk_cache_dir: Optional[str] = None  # Will use temp dir if None
    compress_disk_cache: bool = True  # Compress tensors when saving to disk
    eager_offload_on_wrap: bool = True  # Immediately offload wrapped modules so .to() won't spike VRAM
    offload_1d_parameters: bool = False  # Keep biases/LayerNorm scales resident by default

    # Performance tuning
    offload_batch_size: int = 10  # Max modules to offload in one batch
    prefetch_enabled: bool = True  # Enable predictive loading
    pin_memory: bool = True  # Use pinned memory for faster transfers

    # VRAM cap simulation (treat GPU as if it had at most this VRAM)
    vram_cap_gb: Optional[float] = None  # If set, enforce effective total VRAM

    # Aggressive offload behavior
    aggressive_post_forward_offload: bool = False  # Offload immediately after forward if above threshold

    # Cache management
    empty_cache_after_offload: bool = False  # Call torch.cuda.empty_cache() after offloading

    # Safety settings
    min_free_gpu_memory: float = 0.1  # Always keep 10% GPU memory free
    max_disk_usage_gb: float = 50.0  # Max disk usage for cache

    time_since_forward_threshold: float = 2.0  # Time since last forward pass to offload

    # Group offloading behavior (diffusers-native offloading mechanism)
    group_offload_type: str = "leaf_level"
    group_offload_num_blocks_per_group: Optional[int] = None
    group_offload_use_stream: bool = False
    group_offload_record_stream: bool = False
    group_offload_non_blocking: bool = False
    group_offload_low_cpu_mem_usage: bool = True
    group_offload_offload_device: Union[str, torch.device] = "cpu"
    group_offload_disk_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.gpu_offload_threshold >= self.gpu_emergency_threshold:
            raise ValueError("gpu_offload_threshold must be < gpu_emergency_threshold")

        if self.gpu_reload_threshold >= self.gpu_offload_threshold:
            raise ValueError("gpu_reload_threshold must be < gpu_offload_threshold")

        if self.cpu_offload_threshold >= self.cpu_emergency_threshold:
            raise ValueError("cpu_offload_threshold must be < cpu_emergency_threshold")

    def to_group_offload_kwargs(self, onload_device: torch.device) -> Dict[str, Any]:
        """
        Translate this config into keyword arguments expected by `model.enable_group_offload`.
        """
        offload_device = self.group_offload_offload_device
        try:
            if isinstance(offload_device, torch.device):
                resolved_offload_device = offload_device
            else:
                resolved_offload_device = torch.device(offload_device or "cpu")
        except Exception:
            resolved_offload_device = torch.device("cpu")

        kwargs: Dict[str, Any] = {
            "onload_device": onload_device,
            "offload_device": resolved_offload_device,
            "offload_type": self.group_offload_type,
            "non_blocking": self.group_offload_non_blocking,
            "use_stream": self.group_offload_use_stream,
            "record_stream": self.group_offload_record_stream,
            "low_cpu_mem_usage": self.group_offload_low_cpu_mem_usage,
        }

        if self.group_offload_num_blocks_per_group is not None:
            kwargs["num_blocks_per_group"] = self.group_offload_num_blocks_per_group

        if self.group_offload_disk_path:
            kwargs["offload_to_disk_path"] = self.group_offload_disk_path

        return kwargs

    @classmethod
    def for_low_memory(cls) -> "MemoryConfig":
        """Create config optimized for low memory environments."""
        # Derive a VRAM cap from the actual GPU memory on this machine so that
        # we leave headroom instead of assuming a fixed 16GB limit.
        vram_cap_gb = 16.0
        if torch.cuda.is_available():
            try:
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                total_gb = float(total_bytes) / 1e9
                # Use a conservative fraction of the available VRAM to keep things
                # in a low‑memory regime while still scaling with device size.
                vram_cap_gb = max(1.0, total_gb * 0.7)
            except Exception:
                # Fall back to the previous default if device query fails
                vram_cap_gb = 16.0

        return cls(
            gpu_offload_threshold=0.70,
            gpu_emergency_threshold=0.85,
            gpu_reload_threshold=0.50,
            cpu_offload_threshold=0.80,
            offload_batch_size=5,
            max_disk_usage_gb=20.0,
            vram_cap_gb=vram_cap_gb,
            aggressive_post_forward_offload=True,
            empty_cache_after_offload=True,
            group_offload_type="leaf_level",
            group_offload_num_blocks_per_group=1,
            group_offload_use_stream=True,
            group_offload_record_stream=True,
            group_offload_non_blocking=True,
            group_offload_low_cpu_mem_usage=True,
        )

    @classmethod
    def for_high_performance(cls) -> "MemoryConfig":
        """Create config optimized for high performance."""
        return cls(
            gpu_offload_threshold=0.95,
            gpu_emergency_threshold=0.98,
            gpu_reload_threshold=0.85,
            cpu_offload_threshold=0.95,
            memory_check_interval=0.05,
            prefetch_enabled=True,
            offload_batch_size=20,
            group_offload_type="block_level",
            group_offload_num_blocks_per_group=4,
            group_offload_use_stream=False,
            group_offload_record_stream=False,
            group_offload_non_blocking=False,
            group_offload_low_cpu_mem_usage=False,
        )
