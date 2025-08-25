"""Configuration settings for memory management module."""

import dataclasses
from typing import Optional
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

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.gpu_offload_threshold >= self.gpu_emergency_threshold:
            raise ValueError("gpu_offload_threshold must be < gpu_emergency_threshold")

        if self.gpu_reload_threshold >= self.gpu_offload_threshold:
            raise ValueError("gpu_reload_threshold must be < gpu_offload_threshold")

        if self.cpu_offload_threshold >= self.cpu_emergency_threshold:
            raise ValueError("cpu_offload_threshold must be < cpu_emergency_threshold")

    @classmethod
    def for_low_memory(cls) -> "MemoryConfig":
        """Create config optimized for low memory environments."""
        return cls(
            gpu_offload_threshold=0.70,
            gpu_emergency_threshold=0.85,
            gpu_reload_threshold=0.50,
            cpu_offload_threshold=0.80,
            offload_batch_size=5,
            max_disk_usage_gb=20.0,
            vram_cap_gb=16.0,
            aggressive_post_forward_offload=True,
            empty_cache_after_offload=True,
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
        )
