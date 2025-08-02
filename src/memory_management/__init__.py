"""
Comprehensive Memory Management Module for PyTorch

This module provides intelligent memory management for PyTorch models,
supporting offloading between GPU, CPU, and disk based on memory usage.
"""

from .memory_manager import MemoryManager, auto_manage_model, create_memory_manager
from .module_wrapper import OffloadableModule, wrap_module, wrap_model_layers
from .memory_monitor import MemoryMonitor
from .offload_strategies import OffloadStrategy, CPUOffloadStrategy, DiskOffloadStrategy
from .config import MemoryConfig

__all__ = [
    "MemoryManager",
    "OffloadableModule",
    "MemoryMonitor",
    "OffloadStrategy",
    "CPUOffloadStrategy",
    "DiskOffloadStrategy",
    "MemoryConfig",
    "auto_manage_model",
    "create_memory_manager",
    "wrap_module",
    "wrap_model_layers",
]

__version__ = "1.0.0"
