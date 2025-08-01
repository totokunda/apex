"""Offloading strategies for moving data between GPU, CPU, and disk."""

import os
import pickle
import tempfile
import threading
import time
import weakref
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from .config import MemoryConfig


class OffloadStrategy(ABC):
    """Abstract base class for offloading strategies."""
    
    @abstractmethod
    def offload(self, tensor: torch.Tensor, identifier: str) -> Dict[str, Any]:
        """Offload tensor and return metadata for retrieval."""
        pass
    
    @abstractmethod
    def reload(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """Reload tensor from metadata."""
        pass
    
    @abstractmethod
    def cleanup(self, metadata: Dict[str, Any]) -> bool:
        """Clean up stored data and return success status."""
        pass


class CPUOffloadStrategy(OffloadStrategy):
    """Strategy for offloading tensors from GPU to CPU memory."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.pin_memory = config.pin_memory and torch.cuda.is_available()
        
    def offload(self, tensor: torch.Tensor, identifier: str) -> Dict[str, Any]:
        """Move tensor from GPU to CPU."""
        if not tensor.is_cuda:
            # Already on CPU
            return {
                'strategy': 'cpu',
                'tensor': tensor,
                'device': tensor.device,
                'dtype': tensor.dtype,
                'shape': tensor.shape,
                'pinned': False
            }
        
        # Move to CPU
        cpu_tensor = tensor.cpu()
        
        # Pin memory for faster transfers if enabled
        if self.pin_memory:
            try:
                cpu_tensor = cpu_tensor.pin_memory()
                pinned = True
            except RuntimeError:
                pinned = False
        else:
            pinned = False
            
        return {
            'strategy': 'cpu',
            'tensor': cpu_tensor,
            'device': tensor.device,  # Original device
            'dtype': tensor.dtype,
            'shape': tensor.shape,
            'pinned': pinned
        }
    
    def reload(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """Move tensor back to original device."""
        tensor = metadata['tensor']
        original_device = metadata['device']
        
        if tensor.device == original_device:
            return tensor
            
        # Move back to original device
        if self.pin_memory and metadata.get('pinned', False):
            return tensor.to(original_device, non_blocking=True)
        else:
            return tensor.to(original_device)
    
    def cleanup(self, metadata: Dict[str, Any]) -> bool:
        """Clean up CPU tensor (just remove reference)."""
        if 'tensor' in metadata:
            del metadata['tensor']
        return True


class DiskOffloadStrategy(OffloadStrategy):
    """Strategy for offloading tensors to disk storage."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._setup_cache_dir()
        self._cache_size = 0  # Track cache size in bytes
        self._cache_files: Dict[str, Path] = {}
        self._cache_lock = threading.Lock()
        
    def _setup_cache_dir(self):
        """Setup disk cache directory."""
        if self.config.disk_cache_dir:
            self.cache_dir = Path(self.config.disk_cache_dir)
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "pytorch_memory_cache"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, identifier: str) -> Path:
        """Get cache file path for identifier."""
        filename = f"{identifier}_{int(time.time() * 1000000)}.pt"
        return self.cache_dir / filename
        
    def _check_disk_space(self, estimated_size: int) -> bool:
        """Check if we have enough disk space."""
        max_bytes = self.config.max_disk_usage_gb * 1e9
        return (self._cache_size + estimated_size) <= max_bytes
        
    def _estimate_tensor_size(self, tensor: torch.Tensor) -> int:
        """Estimate serialized tensor size in bytes."""
        # Rough estimate: element size * num elements + overhead
        element_size = tensor.element_size()
        num_elements = tensor.numel()
        overhead = 1024  # Pickle/metadata overhead
        return element_size * num_elements + overhead
        
    def offload(self, tensor: torch.Tensor, identifier: str) -> Dict[str, Any]:
        """Save tensor to disk."""
        # Estimate size and check space
        estimated_size = self._estimate_tensor_size(tensor)
        if not self._check_disk_space(estimated_size):
            raise RuntimeError(f"Insufficient disk space for offloading {identifier}")
        
        # Move to CPU first if on GPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
            
        # Get cache file path
        cache_path = self._get_cache_path(identifier)
        
        try:
            # Save tensor to disk
            if self.config.compress_disk_cache:
                # Use pickle with compression
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'tensor': tensor,
                        'device': tensor.device,
                        'dtype': tensor.dtype,
                        'shape': tensor.shape
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # Use PyTorch's save (faster but larger)
                torch.save(tensor, cache_path)
            
            # Track cache file and size
            with self._cache_lock:
                actual_size = cache_path.stat().st_size
                self._cache_files[identifier] = cache_path
                self._cache_size += actual_size
                
            return {
                'strategy': 'disk',
                'cache_path': cache_path,
                'compressed': self.config.compress_disk_cache,
                'device': tensor.device,
                'dtype': tensor.dtype,
                'shape': tensor.shape,
                'size_bytes': actual_size
            }
            
        except Exception as e:
            # Clean up on failure
            if cache_path.exists():
                cache_path.unlink()
            raise RuntimeError(f"Failed to offload {identifier} to disk: {e}")
    
    def reload(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """Load tensor from disk."""
        cache_path = metadata['cache_path']
        
        if not cache_path.exists():
            raise RuntimeError(f"Cache file not found: {cache_path}")
            
        try:
            if metadata.get('compressed', False):
                # Load from pickle
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    tensor = data['tensor']
            else:
                # Load from PyTorch save
                tensor = torch.load(cache_path, map_location='cpu')
                
            return tensor
            
        except Exception as e:
            raise RuntimeError(f"Failed to reload tensor from {cache_path}: {e}")
    
    def cleanup(self, metadata: Dict[str, Any]) -> bool:
        """Remove cached file from disk."""
        cache_path = metadata.get('cache_path')
        if not cache_path or not cache_path.exists():
            return True
            
        try:
            size_bytes = metadata.get('size_bytes', 0)
            cache_path.unlink()
            
            # Update cache tracking
            with self._cache_lock:
                self._cache_size -= size_bytes
                # Remove from tracking (identifier might not match)
                for identifier, path in list(self._cache_files.items()):
                    if path == cache_path:
                        del self._cache_files[identifier]
                        break
                        
            return True
            
        except Exception:
            return False
    
    def cleanup_all(self) -> int:
        """Clean up all cached files. Returns number of files cleaned."""
        cleaned = 0
        with self._cache_lock:
            for cache_path in list(self._cache_files.values()):
                try:
                    if cache_path.exists():
                        cache_path.unlink()
                        cleaned += 1
                except Exception:
                    pass
            
            self._cache_files.clear()
            self._cache_size = 0
            
        return cleaned
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about disk cache usage."""
        with self._cache_lock:
            return {
                'cache_dir': str(self.cache_dir),
                'num_files': len(self._cache_files),
                'total_size_bytes': self._cache_size,
                'total_size_gb': self._cache_size / 1e9,
                'max_size_gb': self.config.max_disk_usage_gb,
                'usage_ratio': self._cache_size / (self.config.max_disk_usage_gb * 1e9)
            }


class SmartOffloadStrategy:
    """Intelligent strategy that chooses between CPU and disk offloading."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.cpu_strategy = CPUOffloadStrategy(config)
        self.disk_strategy = DiskOffloadStrategy(config) if config.enable_disk_offload else None
        
    def choose_strategy(self, tensor: torch.Tensor, memory_monitor) -> OffloadStrategy:
        """Choose the best offloading strategy based on current conditions."""
        # Always try CPU first if available
        if self.config.enable_cpu_offload:
            cpu_usage = memory_monitor.get_cpu_usage()
            if cpu_usage < self.config.cpu_offload_threshold:
                return self.cpu_strategy
        
        # Fall back to disk if CPU is full and disk is available
        if self.disk_strategy and self.config.enable_disk_offload:
            return self.disk_strategy
            
        # Default to CPU even if usage is high (better than OOM)
        return self.cpu_strategy
    
    def offload(self, tensor: torch.Tensor, identifier: str, memory_monitor) -> Dict[str, Any]:
        """Offload using the best available strategy."""
        strategy = self.choose_strategy(tensor, memory_monitor)
        return strategy.offload(tensor, identifier)
    
    def reload(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """Reload tensor using the strategy specified in metadata."""
        strategy_name = metadata['strategy']
        
        if strategy_name == 'cpu':
            return self.cpu_strategy.reload(metadata)
        elif strategy_name == 'disk' and self.disk_strategy:
            return self.disk_strategy.reload(metadata)
        else:
            raise RuntimeError(f"Unknown or unavailable strategy: {strategy_name}")
    
    def cleanup(self, metadata: Dict[str, Any]) -> bool:
        """Clean up using the appropriate strategy."""
        strategy_name = metadata['strategy']
        
        if strategy_name == 'cpu':
            return self.cpu_strategy.cleanup(metadata)
        elif strategy_name == 'disk' and self.disk_strategy:
            return self.disk_strategy.cleanup(metadata)
        else:
            return False 