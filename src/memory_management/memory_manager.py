"""Main memory manager that coordinates all memory management functionality."""

import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Set
import torch
import torch.nn as nn

from .config import MemoryConfig
from .memory_monitor import MemoryMonitor
from .module_wrapper import OffloadableModule, wrap_module, wrap_model_layers
from .offload_strategies import SmartOffloadStrategy


class MemoryManager:
    """Central memory management system for PyTorch models."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize memory manager with configuration."""
        self.config = config or MemoryConfig()
        
        # Core components
        self.memory_monitor = MemoryMonitor(self.config)
        self.offload_strategy = SmartOffloadStrategy(self.config)
        
        # Module tracking
        self._managed_modules: Dict[str, weakref.ref] = {}
        self._module_lock = threading.RLock()
        
        # State tracking
        self._is_active = False
        self._auto_manage = True
        self._management_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._stats = {
            'total_offloads': 0,
            'total_reloads': 0,
            'bytes_offloaded': 0,
            'bytes_reloaded': 0,
            'start_time': time.time()
        }
        
    def start(self):
        """Start the memory management system."""
        if self._is_active:
            return
            
        self._is_active = True
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        # Start management thread
        if self._auto_manage:
            self._management_thread = threading.Thread(
                target=self._management_loop,
                daemon=True
            )
            self._management_thread.start()
            
        print("Memory management system started")
        
    def stop(self):
        """Stop the memory management system."""
        if not self._is_active:
            return
            
        self._is_active = False
        
        # Stop monitoring
        self.memory_monitor.stop_monitoring()
        
        # Stop management thread
        if self._management_thread:
            self._management_thread.join(timeout=2.0)
            
        print("Memory management system stopped")
        
    def _management_loop(self):
        """Background loop for automatic memory management."""
        while self._is_active:
            try:
                # Check memory pressure and take action
                self._check_and_manage_memory()
                
                # Clean up dead module references
                self._cleanup_dead_references()
                
                time.sleep(self.config.memory_check_interval)
                
            except Exception as e:
                print(f"Memory management error: {e}")
                time.sleep(1.0)
                
    def _check_and_manage_memory(self):
        """Check memory pressure and manage modules accordingly."""
        with self._module_lock:
            # Emergency offloading
            if self.memory_monitor.should_emergency_offload_gpu():
                self._emergency_offload()
                return
                
            # Regular offloading
            if self.memory_monitor.should_offload_from_gpu():
                self._selective_offload()
                
            # Reload opportunities
            if self.memory_monitor.can_reload_to_gpu():
                self._selective_reload()
                
    def _emergency_offload(self):
        """Emergency offloading when memory is critically low."""
        print("Emergency GPU memory offload triggered")
        
        for module_ref in self._managed_modules.values():
            module = module_ref()
            if module is not None:
                try:
                    module.offload_all()
                    self._stats['total_offloads'] += 1
                except Exception as e:
                    print(f"Emergency offload failed: {e}")
                    
    def _selective_offload(self):
        """Selective offloading based on module usage patterns."""
        current_time = time.time()
        modules_to_offload = []
        
        for module_id, module_ref in self._managed_modules.items():
            module:OffloadableModule = module_ref()
            if module is None:
                continue
                
            # Get module info
            info = module.get_memory_info()
            
            # Skip if already mostly offloaded
            if info['offload_ratio'] > 0.8:
                continue
                
            # Offload if inactive for too long
            time_since_forward = current_time - info['last_forward_time']
            if time_since_forward > self.config.time_since_forward_threshold:  # 2 second threshold
                modules_to_offload.append(module)
                
        # Offload selected modules
        for module in modules_to_offload[:self.config.offload_batch_size]:
            try:
                module.offload_all()
                self._stats['total_offloads'] += 1
            except Exception as e:
                print(f"Selective offload failed: {e}")
                
    def _selective_reload(self):
        """Selective reloading when memory pressure is low."""
        # Only reload if we have breathing room
        gpu_usage = self.memory_monitor.get_gpu_usage()
        if gpu_usage > self.config.gpu_reload_threshold + 0.1:  # Add buffer
            return
            
        # Find modules that might benefit from reloading
        for module_ref in self._managed_modules.values():
            module:OffloadableModule = module_ref()
            if module is None:
                continue
                
            info = module.get_memory_info()
            
            # Reload recently active modules
            if info['offload_ratio'] > 0 and info['forward_count'] > 0:
                time_since_forward = time.time() - info['last_forward_time']
                if time_since_forward < 1.0:  # Recently used
                    try:
                        module.load_all()
                        self._stats['total_reloads'] += 1
                        break  # Only reload one at a time
                    except Exception as e:
                        print(f"Selective reload failed: {e}")
                        
    def _cleanup_dead_references(self):
        """Remove references to deleted modules."""
        with self._module_lock:
            dead_refs = []
            for module_id, module_ref in self._managed_modules.items():
                if module_ref() is None:
                    dead_refs.append(module_id)
                    
            for module_id in dead_refs:
                del self._managed_modules[module_id]
                
    def wrap_module(self, module: nn.Module, module_id: Optional[str] = None) -> OffloadableModule:
        """Wrap a PyTorch module with memory management."""
        if module_id is None:
            module_id = f"module_{len(self._managed_modules)}"
            
        wrapped = wrap_module(module, self.config, self.memory_monitor, module_id)
        
        # Track the wrapped module
        with self._module_lock:
            self._managed_modules[module_id] = weakref.ref(wrapped)
            
        return wrapped
        
    def wrap_model(self, model: nn.Module, layer_types: Optional[List[type]] = None) -> nn.Module:
        """Wrap specific layers in a model with memory management."""
        if layer_types is None:
            layer_types = [nn.Linear]
            
        # Wrap the model
        wrapped_model = wrap_model_layers(
            model, 
            self.config, 
            self.memory_monitor, 
            layer_types
        )
        
        # Track wrapped modules in the model
        self._register_wrapped_modules_in_model(wrapped_model)
        
        return wrapped_model
        
    def _register_wrapped_modules_in_model(self, model: nn.Module):
        """Register all OffloadableModule instances in a model."""
        with self._module_lock:
            for name, module in model.named_modules():
                if isinstance(module, OffloadableModule):
                    self._managed_modules[module.module_id] = weakref.ref(module)
                    
    def offload_all_modules(self):
        """Manually offload all managed modules."""
        with self._module_lock:
            for module_ref in self._managed_modules.values():
                module = module_ref()
                if module is not None:
                    try:
                        module.offload_all()
                        self._stats['total_offloads'] += 1
                    except Exception as e:
                        print(f"Failed to offload module: {e}")
                        
    def load_all_modules(self):
        """Manually load all managed modules."""
        with self._module_lock:
            for module_ref in self._managed_modules.values():
                module = module_ref()
                if module is not None:
                    try:
                        module.load_all()
                        self._stats['total_reloads'] += 1
                    except Exception as e:
                        print(f"Failed to load module: {e}")
                        
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        memory_summary = self.memory_monitor.get_memory_summary()
        
        # Add module information
        module_info = self._get_module_summary()
        memory_summary['modules'] = module_info
        
        # Add system stats
        memory_summary['system'] = {
            'active': self._is_active,
            'auto_manage': self._auto_manage,
            'managed_modules': len(self._managed_modules),
            'total_offloads': self._stats['total_offloads'],
            'total_reloads': self._stats['total_reloads'],
            'uptime_seconds': time.time() - self._stats['start_time']
        }
        
        return memory_summary
        
    def _get_module_summary(self) -> Dict[str, Any]:
        """Get summary of all managed modules."""
        summary = {
            'total_modules': 0,
            'active_modules': 0,
            'total_parameters': 0,
            'loaded_parameters': 0,
            'offloaded_parameters': 0,
            'total_memory_mb': 0,
            'loaded_memory_mb': 0,
            'offloaded_memory_mb': 0,
            'modules': {}
        }
        
        with self._module_lock:
            for module_id, module_ref in self._managed_modules.items():
                module = module_ref()
                if module is None:
                    continue
                    
                info = module.get_memory_info()
                summary['modules'][module_id] = info
                
                # Aggregate stats
                summary['total_modules'] += 1
                if info['is_active']:
                    summary['active_modules'] += 1
                    
                summary['total_parameters'] += info['total_parameters']
                summary['loaded_parameters'] += info['loaded_parameters']
                summary['offloaded_parameters'] += info['offloaded_parameters']
                summary['loaded_memory_mb'] += info['loaded_memory_mb']
                summary['offloaded_memory_mb'] += info['offloaded_memory_mb']
                
        summary['total_memory_mb'] = summary['loaded_memory_mb'] + summary['offloaded_memory_mb']
        
        return summary
        
    def set_auto_management(self, enabled: bool):
        """Enable or disable automatic memory management."""
        self._auto_manage = enabled
        
        if enabled and self._is_active and not self._management_thread:
            self._management_thread = threading.Thread(
                target=self._management_loop,
                daemon=True
            )
            self._management_thread.start()
            
    def cleanup(self):
        """Clean up all resources."""
        self.stop()
        
        # Clean up all modules
        with self._module_lock:
            for module_ref in self._managed_modules.values():
                module = module_ref()
                if module is not None:
                    try:
                        module.cleanup()
                    except:
                        pass
                        
            self._managed_modules.clear()
            
        # Clean up disk cache if available
        if hasattr(self.offload_strategy, 'disk_strategy') and self.offload_strategy.disk_strategy:
            self.offload_strategy.disk_strategy.cleanup_all()
            
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass


# Convenience functions for quick setup
def create_memory_manager(config: Optional[MemoryConfig] = None) -> MemoryManager:
    """Create and start a memory manager."""
    manager = MemoryManager(config)
    manager.start()
    return manager


def auto_manage_model(model: nn.Module, config: Optional[MemoryConfig] = None, 
                     layer_types: Optional[List[type]] = None) -> MemoryManager:
    """Automatically wrap a model with memory management and return the manager."""
    manager = create_memory_manager(config)
    manager.wrap_model(model, layer_types)
    return manager 