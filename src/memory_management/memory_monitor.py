"""Memory monitoring utilities for tracking GPU, CPU, and disk usage."""

import time
import threading
import psutil
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
from .config import MemoryConfig


class MemoryStats:
    """Container for memory statistics."""

    def __init__(self, used: float, total: float, timestamp: float):
        self.used = used
        self.total = total
        self.free = total - used
        self.usage_ratio = used / total if total > 0 else 0.0
        self.timestamp = timestamp


class MemoryMonitor:
    """Real-time memory monitoring for GPU, CPU, and disk."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Store recent memory usage history for prediction
        self.gpu_history: deque = deque(maxlen=config.prediction_window * 2)
        self.cpu_history: deque = deque(maxlen=config.prediction_window * 2)
        self.disk_history: deque = deque(maxlen=config.prediction_window * 2)

        # Current stats
        self._current_gpu_stats: Optional[MemoryStats] = None
        self._current_cpu_stats: Optional[MemoryStats] = None
        self._current_disk_stats: Optional[MemoryStats] = None

        # Thread safety
        self._stats_lock = threading.Lock()

        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()

    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                timestamp = time.time()

                # Collect current stats
                gpu_stats = self._get_gpu_stats(timestamp)
                cpu_stats = self._get_cpu_stats(timestamp)
                disk_stats = self._get_disk_stats(timestamp)

                # Update current stats thread-safely
                with self._stats_lock:
                    self._current_gpu_stats = gpu_stats
                    self._current_cpu_stats = cpu_stats
                    self._current_disk_stats = disk_stats

                    # Add to history
                    if gpu_stats:
                        self.gpu_history.append(gpu_stats)
                    self.cpu_history.append(cpu_stats)
                    self.disk_history.append(disk_stats)

                time.sleep(self.config.memory_check_interval)

            except Exception as e:
                print(f"Memory monitoring error: {e}")
                time.sleep(1.0)  # Back off on errors

    def _get_gpu_stats(self, timestamp: float) -> Optional[MemoryStats]:
        """Get current GPU memory statistics."""
        if not self.cuda_available:
            return None

        try:
            # Get stats for current device
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)

            # Use reserved memory as "used" since it's what's actually allocated
            return MemoryStats(used=reserved, total=total, timestamp=timestamp)

        except Exception:
            return None

    def _get_cpu_stats(self, timestamp: float) -> MemoryStats:
        """Get current CPU memory statistics."""
        memory = psutil.virtual_memory()
        return MemoryStats(used=memory.used, total=memory.total, timestamp=timestamp)

    def _get_disk_stats(self, timestamp: float) -> MemoryStats:
        """Get current disk usage statistics."""
        # Use root partition for disk stats
        disk = psutil.disk_usage("/")
        return MemoryStats(used=disk.used, total=disk.total, timestamp=timestamp)

    def get_current_stats(
        self,
    ) -> Tuple[Optional[MemoryStats], MemoryStats, MemoryStats]:
        """Get current memory statistics for GPU, CPU, and disk."""
        with self._stats_lock:
            return (
                self._current_gpu_stats,
                self._current_cpu_stats,
                self._current_disk_stats,
            )

    def get_gpu_usage(self) -> float:
        """Get current GPU memory usage ratio (0.0 to 1.0)."""
        gpu_stats, _, _ = self.get_current_stats()
        return gpu_stats.usage_ratio if gpu_stats else 0.0

    def get_cpu_usage(self) -> float:
        """Get current CPU memory usage ratio (0.0 to 1.0)."""
        _, cpu_stats, _ = self.get_current_stats()
        return cpu_stats.usage_ratio if cpu_stats else 0.0

    def get_disk_usage(self) -> float:
        """Get current disk usage ratio (0.0 to 1.0)."""
        _, _, disk_stats = self.get_current_stats()
        return disk_stats.usage_ratio if disk_stats else 0.0

    def should_offload_from_gpu(self) -> bool:
        """Check if we should start offloading from GPU."""
        usage = self.get_gpu_usage()
        return usage >= self.config.gpu_offload_threshold

    def should_emergency_offload_gpu(self) -> bool:
        """Check if we need emergency GPU offloading."""
        usage = self.get_gpu_usage()
        return usage >= self.config.gpu_emergency_threshold

    def should_offload_from_cpu(self) -> bool:
        """Check if we should offload from CPU to disk."""
        usage = self.get_cpu_usage()
        return usage >= self.config.cpu_offload_threshold

    def can_reload_to_gpu(self) -> bool:
        """Check if we can reload data back to GPU."""
        usage = self.get_gpu_usage()
        return usage <= self.config.gpu_reload_threshold

    def predict_memory_pressure(self) -> Dict[str, float]:
        """Predict future memory pressure based on recent trends."""
        predictions = {"gpu": 0.0, "cpu": 0.0, "disk": 0.0}

        with self._stats_lock:
            # Predict GPU pressure
            if len(self.gpu_history) >= 2 and self.gpu_history:
                recent_gpu = list(self.gpu_history)[-self.config.prediction_window :]
                if len(recent_gpu) >= 2:
                    # Simple linear trend prediction
                    start_usage = recent_gpu[0].usage_ratio
                    end_usage = recent_gpu[-1].usage_ratio
                    trend = (end_usage - start_usage) / len(recent_gpu)
                    predictions["gpu"] = max(0.0, min(1.0, end_usage + trend * 3))

            # Predict CPU pressure
            if len(self.cpu_history) >= 2:
                recent_cpu = list(self.cpu_history)[-self.config.prediction_window :]
                if len(recent_cpu) >= 2:
                    start_usage = recent_cpu[0].usage_ratio
                    end_usage = recent_cpu[-1].usage_ratio
                    trend = (end_usage - start_usage) / len(recent_cpu)
                    predictions["cpu"] = max(0.0, min(1.0, end_usage + trend * 3))

        return predictions

    def get_memory_summary(self) -> Dict:
        """Get comprehensive memory usage summary."""
        gpu_stats, cpu_stats, disk_stats = self.get_current_stats()
        predictions = self.predict_memory_pressure()

        summary = {
            "timestamp": time.time(),
            "gpu": {
                "available": gpu_stats is not None,
                "usage_ratio": gpu_stats.usage_ratio if gpu_stats else 0.0,
                "used_gb": (gpu_stats.used / 1e9) if gpu_stats else 0.0,
                "total_gb": (gpu_stats.total / 1e9) if gpu_stats else 0.0,
                "predicted_usage": predictions["gpu"],
            },
            "cpu": {
                "usage_ratio": cpu_stats.usage_ratio,
                "used_gb": cpu_stats.used / 1e9,
                "total_gb": cpu_stats.total / 1e9,
                "predicted_usage": predictions["cpu"],
            },
            "disk": {
                "usage_ratio": disk_stats.usage_ratio,
                "used_gb": disk_stats.used / 1e9,
                "total_gb": disk_stats.total / 1e9,
            },
            "recommendations": {
                "should_offload_gpu": self.should_offload_from_gpu(),
                "should_emergency_offload": self.should_emergency_offload_gpu(),
                "should_offload_cpu": self.should_offload_from_cpu(),
                "can_reload_gpu": self.can_reload_to_gpu(),
            },
        }

        return summary
