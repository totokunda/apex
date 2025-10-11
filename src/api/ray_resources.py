import os
import subprocess
import ray
from loguru import logger

def _gpu_mem_info_torch():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        infos = []
        for i in range(torch.cuda.device_count()):
            # Make device current, then ask PyTorch for mem info
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()  # bytes
            infos.append({"index": i, "free": free, "total": total})
        return infos
    except Exception:
        return None

def _gpu_mem_info_nvml():
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        n = nvml.nvmlDeviceGetCount()
        infos = []
        for i in range(n):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            infos.append({"index": i, "free": mem.free, "total": mem.total})
        nvml.nvmlShutdown()
        return infos
    except Exception:
        return None

def _gpu_mem_info_nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().splitlines()
        infos = []
        for i, line in enumerate(out):
            total_mb, used_mb = [int(x.strip()) for x in line.split(",")]
            free_mb = max(total_mb - used_mb, 0)
            # convert MB to bytes for consistency
            infos.append({"index": i, "free": free_mb * 1024**2, "total": total_mb * 1024**2})
        return infos if infos else None
    except Exception:
        return None

def _on_mps():
    try:
        import torch
        # Treat MPS as a single logical accelerator
        return getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    except Exception:
        return False

def get_best_gpu():
    """
    Determine the best GPU to use for a task.

    Priority:
      1) If Apple MPS is available -> device 0
      2) If CUDA GPUs exist -> pick GPU with MOST FREE VRAM
      3) Else -> None (use CPU)

    Returns:
        Tuple of (device_index: int or None, device_type: str)
    """
    # 1) MPS (Apple Silicon)
    if _on_mps():
        logger.info("Using MPS device (Apple Silicon)")
        return 0, "mps"

    # 2) CUDA mem info via PyTorch, then NVML, then nvidia-smi
    infos = _gpu_mem_info_torch() or _gpu_mem_info_nvml() or _gpu_mem_info_nvidia_smi()

    if infos:
        # choose the device with maximum free memory
        best = max(infos, key=lambda d: d["free"])
        logger.info(f"Using CUDA device {best['index']} with {best['free'] / (1024**3):.2f} GB free")
        return best["index"], "cuda"

    # 3) Fallback to CPU
    logger.info("No GPU available, using CPU")
    return None, "cpu"

def get_ray_resources(device_index: int = None, device_type: str = "cuda"):
    """
    Get Ray resources specification for scheduling tasks on specific devices.
    
    Args:
        device_index: GPU index to use (None for CPU)
        device_type: Type of device ("cuda", "mps", or "cpu")
        
    Returns:
        Dictionary of resources for Ray task scheduling
    """
    if device_index is None or device_type == "cpu":
        return {"num_cpus": 1}
    
    if device_type == "mps":
        # MPS doesn't support fractional GPU allocation
        return {"num_cpus": 1, "num_gpus": 0}
    
    if device_type == "cuda":
        # Request specific GPU
        return {"num_cpus": 1, "num_gpus": 0.25, "resources": {f"GPU_{device_index}": 1}}
    
    return {"num_cpus": 1}

