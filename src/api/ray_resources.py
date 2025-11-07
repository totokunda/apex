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
    Get Ray resource requirements dynamically based on current availability.

    Behavior:
      - CPU: allocate up to 1 CPU (or whatever is available if < 1 remains)
      - MPS: treat as CPU-only for Ray scheduling
      - CUDA: allocate up to 1 GPU if available; otherwise take a fractional
        share (>= 0.25 if possible, else whatever remains). If the cluster
        exposes per-GPU custom resources like `GPU_{index}`, honor them to
        pin the request to a specific device when `device_index` is provided.

    Args:
        device_index: GPU index to use (None means CPU when device_type is "cpu").
        device_type: "cuda", "mps", or "cpu".

    Returns:
        Dictionary of Ray resource requirements for task/actor scheduling.
    """
    # Ray may not be initialized in all code paths; guard calls accordingly.
    try:
        ray_initialized = ray.is_initialized()
    except Exception:
        ray_initialized = False

    # Helper to read available/cluster resources if Ray is up
    available = {}
    cluster = {}
    if ray_initialized:
        try:
            available = ray.available_resources() or {}
            cluster = ray.cluster_resources() or {}
        except Exception:
            available, cluster = {}, {}

    # CPU-only path or explicit CPU request
    if device_type == "cpu" or device_index is None:
        if ray_initialized:
            cpu_avail = float(available.get("CPU", 0.0))
            # Request 1 CPU if available, otherwise take remaining (down to 0.25 min)
            num_cpus = 1.0 if cpu_avail >= 1.0 else max(0.25, cpu_avail) if cpu_avail > 0 else 1.0
            return {"num_cpus": num_cpus}
        # Fallback when Ray isn't initialized
        return {"num_cpus": 1}

    # Apple MPS: Ray does not track an MPS resource; treat as CPU-only scheduling
    if device_type == "mps":
        if ray_initialized:
            cpu_avail = float(available.get("CPU", 0.0))
            num_cpus = 1.0 if cpu_avail >= 1.0 else max(0.25, cpu_avail) if cpu_avail > 0 else 1.0
            return {"num_cpus": num_cpus, "num_gpus": 0}
        return {"num_cpus": 1, "num_gpus": 0}

    # CUDA path
    if device_type == "cuda":
        if ray_initialized:
            cpu_avail = float(available.get("CPU", 0.0))
            gpu_avail = float(available.get("GPU", 0.0))

            # CPU request: prefer 1, otherwise take what's left (with a small floor)
            num_cpus = 1.0 if cpu_avail >= 1.0 else max(0.25, cpu_avail) if cpu_avail > 0 else 1.0

            # GPU request: prefer 1 full GPU; otherwise try to take at least 0.25 if possible,
            # else take whatever fractional remains (> 0)
            if gpu_avail >= 1.0:
                num_gpus = 1.0
            elif gpu_avail > 0.0:
                num_gpus = 0.25 if gpu_avail >= 0.25 else gpu_avail
            else:
                # No GPUs available right now; fall back to CPU-only scheduling
                return {"num_cpus": num_cpus}

            resources = None
            # If the cluster exposes per-GPU custom resources, honor them to pin the task
            if device_index is not None:
                custom_key = f"GPU_{device_index}"
                if custom_key in cluster:
                    # Reserve the custom resource fully to ensure placement on the target GPU
                    resources = {custom_key: 1}

            result = {"num_cpus": num_cpus, "num_gpus": num_gpus}
            if resources:
                result["resources"] = resources
            return result

        # Fallback when Ray isn't initialized: assume single-node defaults
        return {"num_cpus": 1, "num_gpus": 1}

    # Unknown device type -> default to 1 CPU
    return {"num_cpus": 1}

