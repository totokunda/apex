import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from src.utils.defaults import set_torch_device, get_torch_device, HOME_DIR,  set_cache_path, get_cache_path as get_cache_path_default, set_components_path, get_components_path as get_components_path_default

router = APIRouter(prefix="/config", tags=["config"])

class HomeDirectoryRequest(BaseModel):
    home_dir: str

class HomeDirectoryResponse(BaseModel):
    home_dir: str

class TorchDeviceRequest(BaseModel):
    device: str

class TorchDeviceResponse(BaseModel):
    device: str

class CachePathRequest(BaseModel):
    cache_path: str

class CachePathResponse(BaseModel):
    cache_path: str

class ComponentsPathRequest(BaseModel):
    components_path: str

class ComponentsPathResponse(BaseModel):
    components_path: str

@router.get("/home-dir", response_model=HomeDirectoryResponse)
def get_home_directory():
    """Get the current apex home directory"""
    return HomeDirectoryResponse(home_dir=str(HOME_DIR))

@router.post("/home-dir", response_model=HomeDirectoryResponse)
def set_home_directory(request: HomeDirectoryRequest):
    """Set the apex home directory. Requires restart to take full effect."""
    try:
        home_path = Path(request.home_dir).expanduser().resolve()
        if not home_path.exists():
            home_path.mkdir(parents=True, exist_ok=True)
        
        os.environ["APEX_HOME_DIR"] = str(home_path)
        return HomeDirectoryResponse(home_dir=str(home_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set home directory: {str(e)}")

@router.get("/torch-device", response_model=TorchDeviceResponse)
def get_device():
    """Get the current torch device"""
    device = get_torch_device()
    return TorchDeviceResponse(device=str(device))

@router.post("/torch-device", response_model=TorchDeviceResponse)
def set_device(request: TorchDeviceRequest):
    """Set the torch device (cpu, cuda, mps, cuda:0, etc.)"""
    try:
        valid_devices = ["cpu", "cuda", "mps"]
        device_str = request.device.lower()
        
        if device_str.startswith("cuda:"):
            device_index = int(device_str.split(":")[1])
            if not torch.cuda.is_available():
                raise HTTPException(status_code=400, detail="CUDA is not available")
            if device_index >= torch.cuda.device_count():
                raise HTTPException(status_code=400, detail=f"CUDA device {device_index} not found")
        elif device_str == "cuda":
            if not torch.cuda.is_available():
                raise HTTPException(status_code=400, detail="CUDA is not available")
        elif device_str == "mps":
            if not torch.backends.mps.is_available():
                raise HTTPException(status_code=400, detail="MPS is not available")
        elif device_str != "cpu":
            raise HTTPException(status_code=400, detail=f"Invalid device: {device_str}. Must be one of {valid_devices} or cuda:N")
        
        set_torch_device(device_str)
        return TorchDeviceResponse(device=device_str)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set device: {str(e)}")

@router.get("/cache-path", response_model=CachePathResponse)
def get_cache_path():
    """Get the current cache path for media-related cache items"""
    return CachePathResponse(cache_path=str(get_cache_path_default()))

@router.post("/cache-path", response_model=CachePathResponse)
def set_cache_path(request: CachePathRequest):
    """Set the cache path for media-related cache items"""
    try:
        cache_path = Path(request.cache_path).expanduser().resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        
        set_cache_path(str(cache_path))
        return CachePathResponse(cache_path=str(cache_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set cache path: {str(e)}")


@router.get("/components-path", response_model=ComponentsPathResponse)
def get_components_path():
    """Get the current components path"""
    return ComponentsPathResponse(components_path=str(get_components_path_default()))

@router.post("/components-path", response_model=ComponentsPathResponse)
def set_components_path(request: ComponentsPathRequest):
    """Set the components path"""
    try:
        components_path = Path(request.components_path).expanduser().resolve()
        components_path.mkdir(parents=True, exist_ok=True)
        set_components_path(str(components_path))
        return ComponentsPathResponse(components_path=str(components_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set components path: {str(e)}")