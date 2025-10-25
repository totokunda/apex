import torch
import os
from pathlib import Path

HOME_DIR = Path(os.getenv("APEX_HOME_DIR", Path.home()))

DEFAULT_CONFIG_SAVE_PATH = os.getenv(
    "APEX_CONFIG_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "configs")
)
DEFAULT_SAVE_PATH = os.getenv("APEX_SAVE_PATH", str(HOME_DIR / "apex-diffusion"))

DEFAULT_COMPONENTS_PATH = os.getenv(
    "APEX_COMPONENTS_PATH", str(HOME_DIR / "apex-diffusion" / "components")
)

DEFAULT_PREPROCESSOR_SAVE_PATH = os.getenv(
    "APEX_PREPROCESSOR_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "preprocessors")
)

DEFAULT_POSTPROCESSOR_SAVE_PATH = os.getenv(
    "APEX_POSTPROCESSOR_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "postprocessors")
)

DEFAULT_CACHE_PATH = os.getenv(
    "APEX_CACHE_PATH", str(HOME_DIR / "apex-diffusion" / "cache")
)

# New default path to store LoRA adapters and related artifacts
DEFAULT_LORA_SAVE_PATH = os.getenv(
    "APEX_LORA_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "loras")
)

# make sure all paths exist
os.makedirs(DEFAULT_CONFIG_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_COMPONENTS_PATH, exist_ok=True)
os.makedirs(DEFAULT_PREPROCESSOR_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_POSTPROCESSOR_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_CACHE_PATH, exist_ok=True)
os.makedirs(DEFAULT_LORA_SAVE_PATH, exist_ok=True)

os.environ["HF_HOME"] = os.getenv(
    "APEX_HF_HOME", str(HOME_DIR / "apex-diffusion" / "huggingface")
)

# Check if running in Ray worker (avoid MPS in forked processes)
_IN_RAY_WORKER = os.environ.get('RAY_WORKER_NAME') or 'ray::' in os.environ.get('_', '')

if _IN_RAY_WORKER or os.environ.get('FORCE_CPU', ''):
    # Force CPU in Ray workers to avoid MPS/CUDA fork issues
    DEFAULT_DEVICE = torch.device("cpu")
else:
    DEFAULT_DEVICE = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}

def set_torch_device(device: torch.device | str | None = None) -> None:
    global DEFAULT_DEVICE
    if device is None:
        DEFAULT_DEVICE = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )
        
    else:
        DEFAULT_DEVICE = torch.device(device)
    torch.set_default_device(DEFAULT_DEVICE)


def get_torch_device() -> torch.device:
    return DEFAULT_DEVICE

def get_cache_path() -> str:
    return DEFAULT_CACHE_PATH

def set_cache_path(path: str) -> None:
    global DEFAULT_CACHE_PATH
    DEFAULT_CACHE_PATH = path
    os.makedirs(DEFAULT_CACHE_PATH, exist_ok=True)
    

def get_components_path() -> str:
    return DEFAULT_COMPONENTS_PATH

def set_components_path(path: str) -> None:
    global DEFAULT_COMPONENTS_PATH
    DEFAULT_COMPONENTS_PATH = path
    os.makedirs(DEFAULT_COMPONENTS_PATH, exist_ok=True)

def get_preprocessor_save_path() -> str:
    return DEFAULT_PREPROCESSOR_SAVE_PATH

def set_preprocessor_save_path(path: str) -> None:
    global DEFAULT_PREPROCESSOR_SAVE_PATH
    DEFAULT_PREPROCESSOR_SAVE_PATH = path
    os.makedirs(DEFAULT_PREPROCESSOR_SAVE_PATH, exist_ok=True)

