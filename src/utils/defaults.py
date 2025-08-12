import torch
import os
from pathlib import Path

HOME_DIR = Path.home()


DEFAULT_CONFIG_SAVE_PATH = os.getenv(
    "CONFIG_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "configs")
)
DEFAULT_SAVE_PATH = os.getenv("SAVE_PATH", str(HOME_DIR / "apex-diffusion"))

DEFAULT_COMPONENTS_PATH = os.getenv("COMPONENTS_PATH", str(HOME_DIR / "apex-diffusion" / "components"))

DEFAULT_PREPROCESSOR_SAVE_PATH = os.getenv(
    "PREPROCESSOR_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "preprocessors")
)

DEFAULT_POSTPROCESSOR_SAVE_PATH = os.getenv(
    "POSTPROCESSOR_SAVE_PATH", str(HOME_DIR / "apex-diffusion" / "postprocessors")
)

DEFAULT_CACHE_PATH = os.getenv("CACHE_PATH", str(HOME_DIR / "apex-diffusion" / "cache"))

# make sure all paths exist 
os.makedirs(DEFAULT_CONFIG_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_COMPONENTS_PATH, exist_ok=True)
os.makedirs(DEFAULT_PREPROCESSOR_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_POSTPROCESSOR_SAVE_PATH, exist_ok=True)
os.makedirs(DEFAULT_CACHE_PATH, exist_ok=True)

os.environ["HF_HOME"] = os.getenv(
    "HF_HOME", str(HOME_DIR / "apex-diffusion" / "huggingface")
)


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
