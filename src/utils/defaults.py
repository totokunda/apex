import torch
import os
from pathlib import Path

HOME_DIR = Path.home()


DEFAULT_CONFIG_SAVE_PATH = str(HOME_DIR / "apex-diffusion" / "configs")
DEFAULT_SAVE_PATH = str(HOME_DIR / "apex-diffusion")
DEFAULT_PREPROCESSOR_SAVE_PATH = str(HOME_DIR / "apex-diffusion" / "preprocessors")

# check what system we are on for posix or windows


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
