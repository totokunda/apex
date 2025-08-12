from typing import Dict, Union
import os
import torch
import safetensors


def is_safetensors_file(file_path: str):
    try:
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
            f.keys()
        return True
    except Exception:
        return False


def load_safetensors(
    filename: Union[str, os.PathLike],
    device: Union[str, int] = "cpu",
    dtype: torch.dtype = None,
) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
            if dtype:
                result[k] = result[k].to(dtype)
    return result
