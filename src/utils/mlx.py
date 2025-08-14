import mlx.core as mx
import torch


def convert_dtype_to_torch(dtype: str | torch.dtype | mx.Dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, mx.Dtype):
        dtype_as_str = str(dtype).replace("mlx.core.", "")
        return getattr(torch, dtype_as_str)
    else:
        return torch.dtype(dtype)


def convert_dtype_to_mlx(dtype: str | torch.dtype | mx.Dtype) -> mx.Dtype:
    if isinstance(dtype, mx.Dtype):
        return dtype
    elif isinstance(dtype, torch.dtype):
        dtype_as_str = str(dtype).replace("torch.", "")
        return getattr(mx, dtype_as_str)
    else:
        return mx.Dtype(dtype)


def to_mlx(t: torch.Tensor) -> mx.array:
    torch_dtype = t.dtype
    mx_dtype = convert_dtype_to_mlx(torch_dtype)
    return mx.array(t.detach().to("cpu").numpy(), dtype=mx_dtype)


def to_torch(a: mx.array) -> torch.Tensor:
    import numpy as np

    mx_dtype = a.dtype
    torch_dtype = convert_dtype_to_torch(mx_dtype)
    return torch.from_numpy(np.array(a)).to(torch_dtype)
