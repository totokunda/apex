import torch
import gguf
from src.quantize.dequant import is_quantized, dequantize_tensor
import torch.nn.functional as F
from src.quantize.ggml_tensor import GGMLTensor


def cast_to(tensor, dtype, device, copy=False, non_blocking=False):
    if tensor is None:
        return None
    if tensor.device == device and tensor.dtype == dtype:
        return tensor
    if copy:
        return tensor.to(device=device, dtype=dtype, copy=True)
    return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)


class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """

    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {
        None,
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    }

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias if hasattr(self, "bias") else None
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(
            f"{prefix}bias"
        )
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(
            self, torch.nn.Linear
        ):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + "weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = self.dequant_dtype or torch.float16
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return
        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        return weight

    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = (
                    getattr(input, "dtype", self.weight.dequant_dtype) or torch.float32
                )
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        if hasattr(self, "bias") and self.bias is not None:
            bias = self.get_weight(self.bias.to(device), bias_dtype)
            bias = cast_to(bias, bias_dtype, device, copy=False)

        weight = self.get_weight(self.weight.to(device), dtype)
        weight = cast_to(weight, dtype, device, copy=False)
        return weight, bias


class GGMLLinear(GGMLLayer, torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self.cast_bias_weight(input)
        return F.linear(input, weight, bias)


class GGMLConv2d(GGMLLayer, torch.nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self.cast_bias_weight(input)
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )


class GGMLConv1d(GGMLLayer, torch.nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self.cast_bias_weight(input)
        return F.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )


class GGMLEmbedding(GGMLLayer, torch.nn.Embedding):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, _ = self.cast_bias_weight(self.weight, device=input.device)

        return F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class GGMLLayerNorm(GGMLLayer, torch.nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self.cast_bias_weight(input)
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)


class GGMLGroupNorm(GGMLLayer, torch.nn.GroupNorm):
    def __init__(self, *args, dequant_dtype: torch.dtype = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dequant_dtype = dequant_dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self.cast_bias_weight(input)
        return F.group_norm(input, self.num_groups, weight, bias, self.eps)


def patch_module(module: torch.nn.Module):
    # Iterate over a static list to avoid mutating while iterating
    for name, child in list(module.named_children()):
        # Replace leaf layers we care about, otherwise recurse
        if isinstance(child, torch.nn.Linear):
            new_mod = GGMLLinear(
                child.in_features, child.out_features, bias=(child.bias is not None)
            )
            # Preserve parameters and training mode
            new_mod.weight = child.weight
            if child.bias is not None:
                new_mod.bias = child.bias
            new_mod.training = child.training
            module.add_module(name, new_mod)
        elif isinstance(child, torch.nn.Conv2d):
            new_mod = GGMLConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                bias=(child.bias is not None),
            )
            new_mod.weight = child.weight
            if child.bias is not None:
                new_mod.bias = child.bias
            new_mod.training = child.training
            module.add_module(name, new_mod)
        elif isinstance(child, torch.nn.Conv1d):
            new_mod = GGMLConv1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                bias=(child.bias is not None),
            )
            new_mod.weight = child.weight
            if child.bias is not None:
                new_mod.bias = child.bias
            new_mod.training = child.training
            module.add_module(name, new_mod)
        elif isinstance(child, torch.nn.Embedding):
            new_mod = GGMLEmbedding(
                child.num_embeddings,
                child.embedding_dim,
                child.padding_idx,
                child.max_norm,
                child.norm_type,
                child.scale_grad_by_freq,
                child.sparse,
            )
            new_mod.weight = child.weight
            new_mod.training = child.training
            module.add_module(name, new_mod)
        elif isinstance(child, torch.nn.LayerNorm):
            new_mod = GGMLLayerNorm(
                child.normalized_shape, child.eps, child.elementwise_affine
            )
            if child.weight is not None:
                new_mod.weight = child.weight
            if child.bias is not None:
                new_mod.bias = child.bias
            new_mod.training = child.training
            module.add_module(name, new_mod)
        elif isinstance(child, torch.nn.GroupNorm):
            new_mod = GGMLGroupNorm(
                child.num_groups,
                child.num_channels,
                child.eps,
                child.affine,
            )
            if child.weight is not None:
                new_mod.weight = child.weight
            if child.bias is not None:
                new_mod.bias = child.bias
            new_mod.training = child.training
            module.add_module(name, new_mod)
        else:
            patch_module(child)
