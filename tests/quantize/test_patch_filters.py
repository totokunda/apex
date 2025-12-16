import pytest
import torch
import torch.nn as nn
import gguf

from src.quantize.scaled_layer import FPScaledLayer, FPScaledLinear, patch_fpscaled_model_from_state_dict
from src.quantize.ggml_layer import GGMLLayer, GGMLLinear, patch_model_from_state_dict
from src.quantize.ggml_tensor import GGMLTensor


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4, bias=True)
        self.l2 = nn.Linear(4, 4, bias=True)


def test_patch_fpscaled_only_targets_modules_with_scale_weight_key():
    model = TinyModel()
    state_dict = {"l2.scale_weight": torch.ones(1, dtype=torch.float32)}

    patch_fpscaled_model_from_state_dict(model, state_dict)

    assert not isinstance(model.l1, FPScaledLayer)
    assert isinstance(model.l2, FPScaledLayer)
    assert isinstance(model.l2, FPScaledLinear)


def test_patch_fpscaled_can_target_modules_with_fp8_weight_key_without_scale_weight():
    model = TinyModel()

    try:
        fp8_w = torch.zeros((4, 4), dtype=torch.float8_e4m3fn)
    except Exception:
        pytest.skip("float8 dtype not available/constructible in this torch build")

    state_dict = {"l2.weight": fp8_w}
    patch_fpscaled_model_from_state_dict(model, state_dict)

    assert not isinstance(model.l1, FPScaledLayer)
    assert isinstance(model.l2, FPScaledLayer)


def test_patch_ggml_only_targets_modules_with_quantized_weight():
    model = TinyModel()

    q_w = GGMLTensor(
        torch.zeros(16, dtype=torch.uint8),
        tensor_type=gguf.GGMLQuantizationType.Q4_0,
        tensor_shape=(4, 4),
        dequant_dtype=torch.float16,
    )
    state_dict = {"l1.weight": q_w}

    patch_model_from_state_dict(model, state_dict)

    assert isinstance(model.l1, GGMLLayer)
    assert isinstance(model.l1, GGMLLinear)
    assert not isinstance(model.l2, GGMLLayer)


