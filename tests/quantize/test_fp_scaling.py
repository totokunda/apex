import torch
import os
from safetensors.torch import load_file
from glob import glob
from src.quantize.scaled_layer import quantize_to_fp8, fp8_activation_dequant, quantize_to_fp4, dequantize_from_fp4
path = "/home/tosin_coverquick_co/apex/Wan2.2-Lightning/Wan2.2-T2V-A14B/high_noise_model"
files = glob(os.path.join(path, "*.safetensors"))

def mse(a, b):
    return torch.mean((a - b) ** 2)

def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

for file in files:
    state_dict = load_file(file)
    for key, value in state_dict.items():
        numel = value.numel()
        if numel > 256*256:
            quantized_tensor_fp8, log_scales_fp8 = quantize_to_fp8(value)
            dequantized_tensor = fp8_activation_dequant(quantized_tensor_fp8, log_scales_fp8, dtype=value.dtype)
            quantized_tensor_fp4, log_scales_fp4 = quantize_to_fp4(value)
            dequantized_tensor_fp4 = dequantize_from_fp4(quantized_tensor_fp4, log_scales_fp4, out_dtype=value.dtype)

            # Raw in-memory sizes of the actual tensors we have
            orig_bytes = tensor_nbytes(value)
            fp8_qdq_bytes = tensor_nbytes(quantized_tensor_fp8)
            fp8_scale_bytes = tensor_nbytes(log_scales_fp8)
            fp4_code_bytes = tensor_nbytes(quantized_tensor_fp4)
            fp4_scale_bytes = tensor_nbytes(log_scales_fp4)

            # Theoretical packed sizes assuming:
            #   - FP8 codes: 1 byte per element
            #   - FP4 codes: 4 bits per element (2 values per byte)
            fp8_codes_packed_bytes = value.numel()  # 1 byte/code
            fp4_codes_packed_bytes = (quantized_tensor_fp4.numel() + 1) // 2

            print(f"FP4 {key}: {torch.cosine_similarity(value, dequantized_tensor_fp4, dim=(0)).mean()} {mse(value, dequantized_tensor_fp4)}")
            print(f"FP8 {key}: {torch.cosine_similarity(value, dequantized_tensor, dim=(0)).mean()} {mse(value, dequantized_tensor)}")
            print(
                f"SIZE {key}: "
                f"orig={orig_bytes/1e6:.3f}MB "
                f"fp8_qdq={fp8_qdq_bytes/1e6:.3f}MB "
                f"fp8_scales={fp8_scale_bytes/1e6:.3f}MB "
                f"fp4_codes_raw={fp4_code_bytes/1e6:.3f}MB "
                f"fp4_scales={fp4_scale_bytes/1e6:.3f}MB "
                f"fp8_codes_packed={fp8_codes_packed_bytes/1e6:.3f}MB "
                f"fp4_codes_packed={fp4_codes_packed_bytes/1e6:.3f}MB"
            )