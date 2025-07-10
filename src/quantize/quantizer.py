from diffusers import (
    BitsAndBytesConfig,
    GGUFQuantizationConfig,
    QuantoConfig,
    TorchAoConfig,
)
import torch
import gguf
from enum import Enum
from diffusers.quantizers.auto import DiffusersAutoQuantizer
from diffusers.quantizers.base import DiffusersQuantizer
from typing import Literal, Dict, Union, Optional

from accelerate.utils import (
    get_balanced_memory,
    infer_auto_device_map,
)

from accelerate import dispatch_model

BNB_4BIT_CONFIG_FP16 = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_has_fp16_weight=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

BNB_4BIT_CONFIG_BF16 = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_has_fp16_weight=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)


BNB_8BIT_CONFIG_FP16 = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_has_fp16_weight=True,
    bnb_8bit_compute_dtype=torch.float16,
)

BNB_8BIT_CONFIG_BF16 = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_has_fp16_weight=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

GGUF_CONFIG_FP16 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.F16)
GGUF_CONFIG_BF16 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.BF16)
GGUF_CONFIG_Q8_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q8_K)
GGUF_CONFIG_Q8_0 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q8_0)
GGUF_CONFIG_Q8_1 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q8_1)
GGUF_CONFIG_Q6_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q6_K)

GGUF_CONFIG_Q5_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q5_K)
GGUF_CONFIG_Q5_1 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q5_1)
GGUF_CONFIG_Q5_0 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q5_0)

GGUF_CONFIG_Q4_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q4_K)
GGUF_CONFIG_Q4_0 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q4_0)
GGUF_CONFIG_Q4_1 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q4_1)

GGUF_CONFIG_Q3_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q3_K)
GGUF_CONFIG_Q2_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q2_K)

QUANTO_CONFIG_FLOAT8 = QuantoConfig(weights_dtype="float8")
QUANTO_CONFIG_INT8 = QuantoConfig(weights_dtype="int8")
QUANTO_CONFIG_INT4 = QuantoConfig(weights_dtype="int4")
QUANTO_CONFIG_INT2 = QuantoConfig(weights_dtype="int2")


TORCHAO_CONFIG_INT4WO = TorchAoConfig(quant_type="int4wo")
TORCHAO_CONFIG_INT4DQ = TorchAoConfig(quant_type="int4dq")
TORCHAO_CONFIG_INT8WO = TorchAoConfig(quant_type="int8wo")
TORCHAO_CONFIG_INT8DQ = TorchAoConfig(quant_type="int8dq")

TORCHAO_CONFIG_FLOAT8WO = TorchAoConfig(quant_type="float8wo")
TORCHAO_CONFIG_FLOAT8WO_E5M2 = TorchAoConfig(quant_type="float8wo_e5m2")
TORCHAO_CONFIG_FLOAT8WO_E4M3 = TorchAoConfig(quant_type="float8wo_e4m3")
TORCHAO_CONFIG_FLOAT8DQ = TorchAoConfig(quant_type="float8dq")
TORCHAO_CONFIG_FLOAT8DQ_E4M3 = TorchAoConfig(quant_type="float8dq_e4m3")
TORCHAO_CONFIG_FLOAT8_E4M3_TENSOR = TorchAoConfig(quant_type="float8_e4m3_tensor")
TORCHAO_CONFIG_FLOAT8_E4M3_ROW = TorchAoConfig(quant_type="float8_e4m3_row")

TORCHAO_CONFIG_UINT1WO = TorchAoConfig(quant_type="uint1wo")
TORCHAO_CONFIG_UINT2WO = TorchAoConfig(quant_type="uint2wo")
TORCHAO_CONFIG_UINT3WO = TorchAoConfig(quant_type="uint3wo")
TORCHAO_CONFIG_UINT4WO = TorchAoConfig(quant_type="uint4wo")
TORCHAO_CONFIG_UINT5WO = TorchAoConfig(quant_type="uint5wo")
TORCHAO_CONFIG_UINT6WO = TorchAoConfig(quant_type="uint6wo")
TORCHAO_CONFIG_UINT7WO = TorchAoConfig(quant_type="uint7wo")


QUANTIZER_CONFIGS = {
    "bnb_4bit_config_fp16": BNB_4BIT_CONFIG_FP16,
    "bnb_4bit_config_bf16": BNB_4BIT_CONFIG_BF16,
    "bnb_8bit_config_fp16": BNB_8BIT_CONFIG_FP16,
    "bnb_8bit_config_bf16": BNB_8BIT_CONFIG_BF16,
    "gguf_config_fp16": GGUF_CONFIG_FP16,
    "gguf_config_bf16": GGUF_CONFIG_BF16,
    "gguf_config_q8_k": GGUF_CONFIG_Q8_K,
    "gguf_config_q8_0": GGUF_CONFIG_Q8_0,
    "gguf_config_q8_1": GGUF_CONFIG_Q8_1,
    "gguf_config_q6_k": GGUF_CONFIG_Q6_K,
    "gguf_config_q5_k": GGUF_CONFIG_Q5_K,
    "gguf_config_q5_1": GGUF_CONFIG_Q5_1,
    "gguf_config_q5_0": GGUF_CONFIG_Q5_0,
    "gguf_config_q4_k": GGUF_CONFIG_Q4_K,
    "gguf_config_q4_0": GGUF_CONFIG_Q4_0,
    "gguf_config_q4_1": GGUF_CONFIG_Q4_1,
    "gguf_config_q3_k": GGUF_CONFIG_Q3_K,
    "gguf_config_q2_k": GGUF_CONFIG_Q2_K,
    "quanto_config_float8": QUANTO_CONFIG_FLOAT8,
    "quanto_config_int8": QUANTO_CONFIG_INT8,
    "quanto_config_int4": QUANTO_CONFIG_INT4,
    "quanto_config_int2": QUANTO_CONFIG_INT2,
    "torch_ao_config_int4wo": TORCHAO_CONFIG_INT4WO,
    "torch_ao_config_int4dq": TORCHAO_CONFIG_INT4DQ,
    "torch_ao_config_int8wo": TORCHAO_CONFIG_INT8WO,
    "torch_ao_config_int8dq": TORCHAO_CONFIG_INT8DQ,
    "torch_ao_config_float8wo": TORCHAO_CONFIG_FLOAT8WO,
    "torch_ao_config_float8wo_e5m2": TORCHAO_CONFIG_FLOAT8WO_E5M2,
    "torch_ao_config_float8wo_e4m3": TORCHAO_CONFIG_FLOAT8WO_E4M3,
    "torch_ao_config_float8dq": TORCHAO_CONFIG_FLOAT8DQ,
    "torch_ao_config_float8dq_e4m3": TORCHAO_CONFIG_FLOAT8DQ_E4M3,
    "torch_ao_config_float8_e4m3_tensor": TORCHAO_CONFIG_FLOAT8_E4M3_TENSOR,
    "torch_ao_config_float8_e4m3_row": TORCHAO_CONFIG_FLOAT8_E4M3_ROW,
    "torch_ao_config_uint1wo": TORCHAO_CONFIG_UINT1WO,
    "torch_ao_config_uint2wo": TORCHAO_CONFIG_UINT2WO,
    "torch_ao_config_uint3wo": TORCHAO_CONFIG_UINT3WO,
    "torch_ao_config_uint4wo": TORCHAO_CONFIG_UINT4WO,
    "torch_ao_config_uint5wo": TORCHAO_CONFIG_UINT5WO,
    "torch_ao_config_uint6wo": TORCHAO_CONFIG_UINT6WO,
    "torch_ao_config_uint7wo": TORCHAO_CONFIG_UINT7WO,
}

quant_type = Literal[
    "bnb_4bit_config_fp16",
    "bnb_4bit_config_bf16",
    "bnb_8bit_config_fp16",
    "bnb_8bit_config_bf16",
    "gguf_config_fp16",
    "gguf_config_bf16",
    "gguf_config_q8_k",
    "gguf_config_q8_0",
    "gguf_config_q8_1",
    "gguf_config_q6_k",
    "gguf_config_q5_k",
    "gguf_config_q5_1",
    "gguf_config_q5_0",
    "gguf_config_q4_k",
    "gguf_config_q4_0",
    "gguf_config_q4_1",
    "gguf_config_q3_k",
    "gguf_config_q2_k",
    "quanto_config_float8",
    "quanto_config_int8",
    "quanto_config_int4",
    "quanto_config_int2",
    "torch_ao_config_int4wo",
    "torch_ao_config_int4dq",
    "torch_ao_config_int8wo",
    "torch_ao_config_int8dq",
    "torch_ao_config_float8wo",
    "torch_ao_config_float8wo_e5m2",
    "torch_ao_config_float8wo_e4m3",
    "torch_ao_config_float8dq",
    "torch_ao_config_float8dq_e4m3",
    "torch_ao_config_float8_e4m3_tensor",
    "torch_ao_config_float8_e4m3_row",
    "torch_ao_config_uint1wo",
    "torch_ao_config_uint2wo",
    "torch_ao_config_uint3wo",
    "torch_ao_config_uint4wo",
    "torch_ao_config_uint5wo",
]


class ModelQuantizer:
    """
    Helper wrapper around the new diffusers quantisation back-end.

    Parameters
    ----------
    quant_method:
        One of the keys of your `QUANTIZER_CONFIGS` dict.
    """

    def __init__(self, quant_method: quant_type):
        if quant_method not in QUANTIZER_CONFIGS:
            raise ValueError(
                f"Unknown key {quant_method!r}; choose from {list(QUANTIZER_CONFIGS)}"
            )

        self.config = QUANTIZER_CONFIGS[quant_method]
        self.quantizer: DiffusersQuantizer = DiffusersAutoQuantizer.from_config(
            self.config
        )
        self.quantizer.validate_environment()  # raises early if bitsandbytes / torch-ao etc. are missing

    # ------------------------------------------------------------------------- #
    # Main entry-point
    # ------------------------------------------------------------------------- #
    def quantize(
        self,
        model: torch.nn.Module,
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    ) -> torch.nn.Module:
        """
        Convert `model` to its quantised counterpart and place shards according
        to `max_memory`.  Returns the same Python object, now modified.
        """

        # 1) Let the backend decide which dtype it needs (int8 / int4 / fp8 …)
        target_dtype = self.quantizer.update_torch_dtype(
            getattr(model, "dtype", torch.float32)
        )

        # 2) Replace Linear/Conv layers with quantised versions **before** moving
        #    anything onto a real device.
        self.quantizer.preprocess_model(model)

        # 3) Build (or adapt) a device map
        if max_memory is None:
            max_memory = get_balanced_memory(model, dtype=target_dtype, low_zero=False)
        max_memory = self.quantizer.adjust_max_memory(max_memory)

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=self.quantizer.modules_to_not_convert,
            dtype=target_dtype,
        )
        device_map = self.quantizer.update_device_map(
            device_map
        )  # backend-specific tweaks

        # 4) Ship parameters & buffers
        model = dispatch_model(model, device_map=device_map)

        # 5) Any backend-specific fix-ups (e.g. NF4 scale tables, fp16 cast of LN)
        model = self.quantizer.postprocess_model(model)

        return model
