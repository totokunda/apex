import gguf
from typing import Literal
import warnings
from tqdm import tqdm
import torch
from typing import Dict
from src.quantize.ggml_tensor import GGMLTensor
from src.quantize.dequant import is_quantized, dequantize_tensor
from src.utils.dtype import convert_str_dtype
from loguru import logger
from src.converters.transformer_converters import TransformerConverter

transformer_converter = TransformerConverter()

T5_SD_MAP = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

STEP_SD_MAP = {
    # layers
    "blk.": "transformer.layers.",
    # attention norms
    "attn_norm": "attention_norm",
    # attention projections (unfused path for GGUF)
    "attn_q": "attention.wq",
    "attn_k": "attention.wk",
    "attn_v": "attention.wv",
    "attn_output": "attention.wo",
    # ffn norms
    "ffn_norm": "ffn_norm",
    # feed-forward weights (unfused path for GGUF)
    "ffn_gate": "feed_forward.ffn_gate",
    "ffn_up": "feed_forward.ffn_up",
    "ffn_down": "feed_forward.ffn_down",
    # embeddings
    "token_embd": "tok_embeddings.word_embeddings",
}


def remap_key(key: str, key_map: Literal["t5", "llama", "step"] = "t5"):

    if key_map == "t5":
        key_map = T5_SD_MAP
    elif key_map == "llama":
        key_map = LLAMA_SD_MAP
    elif key_map == "step":
        key_map = STEP_SD_MAP
    else:
        raise ValueError(f"Invalid key map: {key_map}")
    for k, v in key_map.items():
        key = key.replace(k, v)
    return key


def load_text_encoder_gguf(
    path: str,
    key_map: Literal["t5", "llama", "step"] = "t5",
    dequant_dtype: torch.dtype | str = torch.float16,
    **kwargs,
):
    if isinstance(dequant_dtype, str):
        dequant_dtype = convert_str_dtype(dequant_dtype)
    reader = gguf.GGUFReader(path)
    state_dict: Dict[str, GGMLTensor] = {}
    qtype_dict: Dict[str, int] = {}
    for tensor in tqdm(reader.tensors):
        name = remap_key(tensor.name, key_map)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            # Map quantized GGUF buffers to int8 to avoid downstream frameworks casting activations to uint8
            torch_tensor = torch.from_numpy(tensor.data)

            ggml_tensor = GGMLTensor(
                (
                    torch_tensor.view(torch.int8)
                    if is_quantized(torch_tensor)
                    else torch_tensor
                ),
                tensor_type=tensor.tensor_type,
                tensor_shape=shape,
                dequant_dtype=dequant_dtype,
            )

        state_dict[name] = ggml_tensor
        if tensor.name == "token_embd.weight":
            state_dict[name] = dequantize_tensor(
                ggml_tensor, dequant_dtype=dequant_dtype
            )
            if key_map == "t5":  # We duplicate the token embedding for t5
                state_dict["encoder.embed_tokens.weight"] = state_dict[name]

        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    return state_dict, qtype_dict


def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise TypeError(
            f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}"
        )
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def load_transformer_gguf(
    path: str,
    dequant_dtype: torch.dtype | str = torch.float16,
):
    if isinstance(dequant_dtype, str):
        dequant_dtype = convert_str_dtype(dequant_dtype)

    reader = gguf.GGUFReader(path)
    state_dict: Dict[str, GGMLTensor] = {}
    qtype_dict: Dict[str, int] = {}
    

    i = 0 
    for tensor in tqdm(reader.tensors):
        name = tensor.name
        shape = get_orig_shape(reader, name)
        if shape is None:
            # GGUF stores dims reversed
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            base = torch.from_numpy(tensor.data)

            # For F16/F32, present the logical shape now; quantized shapes are handled by dequant
            if tensor.tensor_type in {
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
            }:
                base = base.view(*shape)

            ggml_tensor = GGMLTensor(
                base,
                tensor_type=tensor.tensor_type,
                tensor_shape=shape,
                dequant_dtype=dequant_dtype,
                patches=[],
                requires_grad=False,
            )

        state_dict[name] = ggml_tensor
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

        

    return state_dict, qtype_dict


def load_gguf(
    path: str,
    type: Literal["text_encoder", "transformer"],
    key_map: Literal["t5", "llama", "step"] | None = None,
    **kwargs,
):
    if type == "text_encoder":
        return load_text_encoder_gguf(path, key_map, **kwargs)
    elif type == "transformer":
        return load_transformer_gguf(path, **kwargs)
    else:
        raise ValueError(f"Invalid type: {type}")
