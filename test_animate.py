from src.transformer.wan.animate.model import WanAnimateTransformer3DModel
from safetensors.torch import load_file
from src.converters.transformer_converters import WanAnimateTransformerConverter
from accelerate import init_empty_weights
from src.quantize.scaled_layer import patch_fpscaled_model
from src.quantize.load import load_gguf
import torch
from src.quantize.dequant import dequantize_tensor
from src.quantize.ggml_layer import patch_model

import argparse
import pathlib
from typing import Any, Dict, Tuple

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from src.converters.transformer_converters import WanAnimateTransformerConverter


fp8_weights = load_file("/home/tosin_coverquick_co/apex-diffusion/components/4d070faacff0a3bfc145235c1eb9432a1451c06f5be5da155a6d51fe16c58348_Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors")
config = {
  "_class_name": "WanAnimateTransformer3DModel",
  "_diffusers_version": "0.36.0.dev0",
  "added_kv_proj_dim": 5120,
  "attention_head_dim": 128,
  "cross_attn_norm": True,
  "eps": 1e-06,
  "face_encoder_hidden_dim": 1024,
  "face_encoder_num_heads": 4,
  "ffn_dim": 13824,
  "freq_dim": 256,
  "image_dim": 1280,
  "in_channels": 36,
  "inject_face_latents_blocks": 5,
  "latent_channels": 16,
  "motion_dim": 20,
  "motion_encoder_batch_size": 8,
  "motion_encoder_channel_sizes": None,
  "motion_encoder_dim": 512,
  "motion_encoder_size": 512,
  "motion_style_dim": 512,
  "num_attention_heads": 40,
  "num_layers": 40,
  "out_channels": 16,
  "patch_size": [
    1,
    2,
    2
  ],
  "pos_embed_seq_len": None,
  "qk_norm": "rms_norm_across_heads",
  "rope_max_seq_len": 1024,
  "text_dim": 4096
}

converter = WanAnimateTransformerConverter()
with init_empty_weights():
    model = WanAnimateTransformer3DModel.from_config(config)

patch_fpscaled_model(model)

converter.convert(fp8_weights)
model.load_state_dict(fp8_weights, assign=True, strict=True)
model.to("cpu")
print(model)
