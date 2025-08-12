from src.text_encoder.text_encoder import TextEncoder

model_path = 'umt5.Q4_K.gguf'
config_path = 'apex-diffusion/components/Wan-AI_Wan2.1-I2V-14B-480P-Diffusers/text_encoder/config.json'

text_encoder = TextEncoder(config={
    "config_path": config_path,
    "model_path": model_path,
    "base": "UMT5EncoderModel",
    "type": "text_encoder",
    "gguf_kwargs": {
        "key_map": "t5",
        "dequant_dtype": "float16"
    },
}, enable_cache=False)

prompt_embeds = text_encoder.encode("Hello, world")
print(prompt_embeds.shape, prompt_embeds.dtype)