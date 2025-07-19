import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import LlamaModel

model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
text_encoder = LlamaModel.from_pretrained(
    model_id, subfolder="text_encoder", torch_dtype=torch.float16
)
pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16)

pipe.to("cuda")

# Enable memory 
prompt = "A beautiful woman in a flowing red dress walks through a dimly lit city street, neon lights reflecting off wet pavement, her silhouette glowing as the camera slowly tracks her from behind."

output = pipe(
    prompt=prompt,
    height=480,
    width=832,
    num_frames=61,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(69)
).frames[0]

export_to_video(output, "test_hunyuan_t2v_diffusers_woman.mp4", fps=15)