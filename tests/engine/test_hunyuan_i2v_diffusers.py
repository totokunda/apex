import torch
from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video, load_image

model_id = "hunyuanvideo-community/HunyuanVideo-I2V"
transformer_path = "/dev/shm/models/components/hunyuanvideo-community_HunyuanVideo-I2V_transformer"

transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    transformer_path, subfolder="transformer", torch_dtype=torch.bfloat16
)

pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)

pipe.to("cuda")

# Enable memory 
prompt = "A man with short gray hair plays a red electric guitar."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
)

output = pipe(
    image=image,
    prompt=prompt,
    height=480,
    width=832,
    num_frames=61,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(420)
).frames[0]

export_to_video(output, "test_hunyuan_i2v_diffusers_gina.mp4", fps=15)