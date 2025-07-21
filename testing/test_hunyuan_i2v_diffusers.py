import torch
from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from transformers import LlamaModel
from PIL import Image

model_id = "hunyuanvideo-community/HunyuanVideo-I2V"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)

pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)

pipe.to("cuda")

# Enable memory 
prompt = "A young woman reclines on crisp white sheets, her eyes flashing playful disbelief before her lips curl into a knowing smile—and she offers a single, flirtatious wink as her braids fall softly across her shoulder.."
image = Image.open("image.jpg")

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