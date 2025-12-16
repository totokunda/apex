import torch

dtype = torch.bfloat16
device = "cuda:0"
from diffusers import HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled", torch_dtype=dtype)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

generator = torch.Generator(device=device).manual_seed(42)
prompt = "The woman smile slowly fades as she backs away from the camera with a look of fear, before dropping her sign and running away."
image = load_image("/home/tosin_coverquick_co/apex/images/apex_woman_image.png")

video = pipe(
    prompt=prompt,
    image=image,
    generator=generator,
    num_frames=121,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output_hydiff_woman.mp4", fps=24)
