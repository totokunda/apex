import torch
from dotenv import load_dotenv
load_dotenv()
from diffusers import FluxPipeline



pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

prompt = "A frog holding a sign that says hello world"
generator = torch.Generator(device="cuda").manual_seed(3469497462)
image = pipe(
    prompt,
    height=704,
    width=1280,
    true_cfg_scale=1,
    num_inference_steps=28,
    guidance_scale=4.5,
    generator=generator,
).images[0]
image.save("flux-krea-dev.png")
