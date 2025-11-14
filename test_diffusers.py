from diffusers import HunyuanImagePipeline
import torch

device = "cuda:0"
dtype = torch.bfloat16
repo = "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"

pipe = HunyuanImagePipeline.from_pretrained(repo, torch_dtype=dtype)
pipe = pipe.to(device)

prompt = "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word “Tencent” on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."

generator = torch.Generator(device=device).manual_seed(649151)
out = pipe(
    prompt, 
    num_inference_steps=50, 
    height=2048, 
    width=2048, 
    generator=generator,
).images[0]

out.save("test_hyimage_output.png")
