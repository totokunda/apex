from dotenv import load_dotenv
load_dotenv()
import torch
from diffusers import Flux2Pipeline, FlowMatchEulerDiscreteScheduler
import requests
import io
from huggingface_hub import get_token

repo_id = "/home/tosin_coverquick_co/apex-diffusion/components/black-forest-labs_FLUX.2-dev"
device = "cuda:0"
torch_dtype = torch.bfloat16

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "black-forest-labs/FLUX.2-dev", subfolder="scheduler",
)

def remote_text_encoder(prompts):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }
    )
    prompt_embeds = torch.load(io.BytesIO(response.content))
    return prompt_embeds.to(device)

pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=None, torch_dtype=torch_dtype, scheduler=scheduler,
).to(device)

prompt = "A warmly lit study filled with books, chalk dust drifting gently in the air. Albert Einstein stands at a large whiteboard covered in dense, elegant equations—relativity tensors, integrals, diagrams of spacetime curvature. His iconic wild hair casts a soft silhouette under the overhead lamp. He pauses, thinking deeply, then steps forward and adds a new line of mathematics with quick, confident strokes. The camera moves slowly around him, capturing the intensity in his eyes and the complexity of the symbols on the board. Papers clutter the desk behind him, and faint sunlight pours through tall windows, illuminating motes of dust as he works. The atmosphere feels brilliant and contemplative, as though a breakthrough is moments away."

image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, #28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

image.save("flux2_output_einstein_seed_42_diffusers.png")
