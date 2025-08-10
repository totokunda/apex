from src.engine import create_engine
from diffusers.utils import export_to_video
import torch
from diffusers.utils import export_to_video, load_image


engine = create_engine("cosmos2", "manifest/cosmos/cosmos2_x2v_14b.yml", "i2v", save_path="/workspace/models", postprocessors_to_load=["cosmos.guardrail"], attention_type="sdpa", component_dtypes={"text_encoder": torch.float32})

prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yellow-scrubber.png"
)
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

video = engine.run(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=704,
    width=1280,
    duration=93,
    num_videos=1,
    num_inference_steps=35,
    guidance_scale=7.0,
    generator=torch.Generator('cuda').manual_seed(1)
)


export_to_video(video[0], "test_cosmos2_x2v_14b_scrubber_short.mp4", fps=16, quality=8)