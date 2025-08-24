import os
os.environ["APEX_HOME_DIR"] = "/mnt/localssd"
from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("wan", "wan-2-2-a14b-text-to-video-1-0-0-v1", "t2i", component_dtypes={"text_encoder": torch.float32}, denoise_type="moe")

prompt = "Deep space: a velvet-black void studded with shimmering stars. Center frame: a colossal spiral galaxy, its cerulean and rose arms rotating against cosmic darkness. In the foreground, a sleek silver spacecraft hovers silently, its hull reflecting swirling nebulae.  \
Through panoramic windows, two astronauts drift weightlessly, their visors glowing with the light of a nearby supernova. Suddenly, iridescent plasma tendrils burst from a distant pulsar, casting rippling shadows as electric arcs dance like liquid light through glittering interstellar dust. \
Below, a ringed ice planet spins serenely, its pale rings fractured by twin moons that cast elongated shadows across frozen plains. A golden beam of starlight pierces a drifting gas cloud, igniting the spacecraft in a brief halo before fading into silence. Ethereal synths swell as the spiral galaxy pulses ever so faintly, hinting at a living universe that endures long after the scene fades to black."
negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

video = engine.run(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=1024,
    width=1024,
    num_images=1,
    num_inference_steps=50,
    guidance_scale=[4.0, 3.0]
)

export_to_video(video[0], "test_wan_22_t2i_bunny.mp4", fps=16, quality=10)