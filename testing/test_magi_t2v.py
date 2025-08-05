from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("magi", "manifest/magi/magi_x2v_4_5b.yml", "t2v", save_path="/workspace/models", attention_type="flash", components_to_load=['transformer'], component_dtypes={"text_encoder": torch.float32})

space_prompt = "Deep space: a velvet-black void studded with shimmering stars. Center frame: a colossal spiral galaxy, its cerulean and rose arms rotating against cosmic darkness. In the foreground, a sleek silver spacecraft hovers silently, its hull reflecting swirling nebulae.  \
Through panoramic windows, two astronauts drift weightlessly, their visors glowing with the light of a nearby supernova. Suddenly, iridescent plasma tendrils burst from a distant pulsar, casting rippling shadows as electric arcs dance like liquid light through glittering interstellar dust. \
Below, a ringed ice planet spins serenely, its pale rings fractured by twin moons that cast elongated shadows across frozen plains. A golden beam of starlight pierces a drifting gas cloud, igniting the spacecraft in a brief halo before fading into silence. Ethereal synths swell as the spiral galaxy pulses ever so faintly, hinting at a living universe that endures long after the scene fades to black."

video = engine.run(
    prompt=space_prompt,
    height=480,
    width=832,
    duration=96,
    num_videos=1,
    num_inference_steps=64,
)

export_to_video(video[0], "test_magi_t2v_space.mp4", fps=24, quality=8)