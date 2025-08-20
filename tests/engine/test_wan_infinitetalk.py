from src.engine import create_engine
from diffusers.utils import export_to_video
from PIL import Image

engine = create_engine("wan", "manifest/wan/wan_infinitetalk_14b.yml", "multitalk",  attention_type="sage", save_converted_weights=False)

prompt= "A man is talking.",
negative_prompt="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
video = "assets/video/man_talking.mp4"

video = engine.run(
    audio_paths={
        "person1": "assets/audio/singing.wav",
    },
    video=video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=448,
    width=896,
    duration=81,
    num_videos=1,
    num_inference_steps=40,
    guidance_scale=5.0,
    audio_guidance_scale=4.0
)

outfile_path = "test_wan_infinitetalk_v2v.mp4"
export_to_video(video[0], outfile_path, fps=25, quality=8)
