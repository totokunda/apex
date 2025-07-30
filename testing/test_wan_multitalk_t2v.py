from src.engine import create_engine
from diffusers.utils import export_to_video
import torch

engine = create_engine("wan", "manifest/wan/wan_multitalk_14b.yml", "multitalk", save_path="/workspace/models", attention_type="flash3", component_dtypes={"text_encoder": torch.float32})

prompt = "A warm, intimate close‑up of a young Black man with a neatly trimmed beard and round, gold‑rimmed glasses, standing shirtless in a softly lit living room. His curly hair catches the morning light as his eyes flutter closed, and his breath catches with each heartfelt note of a gentle, acoustic melody. The camera slowly tracks in on the subtle reflections dancing in his lenses, the slight tremor at the corners of his mouth, and the raw emotion in his voice as it fills the quiet space—every resonant syllable unfolding like a tender confession."
negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
image = '/workspace/apex/data/face.jpg'
audio = '/workspace/apex/data/sound.m4a'

video = engine.run(
    audio_paths={
        "person1": audio,
    },
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=832,
    width=448,
    num_videos=1,
    num_inference_steps=40,
    guidance_scale=5.0,
    audio_guidance_scale=4.0,
)
outfile_path = "test_wan_multitalk_t2v.mp4"
export_to_video(video[0], outfile_path, fps=25, quality=8)

## use ffmpeg to merge the audio and video
import subprocess
subprocess.run(["ffmpeg", "-i", outfile_path, "-i", audio, "-c:v", "copy", "-c:a", "aac", "-shortest", "-strict", "experimental", f"{outfile_path}_merged.mp4"])