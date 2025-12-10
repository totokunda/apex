from dotenv import load_dotenv
load_dotenv()
from diffusers.utils import export_to_video
from src.engine.registry import UniversalEngine
import subprocess

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/wan/wan-2.2-14b-speech-to-video-1.0.0.v1.yml"
audio_path = "/home/tosin_coverquick_co/apex/only_one.m4a"

engine = UniversalEngine(yaml_path=yaml_path, attention_type="flash")
out = engine.run(
    image='/home/tosin_coverquick_co/apex/output3_fibo.png',
    audio=audio_path,
    sampling_rate=16000,
    prompt="The man sings passionately as his eyes are closed with a sad expression.",
    negative_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, \
        ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, \
        poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, \
        bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross \
        proportions, malformed limbs, missing arms, missing legs, extra arms, extra \
        legs, fused fingers, too many fingers, long neck, username, watermark, signature",
    num_inference_steps=30
)

video_path = "output3_fibo_singing.mp4"
export_to_video(out[0], video_path, fps=16)

# Add the original audio to the generated video using ffmpeg
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "output3_fibo_singing_with_audio.mp4",
    ],
    check=True,
)