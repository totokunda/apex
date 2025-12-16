import os 
os.environ['APEX_HOME_DIR'] = "/mnt/localssd"
from src.engine import create_engine
from diffusers.utils import export_to_video
from PIL import Image

engine = create_engine("wan", "manifest/wan/wan_infinitetalk_14b.yml", "multitalk",  attention_type="flash", save_converted_weights=False)

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

# Mux audio into the generated video using ffmpeg via os.system
audio_file = "assets/audio/singing.wav"
muxed_outfile_path = outfile_path.replace(".mp4", "_with_audio.mp4")
ffmpeg_cmd = f'ffmpeg -y -i "{outfile_path}" -i "{audio_file}" -c:v copy -c:a aac -shortest "{muxed_outfile_path}"'
os.system(ffmpeg_cmd)
os.replace(muxed_outfile_path, outfile_path)