from diffusers.utils import export_to_video
from PIL import Image
import torch
from src.engine.wan_engine import WanEngine, ModelType

engine = WanEngine(
        yaml_path="manifest/wan_x2v_camera_1.3b.yml",
        model_type=ModelType.CAMERA,
        save_path="/mnt/localssd/apex-models",  # Change this to your desired save path,  # Change this to your desired save path
        components_to_load=["transformer"],
        component_dtypes={"vae": torch.bfloat16}
    )

poses = "data/poses/pan_left.txt"
image = "data/images/img.png"
prompt = "A lone boat battles against torrential waves under a stormy sky. The ocean roars with fury, crashing against the hull as the vessel rocks violently. The camera performs a smooth cinematic pan from right to left, encircling the boat to reveal its struggle from multiple angles, capturing the chaos of the sea and the intensity of the storm."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

height = 480
width = 640

video = engine.run(
    camera_poses=poses,
    start_image=image,
    height=height,
    width=width,
    prompt=prompt,
    negative_prompt=negative_prompt,
    use_cfg_guidance=True,
    duration="81f",
    num_videos=1,
    guidance_scale=6.0,
    num_inference_steps=50,
)

export_to_video(video[0], "t2v_1.3b_camera_420.mp4", fps=16, quality=8)