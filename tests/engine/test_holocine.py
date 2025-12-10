from dotenv import load_dotenv
import torch
load_dotenv()
from src.engine.registry import UniversalEngine
from diffusers.utils import export_to_video
yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/wan/holocine-2.2-14b-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)

out = engine.run(
    global_caption="The scene is set in a lavish, 1920s Art Deco ballroom during a masquerade party. [character1] is a mysterious woman with a sleek bob, wearing a sequined silver dress and an ornate feather mask. [character2] is a dapper gentleman in a black tuxedo, his face half-hidden by a simple black domino mask. The environment is filled with champagne fountains, a live jazz band, and dancing couples in extravagant costumes. This scene contains 5 shots.",
    shot_captions=[
        "Medium shot of [character1] standing by a pillar, observing the crowd, a champagne flute in her hand.",
        "Close-up of [character2] watching her from across the room, a look of intrigue on his visible features.",
        "Medium shot as [character2] navigates the crowd and approaches [character1], offering a polite bow.",
        "Close-up on [character1]'s eyes through her mask, as they crinkle in a subtle, amused smile.",
        "A stylish medium two-shot of them standing together, the swirling party out of focus behind them, as they begin to converse."
    ],
    duration=241,
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=40,
    height=480,
    width=832,
    guidance_scale=[4.0, 3.0],
    seed=42,
)

export_to_video(out[0], "output_holocine.mp4", fps=15)