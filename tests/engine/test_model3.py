from dotenv import load_dotenv
load_dotenv()
import torch
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/qwenimage/nunchaku-qwenimage-1.0.0.v1.yml"
engine = UniversalEngine(engine_type="flux", yaml_path=yaml_path, selected_components={
    "transformer": {
        "path": "nunchaku-tech/nunchaku-qwen-image/svdq-int4_r128-qwen-image-lightningv1.1-8steps.safetensors",
        "variant": "int4_r128_lightning_v1_1_8steps"
    }
})
prompt = "The Death of Ophelia by John Everett Millais, Pre-Raphaelite painting, Ophelia floating in a river surrounded by flowers, detailed natural elements, melancholic and tragic atmosphere"

out = engine.run(
    prompt=prompt,
    num_inference_steps=8,
    height=1024,
    width=1024
)

out[0].save("output3_fibo.png")