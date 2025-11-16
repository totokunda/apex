from dotenv import load_dotenv
load_dotenv()
from src.engine.registry import UniversalEngine
import json 

#with open("inputs.json", "r") as f:
#    data = json.load(f)
#
#print(data)
#engine_kwargs = data["engine_kwargs"]
#inputs = data["inputs"]

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/hunyuanimage3/hunyuanimage-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path, attention_type="flash", 
    selected_components={
        "transformer": {
            "path": "/home/tosin_coverquick_co/apex-diffusion/gguf/hunyuanimage3_q2.Q2_K.gguf",
            "variant": "GGUF_Q2_K",
            "precision": "q2_k",
            "type": "gguf"
        }
    })

out = engine.run(
    prompt="The Death of Ophelia by John Everett Millais, Pre-Raphaelite painting, Ophelia floating in a river surrounded by flowers, detailed natural elements, melancholic and tragic atmosphere",
    num_inference_steps=30,
    height=1024,
    width=1024,
)

out[0].save("output3_q2k.png")