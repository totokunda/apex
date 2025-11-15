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
engine = UniversalEngine(yaml_path=yaml_path)

out = engine.run(
    prompt="A beautiful sunset over a calm ocean with a boat in the distance",
    num_inference_steps=50,
    height=1024,
    width=1024,
)

out[0].save("output3.png")