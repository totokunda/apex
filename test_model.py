from dotenv import load_dotenv
load_dotenv()
from src.engine.registry import UniversalEngine
import json 

with open("inputs.json", "r") as f:
    data = json.load(f)

print(data)
engine_kwargs = data["engine_kwargs"]
inputs = data["inputs"]

engine = UniversalEngine(**engine_kwargs)
out = engine.run(**inputs)
out[0].save("output.png")