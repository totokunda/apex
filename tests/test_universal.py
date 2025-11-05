import json 
from src.engine.registry import UniversalEngine

file = "/Users/tosinkuye/apex-workspace/apex-engine/engine_settings.json"
with open(file, "r") as f:
    data = json.load(f)


engine_type = data["engine_type"]
model_type = data["model_type"]
selected_components = data["selected_components"]
inputs = data["inputs"]

engine = UniversalEngine(engine_type=engine_type, yaml_path=data["manifest_path"], model_type=model_type, selected_components=selected_components)

engine.run(**inputs)