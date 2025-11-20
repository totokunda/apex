from dotenv import load_dotenv
load_dotenv()
from src.engine.registry import UniversalEngine
import json 

with open("inputs.json", "r") as f:
   data = json.load(f)


engine_kwargs = data["engine_kwargs"]
inputs = data["inputs"]

engine = UniversalEngine(**engine_kwargs)

out = engine.run(
    **inputs
)

# Save output based on type (list of frames/images or single image)
if isinstance(out, list) and len(out) > 0:
    if hasattr(out[0], 'save'):
        # It's a list of PIL images (video frames)
        out[0].save("output.gif", save_all=True, append_images=out[1:], duration=1000/inputs.get("fps", 16), loop=0)
        print("Saved output.gif")
    elif isinstance(out[0], str):
        # It's a list of paths
        print(f"Output paths: {out}")
elif hasattr(out, 'save'):
    # Single PIL image
    out.save("output.png")
    print("Saved output.png")
else:
    print("Output format not recognized for auto-saving:", type(out))
