from dotenv import load_dotenv
import torch
load_dotenv()
from src.engine.registry import UniversalEngine

yaml_path = "/home/tosin_coverquick_co/apex/manifest/engine/zimage/zimage-turbo-1.0.0.v1.yml"
engine = UniversalEngine(yaml_path=yaml_path)

prompt = "A warmly lit study filled with books, chalk dust drifting gently in the air. Albert Einstein stands at a large whiteboard covered in dense, elegant equations—relativity tensors, integrals, diagrams of spacetime curvature. His iconic wild hair casts a soft silhouette under the overhead lamp. He pauses, thinking deeply, then steps forward and adds a new line of mathematics with quick, confident strokes. The camera moves slowly around him, capturing the intensity in his eyes and the complexity of the symbols on the board. Papers clutter the desk behind him, and faint sunlight pours through tall windows, illuminating motes of dust as he works. The atmosphere feels brilliant and contemplative, as though a breakthrough is moments away."

out = engine.run(
    prompt=prompt,
    height=1024,
    width=1024,
    seed=42
)

out[0].save("output_zimage.png")