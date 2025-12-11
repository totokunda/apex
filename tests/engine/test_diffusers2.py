from dotenv import load_dotenv
load_dotenv()
from diffusers import BriaFiboPipeline
import torch
import json

pipe = BriaFiboPipeline.from_pretrained(
    "briaai/FIBO",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
# pipe.enable_model_cpu_offload() # uncomment if you're getting CUDA OOM errors


json_prompt_generate = """{"short_description":"A dramatic and melancholic scene depicting Ophelia's death, as painted by John Everett Millais. She floats on a river, surrounded by a lush, overgrown natural environment. Her expression is one of sorrow and despair, with her eyes closed and lips slightly parted. The water is calm, reflecting the surrounding trees and the soft, diffused light. The overall atmosphere is somber and poignant, emphasizing the tragic nature of her demise.","objects":[{"description":"Ophelia, a young woman with long, dark hair, dressed in a simple, dark gown. Her face is pale and her eyes are closed, conveying deep sorrow.","location":"center","relationship":"She is the central figure, floating on the water.","relative_size":"large within frame","shape_and_color":"Human form, wearing a dark, flowing dress.","texture":"Fabric appears soft and slightly rumpled.","appearance_details":"Her hair is spread around her head, some strands touching the water. Her hands are clasped loosely.","pose":"Lying down, floating on her back.","expression":"Despairing, sorrowful, peaceful.","clothing":"A dark, simple gown with long sleeves.","action":"Floating on the water.","gender":"female","skin_tone_and_texture":"Fair skin, smooth texture.","orientation":"Horizontal, floating"},{"description":"A dense arrangement of water lilies and other aquatic plants, with vibrant pink and white flowers.","location":"surrounding Ophelia, midground and foreground","relationship":"They form a natural, albeit somber, bed around Ophelia.","relative_size":"medium to large cluster","shape_and_color":"Various organic shapes, predominantly pink and white flowers with green leaves and stems.","texture":"Petals appear soft and delicate, leaves are smooth.","appearance_details":"Some flowers are in full bloom, others are buds. Stems are submerged in the water.","number_of_objects":1,"orientation":"Various, growing upwards and outwards"},{"description":"A calm river, reflecting the surrounding trees and sky.","location":"bottom half of the frame","relationship":"It is the medium on which Ophelia floats.","relative_size":"large","shape_and_color":"Fluid, dark blue-green with lighter reflections.","texture":"Smooth, with subtle ripples.","appearance_details":"Reflections of trees and sky are visible on the surface.","orientation":"Horizontal"},{"description":"A dense, dark forest with tall trees and thick foliage.","location":"background","relationship":"It forms the backdrop to the scene, creating a sense of isolation and nature.","relative_size":"large","shape_and_color":"Vertical tree trunks, dark green and brown foliage.","texture":"Bark appears rough, foliage is dense and textured.","appearance_details":"The trees are closely packed, creating a sense of depth and enclosure.","orientation":"Vertical"}],"background_setting":"A dense, dark forest lines the banks of a calm river. The trees are tall and ancient, with thick foliage that creates a sense of mystery and isolation. The river itself is dark and reflective, mirroring the somber sky and surrounding greenery.","lighting":{"conditions":"soft, diffused daylight","direction":"coming from above and slightly to the left","shadows":"soft, elongated shadows cast by the trees and foliage, creating a gentle contrast"},"aesthetics":{"composition":"centered composition with Ophelia as the focal point, framed by the natural elements","color_scheme":"muted greens, browns, and blues, with accents of pink and white from the flowers","mood_atmosphere":"melancholic, tragic, serene","aesthetic_score":"very high","preference_score":"very high"},"photographic_characteristics":{"depth_of_field":"deep, with elements in the foreground, midground, and background all in focus","focus":"sharp focus on Ophelia and the surrounding flora","camera_angle":"eye-level","lens_focal_length":"standard lens (e.g., 35mm-50mm)"},"style_medium":"oil painting","context":"This is a fine art painting, likely a depiction of a scene from Shakespeare's Hamlet, focusing on the tragic death of Ophelia. It would be suitable for a gallery exhibition, a book cover for a dramatic novel, or as part of an art collection.","artistic_style":"Pre-Raphaelite, dramatic realism"}"""


def get_default_negative_prompt(existing_json: dict) -> str:
    negative_prompt = ""
    style_medium = existing_json.get("style_medium", "").lower()
    if style_medium in ["photograph", "photography", "photo"]:
        negative_prompt = """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
    return negative_prompt


negative_prompt = get_default_negative_prompt(json.loads(json_prompt_generate))

# -------------------------------
# Run Image Generation
# -------------------------------
# Generate the image from the structured json prompt
results_generate = pipe(
    prompt=json_prompt_generate, num_inference_steps=1, guidance_scale=5, negative_prompt=negative_prompt,
    generator=torch.Generator(device="cuda").manual_seed(42)
)
results_generate.images[0].save("image_generate.png")