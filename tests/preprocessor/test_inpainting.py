from src.preprocess.inpainting.inpainting import InpaintingPreprocessor
import os 
from PIL import Image

preprocessor = InpaintingPreprocessor()
test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)

image = "/Users/tosinkuye/apex/assets/image/couple.jpg"
mask = "/Users/tosinkuye/apex/assets/mask/couple_mask.png"

RUN_TESTS = [
    "salient",
    "mask",
    "bbox",
    "label",
    "caption",
    "salientmasktrack",
    "salientbboxtrack",
    "masktrack",
    "maskbboxtrack",
    "maskpointtrack",
    "bboxtrack",
]

for test in RUN_TESTS:
    print(f"Running {test} test")
    output = preprocessor(
        image=image,
        mode=test,
        return_mask=True,
        bbox=[0.01, 0.01, 0.99, 0.99],
        label="couple",
        caption="A couple kissing passionately",
        mask=mask,
    )
    out_image = Image.fromarray(output.image)
    out_image.save(os.path.join(test_dir, f"couple_{test}.png"))
    out_mask = Image.fromarray(output.mask)
    out_mask.save(os.path.join(test_dir, f"couple_{test}_mask.png"))

    print(f"Finished {test} test")
