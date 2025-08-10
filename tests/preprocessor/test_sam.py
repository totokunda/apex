from src.preprocess.sam import SAMPreprocessor
from PIL import Image
import os


image = "assets/image/couple.jpg"
mask = "assets/mask/couple_mask.png"

test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)


print("Testing SAMPreprocessor with input_box")
sam = SAMPreprocessor()
output = sam(image=image, task_type="input_box", input_box=[0.1, 0.1, 0.9, 0.9])
# Save top mask if available
if output.masks is not None:
    top_mask = (output.masks[0] * 255).astype("uint8")
    Image.fromarray(top_mask).save(os.path.join(test_dir, "sam_input_box_mask.png"))


print("Testing SAMPreprocessor with mask input (mask)")
sam = SAMPreprocessor()
output = sam(image=image, task_type="mask", mask=mask)
if output.masks is not None:
    top_mask = (output.masks[0] * 255).astype("uint8")
    Image.fromarray(top_mask).save(os.path.join(test_dir, "sam_mask_input.png"))


print("Testing SAMPreprocessor with mask_point")
sam = SAMPreprocessor()
output = sam(image=image, task_type="mask_point", mask=mask)
if output.masks is not None:
    top_mask = (output.masks[0] * 255).astype("uint8")
    Image.fromarray(top_mask).save(os.path.join(test_dir, "sam_mask_point.png"))


print("Testing SAMPreprocessor with mask_box")
sam = SAMPreprocessor()
output = sam(image=image, task_type="mask_box", mask=mask)
if output.masks is not None:
    top_mask = (output.masks[0] * 255).astype("uint8")
    Image.fromarray(top_mask).save(os.path.join(test_dir, "sam_mask_box.png"))


