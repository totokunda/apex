from src.preprocess.subject import SubjectPreprocessor
from PIL import Image
import numpy as np
import os


image = "assets/image/couple.jpg"
mask = "assets/mask/couple_mask.png"

test_dir = "assets/test"
os.makedirs(test_dir, exist_ok=True)


print("Testing SubjectPreprocessor plain mode")
pre = SubjectPreprocessor(mode="plain", use_crop=False, roi_only=False)
out = pre(image=image)
Image.fromarray(out.image).save(os.path.join(test_dir, "subject_plain.png"))


print("Testing SubjectPreprocessor salient mode")
pre = SubjectPreprocessor(mode="salient", use_crop=False, roi_only=False)
out = pre(image=image)
Image.fromarray(out.image).save(os.path.join(test_dir, "subject_salient.png"))
Image.fromarray(out.mask).save(os.path.join(test_dir, "subject_salient_mask.png"))


print("Testing SubjectPreprocessor mask mode")
pre = SubjectPreprocessor(mode="mask", use_crop=True, roi_only=True)
out = pre(image=image, mask=mask)
Image.fromarray(out.image).save(os.path.join(test_dir, "subject_mask.png"))
Image.fromarray(out.mask).save(os.path.join(test_dir, "subject_mask_mask.png"))


print("Testing SubjectPreprocessor bbox mode")
pre = SubjectPreprocessor(mode="bbox", use_crop=True, roi_only=False)
out = pre(image=image, bbox=[0.1, 0.1, 0.7, 0.7])
Image.fromarray(out.image).save(os.path.join(test_dir, "subject_bbox.png"))
Image.fromarray(out.mask).save(os.path.join(test_dir, "subject_bbox_mask.png"))


