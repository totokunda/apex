from src.preprocess.mask import MaskDrawPreprocessor

image = "assets/image/couple.jpg"
mask = "assets/mask/couple_mask.png"

preprocessor = MaskDrawPreprocessor()
output = preprocessor(image=image, mode="maskbbox", mask=mask)

output.mask.save("assets/test/maskbbox.png")

output = preprocessor(image=image, mode="maskpoint", mask=mask)

output.mask.save("assets/test/maskpoint.png")

output = preprocessor(image=image, mode="mask", mask=mask)

output.mask.save("assets/test/mask.png")

output = preprocessor(image=image, mode="bbox", bbox=[0.1, 0.1, 0.2, 0.2])

output.mask.save("assets/test/bbox.png")