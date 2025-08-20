import numpy as np
from PIL import Image
from src.preprocess.canvas import RegionCanvasPreprocessor

region_canvas_preprocessor = RegionCanvasPreprocessor(use_aug=True)
image = Image.open('assets/image/couple.jpg')
mask = Image.open('assets/mask/couple_mask.png')

canvas = region_canvas_preprocessor(image, mask)

Image.fromarray(canvas).save('assets/image/couple_canvas.png')