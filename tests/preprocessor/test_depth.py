from src.preprocess.depth import MidasDepthPreprocessor, DepthAnythingV2Preprocessor


image = "assets/image/couple.jpg"

print("Testing MidasDepthPreprocessor")
preprocessor = MidasDepthPreprocessor()

output = preprocessor(image)
output.depth.save("assets/test/midas_depth.png")

print("Testing DepthAnythingV2Preprocessor")
preprocessor = DepthAnythingV2Preprocessor()

output = preprocessor(image)
output.depth.save("assets/test/depth_anything_v2.png")
