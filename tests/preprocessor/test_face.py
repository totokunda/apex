from src.preprocess.face import FacePreprocessor


image = "assets/image/man.png"

print("Testing FacePreprocessor")
preprocessor = FacePreprocessor()

output = preprocessor(image)
print(output.bbox)
print(output.kps)
print(output.det_score)
print(output.landmark_3d_68)
print(output.pose)
print(output.landmark_2d_106)
print(output.gender)

output.images[0].save("assets/test/face.png")
