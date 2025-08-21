from src.preprocess import StandinFacePreprocessor

image = "assets/image/man.png"
face_preprocessor = StandinFacePreprocessor()
face_output = face_preprocessor(image, extra_input=True)
face_output.face.save("face.png")