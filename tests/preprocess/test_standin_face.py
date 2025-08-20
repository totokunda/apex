from src.preprocess.standin.face import FacePreprocessor

image = "assets/image/man.png"
face_preprocessor = FacePreprocessor()
face_output = face_preprocessor(image, extra_input=True)
face_output.face.save("face.png")