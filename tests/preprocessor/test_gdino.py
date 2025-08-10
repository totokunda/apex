from src.preprocess.gdino import GDINOPreprocessor

image = "assets/image/dog.png"

preprocessor = GDINOPreprocessor()
output = preprocessor(image, classes=["dog"])

print(output.boxes, output.confidences, output.class_ids, output.class_names)

output = preprocessor(image, caption="a dog")

print(output.boxes, output.confidences, output.class_ids, output.class_names)