from src.preprocess.ram import RAMPreprocessor

image = "assets/image/couple.jpg"

preprocessor = RAMPreprocessor()

output = preprocessor(image=image)

print(output.tags)
print(output.tags_c)