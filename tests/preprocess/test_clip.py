import torch
from src.preprocess.clip import CLIPPreprocessor
from src.utils.defaults import DEFAULT_DEVICE

model_path = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/image_encoder"
config_path = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/image_processor"

preprocessor = CLIPPreprocessor(
    model_path=model_path,
    preprocessor_path=config_path,
    processor_class="CLIPImageProcessor",
).to(device=DEFAULT_DEVICE, dtype=torch.float16)

image = [
    "https://static.independent.co.uk/2024/08/12/18/newFile-2.jpg",
    "https://media.istockphoto.com/id/814423752/photo/eye-of-model-with-colorful-art-make-up-close-up.jpg?s=612x612&w=0&k=20&c=l15OdMWjgCKycMMShP8UK94ELVlEGvt7GmB_esHWPYE=",
]

print(preprocessor(image).shape)