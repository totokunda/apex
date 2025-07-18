from PIL import Image
from src.preprocess.base import BasePreprocessor, preprocessor_registry, PreprocessorType
from typing import Union
from src.utils.defaults import (
    DEFAULT_CONFIG_SAVE_PATH,
    DEFAULT_PREPROCESSOR_SAVE_PATH,
    DEFAULT_HEADERS,
)
import numpy as np
import torch
from typing import Union, Dict, Any, List


@preprocessor_registry("clip")
class CLIPPreprocessor(BasePreprocessor):

    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        model_config_path: str = None,
        model_config: Dict[str, Any] | None = None,
        save_path: str = DEFAULT_PREPROCESSOR_SAVE_PATH,
        config_save_path: str = DEFAULT_CONFIG_SAVE_PATH,
        processor_class: str = "AutoProcessor",
        model_class: str = "AutoModel",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(model_path=model_path, preprocessor_type=PreprocessorType.IMAGE)
        self.config_save_path = config_save_path
        self.processor_class = processor_class
        self.model_class = model_class
        self.processor = self.load_processor(preprocessor_path)
        self.model_path = self._download(model_path, save_path)

        self.model = self._load_model(
            {
                "type": "clip",
                "base": model_class,
                "model_path": self.model_path,
                "config_path": model_config_path,
                "config": model_config,
            },
            module_name="transformers",
        )
        

    def find_key_with_type(self, config: Dict[str, Any]) -> str:
        for key, value in config.items():
            if "type" in key:
                return key
        return None

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
        hidden_states_layer: int = -1,
        **kwargs,
    ):
        if isinstance(image, list):
            images = [self._load_image(img) for img in image]
        else:
            images = [self._load_image(image)]

        device = self.model.device
        dtype = self.model.dtype

        processed_images = self.processor(images, return_tensors="pt", **kwargs).to(
            device=device, dtype=dtype
        )
        image_embeds = self.model(**processed_images, output_hidden_states=True)

        image_embeds = image_embeds.hidden_states[hidden_states_layer]
        return image_embeds

    def __str__(self):
        return f"CLIPPreprocessor(model={self.model}, preprocessor={self.processor})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":

    model_path = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/image_encoder"
    config_path = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/image_processor"

    preprocessor = CLIPPreprocessor(
        model_path=model_path,
        preprocessor_path=config_path,
        processor_class="CLIPImageProcessor",
    ).to(device="cuda", dtype=torch.bfloat16)

    image = [
        "https://static.independent.co.uk/2024/08/12/18/newFile-2.jpg",
        "https://media.istockphoto.com/id/814423752/photo/eye-of-model-with-colorful-art-make-up-close-up.jpg?s=612x612&w=0&k=20&c=l15OdMWjgCKycMMShP8UK94ELVlEGvt7GmB_esHWPYE=",
    ]

    print(preprocessor(image).shape)
