from src.mixins import DownloadMixin, LoaderMixin, ToMixin
from torch import nn
from PIL import Image
import importlib
from typing import Dict, Any
from transformers.image_processing_utils import ImageProcessingMixin
from src.utils.module_utils import find_class_recursive
from transformers import AutoProcessor
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
from src.register import ClassRegister
from enum import Enum

class PreprocessorType(Enum):
    IMAGE = "image"
    IMAGE_TEXT = "image_text"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    OTHER = "other"
    POSE = "pose"

class BasePreprocessor(DownloadMixin, LoaderMixin, ToMixin, nn.Module):
    def __init__(self, model_path: str = None, save_path: str = DEFAULT_PREPROCESSOR_SAVE_PATH, preprocessor_type: PreprocessorType = PreprocessorType.IMAGE, **kwargs):
        super().__init__()
        if model_path:  
            self.model_path = self._download(model_path, save_path)
        else:
            self.model_path = model_path
        self.kwargs = kwargs
        self.preprocessor_type = preprocessor_type

    def load_processor(self, processor_path: Dict[str, Any] | str) -> AutoProcessor:
        try:
            processor_class = find_class_recursive(
                importlib.import_module("transformers"), self.processor_class
            )
            if self._is_huggingface_repo(processor_path):
                if len(processor_path.split("/")) > 2:
                    subfolder = "/".join(processor_path.split("/")[2:])
                    processor_path = "/".join(processor_path.split("/")[:2])
                    return processor_class.from_pretrained(
                        processor_path,
                        subfolder=subfolder,
                        save_dir=self.config_save_path,
                    )
                else:
                    return processor_class.from_pretrained(
                        processor_path, save_dir=self.config_save_path
                    )
            else:
                return processor_class.from_pretrained(processor_path)
        except Exception as e:
            processor_config = self.fetch_config(processor_path)
            processor_class = find_class_recursive(
                importlib.import_module("transformers"),
                processor_config[self.find_key_with_type(processor_config)],
            )
            if not issubclass(processor_class, ImageProcessingMixin):
                processor_class = find_class_recursive(
                    importlib.import_module("transformers"), self.processor_class
                )
            return processor_class(**processor_config)
        
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


preprocessor_registry = ClassRegister()