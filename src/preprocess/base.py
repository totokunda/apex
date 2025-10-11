from src.mixins import LoaderMixin, ToMixin
from torch import nn
from PIL import Image
import importlib
from typing import Dict, Any, List, Tuple
from transformers.image_processing_utils import ImageProcessingMixin
from src.utils.module import find_class_recursive
from transformers import AutoProcessor
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
from src.register import ClassRegister
from enum import Enum
from collections import OrderedDict
import numpy as np
import json


class BaseOutput(OrderedDict):
    """
    Lightweight output container that supports both attribute and dict-style access.

    - Initializes attributes from keyword arguments.
    - Keeps attributes and mapping entries in sync on assignment.
    - Preserves field order using subclass type annotations when available.
    - Provides tuple-style indexing (e.g., out[0]) over non-None fields in order.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        # Collect annotated fields from the MRO to preserve a meaningful order
        annotated_fields: Dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            annotated_fields.update(getattr(cls, "__annotations__", {}) or {})

        ordered_keys = (
            list(annotated_fields.keys()) if annotated_fields else list(kwargs.keys())
        )

        # Initialize annotated fields first (default to None if not provided)
        for key in ordered_keys:
            value = kwargs.get(key, None)
            if value is not None:
                super().__setitem__(key, value)
                super().__setattr__(key, value)
            else:
                # Ensure attribute exists even if not provided
                super().__setattr__(key, None)

        # Add any extra keys not present in annotations
        for key, value in kwargs.items():
            if key not in ordered_keys:
                super().__setitem__(key, value)
                super().__setattr__(key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        # Keep mapping and attributes in sync; add key when first set to non-None
        if not name.startswith("_") and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        # Mirror into attribute for convenience
        super().__setattr__(key, value)

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, str):
            # Dict-like access by key
            return dict(self.items())[k]
        # Tuple-like access by index or slice
        return self.to_tuple()[k]

    def to_tuple(self):
        # Convert to tuple of non-None values preserving key order
        return tuple(value for key, value in self.items() if value is not None)

    def _jsonify(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy scalars
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (list, tuple)):
            return [self._jsonify(v) for v in obj]
        if isinstance(obj, (dict, OrderedDict)):
            return {k: self._jsonify(v) for k, v in obj.items()}
        return obj

    def to_dict(self) -> Dict[str, Any]:
        # Return a JSON-serializable plain dict view
        return {k: self._jsonify(v) for k, v in self.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        return self.to_json()

    def __repr__(self) -> str:
        return self.to_json()


class PreprocessorType(Enum):
    IMAGE = "image"
    IMAGE_TEXT = "image_text"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    OTHER = "other"
    POSE = "pose"


class BasePreprocessor(LoaderMixin, ToMixin, nn.Module):
    def __init__(
        self,
        model_path: str = None,
        config_path: str = None,
        save_path: str = DEFAULT_PREPROCESSOR_SAVE_PATH,
        preprocessor_type: PreprocessorType = PreprocessorType.IMAGE,
        **kwargs
    ):
        super().__init__()
        if model_path:
            self.model_path = self._download(model_path, save_path)
        else:
            self.model_path = model_path
        if config_path:
            self.config_path = self._download(config_path, save_path)
        else:
            self.config_path = config_path
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

    def preprocess_bbox(self, bbox: List[float], shape: Tuple[int, int, int]):
        """Preprocess a bounding box"""
        x1, y1, x2, y2 = bbox
        h, w, _ = shape
        if isinstance(x1, float) and x1 < 1:
            x1 = x1 * w
        if isinstance(y1, float) and y1 < 1:
            y1 = y1 * h
        if isinstance(x2, float) and x2 < 1:
            x2 = x2 * w
        if isinstance(y2, float) and y2 < 1:
            y2 = y2 * h
        return x1, y1, x2, y2


preprocessor_registry = ClassRegister()
