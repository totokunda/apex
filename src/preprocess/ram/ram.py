# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import numpy as np
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from typing import Union, List
from PIL import Image

from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
    BaseOutput,
)

from src.utils.defaults import DEFAULT_DEVICE
from src.utils.preprocessors import MODEL_WEIGHTS

class RAMOutput(BaseOutput):
    tags: List[str]
    tags_c: List[str]

@preprocessor_registry("ram")
class RAMPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_path: str | None = None,
        image_size: int = 384,
        ram_type: str = "swin_l",
        return_lang: List[str] = None,
        device: str | torch.device = DEFAULT_DEVICE,
        **kwargs,
    ):
        if model_path is None:
            model_path = MODEL_WEIGHTS["ram"]
        super().__init__(
            model_path=model_path, preprocessor_type=PreprocessorType.IMAGE, **kwargs
        )

        try:
            from ram.models import ram_plus
            from ram import inference_ram
        except ImportError as e:
            import warnings
            warnings.warn(
                "please pip install ram package"
            )
            raise ImportError("RAM package not available")

        self.return_lang = (
            return_lang if return_lang is not None else ["en"]
        )  # ['en', 'zh']

        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        delete_tag_index = []
        self.model = (
            ram_plus(
                pretrained=self.model_path,
                image_size=image_size,
                vit=ram_type,
                delete_tag_index=delete_tag_index,
            )
            .eval()
            .to(self.device)
        )

        self.ram_transform = Compose(
            [
                Resize((image_size, image_size)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.inference_ram = inference_ram

    def __call__(self, image: Union[Image.Image, np.ndarray, str]):
        image = self._load_image(image)
        image_ann_trans = self.ram_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tags_e, tags_c = self.inference_ram(image_ann_trans, self.model)

        tags_e_list = [tag.strip() for tag in tags_e.strip().split("|")]
        tags_c_list = [tag.strip() for tag in tags_c.strip().split("|")]

        if len(self.return_lang) == 1 and "en" in self.return_lang:
            return RAMOutput(tags=tags_e_list, tags_c=tags_c_list)
        elif len(self.return_lang) == 1 and "zh" in self.return_lang:
            return RAMOutput(tags=tags_c_list, tags_c=tags_c_list)
        else:
            return RAMOutput(tags=tags_e_list, tags_c=tags_c_list)

    def __str__(self): 
        return f"RAMPreprocessor(return_lang={self.return_lang})"

    def __repr__(self):
        return self.__str__()
