from src.mixins import DownloadMixin, LoaderMixin, ToMixin
from torch import nn
from PIL import Image
from io import BytesIO
import requests
from src.utils.defaults import DEFAULT_HEADERS
import numpy as np
import torch
from typing import Union, Callable
import PIL


class BasePreprocessor(DownloadMixin, LoaderMixin, ToMixin, nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
