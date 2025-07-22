from abc import ABC, abstractmethod
import torch
from src.mixins import LoaderMixin
from src.register import ClassRegister


class BasePostprocessor(ABC, LoaderMixin):
    def __init__(self, engine, **kwargs):
        self.engine = engine
        self.device = engine.device
        self.component_conf = kwargs

    @abstractmethod
    def __call__(self, latents: torch.Tensor, **kwargs):
        pass


postprocessor_registry = ClassRegister()
