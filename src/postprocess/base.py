from abc import ABC, abstractmethod
import torch
from src.mixins import LoaderMixin
from src.register import ClassRegister


class BasePostprocessor(torch.nn.Module, LoaderMixin):
    def __init__(self, engine, **kwargs):
        super().__init__()
        self.engine = engine
        self.device = engine.device
        self.component_conf = kwargs

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("BasePostprocessor::__call__ method must be implemented by child classes")


postprocessor_registry = ClassRegister()
