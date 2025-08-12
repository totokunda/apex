import torch
import logging


class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """

    def __init__(
        self, *args, tensor_type, tensor_shape, dequant_dtype=None, patches=[], **kwargs
    ):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
        self.dequant_dtype = dequant_dtype

    def __new__(
        cls, *args, tensor_type, tensor_shape, dequant_dtype=None, patches=[], **kwargs
    ):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        new.dequant_dtype = getattr(self, "dequant_dtype", None)
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
            new_tensor,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
            dequant_dtype=getattr(self, "dequant_dtype", None),
            patches=getattr(self, "patches", []).copy(),
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape
