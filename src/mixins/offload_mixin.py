import torch
import gc


class OffloadMixin:
    """
    Add to any class that owns a torch.nn.Module (e.g. your Trainer or Model
    wrapper).  Call `self._offload(self.model)` when you are finished with a
    module and want to give the accelerator memory back.

    Example
    -------
    class MyRunner(OffloadMixin):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def teardown(self):
            self._offload(self.model)   # <- frees VRAM / MRAM
    """

    @staticmethod
    @torch.no_grad()
    def _offload(module: torch.nn.Module | None, *, recurse: bool = True) -> None:
        """
        Move every weight/buffer to CPU **and** clear CUDA/MPS/CPU caches.

        Parameters
        ----------
        module   : torch.nn.Module
            The module whose tensors you want to off-load.
        recurse  : bool, default True
            Whether to descend into sub-modules (almost always what you want).
        """

        if not module:
            return

        # 1)  Off-load PARAMETERS
        for name, param in module.named_parameters(recurse=recurse):
            if param.device.type != "cpu":
                param.data = param.data.cpu()
                if param.grad is not None:
                    param.grad.data = param.grad.data.cpu()

        # 2)  Off-load BUFFERS (e.g. running stats from BatchNorm)
        for name, buf in module.named_buffers(recurse=recurse):
            if buf.device.type != "cpu":
                # Need to replace the reference inside the owning module
                parent, _, key = name.rpartition(".")
                owner = module.get_submodule(parent) if parent else module
                owner._buffers[key] = buf.cpu()

        # 3)  Reclaim Python-level / CPU RAM
        gc.collect()

        # 4)  Reclaim CUDA VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optional: reset statistics so future profiling starts fresh
            for dev_id in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(dev_id)

        # 5)  Reclaim Apple-silicon MPS memory
        if (
            getattr(torch, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()
