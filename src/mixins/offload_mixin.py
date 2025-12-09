import torch
import gc
import mlx.core as mx
from mlx.nn import Module as MlxModule


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
    def _offload(
        module: torch.nn.Module | None,
        *,
        recurse: bool = True,
        delete_from_cpu: bool = True,
    ) -> None:
        """
        Move every weight/buffer to CPU **and** clear CUDA/MPS/CPU caches.
        Optionally (default) also delete the module's parameters and buffers so it no
        longer occupies CPU or accelerator memory.

        Parameters
        ----------
        module   : torch.nn.Module
            The module whose tensors you want to off-load.
        recurse  : bool, default True
            Whether to descend into sub-modules (almost always what you want).
        delete_from_cpu : bool, default True
            If True, remove parameters and buffers from the module after off-loading,
            allowing them to be garbage-collected so the module holds no tensors.
        """

        if isinstance(module, MlxModule):
            return OffloadMixin._mlx_offload(module, delete_from_cpu=delete_from_cpu)

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

        # 2b)  Optionally drop PARAMETERS and BUFFERS from the module entirely so it
        #      holds no tensor memory (CPU or accelerator).  This does *not* clear
        #      external references (e.g. optimizers), but removes everything owned by
        #      the module itself.
        if delete_from_cpu:
            # Drop parameters
            for name, _ in list(module.named_parameters(recurse=recurse)):
                parent, _, key = name.rpartition(".")
                owner = module.get_submodule(parent) if parent else module
                if hasattr(owner, "_parameters") and key in owner._parameters:
                    owner._parameters[key] = None

            # Drop buffers
            for name, _ in list(module.named_buffers(recurse=recurse)):
                parent, _, key = name.rpartition(".")
                owner = module.get_submodule(parent) if parent else module
                if hasattr(owner, "_buffers") and key in owner._buffers:
                    owner._buffers.pop(key, None)

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

    @staticmethod
    def _mlx_offload(
        module: MlxModule | None,
        *,
        delete_from_cpu: bool = True,
    ) -> None:
        if not module:
            return
        if delete_from_cpu:
            del module
            gc.collect()
        mx.clear_cache()
