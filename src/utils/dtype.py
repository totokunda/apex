import platform
import warnings
import torch


def select_ideal_dtypes(
    *,
    prefer_bfloat16: bool = True,
    quiet: bool = False,
) -> dict[str, torch.dtype]:
    """
    Heuristically choose the best‐supported low-precision ``torch.dtype`` for the three
    major components of a diffusion pipeline – the *transformer diffusion model*,
    *VAE*, and *text-encoder* – **without ever falling back to ``float32``**.

    The rules are:

    1. **CUDA / HIP (NVIDIA or AMD GPUs)**
       *  If the runtime reports native BF16 support **and** ``prefer_bfloat16`` is
          ``True`` → use **``torch.bfloat16``**.
       *  Otherwise use **``torch.float16``**.

    2. **Apple Silicon (MPS backend)**
       *  Use **``torch.float16``** (Apple GPUs expose fast ``float16``; BF16 is
          emulated and slower).

    3. **CPU-only**
       *  If the CPU exposes AVX-512 BF16 (Intel Sapphire Rapids, AMD Zen 4, etc.)
          → use **``torch.bfloat16``**.
       *  Otherwise fall back to **``torch.float16``** (even though the speed-up on
          CPU will be modest, we respect the “no float32” requirement).

    Parameters
    ----------
    prefer_bfloat16 : bool, default ``True``
        When both BF16 *and* FP16 are supported on the active device, pick BF16 if
        ``True`` (recommended on Ampere+/Hopper/MI300 GPUs and AVX-512 machines).
    quiet : bool, default ``False``
        Suppress informational warnings.

    Returns
    -------
    dict[str, torch.dtype]
        ``{"diffusion_model": dtype, "vae": dtype, "text_encoder": dtype}``
    """

    # --------------------------- utility helpers ----------------------------
    def _warn(msg: str):
        if not quiet:
            warnings.warn(msg, stacklevel=2)

    def _device_type() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _has_gpu_bf16() -> bool:
        """
        Unified check for NVIDIA (CUDA) and AMD (HIP/ROCm).
        """
        if not torch.cuda.is_available():
            return False
        # PyTorch ≥2.3 provides torch.cuda.is_bf16_supported()
        is_bf16_fn = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_fn):
            return bool(is_bf16_fn())
        # Fallback: infer from compute capability (NVIDIA only)
        major, minor = torch.cuda.get_device_capability()
        # Ampere (8.x) and newer support BF16
        return major >= 8

    def _has_cpu_bf16() -> bool:
        is_bf16_cpu = getattr(torch.backends.cpu, "is_bf16_supported", None)
        return bool(is_bf16_cpu() if callable(is_bf16_cpu) else False)

    # ----------------------------- main logic -------------------------------
    device = _device_type()

    if device == "cuda":  # includes AMD ROCm (reported as "cuda" by PyTorch)
        if prefer_bfloat16 and _has_gpu_bf16():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        gpu_name = torch.cuda.get_device_name(0)
        _warn(f"Using {dtype} on GPU: {gpu_name}")

    elif device == "mps":  # Apple Silicon
        dtype = torch.float16
        _warn("MPS backend detected (Apple Silicon) – using torch.float16")

    else:  # CPU only
        if prefer_bfloat16 and _has_cpu_bf16():
            dtype = torch.bfloat16
            _warn("CPU BF16 supported – using torch.bfloat16")
        else:
            dtype = torch.float16
            _warn(
                "CPU BF16 not detected – falling back to torch.float16. "
                "Performance may be limited."
            )

    return {
        "transformer": dtype,
        "vae": dtype,
        "text_encoder": dtype,
    }


def supports_double(device):
    device = torch.device(device)
    if device.type == "mps":
        # MPS backend has limited support for float64
        return False
    try:
        torch.zeros(1, dtype=torch.float64, device=device)
        return True
    except RuntimeError:
        return False
