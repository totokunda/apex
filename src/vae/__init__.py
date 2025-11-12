from src.vae.cogvideo.model import AutoencoderKLCogVideoX as CogVideoXVAE
from src.vae.wan.model import AutoencoderKLWan as WanVAE
from src.vae.hunyuan.model import AutoencoderKLHunyuanVideo as HunyuanVideoVAE
from src.vae.ltx.model import AutoencoderKLLTXVideo as LTXVideoVAE
from src.vae.magi.model import AutoencoderKLMagi as MagiVAE
from src.vae.mochi.model import AutoencoderKLMochi as MochiVAE
from src.vae.stepvideo.model import AutoencoderKL as StepVideoVAE
from src.vae.qwenimage.model import AutoencoderKLQwenImage as QwenImageVAE
from src.vae.auto.model import AutoencoderKL as AutoVAE
from src.vae.hunyuanimage.model import AutoencoderKLHunyuanImage as HunyuanImageVAE

__all__ = [
    "CogVideoXVAE",
    "WanVAE",
    "HunyuanVideoVAE",
    "LTXVideoVAE",
    "MagiVAE",
    "MochiVAE",
    "StepVideoVAE",
    "QwenImageVAE",
    "AutoVAE",
    "HunyuanImageVAE",
]

def get_vae(vae_name: str):
    if vae_name == "cogvideo":
        return CogVideoXVAE
    elif vae_name == "wan":
        return WanVAE
    elif vae_name == "hunyuan":
        return HunyuanVideoVAE
    elif vae_name == "ltx":
        return LTXVideoVAE
    elif vae_name == "magi":
        return MagiVAE
    elif vae_name == "mochi":
        return MochiVAE
    elif vae_name == "stepvideo":
        return StepVideoVAE
    elif vae_name == "qwenimage":
        return QwenImageVAE
    elif vae_name == "auto":
        return AutoVAE
    elif vae_name == "hunyuanimage":
        return HunyuanImageVAE
    else:
        raise ValueError(f"VAE {vae_name} not found")
