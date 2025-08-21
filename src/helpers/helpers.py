from src.register import ClassRegister
from all.clip import CLIP
from hunyuan.avatar import HunyuanAvatar
from hunyuan.llama import HunyuanLlama
from stepvideo.text_encoder import StepVideoTextEncoder
from wan.ati import WanATI
from wan.fun_camera import WanFunCamera
from wan.multitalk import WanMultiTalk
from wan.recam import WanRecam

helpers = ClassRegister()

__all__ = [
    "CLIP",
    "HunyuanAvatar",
    "HunyuanLlama",
    "StepVideoTextEncoder",
    "WanATI",
    "WanFunCamera",
    "WanMultiTalk",
    "WanRecam",
]
