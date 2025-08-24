import torch
from typing import Dict, Any, Callable, List
from PIL import Image
from .base import QwenImageBaseEngine


class QwenImageEditEngine(QwenImageBaseEngine):
    """QwenImage Edit Engine Implementation"""

    def run(
        self,
        image: Image.Image | List[Image.Image],
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 550,
        width: int = 992,
        num_inference_steps: int = 30,
        num_images: int = 1,
        seed: int | None = None,
        guidance_scale: float = 9.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        **kwargs,
    ):
        pass
