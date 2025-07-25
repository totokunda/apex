from src.engine.base_engine import BaseEngine
import torch
from typing import Dict, Any, Callable, List, Union, Optional
from enum import Enum
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
from PIL import Image
import numpy as np
from src.engine.denoise import HunyuanDenoise, HunyuanDenoiseType
import torch.nn.functional as F
from src.utils.pos_emb_utils import get_nd_rotary_pos_embed_new


class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    FRAMEPACK = "framepack"  # framepack
    HYAVATAR = "hyavatar"  # hyavatar


class HunyuanEngine(BaseEngine, HunyuanDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: HunyuanDenoiseType = HunyuanDenoiseType.T2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        self.denoise_type = denoise_type

        self.vae_scale_factor_temporal = (
            getattr(self.vae, "temporal_compression_ratio", None) or 4
            if getattr(self, "vae", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            getattr(self.vae, "spatial_compression_ratio", None) or 8
            if getattr(self, "vae", None)
            else 8
        )
        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "latent_channels", 16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self.llama_text_encoder = None

    @torch.inference_mode()
    def run(
        self,
        input_nodes: List[UINode] = None,
        **kwargs,
    ):
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}
        if self.model_type == ModelType.T2V:
            return self.t2v_run(**final_kwargs)
        elif self.model_type == ModelType.I2V:
            return self.i2v_run(**final_kwargs)
        elif self.model_type == ModelType.FRAMEPACK:
            return self.framepack_run(**final_kwargs)
        elif self.model_type == ModelType.HYAVATAR:
            return self.hyavatar_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    ### REQUIRES FIXING FOR CORRECTNESS!!!!

    def _soft_append(
        self, history: torch.Tensor, current: torch.Tensor, overlap: int = 0
    ):
        """Soft append with blending for framepack generation"""
        if overlap <= 0:
            return torch.cat([history, current], dim=2)

        assert (
            history.shape[2] >= overlap
        ), f"Current length ({history.shape[2]}) must be >= overlap ({overlap})"
        assert (
            current.shape[2] >= overlap
        ), f"History length ({current.shape[2]}) must be >= overlap ({overlap})"

        weights = torch.linspace(
            1, 0, overlap, dtype=history.dtype, device=history.device
        ).view(1, 1, -1, 1, 1)
        blended = (
            weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
        )
        output = torch.cat(
            [history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2
        )

        return output.to(history)

    def __str__(self):
        return f"HunyuanEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    # Example usage for text-to-video
    engine = HunyuanEngine(
        yaml_path="manifest/hunyuan_t2v.yml",  # You'll need to create this
        model_type=ModelType.T2V,
        save_path="./apex-models",
        components_to_load=["transformer", "text_encoder", "vae", "scheduler"],
        component_dtypes={"vae": torch.float16},
    )

    prompt = "A cat walks on the grass, realistic"
    video = engine.run(
        height=320,
        width=512,
        num_frames=61,
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=6.0,
        seed=42,
    )

    export_to_video(video[0], "hunyuan_t2v_output.mp4", fps=15)
