from src.engine.base_engine import BaseEngine
import torch
from typing import Dict, Any, Callable, List, Union, Optional
from enum import Enum
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
from PIL import Image
import numpy as np
from src.engine.denoise import CogVideoDenoise, CogVideoDenoiseType
import torch.nn.functional as F
from diffusers.models.embeddings import get_3d_rotary_pos_embed


class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    V2V = "v2v"  # video to video
    CONTROL = "control"  # control video


class CogVideoEngine(BaseEngine, CogVideoDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: CogVideoDenoiseType = CogVideoDenoiseType.T2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        self.denoise_type = denoise_type

        self.vae_scale_factor_temporal = (
            getattr(self.vae, "config", {}).get("temporal_compression_ratio", None) or 4
            if getattr(self, "vae", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(getattr(self.vae, "config", {}).get("block_out_channels", [1, 1, 1])) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.vae_scaling_factor_image = (
            getattr(self.vae, "config", {}).get("scaling_factor", None) or 0.7
            if getattr(self, "vae", None)
            else 0.7
        )
        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "latent_channels", 16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

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
        elif self.model_type == ModelType.V2V:
            return self.v2v_run(**final_kwargs)
        elif self.model_type == ModelType.CONTROL:
            return self.control_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    

    

    

    

    

    

    def __str__(self):
        return f"CogVideoEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    # Example usage for text-to-video
    engine = CogVideoEngine(
        yaml_path="manifest/cogvideox_t2v_5b.yml",  # You'll need to create this
        model_type=ModelType.T2V,
        save_path="./apex-models",
        components_to_load=["transformer", "text_encoder", "vae", "scheduler"],
        component_dtypes={"vae": torch.float16},
    )

    prompt = "A cat walks on the grass, realistic"
    video = engine.run(
        height=480,
        width=720,
        num_frames=49,
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=6.0,
        seed=42,
    )

    export_to_video(video[0], "cogvideox_t2v_output.mp4", fps=8)
