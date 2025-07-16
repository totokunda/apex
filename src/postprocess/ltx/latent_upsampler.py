import torch
from diffusers.pipelines.ltx.modeling_latent_upsampler import LTXLatentUpsamplerModel
from diffusers.pipelines.ltx.pipeline_ltx_latent_upsample import (
    LTXLatentUpsamplePipeline,
)
from src.postprocess.base.base import Postprocessor
from diffusers.models.autoencoders import AutoencoderKL


class LatentUpsamplerPostprocessor(Postprocessor):
    def __init__(self, engine, **kwargs):
        super().__init__(engine, **kwargs)
        self.latent_upsampler = self._load_latent_upsampler()

    def _load_latent_upsampler(self):
        # This is a simplified loading mechanism.
        # In a real scenario, you might want to use a more robust loading from the engine.
        model_path = self.component_conf.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for latent_upscaler")

        latent_upsampler = LTXLatentUpsamplerModel.from_pretrained(model_path)
        latent_upsampler = latent_upsampler.to(self.device)
        return latent_upsampler

    @torch.no_grad()
    def __call__(self, latents: torch.Tensor, vae: AutoencoderKL, **kwargs):
        pipeline = LTXLatentUpsamplePipeline(
            vae=vae,
            latent_upsampler=self.latent_upsampler,
        )

        upsampled_latents = pipeline(
            latents=latents, output_type="latent", **kwargs
        ).frames

        return upsampled_latents
