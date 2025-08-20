from __future__ import annotations
from typing import Iterable, Optional

import mlx.core as mx
from loguru import logger
from tqdm import tqdm
from src.utils.type import EnumType
from src.utils.mlx import convert_dtype_to_mlx, torch_to_mlx, to_torch


class DenoiseType(EnumType):
    BASE = "mlx.base"
    MOE = "mlx.moe"
    DIFFUSION_FORCING = "mlx.diffusion_forcing"
    MULTITALK = "mlx.multitalk"


class WanDenoise:
    def __init__(self, denoise_type: DenoiseType = DenoiseType.BASE, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type
        # The following attributes are expected to be provided by the engine hosting this class:
        # - self.transformer (and optionally self.transformer_2)
        # - self.logger (fallback to loguru if missing)

    def _get_logger(self):
        return getattr(self, "logger", logger)

    def denoise(self, *args, **kwargs) -> mx.array:
        if self.denoise_type == DenoiseType.BASE:
            return self.mlx_base_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.MOE:
            return self.mlx_moe_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.DIFFUSION_FORCING:
            return self.mlx_diffusion_forcing_denoise(*args, **kwargs)
        elif self.denoise_type == DenoiseType.MULTITALK:
            return self.mlx_multitalk_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Denoise type {self.denoise_type} not supported")

    # Placeholder for parity with torch version; can be implemented as needed
    def mlx_diffusion_forcing_denoise(self, *args, **kwargs) -> mx.array:
        raise NotImplementedError("diffusion_forcing_denoise not implemented for MLX")

    def mlx_multitalk_denoise(self, *args, **kwargs) -> mx.array:
        raise NotImplementedError("multitalk_denoise not implemented for MLX")

    def mlx_recam_denoise(self, *args, **kwargs) -> mx.array:
        raise NotImplementedError("recam_denoise not implemented for MLX")

    def _maybe_to_dtype(self, array: mx.array, dtype):
        if dtype is None:
            return array
        if array.dtype == dtype:
            return array

        return array.astype(dtype)

    def _concat_if_needed(
        self, latents: mx.array, latent_condition: Optional[mx.array]
    ) -> mx.array:
        if latent_condition is None:
            return latents
        return mx.concatenate([latents, latent_condition], axis=1)

    def mlx_moe_denoise(self, *args, **kwargs) -> mx.array:

        kwargs = torch_to_mlx(kwargs)

        timesteps: Iterable[mx.array] = kwargs.get("timesteps", None)
        latents: mx.array = kwargs.get("latents", None)
        latent_condition: Optional[mx.array] = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance: bool = kwargs.get("use_cfg_guidance", True)
        render_on_step: bool = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        boundary_timestep = kwargs.get("boundary_timestep", None)

        transformer_dtype = convert_dtype_to_mlx(transformer_dtype)

        log = self._get_logger()

        for i, t in enumerate(tqdm(timesteps, desc="Sampling MOE (MLX)")):
            if latent_condition is not None:
                latent_model_input = self._concat_if_needed(latents, latent_condition)
                latent_model_input = self._maybe_to_dtype(
                    latent_model_input, transformer_dtype
                )
            else:
                latent_model_input = self._maybe_to_dtype(latents, transformer_dtype)

            timestep = mx.broadcast_to(t, (latents.shape[0],))

            if boundary_timestep is None or t >= boundary_timestep:
                if hasattr(self, "transformer_2") and self.transformer_2:
                    self._offload(self.transformer_2)
                    setattr(self, "transformer_2", None)
                if not self.transformer:
                    self.load_component_by_name("transformer")
                transformer = self.transformer
                if isinstance(guidance_scale, list):
                    guidance_scale = guidance_scale[1]
            else:
                if self.transformer:
                    self._offload(self.transformer)
                    setattr(self, "transformer", None)
                if not hasattr(self, "transformer_2") or not self.transformer_2:
                    self.load_component_by_name("transformer_2")
                transformer = self.transformer_2
                if isinstance(guidance_scale, list):
                    guidance_scale = guidance_scale[0]
                    # Standard denoising

            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                return_dict=False,
                **kwargs.get("transformer_kwargs", {}),
            )[0]

            mx.eval(noise_pred)

            if use_cfg_guidance and kwargs.get(
                "unconditional_transformer_kwargs", None
            ):
                uncond_noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("unconditional_transformer_kwargs", {}),
                )[0]

                noise_pred = uncond_noise_pred + guidance_scale * (
                    noise_pred - uncond_noise_pred
                )
            mx.eval(noise_pred)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            mx.eval(latents)

            if render_on_step and render_on_step_callback:
                try:
                    render_on_step_callback(latents)
                except Exception as e:
                    log.warning(f"Render-on-step callback failed: {e}")

        log.info("Denoising completed.")
        return to_torch(latents)

    def mlx_base_denoise(self, *args, **kwargs) -> mx.array:
        kwargs = torch_to_mlx(kwargs)

        timesteps: Iterable[mx.array] = kwargs.get("timesteps", None)
        latents: mx.array = kwargs.get("latents", None)
        latent_condition: Optional[mx.array] = kwargs.get("latent_condition", None)
        transformer_dtype = kwargs.get("transformer_dtype", None)
        use_cfg_guidance: bool = kwargs.get("use_cfg_guidance", True)
        render_on_step: bool = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        scheduler = kwargs.get("scheduler", None)
        guidance_scale = kwargs.get("guidance_scale", 5.0)
        expand_timesteps: bool = kwargs.get("expand_timesteps", False)
        first_frame_mask: Optional[mx.array] = kwargs.get("first_frame_mask", None)

        transformer_dtype = convert_dtype_to_mlx(transformer_dtype)

        if not hasattr(self, "transformer") or self.transformer is None:
            self.load_component_by_type("transformer")
        transformer = self.transformer

        log = self._get_logger()

        for i, t in enumerate(tqdm(timesteps, desc=f"Sampling WAN (MLX)")):
            if expand_timesteps and first_frame_mask is not None:
                mask = mx.ones_like(latents)
            else:
                mask = None

            if expand_timesteps:
                if latent_condition is not None and first_frame_mask is not None:
                    latent_model_input = (
                        1 - first_frame_mask
                    ) * latent_condition + first_frame_mask * latents
                    latent_model_input = self._maybe_to_dtype(
                        latent_model_input, transformer_dtype
                    )
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                else:
                    latent_model_input = self._maybe_to_dtype(
                        latents, transformer_dtype
                    )
                    # mask is ensured non-None above when expand_timesteps
                    temp_ts = (
                        (mask[0][0][:, ::2, ::2] * t).flatten()
                        if mask is not None
                        else t.flatten()
                    )
                timestep = mx.broadcast_to(
                    temp_ts, (latents.shape[0], temp_ts.shape[0])
                )
            else:
                timestep = mx.broadcast_to(t, (latents.shape[0],))
                if latent_condition is not None:
                    latent_model_input = self._concat_if_needed(
                        latents, latent_condition
                    )
                    latent_model_input = self._maybe_to_dtype(
                        latent_model_input, transformer_dtype
                    )
                else:
                    latent_model_input = self._maybe_to_dtype(
                        latents, transformer_dtype
                    )

            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                return_dict=False,
                **kwargs.get("transformer_kwargs", {}),
            )[0]

            mx.eval(noise_pred)

            if use_cfg_guidance and kwargs.get(
                "unconditional_transformer_kwargs", None
            ):
                uncond_noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    return_dict=False,
                    **kwargs.get("unconditional_transformer_kwargs", {}),
                )[0]
                noise_pred = uncond_noise_pred + guidance_scale * (
                    noise_pred - uncond_noise_pred
                )
            mx.eval(noise_pred)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            mx.eval(latents)

            if render_on_step and render_on_step_callback:
                try:
                    render_on_step_callback(latents)
                except Exception as e:
                    log.warning(f"Render-on-step callback failed: {e}")

        if (
            expand_timesteps
            and first_frame_mask is not None
            and latent_condition is not None
        ):
            latents = (
                1 - first_frame_mask
            ) * latent_condition + first_frame_mask * latents

        log.info("Denoising completed.")

        return to_torch(latents)
