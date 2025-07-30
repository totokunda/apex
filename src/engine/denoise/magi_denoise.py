import torch
import math
import numpy as np
from src.utils.type_utils import EnumType
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from tqdm import tqdm
import gc


class MagiDenoiseType(EnumType):
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"


class MagiDenoise:
    def __init__(
        self, denoise_type: MagiDenoiseType = MagiDenoiseType.T2V, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        """Unified denoising method that handles chunk-based generation"""
        if self.denoise_type == MagiDenoiseType.T2V:
            return self.t2v_denoise(*args, **kwargs)
        elif self.denoise_type == MagiDenoiseType.I2V:
            return self.i2v_denoise(*args, **kwargs)
        elif self.denoise_type == MagiDenoiseType.V2V:
            return self.v2v_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Denoise type {self.denoise_type} not supported")

    def t2v_denoise(self, *args, **kwargs) -> torch.Tensor:
        """Text-to-video chunk-based denoising following MAGI's approach"""
        return self._chunk_based_denoise(*args, **kwargs)

    def i2v_denoise(self, *args, **kwargs) -> torch.Tensor:
        """Image-to-video chunk-based denoising following MAGI's approach"""
        return self._chunk_based_denoise(*args, **kwargs)

    def v2v_denoise(self, *args, **kwargs) -> torch.Tensor:
        """Video-to-video chunk-based denoising following MAGI's approach"""
        return self._chunk_based_denoise(*args, **kwargs)

    def _chunk_based_denoise(self, *args, **kwargs) -> torch.Tensor:
        """
        Core chunk-based denoising implementation adapted from MAGI's transport system.
        Instead of using generators, we iterate through chunks like framepack.
        Receives preprocessed inputs from the engine.
        """
        # Extract preprocessed parameters from engine
        latents = kwargs.get("latents", None)
        scheduler = kwargs.get("scheduler", None)
        timesteps = kwargs.get("timesteps", None)  # Preprocessed by engine
        processed_caption_embs = kwargs.get(
            "processed_caption_embs", None
        )  # Preprocessed by engine
        processed_caption_masks = kwargs.get(
            "processed_caption_masks", None
        )  # Preprocessed by engine
        prefix_video = kwargs.get("prefix_video", None)
        guidance_scale = kwargs.get("guidance_scale", 6.0)
        use_cfg_guidance = kwargs.get("use_cfg_guidance", True)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        render_on_step = kwargs.get("render_on_step", False)
        render_on_step_callback = kwargs.get("render_on_step_callback", None)
        attention_kwargs = kwargs.get("attention_kwargs", {})
        transformer_dtype = kwargs.get("transformer_dtype", torch.float16)
        chunk_size = kwargs.get("chunk_size", 16)  # Number of latent frames per chunk
        temporal_downsample_factor = kwargs.get("temporal_downsample_factor", 4)
        num_frames = kwargs.get("num_frames", None)

        device = latents.device
        batch_size = latents.shape[0]

        # Calculate chunking parameters
        if num_frames is None:
            # Calculate from latents if not provided
            num_frames = latents.shape[2] * temporal_downsample_factor

        # Determine clean chunk number based on prefix video
        clean_chunk_num = 0
        if prefix_video is not None:
            clean_chunk_num = prefix_video.shape[2] // chunk_size

        # Calculate total number of chunks needed
        total_latent_frames = latents.shape[2]
        if prefix_video is not None:
            total_latent_frames += prefix_video.shape[2]

        infer_chunk_num = math.ceil(total_latent_frames / chunk_size)

        # Set up the scheduler timesteps
        scheduler.set_timesteps(num_inference_steps, device=device)
        # Use the preprocessed timesteps from engine
        scheduler.timesteps = timesteps

        # Initialize with noise or prefix video
        if prefix_video is not None:
            # Concatenate prefix video with latents
            latents = torch.cat([prefix_video, latents], dim=2)
            self.logger.info(
                f"MAGI denoising with prefix video: prefix frames={prefix_video.shape[2]}, total chunks={infer_chunk_num}"
            )
        else:
            self.logger.info(
                f"MAGI {self.denoise_type.value} denoising: total frames={total_latent_frames}, chunks={infer_chunk_num}"
            )

        # Framepack-style chunk iteration
        generated_chunks = []

        with self._progress_bar(
            total=infer_chunk_num,
            desc=f"MAGI {self.denoise_type.value.upper()} Generation",
        ) as pbar:
            for chunk_idx in range(infer_chunk_num):
                # Calculate chunk boundaries
                start_frame = chunk_idx * chunk_size
                end_frame = min(start_frame + chunk_size, total_latent_frames)
                actual_chunk_size = end_frame - start_frame

                # Extract chunk latents
                chunk_latents = latents[:, :, start_frame:end_frame, :, :]

                # Extract corresponding text embeddings for this chunk
                # The processed embeddings are already in the right format [batch, chunk, seq_len, hidden_dim]
                chunk_caption_embs = processed_caption_embs[
                    :, chunk_idx : chunk_idx + 1, :, :
                ]  # Keep chunk dim
                chunk_masks = (
                    processed_caption_masks[:, chunk_idx : chunk_idx + 1, :]
                    if processed_caption_masks is not None
                    else None
                )

                # Skip denoising for clean chunks (from prefix video)
                if chunk_idx < clean_chunk_num:
                    generated_chunks.append(chunk_latents.to(torch.float32))
                    pbar.update(1)
                    continue

                # Denoise this chunk
                denoised_chunk = self._denoise_single_chunk(
                    chunk_latents=chunk_latents,
                    chunk_caption_embs=chunk_caption_embs,
                    chunk_masks=chunk_masks,
                    timesteps=timesteps,
                    scheduler=scheduler,
                    guidance_scale=guidance_scale,
                    use_cfg_guidance=use_cfg_guidance,
                    transformer_dtype=transformer_dtype,
                    attention_kwargs=attention_kwargs,
                    render_on_step=render_on_step,
                    render_on_step_callback=render_on_step_callback,
                )

                generated_chunks.append(denoised_chunk)

                # Memory cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

                pbar.update(1)

        # Concatenate all chunks into final video
        final_latents = torch.cat(generated_chunks, dim=2)

        # Remove prefix video frames if they were added
        if prefix_video is not None:
            final_latents = final_latents[:, :, prefix_video.shape[2] :]

        self.logger.info(f"MAGI {self.denoise_type.value} denoising completed.")
        return final_latents

    def _denoise_single_chunk(
        self,
        chunk_latents: torch.Tensor,
        chunk_caption_embs: torch.Tensor,
        chunk_masks: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler,
        guidance_scale: float,
        use_cfg_guidance: bool,
        transformer_dtype: torch.dtype,
        attention_kwargs: Dict,
        render_on_step: bool,
        render_on_step_callback: Optional[callable],
    ) -> torch.Tensor:
        """Denoise a single chunk following the standard diffusion process"""

        num_inference_steps = len(timesteps) - 1
        latents = chunk_latents.to(transformer_dtype)

        # Prepare embeddings for this chunk - handle MAGI's specific format
        if chunk_caption_embs.dim() == 4:
            # Format: [batch, chunk, seq_len, hidden_dim]
            caption_embs_flat = chunk_caption_embs.squeeze(1)  # Remove chunk dimension
        else:
            caption_embs_flat = chunk_caption_embs

        mask_flat = (
            chunk_masks.squeeze(1)
            if chunk_masks is not None and chunk_masks.dim() > 2
            else chunk_masks
        )

        with self._progress_bar(
            total=num_inference_steps, desc="Denoising Chunk", leave=False
        ) as pbar:
            for i, t in enumerate(timesteps[:-1]):
                # Expand latents for CFG if needed
                if use_cfg_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                    # Also duplicate text embeddings for CFG
                    if caption_embs_flat.dim() == 3:
                        encoder_hidden_states = torch.cat([caption_embs_flat] * 2)
                    else:
                        encoder_hidden_states = caption_embs_flat

                    encoder_attention_mask = (
                        torch.cat([mask_flat] * 2) if mask_flat is not None else None
                    )
                else:
                    latent_model_input = latents
                    encoder_hidden_states = caption_embs_flat
                    encoder_attention_mask = mask_flat

                # Scale model input (standard diffusion step)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # Expand timestep
                timestep = t.expand(latent_model_input.shape[0]).to(
                    latent_model_input.dtype
                )

                # Predict noise using the MAGI transformer
                # The corrected model now handles the diffusers interface properly
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    encoder_attention_mask=encoder_attention_mask,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # Apply CFG
                if use_cfg_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Scheduler step
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if render_on_step and render_on_step_callback:
                    self._render_step(latents, render_on_step_callback)

                pbar.update(1)

        return latents.to(torch.float32)

    def _calculate_time_interval(
        self, num_steps: int, device: torch.device
    ) -> torch.Tensor:
        """Calculate time intervals for integration (simplified version)"""
        return torch.ones(num_steps, device=device) / num_steps
