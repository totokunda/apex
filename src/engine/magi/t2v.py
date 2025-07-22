import torch
from typing import Dict, Any, Callable, List, Union
import math
from .base import MagiBaseEngine


class MagiT2VEngine(MagiBaseEngine):
    """Magi Text-to-Video Engine Implementation"""

    def run(
        self,
        prompt: Union[List[str], str],
        negative_prompt: Union[List[str], str] = None,
        height: int = 512,
        width: int = 512,
        duration: str | int = 5,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
        fps: int = 24,
        guidance_scale: float = 6.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        chunk_size: int = 16,
        timestep_transform: str = "sd3",
        timestep_shift: float = 3.0,
        special_token_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """Text-to-video generation using MAGI's chunk-based approach"""

        # 1. Encode prompts
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        # MAGI uses a different text encoder (T5-based)
        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )

        prompt_attention_mask = None  # MAGI handles masking differently

        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        if negative_prompt is not None and use_cfg_guidance:
            negative_prompt_embeds = self.text_encoder.encode(
                negative_prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )

        if offload:
            self._offload(self.text_encoder)

        # 2. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        # 3. Prepare latents
        num_frames = self._parse_num_frames(duration, fps)

        # MAGI uses latent frames corresponding to chunks
        latent_num_frames = math.ceil(num_frames / self.vae_scale_factor_temporal)

        latents = self._get_latents(
            height=height,
            width=width,
            duration=latent_num_frames,
            fps=fps,
            num_videos=num_videos,
            num_channels_latents=self.num_channels_latents,
            seed=seed,
            generator=generator,
            dtype=torch.float32,
            parse_frames=False,  # Already calculated
        )

        # 4. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Initialize timestep schedule
        timesteps = self.init_timestep_schedule(
            num_steps=num_inference_steps,
            device=self.device,
            transform_type=timestep_transform,
            shift=timestep_shift,
        )

        # 6. Process text embeddings for chunk-based generation
        # Calculate chunking parameters first
        total_latent_frames = latents.shape[2]
        infer_chunk_num = math.ceil(total_latent_frames / chunk_size)
        clean_chunk_num = 0  # No prefix video in T2V

        processed_caption_embs, processed_caption_masks = self.process_text_embeddings(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            infer_chunk_num=infer_chunk_num,
            clean_chunk_num=clean_chunk_num,
            special_token_kwargs=special_token_kwargs,
        )

        # 7. MAGI chunk-based denoising
        latents = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            timesteps=timesteps,
            processed_caption_embs=processed_caption_embs,
            processed_caption_masks=processed_caption_masks,
            guidance_scale=guidance_scale,
            use_cfg_guidance=use_cfg_guidance,
            num_inference_steps=num_inference_steps,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            attention_kwargs=attention_kwargs,
            transformer_dtype=transformer_dtype,
            chunk_size=chunk_size,
            temporal_downsample_factor=self.vae_scale_factor_temporal,
            num_frames=num_frames,
            **kwargs,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            # Decode latents to video
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video
