import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import CogVideoBaseEngine
import torch.nn.functional as F
from einops import rearrange

class CogVideoFunEngine(CogVideoBaseEngine):
    """CogVideo Fun Engine Implementation"""

    def run(
        self,
        prompt: Union[List[str], str],
        control_video: Union[List[Image.Image], torch.Tensor],
        mask_video: Union[List[Image.Image], torch.Tensor] = None,
        negative_prompt: Union[List[str], str] = "",
        height: int | None = 480,
        width: int | None = 832,
        num_inference_steps: int = 50,
        duration: int = 49,
        fps: int = 8,
        num_videos: int = 1,
        seed: int = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        max_sequence_length: int = 226,
        eta: float = 0.0,
        noise_aug_strength: float = 0.0563,
        **kwargs,
    ):
        """Control video generation following CogVideoXFunControlPipeline"""

            # 1. Process control video
        control_video = self._load_video(control_video)
        
        if height is None or width is None:
            height, width = control_video[0].height, control_video[0].width
        
        control_video = self.video_processor.preprocess_video(
            control_video, height=height, width=width
        )
        control_video = control_video.to(device=self.device)
        video_length = control_video.shape[2]
        
        if mask_video is not None:
            mask_video = self._load_video(mask_video)
            mask_video = self.video_processor.preprocess_video(
                mask_video, height=height, width=width
            )
        
        num_frames = self._parse_num_frames(duration=duration, fps=fps)
        
        local_latent_length = (video_length - 1) // self.vae_scale_factor_temporal + 1
        # For CogVideoX 1.5, the latent frames should be clipped to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        
        if patch_size_t is not None and local_latent_length % patch_size_t != 0:
            additional_frames = local_latent_length % patch_size_t
            num_frames -= additional_frames * self.vae_scale_factor_temporal
        if num_frames <= 0:
            num_frames = 1
        if video_length > num_frames:
            self.logger.warning("The length of condition video is not right, the latent frames should be clipped to make it divisible by patch_size_t. ")
            video_length = num_frames
            control_video = control_video[:, :, :video_length]
            if mask_video is not None:
                mask_video = mask_video[:, :, :video_length]

        # 2. Encode prompts
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
            **text_encoder_kwargs,
        )

        if offload:
            self._offload(self.text_encoder)
            
        
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.device)
            
        # 8. Prepare guidance
        do_classifier_free_guidance = (
            guidance_scale > 1.0 and negative_prompt_embeds is not None
        )
        
        batch_size = num_videos

        # 4. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
        )

        # 6. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, check that latent frames are divisible by patch_size_t
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            raise ValueError(
                f"The number of latent frames must be divisible by `{patch_size_t=}` but the given video "
                f"contains {latent_frames=}, which is not divisible."
            )

        if not self.vae:
            self.load_component_by_type("vae")
        
        # Prepare control video latents using vae_encode
        if control_video is not None:
            control_video = control_video.to(device=self.device, dtype=prompt_embeds.dtype)
            control_video_latents = self._prepare_control_latents(
             None,
             control_video,  
             batch_size,
             height,
             width,
             prompt_embeds.dtype,
             self.device,
             generator,
             do_classifier_free_guidance=do_classifier_free_guidance
            )[1]
        else:
            control_video_latents = None

        
        num_channels_latents = self.vae.config.latent_channels
        
        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latents = self._get_latents(
            height,
            width,
            latent_num_frames,
            num_videos=num_videos,
            num_channels_latents=num_channels_latents,
            seed=seed,
            generator=generator,
            dtype=transformer_dtype,
            parse_frames=False,
            order="BFC",
        )

        # Scale initial noise by scheduler's init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma
        
        prompt_embeds = prompt_embeds.to(dtype=transformer_dtype)
        if negative_prompt is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=transformer_dtype)

        # 7. Prepare rotary embeddings
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height, width, latents.size(1), self.device
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            
        
        num_channels_transformer = self.load_config_by_type("transformer").get("in_channels", 32)
        
        if mask_video is not None:
            if (mask_video == 255).all():
                mask_latents = torch.zeros_like(latents)[:, :, :1].to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
            else:
                # Prepare mask latent variables
                mask_condition = mask_video
                if num_channels_transformer != num_channels_latents:
                    mask_condition_tile = torch.tile(mask_condition, [1, 3, 1, 1, 1])
                    masked_video = control_video * (mask_condition_tile < 0.5) + torch.ones_like(control_video) * (mask_condition_tile > 0.5) * -1

                    _, masked_video_latents = self._prepare_mask_latents(
                        None,
                        masked_video,
                        batch_size,
                        height,
                        width,
                        transformer_dtype,
                        self.device,
                        generator,
                        do_classifier_free_guidance,
                        noise_aug_strength=noise_aug_strength,
                    )
                    mask_latents = self._resize_mask(1 - mask_condition, masked_video_latents)
                    mask_latents = mask_latents.to(masked_video_latents.device) * self.vae.config.scaling_factor

                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    
                    mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                    )

                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    mask_input = rearrange(mask_input, "b c f h w -> b f c h w")
                    masked_video_latents_input = rearrange(masked_video_latents_input, "b c f h w -> b f c h w")

                    inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=2).to(latents.dtype)
                else:
                    mask = torch.tile(mask_condition, [1, num_channels_latents, 1, 1, 1])
                    mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                    mask = rearrange(mask, "b c f h w -> b f c h w")
                    
                    inpaint_latents = None
        else:
            if num_channels_transformer != num_channels_latents:
                mask = torch.zeros_like(latents).to(latents.device, latents.dtype)
                masked_video_latents = torch.zeros_like(latents).to(latents.device, latents.dtype)

                mask_input = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
                masked_video_latents_input = (
                    torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                )
                inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1).to(latents.dtype)
            else:
                mask = torch.zeros_like(control_video[:, :1])
                mask = torch.tile(mask, [1, num_channels_latents, 1, 1, 1])
                mask = F.interpolate(mask, size=latents.size()[-3:], mode='trilinear', align_corners=True).to(latents.device, latents.dtype)
                mask = rearrange(mask, "b c f h w -> b f c h w")
                
                inpaint_latents = None
                
        if offload:
            self._offload(self.vae)
        
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)
        
        noise_pred_kwargs = dict(
            encoder_hidden_states=prompt_embeds,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
        )

        # 9. Denoising loop
        latents = self.denoise(
            latents=latents,
            control_latents=control_video_latents,
            inpaint_latents=inpaint_latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_pred_kwargs=noise_pred_kwargs,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_inference_steps=num_inference_steps,
            transformer_dtype=transformer_dtype,
            extra_step_kwargs=self.prepare_extra_step_kwargs(generator, eta),
            **kwargs,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video
