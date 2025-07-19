import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import HunyuanBaseEngine


class HunyuanHyavatarEngine(HunyuanBaseEngine):
    """Hunyuan Hyavatar Engine Implementation"""
    
    def run(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        audio: Union[str, np.ndarray, torch.Tensor],
        prompt: Union[List[str], str],
        prompt_2: Union[List[str], str] = None,
        negative_prompt: Union[List[str], str] = None,
        negative_prompt_2: Union[List[str], str] = None,
        height: int = 1024,
        width: int = 1024,
        duration: Union[str, int] = None,
        fps: int = 25,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        num_videos: int = 1,
        seed: int = None,
        return_latents: bool = False,
        offload: bool = True,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        text_encoder_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        # 1. Load preprocessor and VAE
        if "hyavatar" not in self.preprocessors:
            self.load_preprocessor_by_type("hyavatar")
        hyavatar_preprocessor = self.preprocessors["hyavatar"]
        self.to_device(hyavatar_preprocessor)

        if not self.vae:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        # 2. Preprocess inputs
        batch = hyavatar_preprocessor(
            image=image,
            audio=audio,
            height=height,
            width=width,
            fps=fps,
            dtype=self.vae.dtype,
        )

        # 3. VAE encode reference images and prepare face masks
        ref_latents = self.vae_encode(
            batch["pixel_value_ref_for_vae"],
            offload=offload,
            sample_mode="sample",
            generator=generator,
        )
        uncond_ref_latents = self.vae_encode(
            batch["uncond_pixel_value_ref_for_vae"],
            offload=offload,
            sample_mode="sample",
            generator=generator,
        )
        face_masks = F.interpolate(
            batch["face_masks"].to(self.device),
            size=(ref_latents.shape[-2], ref_latents.shape[-1]),
            mode="bilinear",
        )

        # 4. Encode prompts
        (
            pooled_prompt_embeds,
            prompt_embeds,
            prompt_attention_mask,
        ) = self._encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            image=batch["pixel_value_llava"],
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )

        if negative_prompt is not None:
            (
                negative_pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                image=batch["uncond_pixel_value_llava"],
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )
        else:
            # Create empty negative prompts if not provided
            (
                negative_pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._encode_prompt(
                prompt=[""] * len(prompt),
                prompt_2=[""] * len(prompt) if prompt_2 else None,
                image=batch["uncond_pixel_value_llava"],
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )

        if offload:
            self._offload(self.text_encoder)
            if self.llama_text_encoder is not None:
                self._offload(self.llama_text_encoder)

        # 5. Load transformer and scheduler
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=self.scheduler,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
        )

        # 7. Get RoPE embeddings
        num_frames = (
            self._parse_num_frames(duration, fps)
            if duration
            else batch["audio_prompts"].shape[1]
        )
        target_ndim = 3
        if "884" in self.vae.config.name:
            latents_size = [(num_frames - 1) // 4 + 1, height // 8, width // 8]
        else:
            latents_size = [num_frames, height // 8, width // 8]

        patch_size = self.transformer.config.patch_size
        hidden_size = self.transformer.config.hidden_size
        num_heads = self.transformer.config.num_heads

        if isinstance(patch_size, int):
            rope_sizes = [s // patch_size for s in latents_size]
        else:
            rope_sizes = [s // patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes
        head_dim = hidden_size // num_heads
        rope_dim_list = self.transformer.config.get(
            "rope_dim_list", [head_dim // target_ndim for _ in range(target_ndim)]
        )

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed_new(
            rope_dim_list,
            rope_sizes,
            theta=self.transformer.config.rope_theta,
            use_real=True,
        )
        freqs_cis = (
            freqs_cos.to(self.device, dtype=transformer_dtype),
            freqs_sin.to(self.device, dtype=transformer_dtype),
        )

        # 8. Prepare inputs for denoising loop
        audio_prompts = batch["audio_prompts"].to(self.device, dtype=transformer_dtype)
        uncond_audio_prompts = batch["uncond_audio_prompts"].to(
            self.device, dtype=transformer_dtype
        )

        motion_exp = batch["motion_exp"].to(self.device, dtype=transformer_dtype)
        motion_pose = batch["motion_pose"].to(self.device, dtype=transformer_dtype)
        fps_tensor = batch["fps"].to(self.device, dtype=transformer_dtype)

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            self.device, dtype=transformer_dtype
        )
        negative_prompt_embeds = negative_prompt_embeds.to(
            self.device, dtype=transformer_dtype
        )
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
            self.device, dtype=transformer_dtype
        )
        prompt_attention_mask = prompt_attention_mask.to(self.device)
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(self.device)

        # 9. Prepare latents
        num_frames = audio_prompts.shape[1]

        if "884" in self.vae.config.name:
            num_latent_frames = (num_frames - 1) // 4 + 1
        else:
            num_latent_frames = num_frames

        latents = self._get_latents(
            height=height,
            width=width,
            duration=num_latent_frames,
            fps=fps,
            num_videos=num_videos,
            seed=seed,
            generator=generator,
            parse_frames=False,
        )

        # 10. Denoising loop
        latents_all = latents.clone()
        infer_length = latents.shape[2]
        video_length = infer_length

        latents_all = self.denoise(
            infer_length=infer_length,
            latents_all=latents_all,
            audio_prompts=audio_prompts,
            uncond_audio_prompts=uncond_audio_prompts,
            face_masks=face_masks,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds=prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            prompt_attention_mask=prompt_attention_mask,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            ref_latents=ref_latents,
            uncond_ref_latents=uncond_ref_latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            motion_exp=motion_exp,
            motion_pose=motion_pose,
            fps_tensor=fps_tensor,
            freqs_cis=freqs_cis,
            num_videos=num_videos,
            **kwargs,
        )

        if offload:
            self._offload(self.transformer)

        latents = latents_all.float()[:, :, :video_length]

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video