import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from src.utils.pos_emb_utils import get_rotary_pos_embed
import torch.nn.functional as F
from .base import HunyuanBaseEngine


class HunyuanAvatarEngine(HunyuanBaseEngine):
    """Hunyuan Avatar Engine Implementation"""

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
        duration: Union[str, int] = 129,
        fps: int = 25,
        num_inference_steps: int = 50,
        use_cfg_guidance: bool = True,
        guidance_scale: float = 3.5,
        dynamic_guidance_start: float = 3.5,
        dynamic_guidance_end: float = 6.5,
        num_videos: int = 1,
        seed: int = None,
        return_latents: bool = False,
        offload: bool = True,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        text_encoder_kwargs: Dict[str, Any] = {},
        image_embed_interleave: Optional[int] = None,
        image_condition_type: Optional[str] = None,
        max_sequence_length: int = 256,
        frame_per_batch: int = 33,
        shift_offset: int = 10,
        no_cache_steps: int = None,
        **kwargs,
    ):
        # 1. Load preprocessor and VAE
        if "hunyuan.avatar" not in self.preprocessors:
            self.load_preprocessor_by_type("hunyuan.avatar")
        hyavatar_preprocessor = self.preprocessors["hunyuan.avatar"]
        self.to_device(hyavatar_preprocessor)

        if not self.vae:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        num_frames = self._parse_num_frames(duration, fps)

        loaded_image = self._load_image(image)

        if self.transformer is not None:
            image_condition_type = getattr(
                self.transformer.config, "image_condition_type", "token_replace"
            )
        else:
            image_condition_type = (
                "token_replace"
                if image_condition_type is None
                else image_condition_type
            )

        image_embed_interleave = (
            image_embed_interleave
            if image_embed_interleave is not None
            else (
                2
                if image_condition_type == "latent_concat"
                else 4 if image_condition_type == "token_replace" else 1
            )
        )

        # Preprocess image
        image_tensor = self.video_processor.preprocess(loaded_image, height, width).to(
            self.device
        )

        # 2. Preprocess inputs
        preprocessed_inputs = hyavatar_preprocessor(
            image=image,
            audio=audio,
            height=height,
            width=width,
            fps=fps,
            num_frames=num_frames,
        )

        face_masks = preprocessed_inputs["face_masks"]
        motion_exp = preprocessed_inputs["motion_exp"]
        motion_pose = preprocessed_inputs["motion_pose"]
        uncond_audio_prompts = preprocessed_inputs["uncond_audio_prompts"]
        audio_prompts = preprocessed_inputs["audio_prompts"]
        fps = preprocessed_inputs["fps"]

        (
            pooled_prompt_embeds,
            prompt_embeds,
            prompt_attention_mask,
        ) = self._encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            image=image,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
            image_embed_interleave=image_embed_interleave,
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
                num_videos_per_prompt=num_videos,
                max_sequence_length=max_sequence_length,
                **text_encoder_kwargs,
            )

        if offload:
            self._offload(self.text_encoder)
            if self.llama_text_encoder is not None:
                self._offload(self.llama_text_encoder)

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            self.device, dtype=transformer_dtype
        )
        prompt_attention_mask = prompt_attention_mask.to(
            self.device, dtype=transformer_dtype
        )

        if negative_prompt is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                self.device, dtype=transformer_dtype
            )

        if image_condition_type == "latent_concat":
            num_channels_latents = (
                getattr(self.transformer.config, "in_channels", 16) - 1
            ) // 2
        elif image_condition_type == "token_replace":
            num_channels_latents = getattr(self.transformer.config, "in_channels", 16)

        # Encode image to latents
        image_tensor_unsqueezed = image_tensor.unsqueeze(2).repeat(
            1, 1, num_frames, 1, 1
        )

        image_latents = self.vae_encode(
            image_tensor_unsqueezed,
            offload=offload,
            sample_mode="mode",
            normalize_latents_dtype=torch.float32,
            dtype=torch.float32,
        )

        uncond_image_latents = self.vae_encode(
            torch.zeros_like(image_tensor_unsqueezed),
            offload=offload,
            sample_mode="mode",
            normalize_latents_dtype=torch.float32,
            dtype=torch.float32,
        )

        face_masks = (
            F.interpolate(
                face_masks.to(self.device).squeeze(2),
                size=(image_latents.shape[-2], image_latents.shape[-1]),
                mode="bilinear",
            )
            .unsqueeze(2)
            .to(transformer_dtype)
        )

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
            else audio_prompts.shape[1]
        )


        patch_size = [self.transformer.config.patch_size_t, self.transformer.config.patch_size, self.transformer.config.patch_size]
        hidden_size = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )
        num_heads = self.transformer.config.num_attention_heads
        rope_axes_dim = self.transformer.config.rope_axes_dim
        rope_theta = self.transformer.config.rope_theta

        freqs_cos, freqs_sin = get_rotary_pos_embed(
            num_frames,
            height,
            width,
            patch_size,
            hidden_size,
            num_heads,
            rope_axes_dim,
            concat_dict={'mode': 'timecat', 'bias': -1},
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            theta=rope_theta,
        )

        freqs_cis = (
            freqs_cos.to(self.device, dtype=transformer_dtype),
            freqs_sin.to(self.device, dtype=transformer_dtype),
        )

        # 8. Prepare inputs for denoising loop
        audio_prompts = audio_prompts.to(self.device, dtype=transformer_dtype)
        uncond_audio_prompts = uncond_audio_prompts.to(
            self.device, dtype=transformer_dtype
        )

        motion_exp = motion_exp.to(self.device, dtype=transformer_dtype)
        motion_pose = motion_pose.to(self.device, dtype=transformer_dtype)
        fps_tensor = fps.to(self.device, dtype=transformer_dtype)

        # 9. Prepare latents
        num_frames = audio_prompts.shape[1]
        infer_length = (audio_prompts.shape[1] // 128 + 1) * 32 + 1

        latents = self._get_latents(
            height=height,
            width=width,
            duration=infer_length,
            num_channels_latents=num_channels_latents,
            fps=fps,
            num_videos=num_videos,
            seed=seed,
            generator=generator,
            parse_frames=False,
        )

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        # 10. Denoising loop
        latents_all = latents.clone()
        infer_length = latents.shape[2]
        video_length = infer_length
        
        if not no_cache_steps:
            no_cache_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] + list(range(15, 42, 5)) + [41, 42, 43, 44, 45, 46, 47, 48, 49]

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
            ref_latents=image_latents,
            uncond_ref_latents=uncond_image_latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_cfg_guidance=use_cfg_guidance,
            dynamic_guidance_start=dynamic_guidance_start,
            dynamic_guidance_end=dynamic_guidance_end,
            hidden_size=hidden_size,
            motion_exp=motion_exp,
            motion_pose=motion_pose,
            fps_tensor=fps_tensor,
            freqs_cis=freqs_cis,
            num_videos=num_videos,
            transformer_dtype=transformer_dtype,
            frame_per_batch=frame_per_batch,
            shift_offset=shift_offset,
            no_cache_steps=no_cache_steps,
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
