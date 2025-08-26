import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
from src.utils.models.hunyuan import get_rotary_pos_embed
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
        image_size: int = 704,
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
        guidance_rescale: float = 0.0,
        use_cache: bool = True,
        **kwargs,
    ):

        hyavatar = self.helpers["hunyuan.avatar"]

        if not self.vae:
            self.load_component_by_type("vae")
        self.to_device(self.vae)

        num_frames = self._parse_num_frames(duration, fps)

        loaded_image = self._load_image(image)
        w, h = loaded_image.size
        scale = image_size / min(w, h)
        width = round(w * scale / 64) * 64
        height = round(h * scale / 64) * 64

        if image_size == 704:
            img_size_long = 1216
        elif image_size == 512:
            img_size_long = 768
        elif image_size == 384:
            img_size_long = 576
        elif image_size == 256:
            img_size_long = 384
        else:
            img_size_long = image_size * 1.5  # Default fallback

        if height * width > image_size * img_size_long:
            import math

            scale = math.sqrt(image_size * img_size_long / w / h)
            width = round(w * scale / 64) * 64
            height = round(h * scale / 64) * 64

        loaded_image = loaded_image.resize((width, height), Image.Resampling.LANCZOS)

        # Preprocess image
        image_tensor = self.video_processor.preprocess(loaded_image, height, width).to(
            self.device
        )

        # 2. Preprocess inputs
        preprocessed_inputs = hyavatar(
            image=loaded_image,
            audio=audio,
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
            image=loaded_image,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
            image_embed_interleave=image_embed_interleave,
            hyavatar=True,
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
                image=loaded_image,
                num_videos_per_prompt=num_videos,
                max_sequence_length=max_sequence_length,
                hyavatar=True,
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
        prompt_attention_mask = prompt_attention_mask.to(self.device)

        if negative_prompt is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                self.device
            )

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

        face_masks = F.interpolate(
            face_masks.to(self.device).squeeze(2),
            size=(image_latents.shape[-2], image_latents.shape[-1]),
            mode="bilinear",
        ).unsqueeze(2)

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

        patch_size = [
            self.transformer.config.patch_size_t,
            self.transformer.config.patch_size,
            self.transformer.config.patch_size,
        ]

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
            concat_dict={"mode": "timecat", "bias": -1},
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            theta=rope_theta,
        )

        freqs_cis = (
            freqs_cos.to(self.device),
            freqs_sin.to(self.device),
        )

        video_length = audio_prompts.shape[1] // 4 * 4 + 1
        infer_length = (audio_prompts.shape[1] // 128 + 1) * 32 + 1
        video_length = (video_length - 1) // 4 + 1

        # 8. Prepare inputs for denoising loop
        audio_prompts = audio_prompts.to(self.device)
        pad_audio_length = (
            (audio_prompts.shape[1] // 128 + 1) * 128 + 4 - audio_prompts.shape[1]
        )
        audio_prompts_all = torch.cat(
            [audio_prompts, torch.zeros_like(audio_prompts[:, :pad_audio_length])],
            dim=1,
        )
        uncond_audio_prompts = uncond_audio_prompts.to(self.device)

        motion_exp = motion_exp.to(self.device)
        motion_pose = motion_pose.to(self.device)
        fps_tensor = fps.to(self.device)
        # 9. Prepare latents

        batch_size = prompt_embeds.shape[0]

        latents = self._get_latents(
            height=height,
            width=width,
            duration=infer_length,
            num_channels_latents=num_channels_latents,
            fps=fps,
            batch_size=batch_size,
            seed=seed,
            generator=generator,
            parse_frames=False,
            dtype=torch.float32,
        )

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        if use_cache and no_cache_steps is None:
            no_cache_steps = (
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                + list(range(15, 42, 5))
                + [41, 42, 43, 44, 45, 46, 47, 48, 49]
            )
        elif not use_cache:
            no_cache_steps = list(range(len(timesteps)))

        latents = self.denoise(
            infer_length=infer_length,
            latents_all=latents,
            audio_prompts_all=audio_prompts_all,
            uncond_audio_prompts=uncond_audio_prompts,
            face_masks=face_masks,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds=prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            prompt_attention_mask=prompt_attention_mask,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            ref_latents=image_latents,
            uncond_ref_latents=image_latents,
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
            guidance_rescale=guidance_rescale,
            video_length=video_length,
            **kwargs,
        )

        if offload:
            self._offload(self.transformer)

        latents = latents.float()[:, :, :video_length]

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._tensor_to_frames(video)
            return postprocessed_video
