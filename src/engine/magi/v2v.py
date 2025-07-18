import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import math
from .base import MagiBaseEngine


class MagiV2VEngine(MagiBaseEngine):
    """Magi Video-to-Video Engine Implementation"""
    
    def run(
        self,
        video: Union[List[Image.Image], List[str], str, np.ndarray, torch.Tensor],
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
        prefix_frames: int = None,  # Number of prefix frames from input video
        **kwargs,
    ):
        """Video-to-video generation using MAGI's chunk-based approach"""

        # 1. Process input video
        loaded_video = self._load_video(video)
        loaded_video, height, width = self._aspect_ratio_resize_video(
            loaded_video, max_area=height * width
        )

        # Take prefix frames if specified
        if prefix_frames is not None and prefix_frames > 0:
            loaded_video = loaded_video[:prefix_frames]

        # Preprocess video for VAE
        video_tensor = self.video_processor.preprocess_video(
            loaded_video, height, width
        ).to(self.device)

        # 2. Encode prompts
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        
        self.to_device(self.text_encoder)
        
        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )
        
        prompt_attention_mask = None
        
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

        # 3. Encode video to latents (prefix video)
        prefix_video = self.vae_encode(
            video_tensor,
            offload=False,
            sample_mode="mode",  # Deterministic for prefix
            dtype=torch.float32,
        )

        # 4. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        # 5. Prepare latents for generation
        num_frames = self._parse_num_frames(duration, fps)
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
            parse_frames=False,
        )

        # 6. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)


        # 8. MAGI chunk-based denoising with prefix video
        latents = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prefix_video=prefix_video,  # Key difference for V2V
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            guidance_scale=guidance_scale,
            use_cfg_guidance=use_cfg_guidance,
            num_inference_steps=num_inference_steps,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            attention_kwargs=attention_kwargs,
            transformer_dtype=transformer_dtype,
            special_token_kwargs=special_token_kwargs,
            chunk_size=chunk_size,
            temporal_downsample_factor=self.vae_scale_factor_temporal,
            fps=fps,
            num_frames=num_frames,
            timestep_transform=timestep_transform,
            timestep_shift=timestep_shift,
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