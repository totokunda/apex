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

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Encode prompts using T5 text encoder"""
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # Tokenize and encode prompt
        text_inputs = self.text_encoder.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        prompt_embeds = self.text_encoder.text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype or self.text_encoder.text_encoder.dtype, device=self.device)

        # Duplicate text embeddings for each generation per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        # Handle negative prompt
        negative_prompt_embeds = None
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            elif len(negative_prompt) != batch_size:
                negative_prompt = [negative_prompt[0]] * batch_size

            negative_text_inputs = self.text_encoder.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_text_input_ids = negative_text_inputs.input_ids.to(self.device)

            negative_prompt_embeds = self.text_encoder.text_encoder(negative_text_input_ids)[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype or self.text_encoder.text_encoder.dtype, device=self.device)

            # Duplicate negative text embeddings for each generation per prompt
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ):
        """Prepare rotary positional embeddings for CogVideoX"""
        if not getattr(self.transformer.config, "use_rotary_positional_embeddings", False):
            return None

        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size
        p_t = getattr(self.transformer.config, "patch_size_t", None)

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            from thirdparty.diffusers.src.diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin

    def t2v_run(
        self,
        prompt: Union[List[str], str],
        negative_prompt: Union[List[str], str] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
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
        sigmas: List[float] = None,
        **kwargs,
    ):
        """Text-to-video generation following CogVideoXPipeline"""

        # 1. Encode prompts
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
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
            negative_prompt_embeds = negative_prompt_embeds.to(self.device, dtype=transformer_dtype)

        # 3. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, pad latent frames to be divisible by patch_size_t
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        num_channels_latents = getattr(self.transformer.config, "in_channels", 16)
        latents = self._get_latents(
            height,
            width,
            num_frames,
            num_videos=num_videos,
            num_channels_latents=num_channels_latents,
            seed=seed,
            generator=generator,
            dtype=torch.float32,
        )

        # Scale initial noise by scheduler's init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma

        # 6. Prepare rotary embeddings
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height, width, latents.size(1), self.device
        )

        # 7. Prepare guidance
        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 8. Denoising loop
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            ),
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_inference_steps=num_inference_steps,
            additional_frames=additional_frames,
            **kwargs,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            # Discard any padding frames that were added for CogVideoX 1.5
            if additional_frames > 0:
                latents = latents[:, additional_frames:]
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video

    def i2v_run(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        prompt: Union[List[str], str],
        negative_prompt: Union[List[str], str] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
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
        sigmas: List[float] = None,
        **kwargs,
    ):
        """Image-to-video generation following CogVideoXImageToVideoPipeline"""

        # 1. Process input image
        loaded_image = self._load_image(image)
        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image, max_area=height * width
        )

        # Preprocess image
        image_tensor = self.video_processor.preprocess(loaded_image, height, width).to(
            self.device
        )

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

        # 3. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.device, dtype=transformer_dtype)

        # 4. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        # 6. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, pad latent frames to be divisible by patch_size_t
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        # Encode image to latents
        image_tensor = image_tensor.to(dtype=prompt_embeds.dtype, device=self.device)
        image_tensor_unsqueezed = image_tensor.unsqueeze(2)  # Add temporal dimension

        # Use VAE to encode image
        if isinstance(generator, list):
            image_latents = [
                self.vae_encode(
                    image_tensor_unsqueezed[i].unsqueeze(0), 
                    sample_mode="sample", 
                    sample_generator=generator[i],
                    dtype=prompt_embeds.dtype
                )
                for i in range(image_tensor_unsqueezed.shape[0])
            ]
        else:
            image_latents = [
                self.vae_encode(
                    img.unsqueeze(0), 
                    sample_mode="sample", 
                    sample_generator=generator,
                    dtype=prompt_embeds.dtype
                )
                for img in image_tensor_unsqueezed
            ]

        image_latents = torch.cat(image_latents, dim=0).permute(0, 2, 1, 3, 4)

        # Create padding for remaining frames
        padding_shape = (
            num_videos,
            latent_frames - 1,
            self.transformer.config.in_channels // 2,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        latent_padding = torch.zeros(padding_shape, device=self.device, dtype=prompt_embeds.dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Handle CogVideoX 1.5 padding
        if patch_size_t is not None:
            first_frame = image_latents[:, : image_latents.size(1) % patch_size_t, ...]
            image_latents = torch.cat([first_frame, image_latents], dim=1)

        # Prepare noise latents
        latent_channels = self.transformer.config.in_channels // 2
        latents = self._get_latents(
            height,
            width,
            num_frames,
            num_videos=num_videos,
            num_channels_latents=latent_channels,
            seed=seed,
            generator=generator,
            dtype=prompt_embeds.dtype,
        )

        # Scale initial noise by scheduler's init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma

        # 7. Prepare rotary embeddings
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height, width, latents.size(1), self.device
        )

        # 8. Prepare ofs embeddings (for CogVideoX 1.5)
        ofs_emb = None
        if getattr(self.transformer.config, "ofs_embed_dim", None) is not None:
            ofs_emb = latents.new_full((1,), fill_value=2.0)

        # 9. Prepare guidance
        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 10. Denoising loop
        latents = self.denoise(
            latents=latents,
            image_latents=image_latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                image_rotary_emb=image_rotary_emb,
                ofs=ofs_emb,
                attention_kwargs=attention_kwargs,
            ),
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_inference_steps=num_inference_steps,
            additional_frames=additional_frames,
            **kwargs,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            # Discard any padding frames that were added for CogVideoX 1.5
            if additional_frames > 0:
                latents = latents[:, additional_frames:]
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video

    def v2v_run(
        self,
        video: Union[List[Image.Image], torch.Tensor],
        prompt: Union[List[str], str],
        negative_prompt: Union[List[str], str] = None,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
        strength: float = 0.8,
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
        latents: torch.Tensor = None,
        **kwargs,
    ):
        """Video-to-video generation following CogVideoXVideoToVideoPipeline"""

        # 1. Process input video
        if latents is None:
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            video = video.to(device=self.device)

        num_frames = len(video) if latents is None else latents.size(1)

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

        # 3. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.device, dtype=transformer_dtype)

        # 4. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
        )

        # Get timesteps for video-to-video (strength-based)
        timesteps, num_inference_steps = self._get_v2v_timesteps(
            num_inference_steps, timesteps, strength
        )
        latent_timestep = timesteps[:1].repeat(num_videos)

        # 6. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, check that latent frames are divisible by patch_size_t
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            raise ValueError(
                f"The number of latent frames must be divisible by `{patch_size_t=}` but the given video "
                f"contains {latent_frames=}, which is not divisible."
            )

        # Prepare latents from input video
        if latents is None:
            video = video.to(dtype=prompt_embeds.dtype, device=self.device)
            latents = self._prepare_v2v_latents(
                video=video,
                batch_size=num_videos,
                num_channels_latents=getattr(self.transformer.config, "in_channels", 16),
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=self.device,
                generator=generator,
                timestep=latent_timestep,
            )

        # 7. Prepare rotary embeddings
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height, width, latents.size(1), self.device
        )

        # 8. Prepare guidance
        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 9. Denoising loop
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            ),
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_inference_steps=num_inference_steps,
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

    def control_run(
        self,
        prompt: Union[List[str], str],
        control_video: Union[List[Image.Image], torch.Tensor],
        negative_prompt: Union[List[str], str] = None,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 50,
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
        **kwargs,
    ):
        """Control video generation following CogVideoXFunControlPipeline"""

        # 1. Process control video
        if isinstance(control_video[0], Image.Image):
            control_video = [control_video]
        
        control_video = self.video_processor.preprocess_video(
            control_video, height=height, width=width
        )
        control_video = control_video.to(device=self.device)

        num_frames = len(control_video[0])

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

        # 3. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(self.device, dtype=transformer_dtype)

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

        # Prepare control video latents using vae_encode
        control_video = control_video.to(device=self.device, dtype=prompt_embeds.dtype)
        control_video_latents = self.vae_encode(
            control_video,
            sample_mode="mode",
            dtype=prompt_embeds.dtype
        )
        control_video_latents = control_video_latents.permute(0, 2, 1, 3, 4)

        # Prepare noise latents
        latent_channels = self.transformer.config.in_channels // 2
        latents = self._get_latents(
            height,
            width,
            num_frames,
            num_videos=num_videos,
            num_channels_latents=latent_channels,
            seed=seed,
            generator=generator,
            dtype=prompt_embeds.dtype,
        )

        # Scale initial noise by scheduler's init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma

        # 7. Prepare rotary embeddings
        image_rotary_emb = self._prepare_rotary_positional_embeddings(
            height, width, latents.size(1), self.device
        )

        # 8. Prepare guidance
        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 9. Denoising loop
        latents = self.denoise(
            latents=latents,
            control_video_latents=control_video_latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            ),
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            num_inference_steps=num_inference_steps,
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

    def _get_v2v_timesteps(self, num_inference_steps: int, timesteps: List[int], strength: float):
        """Get timesteps for video-to-video generation based on strength"""
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]
        
        return timesteps, num_inference_steps - t_start

    def _prepare_v2v_latents(
        self,
        video: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        timestep: Optional[torch.Tensor] = None,
    ):
        """Prepare latents for video-to-video generation"""
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (video.size(2) - 1) // self.vae_scale_factor_temporal + 1
        
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # Encode video to latents using vae_encode
        if isinstance(generator, list):
            init_latents = [
                self.vae_encode(
                    video[i].unsqueeze(0), 
                    sample_mode="sample", 
                    sample_generator=generator[i],
                    dtype=dtype
                )
                for i in range(batch_size)
            ]
        else:
            init_latents = [
                self.vae_encode(
                    vid.unsqueeze(0), 
                    sample_mode="sample", 
                    sample_generator=generator,
                    dtype=dtype
                )
                for vid in video
            ]

        init_latents = torch.cat(init_latents, dim=0).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        # Add noise to the initial latents
        from diffusers.utils.torch_utils import randn_tensor
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.add_noise(init_latents, noise, timestep)

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _retrieve_latents(self, encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
        """Retrieve latents from encoder output"""
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def _prepare_control_latents(
        self, mask: Optional[torch.Tensor] = None, masked_image: Optional[torch.Tensor] = None
    ):
        """Prepare control latents for control video generation"""
        if mask is not None:
            masks = []
            for i in range(mask.size(0)):
                current_mask = mask[i].unsqueeze(0)
                current_mask = self.vae.encode(current_mask)[0]
                current_mask = current_mask.mode()
                masks.append(current_mask)
            mask = torch.cat(masks, dim=0)
            mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            mask_pixel_values = []
            for i in range(masked_image.size(0)):
                mask_pixel_value = masked_image[i].unsqueeze(0)
                mask_pixel_value = self.vae.encode(mask_pixel_value)[0]
                mask_pixel_value = mask_pixel_value.mode()
                mask_pixel_values.append(mask_pixel_value)
            masked_image_latents = torch.cat(mask_pixel_values, dim=0)
            masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents

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
