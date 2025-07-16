from src.engine.base_engine import BaseEngine
import torch
from typing import Dict, Any, Callable, List, Union, Optional
from enum import Enum
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
from PIL import Image
import numpy as np
from src.engine.denoise import HunyuanDenoise, HunyuanDenoiseType
import torch.nn.functional as F
from src.utils.pos_emb_utils import get_nd_rotary_pos_embed_new


class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    FRAMEPACK = "framepack"  # framepack
    HYAVATAR = "hyavatar"  # hyavatar


class HunyuanEngine(BaseEngine, HunyuanDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: HunyuanDenoiseType = HunyuanDenoiseType.T2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        self.denoise_type = denoise_type

        self.vae_scale_factor_temporal = (
            getattr(self.vae, "temporal_compression_ratio", None) or 4
            if getattr(self, "vae", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            getattr(self.vae, "spatial_compression_ratio", None) or 8
            if getattr(self, "vae", None)
            else 8
        )
        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "latent_channels", 16
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        self.llama_text_encoder = None

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
        elif self.model_type == ModelType.FRAMEPACK:
            return self.framepack_run(**final_kwargs)
        elif self.model_type == ModelType.HYAVATAR:
            return self.hyavatar_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        image: Union[
            Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor
        ] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        dtype: Optional[torch.dtype] = None,
        image_embed_interleave: int = 2,
        **kwargs,
    ):
        """Encode prompts using both LLaMA and CLIP text encoders"""
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        if not "llama" in self.preprocessors:
            self.load_preprocessor_by_type("llama")
        self.to_device(self.preprocessors["llama"])

        if self.llama_text_encoder is None:
            self.llama_text_encoder = self.preprocessors["llama"]

        if isinstance(prompt, str):
            prompt = [prompt]

        if isinstance(prompt_2, str):
            prompt_2 = [prompt_2]

        prompt_embeds, prompt_attention_mask = self.llama_text_encoder(
            prompt,
            image=image,
            max_sequence_length=max_sequence_length,
            pad_to_max_length=True,
            num_videos_per_prompt=num_videos_per_prompt,
            dtype=dtype,
            image_embed_interleave=image_embed_interleave,
        )

        pooled_prompt_embeds = self.text_encoder.encode(
            prompt_2,
            max_sequence_length=max_sequence_length,
            pad_to_max_length=True,
            num_videos_per_prompt=num_videos_per_prompt,
            dtype=dtype,
        )

        return pooled_prompt_embeds, prompt_embeds, prompt_attention_mask

    def t2v_run(
        self,
        prompt: Union[List[str], str],
        prompt_2: Union[List[str], str] = None,
        negative_prompt: Union[List[str], str] = None,
        negative_prompt_2: Union[List[str], str] = None,
        height: int = 720,
        width: int = 1280,
        duration: str | int = 10,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
        fps: int = 16,
        guidance_scale: float = 6.0,
        true_guidance_scale: float = 1.0,
        use_true_cfg_guidance: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        timesteps_as_indices: bool = True,
        max_sequence_length: int = 256,
        sigmas: List[float] = None,
        **kwargs,
    ):
        """Text-to-video generation following HunyuanVideoPipeline"""

        # 1. Encode prompts
        (
            pooled_prompt_embeds,
            prompt_embeds,
            prompt_attention_mask,
        ) = self._encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
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

        # 2. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            self.device, dtype=transformer_dtype
        )

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        # 3. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 4. Prepare timesteps
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self._get_timesteps(
            scheduler=self.scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
        )

        # 5. Prepare latents
        num_channels_latents = getattr(self.transformer.config, "in_channels", 16)
        latents = self._get_latents(
            height,
            width,
            duration,
            fps,
            num_videos,
            num_channels_latents,
            seed=seed,
            generator=generator,
            dtype=torch.float32,
        )

        # 6. Prepare guidance
        guidance = (
            torch.tensor(
                [guidance_scale] * latents.shape[0],
                dtype=transformer_dtype,
                device=self.device,
            )
            * 1000.0
        )
        use_true_cfg_guidance = (
            true_guidance_scale > 1.0 and negative_prompt_embeds is not None
        )

        # 7. Denoising loop
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            true_guidance_scale=true_guidance_scale,
            use_true_cfg_guidance=use_true_cfg_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
            ),
            unconditional_noise_pred_kwargs=dict(
                encoder_hidden_states=negative_prompt_embeds,
                encoder_attention_mask=negative_prompt_attention_mask,
                pooled_projections=negative_pooled_prompt_embeds,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
            ),
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
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

    def i2v_run(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        prompt: Union[List[str], str],
        prompt_2: Union[List[str], str] = None,
        negative_prompt: Union[List[str], str] = None,
        negative_prompt_2: Union[List[str], str] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
        fps: int = 16,
        guidance_scale: float = 1.0,
        true_guidance_scale: float = 1.0,
        use_true_cfg_guidance: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        timesteps_as_indices: bool = True,
        max_sequence_length: int = 256,
        sigmas: List[float] = None,
        image_embed_interleave: int = 2,
        **kwargs,
    ):
        """Image-to-video generation following HunyuanVideoImageToVideoPipeline"""

        # 1. Process input image
        loaded_image = self._load_image(image)
        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image, max_area=height * width
        )

        # Preprocess image
        image_tensor = self.video_processor.preprocess(loaded_image, height, width).to(
            self.device
        )

        # 2. Encode prompts with image context
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

        # 3. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            self.device, dtype=transformer_dtype
        )
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                self.device, dtype=transformer_dtype
            )

        # 4. Prepare image latents
        image_condition_type = getattr(
            self.transformer.config, "image_condition_type", "latent_concat"
        )

        if image_condition_type == "latent_concat":
            num_channels_latents = (
                getattr(self.transformer.config, "in_channels", 32) - 1
            ) // 2
        else:
            num_channels_latents = getattr(self.transformer.config, "in_channels", 32)

        # Encode image to latents
        image_tensor_unsqueezed = image_tensor.unsqueeze(2)  # Add temporal dimension
        image_latents = self.vae_encode(
            image_tensor_unsqueezed,
            offload=False,
            sample_mode="argmax",
            dtype=torch.float32,
        )

        # Repeat for all frames
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        image_latents = image_latents.repeat(1, 1, num_latent_frames, 1, 1)

        # 5. Prepare latents
        latents = self._get_latents(
            height=height,
            width=width,
            duration=num_frames,
            fps=fps,
            num_videos=num_videos,
            num_channels_latents=num_channels_latents,
            seed=seed,
            generator=generator,
            dtype=torch.float32,
        )

        # Mix latents with image latents
        t = torch.tensor([0.999]).to(device=self.device)
        latents = latents * t + image_latents * (1 - t)

        if image_condition_type == "token_replace":
            image_latents = image_latents[:, :, :1]

        # Create mask for image conditioning
        if image_condition_type == "latent_concat":
            image_latents[:, :, 1:] = 0
            mask = image_latents.new_ones(
                image_latents.shape[0], 1, *image_latents.shape[2:]
            )
            mask[:, :, 1:] = 0

        # 6. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 7. Prepare timesteps
        if sigmas is None:
            sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self._get_timesteps(
            sigmas=sigmas,
            scheduler=self.scheduler,
            timesteps=timesteps,
            timesteps_as_indices=timesteps_as_indices,
            num_inference_steps=num_inference_steps,
        )

        # 8. Prepare guidance
        guidance = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = (
                torch.tensor(
                    [guidance_scale] * latents.shape[0],
                    dtype=transformer_dtype,
                    device=self.device,
                )
                * 1000.0
            )

        use_true_cfg_guidance = (
            true_guidance_scale > 1.0 and negative_prompt_embeds is not None
        )
        # 9. Denoising loop
        latents = self.denoise(
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            true_guidance_scale=true_guidance_scale,
            use_true_cfg_guidance=use_true_cfg_guidance,
            noise_pred_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
            ),
            unconditional_noise_pred_kwargs=dict(
                encoder_hidden_states=negative_prompt_embeds,
                encoder_attention_mask=negative_prompt_attention_mask,
                pooled_projections=negative_pooled_prompt_embeds,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
            ),
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            image_condition_type=image_condition_type,
            image_latents=image_latents,
            mask=mask,
            **kwargs,
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            if image_condition_type == "latent_concat":
                return latents[:, :, 1:, :, :]
            else:
                return latents
        else:
            if image_condition_type == "latent_concat":
                video_latents = latents[:, :, 4:, :, :]  # Skip first few frames
            else:
                video_latents = latents

            video = self.vae_decode(video_latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video

    def framepack_run(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        prompt: Union[List[str], str],
        last_image: Optional[
            Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor]
        ] = None,
        prompt_2: Union[List[str], str] = None,
        negative_prompt: Union[List[str], str] = None,
        negative_prompt_2: Union[List[str], str] = None,
        height: int = 720,
        width: int = 1280,
        duration: str | int = 10,
        latent_window_size: int = 9,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int = None,
        fps: int = 16,
        guidance_scale: float = 6.0,
        true_guidance_scale: float = 1.0,
        use_true_cfg_guidance: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator = None,
        timesteps: List[int] = None,
        timesteps_as_indices: bool = True,
        max_sequence_length: int = 256,
        sigmas: List[float] = None,
        sampling_type: str = "inverted_anti_drifting",
        **kwargs,
    ):
        """Framepack generation following HunyuanVideoFramepackPipeline"""

        # 1. Process input images
        loaded_image = self._load_image(image)
        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image, max_area=height * width
        )

        image_tensor = self.video_processor.preprocess(loaded_image, height, width).to(
            self.device
        )

        last_image_tensor = None
        if last_image is not None:
            loaded_last_image = self._load_image(last_image)
            loaded_last_image, height, width = self._aspect_ratio_resize(
                loaded_last_image, max_area=height * width
            )
            last_image_tensor = self.video_processor.preprocess(
                loaded_last_image, height, width
            ).to(self.device)

        # 2. Encode prompts
        (
            prompt_embeds,
            pooled_prompt_embeds,
            prompt_attention_mask,
        ) = self._encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            num_videos_per_prompt=num_videos,
            max_sequence_length=max_sequence_length,
        )

        if negative_prompt is not None:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                num_videos_per_prompt=num_videos,
                max_sequence_length=max_sequence_length,
                **text_encoder_kwargs,
            )

        # 3. Encode images
        if not self.preprocessors or "clip" not in self.preprocessors:
            self.load_preprocessor_by_type("clip")

        clip_image_encoder = self.preprocessors["clip"]
        self.to_device(clip_image_encoder)

        preprocessed_image = self.video_processor.preprocess(
            loaded_image, height, width
        )
        # convert to 0 to 1
        preprocessed_image = (preprocessed_image + 1) / 2.0
        image_embeds = clip_image_encoder(
            preprocessed_image, hidden_states_layer=-2
        ).to(self.device)

        if last_image_tensor is not None:
            preprocessed_last_image = self.video_processor.preprocess(
                loaded_last_image, height, width
            )
            preprocessed_last_image = (preprocessed_last_image + 1) / 2.0
            last_image_embeds = clip_image_encoder(
                preprocessed_last_image, hidden_states_layer=-2
            ).to(self.device)
            # Blend embeddings as in the original implementation
            image_embeds = (image_embeds + last_image_embeds) / 2

        if offload:
            self._offload(self.text_encoder)
            if self.llama_text_encoder is not None:
                self._offload(self.llama_text_encoder)

        # 4. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            self.device, dtype=transformer_dtype
        )
        image_embeds = image_embeds.to(self.device, dtype=transformer_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        # 5. Prepare image latents
        num_channels_latents = getattr(self.transformer.config, "in_channels", 16)

        # Encode first image
        image_tensor_unsqueezed = image_tensor.unsqueeze(2)
        image_latents = self.vae_encode(
            image_tensor_unsqueezed,
            offload=offload,
            sample_mode="sample",
            dtype=torch.float32,
        )

        # Encode last image if provided
        last_image_latents = None
        if last_image_tensor is not None:
            last_image_tensor_unsqueezed = last_image_tensor.unsqueeze(2)
            last_image_latents = self.vae_encode(
                last_image_tensor_unsqueezed,
                offload=offload,
                sample_mode="sample",
                dtype=torch.float32,
            )

        # 6. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 7. Framepack generation loop
        num_frames = self._parse_num_frames(duration, fps)
        window_num_frames = (
            latent_window_size - 1
        ) * self.vae_scale_factor_temporal + 1
        num_latent_sections = max(
            1, (num_frames + window_num_frames - 1) // window_num_frames
        )
        history_video = None
        total_generated_latent_frames = 0

        # Initialize history based on sampling type
        if sampling_type == "inverted_anti_drifting":
            history_sizes = [1, 2, 16]
        else:  # vanilla
            history_sizes = [16, 2, 1]

        history_latents = torch.zeros(
            num_videos,
            num_channels_latents,
            sum(history_sizes),
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
            device=self.device,
            dtype=torch.float32,
        )

        if sampling_type == "vanilla":
            history_latents = torch.cat([history_latents, image_latents], dim=2)
            total_generated_latent_frames += 1

        # 8. Guidance preparation
        guidance = (
            torch.tensor(
                [guidance_scale] * num_videos,
                dtype=transformer_dtype,
                device=self.device,
            )
            * 1000.0
        )
        use_true_cfg_guidance = (
            true_guidance_scale > 1.0 and negative_prompt_embeds is not None
        )

        # 9. Generation loop for each section
        for k in range(num_latent_sections):
            # Prepare latents for this section
            latents = self._get_latents(
                height=height,
                width=width,
                duration=window_num_frames,
                fps=fps,
                num_videos=num_videos,
                num_channels_latents=num_channels_latents,
                seed=seed,
                generator=generator,
                dtype=torch.float32,
            )

            # Prepare timesteps with dynamic shift
            if sigmas is None:
                sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]

            # Calculate shift based on sequence length (from framepack implementation)
            image_seq_len = (
                latents.shape[2]
                * latents.shape[3]
                * latents.shape[4]
                / getattr(self.transformer.config, "patch_size", 2) ** 2
            )
            mu = self._calculate_shift(image_seq_len)

            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps, num_inference_steps = self._get_timesteps(
                scheduler=self.scheduler,
                timesteps=timesteps,
                timesteps_as_indices=timesteps_as_indices,
                num_inference_steps=num_inference_steps,
                mu=mu,
                sigmas=sigmas,
            )

            # Prepare history latents for this section
            if sampling_type == "inverted_anti_drifting":
                latent_paddings = list(reversed(range(num_latent_sections)))
                if num_latent_sections > 4:
                    latent_paddings = [3] + [2] * (num_latent_sections - 3) + [1, 0]

                is_first_section = k == 0
                is_last_section = k == num_latent_sections - 1
                latent_padding_size = latent_paddings[k] * latent_window_size

                indices = torch.arange(
                    0, sum([1, latent_padding_size, latent_window_size, *history_sizes])
                )

                (
                    indices_prefix,
                    indices_padding,
                    indices_latents,
                    indices_latents_history_1x,
                    indices_latents_history_2x,
                    indices_latents_history_4x,
                ) = indices.split(
                    [1, latent_padding_size, latent_window_size, *history_sizes], dim=0
                )

                indices_clean_latents = torch.cat(
                    [indices_prefix, indices_latents_history_1x], dim=0
                )

                latents_prefix = image_latents
                latents_history_1x, latents_history_2x, latents_history_4x = (
                    history_latents[:, :, : sum(history_sizes)].split(
                        history_sizes, dim=2
                    )
                )

                if last_image_latents is not None and is_first_section:
                    latents_history_1x = last_image_latents

                latents_clean = torch.cat([latents_prefix, latents_history_1x], dim=2)

            else:  # vanilla
                indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                (
                    indices_prefix,
                    indices_latents_history_4x,
                    indices_latents_history_2x,
                    indices_latents_history_1x,
                    indices_latents,
                ) = indices.split([1, *history_sizes, latent_window_size], dim=0)

                indices_clean_latents = torch.cat(
                    [indices_prefix, indices_latents_history_1x], dim=0
                )

                latents_prefix = image_latents
                latents_history_4x, latents_history_2x, latents_history_1x = (
                    history_latents[:, :, -sum(history_sizes) :].split(
                        history_sizes, dim=2
                    )
                )

                latents_clean = torch.cat([latents_prefix, latents_history_1x], dim=2)

            latents = self.denoise(
                latents=latents,
                timesteps=timesteps,
                scheduler=self.scheduler,
                true_guidance_scale=true_guidance_scale,
                use_true_cfg_guidance=use_true_cfg_guidance,
                noise_pred_kwargs=dict(
                    indices_latents=indices_latents,
                    latents_clean=latents_clean,
                    indices_clean_latents=indices_clean_latents,
                    latents_history_2x=latents_history_2x,
                    indices_latents_history_2x=indices_latents_history_2x,
                    latents_history_4x=latents_history_4x,
                    indices_latents_history_4x=indices_latents_history_4x,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    pooled_projections=pooled_prompt_embeds,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                ),
                unconditional_noise_pred_kwargs=dict(
                    indices_latents=indices_latents,
                    latents_clean=latents_clean,
                    indices_clean_latents=indices_clean_latents,
                    latents_history_2x=latents_history_2x,
                    indices_latents_history_2x=indices_latents_history_2x,
                    latents_history_4x=latents_history_4x,
                    indices_latents_history_4x=indices_latents_history_4x,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_attention_mask=negative_prompt_attention_mask,
                    pooled_projections=negative_pooled_prompt_embeds,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                ),
                render_on_step=render_on_step,
                render_on_step_callback=render_on_step_callback,
                transformer_dtype=transformer_dtype,
                num_inference_steps=num_inference_steps,
                num_latent_sections=num_latent_sections,
                **kwargs,
            )

            # Update history
            if sampling_type == "inverted_anti_drifting":
                if is_last_section:
                    latents = torch.cat([image_latents, latents], dim=2)
                total_generated_latent_frames += latents.shape[2]
                history_latents = torch.cat([latents, history_latents], dim=2)
                real_history_latents = history_latents[
                    :, :, :total_generated_latent_frames
                ]
                section_latent_frames = (
                    (latent_window_size * 2 + 1)
                    if is_last_section
                    else (latent_window_size * 2)
                )
                index_slice = (
                    slice(None),
                    slice(None),
                    slice(0, section_latent_frames),
                )

            else:  # vanilla
                total_generated_latent_frames += latents.shape[2]
                history_latents = torch.cat([history_latents, latents], dim=2)
                real_history_latents = history_latents[
                    :, :, -total_generated_latent_frames:
                ]
                section_latent_frames = latent_window_size * 2
                index_slice = (
                    slice(None),
                    slice(None),
                    slice(-section_latent_frames, None),
                )

            if history_video is None:
                if not return_latents:
                    current_latents = real_history_latents.to(self.vae.dtype) / getattr(
                        self.vae.config, "scaling_factor", 1.0
                    )
                    history_video = self.vae_decode(current_latents, offload=False)
                else:
                    history_video = [real_history_latents]
            else:
                if not return_latents:
                    overlapped_frames = (
                        latent_window_size - 1
                    ) * self.vae_scale_factor_temporal + 1
                    current_latents = real_history_latents[index_slice].to(
                        self.vae.dtype
                    ) / getattr(self.vae.config, "scaling_factor", 1.0)
                    current_video = self.vae_decode(current_latents, offload=False)

                    if sampling_type == "inverted_anti_drifting":
                        history_video = self._soft_append(
                            current_video, history_video, overlapped_frames
                        )
                    else:  # vanilla
                        history_video = self._soft_append(
                            history_video, current_video, overlapped_frames
                        )
                else:
                    history_video.append(real_history_latents)

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return history_video
        else:
            # Ensure proper frame count
            generated_frames = history_video.size(2)
            generated_frames = (
                generated_frames - 1
            ) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            history_video = history_video[:, :, :generated_frames]
            postprocessed_video = self._postprocess(history_video)
            return postprocessed_video

    ### REQUIRES FIXING FOR CORRECTNESS!!!!
    def hyavatar_run(
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
            uncond_audio_prompts=uncond_audio_prompts,
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

    def _soft_append(
        self, history: torch.Tensor, current: torch.Tensor, overlap: int = 0
    ):
        """Soft append with blending for framepack generation"""
        if overlap <= 0:
            return torch.cat([history, current], dim=2)

        assert (
            history.shape[2] >= overlap
        ), f"Current length ({history.shape[2]}) must be >= overlap ({overlap})"
        assert (
            current.shape[2] >= overlap
        ), f"History length ({current.shape[2]}) must be >= overlap ({overlap})"

        weights = torch.linspace(
            1, 0, overlap, dtype=history.dtype, device=history.device
        ).view(1, 1, -1, 1, 1)
        blended = (
            weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
        )
        output = torch.cat(
            [history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2
        )

        return output.to(history)

    def __str__(self):
        return f"HunyuanEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    # Example usage for text-to-video
    engine = HunyuanEngine(
        yaml_path="manifest/hunyuan_t2v.yml",  # You'll need to create this
        model_type=ModelType.T2V,
        save_path="./apex-models",
        components_to_load=["transformer", "text_encoder", "vae", "scheduler"],
        component_dtypes={"vae": torch.float16},
    )

    prompt = "A cat walks on the grass, realistic"
    video = engine.run(
        height=320,
        width=512,
        num_frames=61,
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=6.0,
        seed=42,
    )

    export_to_video(video[0], "hunyuan_t2v_output.mp4", fps=15)
