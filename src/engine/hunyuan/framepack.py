import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import HunyuanBaseEngine


class HunyuanFramepackEngine(HunyuanBaseEngine):
    """Hunyuan Framepack Engine Implementation"""
    
    def run(
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