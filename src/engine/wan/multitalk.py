import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import WanBaseEngine


class WanMultitalkEngine(WanBaseEngine):
    """WAN MultiTalk (Audio-driven) Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        image: Union[Image.Image, str],
        audio_paths: Optional[Dict[str, str]] = None,
        audio_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        audio_type: str = "para",
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        motion_frames: int = 25,
        num_inference_steps: int = 40,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 25,
        guidance_scale: float = 5.0,
        audio_guidance_scale: float = 4.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        max_frames_num: int = 1000,
        face_scale: float = 0.05,
        color_correction_strength: float = 0.0,
        bbox: Optional[Dict[str, List[float]]] = None,
        shift: float = 5.0,
        use_timestep_transform: bool = True,
        duration: int | str = 16,
        **kwargs,
    ):
        """
        Generate MultiTalk video from image, text prompt, and audio inputs.

        Args:
            prompt: Text prompt for the video
            image: Input conditioning image (path or PIL Image)
            audio_paths: Dictionary mapping person names to audio file paths
            audio_embeddings: Pre-computed audio embeddings
            audio_type: Type of audio combination ("para" or "add")
            negative_prompt: Negative text prompt
            height: Output video height
            width: Output video width
            num_frames: Number of frames to generate
            motion_frames: Number of motion frames for extended generation
            num_inference_steps: Number of diffusion steps
            num_videos: Number of videos to generate
            seed: Random seed
            fps: Frames per second
            guidance_scale: Text guidance scale
            audio_guidance_scale: Audio guidance scale
            use_cfg_guidance: Whether to use classifier-free guidance
            bbox: Bounding boxes for multiple people
            shift: Timestep transform shift parameter
            use_timestep_transform: Whether to apply timestep transformation
        """

        try:
            # Set random seed
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                import numpy as np
                import random

                np.random.seed(seed)
                random.seed(seed)

            # Load and preprocess image
            image = self._load_image(image)

            # Process with multitalk preprocessor if available
            processed_audio = None
            human_masks = None
            human_num = 1

            if "multitalk" in self.preprocessors:
                try:
                    preprocessor = self.preprocessors["multitalk"]
                    processed_inputs = preprocessor(
                        image=image,
                        audio_paths=audio_paths,
                        audio_embeddings=audio_embeddings,
                        audio_type=audio_type,
                        num_frames=num_frames,
                        vae_scale=self.vae_scale_factor_temporal,
                        bbox=bbox,
                        face_scale=face_scale,
                    )

                    processed_audio = processed_inputs["audio_embeddings"]
                    human_masks = processed_inputs["human_masks"]
                    human_num = processed_inputs["human_num"]
                except Exception as e:
                    print(f"Warning: MultiTalk preprocessor failed: {e}")
                    # Fallback to basic processing
                    processed_audio = audio_embeddings if audio_embeddings else None
                    human_num = len(audio_paths) if audio_paths else 1
            else:
                # Fallback processing
                processed_audio = audio_embeddings if audio_embeddings else None
                human_num = len(audio_paths) if audio_paths else 1

            # Preprocess image
            resized_image, height, width = self._aspect_ratio_resize(
                image, height * width
            )
            preprocessed_image = self.video_processor.preprocess(
                resized_image, height, width
            )

            # Encode text prompts
            if not self.text_encoder:
                self.load_component_by_type("text_encoder")

            self.to_device(self.text_encoder)

            prompt_embeds = self.text_encoder.encode(
                prompt,
                device=self.device,
                num_videos_per_prompt=num_videos,
                **text_encoder_kwargs,
            )

            if negative_prompt is not None and use_cfg_guidance:
                negative_prompt_embeds = self.text_encoder.encode(
                    negative_prompt,
                    device=self.device,
                    num_videos_per_prompt=num_videos,
                    **text_encoder_kwargs,
                )
            else:
                negative_prompt_embeds = None

            if offload:
                self._offload(self.text_encoder)

            # Encode image with CLIP if available
            image_embeds = None
            if "clip" in self.preprocessors:
                try:
                    clip_processor = self.preprocessors["clip"]
                    image_embeds = clip_processor(preprocessed_image)
                except Exception as e:
                    print(f"Warning: CLIP processing failed: {e}")

            # Load transformer
            if not self.transformer:
                self.load_component_by_type("transformer")

            self.to_device(self.transformer)
            transformer_dtype = self.component_dtypes["transformer"]
            prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    self.device, dtype=transformer_dtype
                )

            # Load scheduler
            if not self.scheduler:
                self.load_component_by_type("scheduler")
            self.to_device(self.scheduler)
            scheduler = self.scheduler

            scheduler.set_timesteps(
                num_inference_steps if timesteps is None else 1000, device=self.device
            )
            timesteps, num_inference_steps = self._get_timesteps(
                scheduler=scheduler,
                timesteps=timesteps,
                timesteps_as_indices=timesteps_as_indices,
                num_inference_steps=num_inference_steps,
            )
            num_frames = self._parse_num_frames(duration, fps)

            # Get latents
            latents = self._get_latents(
                height,
                width,
                duration,
                fps=fps,
                num_videos=num_videos,
                seed=seed,
                dtype=torch.float32,
                generator=generator,
            )

            # Prepare conditioning image
            if preprocessed_image.ndim == 4:
                preprocessed_image = preprocessed_image.unsqueeze(2)

            video_condition = torch.cat(
                [
                    preprocessed_image,
                    preprocessed_image.new_zeros(
                        preprocessed_image.shape[0],
                        preprocessed_image.shape[1],
                        num_frames - 1,
                        height,
                        width,
                    ),
                ],
                dim=2,
            )

            latent_condition = self.vae_encode(
                video_condition,
                offload=offload,
                dtype=latents.dtype,
                normalize_latents_dtype=latents.dtype,
            )
            batch_size, _, _, latent_height, latent_width = latents.shape

            # Create mask for conditioning frames
            mask_lat_size = torch.ones(
                batch_size,
                1,
                num_frames,
                latent_height,
                latent_width,
                device=self.device,
            )
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(
                first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
            )
            mask_lat_size = torch.concat(
                [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
            )
            mask_lat_size = mask_lat_size.view(
                batch_size,
                -1,
                self.vae_scale_factor_temporal,
                latent_height,
                latent_width,
            )

            mask_lat_size = mask_lat_size.transpose(1, 2)
            mask_lat_size = mask_lat_size.to(latents.device)

            latent_condition = torch.concat([mask_lat_size, latent_condition], dim=1)

            # Prepare human masks for latent space if available
            if human_masks is not None:
                try:
                    # Resize masks to latent space
                    human_masks = F.interpolate(
                        human_masks.unsqueeze(0),
                        size=(latent_height, latent_width),
                        mode="nearest",
                    ).squeeze(0)
                    human_masks = (human_masks > 0).float().to(self.device)
                except Exception as e:
                    print(f"Warning: Human mask processing failed: {e}")
                    human_masks = None

            # Run denoising with MultiTalk support
            latents = self.denoise(
                timesteps=timesteps,
                latents=latents,
                latent_condition=latent_condition,
                transformer_kwargs=dict(
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    encoder_hidden_states_audio=processed_audio,
                    ref_target_masks=human_masks,
                    human_num=human_num,
                    attention_kwargs=attention_kwargs,
                ),
                unconditional_transformer_kwargs=(
                    dict(
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        encoder_hidden_states_audio=(
                            torch.zeros_like(processed_audio)
                            if processed_audio is not None
                            else None
                        ),
                        ref_target_masks=human_masks,
                        human_num=human_num,
                        attention_kwargs=attention_kwargs,
                    )
                    if negative_prompt_embeds is not None
                    else None
                ),
                transformer_dtype=transformer_dtype,
                use_cfg_guidance=use_cfg_guidance,
                render_on_step=render_on_step,
                render_on_step_callback=render_on_step_callback,
                scheduler=scheduler,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                shift=shift,
                use_timestep_transform=use_timestep_transform,
            )

            if offload:
                self._offload(self.transformer)

            if return_latents:
                return latents
            else:
                video = self.vae_decode(latents, offload=offload)

                # Apply color correction if needed
                if color_correction_strength > 0.0:
                    try:
                        video = self._apply_color_correction(
                            video, preprocessed_image, color_correction_strength
                        )
                    except Exception as e:
                        print(f"Warning: Color correction failed: {e}")

                postprocessed_video = self._postprocess(video)
                return postprocessed_video

        except Exception as e:
            print(f"Error in multitalk_run: {e}")
            import traceback

            traceback.print_exc()
            raise
