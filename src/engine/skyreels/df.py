import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np
import math
from copy import deepcopy
from .base import SkyReelsBaseEngine


class SkyReelsDFEngine(SkyReelsBaseEngine):
    """SkyReels Diffusion Forcing Engine Implementation"""
    
    def run(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ] = None,
        image: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ] = None,
        end_image: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ] = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 97,
        base_duration: int | str = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        use_cfg_guidance: bool = True,
        seed: int | None = None,
        num_videos: int = 1,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        overlap_history: int = None,
        addnoise_condition: int = 0,
        ar_step: int = 5,
        causal_block_size: int = 1,
        causal_attention: bool = False,
        fps: int = 24,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        return_latents: bool = False,
    ):

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

        output_video = None
        prefix_latent = None
        end_latent = None

        if video is not None:
            loaded_video = self._load_video(video)

            video_height, video_width = self.video_processor.get_default_height_width(
                loaded_video[0]
            )
            base = self.vae_scale_factor_spatial * 8
            if video_height * video_width > height * width:
                scale = min(width / video_width, height / video_height)
                video_height, video_width = int(video_height * scale), int(
                    video_width * scale
                )

            if video_height % base != 0 or video_width % base != 0:
                video_height = (video_height // base) * base
                video_width = (video_width // base) * base

            assert video_height * video_width <= height * width

            preprocessed_video = self.video_processor.preprocess_video(
                loaded_video, video_height, video_width
            )
            output_video = preprocessed_video

        if image is not None:
            image = self._load_image(image)
            image, height, width = self._aspect_ratio_resize(
                image, max_area=height * width
            )

            preprocessed_image = self.video_processor.preprocess(
                image, height=height, width=width
            ).to(self.device, dtype=torch.float32)

            prefix_latent = self.vae_encode(
                preprocessed_image,
                offload=offload,
                dtype=torch.float32,
                normalize_latents_dtype=torch.float32,
            )

        if end_image is not None:
            end_image = self._load_image(end_image)
            end_image, height, width = self._aspect_ratio_resize(
                end_image, max_area=height * width
            )

            preprocessed_end_image = self.video_processor.preprocess(
                end_image, height=height, width=width
            ).to(self.device, dtype=torch.float32)

            end_latent = self.vae_encode(
                preprocessed_end_image,
                offload=offload,
                dtype=torch.float32,
                normalize_latents_dtype=torch.float32,
            )

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        scheduler = self.scheduler
        self.to_device(scheduler)
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self._get_timesteps(timesteps, timesteps_as_indices)

        if not self.transformer:
            self.load_component_by_type("transformer")

        transformer_dtype = self.component_dtypes["transformer"]
        self.to_device(self.transformer)
        fps_embeds = [fps] * prompt_embeds.shape[0]
        fps_embeds = [0 if i == 16 else 1 for i in fps_embeds]
        fps_embeds = torch.tensor(fps_embeds, dtype=torch.long, device=self.device)
        num_frames = self._parse_num_frames(duration, fps)
        base_num_frames = self._parse_num_frames(base_duration, fps)

        if causal_attention:
            self.logger.info(
                f"Setting causal attention with block size {causal_block_size}."
            )
            pt, ph, pw = self.transformer.config.patch_size
            latent_num_frames = math.ceil(
                (base_num_frames + 3) / self.vae_scale_factor_temporal
            )
            latent_height = math.ceil(height / self.vae_scale_factor_spatial) // ph
            latent_width = math.ceil(width / self.vae_scale_factor_spatial) // pw
            self.transformer.set_causal_attention(
                causal_block_size,
                latent_num_frames,
                latent_height,
                latent_width,
                self.device,
            )

        if (
            overlap_history is None
            or base_num_frames is None
            or num_frames <= base_num_frames
        ):
            latents, generator = self._get_latents(
                height,
                width,
                base_num_frames,
                fps=fps,
                num_videos=num_videos,
                seed=seed if not generator else None,
                dtype=torch.float32,
                layout=torch.strided,
                generator=generator,
                return_generator=True,
            )

            latent_length = latents.shape[2]

            latent_base_num_frames = (
                (base_num_frames - 1) // 4 + 1
                if base_num_frames is not None
                else latent_length
            )

            if prefix_latent is not None:
                latents[:, :, : prefix_latent.shape[2]] = prefix_latent.to(
                    self.device, dtype=latents.dtype
                )

            if end_latent is not None:
                latents = torch.cat(
                    [latents, end_latent.to(self.device, dtype=latents.dtype)], dim=2
                )
                latent_base_num_frames += latents.shape[2]

            latent_length = latents.shape[2]
            prefix_latent_length = (
                prefix_latent.shape[2] if prefix_latent is not None else 0
            )
            end_latent_length = end_latent.shape[2] if end_latent is not None else 0

            step_matrix, _, step_update_mask, valid_interval = (
                self.generate_timestep_matrix(
                    latent_length,
                    timesteps,
                    latent_base_num_frames,
                    ar_step,
                    prefix_latent_length,
                    causal_block_size,
                )
            )

            if end_latent is not None:
                step_matrix[:, -end_latent_length:] = 0
                step_update_mask[:, -end_latent_length:] = False

            schedulers = [deepcopy(scheduler) for _ in range(latent_length)]
            schedulers_counter = [0] * latent_length

            latents = self.denoise(
                latents=latents,
                timesteps=timesteps,
                latent_condition=prefix_latent,
                transformer_dtype=transformer_dtype,
                use_cfg_guidance=use_cfg_guidance,
                fps_embeds=fps_embeds,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                attention_kwargs=attention_kwargs,
                addnoise_condition=addnoise_condition,
                encoded_image_length=prefix_latent_length,
                step_matrix=step_matrix,
                step_update_mask=step_update_mask,
                valid_interval=valid_interval,
                generator=generator,
                schedulers=schedulers,
                guidance_scale=guidance_scale,
                schedulers_counter=schedulers_counter,
                render_on_step=render_on_step,
                render_on_step_callback=render_on_step_callback,
            )

            if end_latent is not None:
                latents = latents[:, :, :-end_latent_length]

            if return_latents:
                return latents
            else:
                video = self.vae_decode(latents, offload=offload)
                postprocessed_video = self._postprocess(video)
                return postprocessed_video

        else:
            if return_latents:
                self.logger.warning(
                    "return_latents is not supported for long video generation. Setting return_latents to False."
                )
                return_latents = False

            latent_length = math.ceil((num_frames + 3) / self.vae_scale_factor_temporal)

            latent_base_num_frames = (
                (base_num_frames - 1) // 4 + 1
                if base_num_frames is not None
                else latent_length
            )
            latent_overlap_history_frames = (overlap_history - 1) // 4 + 1
            n_iter = (
                1
                + (latent_length - latent_base_num_frames - 1)
                // (latent_base_num_frames - latent_overlap_history_frames)
                + 1
            )
            prefix_latent_length = (
                prefix_latent.shape[2] if prefix_latent is not None else 0
            )
            self.logger.info(f"Generating {n_iter} iterations of video.")

            with self._progress_bar(total=n_iter, desc="Generating video") as pbar:
                for i in range(n_iter):
                    if output_video is not None:
                        prefix_video = output_video[:, :, -overlap_history:].to(
                            self.device
                        )
                        prefix_latent = self.vae_encode(
                            prefix_video,
                            offload=offload,
                            dtype=torch.float32,
                            normalize_latents_dtype=torch.float32,
                        )
                        if prefix_latent.shape[2] % causal_block_size != 0:
                            truncate_len = prefix_latent.shape[1] % causal_block_size
                            self.logger.warning(
                                "the length of prefix video is truncated for the casual block size alignment."
                            )
                            prefix_latent = prefix_latent[
                                :, : prefix_latent.shape[2] - truncate_len, :, :
                            ]
                        prefix_latent_length = prefix_latent.shape[2]
                        finished_frame_num = (
                            i * (latent_base_num_frames - latent_overlap_history_frames)
                            + latent_overlap_history_frames
                        )
                        left_frame_num = latent_length - finished_frame_num
                        latent_base_num_frames_iter = min(
                            left_frame_num + latent_overlap_history_frames,
                            latent_base_num_frames,
                        )
                    else:  # i == 0
                        latent_base_num_frames_iter = latent_base_num_frames

                    latents, generator = self._get_latents(
                        height,
                        width,
                        latent_base_num_frames_iter,
                        fps=fps,
                        num_videos=num_videos,
                        seed=seed if not generator else None,
                        dtype=torch.float32,
                        layout=torch.strided,
                        generator=generator,
                        return_generator=True,
                        parse_frames=False,
                    )

                    if prefix_latent is not None:
                        latents[:, :, : prefix_latent.shape[2]] = prefix_latent.to(
                            self.device, dtype=latents.dtype
                        )

                    if end_latent is not None:
                        latents = torch.cat(
                            [latents, end_latent.to(self.device, dtype=latents.dtype)],
                            dim=2,
                        )

                    step_matrix, _, step_update_mask, valid_interval = (
                        self.generate_timestep_matrix(
                            latent_base_num_frames_iter,
                            timesteps,
                            latent_base_num_frames_iter,
                            ar_step,
                            prefix_latent_length,
                            causal_block_size,
                        )
                    )

                    if end_latent is not None and i == n_iter - 1:
                        step_matrix[:, -end_latent_length:] = 0
                        step_update_mask[:, -end_latent_length:] = False

                    scheduler_component = [
                        component
                        for component in self.config.get("components", [])
                        if component.get("type") == "scheduler"
                    ][0]
                    schedulers = [
                        self._load_component(scheduler_component)
                        for _ in range(latent_base_num_frames_iter)
                    ]
                    for scheduler in schedulers:
                        scheduler.set_timesteps(num_inference_steps, device=self.device)
                    schedulers_counter = [0] * latent_base_num_frames_iter

                    latents = self.denoise(
                        latents=latents,
                        timesteps=timesteps,
                        latent_condition=prefix_latent,
                        transformer_dtype=transformer_dtype,
                        use_cfg_guidance=use_cfg_guidance,
                        fps_embeds=fps_embeds,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        addnoise_condition=addnoise_condition,
                        encoded_image_length=prefix_latent_length,
                        step_matrix=step_matrix,
                        step_update_mask=step_update_mask,
                        valid_interval=valid_interval,
                        generator=generator,
                        schedulers=schedulers,
                        guidance_scale=guidance_scale,
                        schedulers_counter=schedulers_counter,
                        render_on_step=render_on_step,
                        render_on_step_callback=render_on_step_callback,
                    )

                    if end_latent is not None:
                        latents = latents[:, :, :-end_latent_length]

                    video = self.vae_decode(latents, offload=offload)
                    if output_video is None:
                        output_video = video
                    else:
                        output_video = torch.cat(
                            [output_video, video[:, :, overlap_history:]], dim=2
                        )
                    pbar.update(1)

            if offload:
                self._offload(self.transformer)

            if output_video is not None:
                postprocessed_video = self._postprocess(output_video)
                return postprocessed_video