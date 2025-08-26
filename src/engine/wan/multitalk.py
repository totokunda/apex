import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
from .base import WanBaseEngine
from torch.nn import functional as F
from torchvision import transforms as T
from diffusers.utils.torch_utils import randn_tensor
from src.utils.models.wan import match_and_blend_colors
import numpy as np


class WanMultitalkEngine(WanBaseEngine):
    """WAN MultiTalk (Audio-driven) Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        image: Union[Image.Image, str, np.ndarray, torch.Tensor] | None = None,
        video: Union[List[Image.Image], str, np.ndarray, torch.Tensor, None] = None,
        audio_paths: Optional[Dict[str, str]] = None,
        audio_type: str = "para",
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 81,
        max_num_frames: int = 1000,
        num_inference_steps: int = 40,
        num_videos: int = 1,
        seed: int | None = None,
        motion_frames: int = 25,
        fps: int = 25,
        guidance_scale: float = 5.0,
        audio_guidance_scale: float = 4.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        face_scale: float = 0.05,
        color_correction_strength: float = 1.0,
        bbox: Optional[Dict[str, List[float]]] = None,
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
        """

        num_frames = self._parse_num_frames(duration, fps)

        assert (
            image is not None or video is not None
        ), "Either image or video must be provided"

        if image is not None:
            loaded_image = self._load_image(image)
            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width, mod_value=16
            )

            cond_image = self.video_processor.preprocess(
                loaded_image, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            cond_image = cond_image.unsqueeze(2)

        if video is not None:
            input_video = self._load_video(video)
            image = input_video[0]
            for idx, frame in enumerate(input_video):
                frame, height, width = self._aspect_ratio_resize(
                    frame, max_area=height * width, mod_value=16
                )
                input_video[idx] = frame

            loaded_image = input_video[0]
            input_video = self.video_processor.preprocess_video(
                input_video, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            cond_image = input_video[:, :, :1, :, :]
        else:
            input_video = None

        cond_frame = None

        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image.clone()

        if seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        preprocessor = self.helpers["wan.multitalk"]
        processed_inputs = preprocessor(
            image=image,
            audio_paths=audio_paths,
            audio_type=audio_type,
            num_frames=num_frames,
            vae_scale=self.vae_scale_factor_temporal,
            bbox=bbox,
            face_scale=face_scale,
        )

        human_masks = processed_inputs["human_masks"]
        human_num = processed_inputs["human_num"]
        full_audio_embs = processed_inputs["audio_embeddings"]

        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        clip_length = num_frames
        cur_motion_frames_num = 1
        audio_start_idx = 0
        arrive_last_frame = False
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []
        gen_latents_list = []
        is_first_clip = True

        transformer_dtype = self.component_dtypes["transformer"]

        if not self.text_encoder:
            self.load_component_by_type("text_encoder")

        self.to_device(self.text_encoder)

        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )
        
        batch_size = prompt_embeds.shape[0]

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

        if not self.transformer:
            self.load_component_by_type("transformer")
            self.to_device(self.transformer)


        using_video_input = input_video is not None
        while True:
            audio_embs = []
            # split audio with window size
            for human_idx in range(human_num):
                center_indices = torch.arange(
                    audio_start_idx,
                    audio_end_idx,
                    1,
                ).unsqueeze(1) + indices.unsqueeze(0)
                center_indices = torch.clamp(
                    center_indices, min=0, max=full_audio_embs[human_idx].shape[0] - 1
                )
                audio_emb = full_audio_embs[human_idx][center_indices][None, ...].to(
                    self.device
                )
                audio_embs.append(audio_emb)

            audio_embs = torch.cat(audio_embs, dim=0).to(transformer_dtype)

            latent_height, latent_width = (
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )

            # get mask
            mask_lat_size = torch.ones(
                batch_size,
                1,
                num_frames,
                latent_height,
                latent_width,
                device=self.device,
            )
            # For InfiniteTalk (video input), only the first frame is preserved; for image mode, preserve cur_motion_frames_num
            if using_video_input:
                mask_lat_size[:, :, 1:] = 0
            else:
                mask_lat_size[:, :, cur_motion_frames_num:] = 0
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

            # get clip embedding
            clip_processor = self.helpers["clip"]

            image_embeds = clip_processor(loaded_image, hidden_states_layer=-2).to(
                transformer_dtype
            )

            if offload:
                self._offload(clip_processor)

            # zero padding and vae encode
            # InfiniteTalk: always condition on the current source video frame when a video is provided;
            # MultiTalk (image-only): condition on previous generated frames after the first clip
            if is_first_clip:
                base_condition = cond_image
            else:
                base_condition = cond_image if using_video_input else cond_frame

            video_condition = torch.cat(
                [
                    base_condition,
                    base_condition.new_zeros(
                        base_condition.shape[0],
                        base_condition.shape[1],
                        num_frames - base_condition.shape[2],
                        height,
                        width,
                    ),
                ],
                dim=2,
            )

            latent_condition = self.vae_encode(
                video_condition,
                offload=offload,
                dtype=torch.float32,
                normalize_latents_dtype=torch.float32,
            )

            cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)
            # For InfiniteTalk (video input), match reference behavior:
            #   - first clip: inject latents encoded from the current source frame
            #   - subsequent clips: inject latents encoded from last generated frames
            if using_video_input:
                motion_source = cond_image if is_first_clip else cond_frame
                motion_latents = self.vae_encode(
                    motion_source,
                    offload=offload,
                    dtype=torch.float32,
                    normalize_latents_dtype=torch.float32,
                )
                latent_motion_frames = motion_latents[0]
            else:
                latent_motion_frames = latent_condition[
                    :, :, :cur_motion_frames_latent_num
                ][
                    0
                ]  # C T H W
            latent_condition = torch.concat(
                [mask_lat_size.to(latent_condition), latent_condition], dim=1
            )  # B 4+C T H W

            # prepare masks
            ref_target_masks = self.resize_and_centercrop(human_masks, (height, width))
            ref_target_masks = F.interpolate(
                ref_target_masks.unsqueeze(0),
                size=(latent_height, latent_width),
                mode="nearest",
            ).squeeze()
            ref_target_masks = ref_target_masks > 0
            ref_target_masks = ref_target_masks.float().to(self.device)

            # prepare noise
            latents = randn_tensor(
                (
                    batch_size,
                    self.num_channels_latents,
                    (num_frames - 1) // 4 + 1,
                    latent_height,
                    latent_width,
                ),
                dtype=torch.float32,
                generator=generator,
                device=self.device,
            )

            if not self.scheduler:
                self.load_component_by_type("scheduler")
                self.to_device(self.scheduler)

            scheduler = self.scheduler

            input_timesteps, _ = self._get_timesteps(
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                timesteps_as_indices=timesteps_as_indices,
            )

            if not is_first_clip:
                latent_motion_frames = latent_motion_frames.to(latents.dtype).to(
                    self.device
                )
                motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                add_latent = scheduler.add_noise(
                    latent_motion_frames,
                    motion_add_noise,
                    input_timesteps[0].expand(latents.shape[0]),
                )
                C, T_m, H, W = add_latent.shape
                latents[:, :C, :T_m, :H, :W] = add_latent

            latents = self.denoise(
                latent_motion_frames=latent_motion_frames,
                using_video_input=using_video_input,
                cur_motion_frames_latent_num=cur_motion_frames_latent_num,
                timesteps=input_timesteps,
                latents=latents,
                latent_condition=latent_condition,
                image_embeds=image_embeds.to(self.device).to(transformer_dtype),
                audio_embeds=audio_embs.to(self.device).to(transformer_dtype),
                prompt_embeds=prompt_embeds.to(self.device).to(transformer_dtype),
                negative_prompt_embeds=negative_prompt_embeds.to(self.device).to(
                    transformer_dtype
                ),
                ref_target_masks=ref_target_masks,
                transformer_dtype=transformer_dtype,
                use_cfg_guidance=use_cfg_guidance,
                render_on_step=render_on_step,
                render_on_step_callback=render_on_step_callback,
                scheduler=scheduler,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
            )

            videos = self.vae_decode(latents, offload=offload)

            # >>> START OF COLOR CORRECTION STEP <<<
            if color_correction_strength > 0.0 and original_color_reference is not None:
                videos = match_and_blend_colors(
                    videos.float(),
                    original_color_reference.float(),
                    color_correction_strength,
                )
            # >>> END OF COLOR CORRECTION STEP <<<

            if is_first_clip:
                gen_video_list.append(videos)
            else:
                gen_video_list.append(videos[:, :, cur_motion_frames_num:])

            # decide whether is done
            if arrive_last_frame:
                break

            # update next condition frames
            is_first_clip = False
            cur_motion_frames_num = motion_frames
            cond_frame = (
                videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)
            )

            if video is None:
                loaded_image = cond_frame[:, :, -1, :, :]
            else:
                # Clamp index to available frames
                next_src_idx = min(audio_start_idx, input_video.shape[2] - 1)
                loaded_image = input_video[:, :, next_src_idx, :, :]

            loaded_image = loaded_image.squeeze(0).permute(1, 2, 0)

            loaded_image = Image.fromarray(
                ((loaded_image + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
            )
            # Update cond_image from source video for the next iteration in video mode
            if using_video_input:
                cond_idx = min(audio_start_idx, input_video.shape[2] - 1)
                cond_image = input_video[:, :, cond_idx : cond_idx + 1, :, :]
            audio_start_idx += num_frames - cur_motion_frames_num
            audio_end_idx = audio_start_idx + clip_length
            miss_lengths = []

            if audio_end_idx >= min(max_num_frames, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                source_frames = []
                for human_inx in range(human_num):
                    source_frame = len(full_audio_embs[human_inx])
                    source_frames.append(source_frame)
                    if audio_end_idx >= len(full_audio_embs[human_inx]):
                        miss_length = (
                            audio_end_idx - len(full_audio_embs[human_inx]) + 3
                        )
                        add_audio_emb = torch.flip(
                            full_audio_embs[human_inx][-1 * miss_length :], dims=[0]
                        )
                        full_audio_embs[human_inx] = torch.cat(
                            [full_audio_embs[human_inx], add_audio_emb], dim=0
                        )
                        miss_lengths.append(miss_length)
                    else:
                        miss_lengths.append(0)

            if max_num_frames <= num_frames:
                break

        if offload:
            self._offload(self.transformer)

        if return_latents:
            latents = torch.cat(gen_latents_list, dim=2).cpu()
            return latents
        else:
            # postprocess
            gen_video_samples = torch.cat(gen_video_list, dim=2)[
                :, :, : int(max_num_frames)
            ]
            if max_num_frames > num_frames and sum(miss_lengths) > 0:
                # split video frames
                if using_video_input:
                    gen_video_samples = gen_video_samples[
                        :, :, : full_audio_embs[0].shape[0]
                    ]
                else:
                    gen_video_samples = gen_video_samples[:, :, : -1 * miss_lengths[0]]

            postprocessed_video = self._tensor_to_frames(gen_video_samples)
            return postprocessed_video
