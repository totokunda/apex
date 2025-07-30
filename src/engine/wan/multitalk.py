import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import WanBaseEngine
from torch.nn import functional as F
import math
from torchvision import transforms as T
from diffusers.utils.torch_utils import randn_tensor
from src.utils.color_utils import match_and_blend_colors    

class WanMultitalkEngine(WanBaseEngine):
    """WAN MultiTalk (Audio-driven) Engine Implementation"""

    def run(
        self,
        prompt: List[str] | str,
        image: Union[Image.Image, str],
        audio_paths: Optional[Dict[str, str]] = None,
        audio_type: str = "para",
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
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
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        offload: bool = True,
        render_on_step: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        face_scale: float = 0.05,
        color_correction_strength: float = 0.0,
        bbox: Optional[Dict[str, List[float]]] = None,
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
        """
        
        loaded_image = self._load_image(image)
        resized_image = self.resize_and_centercrop(loaded_image, (height, width))
        
        cond_image = self.video_processor.preprocess(
            resized_image, height, width
        ).unsqueeze(0)
        
        original_color_reference = None
        if color_correction_strength > 0.0:
            original_color_reference = cond_image.clone()
        
        
        if seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        
        if "wan.multitalk" not in self.preprocessors:
            self.load_preprocessor_by_type("wan.multitalk")
            
        preprocessor = self.preprocessors["wan.multitalk"]
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
        clip_length = self._parse_num_frames(duration, fps)
        cur_motion_frames_num = 1
        audio_start_idx = 0
        audio_end_idx = audio_start_idx + clip_length
        gen_video_list = []
        gen_latents_list = []
        is_first_clip = True
        
        transformer_dtype = self.component_dtypes["transformer"]
        transformer_config = self.load_config_by_type("transformer")
        
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
            
            
        
        while True:
            audio_embs = []
            # split audio with window size
            for human_idx in range(human_num):   
                center_indices = torch.arange(
                    audio_start_idx,
                    audio_end_idx,
                    1,
                ).unsqueeze(
                    1
                ) + indices.unsqueeze(0)
                center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0]-1)
                audio_emb = full_audio_embs[human_idx][center_indices][None,...].to(self.device)
                audio_embs.append(audio_emb)
            
            audio_embs = torch.cat(audio_embs, dim=0).to(transformer_dtype)

            h, w = cond_image.shape[-2], cond_image.shape[-1]
            lat_h, lat_w = h // self.vae_scale_factor_spatial, w // self.vae_scale_factor_spatial
        
            # get mask
            msk = torch.ones(1, num_frames, lat_h, lat_w, device=self.device)
            msk[:, cur_motion_frames_num:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
            ],
                            dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2).to(transformer_dtype) # B 4 T H W
            
            # get clip embedding
            clip_processor = self.load_preprocessor_by_type("clip")
            self.to_device(clip_processor)
            # get clip embedding
            image_embeds = clip_processor(cond_image[:, :, -1:, :, :]).to(transformer_dtype) 
            if offload:
                self._offload(clip_processor)
                
            # zero padding and vae encode
            video_frames = torch.zeros(1, cond_image.shape[1], num_frames-cond_image.shape[2], lat_h, lat_w).to(self.device)
            padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
            
                
            latent_condition = self.vae_encode(padding_frames_pixels_values, offload=offload) 
            latent_condition = torch.stack(latent_condition).to(transformer_dtype) # B C T H W
            cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num-1) // 4)
            latent_motion_frames = latent_condition[:, :, :cur_motion_frames_latent_num][0] # C T H W
            latent_condition = torch.concat([msk, latent_condition], dim=1) # B 4+C T H W
            
            # prepare masks
            ref_target_masks = self.resize_and_centercrop(human_masks, (height, width))
            ref_target_masks = F.interpolate(ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode='nearest').squeeze() 
            ref_target_masks = (ref_target_masks > 0) 
            ref_target_masks = ref_target_masks.float().to(self.device)
            
            # prepare noise
            latents = randn_tensor(
                (transformer_config.in_channels, (num_frames - 1) // 4 + 1,
                lat_h,
                lat_w),
                dtype=torch.float32,
                generator=generator,
                device=self.device
            ) 
            
            if not self.scheduler:
                self.load_component_by_type("scheduler")
                self.to_device(self.scheduler)
            
            scheduler = self.scheduler
            
            timesteps = self._get_timesteps(
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                timesteps_as_indices=timesteps_as_indices,
            )
            
            if not is_first_clip:
                latent_motion_frames = latent_motion_frames.to(latents.dtype).to(self.device)
                motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                add_latent = scheduler.add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                _, T_m, _, _ = add_latent.shape
                latents[:, :T_m] = add_latent
                    
            latents = self.denoise(
                timesteps=timesteps,
                latents=latents,
                latent_condition=latent_condition,
                transformer_kwargs=dict(
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    encoder_hidden_states_audio=audio_embs,
                    ref_target_masks=ref_target_masks,
                    human_num=human_num,
                    attention_kwargs=attention_kwargs,
                ),
                unconditional_transformer_kwargs=(
                    dict(
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        encoder_hidden_states_audio=(
                            torch.zeros_like(audio_embs)
                            if audio_embs is not None
                            else None
                        ),
                        ref_target_masks=ref_target_masks,
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
            )
            
            
            videos = self.vae_decode(latents, offload=offload).cpu()
        
            # >>> START OF COLOR CORRECTION STEP <<<
            if color_correction_strength > 0.0 and original_color_reference is not None:
                videos = match_and_blend_colors(videos, original_color_reference, color_correction_strength)
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

            cond_image = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)
            audio_start_idx += (num_frames - cur_motion_frames_num)
            audio_end_idx = audio_start_idx + clip_length

            # Repeat audio emb
            if audio_end_idx >= min(max_num_frames, len(full_audio_embs[0])):
                arrive_last_frame = True
                miss_lengths = []
                source_frames = []
                for human_inx in range(human_num):
                    source_frame = len(full_audio_embs[human_inx])
                    source_frames.append(source_frame)
                    if audio_end_idx >= len(full_audio_embs[human_inx]):
                        miss_length   = audio_end_idx - len(full_audio_embs[human_inx]) + 3 
                        add_audio_emb = torch.flip(full_audio_embs[human_inx][-1*miss_length:], dims=[0])
                        full_audio_embs[human_inx] = torch.cat([full_audio_embs[human_inx], add_audio_emb], dim=0)
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
            video = torch.cat(gen_video_list, dim=2).cpu()
            postprocessed_video = self._postprocess(video)
            return postprocessed_video

        