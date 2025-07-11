from enum import Enum
from src.engine.wan_engine import WanEngine
from src.engine.denoise.wan_denoise import DenoiseType
import torch
from typing import List, Union, Optional
from src.ui.nodes import UINode
from PIL import Image
from typing import Dict, Any, Callable
import numpy as np
import math
from copy import deepcopy


class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    DF = "df"  # diffusion forcing


class SkyReelsEngine(WanEngine):
    def __init__(self, yaml_path: str, model_type: ModelType = ModelType.T2V, **kwargs):
        if model_type == ModelType.DF:
            denoise_type = DenoiseType.DIFFUSION_FORCING
        else:
            denoise_type = DenoiseType.BASE
        super().__init__(
            yaml_path, model_type=model_type, denoise_type=denoise_type, **kwargs
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
            return self.t2v_run(fps=24, **final_kwargs)
        elif self.model_type == ModelType.I2V:
            return self.i2v_run(fps=24, **final_kwargs)
        elif self.model_type == ModelType.DF:
            return self.df_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    # Copied from https://github.com/SkyworkAI/SkyReels-V2/blob/main/skyreels_v2_infer/pipelines/diffusion_forcing_pipeline.py#L87
    def generate_timestep_matrix(
        self,
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        casual_block_size=1,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert (
                ar_step >= min_ar_step
            ), f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"

        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        while torch.all(pre_row >= (num_iterations - 1)) == False:
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            update_mask.append(
                (new_row != pre_row) & (new_row != num_iterations)
            )  # False: no need to update， True: need to update
            step_index.append(new_row)
            step_matrix.append(step_template[new_row])
            pre_row = new_row

        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append(
                (max(terminal_flag - base_num_frames_block, 0), terminal_flag)
            )

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = (
                step_update_mask.unsqueeze(-1)
                .repeat(1, 1, casual_block_size)
                .flatten(1)
                .contiguous()
            )
            step_index = (
                step_index.unsqueeze(-1)
                .repeat(1, 1, casual_block_size)
                .flatten(1)
                .contiguous()
            )
            step_matrix = (
                step_matrix.unsqueeze(-1)
                .repeat(1, 1, casual_block_size)
                .flatten(1)
                .contiguous()
            )
            valid_interval = [
                (s * casual_block_size, e * casual_block_size)
                for s, e in valid_interval
            ]

        return step_matrix, step_index, step_update_mask, valid_interval

    def df_run(
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
                preprocessed_image, offload=offload, dtype=torch.float32, normalize_latents_dtype=torch.float32
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
                preprocessed_end_image, offload=offload, dtype=torch.float32, normalize_latents_dtype=torch.float32
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
                            prefix_video, offload=offload, dtype=torch.float32, normalize_latents_dtype=torch.float32
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


if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    engine = SkyReelsEngine(
        yaml_path="manifest/skyreels_df_540p_1.3b.yml",
        model_type=ModelType.DF,
        attention_type="flash3",
        save_path="/mnt/localssd/apex-diffusion",
        components_to_load=["transformer"],
    )

    # prompt = "A graceful white swan with a curved neck and delicate feathers swimming in a serene lake at dawn, its reflection perfectly mirrored in the still water as mist rises from the surface, with the swan occasionally dipping its head into the water to feed."
    prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
    # negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    height = 544
    width = 960

    # image = Image.open("/mnt/filestore/apex-diffusion/kiss.jpg")

    video = engine.run(
        height=height,
        width=width,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_cfg_guidance=True,
        duration="257f",
        base_duration="97f",
        num_videos=1,
        seed=42,
        ar_step=0,
        overlap_history=17,
        causal_block_size=1,
        guidance_scale=6.0,
        num_inference_steps=30,
    )

    export_to_video(video[0], "skyreels_df_1.3b_async_10.mp4", fps=24, quality=8)
