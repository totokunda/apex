from src.engine.base_engine import BaseEngine
import torch
from typing import Dict, Any, Callable
from enum import Enum
from src.ui.nodes import UINode
from typing import List
from diffusers.video_processor import VideoProcessor
import math
from PIL import Image
import numpy as np
from typing import Union
from src.mixins import OffloadMixin
import torchvision.transforms.functional as TF
from src.engine.denoise.wan_denoise import WanDenoise, DenoiseType


class ModelType(Enum):
    VACE = "vace"  # vace
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    FFLF = "fflf"  # first frame last frame
    CAUSAL = "causal"  # causal


class WanEngine(BaseEngine, WanDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: DenoiseType = DenoiseType.BASE,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        self.denoise_type = denoise_type

        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.temperal_downsample)
            if getattr(self.vae, "temperal_downsample", None)
            else 8
        )
        self.num_channels_latents = getattr(self.vae, "config", {}).get("z_dim", 16)
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
        elif self.model_type == ModelType.VACE:
            return self.vace_run(**final_kwargs)
        elif self.model_type == ModelType.I2V:
            return self.i2v_run(**final_kwargs)
        elif self.model_type == ModelType.FFLF:
            return self.fflf_run(**final_kwargs)
        elif self.model_type == ModelType.CAUSAL:
            return self.causal_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def t2v_run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 16,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
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
        **kwargs,
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

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)
        scheduler = self.scheduler

        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )
        timesteps = self._get_timesteps(timesteps, timesteps_as_indices)

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

        latents = self.denoise(
            timesteps=timesteps,
            latents=latents,
            latent_condition=None,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
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
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video

    def vace_run(
        self,
        video: Union[List[Image.Image], List[str], str, np.ndarray, torch.Tensor],
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        reference_images: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ] = None,
        mask: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor, None
        ] = None,
        conditioning_scale: Union[float, List[float], torch.Tensor] = 1.0,
        height: int = 480,
        width: int = 832,
        duration: int | str | None = None,
        fps: int = 16,
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
        return_latents: bool = False,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        **kwargs,
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

        if not self.transformer:
            self.load_component_by_type("transformer")

        pt, ph, pw = self.transformer.config.patch_size
        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        self.to_device(self.scheduler)
        scheduler = self.scheduler
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )
        timesteps = self._get_timesteps(timesteps, timesteps_as_indices)

        loaded_video = self._load_video(video)
        if mask:
            loaded_mask = self._load_video(mask)

        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [conditioning_scale] * len(
                self.transformer.config.vace_layers
            )
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {len(conditioning_scale)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if isinstance(conditioning_scale, torch.Tensor):
            if conditioning_scale.size(0) != len(self.transformer.config.vace_layers):
                raise ValueError(
                    f"Length of `conditioning_scale` {conditioning_scale.size(0)} does not match number of layers {len(self.transformer.config.vace_layers)}."
                )
            conditioning_scale = conditioning_scale.to(
                device=self.device, dtype=transformer_dtype
            )

        video_height, video_width = self.video_processor.get_default_height_width(
            loaded_video[0]
        )
        base = self.vae_scale_factor_spatial * ph
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

        if not mask:
            preprocessed_mask = torch.ones_like(preprocessed_video)
        else:
            preprocessed_mask = self.video_processor.preprocess_video(
                loaded_mask, video_height, video_width
            )
            preprocessed_mask = torch.clamp((preprocessed_mask + 1) / 2, min=0, max=1)

        if reference_images is None or not isinstance(reference_images, (list, tuple)):
            if reference_images is not None:
                reference_images = self._load_image(reference_images)
            reference_images = [
                [reference_images] for _ in range(preprocessed_video.shape[0])
            ]
        elif isinstance(reference_images, (list, tuple)) and isinstance(
            next(iter(reference_images)), list
        ):
            reference_images = [
                [
                    self._load_image(image)
                    for image in reference_images_batch
                    if image is not None
                ]
                for reference_images_batch in reference_images
            ]
        elif isinstance(reference_images, (list, tuple)):
            reference_images = [
                self._load_image(image)
                for image in reference_images
                if image is not None
            ]
            reference_images = [
                reference_images if reference_images else None
                for _ in range(preprocessed_video.shape[0])
            ]

        assert reference_images is not None, "reference_images must be provided"
        assert isinstance(reference_images, list), "reference_images must be a list"
        assert (
            len(reference_images) == preprocessed_video.shape[0]
        ), "reference_images must be a list of the same length as the video"

        reference_images_preprocessed = []
        for i, reference_images_batch in enumerate(reference_images):
            preprocessed_images = []
            for j, image in enumerate(reference_images_batch):
                if image is None:
                    continue
                image = self.video_processor.preprocess(image, None, None)
                img_height, img_width = image.shape[-2:]
                scale = min(height / img_height, width / img_width)
                new_height, new_width = int(img_height * scale), int(img_width * scale)
                resized_image = torch.nn.functional.interpolate(
                    image,
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(
                    0
                )  # [C, H, W]
                top = (height - new_height) // 2
                left = (width - new_width) // 2
                canvas = torch.ones(
                    3, height, width, device=self.device, dtype=torch.float32
                )
                canvas[:, top : top + new_height, left : left + new_width] = (
                    resized_image
                )
                preprocessed_images.append(canvas)
            reference_images_preprocessed.append(preprocessed_images)

        num_reference_images = len(reference_images_preprocessed[0])

        mask = torch.where(preprocessed_mask > 0.5, 1.0, 0.0)
        inactive = preprocessed_video * (1 - mask)
        reactive = preprocessed_video * mask

        inactive = self.vae_encode(inactive, offload=offload, dtype=torch.float32)
        reactive = self.vae_encode(reactive, offload=offload, dtype=torch.float32)

        latents = torch.cat([inactive, reactive], dim=1)

        latent_list = []
        for latent, reference_images_batch in zip(
            latents, reference_images_preprocessed
        ):
            for reference_image in reference_images_batch:
                assert reference_image.ndim == 3
                reference_image = reference_image.to(dtype=self.vae.dtype)
                reference_image = reference_image[
                    None, :, None, :, :
                ]  # [1, C, 1, H, W]
                reference_latent = self.vae_encode(
                    reference_image, offload=offload, dtype=torch.float32
                )
                reference_latent = reference_latent.squeeze(0)  # [C, 1, H, W]
                reference_latent = torch.cat(
                    [reference_latent, torch.zeros_like(reference_latent)], dim=0
                )
                latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
            latent_list.append(latent)

        mask_list = []

        for mask_, reference_images_batch in zip(
            preprocessed_mask, reference_images_preprocessed
        ):
            num_channels, num_frames, height, width = mask_.shape
            new_num_frames = (
                num_frames + self.vae_scale_factor_temporal - 1
            ) // self.vae_scale_factor_temporal

            new_height = height // (self.vae_scale_factor_spatial * ph) * ph
            new_width = width // (self.vae_scale_factor_spatial * ph) * ph
            mask_ = mask_[0, :, :, :]
            mask_ = mask_.view(
                num_frames,
                new_height,
                self.vae_scale_factor_spatial,
                new_width,
                self.vae_scale_factor_spatial,
            )
            mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(
                0, 1
            )  # [8x8, num_frames, new_height, new_width]
            mask_ = torch.nn.functional.interpolate(
                mask_.unsqueeze(0),
                size=(new_num_frames, new_height, new_width),
                mode="nearest-exact",
            ).squeeze(0)
            num_ref_images = len(reference_images_batch)
            if num_ref_images > 0:
                mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
                mask_ = torch.cat([mask_padding, mask_], dim=1)
            mask_list.append(mask_)

        conditioning_latents = torch.stack(latent_list, dim=0).to(self.device)
        conditioning_masks = torch.stack(mask_list, dim=0).to(self.device)

        conditioning_latents = torch.cat(
            [conditioning_latents, conditioning_masks], dim=1
        )
        conditioning_latents = conditioning_latents.to(transformer_dtype)
        prompt_embeds = prompt_embeds.to(transformer_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if duration is None:
            duration = len(loaded_video)

        num_frames = self._parse_num_frames(duration, fps=fps)
        latents = self._get_latents(
            height,
            width,
            num_frames + num_reference_images * self.vae_scale_factor_temporal,
            num_videos=num_videos,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
        )

        if latents.shape[2] != conditioning_latents.shape[2]:
            self.logger.warning(
                "The number of frames in the conditioning latents does not match the number of frames to be generated. Generation quality may be affected."
            )

        latents = self.denoise(
            timesteps=timesteps,
            latents=latents,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                control_hidden_states=conditioning_latents,
                control_hidden_states_scale=conditioning_scale,
                attention_kwargs=attention_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    control_hidden_states=conditioning_latents,
                    control_hidden_states_scale=conditioning_scale,
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
        image: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        duration: int | str = 16,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        fps: int = 16,
        guidance_scale: float = 5.0,
        use_cfg_guidance: bool = True,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable = None,
        generator: torch.Generator | None = None,
        offload: bool = True,
        render_on_step: bool = False,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
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

        if not self.preprocessors or "clip" not in self.preprocessors:
            self.load_preprocessor_by_type("clip")

        self.to_device(self.preprocessors["clip"])

        loaded_image = self._load_image(image)

        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image, max_area=height * width
        )

        preprocessed_image = self.video_processor.preprocess(
            loaded_image, height=height, width=width
        ).to(self.device, dtype=torch.float32)

        if not self.transformer:
            self.load_component_by_type("transformer")

        transformer_dtype = self.component_dtypes["transformer"]

        self.to_device(self.transformer)

        image_embeds = self.preprocessors["clip"](
            loaded_image, hidden_states_layer=-2
        ).to(self.device, dtype=transformer_dtype)
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if offload:
            self._offload(self.preprocessors["clip"])

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        scheduler = self.scheduler
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )
        timesteps = self._get_timesteps(timesteps, timesteps_as_indices)
        num_frames = self._parse_num_frames(duration, fps)

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
            video_condition, offload=offload, dtype=latents.dtype
        )
        batch_size, _, _, latent_height, latent_width = latents.shape

        mask_lat_size = torch.ones(
            batch_size, 1, num_frames, latent_height, latent_width, device=self.device
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
            batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width
        )

        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latents.device)

        latent_condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        latents = self.denoise(
            timesteps=timesteps,
            latents=latents,
            latent_condition=latent_condition,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
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
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video

    def fflf_run(
        self,
        first_frame: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
        last_frame: Union[
            Image.Image, List[Image.Image], List[str], str, np.ndarray, torch.Tensor
        ],
        prompt: List[str] | str,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
        negative_prompt: List[str] | str = None,
        duration: int | str = 16,
        num_inference_steps: int = 30,
        num_videos: int = 1,
        seed: int | None = None,
        guidance_scale: float = 5.0,
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
        **kwargs,
    ):
        """
        First Frame Last Frame generation method.
        Generates video content conditioned on both first and last frames.
        """

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

        if not self.preprocessors or "clip" not in self.preprocessors:
            self.load_preprocessor_by_type("clip")

        self.to_device(self.preprocessors["clip"])

        loaded_first_frame = self._load_image(first_frame)
        loaded_last_frame = self._load_image(last_frame)

        loaded_first_frame, height, width = self._aspect_ratio_resize(
            loaded_first_frame, max_area=height * width
        )
        loaded_last_frame, height, width = self._center_crop_resize(
            loaded_last_frame, height, width
        )

        preprocessed_first_frame = self.video_processor.preprocess(
            loaded_first_frame, height=height, width=width
        ).to(self.device, dtype=torch.float32)

        preprocessed_last_frame = self.video_processor.preprocess(
            loaded_last_frame, height=height, width=width
        ).to(self.device, dtype=torch.float32)

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        transformer_dtype = self.component_dtypes["transformer"]

        image_embeds = self.preprocessors["clip"](
            [loaded_first_frame, loaded_last_frame], hidden_states_layer=-2
        ).to(self.device, dtype=transformer_dtype)
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if offload:
            self._offload(self.preprocessors["clip"])

        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        scheduler = self.scheduler
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000, device=self.device
        )
        timesteps = self._get_timesteps(timesteps, timesteps_as_indices)
        num_frames = self._parse_num_frames(duration, fps)

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

        if preprocessed_first_frame.ndim == 4:
            preprocessed_first_frame = preprocessed_first_frame.unsqueeze(2)

        if preprocessed_last_frame.ndim == 4:
            preprocessed_last_frame = preprocessed_last_frame.unsqueeze(2)

        video_condition = torch.cat(
            [
                preprocessed_first_frame,
                preprocessed_last_frame.new_zeros(
                    preprocessed_last_frame.shape[0],
                    preprocessed_last_frame.shape[1],
                    num_frames - 2,
                    height,
                    width,
                ),
                preprocessed_last_frame,
            ],
            dim=2,
        )

        latent_condition = self.vae_encode(
            video_condition, offload=offload, sample_mode="mode", dtype=latents.dtype
        )

        batch_size, _, _, latent_height, latent_width = latents.shape

        mask_lat_size = torch.ones(
            batch_size, 1, num_frames, latent_height, latent_width, device=self.device
        )
        mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal
        )
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
        )
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width
        )

        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latents.device)

        latent_condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        latents = self.denoise(
            timesteps=timesteps,
            latents=latents,
            latent_condition=latent_condition,
            transformer_kwargs=dict(
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
            ),
            unconditional_transformer_kwargs=(
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
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
        )

        if offload:
            self._offload(self.transformer)

        if return_latents:
            return latents
        else:
            video = self.vae_decode(latents, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video

    def causal_run(
        self,
        prompt: List[str] | str,
        negative_prompt: List[str] | str | None = None,
        image: Union[
            Image.Image,
            List[Image.Image],
            List[str],
            str,
            np.ndarray,
            torch.Tensor,
            None,
        ] = None,
        video: Union[
            List[Image.Image], List[str], str, np.ndarray, torch.Tensor, None
        ] = None,
        height: int = 480,
        width: int = 832,
        duration: int | str = 16,
        fps: int = 16,
        num_videos: int = 1,
        seed: int | None = None,
        use_cfg_guidance: bool = False,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step: bool = False,
        render_on_step_callback: Callable = None,
        num_frame_per_block: int = 3,
        context_noise: float = 0.0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        offload: bool = True,
        num_inference_steps: int = 4,
        timesteps: List[int] | None = None,
        encoder_cache_size: int = 512,
        generator: torch.Generator | None = None,
        timesteps_as_indices: bool = True,
        **kwargs,
    ):

        kv_cache1 = []

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

        latent_image = None
        latent_video = None
        num_input_frames = 0

        if image is not None:
            loaded_image = self._load_image(image)
            loaded_image, height, width = self._aspect_ratio_resize(
                loaded_image, max_area=height * width
            )

            preprocessed_image = self.video_processor.preprocess(
                loaded_image, height=height, width=width
            ).to(self.device, dtype=torch.float32)

            latent_image = self.vae_encode(
                preprocessed_image.unsqueeze(2), offload=offload, sample_mode="mode"
            )
            num_input_frames = latent_image.shape[2]
        elif video is not None:
            loaded_video = self._load_video(video)
            preprocessed_video = self.video_processor.preprocess_video(
                loaded_video, height=height, width=width
            ).to(self.device, dtype=torch.float32)
            latent_video = self.vae_encode(
                preprocessed_video, offload=offload, sample_mode="mode"
            )
            num_input_frames = latent_video.shape[2]

        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)

        transformer_dtype = self.component_dtypes["transformer"]

        latents, generator = self._get_latents(
            height,
            width,
            duration,
            fps=fps,
            num_videos=num_videos,
            seed=seed,
            dtype=torch.float32,
            generator=generator,
            return_generator=True,
        )

        batch_size, num_channels, latent_frames, latent_height, latent_width = (
            latents.shape
        )

        # Initialize the KV cache
        head_dim = (
            self.transformer.blocks[0].attn1.inner_dim
            // self.transformer.blocks[0].attn1.heads
        )
        heads = self.transformer.blocks[0].attn1.heads
        pt, ph, pw = self.transformer.config.patch_size
        frame_seq_length = latent_height // ph * latent_width // pw
        if local_attn_size == -1:
            kv_cache_size = (
                (latent_frames + num_input_frames) // pt
            ) * frame_seq_length
        else:
            kv_cache_size = frame_seq_length * local_attn_size

        for _ in range(len(self.transformer.blocks)):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, heads, head_dim],
                        dtype=transformer_dtype,
                        device=self.device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, heads, head_dim],
                        dtype=transformer_dtype,
                        device=self.device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=self.device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=self.device
                    ),
                }
            )

        crossattn_cache = []
        for _ in range(len(self.transformer.blocks)):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, encoder_cache_size, heads, head_dim],
                        dtype=transformer_dtype,
                        device=self.device,
                    ),
                    "v": torch.zeros(
                        [batch_size, encoder_cache_size, heads, head_dim],
                        dtype=transformer_dtype,
                        device=self.device,
                    ),
                    "is_init": False,
                }
            )

        if not self.scheduler:
            self.load_component_by_type("scheduler")

        scheduler = self.scheduler
        scheduler.set_timesteps(
            num_inference_steps if timesteps is None else 1000,
            training=True,
            device=self.device,
        )

        timesteps = self._get_timesteps(timesteps, timesteps_as_indices)

        num_output_frames = num_input_frames + latent_frames
        num_blocks = latent_frames // num_frame_per_block
        all_num_frames = [num_frame_per_block] * num_blocks
        current_start_frame = 0
        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, latent_height, latent_width],
            dtype=latents.dtype,
            device=self.device,
        )

        if latent_image is not None or latent_video is not None:
            timestep = (
                torch.ones([batch_size, 1], device=self.device, dtype=torch.int64) * 0
            )
            if latent_image is not None:
                initial_latent = latent_image
                output[:, :, :num_input_frames, :, :] = initial_latent[
                    :, :, :num_input_frames, :, :
                ]
                latent_model_input = initial_latent.to(transformer_dtype)
                assert (num_input_frames - 1) % num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // num_frame_per_block

                self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    return_dict=False,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                    attention_kwargs=attention_kwargs,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                )
                if render_on_step:
                    render_on_step_callback(latent_model_input)

                current_start_frame += 1
            else:
                assert num_input_frames % num_frame_per_block == 0
                num_input_blocks = num_input_frames // num_frame_per_block
                initial_latent = latent_video

            with self._progress_bar(
                total=num_input_blocks,
                desc="Caching Input Frames",
                disable=num_input_blocks == 0,
            ) as pbar:
                for _ in range(num_input_blocks):
                    current_ref_latents = initial_latent[
                        :,
                        :,
                        current_start_frame : current_start_frame + num_frame_per_block,
                        :,
                        :,
                    ]
                    output[
                        :,
                        :,
                        current_start_frame : current_start_frame + num_frame_per_block,
                        :,
                        :,
                    ] = current_ref_latents

                    latent_model_input = current_ref_latents.to(transformer_dtype)

                    self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timestep * 0,
                        kv_cache=kv_cache1,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                        attention_kwargs=attention_kwargs,
                        local_attn_size=local_attn_size,
                        sink_size=sink_size,
                    )

                    pbar.update(1)

                    if render_on_step:
                        render_on_step_callback(current_ref_latents)

                    current_start_frame += num_frame_per_block

        with self._progress_bar(
            total=len(all_num_frames) * len(timesteps), desc="Causal Generation"
        ) as pbar:
            for current_num_frames in all_num_frames:
                latent = latents[
                    :,
                    :,
                    current_start_frame
                    - num_input_frames : current_start_frame
                    + current_num_frames
                    - num_input_frames,
                    :,
                    :,
                ]
                for i, t in enumerate(timesteps):
                    timestep = (
                        torch.ones(
                            [batch_size, current_num_frames],
                            device=self.device,
                            dtype=torch.int64,
                        )
                        * t
                    )
                    latent_model_input = latent.to(transformer_dtype)

                    if i < len(timesteps) - 1:
                        flow_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            current_start=current_start_frame * frame_seq_length,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            return_dict=False,
                            attention_kwargs=attention_kwargs,
                            local_attn_size=local_attn_size,
                            sink_size=sink_size,
                        )[0]

                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)

                        pred_x0 = self.scheduler.convert_flow_pred_to_x0(
                            flow_pred.flatten(0, 1),
                            latent_model_input.permute(0, 2, 1, 3, 4).flatten(0, 1),
                            timestep.flatten(0, 1),
                        ).unflatten(0, flow_pred.shape[:2])

                        next_timestep = timesteps[i + 1]

                        latent = self.scheduler.add_noise(
                            pred_x0.flatten(0, 1),
                            torch.randn_like(pred_x0.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=self.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, flow_pred.shape[:2])
                        latent = latent.permute(0, 2, 1, 3, 4)

                    else:
                        flow_pred = self.transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            current_start=current_start_frame * frame_seq_length,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            return_dict=False,
                            attention_kwargs=attention_kwargs,
                            local_attn_size=local_attn_size,
                            sink_size=sink_size,
                        )[0]
                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                        pred_x0 = self.scheduler.convert_flow_pred_to_x0(
                            flow_pred.flatten(0, 1),
                            latent_model_input.permute(0, 2, 1, 3, 4).flatten(0, 1),
                            timestep.flatten(0, 1),
                        ).unflatten(0, flow_pred.shape[:2])
                        latent = pred_x0.permute(0, 2, 1, 3, 4)

                    if render_on_step:
                        render_on_step_callback(latent)
                    pbar.update(1)

                output[
                    :,
                    :,
                    current_start_frame : current_start_frame + current_num_frames,
                    :,
                    :,
                ] = latent

                context_timestep = torch.ones_like(timestep) * context_noise
                # Clean context cache
                self.transformer(
                    hidden_states=latent,
                    encoder_hidden_states=prompt_embeds,
                    timestep=context_timestep,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                    attention_kwargs=attention_kwargs,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    return_dict=False,
                )
                current_start_frame += current_num_frames

        if return_latents:
            return output
        else:
            video = self.vae_decode(output, offload=offload)
            postprocessed_video = self._postprocess(video)
            return postprocessed_video

    def __str__(self):
        return f"WanEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    from diffusers.utils import export_to_video
    from PIL import Image

    engine = WanEngine(
        yaml_path="manifest/wan_t2v_sf_1.3b.yml",
        model_type=ModelType.CAUSAL,
        save_path="/Users/tosinkuye/apex-models",  # Change this to your desired save path,  # Change this to your desired save path
        components_to_load=["transformer"],
        component_dtypes={"vae": torch.float16},
    )

    prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    # negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    height = 480
    width = 832

    video = engine.run(
        height=height,
        width=width,
        prompt=prompt,
        # negative_prompt=negative_prompt,
        use_cfg_guidance=False,
        duration="5s",
        num_videos=1,
        guidance_scale=5.0,
        seed=420,
    )

    export_to_video(video[0], "t2v_1.3b_sf_420.mp4", fps=16, quality=8)
