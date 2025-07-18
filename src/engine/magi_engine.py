from src.engine.base_engine import BaseEngine
import torch
from typing import Dict, Any, Callable, List, Union, Optional, Tuple
from enum import Enum
from src.ui.nodes import UINode
from diffusers.video_processor import VideoProcessor
from PIL import Image
import numpy as np
from src.engine.denoise import MagiDenoise, MagiDenoiseType
import math

class ModelType(Enum):
    T2V = "t2v"  # text to video
    I2V = "i2v"  # image to video
    V2V = "v2v"  # video to video


class MagiEngine(BaseEngine, MagiDenoise):
    def __init__(
        self,
        yaml_path: str,
        model_type: ModelType = ModelType.T2V,
        denoise_type: MagiDenoiseType = MagiDenoiseType.T2V,
        **kwargs,
    ):
        super().__init__(yaml_path, **kwargs)

        self.model_type = model_type
        self.denoise_type = denoise_type

        # Set up VAE scale factors based on MAGI VAE configuration
        self.vae_scale_factor_temporal = (
            getattr(self.vae, "temporal_compression_ratio", None) or 
            getattr(self.vae, "patch_length", None) or 4
            if getattr(self, "vae", None)
            else 4
        )
        self.vae_scale_factor_spatial = (
            getattr(self.vae, "spatial_compression_ratio", None) or 
            getattr(self.vae, "patch_size", None) or 8
            if getattr(self, "vae", None)
            else 8
        )
        
        # MAGI uses different channel configurations
        self.num_channels_latents = getattr(self.vae, "config", {}).get(
            "z_chans", 4  # MAGI default
        )
        
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

    def init_timestep_schedule(
        self, 
        num_steps: int, 
        device: torch.device, 
        transform_type: str = "sd3", 
        shift: float = 3.0
    ) -> torch.Tensor:
        """Initialize timestep schedule following MAGI's approach"""
        
        if num_steps == 12:
            # MAGI's specific 12-step schedule
            base_t = torch.linspace(0, 1, 4 + 1, device=device) / 4
            accu_num = torch.linspace(0, 1, 4 + 1, device=device)
            base_t = torch.cat([base_t[:1], base_t[2:4]], dim=0)
            t = torch.cat([base_t + accu for accu in accu_num], dim=0)[:num_steps + 1]
        else:
            t = torch.linspace(0, 1, num_steps + 1, device=device)

        # Apply transform
        if transform_type == "sd3":
            def t_resolution_transform(x, shift=3.0):
                assert shift >= 1.0, "shift should >=1"
                shift_inv = 1.0 / shift
                return shift_inv * x / (1 + (shift_inv - 1) * x)
            
            t = t**2
            t = t_resolution_transform(t, shift)
        elif transform_type == "square":
            t = t**2
        elif transform_type == "piecewise":
            def t_transform(x):
                mask = x < 0.875
                x[mask] = x[mask] * (0.5 / 0.875)
                x[~mask] = 0.5 + (x[~mask] - 0.875) * (0.5 / (1 - 0.875))
                return x
            t = t_transform(t)
        # else: identity transform

        return t

    def process_text_embeddings(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_attention_mask: Optional[torch.Tensor],
        infer_chunk_num: int,
        clean_chunk_num: int,
        special_token_kwargs: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process text embeddings following MAGI's special token approach"""
        
        # Get null embedding from transformer (if available)
        null_emb = getattr(self.transformer, 'y_embedder', None)
        if null_emb and hasattr(null_emb, 'null_caption_embedding'):
            null_caption_embedding = null_emb.null_caption_embedding.unsqueeze(0)
        else:
            # Create a dummy null embedding if not available
            null_caption_embedding = torch.zeros_like(prompt_embeds[:1])

        # Expand embeddings for chunks
        denoise_chunks = infer_chunk_num - clean_chunk_num
        
        # Denoise chunks with caption_embs
        caption_embs = prompt_embeds.repeat(1, denoise_chunks, 1, 1)
        emb_masks = prompt_attention_mask.unsqueeze(1).repeat(1, denoise_chunks, 1) if prompt_attention_mask is not None else None

        # Apply special tokens if specified (simplified version of MAGI's special token logic)
        if special_token_kwargs.get("use_special_tokens", False):
            caption_embs, emb_masks = self._pad_special_tokens(caption_embs, emb_masks, special_token_kwargs)

        # Clean chunks with null_emb
        caption_embs = torch.cat([null_caption_embedding.repeat(1, clean_chunk_num, 1, 1), caption_embs], dim=1)
        if emb_masks is not None:
            emb_masks = torch.cat([
                torch.zeros(1, clean_chunk_num, emb_masks.size(2), dtype=emb_masks.dtype, device=emb_masks.device),
                emb_masks
            ], dim=1)

        # Handle CFG by concatenating conditional and unconditional
        if negative_prompt_embeds is not None:
            # Process negative embeddings similarly
            neg_emb_expanded = negative_prompt_embeds.repeat(1, denoise_chunks, 1, 1)
            neg_mask_expanded = negative_prompt_attention_mask.unsqueeze(1).repeat(1, denoise_chunks, 1) if negative_prompt_attention_mask is not None else None
            
            neg_emb_expanded = torch.cat([null_caption_embedding.repeat(1, clean_chunk_num, 1, 1), neg_emb_expanded], dim=1)
            if neg_mask_expanded is not None:
                neg_mask_expanded = torch.cat([
                    torch.zeros(1, clean_chunk_num, neg_mask_expanded.size(2), dtype=neg_mask_expanded.dtype, device=neg_mask_expanded.device),
                    neg_mask_expanded
                ], dim=1)

            # Concatenate for CFG
            caption_embs = torch.cat([caption_embs, neg_emb_expanded], dim=0)
            if emb_masks is not None and neg_mask_expanded is not None:
                emb_masks = torch.cat([emb_masks, neg_mask_expanded], dim=0)

        return caption_embs, emb_masks

    def _pad_special_tokens(self, caption_embs: torch.Tensor, emb_masks: torch.Tensor, special_token_kwargs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified special token padding (can be expanded based on specific needs)"""
        # This is a placeholder for MAGI's complex special token logic
        # In practice, you would implement the full special token system here
        return caption_embs, emb_masks

    @torch.inference_mode()
    def run(
        self,
        input_nodes: List[UINode] = None,
        **kwargs,
    ):
        """Main run method that routes to appropriate generation function"""
        default_kwargs = self._get_default_kwargs("run")
        preprocessed_kwargs = self._preprocess_kwargs(input_nodes, **kwargs)
        final_kwargs = {**default_kwargs, **preprocessed_kwargs}
        
        if self.model_type == ModelType.T2V:
            return self.t2v_run(**final_kwargs)
        elif self.model_type == ModelType.I2V:
            return self.i2v_run(**final_kwargs)
        elif self.model_type == ModelType.V2V:
            return self.v2v_run(**final_kwargs)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def t2v_run(
        self,
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
        **kwargs,
    ):
        """Text-to-video generation using MAGI's chunk-based approach"""

        # 1. Encode prompts
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        
        self.to_device(self.text_encoder)
        
        # MAGI uses a different text encoder (T5-based)
        prompt_embeds = self.text_encoder.encode(
            prompt,
            device=self.device,
            num_videos_per_prompt=num_videos,
            **text_encoder_kwargs,
        )
        
        prompt_attention_mask = None  # MAGI handles masking differently
        
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

        # 2. Load transformer
        if not self.transformer:
            self.load_component_by_type("transformer")

        self.to_device(self.transformer)
        transformer_dtype = self.component_dtypes["transformer"]

        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        # 3. Prepare latents
        num_frames = self._parse_num_frames(duration, fps)
        
        # MAGI uses latent frames corresponding to chunks
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
            parse_frames=False,  # Already calculated
        )

        # 4. Load scheduler
        if not self.scheduler:
            self.load_component_by_type("scheduler")
        self.to_device(self.scheduler)

        # 5. Initialize timestep schedule
        timesteps = self.init_timestep_schedule(
            num_steps=num_inference_steps,
            device=self.device,
            transform_type=timestep_transform,
            shift=timestep_shift
        )

        # 6. Process text embeddings for chunk-based generation
        # Calculate chunking parameters first
        total_latent_frames = latents.shape[2]
        infer_chunk_num = math.ceil(total_latent_frames / chunk_size)
        clean_chunk_num = 0  # No prefix video in T2V

        processed_caption_embs, processed_caption_masks = self.process_text_embeddings(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            infer_chunk_num=infer_chunk_num,
            clean_chunk_num=clean_chunk_num,
            special_token_kwargs=special_token_kwargs
        )

        # 7. MAGI chunk-based denoising
        latents = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            timesteps=timesteps,
            processed_caption_embs=processed_caption_embs,
            processed_caption_masks=processed_caption_masks,
            guidance_scale=guidance_scale,
            use_cfg_guidance=use_cfg_guidance,
            num_inference_steps=num_inference_steps,
            render_on_step=render_on_step,
            render_on_step_callback=render_on_step_callback,
            attention_kwargs=attention_kwargs,
            transformer_dtype=transformer_dtype,
            chunk_size=chunk_size,
            temporal_downsample_factor=self.vae_scale_factor_temporal,
            num_frames=num_frames,
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

    def i2v_run(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
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
        **kwargs,
    ):
        """Image-to-video generation using MAGI's chunk-based approach"""

        # 1. Process input image
        loaded_image = self._load_image(image)
        loaded_image, height, width = self._aspect_ratio_resize(
            loaded_image, max_area=height * width
        )

        # Preprocess image for VAE
        image_tensor = self.video_processor.preprocess(loaded_image, height, width).to(
            self.device
        )

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

        # 3. Encode image to latents (prefix video)
        image_tensor_unsqueezed = image_tensor.unsqueeze(2)  # Add temporal dimension
        prefix_video = self.vae_encode(
            image_tensor_unsqueezed,
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
            prefix_video=prefix_video,  # Key difference for I2V
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

    def v2v_run(
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

    def _load_video(self, video):
        """Load video from various input formats"""
        # Placeholder - implement based on your video loading utilities
        if isinstance(video, str):
            # Load from file path
            pass
        elif isinstance(video, list):
            # Load from list of images or paths
            pass
        elif isinstance(video, (np.ndarray, torch.Tensor)):
            # Already loaded
            pass
        # Add more loading logic as needed
        return video

    def _aspect_ratio_resize_video(self, video, max_area):
        """Resize video maintaining aspect ratio"""
        # Placeholder - implement based on your video processing utilities
        height, width = video.shape[-2:]  # Assuming THWC or similar format
        return video, height, width

    def __str__(self):
        return f"MagiEngine(config={self.config}, device={self.device}, model_type={self.model_type})"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    from diffusers.utils import export_to_video

    # Example usage for text-to-video
    engine = MagiEngine(
        yaml_path="manifest/magi_t2v.yml",  # You'll need to create this
        model_type=ModelType.T2V,
        save_path="./apex-models",
        components_to_load=["transformer", "text_encoder", "vae", "scheduler"],
        component_dtypes={"vae": torch.float16, "transformer": torch.float16},
    )

    prompt = "A serene sunset over a calm lake with gentle ripples"
    video = engine.run(
        height=512,
        width=512,
        duration=5,
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=6.0,
        fps=24,
        seed=42,
    )

    export_to_video(video[0], "magi_t2v_output.mp4", fps=24) 