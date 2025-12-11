import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from .shared import WanShared
from src.helpers.wan.lynx import WanLynxHelper
from src.types import InputImage
from src.utils.progress import make_mapped_progress, safe_emit_progress


class LynxEngine(WanShared):
    """Personalized Wan (Lynx) engine with IPA and Ref adapters."""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, auto_apply_loras=False, **kwargs)
        self._adapter_path = self._resolve_adapter_path()
        self._lynx_helper = WanLynxHelper(adapter_path=self._adapter_path)

    # --------------------------- Internal utilities --------------------------- #
    def _resolve_adapter_path(self, override: str | None = None) -> str | None:
        if override:
            return override
        env_path = os.getenv("LYNX_ADAPTER_PATH")
        if env_path:
            return env_path
        cfg = getattr(self, "config", {}) or {}
        transformer_component = next(
            (x for x in cfg.get("components", []) if x.get("type") == "transformer"),
            None,
        )
        extra_model_path = transformer_component.get("extra_model_paths", [])[0]
        return (
            extra_model_path
            or cfg.get("adapter_path")
            or cfg.get("lynx_adapter_path")
            or cfg.get("adapter_dir")
            or override
        )

    # ------------------------------- Main entry ------------------------------- #
    def run(
        self,
        subject_image: InputImage,
        prompt: List[str] | str,
        negative_prompt: List[str] | str = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        fps: int = 16,
        num_inference_steps: int = 50,
        num_videos: int = 1,
        seed: int | None = None,
        guidance_scale: float = 5.0,
        guidance_scale_i: float | None = 2.0,
        ip_scale: float = 1.0,
        ref_scale: float = 1.0,
        adapter_path: str | None = None,
        face_embeds: Optional[np.ndarray | torch.Tensor] = None,
        landmarks: Optional[np.ndarray | torch.Tensor] = None,
        return_latents: bool = False,
        text_encoder_kwargs: Dict[str, Any] = {},
        attention_kwargs: Dict[str, Any] = {},
        render_on_step_callback: Callable | None = None,
        render_on_step: bool = False,
        render_on_step_interval: int = 3,
        offload: bool = True,
        generator: torch.Generator | None = None,
        timesteps: List[int] | None = None,
        timesteps_as_indices: bool = True,
        progress_callback: Callable | None = None,
        denoise_progress_callback: Callable | None = None,
        output_type: str = "pil",
        **kwargs,
    ):
        safe_emit_progress(progress_callback, 0.0, "Starting Lynx pipeline")

        use_cfg_guidance = guidance_scale > 1.0 and negative_prompt is not None

        helper = self._lynx_helper
        adapter_root = helper.resolve_adapter_path(
            config=getattr(self, "config", {}),
            override=adapter_path or self._adapter_path,
        )
        embeds, face_landmarks, face_image = helper.prepare_face_data(
            subject_image,
            face_embeds,
            landmarks,
            device=self.device or "cuda",
            load_image_fn=self._load_image,
        )

        safe_emit_progress(progress_callback, 0.08, "Prepared face embeddings")

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            use_cfg_guidance=use_cfg_guidance,
            num_videos=num_videos,
            max_sequence_length=512,
            progress_callback=make_mapped_progress(progress_callback, 0.08, 0.18),
            text_encoder_kwargs=text_encoder_kwargs,
            offload=False,
        )

        batch_size = prompt_embeds.shape[0]

        if offload:
            self._offload(self.text_encoder)
        safe_emit_progress(progress_callback, 0.20, "Text encoder ready")

        if self.scheduler is None:
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
        safe_emit_progress(progress_callback, 0.26, "Scheduler prepared")

        transformer_dtype = self.component_dtypes["transformer"]
        prompt_embeds = prompt_embeds.to(self.device, dtype=transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                self.device, dtype=transformer_dtype
            )

        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        latents = self._get_latents(
            height,
            width,
            num_frames,
            fps=fps,
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            vae_scale_factor_spatial=self.vae_scale_factor_spatial,
            vae_scale_factor_temporal=self.vae_scale_factor_temporal,
            dtype=transformer_dtype,
            generator=generator,
        )
        safe_emit_progress(progress_callback, 0.34, "Latent noise initialized")

        if self.transformer is None:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)
        self.transformer = helper.load_adapters(
            self.transformer, adapter_root, device=self.device, dtype=transformer_dtype
        )

        lora_items, adapter_names = list(self.preloaded_loras.values()), list(
            self.preloaded_loras.keys()
        )
        if lora_items:
            self.apply_loras(lora_items, adapter_names=adapter_names)

        ip_states, ip_states_uncond = helper.build_ip_states(
            embeds, device=self.device, dtype=transformer_dtype
        )

        ref_buffer = ref_buffer_uncond = None
        if helper.ref_loaded and ref_scale is not None:
            if self.vae is None:
                self.load_component_by_type("vae")
            if self.text_encoder is None:
                self.load_component_by_type("text_encoder")
            self.to_device(self.vae)
            self.to_device(self.text_encoder)
            self.to_device(self.transformer)

            aligned_face = helper.align_face(face_image, face_landmarks, face_size=256)
            ref_gen = (
                torch.Generator(device=self.device).manual_seed(seed + 1)
                if seed is not None
                else None
            )
            ref_buffer = helper.encode_reference_buffer(
                self.vae,
                self.text_encoder,
                self.transformer,
                aligned_face,
                device=self.device,
                dtype=transformer_dtype,
                drop=False,
                generator=ref_gen,
            )
            ref_buffer_uncond = helper.encode_reference_buffer(
                self.vae,
                self.text_encoder,
                self.transformer,
                aligned_face,
                device=self.device,
                dtype=transformer_dtype,
                drop=True,
                generator=ref_gen,
            )

        merged_attention_kwargs = {
            **attention_kwargs,
            "ip_hidden_states": ip_states,
            "ip_scale": ip_scale,
        }
        if ref_buffer is not None:
            merged_attention_kwargs.update(
                {"ref_buffer": ref_buffer, "ref_scale": ref_scale}
            )
        merged_attention_kwargs_uncond = {
            "ip_hidden_states": ip_states_uncond,
            "ip_scale": ip_scale,
        }
        if ref_buffer_uncond is not None:
            merged_attention_kwargs_uncond.update(
                {"ref_buffer": ref_buffer_uncond, "ref_scale": ref_scale}
            )

        do_cfg = use_cfg_guidance and negative_prompt_embeds is not None
        self._preview_height = height
        self._preview_width = width
        self._preview_offload = offload

        denoise_progress_callback = denoise_progress_callback or make_mapped_progress(
            progress_callback, 0.50, 0.90
        )
        safe_emit_progress(progress_callback, 0.45, "Starting denoise phase")

        total_steps = len(timesteps)
        with self._progress_bar(total=total_steps, desc="Denoising steps") as pbar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=merged_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_cfg:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=merged_attention_kwargs_uncond,
                        return_dict=False,
                    )[0]

                    if guidance_scale_i is not None:
                        noise_i = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=merged_attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = (
                            noise_uncond
                            + guidance_scale_i * (noise_i - noise_uncond)
                            + guidance_scale * (noise_pred - noise_i)
                        )
                    else:
                        noise_pred = noise_uncond + guidance_scale * (
                            noise_pred - noise_uncond
                        )

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if (
                    render_on_step
                    and render_on_step_callback
                    and ((i + 1) % render_on_step_interval == 0 or i == 0)
                    and i != total_steps - 1
                ):
                    self._render_step(latents, render_on_step_callback)

                safe_emit_progress(
                    denoise_progress_callback,
                    float(i + 1) / float(total_steps) if total_steps else 1.0,
                    f"Denoising step {i + 1}/{total_steps}",
                )
                pbar.update(1)

        if offload:
            self._offload(self.transformer)

        if return_latents:
            safe_emit_progress(progress_callback, 1.0, "Returning latents")
            return latents

        video = self.vae_decode(latents, offload=offload)
        frames = self._tensor_to_frames(video, output_type=output_type)
        safe_emit_progress(progress_callback, 1.0, "Lynx generation complete")
        return frames
