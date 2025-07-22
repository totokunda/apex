import torch
from typing import Dict, Any, Callable, List, Union, Optional, Tuple
import math


class MagiBaseEngine:
    """Base class for Magi engine implementations containing common functionality"""

    def __init__(self, main_engine):
        self.main_engine = main_engine
        # Delegate common properties to the main engine
        self.device = main_engine.device
        self.logger = main_engine.logger
        self.vae_scale_factor_temporal = main_engine.vae_scale_factor_temporal
        self.vae_scale_factor_spatial = main_engine.vae_scale_factor_spatial
        self.num_channels_latents = main_engine.num_channels_latents
        self.video_processor = main_engine.video_processor

    @property
    def text_encoder(self):
        return self.main_engine.text_encoder

    @property
    def transformer(self):
        return self.main_engine.transformer

    @property
    def scheduler(self):
        return self.main_engine.scheduler

    @property
    def vae(self):
        return self.main_engine.vae

    @property
    def preprocessors(self):
        return self.main_engine.preprocessors

    @property
    def component_dtypes(self):
        return self.main_engine.component_dtypes

    def load_component_by_type(self, component_type: str):
        """Load a component by type"""
        return self.main_engine.load_component_by_type(component_type)

    def load_preprocessor_by_type(self, preprocessor_type: str):
        """Load a preprocessor by type"""
        return self.main_engine.load_preprocessor_by_type(preprocessor_type)

    def to_device(self, component):
        """Move component to device"""
        return self.main_engine.to_device(component)

    def _offload(self, component):
        """Offload component"""
        return self.main_engine._offload(component)

    def _get_latents(self, *args, **kwargs):
        """Get latents"""
        return self.main_engine._get_latents(*args, **kwargs)

    def _get_timesteps(self, *args, **kwargs):
        """Get timesteps"""
        return self.main_engine._get_timesteps(*args, **kwargs)

    def _parse_num_frames(self, *args, **kwargs):
        """Parse number of frames"""
        return self.main_engine._parse_num_frames(*args, **kwargs)

    def _aspect_ratio_resize(self, *args, **kwargs):
        """Aspect ratio resize"""
        return self.main_engine._aspect_ratio_resize(*args, **kwargs)

    def _load_image(self, *args, **kwargs):
        """Load image"""
        return self.main_engine._load_image(*args, **kwargs)

    def _load_video(self, *args, **kwargs):
        """Load video"""
        return self.main_engine._load_video(*args, **kwargs)

    def _progress_bar(self, *args, **kwargs):
        """Progress bar context manager"""
        return self.main_engine._progress_bar(*args, **kwargs)

    def _postprocess(self, *args, **kwargs):
        """Postprocess video"""
        return self.main_engine._postprocess(*args, **kwargs)

    def vae_encode(self, *args, **kwargs):
        """VAE encode"""
        return self.main_engine.vae_encode(*args, **kwargs)

    def vae_decode(self, *args, **kwargs):
        """VAE decode"""
        return self.main_engine.vae_decode(*args, **kwargs)

    def denoise(self, *args, **kwargs):
        """Denoise function"""
        return self.main_engine.denoise(*args, **kwargs)

    def init_timestep_schedule(
        self,
        num_steps: int,
        device: torch.device,
        transform_type: str = "sd3",
        shift: float = 3.0,
    ) -> torch.Tensor:
        """Initialize timestep schedule following MAGI's approach"""

        if num_steps == 12:
            # MAGI's specific 12-step schedule
            base_t = torch.linspace(0, 1, 4 + 1, device=device) / 4
            accu_num = torch.linspace(0, 1, 4 + 1, device=device)
            base_t = torch.cat([base_t[:1], base_t[2:4]], dim=0)
            t = torch.cat([base_t + accu for accu in accu_num], dim=0)[: num_steps + 1]
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
        special_token_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process text embeddings following MAGI's special token approach"""

        # Get null embedding from transformer (if available)
        null_emb = getattr(self.transformer, "y_embedder", None)
        if null_emb and hasattr(null_emb, "null_caption_embedding"):
            null_caption_embedding = null_emb.null_caption_embedding.unsqueeze(0)
        else:
            # Create a dummy null embedding if not available
            null_caption_embedding = torch.zeros_like(prompt_embeds[:1])

        # Expand embeddings for chunks
        denoise_chunks = infer_chunk_num - clean_chunk_num

        # Denoise chunks with caption_embs
        caption_embs = prompt_embeds.repeat(1, denoise_chunks, 1, 1)
        emb_masks = (
            prompt_attention_mask.unsqueeze(1).repeat(1, denoise_chunks, 1)
            if prompt_attention_mask is not None
            else None
        )

        # Apply special tokens if specified (simplified version of MAGI's special token logic)
        if special_token_kwargs.get("use_special_tokens", False):
            caption_embs, emb_masks = self._pad_special_tokens(
                caption_embs, emb_masks, special_token_kwargs
            )

        # Clean chunks with null_emb
        caption_embs = torch.cat(
            [null_caption_embedding.repeat(1, clean_chunk_num, 1, 1), caption_embs],
            dim=1,
        )
        if emb_masks is not None:
            emb_masks = torch.cat(
                [
                    torch.zeros(
                        1,
                        clean_chunk_num,
                        emb_masks.size(2),
                        dtype=emb_masks.dtype,
                        device=emb_masks.device,
                    ),
                    emb_masks,
                ],
                dim=1,
            )

        # Handle CFG by concatenating conditional and unconditional
        if negative_prompt_embeds is not None:
            # Process negative embeddings similarly
            neg_emb_expanded = negative_prompt_embeds.repeat(1, denoise_chunks, 1, 1)
            neg_mask_expanded = (
                negative_prompt_attention_mask.unsqueeze(1).repeat(1, denoise_chunks, 1)
                if negative_prompt_attention_mask is not None
                else None
            )

            neg_emb_expanded = torch.cat(
                [
                    null_caption_embedding.repeat(1, clean_chunk_num, 1, 1),
                    neg_emb_expanded,
                ],
                dim=1,
            )
            if neg_mask_expanded is not None:
                neg_mask_expanded = torch.cat(
                    [
                        torch.zeros(
                            1,
                            clean_chunk_num,
                            neg_mask_expanded.size(2),
                            dtype=neg_mask_expanded.dtype,
                            device=neg_mask_expanded.device,
                        ),
                        neg_mask_expanded,
                    ],
                    dim=1,
                )

            # Concatenate for CFG
            caption_embs = torch.cat([caption_embs, neg_emb_expanded], dim=0)
            if emb_masks is not None and neg_mask_expanded is not None:
                emb_masks = torch.cat([emb_masks, neg_mask_expanded], dim=0)

        return caption_embs, emb_masks

    def _pad_special_tokens(
        self,
        caption_embs: torch.Tensor,
        emb_masks: torch.Tensor,
        special_token_kwargs: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified special token padding (can be expanded based on specific needs)"""
        # This is a placeholder for MAGI's complex special token logic
        # In practice, you would implement the full special token system here
        return caption_embs, emb_masks
