import torch
from typing import Dict, Any, Callable, List, Union, Optional, Tuple
import math
import os
import numpy as np
from PIL import Image
import io
import ffmpeg

SPECIAL_TOKEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "assets",
    "magi",
    "special_tokens.npz",
)

SPECIAL_TOKEN = np.load(SPECIAL_TOKEN_PATH)
CAPTION_TOKEN = torch.tensor(SPECIAL_TOKEN["caption_token"].astype(np.float16))
LOGO_TOKEN = torch.tensor(SPECIAL_TOKEN["logo_token"].astype(np.float16))
TRANS_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][:1].astype(np.float16))
HQ_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][1:2].astype(np.float16))
STATIC_FIRST_FRAMES_TOKEN = torch.tensor(
    SPECIAL_TOKEN["other_tokens"][2:3].astype(np.float16)
)  # static first frames
DYNAMIC_FIRST_FRAMES_TOKEN = torch.tensor(
    SPECIAL_TOKEN["other_tokens"][3:4].astype(np.float16)
)  # dynamic first frames
BORDERNESS_TOKEN = torch.tensor(SPECIAL_TOKEN["other_tokens"][4:5].astype(np.float16))
DURATION_TOKEN_LIST = [
    torch.tensor(SPECIAL_TOKEN["other_tokens"][i : i + 1].astype(np.float16))
    for i in range(0 + 7, 8 + 7)
]
THREE_D_MODEL_TOKEN = torch.tensor(
    SPECIAL_TOKEN["other_tokens"][15:16].astype(np.float16)
)
TWO_D_ANIME_TOKEN = torch.tensor(
    SPECIAL_TOKEN["other_tokens"][16:17].astype(np.float16)
)

SPECIAL_TOKEN_DICT = {
    "CAPTION_TOKEN": CAPTION_TOKEN,
    "LOGO_TOKEN": LOGO_TOKEN,
    "TRANS_TOKEN": TRANS_TOKEN,
    "HQ_TOKEN": HQ_TOKEN,
    "STATIC_FIRST_FRAMES_TOKEN": STATIC_FIRST_FRAMES_TOKEN,
    "DYNAMIC_FIRST_FRAMES_TOKEN": DYNAMIC_FIRST_FRAMES_TOKEN,
    "BORDERNESS_TOKEN": BORDERNESS_TOKEN,
    "THREE_D_MODEL_TOKEN": THREE_D_MODEL_TOKEN,
    "TWO_D_ANIME_TOKEN": TWO_D_ANIME_TOKEN,
}


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

    def _load_video(self, *args, **kwargs):
        """Load video"""
        return self.main_engine._load_video(*args, **kwargs)

    def _load_image(self, *args, **kwargs):
        """Load image"""
        return self.main_engine._load_image(*args, **kwargs)

    def _parse_num_frames(self, *args, **kwargs):
        """Parse number of frames"""
        return self.main_engine._parse_num_frames(*args, **kwargs)

    def _aspect_ratio_resize(self, *args, **kwargs):
        """Aspect ratio resize"""
        return self.main_engine._aspect_ratio_resize(*args, **kwargs)

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

    def pad_duration_token_keys(
        self, special_token_keys: List[str], pad_duration: bool = True
    ) -> List[str]:
        if "DURATION_TOKEN" in set(special_token_keys):
            return special_token_keys
        if pad_duration:
            return special_token_keys + ["DURATION_TOKEN"]
        return special_token_keys

    def get_special_token_keys(
        self,
        add_pad_static: bool = False,
        add_pad_dynamic: bool = False,
        add_pad_borderness: bool = False,
        add_pad_hq: bool = True,
        add_pad_three_d_model: bool = False,
        add_pad_two_d_anime: bool = False,
        pad_duration: bool = True,
    ) -> List[str]:
        special_token_keys = []
        if add_pad_static:
            special_token_keys.append("STATIC_FIRST_FRAMES_TOKEN")
        if add_pad_dynamic:
            special_token_keys.append("DYNAMIC_FIRST_FRAMES_TOKEN")
        if add_pad_borderness:
            special_token_keys.append("BORDERNESS_TOKEN")
        if add_pad_hq:
            special_token_keys.append("HQ_TOKEN")
        if add_pad_three_d_model:
            special_token_keys.append("THREE_D_MODEL_TOKEN")
        if add_pad_two_d_anime:
            special_token_keys.append("TWO_D_ANIME_TOKEN")

        special_token_keys = self.pad_duration_token_keys(
            special_token_keys, pad_duration
        )
        return special_token_keys

    def get_negative_special_token_keys(
        self, is_negative_prompt: bool = False
    ) -> List[str]:
        if is_negative_prompt:
            return ["CAPTION_TOKEN", "LOGO_TOKEN", "TRANS_TOKEN", "BORDERNESS_TOKEN"]
        return None

    def pad_special_token_tensor(
        self,
        special_token: torch.Tensor,
        txt_feat: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ):
        _device = txt_feat.device
        _dtype = txt_feat.dtype
        N, C, _, D = txt_feat.size()
        txt_feat = torch.cat(
            [
                special_token.unsqueeze(0)
                .unsqueeze(0)
                .to(_device)
                .to(_dtype)
                .expand(N, C, -1, D),
                txt_feat,
            ],
            dim=2,
        )[:, :, :800, :]
        if attn_mask is not None:
            attn_mask = torch.cat(
                [
                    torch.ones(N, C, 1, dtype=_dtype, device=_device),
                    attn_mask.to(_dtype).to(_device),
                ],
                dim=-1,
            )[:, :, :800]
        return txt_feat, attn_mask

    def pad_special_token(
        self,
        special_token_keys: List[str],
        caption_embs: torch.Tensor,
        emb_masks: torch.Tensor,
    ):
        device = self.device
        if not special_token_keys:
            return caption_embs, emb_masks
        for special_token_key in special_token_keys:
            if special_token_key == "DURATION_TOKEN":
                new_caption_embs, new_emb_masks = [], []
                num_chunks = caption_embs.size(1)
                for i in range(num_chunks):
                    chunk_caption_embs, chunk_emb_masks = self.pad_special_token_tensor(
                        DURATION_TOKEN_LIST[min(num_chunks - i - 1, 7)].to(device),
                        caption_embs[:, i : i + 1],
                        emb_masks[:, i : i + 1],
                    )
                    new_caption_embs.append(chunk_caption_embs)
                    new_emb_masks.append(chunk_emb_masks)
                caption_embs = torch.cat(new_caption_embs, dim=1)
                emb_masks = torch.cat(new_emb_masks, dim=1)
            else:
                special_token = SPECIAL_TOKEN_DICT.get(special_token_key)
                if special_token is not None:
                    caption_embs, emb_masks = self.pad_special_token_tensor(
                        special_token.to(device), caption_embs, emb_masks
                    )
        return caption_embs, emb_masks

    def _process_txt_embeddings(
        self,
        caption_embs: torch.Tensor,
        emb_masks: torch.Tensor,
        null_emb: torch.Tensor,
        infer_chunk_num: int,
        clean_chunk_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        special_token_keys = self.get_special_token_keys()

        # denoise chunk with caption_embs
        caption_embs = caption_embs.repeat(1, infer_chunk_num - clean_chunk_num, 1, 1)
        emb_masks = emb_masks.unsqueeze(1).repeat(
            1, infer_chunk_num - clean_chunk_num, 1
        )

        caption_embs, emb_masks = self.pad_special_token(
            special_token_keys, caption_embs, emb_masks
        )

        # clean chunk with null_emb
        caption_embs = torch.cat(
            [null_emb.repeat(1, clean_chunk_num, 1, 1), caption_embs], dim=1
        )
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

        return caption_embs, emb_masks

    def process_null_embeddings(
        self,
        null_caption_embedding: torch.Tensor,
        null_emb_masks: torch.Tensor,
        infer_chunk_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        null_embs = null_caption_embedding.repeat(1, infer_chunk_num, 1, 1)
        negative_special_token_keys = self.get_negative_special_token_keys()

        if negative_special_token_keys:
            null_embs, _ = self.pad_special_token(
                negative_special_token_keys, null_embs, None
            )

        null_token_length = 50
        null_emb_masks[:, :, :null_token_length] = 1
        null_emb_masks[:, :, null_token_length:] = 0

        return null_embs, null_emb_masks
