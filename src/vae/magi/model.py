# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)

from .module import ViTEncoder, ViTDecoder, VideoTokenizerABC


class AutoencoderKLMagi(
    ModelMixin, ConfigMixin, FromOriginalModelMixin, VideoTokenizerABC
):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Used in MAGI.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for its generic methods implemented
    for all models (such as downloading or saving).

    Args:
        video_size (`int`, defaults to `256`):
            The size of the input video frames.
        video_length (`int`, defaults to `16`):
            The number of frames in the input video.
        patch_size (`int`, defaults to `8`):
            The size of the spatial patches.
        patch_length (`int`, defaults to `4`):
            The size of the temporal patches.
        in_chans (`int`, defaults to `3`):
            Number of input channels.
        z_chans (`int`, defaults to `4`):
            Number of latent channels.
        double_z (`bool`, defaults to `True`):
            Whether to double the latent channels for mean and variance.
        embed_dim (`int`, defaults to `768`):
            The embedding dimension.
        depth (`int`, defaults to `12`):
            The number of transformer layers.
        num_heads (`int`, defaults to `12`):
            The number of attention heads.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of MLP hidden dimension to embedding dimension.
        scaling_factor (`float`, defaults to `1.0`):
            The component-wise standard deviation of the trained latent space.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        video_size: int = 256,
        video_length: int = 16,
        patch_size: int = 8,
        patch_length: int = 4,
        in_chans: int = 3,
        z_chans: int = 4,
        double_z: bool = True,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        with_cls_token: bool = True,
        norm_code: bool = False,
        ln_in_attn: bool = False,
        conv_last_layer: bool = False,
        use_rope: bool = False,
        use_final_proj: bool = False,
        scaling_factor: float = 0.18215,
        spatial_compression_ratio: Optional[int] = None,
        temporal_compression_ratio: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Create encoder and decoder config
        ddconfig = {
            "video_size": video_size,
            "video_length": video_length,
            "patch_size": patch_size,
            "patch_length": patch_length,
            "in_chans": in_chans,
            "z_chans": z_chans,
            "double_z": double_z,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "qkv_bias": qkv_bias,
            "qk_scale": qk_scale,
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "drop_path_rate": drop_path_rate,
            "with_cls_token": with_cls_token,
            "norm_code": norm_code,
            "ln_in_attn": ln_in_attn,
            "conv_last_layer": conv_last_layer,
            "use_rope": use_rope,
            "use_final_proj": use_final_proj,
        }

        self.encoder = ViTEncoder(**ddconfig)
        self.decoder = ViTDecoder(**ddconfig)

        self._temporal_downsample_factor = patch_length
        self._spatial_downsample_factor = patch_size

        self.spatial_compression_ratio = (
            patch_size
            if spatial_compression_ratio is None
            else spatial_compression_ratio
        )
        self.temporal_compression_ratio = (
            patch_length
            if temporal_compression_ratio is None
            else temporal_compression_ratio
        )

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
        # at a fixed frame batch size, the memory requirement can be lowered.
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

        # This can be configured based on the amount of GPU memory available.
        self.num_sample_frames_batch_size = 16
        self.num_latent_frames_batch_size = 2

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 16

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 224
        self.tile_sample_stride_width = 224
        self.tile_sample_stride_num_frames = 8

    @property
    def spatial_downsample_factor(self):
        return self._spatial_downsample_factor

    @property
    def temporal_downsample_factor(self):
        return self._temporal_downsample_factor

    @property
    def first_frame_as_image(self):
        """
        Property representing the first frame as image.
        """
        return False

    @property
    def allow_spatial_tiling(self):
        """
        Determines whether spatial tiling is allowed or not.
        """
        return False

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_num_frames: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_sample_stride_num_frames: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger videos.
        """
        self.use_tiling = True
        self.tile_sample_min_height = (
            tile_sample_min_height or self.tile_sample_min_height
        )
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = (
            tile_sample_min_num_frames or self.tile_sample_min_num_frames
        )
        self.tile_sample_stride_height = (
            tile_sample_stride_height or self.tile_sample_stride_height
        )
        self.tile_sample_stride_width = (
            tile_sample_stride_width or self.tile_sample_stride_width
        )
        self.tile_sample_stride_num_frames = (
            tile_sample_stride_num_frames or self.tile_sample_stride_num_frames
        )

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Internal encode method"""
        N, C, T, H, W = x.shape
        if T == 1 and self._temporal_downsample_factor > 1:
            x = x.expand(-1, -1, 4, -1, -1)
            x = self.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            z = posterior.mode()
            return z[:, :, :1, :, :].type(x.dtype)
        else:
            x = self.encoder(x)
            posterior = DiagonalGaussianDistribution(x)
            z = posterior.mode()
            return z.type(x.dtype)

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of videos into latents.

        Args:
            x (`torch.Tensor`): Input batch of videos.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """Internal decode method"""
        N, C, T, H, W = z.shape
        if T == 1:
            z = z.expand(-1, -1, 1, -1, -1)
            dec = self.decoder(z)
            dec = dec[:, :, :1, :, :]
        else:
            dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if not return_dict:
            return (dec.sample,)
        return dec

    def get_last_layer(self):
        """
        Get the last layer of the decoder.

        Returns:
            torch.Tensor: Last layer of the decoder.
        """
        return self.decoder.last_layer.weight

    def normalize_latents(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        if hasattr(self.config, "shift_factor") and self.config.shift_factor:
            latents = (latents - self.config.shift_factor) * self.config.scaling_factor
        else:
            latents = latents * self.config.scaling_factor
        return latents

    def denormalize_latents(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        if hasattr(self.config, "shift_factor") and self.config.shift_factor:
            latents = latents / self.config.scaling_factor + self.config.shift_factor
        else:
            latents = latents / self.config.scaling_factor
        return latents
