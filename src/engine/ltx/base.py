import torch
import math
from typing import List, Union, Optional, Tuple, TYPE_CHECKING
from diffusers.utils.torch_utils import randn_tensor
import inspect
from PIL import Image
from src.mixins import LoaderMixin
from src.helpers.ltx.patchifier import Patchifier
import copy
from einops import rearrange
import torch.nn.functional as F
import io
import av
import numpy as np
import torchvision.transforms.functional as TVF
from src.engine.base_engine import AutoLoadingHelperDict

# Typing-only linkage to BaseEngine for IDE navigation and autocompletion,
# while avoiding a runtime dependency/import cycle.
if TYPE_CHECKING:
    from src.engine.base_engine import BaseEngine  # noqa: F401
    BaseClass = BaseEngine  # type: ignore
else:
    BaseClass = object


def _encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def _decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def compress(image: torch.Tensor, crf=29):
    if crf == 0:
        return image

    image_array = (
        (image[: (image.shape[0] // 2) * 2, : (image.shape[1] // 2) * 2] * 255.0)
        .byte()
        .cpu()
        .numpy()
    )
    with io.BytesIO() as output_file:
        _encode_single_frame(output_file, image_array, crf)
        video_bytes = output_file.getvalue()
    with io.BytesIO(video_bytes) as video_file:
        image_array = _decode_single_frame(video_file)
    tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
    return tensor


class LTXVideoCondition(LoaderMixin):
    def __init__(
        self,
        image: Optional[Image.Image] = None,
        video: Optional[List[Image.Image]] = None,
        height: int = 704,
        width: int = 1216,
        frame_number: int = 0,
        target_frames: int | None = None,
        conditioning_strength: float = 1.0,
        vae_scale_factor_spatial: int = 32,
        media_x: Optional[int] = None,
        media_y: Optional[int] = None,
        padding: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 0),
    ):
        if image is not None:
            self._media_item = self._load_image(image)
        elif video is not None:
            self._media_item = self._load_video(video)
        else:
            raise ValueError("No media item provided")
        self.frame_number = frame_number
        self.conditioning_strength = conditioning_strength
        self.media_x = media_x
        self.media_y = media_y
        self.height = height
        self.width = width
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        num_frames = (
            1 if isinstance(self._media_item, Image.Image) else len(self._media_item)
        )

        num_frames = self.trim_conditioning_sequence(
            self.frame_number, num_frames, target_frames or num_frames
        )

        self.media_item = self.load_media_file(
            self._media_item,
            self.height,
            self.width,
            num_frames,
            padding,
            just_crop=True,
        )

    def load_media_file(
        self,
        media: List[Image.Image] | Image.Image,
        height: int,
        width: int,
        max_frames: int,
        padding: tuple[int, int, int, int],
        just_crop: bool = False,
    ) -> torch.Tensor:

        if isinstance(media, List):
            num_input_frames = min(len(media), max_frames)

            # Read and preprocess the relevant frames from the video file.
            frames = []
            for i in range(num_input_frames):
                frame = media[i]
                frame_tensor = self.load_image_to_tensor_with_resize_and_crop(
                    frame, height, width, just_crop=just_crop
                )
                frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
                frames.append(frame_tensor)
            # Stack frames along the temporal dimension
            media_tensor = torch.cat(frames, dim=2)
        else:  # Input image
            media_tensor = self.load_image_to_tensor_with_resize_and_crop(
                media, height, width, just_crop=just_crop
            )
            media_tensor = torch.nn.functional.pad(media_tensor, padding)
        return media_tensor

    def load_image_to_tensor_with_resize_and_crop(
        self,
        image_input: Union[str, Image.Image],
        target_height: int = 512,
        target_width: int = 768,
        just_crop: bool = False,
    ) -> torch.Tensor:
        """Load and process an image into a tensor.

        Args:
            image_input: Either a file path (str) or a PIL Image object
            target_height: Desired height of output tensor
            target_width: Desired width of output tensor
            just_crop: If True, only crop the image to the target size without resizing
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError(
                "image_input must be either a file path or a PIL Image object"
            )

        input_width, input_height = image.size
        aspect_ratio_target = target_width / target_height
        aspect_ratio_frame = input_width / input_height

        if aspect_ratio_frame > aspect_ratio_target:
            new_width = int(input_height * aspect_ratio_target)
            new_height = input_height
            x_start = (input_width - new_width) // 2
            y_start = 0
        else:
            new_width = input_width
            new_height = int(input_width / aspect_ratio_target)
            x_start = 0
            y_start = (input_height - new_height) // 2

        image = image.crop(
            (x_start, y_start, x_start + new_width, y_start + new_height)
        )
        if not just_crop:
            image = image.resize((target_width, target_height))

        frame_tensor = TVF.to_tensor(image)  # PIL -> tensor (C, H, W), [0,1]
        frame_tensor = TVF.gaussian_blur(frame_tensor, kernel_size=3, sigma=1.0)
        frame_tensor_hwc = frame_tensor.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        frame_tensor_hwc = compress(frame_tensor_hwc)
        frame_tensor = (
            frame_tensor_hwc.permute(2, 0, 1) * 255.0
        )  # (H, W, C) -> (C, H, W)
        frame_tensor = (frame_tensor / 127.5) - 1.0
        # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
        return frame_tensor.unsqueeze(0).unsqueeze(2)

    def trim_conditioning_sequence(
        self, start_frame: int, sequence_num_frames: int, target_num_frames: int
    ):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.

        Returns:
            int: updated sequence length
        """

        scale_factor = self.vae_scale_factor_spatial
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames


class LTXBaseEngine(BaseClass):
    """Base class for LTX engine implementations"""

    def __init__(self, main_engine: "BaseEngine"):
        self.main_engine = main_engine
        # Delegate properties to main engine

    # Dynamic delegation: forward unknown attributes/methods to the underlying BaseEngine
    def __getattr__(self, name: str):  # noqa: D401
        """Delegate attribute access to the composed BaseEngine when not found here."""
        try:
            return getattr(self.main_engine, name)
        except AttributeError as exc:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute '{name}'") from exc

    # Improve editor introspection (e.g., autocomplete) by exposing attributes of main_engine
    def __dir__(self):
        return sorted(set(list(super().__dir__()) + dir(self.main_engine)))

    @property
    def text_encoder(self):
        return self.main_engine.text_encoder

    @property
    def device(self):
        return self.main_engine.device

    @property
    def component_dtypes(self):
        return self.main_engine.component_dtypes

    @text_encoder.setter
    def text_encoder(self, text_encoder):
        self.main_engine.text_encoder = text_encoder

    @property
    def transformer(self):
        return self.main_engine.transformer

    @transformer.setter
    def transformer(self, transformer):
        self.main_engine.transformer = transformer

    @property
    def scheduler(self):
        return self.main_engine.scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        self.main_engine.scheduler = scheduler

    @property
    def vae(self):
        return self.main_engine.vae

    @vae.setter
    def vae(self, vae):
        self.main_engine.vae = vae

    def _get_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
        **kwargs,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.
            max_timestep ('float', *optional*, defaults to 1.0):
                The initial noising level for image-to-image/video-to-video. The list if timestamps will be
                truncated to start with a timestamp greater or equal to this.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps

            if (
                skip_initial_inference_steps < 0
                or skip_final_inference_steps < 0
                or skip_initial_inference_steps + skip_final_inference_steps
                >= num_inference_steps
            ):
                raise ValueError(
                    "invalid skip inference step values: must be non-negative and the sum of skip_initial_inference_steps and skip_final_inference_steps must be less than the number of inference steps"
                )

            timesteps = timesteps[
                skip_initial_inference_steps : len(timesteps)
                - skip_final_inference_steps
            ]
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            num_inference_steps = len(timesteps)

        return timesteps, num_inference_steps

    def _get_latents(
        self,
        height: int,
        width: int,
        duration: int | str,
        fps: int = 16,
        num_videos: int = 1,
        shape: Tuple[int, int, int, int, int] = None,
        num_channels_latents: int = None,
        seed: int | None = None,
        dtype: torch.dtype = None,
        layout: torch.layout = None,
        generator: torch.Generator | None = None,
        return_generator: bool = False,
        parse_frames: bool = True,
    ):
        if parse_frames or isinstance(duration, str):
            num_frames = self._parse_num_frames(duration, fps)
            latent_num_frames = math.ceil(
                (num_frames + 3) / self.vae_scale_factor_temporal
            )
        else:
            latent_num_frames = duration

        latent_height = math.ceil(height / self.vae_scale_factor_spatial)
        latent_width = math.ceil(width / self.vae_scale_factor_spatial)

        if seed is not None and generator is not None:
            self.logger.warning(
                f"Both `seed` and `generator` are provided. `seed` will be ignored."
            )

        if generator is None:
            device = self.device
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)
        else:
            device = generator.device

        if shape is not None:
            b, c, f, h, w = shape
        else:
            b, c, f, h, w = (
                num_videos,
                num_channels_latents or self.num_channels_latents,
                latent_num_frames,
                latent_height,
                latent_width,
            )

        noise = randn_tensor(
            (b, f * h * w, c), generator=generator, device=device, dtype=dtype
        )
        noise = rearrange(noise, "b (f h w) c -> b c f h w", f=f, h=h, w=w)

        if return_generator:
            return noise, generator
        else:
            return noise

    def latent_to_pixel_coords(
        self, latent_coords: torch.Tensor, causal_fix: bool = False
    ) -> torch.Tensor:
        """
        Converts latent coordinates to pixel coordinates by scaling them according to the VAE's
        configuration.

        Args:
            latent_coords (Tensor): A tensor of shape [batch_size, 3, num_latents]
            containing the latent corner coordinates of each token.
            vae (AutoencoderKL): The VAE model
            causal_fix (bool): Whether to take into account the different temporal scale
                of the first frame. Default = False for backwards compatibility.
        Returns:
            Tensor: A tensor of pixel coordinates corresponding to the input latent coordinates.
        """

        scale_factors = (
            self.vae_scale_factor_temporal,
            self.vae_scale_factor_spatial,
            self.vae_scale_factor_spatial,
        )
        pixel_coords = self.latent_to_pixel_coords_from_factors(
            latent_coords, scale_factors, causal_fix
        )
        return pixel_coords

    def latent_to_pixel_coords_from_factors(
        self,
        latent_coords: torch.Tensor,
        scale_factors: Tuple[int, int, int],
        causal_fix: bool = False,
    ) -> torch.Tensor:
        pixel_coords = (
            latent_coords
            * torch.tensor(scale_factors, device=latent_coords.device)[None, :, None]
        )
        if causal_fix:
            # Fix temporal scale for first frame to 1 due to causality
            pixel_coords[:, 0] = (pixel_coords[:, 0] + 1 - scale_factors[0]).clamp(
                min=0
            )
        return pixel_coords

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_output(
        self,
        latents: torch.Tensor,
        offload: bool = True,
        generator: torch.Generator | None = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        tone_map_compression_ratio: float = 0.0,
    ):

        batch_size = latents.shape[0]

        if not self.vae:
            self.load_component_by_type("vae")

        self.to_device(self.vae)

        latents = self.vae.denormalize_latents(latents)

        latents = latents.to(self.component_dtypes["vae"])

        if not self.vae.config.timestep_conditioning:
            timestep = None
        else:
            noise = randn_tensor(
                latents.shape,
                generator=generator,
                device=self.device,
                dtype=latents.dtype,
            )
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * batch_size
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * batch_size
            timestep = torch.tensor(
                decode_timestep, device=self.device, dtype=latents.dtype
            )
            decode_noise_scale = torch.tensor(
                decode_noise_scale, device=self.device, dtype=latents.dtype
            )[:, None, None, None, None]
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        latents = self.tone_map_latents(latents, tone_map_compression_ratio)
        *_, fl, hl, wl = latents.shape

        decoded_video = self.vae.decode(
            latents,
            target_shape=(
                1,
                3,
                fl * self.vae_scale_factor_temporal,
                hl * self.vae_scale_factor_spatial,
                wl * self.vae_scale_factor_spatial,
            ),
            timestep=timestep,
            return_dict=False,
        )[0]
        video = self._tensor_to_frames(decoded_video)

        if offload:
            self._offload(self.vae)

        return video

    @staticmethod
    def tone_map_latents(
        latents: torch.Tensor,
        compression: float,
    ) -> torch.Tensor:
        """
        Applies a non-linear tone-mapping function to latent values to reduce their dynamic range
        in a perceptually smooth way using a sigmoid-based compression.

        This is useful for regularizing high-variance latents or for conditioning outputs
        during generation, especially when controlling dynamic behavior with a `compression` factor.

        Parameters:
        ----------
        latents : torch.Tensor
            Input latent tensor with arbitrary shape. Expected to be roughly in [-1, 1] or [0, 1] range.
        compression : float
            Compression strength in the range [0, 1].
            - 0.0: No tone-mapping (identity transform)
            - 1.0: Full compression effect

        Returns:
        -------
        torch.Tensor
            The tone-mapped latent tensor of the same shape as input.
        """
        if not (0 <= compression <= 1):
            raise ValueError("Compression must be in the range [0, 1]")

        # Remap [0-1] to [0-0.75] and apply sigmoid compression in one shot
        scale_factor = compression * 0.75
        abs_latents = torch.abs(latents)

        # Sigmoid compression: sigmoid shifts large values toward 0.2, small values stay ~1.0
        # When scale_factor=0, sigmoid term vanishes, when scale_factor=0.75, full effect
        sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale_factor * sigmoid_term

        filtered = latents * scales
        return filtered

    def prepare_conditioning(
        self,
        conditioning_items: Optional[List[LTXVideoCondition]],
        patchifier: Patchifier,
        init_latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        causal_fix: bool = False,
        generator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Prepare conditioning tokens based on the provided conditioning items.

        This method encodes provided conditioning items (video frames or single frames) into latents
        and integrates them with the initial latent tensor. It also calculates corresponding pixel
        coordinates, a mask indicating the influence of conditioning latents, and the total number of
        conditioning latents.

        Args:
            conditioning_items (Optional[List[LTXVideoCondition]]): A list of LTXVideoCondition objects.
            patchifier: The patchifier to use.
            init_latents (torch.Tensor): The initial latent tensor of shape (b, c, f, h, w), where
                `f` is the number of latent frames, and `h` and `w` are latent spatial dimensions.
            num_frames, height, width: The dimensions of the generated video.
            generator: The random generator
            init_latents (torch.Tensor): The initial latent tensor of shape (b, c, f_l, h_l, w_l), where
                `f_l` is the number of latent frames, and `h_l` and `w_l` are latent spatial dimensions.
            num_frames, height, width: The dimensions of the generated video.
                Defaults to `False`.
            generator: The random generator

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - `init_latents` (torch.Tensor): The updated latent tensor including conditioning latents,
                  patchified into (b, n, c) shape.
                - `init_pixel_coords` (torch.Tensor): The pixel coordinates corresponding to the updated
                  latent tensor.
                - `conditioning_mask` (torch.Tensor): A mask indicating the conditioning-strength of each
                  latent token.
                - `num_cond_latents` (int): The total number of latent tokens added from conditioning items.

        Raises:
            AssertionError: If input shapes, dimensions, or conditions for applying conditioning are invalid.
        """

        if conditioning_items:
            batch_size, _, num_latent_frames = init_latents.shape[:3]

            init_conditioning_mask = torch.zeros(
                init_latents[:, 0, :, :, :].shape,
                dtype=torch.float32,
                device=init_latents.device,
            )

            extra_conditioning_latents = []
            extra_conditioning_pixel_coords = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0  # Number of extra conditioning latents added (should be removed before decoding)

            # Process each conditioning item
            for conditioning_item in conditioning_items:

                conditioning_item = self.resize_conditioning_item(
                    conditioning_item, height, width
                )
                media_item = conditioning_item.media_item
                media_frame_number = conditioning_item.frame_number
                strength = conditioning_item.conditioning_strength
                assert media_item.ndim == 5  # (b, c, f, h, w)
                b, c, n_frames, h, w = media_item.shape
                assert (
                    height == h and width == w
                ) or media_frame_number == 0, f"Dimensions do not match: {height}x{width} != {h}x{w} - allowed only when media_frame_number == 0"
                assert n_frames % 8 == 1
                assert (
                    media_frame_number >= 0
                    and media_frame_number + n_frames <= num_frames
                )

                # Encode the provided conditioning media item
                media_item_latents = self.vae_encode(
                    media_item, dtype=init_latents.dtype, sample_mode="mode"
                )

                # Handle the different conditioning cases
                if media_frame_number == 0:
                    # Get the target spatial position of the latent conditioning item
                    media_item_latents, l_x, l_y = self._get_latent_spatial_position(
                        media_item_latents,
                        conditioning_item,
                        height,
                        width,
                        strip_latent_border=True,
                    )
                    b, c_l, f_l, h_l, w_l = media_item_latents.shape

                    # First frame or sequence - just update the initial noise latents and the mask
                    init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l] = (
                        torch.lerp(
                            init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l],
                            media_item_latents,
                            strength,
                        )
                    )
                    init_conditioning_mask[
                        :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l
                    ] = strength
                else:
                    # Non-first frame or sequence
                    if n_frames > 1:
                        # Handle non-first sequence.
                        # Encoded latents are either fully consumed, or the prefix is handled separately below.
                        (
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                        ) = self._handle_non_first_conditioning_sequence(
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                            media_frame_number,
                            strength,
                        )

                    # Single frame or sequence-prefix latents
                    if media_item_latents is not None:
                        noise = randn_tensor(
                            media_item_latents.shape,
                            generator=generator,
                            device=media_item_latents.device,
                            dtype=media_item_latents.dtype,
                        )

                        media_item_latents = torch.lerp(
                            noise, media_item_latents, strength
                        )

                        # Patchify the extra conditioning latents and calculate their pixel coordinates
                        media_item_latents, latent_coords = patchifier.patchify(
                            latents=media_item_latents
                        )
                        pixel_coords = self.latent_to_pixel_coords(
                            latent_coords,
                            causal_fix=causal_fix,
                        )

                        # Update the frame numbers to match the target frame number
                        pixel_coords[:, 0] += media_frame_number
                        extra_conditioning_num_latents += media_item_latents.shape[1]

                        conditioning_mask = torch.full(
                            media_item_latents.shape[:2],
                            strength,
                            dtype=torch.float32,
                            device=init_latents.device,
                        )

                        extra_conditioning_latents.append(media_item_latents)
                        extra_conditioning_pixel_coords.append(pixel_coords)
                        extra_conditioning_mask.append(conditioning_mask)

        # Patchify the updated latents and calculate their pixel coordinates
        init_latents, init_latent_coords = patchifier.patchify(latents=init_latents)

        init_pixel_coords = self.latent_to_pixel_coords(
            init_latent_coords,
            causal_fix=causal_fix,
        )

        if not conditioning_items:
            return init_latents, init_pixel_coords, None, 0

        init_conditioning_mask, _ = patchifier.patchify(
            latents=init_conditioning_mask.unsqueeze(1)
        )

        init_conditioning_mask = init_conditioning_mask.squeeze(-1)

        if extra_conditioning_latents:
            # Stack the extra conditioning latents, pixel coordinates and mask
            init_latents = torch.cat([*extra_conditioning_latents, init_latents], dim=1)
            init_pixel_coords = torch.cat(
                [*extra_conditioning_pixel_coords, init_pixel_coords], dim=2
            )
            init_conditioning_mask = torch.cat(
                [*extra_conditioning_mask, init_conditioning_mask], dim=1
            )

            if self.transformer.use_tpu_flash_attention:
                # When flash attention is used, keep the original number of tokens by removing
                #   tokens from the end.
                init_latents = init_latents[:, :-extra_conditioning_num_latents]
                init_pixel_coords = init_pixel_coords[
                    :, :, :-extra_conditioning_num_latents
                ]
                init_conditioning_mask = init_conditioning_mask[
                    :, :-extra_conditioning_num_latents
                ]

        return (
            init_latents,
            init_pixel_coords,
            init_conditioning_mask,
            extra_conditioning_num_latents,
        )

    def _get_latent_spatial_position(
        self,
        latents: torch.Tensor,
        conditioning_item: LTXVideoCondition,
        height: int,
        width: int,
        strip_latent_border,
    ):
        """
        Get the spatial position of the conditioning item in the latent space.
        If requested, strip the conditioning latent borders that do not align with target borders.
        (border latents look different then other latents and might confuse the model)
        """
        scale = self.vae_scale_factor_spatial
        h, w = conditioning_item.media_item.shape[-2:]
        assert (
            h <= height and w <= width
        ), f"Conditioning item size {h}x{w} is larger than target size {height}x{width}"
        assert h % scale == 0 and w % scale == 0

        # Compute the start and end spatial positions of the media item
        x_start, y_start = conditioning_item.media_x, conditioning_item.media_y
        x_start = (width - w) // 2 if x_start is None else x_start
        y_start = (height - h) // 2 if y_start is None else y_start
        x_end, y_end = x_start + w, y_start + h
        assert (
            x_end <= width and y_end <= height
        ), f"Conditioning item {x_start}:{x_end}x{y_start}:{y_end} is out of bounds for target size {width}x{height}"

        if strip_latent_border:
            # Strip one latent from left/right and/or top/bottom, update x, y accordingly
            if x_start > 0:
                x_start += scale
                latents = latents[:, :, :, :, 1:]

            if y_start > 0:
                y_start += scale
                latents = latents[:, :, :, 1:, :]

            if x_end < width:
                latents = latents[:, :, :, :, :-1]

            if y_end < height:
                latents = latents[:, :, :, :-1, :]

        return latents, x_start // scale, y_start // scale

    @staticmethod
    def _handle_non_first_conditioning_sequence(
        init_latents: torch.Tensor,
        init_conditioning_mask: torch.Tensor,
        latents: torch.Tensor,
        media_frame_number: int,
        strength: float,
        num_prefix_latent_frames: int = 2,
        prefix_latents_mode: str = "concat",
        prefix_soft_conditioning_strength: float = 0.15,
    ):
        """
        Special handling for a conditioning sequence that does not start on the first frame.
        The special handling is required to allow a short encoded video to be used as middle
        (or last) sequence in a longer video.
        Args:
            init_latents (torch.Tensor): The initial noise latents to be updated.
            init_conditioning_mask (torch.Tensor): The initial conditioning mask to be updated.
            latents (torch.Tensor): The encoded conditioning item.
            media_frame_number (int): The target frame number of the first frame in the conditioning sequence.
            strength (float): The conditioning strength for the conditioning latents.
            num_prefix_latent_frames (int, optional): The length of the sequence prefix, to be handled
                separately. Defaults to 2.
            prefix_latents_mode (str, optional): Special treatment for prefix (boundary) latents.
                - "drop": Drop the prefix latents.
                - "soft": Use the prefix latents, but with soft-conditioning
                - "concat": Add the prefix latents as extra tokens (like single frames)
            prefix_soft_conditioning_strength (float, optional): The strength of the soft-conditioning for
                the prefix latents, relevant if `prefix_latents_mode` is "soft". Defaults to 0.1.

        """
        f_l = latents.shape[2]
        f_l_p = num_prefix_latent_frames
        assert f_l >= f_l_p
        assert media_frame_number % 8 == 0
        if f_l > f_l_p:
            # Insert the conditioning latents **excluding the prefix** into the sequence
            f_l_start = media_frame_number // 8 + f_l_p
            f_l_end = f_l_start + f_l - f_l_p
            init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                init_latents[:, :, f_l_start:f_l_end],
                latents[:, :, f_l_p:],
                strength,
            )
            # Mark these latent frames as conditioning latents
            init_conditioning_mask[:, f_l_start:f_l_end] = strength

        # Handle the prefix-latents
        if prefix_latents_mode == "soft":
            if f_l_p > 1:
                # Drop the first (single-frame) latent and soft-condition the remaining prefix
                f_l_start = media_frame_number // 8 + 1
                f_l_end = f_l_start + f_l_p - 1
                strength = min(prefix_soft_conditioning_strength, strength)
                init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                    init_latents[:, :, f_l_start:f_l_end],
                    latents[:, :, 1:f_l_p],
                    strength,
                )
                # Mark these latent frames as conditioning latents
                init_conditioning_mask[:, f_l_start:f_l_end] = strength
            latents = None  # No more latents to handle
        elif prefix_latents_mode == "drop":
            # Drop the prefix latents
            latents = None
        elif prefix_latents_mode == "concat":
            # Pass-on the prefix latents to be handled as extra conditioning frames
            latents = latents[:, :, :f_l_p]
        else:
            raise ValueError(f"Invalid prefix_latents_mode: {prefix_latents_mode}")
        return (
            init_latents,
            init_conditioning_mask,
            latents,
        )

    @staticmethod
    def resize_conditioning_item(
        conditioning_item: LTXVideoCondition,
        height: int,
        width: int,
    ):
        if conditioning_item.media_x or conditioning_item.media_y:
            raise ValueError(
                "Provide media_item in the target size for spatial conditioning."
            )
        new_conditioning_item = copy.copy(conditioning_item)
        new_conditioning_item.media_item = LTXBaseEngine.resize_tensor(
            conditioning_item.media_item, height, width
        )
        return new_conditioning_item

    @staticmethod
    def resize_tensor(media_items, height, width):
        n_frames = media_items.shape[2]
        if media_items.shape[-2:] != (height, width):
            media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
            media_items = F.interpolate(
                media_items,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            media_items = rearrange(media_items, "(b n) c h w -> b c n h w", n=n_frames)
        return media_items
    
    
    
    
    
