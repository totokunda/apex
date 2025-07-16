from typing import Dict, Any
from PIL import Image
from transformers import WhisperModel, AutoFeatureExtractor
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
from src.preprocess.hunyuan.align import AlignImage
from src.preprocess.base.base import BasePreprocessor
from torchvision import transforms
from transformers import CLIPImageProcessor
import torch
import librosa
import numpy as np
from typing import Union, List
from src.mixins.loader_mixin import LoaderMixin
from src.mixins.offload_mixin import OffloadMixin
from einops import rearrange
from src.preprocess.hunyuan.align import get_facemask


class HyAvatarPreprocessor(BasePreprocessor, LoaderMixin, OffloadMixin):
    def __init__(
        self,
        model_path: str,
        save_path: str = DEFAULT_PREPROCESSOR_SAVE_PATH,
        align_pt_path: str = None,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.save_path = save_path
        self.wav2vec_model = WhisperModel.from_pretrained(
            model_path, cache_dir=save_path
        )
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_path, cache_dir=save_path
        )
        self.align_image = AlignImage(
            save_path=save_path, pt_path=align_pt_path, device=device
        )
        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(
                    (336, 336), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.4082107),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.clip_image_processor = CLIPImageProcessor()

    def __call__(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        audio: str = None,
        image_height: int = 720,
        image_width: int = 1280,
        fps: int = 25,
        dtype: torch.dtype = torch.float32,
    ):

        loaded_image = self._load_image(image)
        motion_bucket_id_heads = np.array([25] * 4)
        motion_bucket_id_exps = np.array([30] * 4)
        motion_bucket_id_heads = torch.from_numpy(motion_bucket_id_heads)
        motion_bucket_id_exps = torch.from_numpy(motion_bucket_id_exps)

        resized_image, image_height, image_width = self._aspect_ratio_resize(
            loaded_image, max_area=image_height * image_width, mod_value=64
        )

        fps = torch.from_numpy(np.array(fps))
        audio_features, audio_length = self._extract_audio_features(audio)
        audio_prompts = [
            self._encode_audio(audio_feature, fps, num_frames=129)
            for audio_feature in audio_features
        ]

        if audio_prompts.shape[1] <= 129:
            audio_prompts = torch.cat(
                [
                    audio_prompts,
                    torch.zeros_like(audio_prompts[:, :1]).repeat(
                        1, 129 - audio_prompts.shape[1], 1, 1, 1
                    ),
                ],
                dim=1,
            )
        else:
            audio_prompts = torch.cat(
                [
                    audio_prompts,
                    torch.zeros_like(audio_prompts[:, :1]).repeat(1, 5, 1, 1, 1),
                ],
                dim=1,
            )

        pixel_value_ref_llava = [self.llava_transform(resized_image)]
        pixel_value_ref_llava = torch.stack(pixel_value_ref_llava, dim=0)
        pixel_value_ref_clip = self.clip_image_processor(
            images=Image.fromarray(
                (resized_image.permute(1, 2, 0)).data.cpu().numpy().astype(np.uint8)
            ),
            return_tensors="pt",
        ).pixel_values[0]

        pixel_value_ref_clip = pixel_value_ref_clip.unsqueeze(0)

        uncond_audio_prompts = torch.zeros_like(audio_prompts[:, :129])
        motion_exp = motion_bucket_id_exps.to(self.device)
        motion_pose = motion_bucket_id_heads.to(self.device)

        pixel_value_ref = pixel_value_ref_llava.to(
            self.device
        )  # (b f c h w) 取值范围[0,255]
        face_masks = get_facemask(pixel_value_ref.clone(), self.align_image, area=3.0)

        pixel_value_ref = pixel_value_ref.clone().repeat(1, 129, 1, 1, 1)
        uncond_pixel_value_ref = torch.zeros_like(pixel_value_ref)
        pixel_value_ref = pixel_value_ref / 127.5 - 1.0
        uncond_pixel_value_ref = uncond_pixel_value_ref * 2 - 1

        pixel_value_ref_for_vae = rearrange(pixel_value_ref, "b f c h w -> b c f h w")
        uncond_uncond_pixel_value_ref = rearrange(
            uncond_pixel_value_ref, "b f c h w -> b c f h w"
        )

        pixel_value_llava = pixel_value_ref_llava.to(self.device)
        pixel_value_llava = rearrange(pixel_value_llava, "b f c h w -> (b f) c h w")
        uncond_pixel_value_llava = pixel_value_llava.clone()

        return {
            "pixel_value_ref": pixel_value_ref,
            "uncond_pixel_value_ref": uncond_pixel_value_ref,
            "pixel_value_ref_for_vae": pixel_value_ref_for_vae,
            "uncond_uncond_pixel_value_ref": uncond_uncond_pixel_value_ref,
            "pixel_value_llava": pixel_value_llava,
            "uncond_pixel_value_llava": uncond_pixel_value_llava,
            "face_masks": face_masks,
            "motion_exp": motion_exp,
            "motion_pose": motion_pose,
            "uncond_audio_prompts": uncond_audio_prompts,
            "audio_prompts": audio_prompts,
            "pixel_value_ref_clip": pixel_value_ref_clip,
        }

    def _extract_audio_features(self, audio: str):
        audio_input, sampling_rate = librosa.load(audio, sr=16000)
        assert sampling_rate == 16000

        audio_features = []
        window = 750 * 640
        for i in range(0, len(audio_input), window):
            audio_feature = self.audio_feature_extractor(
                audio_input[i : i + window],
                sampling_rate=sampling_rate,
                return_tensors="pt",
            ).input_features
        audio_features.append(audio_feature)

        audio_features = torch.cat(audio_features, dim=-1)
        return audio_features, len(audio_input) // 640

    def _encode_image(self, image: Image.Image):
        pass

    def _encode_prompt(self, prompt: str):
        pass

    def _encode_audio(self, audio_feats, fps, num_frames=129):
        if fps == 25:
            start_ts = [0]
            step_ts = [1]
        elif fps == 12.5:
            start_ts = [0]
            step_ts = [2]
        num_frames = min(num_frames, 400)
        audio_feats = self.wav2vec_model.encoder(
            audio_feats.unsqueeze(0)[:, :, :3000], output_hidden_states=True
        ).hidden_states
        audio_feats = torch.stack(audio_feats, dim=2)
        audio_feats = torch.cat([torch.zeros_like(audio_feats[:, :4]), audio_feats], 1)

        audio_prompts = []
        for bb in range(1):
            audio_feats_list = []
            for f in range(num_frames):
                cur_t = (start_ts[bb] + f * step_ts[bb]) * 2
                audio_clip = audio_feats[bb : bb + 1, cur_t : cur_t + 10]
                audio_feats_list.append(audio_clip)
            audio_feats_list = torch.stack(audio_feats_list, 1)
            audio_prompts.append(audio_feats_list)
        audio_prompts = torch.cat(audio_prompts)

        return audio_prompts

    def _aspect_ratio_resize(self, image, max_area=720 * 1280, mod_value=16):
        aspect_ratio = image.height / image.width
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        return image, height, width
