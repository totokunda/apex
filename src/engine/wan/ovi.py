from .shared import WanShared
from src.types import InputImage, InputAudio, InputVideo
from typing import List, Union, Optional, Dict, Any, Tuple, Callable
import torch
from src.utils.progress import safe_emit_progress
import numpy as np
import torch.nn.functional as F
import math
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
import torch.amp as amp
import re
import logging
import cv2
from PIL import Image

NAME_TO_MODEL_SPECS_MAP = {
    "720x720_5s": {
        "path": "model.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 720 * 720,
        "formatter": lambda text: re.sub(r"Audio:\s*(.*)", r"<AUDCAP>\1<ENDAUDCAP>", text, flags=re.S)
    },
    "960x960_5s": {
        "path": "model_960x960.safetensors",
        "video_latent_length": 31,
        "audio_latent_length": 157,
        "video_area": 960 * 960,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S)
    }, 
    "960x960_10s": {
        "path": "model_960x960_10s.safetensors",
        "video_latent_length": 61,
        "audio_latent_length": 314,
        "video_area": 960 * 960,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S)
    }
}


class OviEngine(WanShared):
    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        
        self.transformer_name = None 
        for component in self.config.get("components", []):
            if component.get("type") == "transformer":
                self.transformer_name = component.get("name")
                break
        if self.transformer_name is None:
            raise ValueError("Transformer component not found in config")
        logging.info(f"Using transformer: {self.transformer_name}")
    
    @staticmethod
    def snap_hw_to_multiple_of_32(h: int, w: int, area = 720 * 720) -> tuple[int, int]:
        """
        Scale (h, w) to match a target area if provided, then snap both
        dimensions to the nearest multiple of 32 (min 32).

        Args:
            h (int): original height
            w (int): original width
            area (int, optional): target area to scale to. If None, no scaling is applied.

        Returns:
            (new_h, new_w): dimensions adjusted
        """
        if h <= 0 or w <= 0:
            raise ValueError(f"h and w must be positive, got {(h, w)}")

        # If a target area is provided, rescale h, w proportionally
        if area is not None and area > 0:
            current_area = h * w
            scale = math.sqrt(area / float(current_area))
            h = int(round(h * scale))
            w = int(round(w * scale))

        # Snap to nearest multiple of 32
        def _n32(x: int) -> int:
            return max(32, int(round(x / 32)) * 32)

        return _n32(h), _n32(w)
    
    @staticmethod
    def preprocess_image_tensor(image_path, device, target_dtype, h_w_multiple_of=32, resize_total_area=720*720):
        """Preprocess video data into standardized tensor format and (optionally) resize area."""
        def _parse_area(val):
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, (tuple, list)) and len(val) == 2:
                return int(val[0]) * int(val[1])
            if isinstance(val, str):
                m = re.match(r"\s*(\d+)\s*[x\*\s]\s*(\d+)\s*$", val, flags=re.IGNORECASE)
                if m:
                    return int(m.group(1)) * int(m.group(2))
                if val.strip().isdigit():
                    return int(val.strip())
            raise ValueError(f"resize_total_area={val!r} is not a valid area or WxH.")

        def _best_hw_for_area(h, w, area_target, multiple):
            if area_target <= 0:
                return h, w
            ratio_wh = w / float(h)
            area_unit = multiple * multiple
            tgt_units = max(1, area_target // area_unit)
            p0 = max(1, int(round(np.sqrt(tgt_units / max(ratio_wh, 1e-8)))))
            candidates = []
            for dp in range(-3, 4):
                p = max(1, p0 + dp)
                q = max(1, int(round(p * ratio_wh)))
                H = p * multiple
                W = q * multiple
                candidates.append((H, W))
            scale = np.sqrt(area_target / (h * float(w)))
            H_sc = max(multiple, int(round(h * scale / multiple)) * multiple)
            W_sc = max(multiple, int(round(w * scale / multiple)) * multiple)
            candidates.append((H_sc, W_sc))
            def score(HW):
                H, W = HW
                area = H * W
                return (abs(area - area_target), abs((W / max(H, 1e-8)) - ratio_wh))
            H_best, W_best = min(candidates, key=score)
            return H_best, W_best

        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            assert isinstance(image_path, Image.Image)
            if image_path.mode != "RGB":
                image_path = image_path.convert("RGB")
            image = np.array(image_path)

        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(image).float().to(device, dtype=target_dtype).unsqueeze(0) ## b c h w
        image_tensor = image_tensor * 2.0 - 1.0 ## -1 to 1

        _, c, h, w = image_tensor.shape
        area_target = _parse_area(resize_total_area)
        if area_target is not None:
            target_h, target_w = _best_hw_for_area(h, w, area_target, h_w_multiple_of)
        else:
            target_h = (h // h_w_multiple_of) * h_w_multiple_of
            target_w = (w // h_w_multiple_of) * h_w_multiple_of

        target_h = max(h_w_multiple_of, int(target_h))
        target_w = max(h_w_multiple_of, int(target_w))

        if (h != target_h) or (w != target_w):
            image_tensor = torch.nn.functional.interpolate(
                image_tensor,
                size=(target_h, target_w),
                mode='bicubic',
                align_corners=False
            )

        return image_tensor
    
    def _get_model_specs(self, transformer_name: str | None = None) -> dict:
        if transformer_name is None:
            transformer_name = self.transformer_name
        if transformer_name not in NAME_TO_MODEL_SPECS_MAP:
            logging.warning(f"Transformer name {transformer_name} not found in map. Defaulting to 960x960_10s")
            transformer_name = "960x960_10s"
        return NAME_TO_MODEL_SPECS_MAP[transformer_name]

    def run(self,
            prompt: str,
            image: InputImage | None = None,
            height: int = 720,
            width: int = 720,
            seed: int = 100,
            num_inference_steps: int = 50,
            video_guidance_scale: float = 5.0,
            audio_guidance_scale: float = 4.0,
            slg_layer: int = 11,
            negative_prompt: str = "",
            audio_negative_prompt: str = "",
            offload: bool = True,
            output_type: str = "pil",
            **kwargs
            ):
         # Audio VAE
        specs = self._get_model_specs()    
        video_latent_length = specs["video_latent_length"]
        audio_latent_length = specs["audio_latent_length"]
        target_area = specs["video_area"]
        text_formatter = specs["formatter"]
        if image is not None:
            image = self._load_image(image)

        device = self.device
        target_dtype = torch.bfloat16 # Ovi uses bfloat16 by default
        
        if not self.vae:
             self.load_component_by_name("transformer_vae")
        self.to_device(self.transformer_vae)
        
        # Audio VAE
        if not hasattr(self, "audio_vae") or self.audio_vae is None:
             self.load_component_by_name("audio_vae")
    
        # Text formatting
        formatted_text_prompt = text_formatter(prompt)
        if formatted_text_prompt != prompt:
            logging.info(f"Formatted prompt: {formatted_text_prompt}")
            prompt = formatted_text_prompt

        # Encode Prompts
        # Ovi expects: [prompt, video_neg, audio_neg]
        prompts = [prompt, negative_prompt, audio_negative_prompt]
        if not self.text_encoder:
            self.load_component_by_type("text_encoder")
        # Use .encode() from TextEncoder
        raw_output, attention_mask = self.text_encoder.encode(
            prompts,
            device=device,
            dtype=target_dtype,
            use_attention_mask=True,
            max_sequence_length=self.text_encoder.config.get("max_sequence_length", 512),
            pad_with_zero=False,
            clean_text=False,
            output_type="raw",
            return_attention_mask=True
        )
        seq_lens = attention_mask.sum(dim=1).long()
        
        if hasattr(raw_output, "last_hidden_state"):
            text_embeddings = torch.stack([u for u in raw_output.last_hidden_state])
        else:
            text_embeddings = raw_output
        
        if self.text_encoder.enable_cache:
            prompt_hash = self.text_encoder.get_prompt_hash(
                prompts,
                device=device,
                dtype=target_dtype,
                use_attention_mask=True,
                max_sequence_length=self.text_encoder.config.get("max_sequence_length", 512),
                pad_with_zero=False,
                clean_text=False,
                output_type="raw",
                return_attention_mask=True
            )
            self.text_encoder.cache_prompt(prompt_hash, text_embeddings, attention_mask)
        
        
        text_embeddings = [u[:v] for u, v in zip(text_embeddings, seq_lens)]
        
        
        
        text_embeddings_audio_pos = text_embeddings[0]
        text_embeddings_video_pos = text_embeddings[0]
        text_embeddings_video_neg = text_embeddings[1]
        text_embeddings_audio_neg = text_embeddings[2]
        
        if offload:
            self._offload(self.text_encoder)

        # 2. Image Encoding (I2V)
        is_t2v = image is None
        is_i2v = not is_t2v
        first_frame = None
        latents_images = None
        video_latent_h, video_latent_w = 0, 0

        if is_i2v:
            if not self.vae:
                self.load_component_by_type("vae")
            self.to_device(self.vae)
            
            first_frame = self.preprocess_image_tensor(image, device, target_dtype, resize_total_area=target_area)
    
                # Add temporal dim: (1, 3, 1, H, W)
            input_tensor = first_frame.unsqueeze(2)
            latents_images = self.vae_encode(input_tensor, dtype=target_dtype)
            latents_images = latents_images.squeeze(0) # (C, T, H, W)
            
            video_latent_h, video_latent_w = latents_images.shape[2], latents_images.shape[3]
            
            if offload:
                self._offload(self.vae)
        else:
            # T2V
            video_h, video_w = height, width
            video_h, video_w = self.snap_hw_to_multiple_of_32(video_h, video_w, area=target_area)
            video_latent_h, video_latent_w = video_h // 16, video_w // 16

        # 3. Schedulers
        if not hasattr(self, "transformer_scheduler") or self.transformer_scheduler is None:
             self.load_component_by_name("transformer_scheduler")

        self.to_device(self.transformer_scheduler)
        
        if not hasattr(self, "audio_scheduler") or self.audio_scheduler is None:
             self.load_component_by_name("audio_scheduler")
        self.to_device(self.audio_scheduler)

        # Set timesteps

        timesteps_video, num_inference_steps_video = self._get_timesteps(self.transformer_scheduler, num_inference_steps)
        timesteps_audio, num_inference_steps_audio = self._get_timesteps(self.audio_scheduler, num_inference_steps)

        # 4. Latents Initialization
        transformer_config = self.load_config_by_type("transformer") 
        video_latent_channel = transformer_config.get("video", {}).get("in_dim", 48)
        audio_latent_channel = transformer_config.get("audio", {}).get("in_dim", 20)

        video_noise = randn_tensor(shape=(video_latent_channel, video_latent_length, video_latent_h, video_latent_w), device=device, dtype=target_dtype, generator=torch.Generator(device=device).manual_seed(seed))
        audio_noise = randn_tensor(shape=(audio_latent_length, audio_latent_channel), device=device, dtype=target_dtype, generator=torch.Generator(device=device).manual_seed(seed))
        max_seq_len_audio = audio_noise.shape[0]
        

        
        # 5. Transformer
        if not self.transformer:
            self.load_component_by_type("transformer")
        self.to_device(self.transformer)
        
        _patch_size_h, _patch_size_w = self.transformer.video_model.patch_size[1], self.transformer.video_model.patch_size[2]
        max_seq_len_video = video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (_patch_size_h*_patch_size_w)

        with torch.amp.autocast(device.type, enabled=target_dtype != torch.float32, dtype=target_dtype):
            for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio)), total=len(timesteps_video)):
                timestep_input = torch.full((1,), t_v, device=self.device)

                if is_i2v:
                    video_noise[:, :1] = latents_images

                # Positive forward
                pos_forward_args = {
                    'audio_context': [text_embeddings_audio_pos],
                    'vid_context': [text_embeddings_video_pos],
                    'vid_seq_len': max_seq_len_video,
                    'audio_seq_len': max_seq_len_audio,
                    'first_frame_is_clean': is_i2v
                }
                
                

                pred_vid_pos, pred_audio_pos = self.transformer(
                    vid=[video_noise],
                    audio=[audio_noise],
                    t=timestep_input,
                    **pos_forward_args
                )

                # Negative forward
                neg_forward_args = {
                    'audio_context': [text_embeddings_audio_neg],
                    'vid_context': [text_embeddings_video_neg],
                    'vid_seq_len': max_seq_len_video,
                    'audio_seq_len': max_seq_len_audio,
                    'first_frame_is_clean': is_i2v,
                    'slg_layer': slg_layer
                }

                pred_vid_neg, pred_audio_neg = self.transformer(
                    vid=[video_noise],
                    audio=[audio_noise],
                    t=timestep_input,
                    **neg_forward_args
                )
                

                # Guidance
                pred_video_guided = pred_vid_neg[0] + video_guidance_scale * (pred_vid_pos[0] - pred_vid_neg[0])
                pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])

                # Step
                video_noise = self.transformer_scheduler.step(
                    pred_video_guided.unsqueeze(0), t_v, video_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)

                audio_noise = self.audio_scheduler.step(
                    pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)

        if offload:
            self._offload(self.transformer)

        if is_i2v:
            video_noise[:, :1] = latents_images

        # 6. Decoding
        # Video VAE
        if not self.vae:
             self.load_component_by_name("transformer_vae")
        self.to_device(self.transformer_vae)
        
        # Audio VAE
        if not hasattr(self, "audio_vae") or self.audio_vae is None:
             self.load_component_by_name("audio_vae")
            
        self.audio_vae.tod.remove_weight_norm()
        self.to_device(self.audio_vae)

        # Decode Audio
        audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2) # 1, c, l
        generated_audio = self.vae_decode(audio_latents_for_vae, component_name="audio_vae")
        generated_audio = generated_audio.squeeze().cpu().float().numpy()

        # Decode Video
        video_latents_for_vae = video_noise.unsqueeze(0) # 1, c, f, h, w
        generated_video_tensor = self.vae_decode(video_latents_for_vae, component_name="transformer_vae")
        # Post-process
        generated_video_frames = self._tensor_to_frames(generated_video_tensor, output_type=output_type)

        if offload:
            self._offload(self.vae)
            self._offload(self.audio_vae)

        return generated_video_tensor, generated_audio
