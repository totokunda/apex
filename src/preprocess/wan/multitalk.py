from typing import Dict, Any, Union, List, Optional
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor
from src.utils.defaults import DEFAULT_PREPROCESSOR_SAVE_PATH
from src.preprocess.base import (
    BasePreprocessor,
    preprocessor_registry,
    PreprocessorType,
)
from src.mixins.loader_mixin import LoaderMixin
from src.mixins.offload_mixin import OffloadMixin
import torch
import librosa
import numpy as np
from einops import rearrange
import soundfile as sf
import os
import tempfile


@preprocessor_registry("wan.multitalk")
class MultiTalkPreprocessor(BasePreprocessor, LoaderMixin, OffloadMixin):
    def __init__(
        self,
        model_path: str,
        save_path: str = DEFAULT_PREPROCESSOR_SAVE_PATH,
        device: str = "cuda",
    ):
        super().__init__(
            model_path, save_path, preprocessor_type=PreprocessorType.AUDIO
        )
        self.model_path = model_path
        self.save_path = save_path
        self.device = device

        # Initialize Wav2Vec2 components if available
        try:
            # Try to import and initialize the wav2vec model for audio feature extraction
            from transformers import Wav2Vec2Model

            self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                model_path, cache_dir=save_path
            )
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_path, cache_dir=save_path
            )
        except Exception as e:
            print(f"Warning: Could not load Wav2Vec2 model from {model_path}: {e}")
            self.wav2vec_model = None
            self.wav2vec_feature_extractor = None

    def __call__(
        self,
        image: Union[Image.Image, List[Image.Image], str, np.ndarray, torch.Tensor],
        audio: Optional[Union[str, List[str]]] = None,
        audio_paths: Optional[Dict[str, str]] = None,
        audio_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        audio_type: str = "para",  # "para" for parallel, "add" for sequential
        num_frames: int = 81,
        vae_scale: int = 4,
        dtype: torch.dtype = torch.float32,
        bbox: Optional[Dict[str, List[float]]] = None,
        face_scale: float = 0.05,
        **kwargs,
    ):
        """
        Preprocess inputs for MultiTalk model.

        Args:
            image: Input conditioning image
            audio: Single audio file path or list of audio file paths
            audio_paths: Dictionary mapping person names to audio file paths
            audio_embeddings: Pre-computed audio embeddings
            audio_type: Type of audio combination ("para" or "add")
            num_frames: Number of video frames to generate
            vae_scale: VAE temporal downsampling scale
            dtype: Target data type
            bbox: Bounding boxes for multiple people
            face_scale: Scale factor for face regions
        """

        # Load and process image
        loaded_image = self._load_image(image)

        # Determine number of people from inputs
        if audio_paths is not None:
            human_num = len(audio_paths)
            audio_files = list(audio_paths.values())
        elif audio_embeddings is not None:
            human_num = len(audio_embeddings)
            audio_files = None
        elif isinstance(audio, list):
            human_num = len(audio)
            audio_files = audio
        elif audio is not None:
            human_num = 1
            audio_files = [audio]
        else:
            raise ValueError(
                "Must provide either audio paths, embeddings, or audio files"
            )

        # Process audio inputs
        if audio_embeddings is not None:
            # Use pre-computed embeddings
            processed_audio = self._process_audio_embeddings(
                audio_embeddings, num_frames, vae_scale
            )
        else:
            # Extract features from audio files
            processed_audio = self._process_audio_files(
                audio_files, num_frames, vae_scale, audio_type
            )

        # Generate human masks for spatial attention
        human_masks = self._generate_human_masks(
            loaded_image, human_num, bbox, face_scale
        )

        # Prepare outputs
        result = {
            "image": loaded_image,
            "audio_embeddings": processed_audio,
            "human_masks": human_masks,
            "human_num": human_num,
            "num_frames": num_frames,
        }

        return result

    def _process_audio_files(
        self,
        audio_files: List[str],
        num_frames: int,
        vae_scale: int,
        audio_type: str = "para",
    ) -> torch.Tensor:
        """Process audio files into embeddings."""
        if self.wav2vec_model is None:
            raise ValueError(
                "Wav2Vec2 model not available. Please provide pre-computed embeddings."
            )

        # Load and prepare audio
        audio_arrays = []
        for audio_file in audio_files:
            if audio_file is None or audio_file == "None":
                # Create silent audio placeholder
                # Use duration based on typical audio length for video
                duration_seconds = num_frames / 25.0  # Assume 25 fps
                audio_array = np.zeros(int(duration_seconds * 16000))
            else:
                audio_array = self._load_audio_file(audio_file)
            audio_arrays.append(audio_array)

        # Combine audio based on type
        if audio_type == "para":
            # Parallel - keep separate
            combined_audio = audio_arrays
        elif audio_type == "add":
            # Sequential - concatenate
            max_len = max(len(arr) for arr in audio_arrays)
            padded_arrays = []
            for arr in audio_arrays:
                padded = np.concatenate([arr, np.zeros(max_len - len(arr))])
                padded_arrays.append(padded)

            # Create sequential version
            combined_audio = []
            for i, arr in enumerate(padded_arrays):
                if i == 0:
                    seq_audio = np.concatenate([arr, np.zeros(max_len)])
                else:
                    seq_audio = np.concatenate([np.zeros(max_len), arr])
                combined_audio.append(seq_audio)
        else:
            raise ValueError(f"Unknown audio_type: {audio_type}")

        # Extract features using Wav2Vec2
        audio_embeddings = []
        for audio_array in combined_audio:
            embedding = self._extract_audio_features(audio_array, num_frames)
            audio_embeddings.append(embedding)

        # Convert to proper format: [num_humans, frames, window, blocks, channels]
        # Stack embeddings for multiple humans
        if len(audio_embeddings) > 1:
            stacked_embeddings = torch.stack(audio_embeddings, dim=0)
        else:
            stacked_embeddings = audio_embeddings[0].unsqueeze(0)

        return stacked_embeddings

    def _process_audio_embeddings(
        self, audio_embeddings: Dict[str, torch.Tensor], num_frames: int, vae_scale: int
    ) -> torch.Tensor:
        """Process pre-computed audio embeddings."""
        embeddings_list = []
        for person_key in sorted(audio_embeddings.keys()):
            embedding = audio_embeddings[person_key]
            # Ensure proper format and length
            if embedding.shape[0] < num_frames:
                # Pad if too short
                padding_frames = num_frames - embedding.shape[0]
                padding = torch.zeros(padding_frames, *embedding.shape[1:])
                embedding = torch.cat([embedding, padding], dim=0)
            elif embedding.shape[0] > num_frames:
                # Truncate if too long
                embedding = embedding[:num_frames]

            embeddings_list.append(embedding)

        # Stack embeddings: [num_humans, frames, ...]
        stacked_embeddings = torch.stack(embeddings_list, dim=0)
        return stacked_embeddings

    def _load_audio_file(self, audio_path: str, sample_rate: int = 16000) -> np.ndarray:
        """Load and normalize audio file."""
        # Check if it's a video file
        video_extensions = [".mp4", ".mov", ".avi", ".mkv"]
        if any(audio_path.lower().endswith(ext) for ext in video_extensions):
            return self._extract_audio_from_video(audio_path, sample_rate)
        else:
            # Load audio file directly
            audio_array, sr = librosa.load(audio_path, sr=sample_rate)
            return self._normalize_audio(audio_array, sr)

    def _extract_audio_from_video(
        self, video_path: str, sample_rate: int
    ) -> np.ndarray:
        """Extract audio from video file."""
        import subprocess

        # Create temporary file for extracted audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_audio_path = temp_file.name

        try:
            # Use ffmpeg to extract audio
            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                temp_audio_path,
            ]
            subprocess.run(ffmpeg_command, check=True, capture_output=True)

            # Load the extracted audio
            audio_array, sr = librosa.load(temp_audio_path, sr=sample_rate)
            return self._normalize_audio(audio_array, sr)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def _normalize_audio(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """Normalize audio loudness."""
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio_array)
            if abs(loudness) < 100:  # Valid loudness measurement
                normalized_audio = pyln.normalize.loudness(audio_array, loudness, -23)
                return normalized_audio
        except ImportError:
            pass  # pyloudnorm not available, skip normalization

        return audio_array

    def _extract_audio_features(
        self, audio_array: np.ndarray, num_frames: int
    ) -> torch.Tensor:
        """Extract audio features using Wav2Vec2."""
        if self.wav2vec_model is None or self.wav2vec_feature_extractor is None:
            raise ValueError("Wav2Vec2 components not available")

        # Process audio with feature extractor
        audio_features = self.wav2vec_feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_values

        # Extract embeddings using Wav2Vec2 model
        with torch.no_grad():
            embeddings = self.wav2vec_model(audio_features, output_hidden_states=True)

        # Stack hidden states from different layers
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        # Ensure we have the right number of frames
        target_length = num_frames
        if audio_emb.shape[0] < target_length:
            # Pad if too short
            padding_frames = target_length - audio_emb.shape[0]
            padding = torch.zeros(padding_frames, *audio_emb.shape[1:])
            audio_emb = torch.cat([audio_emb, padding], dim=0)
        elif audio_emb.shape[0] > target_length:
            # Interpolate if too long
            audio_emb = audio_emb[:target_length]

        return audio_emb

    def _generate_human_masks(
        self,
        image: Image.Image,
        human_num: int,
        bbox: Optional[Dict[str, List[float]]] = None,
        face_scale: float = 0.05,
    ) -> torch.Tensor:
        """Generate spatial masks for humans in the image."""
        height, width = image.height, image.width

        if human_num == 1:
            # Single person - use full image
            human_mask = torch.ones([height, width])
            background_mask = torch.ones([height, width])
            masks = [human_mask, torch.ones_like(human_mask), background_mask]

        elif human_num == 2:
            if bbox is not None:
                # Use provided bounding boxes
                background_mask = torch.zeros([height, width])
                human_masks = []

                for person_key in sorted(bbox.keys()):
                    x_min, y_min, x_max, y_max = bbox[person_key]
                    human_mask = torch.zeros([height, width])
                    human_mask[int(x_min) : int(x_max), int(y_min) : int(y_max)] = 1
                    background_mask += human_mask
                    human_masks.append(human_mask)

                # Background is where no humans are
                background_mask = torch.where(
                    background_mask > 0, torch.tensor(0), torch.tensor(1)
                )
                human_masks.append(background_mask)
                masks = human_masks
            else:
                # Default: split image in half vertically
                x_min, x_max = int(height * face_scale), int(height * (1 - face_scale))

                # Left person
                human_mask1 = torch.zeros([height, width])
                lefty_min, lefty_max = int((width // 2) * face_scale), int(
                    (width // 2) * (1 - face_scale)
                )
                human_mask1[x_min:x_max, lefty_min:lefty_max] = 1

                # Right person
                human_mask2 = torch.zeros([height, width])
                righty_min = int((width // 2) * face_scale + (width // 2))
                righty_max = int((width // 2) * (1 - face_scale) + (width // 2))
                human_mask2[x_min:x_max, righty_min:righty_max] = 1

                # Background
                background_mask = torch.where(
                    (human_mask1 + human_mask2) > 0, torch.tensor(0), torch.tensor(1)
                )

                masks = [human_mask1, human_mask2, background_mask]
        else:
            raise ValueError(f"Unsupported number of humans: {human_num}")

        return torch.stack(masks, dim=0).float()

    def _encode_image(self, image: Image.Image):
        """Encode image if needed."""
        # This method can be extended for additional image processing
        return image

    def _encode_prompt(self, prompt: str):
        """Encode text prompt if needed."""
        # This method can be extended for additional text processing
        return prompt

    def _encode_audio(self, audio_path: str):
        """Encode audio if needed."""
        # This method can be extended for additional audio processing
        return self._load_audio_file(audio_path)
