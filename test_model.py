from dotenv import load_dotenv
load_dotenv()
from src.engine.registry import UniversalEngine
import json 
from diffusers.utils import export_to_video
import numpy as np
from typing import Optional
import tempfile
import soundfile as wavfile

def save_video(
    output_path: str,
    video_numpy: np.ndarray,
    audio_numpy: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    fps: int = 24,
) -> str:
    """
    Combine a sequence of video frames with an optional audio track and save as an MP4.

    Args:
        output_path (str): Path to the output MP4 file.
        video_numpy (np.ndarray): Numpy array of frames. Shape (C, F, H, W).
                                  Values can be in range [-1, 1] or [0, 255].
        audio_numpy (Optional[np.ndarray]): 1D or 2D numpy array of audio samples, range [-1, 1].
        sample_rate (int): Sample rate of the audio in Hz. Defaults to 16000.
        fps (int): Frames per second for the video. Defaults to 24.

    Returns:
        str: Path to the saved MP4 file.
    """

    # Validate inputs
    assert isinstance(video_numpy, np.ndarray), "video_numpy must be a numpy array"
    assert video_numpy.ndim == 4, "video_numpy must have shape (C, F, H, W)"
    assert video_numpy.shape[0] in {1, 3}, "video_numpy must have 1 or 3 channels"

    if audio_numpy is not None:
        assert isinstance(audio_numpy, np.ndarray), "audio_numpy must be a numpy array"
        assert np.abs(audio_numpy).max() <= 1.0, "audio_numpy values must be in range [-1, 1]"

    # Reorder dimensions: (C, F, H, W) → (F, H, W, C)
    video_numpy = video_numpy.transpose(1, 2, 3, 0)

    # Normalize frames if values are in [-1, 1]
    if video_numpy.max() <= 1.0:
        video_numpy = np.clip(video_numpy, -1, 1)
        video_numpy = ((video_numpy + 1) / 2 * 255).astype(np.uint8)
    else:
        video_numpy = video_numpy.astype(np.uint8)

    # Convert numpy array to a list of frames
    frames = list(video_numpy)

    # Create video clip
    clip = ImageSequenceClip(frames, fps=fps)

    # Add audio if provided
    if audio_numpy is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", mode='wb', delete=False) as temp_audio_file:
            wavfile.write(
                temp_audio_file.name,
                (audio_numpy * 32767).astype(np.int16),
                sample_rate,
            )
            audio_clip = AudioFileClip(temp_audio_file.name)
            final_clip = clip.set_audio(audio_clip)
    else:
        final_clip = clip

    # Write final video to disk
    final_clip.write_videofile(
        output_path, codec="libx264", audio_codec="aac", fps=fps, verbose=False, logger=None
    )
    final_clip.close()

    return output_path

with open("inputs.json", "r") as f:
   data = json.load(f)


engine_kwargs = data["engine_kwargs"]
inputs = data["inputs"]


yaml_path = engine_kwargs.get("yaml_path")
engine = UniversalEngine(yaml_path=yaml_path)

out = engine.run(
    **inputs
)

print(len(out[0]))

export_to_video(out[0], "output_16.mp4", fps=16)
export_to_video(out[0], "output_24.mp4", fps=24)

