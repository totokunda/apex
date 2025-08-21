# src/postprocessors/rife_postprocessor.py

from __future__ import annotations

import os
import math
import zipfile
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F
from loguru import logger
from src.utils.defaults import DEFAULT_POSTPROCESSOR_SAVE_PATH, DEFAULT_DEVICE
from src.postprocess.rife.module import Model
from src.postprocess.rife.ssim import ssim_matlab
from src.postprocess.base import BasePostprocessor, PostprocessorCategory, postprocessor_registry 
from PIL import Image
from collections import deque
from tqdm import tqdm


def _load_rife_model(model_dir: str, device: torch.device, logger=None):
    """
    Import and load the RIFE model dynamically, matching the fallback chain used in the RIFE scripts.
    """
    model = Model()
    model.load_model(model_dir)
    model.eval()
    model.device()
    if logger:
        logger.info(f"Loaded RIFE model from: {model_dir}")
    return model


@postprocessor_registry("video.rife")
class RifePostprocessor(BasePostprocessor):
    """
    RIFE frame interpolation postprocessor.

    Usage:
        pp = RifePostprocessor(engine, target_fps=60)  # or exp=1 for 2x
        video_out = pp(video_in)  # returns tensor (1, C, F_out, H, W) in [-1, 1]
    """

    def __init__(
        self,
        engine = None,
        # Targeting / control
        target_fps: Optional[float] = None,
        exp: Optional[int] = None,            # if set, overrides target_fps
        scale: float = 1.0,                   # RIFE's 'scale' (try 0.5 for UHD)
        # SSIM gating (optional, mirrors RIFE script behavior)
        ssim_static_thresh: float = 0.996,    # treat near-identical frames as static
        ssim_hardcut_thresh: float = 0.20,    # when < thresh, fill with duplicates
        # Weights / code locations
        device: torch.device = DEFAULT_DEVICE,
        save_path: str = DEFAULT_POSTPROCESSOR_SAVE_PATH,
        model_dir: str = 'https://drive.google.com/uc?id=1gViYvvQrtETBgU1w8axZSsr7YUuw31uy',
        **kwargs: Any,
    ):
        super().__init__(engine, PostprocessorCategory.FRAME_INTERPOLATION, **kwargs)
        self.save_path = save_path
        self.scale = scale
        self.target_fps = target_fps
        self.exp = exp
        self.ssim_static_thresh = ssim_static_thresh
        self.ssim_hardcut_thresh = ssim_hardcut_thresh
        model_dir = self.download_rife(model_dir, save_path=save_path)

        # Build model
        self.device = device
        self.model = _load_rife_model(model_dir, self.device, logger=logger)
        
    def download_rife(self, model_dir: str, save_path: str):
        if self._is_url(model_dir):
            # check if the save_path exists
            save_rife_path = os.path.join(save_path, 'rife')
            if os.path.exists(os.path.join(save_rife_path, 'train_log')):
                return os.path.join(save_rife_path, 'train_log')
            os.makedirs(save_rife_path, exist_ok=True)
            path = self._download_from_url(model_dir, save_path=save_path)
            # extract from train_log directory and only extract the flownet.pkl file

            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extract('train_log/flownet.pkl', save_rife_path)
            os.remove(path)
            return os.path.join(save_rife_path, 'train_log')
        else:
            return model_dir

    @torch.no_grad()
    def __call__(self, video, **kwargs) -> List[Image.Image]:
        """
        RIFE interpolation that mirrors the provided reference script:
          - multi expansion (default 2, or 2**exp when exp != 1)
          - padding to multiples of max(128, int(128/scale))
          - version-aware inference: v>=3.9 uses t ∈ (0,1), else recursive midpoint
          - SSIM gates for static frames (>0.996) and hard cuts (<0.2)

        Returns:
            (1, C, F_out, H, W) in [-1, 1]
        """

        # ---- 1) Load frames & original FPS via LoaderMixin ----
        frames, orig_fps = self._load_video(video, return_fps=True)
        if not frames:
            return torch.empty(1, 3, 0, 0, 0)

        self.target_fps = kwargs.get("target_fps", self.target_fps)
        self.exp = kwargs.get("exp", self.exp)
        self.scale = kwargs.get("scale", self.scale)
        self.ssim_static_thresh = kwargs.get("ssim_static_thresh", self.ssim_static_thresh)
        self.ssim_hardcut_thresh = kwargs.get("ssim_hardcut_thresh", self.ssim_hardcut_thresh)


        # ---- 2) Determine multi (like argparse logic) ----
        # Priority: explicit kwargs.multi -> exp (2**exp if exp != 1) -> target_fps vs orig_fps -> default 2
        multi = int(kwargs.get("multi", 0)) if kwargs.get("multi", None) is not None else 0
        if multi < 2:
            if self.exp is not None:
                multi = (2 ** int(self.exp)) if int(self.exp) != 1 else 2
            elif self.target_fps and orig_fps and orig_fps > 0:
                # Allow any integer multi (script multiplies fps by multi)
                multi = max(2, int(round(float(self.target_fps) / float(orig_fps))))
            else:
                multi = 2
        n_inserts = multi - 1

        # ---- 3) Tensorify frames to device in [0,1], RGB, (F, C, H, W) ----
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t_list = []
        for im in frames:
            arr = np.asarray(im)  # HWC
            if arr.ndim == 2:
                arr = np.expand_dims(arr, 2)
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)  # grayscale → RGB
            t = torch.from_numpy(arr.transpose(2, 0, 1)).to(dev).float() / 255.0  # C,H,W
            t_list.append(t.unsqueeze(0))
        seq = torch.cat(t_list, dim=0)  # (F, C, H, W)

        F_total, C, H, W = seq.shape

        # ---- 4) Padding logic (like script: multiples of 128/scale) ----
        tmp = max(128, int(128 / self.scale))
        ph = ((H - 1) // tmp + 1) * tmp
        pw = ((W - 1) // tmp + 1) * tmp
        padding = (0, pw - W, 0, ph - H)

        def pad_image(x: torch.Tensor) -> torch.Tensor:
            x = F.pad(x, padding)
            return x

        def unpad_image(x: torch.Tensor) -> torch.Tensor:
            return x[..., :H, :W]

        # ---- 5) Version-aware make_inference (script parity) ----
        version = float(getattr(self.model, "version", 0))
        if not hasattr(self.model, "version"):
            # Mirror the script's behavior
            setattr(self.model, "version", version)

        def make_inference(I0: torch.Tensor, I1: torch.Tensor, n: int):
            if n <= 0:
                return []
            if version >= 3.9:
                # Evenly spaced t in (0,1) with n samples
                return [self.model.inference(I0, I1, (i + 1) * 1.0 / (n + 1), self.scale) for i in range(n)]
            else:
                # Classic recursive midpoint
                mid = self.model.inference(I0, I1, self.scale)
                if n == 1:
                    return [mid]
                left = make_inference(I0, mid, n // 2)
                right = make_inference(mid, I1, n // 2)
                return ([*left, mid, *right] if (n % 2) else [*left, *right])

        # ---- 6) SSIM helper (32x32 like script) ----
        def to_small(x: torch.Tensor) -> torch.Tensor:
            # x is 1,C,H,W in [0,1]
            return F.interpolate(x, (32, 32), mode="bilinear", align_corners=False)

        # Prefer the RIFE SSIM if available; otherwise fallback already set in module
        def ssim_val(a: torch.Tensor, b: torch.Tensor) -> float:
            a32 = to_small(a)
            b32 = to_small(b)
            try:
                from src.postprocess.rife.ssim import ssim_matlab as _ssim
                return float(_ssim(a32[:, :3], b32[:, :3]))
            except Exception:
                # Lightweight proxy if module not importable
                return max(0.0, 1.0 - (a32 - b32).abs().mean().item() * 4.0)

        # ---- 7) Main loop (mirrors script ordering & gates) ----
        # Use a deque so we can "push back" the real next frame when we insert a synthesized mid-frame on static scenes.
        remaining = deque([seq[i : i + 1, ...] for i in range(1, F_total)])  # list of 1,C,H,W
        cur = seq[0:1, ...]  # 1,C,H,W
        cur_pad = pad_image(cur)

        out_frames: List[torch.Tensor] = []
        # The script writes the first original frame before any inserts of the first pair.
        out_frames.append(unpad_image(cur_pad))  # write first frame
        
        pbar = tqdm(total=F_total, desc="RIFE Interpolation")

        while True:
            if not remaining:
                break
            nxt = remaining.popleft()  # 1,C,H,W
            nxt_pad = pad_image(nxt)

            # SSIM gate between padded smalls (script computes on padded tensors downscaled to 32x32)
            ssim = ssim_val(cur_pad, nxt_pad)

            # Branching like the script
            if ssim > 0.996:
                # Read a new frame: here we synthesize the MID and push the real nxt back for the next iteration
                mid_pad = self.model.inference(cur_pad, nxt_pad, self.scale) if version < 3.9 \
                          else self.model.inference(cur_pad, nxt_pad, 0.5, self.scale)
                # Recompute ssim as in script (not used for branching beyond parity, but we keep it)
                _ = ssim_val(cur_pad, mid_pad)

                # Inserts between cur and mid
                inserts = make_inference(cur_pad, mid_pad, n_inserts)
                for m in inserts:
                    out_frames.append(unpad_image(m))
                # Then write the mid (this becomes the next "original" frame)
                out_frames.append(unpad_image(mid_pad))

                # Push the real next back for the following loop
                remaining.appendleft(nxt)
                # Advance cur to mid
                cur_pad = pad_image(unpad_image(mid_pad))  # keep padding fresh
                pbar.update(1)
                continue

            elif ssim < 0.2:
                # Hard cut: duplicate previous frame (I0) n_inserts times
                inserts = [cur_pad for _ in range(n_inserts)]
                for m in inserts:
                    out_frames.append(unpad_image(m))
                # Write the real next frame
                out_frames.append(unpad_image(nxt_pad))
                # Advance
                cur_pad = nxt_pad
                pbar.update(1)
                continue

            else:
                # Normal case: interpolate between cur and nxt
                inserts = make_inference(cur_pad, nxt_pad, n_inserts)
                for m in inserts:
                    out_frames.append(unpad_image(m))
                out_frames.append(unpad_image(nxt_pad))
                cur_pad = nxt_pad
                pbar.update(1)
                continue

        # ---- 8) Stack & normalize to [-1,1], (1, C, F_out, H, W) ----
        torch_frames = torch.stack(out_frames, dim=0)          # (F_out, 1, C, H, W)
        torch_frames = torch_frames.squeeze(1)                 # (F_out, C, H, W)
        pbar.close()
        frames = torch_frames.clamp(0, 1)
        # convert to PIL Images
        frames = [Image.fromarray((frame.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)) for frame in frames]
        return frames
