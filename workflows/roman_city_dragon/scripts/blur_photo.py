#!/usr/bin/env python3
"""
motion_blur.py — Apply directional motion blur to an image.

Usage:
  python motion_blur.py input.jpg -o output.jpg --length 25 --angle 45
  # Optional softness (feathering) and border mode:
  python motion_blur.py input.png -o blurred.png --length 35 --angle -20 --sigma 1.2 --border reflect

Args:
  input            Path to input image
  -o, --output     Path to save the blurred image (default: <input>_motion.jpg)
  --length         Blur length in pixels (kernel size). Must be >= 1.
  --angle          Blur direction in degrees. 0 = left→right, 90 = top→bottom.
  --sigma          Optional softness of the blur line (0 = sharp/boxy). Typical 0–2.
  --border         How to handle edges: reflect | replicate | constant | wrap (default: reflect)

Requires: opencv-python, numpy
Install:  pip install opencv-python numpy
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


BORDER_MAP = {
    "reflect":   cv2.BORDER_REFLECT_101,  # nice for most photos
    "replicate": cv2.BORDER_REPLICATE,
    "constant":  cv2.BORDER_CONSTANT,
    "wrap":      cv2.BORDER_WRAP,
}


def _make_odd(n: int) -> int:
    return int(n) | 1  # ensure odd (needed so kernel has a true center)


def build_motion_kernel(length: int, angle_deg: float, sigma: float = 0.0) -> np.ndarray:
    """
    Create a 2D motion blur kernel: a normalized line at a given angle.
    length: kernel size (and approximate blur length) in pixels, >=1.
    angle_deg: direction in degrees (0 = horizontal, +CCW).
    sigma: optional Gaussian feathering; 0 = sharp line.
    """
    k = max(1, int(round(length)))
    k = _make_odd(k)
    kernel = np.zeros((k, k), dtype=np.float32)

    # Line endpoints across the kernel at the specified angle
    theta = math.radians(angle_deg)
    cx = cy = (k - 1) / 2.0
    half = (k - 1) / 2.0
    dx = half * math.cos(theta)
    dy = half * math.sin(theta)

    x1, y1 = int(round(cx - dx)), int(round(cy - dy))
    x2, y2 = int(round(cx + dx)), int(round(cy + dy))

    # Draw a 1-pixel line; thickness can be increased if you want "fatter" blur
    cv2.line(kernel, (x1, y1), (x2, y2), color=1.0, thickness=1)

    # Optional softness: lightly blur the kernel itself to feather the line
    if sigma and sigma > 0:
        # Choose an odd Gaussian size based on sigma; clamp to kernel size
        gk = min(k, _make_odd(max(3, int(6 * sigma + 1))))
        kernel = cv2.GaussianBlur(kernel, (gk, gk), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    s = kernel.sum()
    if s <= 1e-8:
        # Safety: fall back to identity if the line somehow vanished
        kernel[(k // 2), (k // 2)] = 1.0
        s = 1.0
    kernel /= s
    return kernel


def apply_motion_blur(img: np.ndarray, length: int, angle_deg: float, sigma: float, border_mode: str) -> np.ndarray:
    """
    Apply the motion blur kernel to an image (supports grayscale, BGR, or BGRA).
    """
    kernel = build_motion_kernel(length, angle_deg, sigma)
    border = BORDER_MAP.get(border_mode.lower(), cv2.BORDER_REFLECT_101)

    if img.ndim == 2:  # grayscale
        return cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=border)

    if img.shape[2] == 4:  # BGRA: preserve alpha by filtering only color channels
        b, g, r, a = cv2.split(img)
        b = cv2.filter2D(b, -1, kernel, borderType=border)
        g = cv2.filter2D(g, -1, kernel, borderType=border)
        r = cv2.filter2D(r, -1, kernel, borderType=border)
        return cv2.merge((b, g, r, a))

    # BGR
    return cv2.filter2D(img, ddepth=-1, kernel=kernel, borderType=border)


def main():
    p = argparse.ArgumentParser(description="Apply directional motion blur to an image.")
    p.add_argument("input", type=str, help="Path to input image")
    p.add_argument("-o", "--output", type=str, default=None, help="Path to save output image")
    p.add_argument("--length", type=float, required=True, help="Blur length (pixels). Example: 25")
    p.add_argument("--angle", type=float, required=True, help="Blur angle (degrees). 0=→, 90=↓")
    p.add_argument("--sigma", type=float, default=0.0, help="Optional softness of blur line (0–2 typical)")
    p.add_argument("--border", type=str, default="reflect",
                   choices=list(BORDER_MAP.keys()), help="Edge handling mode")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_motion" + inp.suffix)

    # Read unchanged so alpha is retained if present
    img = cv2.imread(str(inp), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {inp}")

    # Apply blur
    blurred = apply_motion_blur(img, length=int(round(args.length)), angle_deg=args.angle,
                                sigma=args.sigma, border_mode=args.border)

    # Save
    if not cv2.imwrite(str(out), blurred):
        raise RuntimeError(f"Failed to write output: {out}")

    print(f"Saved: {out}  (length={int(round(args.length))}, angle={args.angle}°, sigma={args.sigma}, border={args.border})")


if __name__ == "__main__":
    main()
