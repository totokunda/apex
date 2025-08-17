#!/usr/bin/env python3
"""
Video merger with smooth blending transitions using OpenCV.
Merges two video files with crossfade effects for fluid transitions.
"""

import cv2
import numpy as np
import argparse
import os
import imageio
from typing import Tuple, Optional


def get_video_info(video_path: str) -> Tuple[int, int, int, float, int]:
    """Extract video properties."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    cap.release()
    return width, height, fps, duration, total_frames


def create_crossfade_transition(frame1: np.ndarray, frame2: np.ndarray, 
                               alpha: float) -> np.ndarray:
    """Create smooth crossfade between two frames."""
    # Ensure frames have same dimensions
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    
    # Blend frames using weighted addition
    blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
    return blended.astype(np.uint8)


def merge_videos_with_transitions(video1_path: str, video2_path: str, 
                                output_path: str, transition_frames: int = 30):
    """Merge two videos with smooth crossfade transitions."""
    
    # Get video properties
    width1, height1, fps1, duration1, frames1 = get_video_info(video1_path)
    width2, height2, fps2, duration2, frames2 = get_video_info(video2_path)
    
    # Use the first video's properties for output
    output_width, output_height, output_fps = width1, height1, fps1
    
    # Open video captures
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Cannot open one or both video files")
    
    # Collect all frames for high-quality output
    all_frames = []
    
    # Read first video (excluding transition frames)
    frames_to_write = frames1 - transition_frames
    print("Reading frames from first video...")
    for i in range(frames_to_write):
        ret, frame = cap1.read()
        if not ret:
            break
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)
    
    # Create transition section
    print(f"Creating {transition_frames} frame transition...")
    for i in range(transition_frames):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Calculate alpha for smooth transition
        alpha = i / transition_frames
        
        # Create crossfade
        blended_frame = create_crossfade_transition(frame1, frame2, alpha)
        # Convert BGR to RGB for imageio
        blended_frame_rgb = cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGB)
        all_frames.append(blended_frame_rgb)
    
    # Read remaining frames from second video
    print("Reading frames from second video...")
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        
        # Resize if dimensions differ
        if frame.shape[:2] != (output_height, output_width):
            frame = cv2.resize(frame, (output_width, output_height))
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)
    
    # Cleanup video captures
    cap1.release()
    cap2.release()
    
    # Write high-quality output using imageio
    print(f"Writing {len(all_frames)} frames to output file...")
    imageio.mimsave(output_path, all_frames, fps=output_fps, quality=9, 
                    codec='libx264', pixelformat='yuv420p')
    
    print(f"Video merge complete! Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge two videos with smooth transitions")
    parser.add_argument("video1", help="Path to first video file")
    parser.add_argument("video2", help="Path to second video file")
    parser.add_argument("output", help="Path for output video file")
    parser.add_argument("--transition-frames", "-t", type=int, default=30,
                       help="Number of frames for transition (default: 30)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.video1):
        print(f"Error: First video file not found: {args.video1}")
        return
    
    if not os.path.exists(args.video2):
        print(f"Error: Second video file not found: {args.video2}")
        return
    
    try:
        merge_videos_with_transitions(args.video1, args.video2, args.output, args.transition_frames)
    except Exception as e:
        print(f"Error during video merge: {e}")


if __name__ == "__main__":
    main()
