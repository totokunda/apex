#!/usr/bin/env python3
"""
Generate synthetic video from image and run flow preprocessors
"""
import asyncio
import requests
import websockets
import json
import shutil
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
API_BASE_URL = "http://localhost:8765"
WS_BASE_URL = "ws://localhost:8765"
INPUT_IMAGE = "/path/to/image"
SYNTHETIC_VIDEO = "/path/to/synthetic_video.mp4"
OUTPUT_DIR = Path.cwd() / "preprocessor_results"

# Flow preprocessors
FLOW_PREPROCESSORS = ["ptlflow", "unimatch"]

def create_panning_video(image_path: str, output_path: Path, duration_seconds: float = 2.0, fps: int = 30):
    """Create a video with panning motion from a static image"""
    print(f"\nCreating synthetic video with panning motion...")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return False, f"Failed to load image: {image_path}"
    
    h, w = img.shape[:2]
    
    # Create a larger canvas for panning effect
    scale_factor = 1.3
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Resize image to larger canvas
    large_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Video settings
    total_frames = int(duration_seconds * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    if not out.isOpened():
        return False, "Failed to create video writer"
    
    # Generate frames with panning motion
    for i in range(total_frames):
        # Calculate panning position (diagonal pan from top-left to bottom-right)
        progress = i / (total_frames - 1)
        x_offset = int((new_w - w) * progress)
        y_offset = int((new_h - h) * progress)
        
        # Crop frame from large image
        frame = large_img[y_offset:y_offset+h, x_offset:x_offset+w]
        
        out.write(frame)
    
    out.release()
    
    print(f"✓ Created synthetic video: {output_path}")
    print(f"  Duration: {duration_seconds}s, FPS: {fps}, Frames: {total_frames}")
    print(f"  Resolution: {w}x{h}\n")
    
    return True, str(output_path)

async def track_job_progress(job_id: str, pbar: tqdm):
    """Connect to websocket and track job progress"""
    ws_url = f"{WS_BASE_URL}/ws/job/{job_id}"
    
    try:
        async with websockets.connect(ws_url) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    status = data.get("status", "unknown")
                    progress = data.get("progress", 0)
                    message_text = data.get("message", "")
                    
                    pbar.set_postfix_str(f"{message_text[:50]}")
                    
                    if status == "complete":
                        return True, None
                    elif status == "error":
                        error = data.get("error", "Unknown error")
                        return False, error
                        
                except websockets.exceptions.ConnectionClosed:
                    break
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        return False, str(e)
    
    return False, "Connection closed unexpectedly"

def submit_preprocessor_job(preprocessor_name: str, input_path: str):
    """Submit a preprocessor job via the API"""
    url = f"{API_BASE_URL}/preprocessor/run"
    
    payload = {
        "preprocessor_name": preprocessor_name,
        "input_path": input_path,
        "params": {}
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("job_id"), None
    else:
        return None, f"Status {response.status_code}: {response.text}"

def get_job_result(job_id: str):
    """Get the result file path for a completed job"""
    url = f"{API_BASE_URL}/preprocessor/result/{job_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("status") == "complete":
            return result.get("result_path"), None
        else:
            return None, result.get("error", f"Job status: {result.get('status')}")
    else:
        return None, f"Status {response.status_code}"

def extract_frame_from_video(source: str, preprocessor_name: str, output_dir: Path):
    """Extract a representative frame from flow video and save as image"""
    if not source:
        return False, "No source path"
    
    source_path = Path(source)
    
    if not source_path.exists():
        return False, f"Source does not exist: {source_path}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video and extract middle frame
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return False, "Failed to open video"
    
    # Get total frame count and seek to middle
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, "Failed to read frame"
    
    # Save as PNG image
    destination = output_dir / f"{preprocessor_name}.png"
    
    try:
        cv2.imwrite(str(destination), frame)
        return True, str(destination)
    except Exception as e:
        return False, str(e)

async def process_preprocessor(preprocessor_id: str, preprocessor_name: str, video_path: str, pbar: tqdm):
    """Run a single preprocessor and save the result"""
    pbar.set_description(f"Processing {preprocessor_name[:30]}")
    
    # Submit job
    job_id, error = submit_preprocessor_job(preprocessor_id, video_path)
    if error:
        pbar.write(f"❌ {preprocessor_name}: Failed to submit - {error}")
        return False
    
    # Track progress
    success, error = await track_job_progress(job_id, pbar)
    if not success:
        pbar.write(f"❌ {preprocessor_name}: Job failed - {error}")
        return False
    
    # Small delay for result to be ready
    await asyncio.sleep(1)
    
    # Get result
    result_path, error = get_job_result(job_id)
    if error:
        pbar.write(f"❌ {preprocessor_name}: Failed to get result - {error}")
        return False
    
    # Extract frame from flow video
    success, output = extract_frame_from_video(result_path, preprocessor_id, OUTPUT_DIR)
    if success:
        pbar.write(f"✓ {preprocessor_name}: Saved to {Path(output).name}")
        return True
    else:
        pbar.write(f"❌ {preprocessor_name}: Failed to extract frame - {output}")
        return False

def get_preprocessors():
    """Fetch list of flow preprocessors from API"""
    url = f"{API_BASE_URL}/preprocessor/list"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        preprocessors = data.get("preprocessors", [])
        # Filter for flow preprocessors only
        filtered = [
            p for p in preprocessors 
            if p["id"] in FLOW_PREPROCESSORS
        ]
        return filtered, None
    else:
        return None, f"Failed to fetch preprocessors: {response.status_code}"

async def main():
    """Main function"""
    # Check input file
    if not Path(INPUT_IMAGE).exists():
        print(f"❌ Input image not found: {INPUT_IMAGE}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Running Flow Preprocessors on Synthetic Video")
    print(f"{'='*60}")
    print(f"Input Image: {INPUT_IMAGE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    # Step 1: Create synthetic video with motion
    success, result = create_panning_video(INPUT_IMAGE, SYNTHETIC_VIDEO)
    if not success:
        print(f"❌ Failed to create synthetic video: {result}")
        sys.exit(1)
    
    video_path = result
    
    # Step 2: Get list of flow preprocessors
    print("Fetching flow preprocessor list...")
    preprocessors, error = get_preprocessors()
    if error:
        print(f"❌ {error}")
        sys.exit(1)
    
    if not preprocessors:
        print("❌ No flow preprocessors found")
        sys.exit(1)
    
    print(f"Found {len(preprocessors)} flow preprocessors\n")
    
    # Step 3: Process each flow preprocessor
    success_count = 0
    failed_count = 0
    
    with tqdm(total=len(preprocessors), desc="Overall Progress", unit="preprocessor") as pbar:
        for preprocessor in preprocessors:
            preprocessor_id = preprocessor["id"]
            preprocessor_name = preprocessor["name"]
            
            success = await process_preprocessor(preprocessor_id, preprocessor_name, video_path, pbar)
            if success:
                success_count += 1
            else:
                failed_count += 1
            
            pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count} successful, {failed_count} failed")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Synthetic video: {SYNTHETIC_VIDEO}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(main())

