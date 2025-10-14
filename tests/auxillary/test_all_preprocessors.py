#!/usr/bin/env python3
"""
Run all preprocessors on a single image and save results with preprocessor names
"""
import asyncio
import requests
import websockets
import json
import shutil
from pathlib import Path
import sys
from tqdm import tqdm

# Configuration
API_BASE_URL = "http://localhost:8765"
WS_BASE_URL = "ws://localhost:8765"
INPUT_IMAGE = "/path/to/image"
OUTPUT_DIR = Path.cwd() / "preprocessor_results"

# Preprocessors to exclude (flow-related)
EXCLUDE_PREPROCESSORS = ["ptlflow", "unimatch"]

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

def copy_result(source: str, preprocessor_name: str, output_dir: Path):
    """Copy the result file to output directory with preprocessor name"""
    if not source:
        return False, "No source path"
    
    source_path = Path(source)
    
    if not source_path.exists():
        return False, f"Source does not exist: {source_path}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extension = source_path.suffix
    destination = output_dir / f"{preprocessor_name}{extension}"
    
    try:
        shutil.copy2(source_path, destination)
        return True, str(destination)
    except Exception as e:
        return False, str(e)

def get_preprocessors():
    """Fetch list of all preprocessors from API"""
    url = f"{API_BASE_URL}/preprocessor/list"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        preprocessors = data.get("preprocessors", [])
        # Filter out flow preprocessors and those that don't support images
        filtered = [
            p for p in preprocessors 
            if p["id"] not in EXCLUDE_PREPROCESSORS 
            and p.get("supports_image", True)
        ]
        return filtered, None
    else:
        return None, f"Failed to fetch preprocessors: {response.status_code}"

async def process_preprocessor(preprocessor_id: str, preprocessor_name: str, pbar: tqdm):
    """Run a single preprocessor and save the result"""
    pbar.set_description(f"Processing {preprocessor_name[:30]}")
    
    # Submit job
    job_id, error = submit_preprocessor_job(preprocessor_id, INPUT_IMAGE)
    print(f"Job ID: {job_id}")
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
    
    # Copy result
    success, output = copy_result(result_path, preprocessor_id, OUTPUT_DIR)
    if success:
        pbar.write(f"✓ {preprocessor_name}: Saved to {Path(output).name}")
        return True
    else:
        pbar.write(f"❌ {preprocessor_name}: Failed to copy - {output}")
        return False

async def main():
    """Main function"""
    # Check input file
    if not Path(INPUT_IMAGE).exists():
        print(f"❌ Input image not found: {INPUT_IMAGE}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Running All Preprocessors on Image")
    print(f"{'='*60}")
    print(f"Input: {INPUT_IMAGE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # Get list of preprocessors
    print("Fetching preprocessor list...")
    preprocessors, error = get_preprocessors()
    if error:
        print(f"❌ {error}")
        sys.exit(1)
    
    print(f"Found {len(preprocessors)} preprocessors (excluding flow preprocessors)\n")
    
    # Process each preprocessor
    success_count = 0
    failed_count = 0
    
    with tqdm(total=len(preprocessors), desc="Overall Progress", unit="preprocessor") as pbar:
        for preprocessor in preprocessors:
            preprocessor_id = preprocessor["id"]
            preprocessor_name = preprocessor["name"]
            
            if preprocessor_id != "animalpose":
                continue
            
            success = await process_preprocessor(preprocessor_id, preprocessor_name, pbar)
            if success:
                success_count += 1
            else:
                failed_count += 1
            
            pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count} successful, {failed_count} failed")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(main())
