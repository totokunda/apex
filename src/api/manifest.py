import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/manifest", tags=["manifest"])

# Base path to manifest directory
MANIFEST_BASE_PATH = Path(__file__).parent.parent.parent / "manifest/"

class ManifestInfo(BaseModel):
    name: str
    model: str
    model_type: str
    full_path: str

def get_all_manifest_files() -> List[ManifestInfo]:
    """Scan manifest directory and return all YAML files with their metadata."""
    manifests = []
    print(MANIFEST_BASE_PATH)
    
    for root, dirs, files in os.walk(MANIFEST_BASE_PATH):
        for file in files:
            if file.endswith('.yml') and not file.startswith('shared'):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(MANIFEST_BASE_PATH)
                
                # Extract model and model_type from path structure
                path_parts = relative_path.parts
                if len(path_parts) >= 2:
                    model = path_parts[0]
                    model_type = path_parts[1] if len(path_parts) > 2 else path_parts[1]
                    manifest_name = file.replace('.yml', '')
                    
                    manifests.append(ManifestInfo(
                        name=manifest_name,
                        model=model,
                        model_type=model_type,
                        full_path=str(relative_path)
                    ))
    
    return manifests

def load_yaml_content(file_path: Path) -> Dict[Any, Any]:
    """Load and return YAML file content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Manifest file not found")
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing YAML file: {str(e)}")

@router.get("/list", response_model=List[str])
def list_all_manifests():
    """List all available manifest names."""
    manifests = get_all_manifest_files()
    return [manifest.name for manifest in manifests]

@router.get("/list/model/{model}", response_model=List[str])
def list_manifests_by_model(model: str):
    """List all manifest names for a specific model."""
    manifests = get_all_manifest_files()
    filtered = [m.name for m in manifests if m.model == model]
    if not filtered:
        raise HTTPException(status_code=404, detail=f"No manifests found for model: {model}")
    return filtered

@router.get("/list/type/{model_type}", response_model=List[str])
def list_manifests_by_model_type(model_type: str):
    """List all manifest names for a specific model type."""
    manifests = get_all_manifest_files()
    
    filtered = [m.name for m in manifests if m.model_type == model_type]
    if not filtered:
        raise HTTPException(status_code=404, detail=f"No manifests found for model_type: {model_type}")
    return filtered

@router.get("/list/model/{model}/model_type/{model_type}", response_model=List[str])
def list_manifests_by_model_and_type(model: str, model_type: str):
    """List all manifest names for a specific model and model type combination."""
    manifests = get_all_manifest_files()
    filtered = [m.name for m in manifests if m.model == model and m.model_type == model_type]
    if not filtered:
        raise HTTPException(
            status_code=404, 
            detail=f"No manifests found for model: {model} and model_type: {model_type}"
        )
    return filtered

@router.get("/get/{manifest_name}")
def get_manifest_content(manifest_name: str):
    """Get the actual YAML content of a specific manifest by name."""
    manifests = get_all_manifest_files()
    
    # Find manifest by name
    manifest = next((m for m in manifests if m.name == manifest_name), None)
    if not manifest:
        raise HTTPException(status_code=404, detail=f"Manifest not found: {manifest_name}")
    
    # Load and return YAML content
    file_path = MANIFEST_BASE_PATH / manifest.full_path
    return load_yaml_content(file_path)

@router.get("/info", response_model=List[ManifestInfo])
def get_manifest_info(
    model: Optional[str] = Query(None, description="Filter by model name"),
    model_type: Optional[str] = Query(None, description="Filter by model type")
):
    """Get detailed information about manifests with optional filtering."""
    manifests = get_all_manifest_files()
    
    if model:
        manifests = [m for m in manifests if m.model == model]
    
    if model_type:
        manifests = [m for m in manifests if m.model_type == model_type]
    
    if not manifests:
        filter_desc = []
        if model:
            filter_desc.append(f"model: {model}")
        if model_type:
            filter_desc.append(f"model_type: {model_type}")
        
        detail = f"No manifests found" + (f" for {', '.join(filter_desc)}" if filter_desc else "")
        raise HTTPException(status_code=404, detail=detail)
    
    return manifests
