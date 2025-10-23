import os
from functools import lru_cache
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/manifest", tags=["manifest"])

# Base path to manifest directory
MANIFEST_BASE_PATH = Path(__file__).parent.parent.parent / "manifest_updated/"

class ManifestInfo(BaseModel):
    id: str
    name: str
    model: str
    model_type: List[str] | str
    full_path: str
    version: str
    description: str
    tags: List[str]
    author: str
    license: str
    demo_path: str
    downloaded: bool = False

class ModelTypeInfo(BaseModel):
    key: str
    label: str
    description: str
    
    


MODEL_TYPE_MAPPING = {
    "vace": "control"
}


@lru_cache(maxsize=1000)
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
                file_data = load_yaml_content(file_path)
                model_type = file_data.get("spec", {}).get("model_type", [])
                if isinstance(model_type, list):
                    model_type = [MODEL_TYPE_MAPPING.get(t, t) for t in model_type]
                else:
                    model_type = MODEL_TYPE_MAPPING.get(model_type, model_type)
                model = file_data.get("metadata", {}).get("model", "")
                manifest_name = file_data.get("metadata", {}).get("name", file.replace('.yml', ''))
                manifests.append(ManifestInfo(
                        id=file_data.get("metadata", {}).get("id", ""),
                        name=manifest_name,
                        model=model,
                        model_type=model_type,
                        full_path=str(relative_path),
                        version=str(file_data.get("metadata", {}).get("version", "")),
                        description=file_data.get("metadata", {}).get("description", ""),
                        tags=[str(t) for t in file_data.get("metadata", {}).get("tags", [])],
                        author=file_data.get("metadata", {}).get("author", ""),
                        license=file_data.get("metadata", {}).get("license", ""),
                        demo_path=file_data.get("metadata", {}).get("demo_path", "")
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

@router.get("/types", response_model=List[ModelTypeInfo])
def list_model_types() -> List[ModelTypeInfo]:
    """List distinct spec.model_type values across manifests with label and description.

    Scans all YAML files under ``manifest_updated`` and aggregates the unique
    values found in ``spec.model_type`` (supports both string and list values).
    """
    if not MANIFEST_BASE_PATH.exists():
        return []

    # Friendly labels and short descriptions per known type
    label_map = {
        "t2v": "Text to Video",
        "i2v": "Image to Video",
        "v2v": "Video to Video",
        "x2v": "Any to Video ",
        "edit": "Edit",
        "control": "Control",
    }
    description_map = {
        "t2v": "Generate videos from text prompts.",
        "i2v": "Animate a single image into a video.",
        "v2v": "Transform an input video with a new style or prompt.",
        "x2v": "Flexible any-to-video generation.",
        "edit": "Edit or modify images using prompts and tools.",
        "control": "Guide generation with control signals (e.g., canny, pose)."
    }

    discovered_types = set()

    for root, _, files in os.walk(MANIFEST_BASE_PATH):
        for file in files:
            if not file.endswith(".yml") or file.startswith("shared"):
                continue
            file_path = Path(root) / file
            try:
                data = load_yaml_content(file_path)
            except HTTPException:
                # Skip invalid YAMLs
                continue

            spec = data.get("spec", {}) if isinstance(data, dict) else {}
            model_type_field = spec.get("model_type")
            if isinstance(model_type_field, list):
                for t in model_type_field:
                    if isinstance(t, str) and t.strip():
                        model_type = MODEL_TYPE_MAPPING.get(t.strip(), t.strip())
                        discovered_types.add(model_type)
            elif isinstance(model_type_field, str) and model_type_field.strip():
                model_type = MODEL_TYPE_MAPPING.get(model_type_field.strip(), model_type_field.strip())
                discovered_types.add(model_type)

    results: List[ModelTypeInfo] = []
    for key in sorted(discovered_types):
        label = label_map.get(key, key.replace('_', ' ').replace('-', ' ').upper())
        description = description_map.get(key, f"Models with '{key}' capability.")
        results.append(ModelTypeInfo(key=key, label=label, description=description))

    return results

@router.get("/list", response_model=List[ManifestInfo])
def list_all_manifests():
    """List all available manifest names."""
    manifests = get_all_manifest_files()
    return manifests

@router.get("/list/model/{model}", response_model=List[ManifestInfo])
def list_manifests_by_model(model: str):
    """List all manifest names for a specific model."""
    manifests = get_all_manifest_files()
    filtered = [m for m in manifests if m.model == model]
    if not filtered:
        raise HTTPException(status_code=404, detail=f"No manifests found for model: {model}")
    return filtered

@router.get("/list/type/{model_type}", response_model=List[ManifestInfo])
def list_manifests_by_model_type(model_type: str):
    """List all manifest names for a specific model type."""
    manifests = get_all_manifest_files()
    
    filtered = [m for m in manifests if m.model_type == model_type]
    if not filtered:
        raise HTTPException(status_code=404, detail=f"No manifests found for model_type: {model_type}")
    return filtered

@router.get("/list/model/{model}/model_type/{model_type}", response_model=List[ManifestInfo])
def list_manifests_by_model_and_type(model: str, model_type: str):
    """List all manifest names for a specific model and model type combination."""
    manifests = get_all_manifest_files()
    filtered = [m for m in manifests if m.model == model and m.model_type == model_type]
    if not filtered:
        raise HTTPException(
            status_code=404, 
            detail=f"No manifests found for model: {model} and model_type: {model_type}"
        )
    return filtered

@router.get("/{manifest_id}")
def get_manifest_content(manifest_id: str):
    """Get the actual YAML content of a specific manifest by name."""
    manifests = get_all_manifest_files()
    
    # Find manifest by name
    manifest = next((m for m in manifests if m.id == manifest_id), None)
    if not manifest:
        raise HTTPException(status_code=404, detail=f"Manifest not found: {manifest_id}")
    
    # Load and return YAML content
    file_path = MANIFEST_BASE_PATH / manifest.full_path
    return load_yaml_content(file_path)
