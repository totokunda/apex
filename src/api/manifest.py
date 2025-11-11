import os
from functools import lru_cache
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import get_components_path, get_config_path

router = APIRouter(prefix="/manifest", tags=["manifest"])

# Base path to manifest directory
MANIFEST_BASE_PATH = Path(__file__).parent.parent.parent / "manifest" / "engine"

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


def get_all_manifest_files() -> List[ManifestInfo]:
    """Scan manifest directory and return all YAML files with their metadata."""
    manifests = []
    
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
                # Determine if manifest is considered downloaded: at least one model_path per component is present,
                # and if a config_path exists it must also be present.
                components = file_data.get("spec", {}).get("components", [])
                manifest_downloaded = bool(components)
                for component in components:
                    comp_type = component.get("type")
                    component_ok = True

                    if comp_type == "scheduler":
                        # For schedulers, iterate all options: OK if any option with a config_path is downloaded.
                        options = component.get("scheduler_options", []) or []
                        if isinstance(options, list) and options:
                            any_downloaded = False
                            has_downloadable_config = False
                            for opt in options:
                                if not isinstance(opt, dict):
                                    continue
                                config_path = opt.get("config_path")
                                if config_path:
                                    has_downloadable_config = True
                                    if DownloadMixin.is_downloaded(config_path, get_config_path()) is not None:
                                        any_downloaded = True
                                        break
                                    if DownloadMixin.is_downloaded(config_path, get_components_path()) is not None:
                                        any_downloaded = True
                                        break
                            if has_downloadable_config:
                                component_ok = any_downloaded
                            else:
                                # Built-in or inline-config schedulers don't require downloads
                                component_ok = True
                        else:
                            # No scheduler options; treat as OK
                            component_ok = True
                        
                    else:
                        model_paths = component.get("model_path", [])
                        if isinstance(model_paths, (str, dict)):
                            model_paths = [model_paths]
                        model_ok = len(model_paths) == 0
                        for mp in model_paths:
                            if isinstance(mp, str):
                                if DownloadMixin.is_downloaded(mp, get_components_path()) is not None:
                                    model_ok = True
                                    break
                            elif isinstance(mp, dict):
                                path_val = mp.get("path")
                                if path_val and DownloadMixin.is_downloaded(path_val, get_components_path()) is not None:
                                    model_ok = True
                                    break
                                
                        component_ok =  model_ok
                        
                    if not component_ok:
                        manifest_downloaded = False
                        break
                
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
                        demo_path=file_data.get("metadata", {}).get("demo_path", ""),
                        downloaded=manifest_downloaded
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


# ----------------------------- Attention Helpers ----------------------------- #
def _attention_label_description_maps() -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Centralised mapping for attention backend labels and short descriptions.
    """
    label_map = {
        "sdpa": "PyTorch SDPA",
        "sdpa_varlen": "PyTorch SDPA (VarLen)",
        "sdpa_streaming": "SDPA Streaming",
        "flash": "FlashAttention-2",
        "flash3": "FlashAttention-3",
        "sage": "SageAttention",
        "xformers": "xFormers",
        "flex": "Flex Attention",
        "xla_flash": "XLA Flash Attention",
    }

    description_map = {
        "sdpa": "Built-in torch scaled_dot_product_attention backend.",
        "sdpa_varlen": "VarLen wrapper using SDPA compatible with flash-attn varlen APIs.",
        "sdpa_streaming": "Streaming softmax SDPA variant for long sequences.",
        "flash": "NVIDIA FlashAttention-2 kernel (fast, memory-efficient).",
        "flash3": "FlashAttention-3 kernel via flash_attn_interface.",
        "sage": "SageAttention kernel backend.",
        "xformers": "xFormers memory-efficient attention implementation.",
        "flex": "PyTorch Flex Attention (experimental flexible masks).",
        "xla_flash": "XLA/TPU Flash Attention kernel.",
    }

    return label_map, description_map


def _build_attention_options(allowed: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Build a list of attention backend options from the attention registry, each
    containing name, label and description. If `allowed` is provided, the list is
    filtered to those names; otherwise all installed/available backends are used.
    """
    # Local import to avoid import cycles at startup
    try:
        from src.attention.functions import attention_register
    except Exception as e:
        # If attention stack cannot be imported, return an empty list gracefully
        return []

    label_map, description_map = _attention_label_description_maps()

    # "Installed" or runtime-available attention backends
    available_keys = set(attention_register.all_available().keys())

    if allowed is not None:
        allowed_set = {a for a in allowed if isinstance(a, str)}
        final_keys = sorted(available_keys.intersection(allowed_set))
    else:
        # If the manifest does not specify support/attention types, expose all
        # available implementations.
        final_keys = sorted(available_keys)

    results: List[Dict[str, str]] = []
    for key in final_keys:
        label = label_map.get(key, key.replace('_', ' ').title())
        desc = description_map.get(key, f"{key} attention backend.")
        results.append({
            "name": key,
            "label": label,
            "description": desc,
        })

    return results

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
        "t2i": "Text to Image",
    }
    description_map = {
        "t2v": "Generate videos from text prompts.",
        "i2v": "Animate a single image into a video.",
        "t2i": "Generate images from text prompts.",
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
    content = load_yaml_content(file_path)

    # ----------------- Attention backends enrichment (name/label/desc) ----------------- #
    spec = content.get("spec", {}) if isinstance(content, dict) else {}
    # Support both historical and current field names
    configured_attention = spec.get("support_attention")
    if configured_attention is None:
        configured_attention = spec.get("attention_types")
    if isinstance(configured_attention, list):
        attention_allowed: Optional[List[str]] = [x for x in configured_attention if isinstance(x, str)]
    else:
        attention_allowed = None  # fall back to all available backends

    attention_options = _build_attention_options(attention_allowed)
    # Keep the original field as-is; add a parallel enriched field for the UI/API
    if "spec" not in content:
        content["spec"] = {}
    content["spec"]["attention_types_detail"] = attention_options
    for component_index, component in enumerate(content.get("spec", {}).get("components", [])):
        # check config path too
        is_component_downloaded = True
        if config_path := component.get("config_path"):
            is_downloaded = DownloadMixin.is_downloaded(config_path, get_components_path())
            if is_downloaded is None:
                is_component_downloaded = False
                
        
        if component.get("type") == "scheduler":
            options = component.get("scheduler_options", []) or []
            is_scheduler_downloaded = False
            has_downloadable_config = False
            for option in options:
                if option.get("config_path"):
                    has_downloadable_config = True
                    is_downloaded = DownloadMixin.is_downloaded(option.get("config_path"), get_config_path())
                    if is_downloaded is not None:
                        is_scheduler_downloaded = True
                        break
                    is_downloaded = DownloadMixin.is_downloaded(option.get("config_path"), get_components_path())
                    if is_downloaded is not None:
                        is_scheduler_downloaded = True
                        break

            if not is_scheduler_downloaded and has_downloadable_config:
                is_component_downloaded = False
            elif not has_downloadable_config:
                is_component_downloaded = True

        any_path_downloaded = False
        for index, model_path in enumerate(component.get("model_path", [])):
            # we check if model path is downloaded
            if isinstance(model_path, str):
                is_downloaded = DownloadMixin.is_downloaded(model_path, get_components_path())
                model_path = {
                    "path": model_path,
                    "is_downloaded": is_downloaded is not None,
                    "type": "safetensors"
                }
            else:
                is_downloaded = DownloadMixin.is_downloaded(model_path.get("path"), get_components_path())
                

                if is_downloaded is not None:
                    model_path["is_downloaded"] = True
                    model_path["path"] = is_downloaded
                    any_path_downloaded = True
                else:
                    model_path["is_downloaded"] = False
                    
        
            component["model_path"][index] = model_path
        
        if not any_path_downloaded and len(component.get("model_path", [])) > 0:
            is_component_downloaded = False
        component['is_downloaded'] = is_component_downloaded
        content["spec"]["components"][component_index] = component
        
        
                
    return content
