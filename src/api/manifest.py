import os
from functools import lru_cache
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import get_components_path, get_config_path, get_lora_path
from src.utils.compute import get_compute_capability, validate_compute_requirements, ComputeCapability

router = APIRouter(prefix="/manifest", tags=["manifest"])

# Base path to manifest directory
MANIFEST_BASE_PATH = Path(__file__).parent.parent.parent / "manifest" / "engine"

# Cache the system's compute capability (it doesn't change during runtime)
_SYSTEM_COMPUTE_CAPABILITY: Optional[ComputeCapability] = None

def _get_system_compute_capability() -> ComputeCapability:
    """Get the system's compute capability (cached)."""
    global _SYSTEM_COMPUTE_CAPABILITY
    if _SYSTEM_COMPUTE_CAPABILITY is None:
        _SYSTEM_COMPUTE_CAPABILITY = get_compute_capability()
    return _SYSTEM_COMPUTE_CAPABILITY


class ModelTypeInfo(BaseModel):
    key: str
    label: str
    description: str


MODEL_TYPE_MAPPING = {
    "vace": "control",
    "fill": "inpaint",
    "kontext": "edit",
    "edit_plus": "edit",
    "dreamomni2": "edit",
    "humo": "a2v",
    "s2v": "a2v",
    "animate": "control"
}

def _load_and_enrich_manifest(relative_path: str) -> Dict[Any, Any]:
    """Load a manifest by relative path and enrich it with runtime info."""
    file_path = MANIFEST_BASE_PATH / relative_path
    content = load_yaml_content(file_path)

    # ----------------- Attention backends enrichment (name/label/desc) ----------------- #
    spec = content.get("spec", {}) if isinstance(content, dict) else {}
    metadata = content.get("metadata", {}) if isinstance(content, dict) else {}
    configured_attention = spec.get("support_attention")
    if configured_attention is None:
        configured_attention = spec.get("attention_types")
    if isinstance(configured_attention, list):
        attention_allowed: Optional[List[str]] = [x for x in configured_attention if isinstance(x, str)]
    else:
        attention_allowed = None  # fall back to all available backends

    attention_options = _build_attention_options(attention_allowed)
    if "spec" not in content:
        content["spec"] = {}
    content["spec"]["attention_types_detail"] = attention_options

    # Enrich LoRA entries
    for lora_index, lora in enumerate(content.get("spec", {}).get("loras", [])):
        if isinstance(lora, str):
            is_downloaded = DownloadMixin.is_downloaded(lora, get_lora_path())
            lora_basename = os.path.basename(lora)
            lora_name = lora_basename.split(".")[0]
            out_lora = {
                "label": lora_name,
                "name": lora_name,
                "scale": 1.0,
            }
            if is_downloaded is not None:
                out_lora["is_downloaded"] = True
                out_lora["source"] = is_downloaded
            else:
                out_lora["is_downloaded"] = False
                out_lora["source"] = lora
            content["spec"]["loras"][lora_index] = out_lora
        elif isinstance(lora, dict):
            is_downloaded = DownloadMixin.is_downloaded(lora.get("source"), get_lora_path())
            if is_downloaded is not None:
                lora["is_downloaded"] = True
                lora["source"] = is_downloaded
            else:
                lora["is_downloaded"] = False
                lora["source"] = lora.get("source")
            content["spec"]["loras"][lora_index] = lora

    # Enrich components entries
    for component_index, component in enumerate(content.get("spec", {}).get("components", [])):
        is_component_downloaded = True
        if config_path := component.get("config_path"):
            is_downloaded = DownloadMixin.is_downloaded(config_path, get_components_path())
            if is_downloaded is None:
                is_component_downloaded = False
            else:
                component["config_path"] = is_downloaded

        if component.get("type") == "scheduler":
            options = component.get("scheduler_options", []) or []
            is_scheduler_downloaded = False
            has_downloadable_config = False
            for idx, option in enumerate(options):
                if option.get("config_path"):
                    has_downloadable_config = True
                    is_downloaded = DownloadMixin.is_downloaded(option.get("config_path"), get_config_path())
                    if is_downloaded is not None:
                        is_scheduler_downloaded = True
                        options[idx]['config_path'] = is_downloaded                   
                    is_downloaded = DownloadMixin.is_downloaded(option.get("config_path"), get_components_path())
                    if is_downloaded is not None:
                        is_scheduler_downloaded = True
                        options[idx]['config_path'] = is_downloaded

            if not is_scheduler_downloaded and has_downloadable_config:
                is_component_downloaded = False
            
        
            component['scheduler_options'] = options

        any_path_downloaded = False
        for index, model_path in enumerate(component.get("model_path", [])):
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

    # Convenience fields for filtering and compatibility with previous ManifestInfo
    # Normalize and expose common metadata at the top level
    # ID, name, model
    content["id"] = metadata.get("id", "")
    content["name"] = metadata.get("name", "")
    content["model"] = metadata.get("model", "")
    # Model type mapping
    model_type_field = spec.get("model_type", [])
    if isinstance(model_type_field, list):
        mapped_model_type = [MODEL_TYPE_MAPPING.get(t, t) for t in model_type_field]
    else:
        mapped_model_type = MODEL_TYPE_MAPPING.get(model_type_field, model_type_field)
    content["model_type"] = mapped_model_type
    # Other top-level convenience fields
    content["version"] = str(metadata.get("version", ""))
    content["description"] = metadata.get("description", "")
    content["tags"] = [str(t) for t in metadata.get("tags", [])]
    content["author"] = metadata.get("author", "")
    content["license"] = metadata.get("license", "")
    content["demo_path"] = metadata.get("demo_path", "")
    # Keep relative path for downstream use
    content["full_path"] = relative_path
    # Manifest-level downloaded flag: true if there are components and all are downloaded
    components_list = content.get("spec", {}).get("components", []) or []
    content["downloaded"] = bool(components_list) and all(
        isinstance(c, dict) and c.get("is_downloaded", False) for c in components_list
    )
    
    # Compute compatibility check
    compute_requirements = spec.get("compute_requirements")
    if compute_requirements:
        system_capability = _get_system_compute_capability()
        is_compatible, compatibility_error = validate_compute_requirements(
            compute_requirements, 
            system_capability
        )
        content["compute_compatible"] = is_compatible
        content["compute_compatibility_error"] = compatibility_error
        content["compute_requirements_present"] = True
    else:
        # No compute requirements means it's compatible with all systems
        content["compute_compatible"] = True
        content["compute_compatibility_error"] = None
        content["compute_requirements_present"] = False

    return content

def get_manifest(manifest_id: str):
    """Get the actual YAML content of a specific manifest by name."""
    # Resolve manifest path via cached id->path index to avoid full list load
    id_index = _get_manifest_id_index()
    relative_path = id_index.get(manifest_id)
    if not relative_path:
        raise HTTPException(status_code=404, detail=f"Manifest not found: {manifest_id}")
    # Load and enrich only the requested manifest
    return _load_and_enrich_manifest(relative_path)



def _get_all_manifest_files_uncached() -> List[Dict[str, Any]]:
    """Scan manifest directory and return all enriched manifest contents (no cache).
    
    Filters out manifests that are not compatible with the current system's compute capabilities.
    """
    manifests: List[Dict[str, Any]] = []
    
    for root, dirs, files in os.walk(MANIFEST_BASE_PATH):
        for file in files:
            if file.endswith('.yml') and not file.startswith('shared'):
                file_path = Path(root) / file
                relative_path = file_path.relative_to(MANIFEST_BASE_PATH)

                # Load and enrich each manifest directly
                enriched = _load_and_enrich_manifest(str(relative_path))
                
                # Only include manifests that are compatible with the current system
                if enriched.get("compute_compatible", True):
                    manifests.append(enriched)
                
    return manifests


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _get_all_manifest_files_cached(cache_key: str) -> List[Dict[str, Any]]:
    # cache_key is only used to differentiate cache entries; actual logic ignores it
    return _get_all_manifest_files_uncached()


def get_all_manifest_files() -> List[Dict[str, Any]]:
    """Return all enriched manifests, optionally cached based on environment variables.
    
    Controls:
    - APEX_MANIFEST_CACHE or APEX_MANIFEST_CACHE_ENABLED: enable caching when truthy (default disabled)
    - APEX_MANIFEST_CACHE_BUSTER: changing this string invalidates the cache (e.g., set to a timestamp)
    """
    enabled = _env_truthy(os.getenv("APEX_MANIFEST_CACHE", os.getenv("APEX_MANIFEST_CACHE_ENABLED", "0")))
    if not enabled:
        return _get_all_manifest_files_uncached()
    buster = os.getenv("APEX_MANIFEST_CACHE_BUSTER", "")
    cache_key = f"v1:{buster}"
    return _get_all_manifest_files_cached(cache_key)

def _build_manifest_id_index_uncached() -> Dict[str, str]:
    """
    Build a mapping of manifest_id -> relative_path (str) without enriching all manifests.
    """
    index: Dict[str, str] = {}
    for root, dirs, files in os.walk(MANIFEST_BASE_PATH):
        for file in files:
            if not file.endswith(".yml") or file.startswith("shared"):
                continue
            file_path = Path(root) / file
            relative_path = str(file_path.relative_to(MANIFEST_BASE_PATH))
            try:
                data = load_yaml_content(file_path)
                manifest_id = ""
                if isinstance(data, dict):
                    meta = data.get("metadata", {})
                    if isinstance(meta, dict):
                        manifest_id = str(meta.get("id", "")).strip()
                if manifest_id:
                    index.setdefault(manifest_id, relative_path)
            except HTTPException:
                continue
    return index

@lru_cache(maxsize=1)
def _get_manifest_id_index_cached(cache_key: str) -> Dict[str, str]:
    return _build_manifest_id_index_uncached()

def _get_manifest_id_index() -> Dict[str, str]:
    """
    Get the manifest id index, optionally cached controlled by the same env flags
    used for manifest list caching.
    """
    enabled = _env_truthy(os.getenv("APEX_MANIFEST_CACHE", os.getenv("APEX_MANIFEST_CACHE_ENABLED", "0")))
    if not enabled:
        return _build_manifest_id_index_uncached()
    buster = os.getenv("APEX_MANIFEST_CACHE_BUSTER", "")
    cache_key = f"v1:{buster}"
    return _get_manifest_id_index_cached(cache_key)

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
        "inpaint": "Inpaint",
        "a2v": "Audio to Video",
    }
    description_map = {
        "t2v": "Generate videos from text prompts.",
        "i2v": "Animate a single image into a video.",
        "t2i": "Generate images from text prompts.",
        "v2v": "Transform an input video with a new style or prompt.",
        "x2v": "Flexible any-to-video generation.",
        "edit": "Edit or modify images using prompts and tools.",
        "control": "Guide generation with control signals (e.g., canny, pose).",
        "inpaint": "Inpaint images using prompts and masks.",
        "a2v": "Generate videos from audio and images.",
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



@router.get("/system/compute")
def get_system_compute_info():
    """Get information about the current system's compute capabilities."""
    capability = _get_system_compute_capability()
    return capability.to_dict()


@router.get("/list")
def list_all_manifests(include_incompatible: bool = False):
    """List all available manifests.
    
    Args:
        include_incompatible: If True, include manifests that are not compatible 
                            with the current system's compute capabilities.
                            Default is False (only show compatible manifests).
    """
    if include_incompatible:
        # Load all manifests without filtering
        manifests: List[Dict[str, Any]] = []
        for root, dirs, files in os.walk(MANIFEST_BASE_PATH):
            for file in files:
                if file.endswith('.yml') and not file.startswith('shared'):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(MANIFEST_BASE_PATH)
                    enriched = _load_and_enrich_manifest(str(relative_path))
                    manifests.append(enriched)
        return manifests
    else:
        # Use the normal filtered list
        manifests = get_all_manifest_files()
        return manifests

@router.get("/list/model/{model}")
def list_manifests_by_model(model: str, include_incompatible: bool = False):
    """List all manifest names for a specific model.
    
    Args:
        model: The model name to filter by
        include_incompatible: If True, include manifests not compatible with current system
    """
    if include_incompatible:
        manifests = list_all_manifests(include_incompatible=True)
    else:
        manifests = get_all_manifest_files()
    filtered = [m for m in manifests if m.get("model") == model]
    if not filtered:
        raise HTTPException(status_code=404, detail=f"No manifests found for model: {model}")
    return filtered

@router.get("/list/type/{model_type}")
def list_manifests_by_model_type(model_type: str, include_incompatible: bool = False):
    """List all manifest names for a specific model type.
    
    Args:
        model_type: The model type to filter by
        include_incompatible: If True, include manifests not compatible with current system
    """
    if include_incompatible:
        manifests = list_all_manifests(include_incompatible=True)
    else:
        manifests = get_all_manifest_files()
    filtered: List[Dict[str, Any]] = []
    for m in manifests:
        mt = m.get("model_type")
        if isinstance(mt, list):
            if model_type in mt:
                filtered.append(m)
        else:
            if mt == model_type:
                filtered.append(m)
    if not filtered:
        raise HTTPException(status_code=404, detail=f"No manifests found for model_type: {model_type}")
    return filtered

@router.get("/list/model/{model}/model_type/{model_type}")
def list_manifests_by_model_and_type(model: str, model_type: str, include_incompatible: bool = False):
    """List all manifest names for a specific model and model type combination.
    
    Args:
        model: The model name to filter by
        model_type: The model type to filter by
        include_incompatible: If True, include manifests not compatible with current system
    """
    if include_incompatible:
        manifests = list_all_manifests(include_incompatible=True)
    else:
        manifests = get_all_manifest_files()
    filtered: List[Dict[str, Any]] = []
    for m in manifests:
        model_match = (m.get("model") == model)
        mt = m.get("model_type")
        if isinstance(mt, list):
            type_match = (model_type in mt)
        else:
            type_match = (mt == model_type)
        if model_match and type_match:
            filtered.append(m)
    if not filtered:
        raise HTTPException(status_code=404, detail=f"No manifests found for model: {model} and model_type: {model_type}")
    return filtered


@router.get("/{manifest_id}")
def get_manifest_by_id(manifest_id: str) -> Dict[Any, Any]:
    return get_manifest(manifest_id)

@router.get("/{manifest_id}/part")
def get_manifest_part(manifest_id: str, path: Optional[str] = None):
    """
    Return a specific part of the enriched manifest given a dot-separated path.
    Examples:
      - path=spec.loras
      - path=spec.components
      - path=spec.components.0.model_path
    Supports numeric tokens to index into lists.
    """
    doc = get_manifest(manifest_id)
    if not path:
        return doc
    value: Any = doc
    for token in path.split("."):
        if isinstance(value, dict):
            if token in value:
                value = value[token]
            else:
                raise HTTPException(status_code=404, detail=f"Key not found at segment '{token}'")
        elif isinstance(value, list):
            try:
                idx = int(token)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Expected list index at segment '{token}'")
            if idx < 0 or idx >= len(value):
                raise HTTPException(status_code=404, detail=f"List index out of range at segment '{token}'")
            value = value[idx]
        else:
            raise HTTPException(status_code=404, detail=f"Path not traversable at segment '{token}'")

    return value
