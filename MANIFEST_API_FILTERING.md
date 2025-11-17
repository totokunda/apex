# Manifest API Compute Filtering

## Overview

The manifest API now automatically filters out models that are incompatible with the current system's compute capabilities. This prevents users from seeing or attempting to run models that won't work on their hardware.

## Changes Made

### 1. Automatic Filtering in List Endpoints

All manifest list endpoints now filter results based on compute compatibility:

- `/manifest/list` - Lists only compatible manifests
- `/manifest/list/model/{model}` - Lists compatible manifests for a model
- `/manifest/list/type/{model_type}` - Lists compatible manifests for a type
- `/manifest/list/model/{model}/model_type/{model_type}` - Lists compatible manifests for model+type

### 2. New Fields in Manifest Response

Each manifest now includes compute compatibility information:

```json
{
  "id": "nunchaku-flux-dev-text-to-image",
  "name": "Nunchaku Flux Dev Text to Image",
  "compute_compatible": false,
  "compute_compatibility_error": "CUDA architecture 'sm_90' (Hopper) is excluded...",
  "compute_requirements_present": true,
  ...
}
```

**Fields:**
- `compute_compatible` (bool): Whether the manifest can run on the current system
- `compute_compatibility_error` (string|null): Detailed error message if incompatible
- `compute_requirements_present` (bool): Whether the manifest specifies compute requirements

### 3. New System Info Endpoint

Get information about the current system's compute capabilities:

**Endpoint:** `GET /manifest/system/compute`

**Response:**
```json
{
  "compute_type": "cuda",
  "cuda_compute_capability": 9.0,
  "device_name": "NVIDIA H100 80GB HBM3",
  "device_count": 1,
  "metal_version": null,
  "cpu_info": {}
}
```

### 4. Optional Include Incompatible Flag

All list endpoints support an optional `include_incompatible` query parameter:

**Example:**
```
GET /manifest/list?include_incompatible=true
```

When `include_incompatible=true`, the API returns ALL manifests including those incompatible with the current system. Each manifest still includes the compatibility fields so clients can filter or display warnings.

## API Examples

### Get System Compute Info

```bash
curl http://localhost:8000/manifest/system/compute
```

```json
{
  "compute_type": "cuda",
  "cuda_compute_capability": 9.0,
  "device_name": "NVIDIA H100 80GB HBM3",
  "device_count": 1
}
```

### List Compatible Manifests (Default)

```bash
curl http://localhost:8000/manifest/list
```

On H100 (sm_90), this will NOT include any nunchaku models since they exclude sm_90.

### List All Manifests Including Incompatible

```bash
curl http://localhost:8000/manifest/list?include_incompatible=true
```

This returns ALL manifests, including nunchaku models on H100. Each manifest will have `compute_compatible: false` and include an error message.

### List Compatible Flux Models

```bash
curl http://localhost:8000/manifest/list/model/flux
```

On H100, this will exclude nunchaku-flux models but include other flux variants.

### List All Flux Models Including Incompatible

```bash
curl http://localhost:8000/manifest/list/model/flux?include_incompatible=true
```

Returns all flux models with compatibility information.

## Behavior on Different Systems

### On H100 (Hopper / sm_90)

**Without `include_incompatible`:**
- ❌ All 7 nunchaku models are excluded from lists
- ✅ Other models (non-nunchaku) are included

**With `include_incompatible=true`:**
- ⚠️ All models included, nunchaku marked as incompatible
- Each nunchaku model shows: `"compute_compatible": false`

### On RTX 4090 (Ada Lovelace / sm_89)

**Without `include_incompatible`:**
- ✅ All 7 nunchaku models are included
- ✅ All other models are included

### On RTX 3090 (Ampere / sm_86)

**Without `include_incompatible`:**
- ✅ All 7 nunchaku models are included
- ✅ All other models are included

### On CPU-only Systems

**Without `include_incompatible`:**
- ✅ Nunchaku models that support CPU are included
- ❌ CUDA-only models are excluded

## Implementation Details

### Compute Capability Caching

The system's compute capability is detected once at startup and cached:

```python
_SYSTEM_COMPUTE_CAPABILITY: Optional[ComputeCapability] = None

def _get_system_compute_capability() -> ComputeCapability:
    """Get the system's compute capability (cached)."""
    global _SYSTEM_COMPUTE_CAPABILITY
    if _SYSTEM_COMPUTE_CAPABILITY is None:
        _SYSTEM_COMPUTE_CAPABILITY = get_compute_capability()
    return _SYSTEM_COMPUTE_CAPABILITY
```

### Manifest Enrichment

During manifest loading, compute compatibility is checked:

```python
# In _load_and_enrich_manifest()
compute_requirements = spec.get("compute_requirements")
if compute_requirements:
    system_capability = _get_system_compute_capability()
    is_compatible, compatibility_error = validate_compute_requirements(
        compute_requirements, 
        system_capability
    )
    content["compute_compatible"] = is_compatible
    content["compute_compatibility_error"] = compatibility_error
else:
    # No requirements = compatible with all systems
    content["compute_compatible"] = True
```

### List Filtering

The filtered list implementation:

```python
def _get_all_manifest_files_uncached() -> List[Dict[str, Any]]:
    manifests: List[Dict[str, Any]] = []
    
    for root, dirs, files in os.walk(MANIFEST_BASE_PATH):
        for file in files:
            if file.endswith('.yml') and not file.startswith('shared'):
                enriched = _load_and_enrich_manifest(str(relative_path))
                
                # Only include manifests compatible with current system
                if enriched.get("compute_compatible", True):
                    manifests.append(enriched)
                
    return manifests
```

## Benefits

### For Users
1. **No Confusing Errors**: Users don't see models that won't work on their system
2. **Clear Information**: When viewing all models, compatibility status is clear
3. **Better UX**: Model lists are relevant to the current hardware

### For Developers
1. **Automatic Filtering**: No manual filtering needed in client code
2. **Backward Compatible**: Manifests without requirements are always shown
3. **Flexible**: Can opt-in to see all models with `include_incompatible=true`

### For Operations
1. **H100 Compatibility**: Nunchaku models automatically hidden on H100
2. **Future-Proof**: New architectures can be handled via YAML updates
3. **Transparent**: System info endpoint shows current capabilities

## Testing

Verify filtering works correctly:

```bash
cd /home/tosin_coverquick_co/apex
PYTHONPATH=/home/tosin_coverquick_co/apex python3 test_manifest_filtering_simple.py
```

Expected output on H100:
```
✓ PASS: All nunchaku manifests correctly exclude H100
Compatible: 0
Incompatible: 7
```

Expected output on RTX 3090:
```
✓ PASS: All nunchaku manifests are compatible
Compatible: 7
Incompatible: 0
```

## Migration Notes

### For Existing Clients

**Breaking Change:** On H100 systems, nunchaku models will no longer appear in default list responses.

**Migration Path:**
1. If you need to show all models regardless of compatibility, add `?include_incompatible=true`
2. Check the `compute_compatible` field before attempting to run a model
3. Display `compute_compatibility_error` to users when showing incompatible models

### For New Manifests

When creating new manifests, add compute requirements if your model has hardware limitations:

```yaml
spec:
  compute_requirements:
    min_cuda_compute_capability: 7.5
    supported_compute_types:
    - cuda
    - cpu
    excluded_cuda_architectures:
    - sm_90  # Exclude H100 if not supported
```

## Related Documentation

- `docs/compute_requirements.md` - Compute requirements YAML specification
- `ARCHITECTURE_FILTERING_UPDATE.md` - Architecture filtering feature details
- `NUNCHAKU_H100_EXCLUSION.md` - Nunchaku H100 exclusion specifics
- `src/utils/compute.py` - Compute capability detection utilities
- `src/engine/base_engine.py` - Engine-level validation

## API Reference Summary

| Endpoint | Default Behavior | With `include_incompatible=true` |
|----------|------------------|----------------------------------|
| `/manifest/list` | Only compatible | All manifests |
| `/manifest/list/model/{model}` | Only compatible for model | All for model |
| `/manifest/list/type/{type}` | Only compatible for type | All for type |
| `/manifest/list/model/{model}/model_type/{type}` | Only compatible | All matching |
| `/manifest/{id}` | Always returns (with compatibility info) | N/A |
| `/manifest/system/compute` | Current system info | N/A |

## Future Enhancements

Potential improvements:
1. Add compute requirements to more models
2. Support version-specific requirements (e.g., CUDA toolkit version)
3. Add driver version requirements
4. Support compound requirements (e.g., "sm_80+ with 40GB+ VRAM")
5. Cache compatibility checks per manifest to improve performance

