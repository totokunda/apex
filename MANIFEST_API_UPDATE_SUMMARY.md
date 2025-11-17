# Manifest API Compute Filtering - Update Summary

## What Was Changed

The manifest API (`src/api/manifest.py`) now automatically filters manifests based on compute compatibility with the current system.

## Key Changes

### 1. Added Compute Capability Detection

```python
from src.utils.compute import get_compute_capability, validate_compute_requirements

_SYSTEM_COMPUTE_CAPABILITY: Optional[ComputeCapability] = None

def _get_system_compute_capability() -> ComputeCapability:
    """Get the system's compute capability (cached)."""
    # Cached at module level for performance
```

### 2. Enriched Manifests with Compatibility Info

Each manifest now includes:
- `compute_compatible` - Boolean indicating if it can run on current system
- `compute_compatibility_error` - Error message if incompatible
- `compute_requirements_present` - Whether manifest specifies requirements

### 3. Automatic Filtering in List Endpoints

By default, all list endpoints now return ONLY compatible manifests:
- `/manifest/list`
- `/manifest/list/model/{model}`
- `/manifest/list/type/{model_type}`
- `/manifest/list/model/{model}/model_type/{model_type}`

### 4. Optional Include All Flag

Added `include_incompatible` query parameter to all list endpoints:
- `?include_incompatible=false` (default) - Only compatible manifests
- `?include_incompatible=true` - All manifests with compatibility info

### 5. New System Info Endpoint

```
GET /manifest/system/compute
```

Returns current system's compute capabilities.

## Impact on H100 Systems

### Before This Change
- All 7 nunchaku models appeared in manifest lists
- Users would get cryptic CUDA errors when trying to run them
- No indication that models were incompatible

### After This Change
- Nunchaku models are automatically filtered out on H100
- Users only see models that will actually work
- Individual manifest GET still works, but shows compatibility status
- Can view all models with `?include_incompatible=true`

## Example Behavior

### On H100 (sm_90)

**Default list request:**
```bash
curl http://localhost:8000/manifest/list
```
Result: Returns all manifests EXCEPT 7 nunchaku models

**Include all:**
```bash
curl http://localhost:8000/manifest/list?include_incompatible=true
```
Result: Returns all manifests, nunchaku models have `compute_compatible: false`

### On RTX 3090 (sm_86)

**Default list request:**
```bash
curl http://localhost:8000/manifest/list
```
Result: Returns all manifests INCLUDING nunchaku models

## Testing Results

Test script: `test_manifest_filtering_simple.py`

**On H100:**
```
✓ PASS: All nunchaku manifests correctly exclude H100
Compatible: 0
Incompatible: 7
Total: 7
```

All 7 nunchaku models are correctly identified as incompatible and will be filtered from default lists.

## Files Modified

1. **src/api/manifest.py**
   - Added compute capability imports
   - Added `_get_system_compute_capability()` function
   - Modified `_load_and_enrich_manifest()` to add compatibility fields
   - Modified `_get_all_manifest_files_uncached()` to filter incompatible manifests
   - Added `include_incompatible` parameter to list endpoints
   - Added `/manifest/system/compute` endpoint

## Files Created

1. **MANIFEST_API_FILTERING.md** - Complete documentation of the feature
2. **test_manifest_filtering_simple.py** - Test script for verification
3. **MANIFEST_API_UPDATE_SUMMARY.md** - This file

## Backward Compatibility

### Safe Changes
- Manifests without compute requirements: Always show (compatible with all)
- Individual manifest GET by ID: Still works, just adds compatibility fields
- New query parameter: Optional, defaults to filtering behavior

### Potential Breaking Changes
- **H100 users:** Will no longer see nunchaku models in default lists
  - **Mitigation:** Use `?include_incompatible=true` to see all models
  
- **Clients expecting fixed manifest counts:** Count will vary by system
  - **Mitigation:** Check `compute_compatible` field if you need all models

## Benefits

1. **Better UX:** Users only see models they can actually run
2. **Fewer Errors:** No confusing CUDA errors from incompatible models
3. **Clear Status:** Compatibility information always available
4. **Flexible:** Can opt-in to see all models when needed
5. **Automatic:** No manual filtering needed in client code
6. **Future-Proof:** New hardware restrictions handled via YAML

## Integration Guide

### For Frontend Developers

**Default behavior (recommended):**
```javascript
// This will only return compatible models
const response = await fetch('/manifest/list');
const manifests = await response.json();
// All manifests in this list will work on current system
```

**Show all models with warnings:**
```javascript
// Include incompatible models
const response = await fetch('/manifest/list?include_incompatible=true');
const manifests = await response.json();

// Filter and display appropriately
const compatible = manifests.filter(m => m.compute_compatible);
const incompatible = manifests.filter(m => !m.compute_compatible);

// Show incompatible with warning message
incompatible.forEach(m => {
  console.warn(`${m.name}: ${m.compute_compatibility_error}`);
});
```

**Get system info:**
```javascript
const response = await fetch('/manifest/system/compute');
const systemInfo = await response.json();
console.log(`Running on ${systemInfo.compute_type}`);
if (systemInfo.compute_type === 'cuda') {
  console.log(`GPU: ${systemInfo.device_name}`);
  console.log(`Compute: ${systemInfo.cuda_compute_capability}`);
}
```

### For Backend Developers

The filtering happens automatically. If you're using the manifest API:

```python
from src.api.manifest import get_all_manifest_files

# Returns only compatible manifests
manifests = get_all_manifest_files()

# Each manifest has compatibility info
for manifest in manifests:
    print(f"{manifest['name']}: {manifest['compute_compatible']}")
```

## Performance Considerations

1. **Compute capability detection:** Cached at module level (one-time cost)
2. **Validation per manifest:** Runs once during enrichment
3. **List caching:** Still works with `APEX_MANIFEST_CACHE` env var
4. **No extra API calls:** All info included in manifest response

## Related Changes

This update is part of a larger compute requirements system:

1. ✅ Core compute detection utilities (`src/utils/compute.py`)
2. ✅ Engine-level validation (`src/engine/base_engine.py`)
3. ✅ Manifest loading normalization (`src/manifest/loader.py`)
4. ✅ All 7 nunchaku manifests updated with H100 exclusion
5. ✅ **Manifest API filtering (this update)**

## Next Steps

Consider these future enhancements:

1. Add compute requirements to more models as needed
2. Add UI indicators for compute compatibility in frontend
3. Add metrics/logging for filtered manifests
4. Consider caching validation results per manifest
5. Add API endpoint to validate arbitrary compute requirements

## Questions?

See full documentation in:
- `MANIFEST_API_FILTERING.md` - Complete API documentation
- `docs/compute_requirements.md` - YAML specification
- `NUNCHAKU_H100_EXCLUSION.md` - Nunchaku-specific details

