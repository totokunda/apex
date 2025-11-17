# Architecture Filtering Update

## Summary

Extended the compute requirements system to support fine-grained control over which CUDA GPU architectures (SM versions) are allowed or excluded. This allows you to skip problematic GPU generations like Hopper (sm_90) or Ada Lovelace (sm_89), or restrict execution to only tested hardware.

## What's New

### 1. Architecture-Level Control

You can now specify:
- **`excluded_cuda_architectures`**: List of SM versions to reject (e.g., skip Hopper)
- **`allowed_cuda_architectures`**: Whitelist of SM versions allowed (e.g., only Ampere)

### 2. New Utility Function

Added `get_cuda_architecture_from_capability(compute_capability: float) -> str`:
- Converts compute capability (e.g., 9.0) to SM version (e.g., "sm_90")
- Used internally for validation and available for external use

### 3. Enhanced Validation

The `validate_compute_requirements()` function now checks:
1. Compute type support (cuda/metal/cpu)
2. Minimum CUDA compute capability
3. **NEW**: Architecture exclusion list
4. **NEW**: Architecture allowlist

## YAML Examples

### Skip Hopper GPUs

```yaml
spec:
  compute_requirements:
    supported_compute_types:
    - cuda
    excluded_cuda_architectures:
    - sm_90  # H100, H800 (Hopper)
```

### Skip Multiple Architectures

```yaml
spec:
  compute_requirements:
    min_cuda_compute_capability: 7.5
    supported_compute_types:
    - cuda
    excluded_cuda_architectures:
    - sm_90  # Hopper
    - sm_89  # Ada Lovelace (RTX 4090)
```

### Allow Only Tested Hardware

```yaml
spec:
  compute_requirements:
    supported_compute_types:
    - cuda
    allowed_cuda_architectures:
    - sm_80  # A100
    - sm_86  # RTX 3090, RTX 3080
```

## SM Version Reference

| SM Version | Architecture | Example GPUs |
|------------|--------------|--------------|
| sm_70 | Volta | Tesla V100 |
| sm_75 | Turing | RTX 2080, RTX 2080 Ti |
| sm_80 | Ampere | A100, A30, A10 |
| sm_86 | Ampere | RTX 3090, RTX 3080, A40 |
| sm_89 | Ada Lovelace | RTX 4090, L4, L40 |
| sm_90 | Hopper | H100, H800 |

## Use Cases

### 1. Skip Problematic GPU Generation

If you discover a bug specific to Hopper GPUs:

```yaml
excluded_cuda_architectures:
- sm_90
```

### 2. Limit to Tested Hardware

Only allow execution on GPUs you've validated:

```yaml
allowed_cuda_architectures:
- sm_80  # A100 - tested
- sm_86  # RTX 3090 - tested
```

### 3. Avoid Cutting-Edge Hardware

Skip the newest architectures that may have driver issues:

```yaml
excluded_cuda_architectures:
- sm_90  # Hopper - too new
- sm_89  # Ada Lovelace - too new
```

### 4. Target Specific Data Center GPUs

For models optimized for A100:

```yaml
min_cuda_compute_capability: 8.0
allowed_cuda_architectures:
- sm_80  # A100 only
```

## Code Examples

### Detect Current Architecture

```python
from src.utils.compute import get_compute_capability, get_cuda_architecture_from_capability

cap = get_compute_capability()
if cap.compute_type == "cuda":
    arch = get_cuda_architecture_from_capability(cap.cuda_compute_capability)
    print(f"Running on {arch}")  # e.g., "sm_90"
```

### Validate Requirements

```python
from src.utils.compute import validate_compute_requirements

requirements = {
    "excluded_cuda_architectures": ["sm_90"],
    "supported_compute_types": ["cuda"]
}

is_valid, error = validate_compute_requirements(requirements)
if not is_valid:
    print(f"Cannot run on this GPU: {error}")
```

## Error Messages

### Excluded Architecture

```
Compute Validation Failed:
  CUDA architecture 'sm_90' (Hopper) is excluded. 
  Excluded: sm_90. 
  Device: NVIDIA H100 80GB HBM3

Current System:
  Compute Type: cuda
  CUDA Capability: 9.0
  Device: NVIDIA H100 80GB HBM3

Required:
  Supported Types: cuda
```

### Not in Allowed List

```
Compute Validation Failed:
  CUDA architecture 'sm_90' (Hopper) is not in allowed list. 
  Allowed: sm_80, sm_86. 
  Device: NVIDIA H100 80GB HBM3

Current System:
  Compute Type: cuda
  CUDA Capability: 9.0
  Device: NVIDIA H100 80GB HBM3
```

## Testing

All architecture filtering features are tested:

```bash
PYTHONPATH=/home/tosin_coverquick_co/apex python3 examples/compute_capability_example.py
```

Or run the test suite:

```python
# See tests/test_compute_capability.py for:
- test_cuda_architecture_from_capability()
- test_excluded_cuda_architectures()
- test_allowed_cuda_architectures()
- test_combined_architecture_constraints()
```

## Files Modified/Created

### Modified
1. `src/utils/compute.py` - Added `get_cuda_architecture_from_capability()` and architecture validation
2. `docs/compute_requirements.md` - Added architecture filtering documentation
3. `examples/compute_capability_example.py` - Added architecture filtering examples
4. `tests/test_compute_capability.py` - Added architecture filtering tests

### Created
1. `examples/compute_requirements_skip_hopper.yml` - Example: Skip Hopper GPUs
2. `examples/compute_requirements_ampere_only.yml` - Example: Allow only Ampere GPUs

## Backward Compatibility

All existing compute requirements continue to work:
- `min_cuda_compute_capability` still works as before
- `supported_compute_types` still works as before
- New fields are optional and ignored if not specified

## Integration with setup.py

This feature aligns with the SM target detection in `thirdparty/nunchaku/setup.py`:

```python
# In setup.py (line 47-58)
capability = torch.cuda.get_device_capability(i)
sm = f"{capability[0]}{capability[1]}"
# Maps to our SM version format: sm_75, sm_80, sm_86, sm_89, sm_90
```

You can now use the same SM version identifiers in your YAML manifests to control which GPUs can run your models.

