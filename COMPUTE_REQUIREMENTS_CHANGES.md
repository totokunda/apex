# Compute Requirements Implementation Summary

## Overview
Added compute capability validation for model engines to ensure they only run on compatible hardware.

## Changes Made

### 1. New Utility Module: `src/utils/compute.py`
Created comprehensive compute capability detection and validation utilities:

- `get_compute_capability()` - Detects current system capabilities (CUDA/Metal/CPU)
- `validate_compute_requirements()` - Validates system meets specified requirements
- `get_cuda_compute_capability()` - Returns CUDA compute capability (e.g., 7.0, 8.6, 9.0)
- `get_cuda_architecture_from_capability()` - Converts capability to SM version (e.g., sm_90)
- `get_metal_version()` - Detects Metal version for Apple Silicon
- `get_cpu_info()` - Returns CPU platform and capability info
- `get_cuda_capability_name()` - Maps compute capability to architecture names
- `ComputeCapability` class - Encapsulates system compute info

### 2. Engine Validation: `src/engine/base_engine.py`
Modified BaseEngine to validate compute requirements during initialization:

- Added import for compute validation utilities
- Added `_validate_compute_requirements()` method
- Validation runs automatically after config loading
- Raises RuntimeError with detailed error message if requirements not met

### 3. Manifest Loader: `src/manifest/loader.py`
Updated to pass through compute requirements from YAML specs:

- Added `compute_requirements` to normalized fields list
- Also added `attention_types` for completeness

### 4. YAML Example: `manifest/engine/qwenimage/nunchaku-qwenimage-1.0.0.v1.yml`
Added compute requirements specification:

```yaml
spec:
  compute_requirements:
    min_cuda_compute_capability: 7.0
    supported_compute_types:
    - cuda
    - cpu
    - metal
```

### 5. Documentation: `docs/compute_requirements.md`
Created comprehensive documentation covering:

- Configuration syntax and options
- Common CUDA compute capabilities
- Example configurations
- Programmatic API usage
- Error messages and troubleshooting

### 6. Tests: `tests/test_compute_capability.py`
Created unit tests for:

- Compute capability detection
- Requirement validation (supported types, CUDA capability)
- Architecture name mapping
- Edge cases and error conditions

### 7. Example: `examples/compute_capability_example.py`
Created demonstration script showing:

- System capability detection
- Validation scenarios
- Integration with engine initialization

## Usage

### In YAML Manifests

```yaml
spec:
  compute_requirements:
    min_cuda_compute_capability: 7.0  # Optional: minimum CUDA version
    supported_compute_types:          # Optional: allowed compute types
    - cuda
    - cpu
    - metal
    excluded_cuda_architectures:      # Optional: skip specific GPUs
    - sm_90  # Skip Hopper (H100)
    - sm_89  # Skip Ada Lovelace (RTX 4090)
    # OR use allowed list instead:
    # allowed_cuda_architectures:
    # - sm_80  # A100
    # - sm_86  # RTX 3090
```

### Programmatic

```python
from src.utils.compute import get_compute_capability, validate_compute_requirements

# Detect system
cap = get_compute_capability()
print(f"Running on {cap.compute_type}")

# Validate requirements
requirements = {
    "min_cuda_compute_capability": 7.0,
    "supported_compute_types": ["cuda"]
}
is_valid, error = validate_compute_requirements(requirements)
if not is_valid:
    print(f"Error: {error}")
```

## Supported Platforms

- **CUDA**: NVIDIA GPUs with compute capability detection
- **Metal**: Apple Silicon (M1/M2/M3/M4) with Metal 2/3
- **CPU**: Intel/AMD/ARM with BF16 capability detection

## Common CUDA Compute Capabilities

- 7.0: Tesla V100 (Volta) - sm_70
- 7.5: RTX 2080 series (Turing) - sm_75
- 8.0: A100 (Ampere) - sm_80
- 8.6: RTX 3090, RTX 4090 (Ampere) - sm_86
- 8.9: L4, L40 (Ada Lovelace) - sm_89
- 9.0: H100 (Hopper) - sm_90

## Architecture Filtering

You can now control which specific GPU architectures are allowed or excluded:

### Use Cases

1. **Skip problematic architectures**: If you have a bug specific to Hopper GPUs, exclude `sm_90`
2. **Limit to tested hardware**: Only allow architectures you've tested, like `sm_80` and `sm_86`
3. **Avoid newer GPUs**: Exclude cutting-edge architectures that may have driver issues
4. **Target specific generations**: Combine with `min_cuda_compute_capability` for fine control

## Testing

Run tests with:
```bash
pytest tests/test_compute_capability.py
```

Run example:
```bash
PYTHONPATH=/home/tosin_coverquick_co/apex python3 examples/compute_capability_example.py
```

## Error Behavior

When requirements aren't met, engine initialization fails with detailed error:

```
Compute Validation Failed:
  CUDA compute capability 7.5 is below minimum required 8.0
  Device: NVIDIA GeForce RTX 2080 Ti

Current System:
  Compute Type: cuda
  CUDA Capability: 7.5
  Device: NVIDIA GeForce RTX 2080 Ti

Required:
  Min CUDA Capability: 8.0
  Supported Types: cuda
```

## Files Modified/Created

1. `src/utils/compute.py` (NEW)
2. `src/engine/base_engine.py` (MODIFIED)
3. `src/manifest/loader.py` (MODIFIED)
4. `manifest/engine/qwenimage/nunchaku-qwenimage-1.0.0.v1.yml` (MODIFIED)
5. `docs/compute_requirements.md` (NEW)
6. `tests/test_compute_capability.py` (NEW)
7. `examples/compute_capability_example.py` (NEW)

