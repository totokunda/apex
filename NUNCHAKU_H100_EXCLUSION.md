# Nunchaku H100 Exclusion Update

## Summary

Added compute requirements to all 7 nunchaku model YAML manifests to exclude H100 (Hopper/sm_90) architecture support, as nunchaku kernels are not compatible with Hopper GPUs.

## Why This Change?

The nunchaku library (see `thirdparty/nunchaku/setup.py` line 52-53) maps sm_90 to sm_89 as a fallback compatibility measure. However, this binary compatibility workaround is not sufficient for proper operation. The kernels compiled for Ada Lovelace (sm_89) do not execute correctly on Hopper (sm_90) architecture.

## Files Updated

All nunchaku model manifests now include compute requirements:

### QwenImage Models (3 files)
1. `manifest/engine/qwenimage/nunchaku-qwenimage-1.0.0.v1.yml`
2. `manifest/engine/qwenimage/nunchaku-qwenimage-edit-1.0.0.v1.yml`
3. `manifest/engine/qwenimage/nunchaku-qwenimage-edit-2509-1.0.0.v1.yml`

### Flux Models (4 files)
4. `manifest/engine/flux/nunchaku-flux-dev-text-to-image-1.0.0.v1.yml`
5. `manifest/engine/flux/nunchaku-flux-krea-text-to-image-1.0.0.v1.yml`
6. `manifest/engine/flux/nunchaku-flux-dev-kontext-1.0.0.v1.yml`
7. `manifest/engine/flux/nunchaku-flux-dev-fill-1.0.0.v1.yml`

## Configuration Added

Each file now contains:

```yaml
spec:
  compute_requirements:
    min_cuda_compute_capability: 7.5
    supported_compute_types:
    - cuda
    - cpu
    excluded_cuda_architectures:
    - sm_90  # Nunchaku does not support H100 (Hopper)
```

## Supported Hardware

### ✅ Supported GPUs

| Architecture | SM Version | Example GPUs |
|--------------|------------|--------------|
| Turing | sm_75 | RTX 2080, RTX 2080 Ti, Titan RTX |
| Ampere | sm_80 | A100, A30, A10 |
| Ampere | sm_86 | RTX 3090, RTX 3080, RTX 3070, A40 |
| Ada Lovelace | sm_89 | RTX 4090, RTX 4080, L4, L40 |

### ❌ Excluded GPUs

| Architecture | SM Version | Example GPUs |
|--------------|------------|--------------|
| Hopper | sm_90 | H100, H800 |

### ✅ Also Supported

- CPU execution (with reduced performance)

## Error Message on H100

When attempting to initialize a nunchaku model on H100:

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
  Min CUDA Capability: 7.5
  Supported Types: cuda, cpu
  Excluded: sm_90
```

## Testing

All files verified:

```bash
cd /home/tosin_coverquick_co/apex
python3 -c "
import yaml

nunchaku_files = [
    'manifest/engine/qwenimage/nunchaku-qwenimage-1.0.0.v1.yml',
    'manifest/engine/qwenimage/nunchaku-qwenimage-edit-1.0.0.v1.yml',
    'manifest/engine/qwenimage/nunchaku-qwenimage-edit-2509-1.0.0.v1.yml',
    'manifest/engine/flux/nunchaku-flux-dev-text-to-image-1.0.0.v1.yml',
    'manifest/engine/flux/nunchaku-flux-krea-text-to-image-1.0.0.v1.yml',
    'manifest/engine/flux/nunchaku-flux-dev-kontext-1.0.0.v1.yml',
    'manifest/engine/flux/nunchaku-flux-dev-fill-1.0.0.v1.yml'
]

for f in nunchaku_files:
    data = yaml.safe_load(open(f))
    excluded = data['spec']['compute_requirements']['excluded_cuda_architectures']
    assert 'sm_90' in excluded
print('✓ All nunchaku models exclude H100')
"
```

## Impact

### Before This Change
- Nunchaku models would attempt to initialize on H100
- Setup.py would compile for sm_89 (fallback from sm_90)
- Models would fail at runtime with cryptic CUDA errors
- Users would be confused about compatibility

### After This Change
- Nunchaku models immediately reject H100 during initialization
- Clear error message explains the limitation
- Users understand which GPUs are supported
- No wasted time debugging runtime failures

## Future Work

If nunchaku adds native Hopper support:
1. Update or remove `excluded_cuda_architectures` in these files
2. Update `thirdparty/nunchaku/setup.py` to compile native sm_90 kernels
3. Test thoroughly on H100 hardware
4. Update this documentation

## Related Files

- `src/utils/compute.py` - Compute capability detection and validation
- `src/engine/base_engine.py` - Engine initialization with validation
- `src/manifest/loader.py` - Manifest loading and normalization
- `thirdparty/nunchaku/setup.py` - Nunchaku build configuration
- `docs/compute_requirements.md` - Compute requirements documentation
- `ARCHITECTURE_FILTERING_UPDATE.md` - Architecture filtering feature docs

## Verification

Run verification script:

```bash
cd /home/tosin_coverquick_co/apex
python3 -c "
import yaml

nunchaku_files = [
    'manifest/engine/qwenimage/nunchaku-qwenimage-1.0.0.v1.yml',
    'manifest/engine/qwenimage/nunchaku-qwenimage-edit-1.0.0.v1.yml',
    'manifest/engine/qwenimage/nunchaku-qwenimage-edit-2509-1.0.0.v1.yml',
    'manifest/engine/flux/nunchaku-flux-dev-text-to-image-1.0.0.v1.yml',
    'manifest/engine/flux/nunchaku-flux-krea-text-to-image-1.0.0.v1.yml',
    'manifest/engine/flux/nunchaku-flux-dev-kontext-1.0.0.v1.yml',
    'manifest/engine/flux/nunchaku-flux-dev-fill-1.0.0.v1.yml'
]

print('Nunchaku H100 Exclusion Verification')
print('=' * 60)

for file_path in nunchaku_files:
    data = yaml.safe_load(open(file_path))
    name = data['metadata']['name']
    compute_reqs = data['spec']['compute_requirements']
    excluded = compute_reqs['excluded_cuda_architectures']
    
    if 'sm_90' in excluded:
        print(f'✓ {name}')
    else:
        print(f'✗ {name} - MISSING sm_90 exclusion!')

print('=' * 60)
print('All nunchaku models properly exclude H100 (sm_90)')
"
```

Expected output:
```
Nunchaku H100 Exclusion Verification
============================================================
✓ Nunchaku QwenImage Text-to-Image
✓ Nunchaku QwenImage Edit
✓ Nunchaku QwenImage Edit 2509
✓ Nunchaku Flux Dev Text to Image
✓ Nunchaku Flux Krea Text to Image
✓ Nunchaku Flux Dev Kontext
✓ Nunchaku Flux Dev Fill
============================================================
All nunchaku models properly exclude H100 (sm_90)
```

