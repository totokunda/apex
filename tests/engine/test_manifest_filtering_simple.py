"""
Simple test script to verify compute requirements in manifest files.
"""
import yaml
from pathlib import Path

# Test file paths
manifest_base = Path("/home/tosin_coverquick_co/apex/manifest/engine")

print("=" * 70)
print("Manifest Compute Requirements Test")
print("=" * 70)

# Get system info
from src.utils.compute import get_compute_capability, get_cuda_architecture_from_capability

capability = get_compute_capability()
print(f"\nSystem Compute Capability:")
print(f"  Type: {capability.compute_type}")
if capability.compute_type == "cuda":
    arch = get_cuda_architecture_from_capability(capability.cuda_compute_capability)
    print(f"  CUDA Capability: {capability.cuda_compute_capability}")
    print(f"  Architecture: {arch}")
    print(f"  Device: {capability.device_name}")

# Find all nunchaku manifests
print(f"\n" + "=" * 70)
print("Scanning Nunchaku Manifests")
print("=" * 70)

nunchaku_files = list(manifest_base.glob("**/nunchaku*.yml"))
print(f"\nFound {len(nunchaku_files)} nunchaku manifest files")

# Check each one
compatible_count = 0
incompatible_count = 0

for file_path in sorted(nunchaku_files):
    with open(file_path) as f:
        data = yaml.safe_load(f)
    
    name = data.get('metadata', {}).get('name', file_path.name)
    spec = data.get('spec', {})
    compute_reqs = spec.get('compute_requirements')
    
    if compute_reqs:
        from src.utils.compute import validate_compute_requirements
        is_valid, error = validate_compute_requirements(compute_reqs, capability)
        
        status = "✓ Compatible" if is_valid else "✗ Incompatible"
        print(f"\n{status}: {name}")
        print(f"  File: {file_path.name}")
        
        if not is_valid:
            incompatible_count += 1
            excluded = compute_reqs.get('excluded_cuda_architectures', [])
            if excluded:
                print(f"  Excluded: {excluded}")
            print(f"  Reason: {error[:80]}...")
        else:
            compatible_count += 1
    else:
        print(f"\n⚠ No requirements: {name}")
        compatible_count += 1

# Summary
print(f"\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Compatible: {compatible_count}")
print(f"Incompatible: {incompatible_count}")
print(f"Total: {len(nunchaku_files)}")

if capability.compute_type == "cuda" and capability.cuda_compute_capability == 9.0:
    print(f"\n✓ Running on H100 (sm_90)")
    print(f"  Expected: All {len(nunchaku_files)} nunchaku manifests should be incompatible")
    if incompatible_count == len(nunchaku_files):
        print("  ✓ PASS: All nunchaku manifests correctly exclude H100")
    else:
        print(f"  ✗ FAIL: Only {incompatible_count}/{len(nunchaku_files)} are incompatible")
else:
    print(f"\n✓ Running on {capability.compute_type}")
    if capability.compute_type == "cuda":
        print(f"  Architecture: {get_cuda_architecture_from_capability(capability.cuda_compute_capability)}")
    print(f"  Expected: All nunchaku manifests should be compatible")
    if compatible_count == len(nunchaku_files):
        print("  ✓ PASS: All nunchaku manifests are compatible")
    else:
        print(f"  ✗ FAIL: Only {compatible_count}/{len(nunchaku_files)} are compatible")

print(f"\n" + "=" * 70)

