"""
Tests for compute capability detection and validation.
"""
import pytest
from src.utils.compute import (
    get_compute_capability,
    validate_compute_requirements,
    get_cuda_compute_capability,
    get_cuda_capability_name,
    get_cuda_architecture_from_capability,
    ComputeCapability,
)


def test_get_compute_capability():
    """Test that compute capability detection returns valid results."""
    cap = get_compute_capability()
    
    assert cap is not None
    assert cap.compute_type in ["cuda", "metal", "cpu"]
    assert isinstance(cap.to_dict(), dict)
    
    if cap.compute_type == "cuda":
        assert cap.cuda_compute_capability is not None
        assert cap.cuda_compute_capability >= 3.0
        assert cap.device_name is not None
    elif cap.compute_type == "metal":
        assert cap.metal_version is not None
    else:
        assert cap.cpu_info is not None
        assert "platform" in cap.cpu_info


def test_validate_compute_requirements_supported_types():
    """Test validation of supported compute types."""
    # Get current capability
    current = get_compute_capability()
    
    # Should pass if current type is in supported list
    requirements = {
        "supported_compute_types": [current.compute_type]
    }
    is_valid, error = validate_compute_requirements(requirements, current)
    assert is_valid
    assert error is None
    
    # Should fail if current type is not in supported list
    if current.compute_type == "cuda":
        unsupported_type = "metal"
    else:
        unsupported_type = "cuda"
    
    requirements = {
        "supported_compute_types": [unsupported_type]
    }
    is_valid, error = validate_compute_requirements(requirements, current)
    assert not is_valid
    assert error is not None
    assert unsupported_type in error


def test_validate_compute_requirements_cuda_capability():
    """Test validation of minimum CUDA compute capability."""
    current = get_compute_capability()
    
    if current.compute_type != "cuda":
        pytest.skip("CUDA not available")
    
    # Should pass with low requirement
    requirements = {
        "min_cuda_compute_capability": 3.0,
        "supported_compute_types": ["cuda"]
    }
    is_valid, error = validate_compute_requirements(requirements, current)
    assert is_valid
    assert error is None
    
    # Should fail with very high requirement
    requirements = {
        "min_cuda_compute_capability": 99.0,
        "supported_compute_types": ["cuda"]
    }
    is_valid, error = validate_compute_requirements(requirements, current)
    assert not is_valid
    assert error is not None
    assert "99.0" in error


def test_cuda_capability_names():
    """Test CUDA architecture name lookup."""
    assert "Ampere" in get_cuda_capability_name(8.0)
    assert "Ampere" in get_cuda_capability_name(8.6)
    assert "Turing" in get_cuda_capability_name(7.5)
    assert "Volta" in get_cuda_capability_name(7.0)
    assert "Hopper" in get_cuda_capability_name(9.0)


def test_compute_capability_repr():
    """Test string representation of ComputeCapability."""
    cuda_cap = ComputeCapability(
        compute_type="cuda",
        cuda_compute_capability=8.6,
        device_name="RTX 3090"
    )
    repr_str = repr(cuda_cap)
    assert "cuda" in repr_str
    assert "8.6" in repr_str
    
    cpu_cap = ComputeCapability(
        compute_type="cpu",
        cpu_info={"platform": "Linux"}
    )
    repr_str = repr(cpu_cap)
    assert "cpu" in repr_str


def test_empty_requirements():
    """Test that empty requirements always pass."""
    requirements = {}
    is_valid, error = validate_compute_requirements(requirements)
    assert is_valid
    assert error is None


def test_cuda_architecture_from_capability():
    """Test conversion of compute capability to SM version."""
    assert get_cuda_architecture_from_capability(7.5) == "sm_75"
    assert get_cuda_architecture_from_capability(8.0) == "sm_80"
    assert get_cuda_architecture_from_capability(8.6) == "sm_86"
    assert get_cuda_architecture_from_capability(8.9) == "sm_89"
    assert get_cuda_architecture_from_capability(9.0) == "sm_90"


def test_excluded_cuda_architectures():
    """Test excluding specific CUDA architectures."""
    current = get_compute_capability()
    
    if current.compute_type != "cuda":
        pytest.skip("CUDA not available")
    
    current_arch = get_cuda_architecture_from_capability(current.cuda_compute_capability)
    
    # Should fail when excluding current architecture
    requirements = {
        "supported_compute_types": ["cuda"],
        "excluded_cuda_architectures": [current_arch]
    }
    is_valid, error = validate_compute_requirements(requirements, current)
    assert not is_valid
    assert error is not None
    assert "excluded" in error.lower()
    assert current_arch in error
    
    # Should pass when excluding a different architecture
    if current_arch != "sm_75":
        different_arch = "sm_75"
    else:
        different_arch = "sm_80"
    
    requirements = {
        "supported_compute_types": ["cuda"],
        "excluded_cuda_architectures": [different_arch]
    }
    is_valid, error = validate_compute_requirements(requirements, current)
    assert is_valid
    assert error is None


def test_allowed_cuda_architectures():
    """Test allowing only specific CUDA architectures."""
    current = get_compute_capability()
    
    if current.compute_type != "cuda":
        pytest.skip("CUDA not available")
    
    current_arch = get_cuda_architecture_from_capability(current.cuda_compute_capability)
    
    # Should pass when current architecture is in allowed list
    requirements = {
        "supported_compute_types": ["cuda"],
        "allowed_cuda_architectures": [current_arch, "sm_75", "sm_80"]
    }
    is_valid, error = validate_compute_requirements(requirements, current)
    assert is_valid
    assert error is None
    
    # Should fail when current architecture is not in allowed list
    if current_arch != "sm_75":
        allowed_list = ["sm_75"]
    else:
        allowed_list = ["sm_80"]
    
    requirements = {
        "supported_compute_types": ["cuda"],
        "allowed_cuda_architectures": allowed_list
    }
    is_valid, error = validate_compute_requirements(requirements, current)
    assert not is_valid
    assert error is not None
    assert "not in allowed list" in error.lower()


def test_combined_architecture_constraints():
    """Test combining minimum capability with architecture filters."""
    # Create a mock capability for testing
    mock_cap = ComputeCapability(
        compute_type="cuda",
        cuda_compute_capability=8.6,
        device_name="Test RTX 3090"
    )
    
    # Should pass all constraints
    requirements = {
        "min_cuda_compute_capability": 8.0,
        "supported_compute_types": ["cuda"],
        "allowed_cuda_architectures": ["sm_86", "sm_80"]
    }
    is_valid, error = validate_compute_requirements(requirements, mock_cap)
    assert is_valid
    assert error is None
    
    # Should fail on architecture exclusion even though capability is sufficient
    requirements = {
        "min_cuda_compute_capability": 8.0,
        "supported_compute_types": ["cuda"],
        "excluded_cuda_architectures": ["sm_86"]
    }
    is_valid, error = validate_compute_requirements(requirements, mock_cap)
    assert not is_valid
    assert "excluded" in error.lower()

