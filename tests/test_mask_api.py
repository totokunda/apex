"""
Tests for mask API functionality
"""
import pytest
import numpy as np
from pathlib import Path

# Test imports work correctly
def test_imports():
    """Test that all mask-related imports work."""
    from src.mask.mask import (
        ModelType,
        SAM2ModelActor,
        get_sam2_actor,
        extract_video_frame,
        mask_to_contours,
        MODEL_WEIGHTS,
        MODEL_CONFIGS
    )
    
    assert ModelType.SAM2_SMALL is not None
    assert len(MODEL_WEIGHTS) == 4
    assert len(MODEL_CONFIGS) == 4


def test_model_type_enum():
    """Test ModelType enum values."""
    from src.mask.mask import ModelType
    
    assert ModelType.SAM2_TINY.value == "sam2_tiny"
    assert ModelType.SAM2_SMALL.value == "sam2_small"
    assert ModelType.SAM2_BASE_PLUS.value == "sam2_base_plus"
    assert ModelType.SAM2_LARGE.value == "sam2_large"


def test_mask_to_contours():
    """Test mask to contours conversion."""
    from src.mask.mask import mask_to_contours
    
    # Create a simple binary mask (square)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1
    
    # Convert to contours
    contours = mask_to_contours(mask, simplify_tolerance=1.0)
    
    # Should have at least one contour
    assert len(contours) > 0
    
    # Contour should be a flat list with even number of values (x,y pairs)
    for contour in contours:
        assert isinstance(contour, list)
        assert len(contour) >= 6  # At least 3 points (x,y pairs)
        assert len(contour) % 2 == 0  # Even number (x,y pairs)


def test_mask_to_contours_multiple():
    """Test mask with multiple disconnected regions."""
    from src.mask.mask import mask_to_contours
    
    # Create mask with two separate squares
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1  # Small square top-left
    mask[60:90, 60:90] = 1  # Larger square bottom-right
    
    contours = mask_to_contours(mask, simplify_tolerance=1.0)
    
    # Should have two contours
    assert len(contours) == 2


def test_mask_to_contours_empty():
    """Test empty mask."""
    from src.mask.mask import mask_to_contours
    
    # Empty mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    contours = mask_to_contours(mask)
    
    # Should have no contours
    assert len(contours) == 0


def test_api_models():
    """Test Pydantic models for the API."""
    from src.api.mask import MaskRequest, MaskResponse
    
    # Test valid request
    request = MaskRequest(
        input_path="/path/to/image.jpg",
        tool="touch",
        points=[{"x": 100.0, "y": 200.0}]
    )
    
    assert request.input_path == "/path/to/image.jpg"
    assert request.tool == "touch"
    assert request.frame_number is None
    assert len(request.points) == 1
    
    # Test response
    response = MaskResponse(
        status="success",
        contours=[[1.0, 2.0, 3.0, 4.0]],
        message="Test message"
    )
    
    assert response.status == "success"
    assert len(response.contours) == 1


def test_api_prepare_mask_inputs():
    """Test input preparation for different tools."""
    from src.api.mask import prepare_mask_inputs, MaskRequest
    
    # Test brush tool
    request = MaskRequest(
        input_path="/test.jpg",
        tool="brush",
        points=[
            {"x": 10.0, "y": 20.0},
            {"x": 30.0, "y": 40.0}
        ]
    )
    
    inputs = prepare_mask_inputs(request, (480, 640, 3))
    
    assert inputs['point_coords'] is not None
    assert inputs['point_coords'].shape == (2, 2)
    assert inputs['point_labels'] is not None
    assert np.all(inputs['point_labels'] == 1)  # All positive by default
    
    # Test shape tool with box
    request_shape = MaskRequest(
        input_path="/test.jpg",
        tool="shape",
        box={"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0}
    )
    
    inputs_shape = prepare_mask_inputs(request_shape, (480, 640, 3))
    
    assert inputs_shape['box'] is not None
    assert inputs_shape['box'].shape == (4,)
    assert inputs_shape['box'][0] == 10.0
    assert inputs_shape['box'][3] == 200.0


def test_api_prepare_mask_inputs_with_labels():
    """Test input preparation with custom point labels."""
    from src.api.mask import prepare_mask_inputs, MaskRequest
    
    request = MaskRequest(
        input_path="/test.jpg",
        tool="touch",
        points=[
            {"x": 10.0, "y": 20.0},
            {"x": 30.0, "y": 40.0}
        ],
        point_labels=[1, 0]  # First positive, second negative
    )
    
    inputs = prepare_mask_inputs(request, (480, 640, 3))
    
    assert inputs['point_coords'] is not None
    assert inputs['point_labels'] is not None
    assert inputs['point_labels'][0] == 1
    assert inputs['point_labels'][1] == 0


def test_api_lasso_generates_box():
    """Test that lasso tool generates a bounding box from points."""
    from src.api.mask import prepare_mask_inputs, MaskRequest
    
    request = MaskRequest(
        input_path="/test.jpg",
        tool="lasso",
        points=[
            {"x": 10.0, "y": 20.0},
            {"x": 100.0, "y": 200.0},
            {"x": 50.0, "y": 150.0}
        ]
    )
    
    inputs = prepare_mask_inputs(request, (480, 640, 3))
    
    # Lasso should generate both points and a bounding box
    assert inputs['point_coords'] is not None
    assert inputs['box'] is not None
    
    # Box should encompass all points
    assert inputs['box'][0] == 10.0   # min x
    assert inputs['box'][1] == 20.0   # min y
    assert inputs['box'][2] == 100.0  # max x
    assert inputs['box'][3] == 200.0  # max y


if __name__ == "__main__":
    # Run basic tests that don't require heavy dependencies
    print("Testing imports...")
    test_imports()
    print("✓ Imports work")
    
    print("Testing ModelType enum...")
    test_model_type_enum()
    print("✓ ModelType enum works")
    
    print("Testing mask_to_contours...")
    test_mask_to_contours()
    test_mask_to_contours_multiple()
    test_mask_to_contours_empty()
    print("✓ mask_to_contours works")
    
    print("Testing API models...")
    test_api_models()
    print("✓ API models work")
    
    print("Testing input preparation...")
    test_api_prepare_mask_inputs()
    test_api_prepare_mask_inputs_with_labels()
    test_api_lasso_generates_box()
    print("✓ Input preparation works")
    
    print("\n✓ All tests passed!")

