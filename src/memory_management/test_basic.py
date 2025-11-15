"""Basic tests for memory management system."""

import torch
import torch.nn as nn
import time
import sys
import os

# Add the memory_management module directory to the path when executed as a script
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .memory_manager import MemoryManager, auto_manage_model
    from .config import MemoryConfig
    from .memory_monitor import MemoryMonitor
except ImportError:
    from memory_manager import MemoryManager, auto_manage_model  # type: ignore
    from config import MemoryConfig  # type: ignore
    from memory_monitor import MemoryMonitor  # type: ignore


def test_memory_monitor():
    """Test memory monitoring functionality."""
    print("Testing MemoryMonitor...")

    config = MemoryConfig()
    monitor = MemoryMonitor(config)

    # Test basic functionality
    monitor.start_monitoring()
    time.sleep(0.5)  # Let it collect some data

    # Check if we can get stats
    gpu_usage = monitor.get_gpu_usage()
    cpu_usage = monitor.get_cpu_usage()
    disk_usage = monitor.get_disk_usage()

    print(f"GPU usage: {gpu_usage:.2%}")
    print(f"CPU usage: {cpu_usage:.2%}")
    print(f"Disk usage: {disk_usage:.2%}")

    # Get memory summary
    summary = monitor.get_memory_summary()
    assert "gpu" in summary
    assert "cpu" in summary
    assert "disk" in summary

    monitor.stop_monitoring()
    print("✓ MemoryMonitor test passed\n")


def test_config():
    """Test configuration system."""
    print("Testing MemoryConfig...")

    # Test default config
    config = MemoryConfig()
    assert config.gpu_offload_threshold == 0.85
    assert config.enable_cpu_offload == True

    # Test preset configs
    low_mem_config = MemoryConfig.for_low_memory()
    assert low_mem_config.gpu_offload_threshold < config.gpu_offload_threshold

    high_perf_config = MemoryConfig.for_high_performance()
    assert high_perf_config.gpu_offload_threshold > config.gpu_offload_threshold

    # Test validation
    try:
        invalid_config = MemoryConfig(
            gpu_offload_threshold=0.95,
            gpu_emergency_threshold=0.90,  # Invalid: emergency < offload
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

    print("✓ MemoryConfig test passed\n")


def test_basic_module_wrapping():
    """Test basic module wrapping functionality."""
    print("Testing basic module wrapping...")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
    )

    # Test with auto_manage_model
    manager = auto_manage_model(model)

    # Get initial summary
    summary = manager.get_memory_summary()
    print(f"Managed modules: {summary['system']['managed_modules']}")

    # Test basic forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = torch.randn(8, 64, device=device)
    output = model(x)

    assert output.shape == (8, 10), f"Expected shape (8, 10), got {output.shape}"

    # Test memory info
    summary = manager.get_memory_summary()
    assert summary["system"]["managed_modules"] > 0

    manager.cleanup()
    print("✓ Basic module wrapping test passed\n")


def test_manual_offloading():
    """Test manual offloading functionality."""
    print("Testing manual offloading...")

    # Create a larger model for testing
    model = nn.Sequential(nn.Linear(256, 512), nn.Linear(512, 512), nn.Linear(512, 256))

    config = MemoryConfig()
    manager = MemoryManager(config)
    manager.start()

    # Wrap the model
    wrapped_model = manager.wrap_model(model)

    # Get a reference to one of the wrapped modules
    wrapped_modules = []
    for name, module in wrapped_model.named_modules():
        if hasattr(module, "offload_all"):
            wrapped_modules.append(module)

    assert len(wrapped_modules) > 0, "No wrapped modules found"

    # Test manual offloading
    test_module = wrapped_modules[0]

    # Get initial state
    initial_info = test_module.get_memory_info()
    print(f"Initial offload ratio: {initial_info['offload_ratio']:.2%}")

    # Manual offload
    test_module.offload_all()

    # Check state after offload
    offloaded_info = test_module.get_memory_info()
    print(f"After offload ratio: {offloaded_info['offload_ratio']:.2%}")

    # Manual reload
    test_module.load_all()

    # Check state after reload
    reloaded_info = test_module.get_memory_info()
    print(f"After reload ratio: {reloaded_info['offload_ratio']:.2%}")

    manager.cleanup()
    print("✓ Manual offloading test passed\n")


def test_eager_offload_on_wrap_behavior():
    """Ensure eager_offload_on_wrap toggles initial offload state."""
    print("Testing eager offload on wrap behavior...")

    config = MemoryConfig(eager_offload_on_wrap=True)
    manager = MemoryManager(config)
    manager.start()
    model = nn.Linear(32, 32)
    wrapped = manager.wrap_module(model, module_id="eager_linear")

    info = wrapped.get_memory_info()
    retained_1d = sum(1 for meta in wrapped._param_metadata.values() if len(meta.shape) <= 1)
    assert info["offloaded_parameters"] == info["total_parameters"] - retained_1d

    manager.cleanup()

    config_lazy = MemoryConfig(eager_offload_on_wrap=False)
    manager_lazy = MemoryManager(config_lazy)
    manager_lazy.start()
    model_lazy = nn.Linear(32, 32)
    wrapped_lazy = manager_lazy.wrap_module(model_lazy, module_id="lazy_linear")
    lazy_info = wrapped_lazy.get_memory_info()
    assert lazy_info["offloaded_parameters"] == 0
    manager_lazy.cleanup()

    print("✓ Eager offload on wrap test passed\n")


def test_bias_offload_policy():
    """Ensure 1D parameters stay resident unless explicitly allowed."""
    print("Testing bias offload policy...")

    config = MemoryConfig(eager_offload_on_wrap=True, offload_1d_parameters=False)
    manager = MemoryManager(config)
    manager.start()
    layer = nn.Linear(8, 4)
    wrapped = manager.wrap_module(layer, module_id="bias_default")
    assert "bias" in wrapped._param_metadata
    assert not wrapped._param_metadata["bias"].is_offloaded
    assert wrapped._original_module.bias.shape == torch.Size([4])
    manager.cleanup()

    config_on = MemoryConfig(eager_offload_on_wrap=True, offload_1d_parameters=True)
    manager_on = MemoryManager(config_on)
    manager_on.start()
    layer_on = nn.Linear(8, 4)
    wrapped_on = manager_on.wrap_module(layer_on, module_id="bias_enabled")
    assert wrapped_on._param_metadata["bias"].is_offloaded
    sample = torch.randn(2, 8)
    wrapped_on(sample)
    assert wrapped_on._original_module.bias.shape == torch.Size([4])
    manager_on.cleanup()

    print("✓ Bias offload policy test passed\n")


def test_placeholder_weights_reload_on_forward():
    """Simulate lost tracking and ensure placeholders reload before forward."""
    print("Testing placeholder reload safety...")

    config = MemoryConfig(eager_offload_on_wrap=True)
    manager = MemoryManager(config)
    manager.start()
    model = nn.Linear(16, 16)
    wrapped = manager.wrap_module(model, module_id="placeholder_test")

    # Force placeholder state
    wrapped.offload_all()
    weight_meta = wrapped._param_metadata["weight"]
    bias_meta = wrapped._param_metadata.get("bias")

    # Simulate stale tracking information
    wrapped._offloaded_params.clear()
    weight_meta.is_offloaded = False
    if bias_meta:
        bias_meta.is_offloaded = False

    x = torch.randn(4, 16)
    wrapped(x)

    assert wrapped._original_module.weight.shape == weight_meta.shape
    if bias_meta:
        assert wrapped._original_module.bias.shape == bias_meta.shape

    manager.cleanup()
    print("✓ Placeholder reload safety test passed\n")


def test_forward_pass_with_management():
    """Test that forward passes work correctly with memory management."""
    print("Testing forward pass with memory management...")

    # Create model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(100, 200)
            self.layer2 = nn.Linear(200, 200)
            self.layer3 = nn.Linear(200, 100)
            self.layer4 = nn.Linear(100, 10)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
            return self.layer4(x)

    model = TestModel()

    # Use low memory config to trigger offloading
    config = MemoryConfig.for_low_memory()
    manager = auto_manage_model(model, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Run multiple forward passes
    results = []
    for i in range(5):
        x = torch.randn(16, 100, device=device)
        output = model(x)
        results.append(output.mean().item())

        print(f"Forward pass {i+1}: mean output = {results[-1]:.4f}")

        # Small delay to allow memory management
        time.sleep(0.2)

    # Check that forward passes are producing reasonable outputs
    assert len(results) == 5
    assert all(isinstance(r, float) for r in results)

    # Get final summary
    summary = manager.get_memory_summary()
    print(
        f"Final summary - Offloads: {summary['system']['total_offloads']}, "
        f"Reloads: {summary['system']['total_reloads']}"
    )

    manager.cleanup()
    print("✓ Forward pass with management test passed\n")


def run_all_tests():
    """Run all basic tests."""
    print("Running Memory Management Basic Tests")
    print("=" * 50)

    try:
        test_config()
        test_memory_monitor()
        test_basic_module_wrapping()
        test_manual_offloading()
        test_eager_offload_on_wrap_behavior()
        test_bias_offload_policy()
        test_placeholder_weights_reload_on_forward()
        test_forward_pass_with_management()

        print("🎉 All tests passed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
