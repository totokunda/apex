#!/usr/bin/env python3
"""
Demonstration script for the improved quantization system.

This script shows practical usage examples of the quantization system
integrated into the engine architecture.
"""

import os
import sys
import torch
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/workspace/apex/src')

def demo_basic_usage():
    """Demonstrate basic quantization usage"""
    print("=" * 60)
    print("DEMO: Basic Quantization Usage")
    print("=" * 60)
    
    from quantize.quantizer import quantize_model, QUANTIZER_CONFIGS
    
    print(f"Available quantization methods: {len(QUANTIZER_CONFIGS)} methods")
    print(f"Methods: {', '.join(list(QUANTIZER_CONFIGS.keys())[:5])}...")
    
    # Create a sample model
    class SampleTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(1000, 512)
            self.transformer_layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(512, 8, batch_first=True) 
                for _ in range(6)
            ])
            self.layer_norm = torch.nn.LayerNorm(512)
            self.output_proj = torch.nn.Linear(512, 256)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.transformer_layers:
                x = layer(x)
            x = self.layer_norm(x)
            return self.output_proj(x)
    
    model = SampleTransformer()
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Basic quantization
    print("\n1. Basic FP16 quantization:")
    quantized_fp16 = quantize_model(model, quant_method="basic_fp16", target_memory_gb=1.0)
    print("✓ FP16 quantization completed")
    
    # Advanced quantization with dtype preservation
    print("\n2. Advanced quantization with dtype preservation:")
    preserve_dtypes = {
        ".*embedding.*": torch.float32,  # Keep embeddings in FP32
        ".*layer_norm.*": torch.float16  # LayerNorm in FP16
    }
    
    quantized_advanced = quantize_model(
        model, 
        quant_method="basic_bf16",
        preserve_dtypes=preserve_dtypes,
        target_memory_gb=0.8
    )
    print("✓ Advanced quantization with dtype preservation completed")
    
    # Show dtype information
    print("\nModel layer dtypes after quantization:")
    for name, param in quantized_advanced.named_parameters():
        if any(key in name for key in ["embedding", "layer_norm", "output_proj"]):
            print(f"  {name}: {param.dtype}")

def demo_save_load():
    """Demonstrate save and load functionality"""
    print("\n" + "=" * 60)
    print("DEMO: Save and Load Functionality")
    print("=" * 60)
    
    from quantize.quantizer import ModelQuantizer
    
    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 10)
    )
    
    # Quantize with specific settings
    quantizer = ModelQuantizer(
        quant_method="basic_fp16",
        target_memory_gb=0.5,
        preserve_dtypes={".*0.*": torch.float32}  # Keep first layer in FP32
    )
    
    quantized_model = quantizer.quantize(model)
    
    # Save model
    save_path = "/tmp/demo_quantized_model"
    print(f"Saving quantized model to: {save_path}")
    quantizer.save_quantized_model(quantized_model, save_path)
    
    # Verify save
    save_path_obj = Path(save_path)
    files = list(save_path_obj.iterdir()) if save_path_obj.exists() else []
    print(f"Saved files: {[f.name for f in files]}")
    
    # Load model
    print("\nLoading quantized model...")
    loaded_model = ModelQuantizer.load_quantized_model(
        load_path=save_path,
        model_class=torch.nn.Sequential
    )
    
    print("✓ Model loaded successfully")
    
    # Test inference
    test_input = torch.randn(4, 128)
    
    # Convert input to match model's expected dtype
    if hasattr(quantized_model, 'parameters'):
        first_param = next(quantized_model.parameters())
        test_input = test_input.to(first_param.dtype)
    
    with torch.inference_mode():
        original_output = quantized_model(test_input)
        loaded_output = loaded_model(test_input)
    
    diff = torch.abs(original_output - loaded_output).mean()
    print(f"Output difference: {diff:.8f}")
    
    # Cleanup
    import shutil
    if save_path_obj.exists():
        shutil.rmtree(save_path)
        print(f"Cleaned up: {save_path}")

def demo_config_file_usage():
    """Demonstrate using config files for dtype preservation"""
    print("\n" + "=" * 60)
    print("DEMO: Config File Usage for Dtype Preservation")
    print("=" * 60)
    
    from quantize.quantizer import ModelQuantizer
    import json
    import tempfile
    
    # Create a dtype preservation config
    dtype_config = {
        ".*embedding.*": "float32",
        ".*norm.*": "float16",
        ".*classifier.*": "bfloat16"
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(dtype_config, f, indent=2)
        config_file = f.name
    
    print(f"Created config file: {config_file}")
    
    # Create model
    model = torch.nn.Sequential(
        torch.nn.Embedding(100, 64),
        torch.nn.Flatten(),
        torch.nn.LayerNorm(6400),
        torch.nn.Linear(6400, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)  # classifier
    )
    
    # Use config file for quantization
    quantizer = ModelQuantizer(
        quant_method="basic_bf16",
        preserve_dtypes=config_file  # Pass file path
    )
    
    quantized_model = quantizer.quantize(model)
    
    print("\nModel dtypes after config-based quantization:")
    for name, param in quantized_model.named_parameters():
        print(f"  {name}: {param.dtype}")
    
    # Cleanup
    os.unlink(config_file)
    print(f"\nCleaned up config file: {config_file}")

def demo_memory_optimization():
    """Demonstrate memory optimization features"""
    print("\n" + "=" * 60)
    print("DEMO: Memory Optimization")
    print("=" * 60)
    
    from quantize.quantizer import ModelQuantizer
    
    # Create a large model
    class LargeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(1024, 1024) for _ in range(10)
            ])
            self.output = torch.nn.Linear(1024, 100)
        
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.output(x)
    
    model = LargeModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Large model parameters: {total_params:,}")
    
    # Test different memory targets
    memory_targets = [1.0, 5.0, 10.0]  # GB
    
    for target_gb in memory_targets:
        print(f"\n--- Target memory: {target_gb}GB ---")
        
        quantizer = ModelQuantizer(
            quant_method="basic_fp16",
            target_memory_gb=target_gb,
            auto_optimize=True
        )
        
        quantized_model = quantizer.quantize(model)
        
        if hasattr(quantized_model, '_quantization_info'):
            info = quantized_model._quantization_info
            print(f"Original: {info.get('original_memory_gb', 0):.3f}GB")
            print(f"Final: {info.get('final_memory_gb', 0):.3f}GB")
            print(f"Target: {target_gb}GB")
            
            reduction = ((info.get('original_memory_gb', 0) - info.get('final_memory_gb', 0)) 
                        / info.get('original_memory_gb', 1) * 100)
            print(f"Memory reduction: {reduction:.1f}%")

def demo_real_world_usage():
    """Demonstrate real-world usage patterns"""
    print("\n" + "=" * 60)
    print("DEMO: Real-World Usage Patterns")
    print("=" * 60)
    
    from quantize.quantizer import quantize_model
    
    # Scenario 1: Inference optimization
    print("Scenario 1: Optimizing for inference latency")
    
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 512),
        torch.nn.LayerNorm(512),
        torch.nn.Linear(512, 256)
    )
    
    # For inference, we want fast FP16
    inference_model = quantize_model(
        model, 
        quant_method="basic_fp16",
        preserve_dtypes={".*norm.*": torch.float32}  # Keep norms stable
    )
    
    print("✓ Optimized for inference with FP16 + stable norms")
    
    # Scenario 2: Memory-constrained environment
    print("\nScenario 2: Memory-constrained deployment")
    
    memory_constrained_model = quantize_model(
        model,
        quant_method="basic_fp16", 
        target_memory_gb=0.1,  # Very tight constraint
        auto_optimize=True
    )
    
    print("✓ Optimized for minimal memory usage")
    
    # Scenario 3: Mixed precision for training
    print("\nScenario 3: Mixed precision setup")
    
    mixed_precision_dtypes = {
        ".*0.*": torch.float32,    # First layer in FP32
        ".*4.*": torch.float32,    # Last layer in FP32  
        ".*norm.*": torch.float32  # Norms in FP32
    }
    
    mixed_model = quantize_model(
        model,
        quant_method="basic_fp16",
        preserve_dtypes=mixed_precision_dtypes
    )
    
    print("✓ Set up mixed precision configuration")
    
    print("\nFinal model dtypes:")
    for name, param in mixed_model.named_parameters():
        print(f"  {name}: {param.dtype}")

def main():
    """Run all demonstrations"""
    print("🚀 Quantization System Demonstration")
    print("Environment: apex conda environment")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        demo_basic_usage()
        demo_save_load()
        demo_config_file_usage()
        demo_memory_optimization()
        demo_real_world_usage()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("=" * 60)
        
        print("\nKey Features Demonstrated:")
        print("✓ Multiple quantization methods (basic + advanced)")
        print("✓ Dtype preservation with flexible configuration")
        print("✓ Save/load functionality with metadata")
        print("✓ Memory optimization and targeting")
        print("✓ Config file support for dtype preservation")
        print("✓ Real-world usage patterns")
        
        print("\nQuantization system is ready for production use! [[memory:4565082]]")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()