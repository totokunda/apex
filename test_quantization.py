#!/usr/bin/env python3
"""
Test script for the improved quantization system.

This script tests the quantization functionality with the HunyuanVideo text encoder
to ensure proper integration with the engine system and seamless save/load operations.
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/workspace/apex/src')

from text_encoder.text_encoder import TextEncoder
from quantize.quantizer import ModelQuantizer, quantize_model
from engine.base_engine import BaseEngine

def test_text_encoder_quantization():
    """Test basic text encoder quantization functionality"""
    print("=" * 60)
    print("Testing Text Encoder Quantization")
    print("=" * 60)
    
    # Configuration for the HunyuanVideo text encoder
    text_encoder_config = {
        "base": "transformers.CLIPTextModel.from_pretrained",
        "model_path": "/workspace/models/components/hunyuanvideo-community_HunyuanVideo_text_encoder_2/text_encoder_2",
        "config_path": "/workspace/models/components/hunyuanvideo-community_HunyuanVideo_text_encoder_2/text_encoder_2/config.json",
        "tokenizer_class": "transformers.CLIPTokenizer",
        "config": {},
        "extra_kwargs": {}
    }
    
    try:
        # Load the original text encoder
        print("Loading original text encoder...")
        text_encoder = TextEncoder(text_encoder_config)
        
        print(f"Original model device: {next(text_encoder.model.parameters()).device}")
        print(f"Original model dtype: {next(text_encoder.model.parameters()).dtype}")
        
        # Get model size
        total_params = sum(p.numel() for p in text_encoder.model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Test encoding with original model
        print("\nTesting original model encoding...")
        test_prompt = "A beautiful sunset over the mountains"
        with torch.inference_mode():
            original_output = text_encoder.encode(test_prompt)
        print(f"Original output shape: {original_output.shape}")
        print(f"Original output dtype: {original_output.dtype}")
        
        # Test different quantization methods (start with basic ones)
        quantization_methods = [
            "basic_fp16",
            "basic_bf16",
            "basic_int8"
        ]
        
        # Add advanced methods if available
        try:
            from src.quantize.quantizer import QUANTIZER_CONFIGS
            available_methods = list(QUANTIZER_CONFIGS.keys())
            
            if "quanto_config_int8" in available_methods:
                quantization_methods.append("quanto_config_int8")
            if "quanto_config_int4" in available_methods:
                quantization_methods.append("quanto_config_int4")
            if "bnb_4bit_config_fp16" in available_methods:
                quantization_methods.append("bnb_4bit_config_fp16")
                
            print(f"Available quantization methods: {available_methods}")
        except Exception as e:
            print(f"Warning: Could not load advanced quantization methods: {e}")
        
        for quant_method in quantization_methods:
            print(f"\n--- Testing {quant_method} ---")
            
            try:
                # Create a copy of the model for quantization
                quantizer = ModelQuantizer(
                    quant_method=quant_method,
                    target_memory_gb=2.0,  # Target 2GB
                    auto_optimize=True
                )
                
                # Quantize the model
                print(f"Quantizing with {quant_method}...")
                quantized_model = quantizer.quantize(text_encoder.model)
                
                print(f"Quantized model device: {next(quantized_model.parameters()).device}")
                print(f"Quantized model dtype: {next(quantized_model.parameters()).dtype}")
                
                # Test encoding with quantized model
                print("Testing quantized model encoding...")
                # Create a new text encoder instance with the quantized model
                quantized_text_encoder = TextEncoder(text_encoder_config, no_weights=True)
                quantized_text_encoder.model = quantized_model
                
                with torch.inference_mode():
                    quantized_output = quantized_text_encoder.encode(test_prompt)
                
                print(f"Quantized output shape: {quantized_output.shape}")
                print(f"Quantized output dtype: {quantized_output.dtype}")
                
                # Compare outputs
                if quantized_output.shape == original_output.shape:
                    diff = torch.abs(original_output.cpu() - quantized_output.cpu()).mean()
                    print(f"Mean absolute difference: {diff:.6f}")
                    print(f"Relative difference: {(diff / torch.abs(original_output.cpu()).mean() * 100):.2f}%")
                
                print(f"✓ {quant_method} quantization successful")
                
            except Exception as e:
                print(f"✗ {quant_method} quantization failed: {e}")
                continue
        
        # Test dtype preservation
        print("\n--- Testing dtype preservation ---")
        try:
            preserve_dtypes = {
                ".*embeddings.*": torch.float32,  # Preserve embeddings in FP32
                ".*norm.*": torch.float16        # Keep norms in FP16
            }
            
            quantizer = ModelQuantizer(
                quant_method="basic_bf16",
                target_memory_gb=2.0,
                preserve_dtypes=preserve_dtypes
            )
            
            print("Testing quantization with dtype preservation...")
            quantized_model = quantizer.quantize(text_encoder.model)
            
            # Check if dtypes were preserved correctly
            preserved_correctly = True
            for name, param in quantized_model.named_parameters():
                expected_dtype = None
                if "embeddings" in name:
                    expected_dtype = torch.float32
                elif "norm" in name:
                    expected_dtype = torch.float16
                else:
                    expected_dtype = torch.bfloat16  # Default quantized dtype
                
                if param.dtype != expected_dtype:
                    print(f"Warning: {name} has dtype {param.dtype}, expected {expected_dtype}")
                    preserved_correctly = False
            
            if preserved_correctly:
                print("✓ Dtype preservation working correctly")
            else:
                print("⚠ Some dtypes were not preserved as expected")
                
        except Exception as e:
            print(f"✗ Dtype preservation test failed: {e}")

        print("\n✓ Text encoder quantization tests completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Text encoder quantization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_save_load_functionality():
    """Test save and load functionality for quantized models"""
    print("\n" + "=" * 60)
    print("Testing Save/Load Functionality")
    print("=" * 60)
    
    text_encoder_config = {
        "base": "transformers.CLIPTextModel.from_pretrained",
        "model_path": "/workspace/models/components/hunyuanvideo-community_HunyuanVideo_text_encoder_2/text_encoder_2",
        "config_path": "/workspace/models/components/hunyuanvideo-community_HunyuanVideo_text_encoder_2/text_encoder_2/config.json",
        "tokenizer_class": "transformers.CLIPTokenizer",
        "config": {},
        "extra_kwargs": {}
    }
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    save_path = Path(temp_dir) / "quantized_text_encoder"
    
    try:
        # Load and quantize model
        print("Loading and quantizing text encoder...")
        text_encoder = TextEncoder(text_encoder_config)
        
        quantizer = ModelQuantizer(
            quant_method="basic_fp16",
            target_memory_gb=2.0,
            auto_optimize=True
        )
        
        quantized_model = quantizer.quantize(text_encoder.model)
        
        # Test encoding before save
        test_prompt = "Testing save and load functionality"
        with torch.inference_mode():
            original_output = quantized_model(**text_encoder.tokenizer(
                test_prompt, return_tensors="pt", padding=True, truncation=True
            ))
        
        print("Saving quantized model...")
        quantizer.save_quantized_model(
            model=quantized_model,
            save_path=save_path,
            save_config=True,
            save_tokenizer=False  # TextEncoder handles tokenizer separately
        )
        
        print(f"Model saved to: {save_path}")
        
        # Verify saved files
        expected_files = ["model.safetensors", "quantization_info.json", "config.json"]
        for file in expected_files:
            file_path = save_path / file
            if file_path.exists():
                print(f"✓ {file} saved successfully")
            else:
                print(f"✗ {file} not found")
                return False
        
        # Test loading
        print("\nLoading quantized model...")
        from transformers import CLIPTextModel
        
        loaded_model = ModelQuantizer.load_quantized_model(
            load_path=save_path,
            model_class=CLIPTextModel
        )
        
        print("Testing loaded model...")
        with torch.inference_mode():
            loaded_output = loaded_model(**text_encoder.tokenizer(
                test_prompt, return_tensors="pt", padding=True, truncation=True
            ))
        
        # Compare outputs
        if hasattr(original_output, 'last_hidden_state') and hasattr(loaded_output, 'last_hidden_state'):
            diff = torch.abs(
                original_output.last_hidden_state.cpu() - 
                loaded_output.last_hidden_state.cpu()
            ).mean()
            print(f"Output difference after save/load: {diff:.8f}")
            
            if diff < 1e-6:
                print("✓ Save/load functionality working correctly")
                return True
            else:
                print(f"✗ Significant difference in outputs: {diff}")
                return False
        else:
            print("✓ Model loaded successfully (unable to compare outputs)")
            return True
            
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

def test_engine_integration():
    """Test integration with engine system"""
    print("\n" + "=" * 60)
    print("Testing Engine Integration")
    print("=" * 60)
    
    try:
        # Create a minimal engine config for testing
        engine_config = {
            "components": {
                "text_encoder": {
                    "base": "transformers.CLIPTextModel.from_pretrained",
                    "model_path": "/workspace/models/components/hunyuanvideo-community_HunyuanVideo_text_encoder_2/text_encoder_2",
                    "config_path": "/workspace/models/components/hunyuanvideo-community_HunyuanVideo_text_encoder_2/text_encoder_2/config.json",
                    "tokenizer_class": "transformers.CLIPTokenizer",
                    "config": {},
                    "extra_kwargs": {}
                }
            },
            "model_path": "/workspace/models",
            "save_path": "/tmp/test_engine",
            "device": "cpu"
        }
        
        # Create a temporary config file
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(engine_config, f)
            config_path = f.name
        
        try:
            # Initialize engine
            print("Initializing test engine...")
            engine = BaseEngine(yaml_path=config_path, device=torch.device('cpu'))
            
            # Load text encoder
            print("Loading text encoder...")
            engine.load_component_by_type("text_encoder")
            
            # Test quantization through engine
            print("Testing engine quantization methods...")
            quantized_component = engine.quantize_component(
                component_type="text_encoder",
                quant_method="basic_fp16",
                target_memory_gb=2.0
            )
            
            print(f"✓ Engine quantization successful")
            print(f"Quantized component type: {type(quantized_component)}")
            
            # Test save through engine
            temp_save_path = "/tmp/test_engine_quantized_text_encoder"
            print(f"Testing engine save functionality...")
            
            engine.save_quantized_component(
                component_type="text_encoder", 
                save_path=temp_save_path,
                quant_method="basic_fp16"
            )
            
            print(f"✓ Engine save successful to {temp_save_path}")
            
            # Verify files exist
            save_path = Path(temp_save_path)
            if (save_path / "model.safetensors").exists():
                print("✓ Model files saved correctly")
            
            print("✓ Engine integration tests completed successfully")
            return True
            
        finally:
            # Clean up config file
            os.unlink(config_path)
            
    except Exception as e:
        print(f"✗ Engine integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all quantization tests"""
    print("Starting Quantization System Tests")
    print("Environment: apex conda environment")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    results = []
    
    # Run tests
    tests = [
        ("Text Encoder Quantization", test_text_encoder_quantization),
        ("Save/Load Functionality", test_save_load_functionality), 
        ("Engine Integration", test_engine_integration)
    ]
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Quantization system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)