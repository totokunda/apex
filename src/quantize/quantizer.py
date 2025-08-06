import torch
import os
import json
from enum import Enum
from typing import Literal, Dict, Union, Optional, Any
from pathlib import Path
import safetensors.torch as st

# Try to import quantization libraries with graceful fallbacks
try:
    from diffusers import (
        BitsAndBytesConfig,
        GGUFQuantizationConfig, 
        QuantoConfig,
        TorchAoConfig,
    )
    from diffusers.quantizers.auto import DiffusersAutoQuantizer
    from diffusers.quantizers.base import DiffusersQuantizer
    HAS_DIFFUSERS_QUANTIZERS = True
except ImportError:
    print("Warning: Diffusers quantizers not available. Using basic torch quantization.")
    HAS_DIFFUSERS_QUANTIZERS = False

try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

try:
    from accelerate.utils import (
        get_balanced_memory,
        infer_auto_device_map,
    )
    from accelerate import dispatch_model
    HAS_ACCELERATE = True
except ImportError:
    print("Warning: Accelerate not available. Using basic device management.")
    HAS_ACCELERATE = False

# Basic torch quantization fallback
class BasicQuantConfig:
    def __init__(self, dtype=torch.float16):
        self.dtype = dtype

# Initialize configurations based on available libraries
QUANTIZER_CONFIGS = {}

# Basic torch quantization (always available)
BASIC_FP16_CONFIG = BasicQuantConfig(torch.float16)
BASIC_BF16_CONFIG = BasicQuantConfig(torch.bfloat16)
BASIC_INT8_CONFIG = BasicQuantConfig(torch.int8)

QUANTIZER_CONFIGS.update({
    "basic_fp16": BASIC_FP16_CONFIG,
    "basic_bf16": BASIC_BF16_CONFIG,
    "basic_int8": BASIC_INT8_CONFIG,
})

# Add diffusers quantizers if available
if HAS_DIFFUSERS_QUANTIZERS:
    try:
        BNB_4BIT_CONFIG_FP16 = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        
        BNB_4BIT_CONFIG_BF16 = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        
        BNB_8BIT_CONFIG_FP16 = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        
        BNB_8BIT_CONFIG_BF16 = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )

        QUANTIZER_CONFIGS.update({
            "bnb_4bit_config_fp16": BNB_4BIT_CONFIG_FP16,
            "bnb_4bit_config_bf16": BNB_4BIT_CONFIG_BF16,
            "bnb_8bit_config_fp16": BNB_8BIT_CONFIG_FP16,
            "bnb_8bit_config_bf16": BNB_8BIT_CONFIG_BF16,
        })
        
        # Add GGUF configs if available
        if HAS_GGUF:
            GGUF_CONFIG_FP16 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.F16)
            GGUF_CONFIG_BF16 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.BF16)
            GGUF_CONFIG_Q8_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q8_K)
            GGUF_CONFIG_Q8_0 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q8_0)
            GGUF_CONFIG_Q8_1 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q8_1)
            GGUF_CONFIG_Q6_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q6_K)
            GGUF_CONFIG_Q5_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q5_K)
            GGUF_CONFIG_Q5_1 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q5_1)
            GGUF_CONFIG_Q5_0 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q5_0)
            GGUF_CONFIG_Q4_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q4_K)
            GGUF_CONFIG_Q4_0 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q4_0)
            GGUF_CONFIG_Q4_1 = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q4_1)
            GGUF_CONFIG_Q3_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q3_K)
            GGUF_CONFIG_Q2_K = GGUFQuantizationConfig(compute_dtype=gguf.GGMLQuantizationType.Q2_K)
            
            QUANTIZER_CONFIGS.update({
                "gguf_config_fp16": GGUF_CONFIG_FP16,
                "gguf_config_bf16": GGUF_CONFIG_BF16,
                "gguf_config_q8_k": GGUF_CONFIG_Q8_K,
                "gguf_config_q8_0": GGUF_CONFIG_Q8_0,
                "gguf_config_q8_1": GGUF_CONFIG_Q8_1,
                "gguf_config_q6_k": GGUF_CONFIG_Q6_K,
                "gguf_config_q5_k": GGUF_CONFIG_Q5_K,
                "gguf_config_q5_1": GGUF_CONFIG_Q5_1,
                "gguf_config_q5_0": GGUF_CONFIG_Q5_0,
                "gguf_config_q4_k": GGUF_CONFIG_Q4_K,
                "gguf_config_q4_0": GGUF_CONFIG_Q4_0,
                "gguf_config_q4_1": GGUF_CONFIG_Q4_1,
                "gguf_config_q3_k": GGUF_CONFIG_Q3_K,
                "gguf_config_q2_k": GGUF_CONFIG_Q2_K,
            })
        
        # Add Quanto configs if available
        try:
            QUANTO_CONFIG_FLOAT8 = QuantoConfig(weights_dtype="float8")
            QUANTO_CONFIG_INT8 = QuantoConfig(weights_dtype="int8")
            QUANTO_CONFIG_INT4 = QuantoConfig(weights_dtype="int4")
            QUANTO_CONFIG_INT2 = QuantoConfig(weights_dtype="int2")
            
            QUANTIZER_CONFIGS.update({
                "quanto_config_float8": QUANTO_CONFIG_FLOAT8,
                "quanto_config_int8": QUANTO_CONFIG_INT8,
                "quanto_config_int4": QUANTO_CONFIG_INT4,
                "quanto_config_int2": QUANTO_CONFIG_INT2,
            })
        except:
            pass
        
        # Add TorchAO configs if available
        try:
            TORCHAO_CONFIG_INT4WO = TorchAoConfig(quant_type="int4wo")
            TORCHAO_CONFIG_INT4DQ = TorchAoConfig(quant_type="int4dq")
            TORCHAO_CONFIG_INT8WO = TorchAoConfig(quant_type="int8wo")
            TORCHAO_CONFIG_INT8DQ = TorchAoConfig(quant_type="int8dq")
            TORCHAO_CONFIG_FLOAT8WO = TorchAoConfig(quant_type="float8wo")
            TORCHAO_CONFIG_FLOAT8WO_E5M2 = TorchAoConfig(quant_type="float8wo_e5m2")
            TORCHAO_CONFIG_FLOAT8WO_E4M3 = TorchAoConfig(quant_type="float8wo_e4m3")
            TORCHAO_CONFIG_FLOAT8DQ = TorchAoConfig(quant_type="float8dq")
            TORCHAO_CONFIG_FLOAT8DQ_E4M3 = TorchAoConfig(quant_type="float8dq_e4m3")
            TORCHAO_CONFIG_FLOAT8_E4M3_TENSOR = TorchAoConfig(quant_type="float8_e4m3_tensor")
            TORCHAO_CONFIG_FLOAT8_E4M3_ROW = TorchAoConfig(quant_type="float8_e4m3_row")
            TORCHAO_CONFIG_UINT1WO = TorchAoConfig(quant_type="uint1wo")
            TORCHAO_CONFIG_UINT2WO = TorchAoConfig(quant_type="uint2wo")
            TORCHAO_CONFIG_UINT3WO = TorchAoConfig(quant_type="uint3wo")
            TORCHAO_CONFIG_UINT4WO = TorchAoConfig(quant_type="uint4wo")
            TORCHAO_CONFIG_UINT5WO = TorchAoConfig(quant_type="uint5wo")
            TORCHAO_CONFIG_UINT6WO = TorchAoConfig(quant_type="uint6wo")
            TORCHAO_CONFIG_UINT7WO = TorchAoConfig(quant_type="uint7wo")
            
            QUANTIZER_CONFIGS.update({
                "torch_ao_config_int4wo": TORCHAO_CONFIG_INT4WO,
                "torch_ao_config_int4dq": TORCHAO_CONFIG_INT4DQ,
                "torch_ao_config_int8wo": TORCHAO_CONFIG_INT8WO,
                "torch_ao_config_int8dq": TORCHAO_CONFIG_INT8DQ,
                "torch_ao_config_float8wo": TORCHAO_CONFIG_FLOAT8WO,
                "torch_ao_config_float8wo_e5m2": TORCHAO_CONFIG_FLOAT8WO_E5M2,
                "torch_ao_config_float8wo_e4m3": TORCHAO_CONFIG_FLOAT8WO_E4M3,
                "torch_ao_config_float8dq": TORCHAO_CONFIG_FLOAT8DQ,
                "torch_ao_config_float8dq_e4m3": TORCHAO_CONFIG_FLOAT8DQ_E4M3,
                "torch_ao_config_float8_e4m3_tensor": TORCHAO_CONFIG_FLOAT8_E4M3_TENSOR,
                "torch_ao_config_float8_e4m3_row": TORCHAO_CONFIG_FLOAT8_E4M3_ROW,
                "torch_ao_config_uint1wo": TORCHAO_CONFIG_UINT1WO,
                "torch_ao_config_uint2wo": TORCHAO_CONFIG_UINT2WO,
                "torch_ao_config_uint3wo": TORCHAO_CONFIG_UINT3WO,
                "torch_ao_config_uint4wo": TORCHAO_CONFIG_UINT4WO,
                "torch_ao_config_uint5wo": TORCHAO_CONFIG_UINT5WO,
                "torch_ao_config_uint6wo": TORCHAO_CONFIG_UINT6WO,
                "torch_ao_config_uint7wo": TORCHAO_CONFIG_UINT7WO,
            })
        except:
            pass
            
    except Exception as e:
        print(f"Warning: Some quantization configs failed to initialize: {e}")

print(f"Available quantization methods: {list(QUANTIZER_CONFIGS.keys())}")

# Dynamic type based on available configurations  
quant_type = str  # Will be validated against QUANTIZER_CONFIGS at runtime


class ModelQuantizer:
    """
    Advanced quantization system for PyTorch models with save/load functionality.
    
    This class provides a comprehensive solution for model quantization that integrates
    seamlessly with the engine system, supporting various quantization backends and
    automatic memory management with dtype preservation capabilities.
    
    Parameters
    ----------
    quant_method : str
        Quantization method key from QUANTIZER_CONFIGS
    target_memory_gb : Optional[float]
        Target memory usage in GB. If specified, will choose optimal settings
    auto_optimize : bool
        Whether to automatically optimize quantization settings for performance
    preserve_dtypes : Optional[Union[Dict, str, Path]]
        Dtypes to preserve during quantization. Can be:
        - Dict: {"layer_name_pattern": dtype} mapping
        - str: Path to config file or single dtype name
        - Path: Path to config file
    """

    def __init__(
        self, 
        quant_method: quant_type,
        target_memory_gb: Optional[float] = None,
        auto_optimize: bool = True,
        preserve_dtypes: Optional[Union[Dict[str, torch.dtype], str, Path]] = None
    ):
        if quant_method not in QUANTIZER_CONFIGS:
            available_methods = list(QUANTIZER_CONFIGS.keys())
            raise ValueError(
                f"Unknown quantization method {quant_method!r}. Available methods: {available_methods}"
            )

        self.quant_method = quant_method
        self.target_memory_gb = target_memory_gb
        self.auto_optimize = auto_optimize
        self.config = QUANTIZER_CONFIGS[quant_method]
        self.preserve_dtypes = self._parse_preserve_dtypes(preserve_dtypes)
        
        # Initialize quantizer based on type
        if isinstance(self.config, BasicQuantConfig):
            self.quantizer = None  # Will use basic torch quantization
            self.is_basic_quantization = True
        else:
            if HAS_DIFFUSERS_QUANTIZERS:
                self.quantizer: DiffusersQuantizer = DiffusersAutoQuantizer.from_config(
                    self.config
                )
                try:
                    self.quantizer.validate_environment()
                except Exception as e:
                    print(f"Warning: Quantizer validation failed: {e}. Falling back to basic quantization.")
                    self.quantizer = None
                    self.is_basic_quantization = True
                else:
                    self.is_basic_quantization = False
            else:
                self.quantizer = None
                self.is_basic_quantization = True
        
        # Store quantization metadata
        self.quantization_info = {
            "method": quant_method,
            "config": self._serialize_config(self.config),
            "target_memory_gb": target_memory_gb,
            "auto_optimize": auto_optimize,
            "preserve_dtypes": self._serialize_preserve_dtypes(self.preserve_dtypes),
            "is_basic_quantization": self.is_basic_quantization
        }

    def _parse_preserve_dtypes(self, preserve_dtypes: Optional[Union[Dict, str, Path]]) -> Dict[str, torch.dtype]:
        """Parse dtype preservation configuration from various input formats"""
        if preserve_dtypes is None:
            return {}
        
        if isinstance(preserve_dtypes, dict):
            # Convert string dtypes to torch dtypes
            result = {}
            for pattern, dtype in preserve_dtypes.items():
                if isinstance(dtype, str):
                    dtype = getattr(torch, dtype, dtype)
                result[pattern] = dtype
            return result
        
        if isinstance(preserve_dtypes, (str, Path)):
            preserve_dtypes = Path(preserve_dtypes)
            
            # Check if it's a file path
            if preserve_dtypes.exists() and preserve_dtypes.suffix in ['.json', '.yaml', '.yml']:
                if preserve_dtypes.suffix == '.json':
                    with open(preserve_dtypes, 'r') as f:
                        config = json.load(f)
                else:
                    import yaml
                    with open(preserve_dtypes, 'r') as f:
                        config = yaml.safe_load(f)
                
                return self._parse_preserve_dtypes(config)
            
            # Single dtype name
            dtype_str = str(preserve_dtypes)
            if hasattr(torch, dtype_str):
                return {".*": getattr(torch, dtype_str)}
            else:
                print(f"Warning: Unknown dtype {dtype_str}, ignoring")
                return {}
        
        return {}

    def _serialize_preserve_dtypes(self, preserve_dtypes: Dict[str, torch.dtype]) -> Dict[str, str]:
        """Serialize preserve_dtypes for JSON storage"""
        return {pattern: str(dtype) for pattern, dtype in preserve_dtypes.items()}

    def _should_preserve_layer(self, layer_name: str) -> Optional[torch.dtype]:
        """Check if a layer should preserve its dtype based on patterns"""
        import re
        for pattern, dtype in self.preserve_dtypes.items():
            if re.search(pattern, layer_name):
                return dtype
        return None

    def _serialize_config(self, config) -> Dict[str, Any]:
        """Serialize quantization config for saving"""
        if hasattr(config, '__dict__'):
            return {k: str(v) for k, v in config.__dict__.items()}
        return {"type": str(type(config))}

    def _estimate_memory_usage(self, model: torch.nn.Module) -> float:
        """Estimate model memory usage in GB"""
        total_params = sum(p.numel() for p in model.parameters())
        # Rough estimate: 4 bytes per parameter for fp32
        memory_gb = (total_params * 4) / (1024**3)
        return memory_gb

    def _optimize_max_memory(
        self, 
        model: torch.nn.Module, 
        target_dtype: torch.dtype
    ) -> Dict[Union[int, str], Union[int, str]]:
        """Optimize memory allocation based on target memory"""
        if self.target_memory_gb is None:
            return get_balanced_memory(model, dtype=target_dtype, low_zero=False)
        
        # Calculate target memory per device
        available_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        target_bytes = int(self.target_memory_gb * (1024**3))
        
        if available_devices > 1:
            per_device_memory = target_bytes // available_devices
            max_memory = {i: per_device_memory for i in range(available_devices)}
        else:
            max_memory = {0: target_bytes}
            
        return max_memory

    def _basic_quantize(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply basic torch quantization with dtype preservation"""
        # Store original dtypes for layers that should be preserved
        preserved_layers = {}
        for name, param in model.named_parameters():
            preserve_dtype = self._should_preserve_layer(name)
            if preserve_dtype:
                preserved_layers[name] = preserve_dtype
        
        # Apply basic quantization
        target_dtype = self.config.dtype
        print(f"Applying basic quantization to dtype: {target_dtype}")
        
        # Convert model to target dtype, preserving specified layers
        model = model.to(target_dtype)  # Convert entire model first
        
        # Then apply specific dtype preservation
        for name, param in model.named_parameters():
            if name in preserved_layers:
                param.data = param.data.to(preserved_layers[name])
        
        return model

    def quantize(
        self,
        model: torch.nn.Module,
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    ) -> torch.nn.Module:
        """
        Quantize model with optimal memory allocation and performance settings.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to quantize
        max_memory : Optional[Dict]
            Memory constraints per device
            
        Returns
        -------
        torch.nn.Module
            Quantized model
        """
        original_memory = self._estimate_memory_usage(model)
        print(f"Original model memory: {original_memory:.2f} GB")
        
        if self.is_basic_quantization or self.quantizer is None:
            # Use basic quantization
            print(f"Using basic quantization with method: {self.quant_method}")
            quantized_model = self._basic_quantize(model)
            device_map = None
            max_memory_used = None
        else:
            # Use advanced diffusers quantization
            print(f"Using advanced quantization with method: {self.quant_method}")
            
            # 1) Determine target dtype
            target_dtype = self.quantizer.update_torch_dtype(
                getattr(model, "dtype", torch.float32)
            )

            # 2) Preserve specified dtypes before preprocessing
            preserved_params = {}
            for name, param in model.named_parameters():
                preserve_dtype = self._should_preserve_layer(name)
                if preserve_dtype:
                    preserved_params[name] = param.data.clone().to(preserve_dtype)

            # 3) Preprocess model for quantization
            self.quantizer.preprocess_model(model)

            # 4) Restore preserved dtypes
            for name, preserved_data in preserved_params.items():
                if hasattr(model, name.replace('.', '_')):
                    param = model.get_parameter(name)
                    param.data = preserved_data

            # 5) Optimize memory allocation
            if max_memory is None:
                if self.auto_optimize and self.target_memory_gb:
                    max_memory = self._optimize_max_memory(model, target_dtype)
                elif HAS_ACCELERATE:
                    max_memory = get_balanced_memory(model, dtype=target_dtype, low_zero=False)
                else:
                    max_memory = None
            
            if max_memory and self.quantizer:
                max_memory = self.quantizer.adjust_max_memory(max_memory)

            # 6) Create device map if accelerate is available
            device_map = None
            if HAS_ACCELERATE and max_memory:
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=getattr(self.quantizer, 'modules_to_not_convert', []),
                    dtype=target_dtype,
                )
                if self.quantizer:
                    device_map = self.quantizer.update_device_map(device_map)

                # 7) Dispatch model to devices
                model = dispatch_model(model, device_map=device_map)

            # 8) Post-process quantized model
            quantized_model = self.quantizer.postprocess_model(model)
            max_memory_used = max_memory
        
        # Store quantization metadata on model
        final_memory = self._estimate_memory_usage(quantized_model)
        print(f"Quantized model memory: {final_memory:.2f} GB")
        
        if original_memory > 0:
            reduction = ((original_memory - final_memory) / original_memory * 100)
            print(f"Memory reduction: {reduction:.1f}%")
        else:
            print("Memory reduction: N/A (original memory too small to measure)")
        
        quantized_model._quantization_info = self.quantization_info.copy()
        quantized_model._quantization_info.update({
            "original_memory_gb": original_memory,
            "final_memory_gb": final_memory,
            "device_map": device_map,
            "max_memory": max_memory_used
        })

        return quantized_model

    def save_quantized_model(
        self, 
        model: torch.nn.Module, 
        save_path: Union[str, Path],
        save_config: bool = True,
        save_tokenizer: bool = True
    ) -> None:
        """
        Save quantized model with metadata for seamless loading.
        
        Parameters
        ----------
        model : torch.nn.Module
            Quantized model to save
        save_path : Union[str, Path]
            Directory to save the model
        save_config : bool
            Whether to save model config
        save_tokenizer : bool
            Whether to save tokenizer if present
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save quantization metadata
        quant_info = getattr(model, '_quantization_info', self.quantization_info)
        with open(save_path / "quantization_info.json", "w") as f:
            json.dump(quant_info, f, indent=2, default=str)
        
        # Save model state dict using safetensors
        state_dict = model.state_dict()
        st.save_file(state_dict, save_path / "model.safetensors")
        
        # Save model config if available and requested
        if save_config and hasattr(model, 'config'):
            if hasattr(model.config, 'to_json_file'):
                model.config.to_json_file(save_path / "config.json")
            elif isinstance(model.config, dict):
                with open(save_path / "config.json", "w") as f:
                    json.dump(model.config, f, indent=2, default=str)
        
        # Save tokenizer if present and requested
        if save_tokenizer and hasattr(model, 'tokenizer'):
            tokenizer_path = save_path / "tokenizer"
            tokenizer_path.mkdir(exist_ok=True)
            model.tokenizer.save_pretrained(str(tokenizer_path))

    @classmethod
    def load_quantized_model(
        cls,
        load_path: Union[str, Path],
        model_class: Optional[type] = None,
        **model_kwargs
    ) -> torch.nn.Module:
        """
        Load a quantized model with automatic quantization restoration.
        
        Parameters
        ----------
        load_path : Union[str, Path]
            Path to the saved quantized model
        model_class : Optional[type]
            Model class to instantiate. If None, will try to infer
        **model_kwargs
            Additional arguments for model instantiation
            
        Returns
        -------
        torch.nn.Module
            Loaded quantized model
        """
        load_path = Path(load_path)
        
        # Load quantization metadata
        with open(load_path / "quantization_info.json", "r") as f:
            quant_info = json.load(f)
        
        # Create quantizer instance
        quantizer = cls(
            quant_method=quant_info["method"],
            target_memory_gb=quant_info.get("target_memory_gb"),
            auto_optimize=quant_info.get("auto_optimize", True)
        )
        
        # Load model config if available
        config_path = load_path / "config.json"
        config = None
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        
        # Instantiate model
        if model_class is None:
            raise ValueError("model_class must be provided for loading")
            
        if config:
            model = model_class(config, **model_kwargs)
        else:
            model = model_class(**model_kwargs)
        
        # Load state dict
        state_dict = st.load_file(load_path / "model.safetensors")
        model.load_state_dict(state_dict, strict=False)
        
        # Apply quantization
        quantized_model = quantizer.quantize(
            model, 
            max_memory=quant_info.get("max_memory")
        )
        
        return quantized_model


def quantize_model(
    model: torch.nn.Module,
    quant_method: quant_type = "basic_fp16",
    target_memory_gb: Optional[float] = None,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    auto_optimize: bool = True,
    preserve_dtypes: Optional[Union[Dict[str, torch.dtype], str, Path]] = None
) -> torch.nn.Module:
    """
    Convenience function for quantizing models with dtype preservation.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to quantize
    quant_method : str
        Quantization method to use (default: basic_fp16)
    target_memory_gb : Optional[float]
        Target memory usage in GB
    max_memory : Optional[Dict]
        Manual memory constraints
    auto_optimize : bool
        Enable automatic optimization
    preserve_dtypes : Optional[Union[Dict, str, Path]]
        Dtypes to preserve during quantization. Can be:
        - Dict: {"layer_name_pattern": dtype} mapping
        - str: Path to config file or single dtype name
        - Path: Path to config file
        
    Returns
    -------
    torch.nn.Module
        Quantized model
    """
    quantizer = ModelQuantizer(
        quant_method=quant_method,
        target_memory_gb=target_memory_gb,
        auto_optimize=auto_optimize,
        preserve_dtypes=preserve_dtypes
    )
    return quantizer.quantize(model, max_memory=max_memory)
