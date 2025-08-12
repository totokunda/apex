import gguf
from src.quantize.text_encoder import (
    get_model_architecture,
    ModelBase,
    ModelType,
)
from src.quantize.base import BaseQuantizer
from tempfile import NamedTemporaryFile
import os
import shutil
import subprocess
from loguru import logger
from enum import Enum
from src.utils.defaults import DEFAULT_COMPONENTS_PATH
from pathlib import Path
import torch

class QuantizationType(Enum):
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K_M = "Q4_K_M"
    Q4_K_S = "Q4_K_S"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q8_0 = "Q8_0"
    Q2_K = "Q2_K"
    Q2_K_S = "Q2_K_S"
    Q3_K = "Q3_K"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q5_K_M = "Q5_K_M"
    Q5_K_S = "Q5_K_S"
    Q6_K = "Q6_K"
    BF16 = "BF16"
    F16 = "F16"
    F32 = "F32"
    

FTYPE_MAP: dict[str, gguf.LlamaFileType] = {
    "f32": gguf.LlamaFileType.ALL_F32,
    "f16": gguf.LlamaFileType.MOSTLY_F16,
    "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
    "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
    "auto": gguf.LlamaFileType.GUESSED,
}
    
class TextEncoderQuantizer(BaseQuantizer):
    model_path: str = None
    tokenizer_path: str = None
    model_type: ModelType = None
    kwargs: dict = {}
    quantization: QuantizationType = QuantizationType.F16
    file_type: str = "f16"

    def __init__(
        self,
        output_path: str,
        model_path: str = None,
        file_type: str = "f16",
        tokenizer_path: str = None,
        model_type: ModelType = ModelType.TEXT,
        quantization: QuantizationType | str = QuantizationType.F16,
        **kwargs
    ):
        self.output_path = output_path
        self.file_type = file_type
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.model_type = model_type
        self.kwargs = kwargs
        
        if isinstance(quantization, str):
            quantization = QuantizationType(quantization)
        
        self.quantization = quantization

        self.model_path = self._download(self.model_path, DEFAULT_COMPONENTS_PATH)
        self.tokenizer_path = self._download(self.tokenizer_path, DEFAULT_COMPONENTS_PATH)  
        
    def _fix_output_path(self, output_path: str, quantization_str: str):
        file_ending = f".gguf"
        if output_path.endswith(file_ending) and file_ending.endswith(f".{quantization_str}.gguf"):
            return output_path
        elif output_path.endswith(file_ending):
            return output_path.replace(file_ending, f".{quantization_str}.gguf")
        else:
            return output_path + file_ending

    def _llama_cpp_quant(
        self,
        fp16_quant_path: str,
        output_path: str,
        quantization: QuantizationType = QuantizationType.F16,
        llama_quantize_path: str = "llama-quantize",
    ):
        logger.info(f"Quantizing model with quantization type {quantization}")

        # Resolve llama-quantize binary location with robust fallbacks:
        # 1) explicit arg if it exists
        # 2) env var APEX_LLAMA_QUANTIZE_BIN
        # 3) submodule build path: thirdparty/llama.cpp/build/bin/llama-quantize
        # 4) repo bin path: bin/llama-quantize
        # 5) PATH lookup
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates: list[str] = []

        if llama_quantize_path and os.path.isabs(llama_quantize_path):
            candidates.append(llama_quantize_path)

        env_bin = os.getenv("APEX_LLAMA_QUANTIZE_BIN")
        if env_bin:
            candidates.append(env_bin)

        submodule_bin = os.path.join(repo_root, "thirdparty", "llama.cpp", "build", "bin", "llama-quantize")
        candidates.append(submodule_bin)

        repo_bin = os.path.join(repo_root, "bin", "llama-quantize")
        candidates.append(repo_bin)

        path_bin = shutil.which("llama-quantize")
        if path_bin:
            candidates.append(path_bin)

        resolved = next((p for p in candidates if p and os.path.exists(p)), None)
        if not resolved:
            search_list = "\n".join(candidates)
            raise FileNotFoundError(
                "Could not locate 'llama-quantize'. Tried:\n" + search_list +
                "\nBuild the submodule and/or set APEX_LLAMA_QUANTIZE_BIN to the binary."
            )

        quantization_str = quantization.value
        output_path = self._fix_output_path(output_path, quantization_str)

        cmd = [resolved, fp16_quant_path, output_path, quantization_str]
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if proc.stdout:
                logger.debug(proc.stdout)
            if proc.stderr:
                logger.debug(proc.stderr)
        except subprocess.CalledProcessError as exc:
            logger.error(exc.stdout or "")
            logger.error(exc.stderr or "")
            raise RuntimeError(
                f"Failed to quantize model with exit code {exc.returncode}. Command: {' '.join(cmd)}"
            ) from exc

        return output_path

    def _requires_llama_cpp_quant(self, quantization: QuantizationType):
        print(f"quantization: {quantization}")
        return quantization not in [
            QuantizationType.F16,
            QuantizationType.F32
        ]

    @torch.inference_mode()
    def quantize(
        self,
        output_path: str = None,
        quantization: QuantizationType | str = None,
        file_type: str = None,
        split_max_tensors: int = 0,
        split_max_size: int = 0,
        dry_run: bool = False,
        small_first_shard: bool = False,
        remote_hf_model_id: str = None,
        hf_repo_id: str = None,
        use_temp_file: bool = False,
        bigendian: bool = False,
        llama_quantize_path: str = "llama-quantize",
        **kwargs
    ):
        if isinstance(quantization, str):
            quantization = QuantizationType(quantization)

        if file_type is None:
            file_type = self.file_type  

        if quantization is None:
            quantization = self.quantization

        if output_path is None:
            output_path = self.output_path
            
        requires_llama_cpp_quant = self._requires_llama_cpp_quant(quantization)
            
        hparams = ModelBase.load_hparams(Path(self.model_path), False)
        model_architecture = get_model_architecture(hparams, self.model_type)
        
        model_class = ModelBase.from_model_architecture(
            model_architecture, model_type=self.model_type
        )

        if requires_llama_cpp_quant:
            with NamedTemporaryFile(delete=True) as temp_file:
                quant_path = temp_file.name
        else:
            temp_file = None
            quant_path = self._fix_output_path(output_path, quantization.value)

        model_instance = model_class(
            Path(self.model_path),
            Path(self.tokenizer_path) if self.tokenizer_path is not None else None,
            FTYPE_MAP[file_type],
            Path(quant_path),
            is_big_endian=bigendian,
            use_temp_file=use_temp_file,
            eager=self.kwargs.get("no_lazy", False),
            metadata_override=self.kwargs.get("metadata", None),
            model_name=self.kwargs.get("model_name", None),
            split_max_tensors=split_max_tensors,
            split_max_size=split_max_size,
            dry_run=dry_run,
            small_first_shard=small_first_shard,
            remote_hf_model_id=hf_repo_id,
        )

        model_instance.write()

        if requires_llama_cpp_quant:    
            save_path = self._llama_cpp_quant(quant_path, output_path, quantization, llama_quantize_path)
        else:
            save_path = quant_path
        
        if temp_file is not None:
            temp_file.close()
            
        return save_path
        
