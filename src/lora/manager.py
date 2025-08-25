import os
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from src.converters.convert import get_transformer_converter_by_model_name, strip_common_prefix
import torch
from loguru import logger
from diffusers.loaders import PeftAdapterMixin
from safetensors.torch import load_file
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import DEFAULT_LORA_SAVE_PATH


@dataclass
class LoraItem:
    # A single LoRA entry that may consist of 1+ files (.safetensors/.bin and optional .json config)
    source: str
    local_paths: List[str]
    scale: float = 1.0
    name: Optional[str] = None


class LoraManager(DownloadMixin):
    def __init__(self, save_dir: str = DEFAULT_LORA_SAVE_PATH) -> None:
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # cache source->LoraItem to avoid repeated downloads
        self._cache: Dict[str, LoraItem] = {}

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def resolve(self, source: str, prefer_name: Optional[str] = None) -> LoraItem:
        """
        Resolve and download a LoRA from any supported source:
        - HuggingFace repo or file path
        - CivitAI model or file URL (e.g., https://civitai.com/api/download/models/<id>)
        - Generic direct URL
        - Local path (dir or file)
        Returns a LoraItem with one or more local file paths.
        """
        if source in self._cache:
            return self._cache[source]

        # Local path
        if os.path.exists(source):
            paths = self._collect_lora_files(source)
            item = LoraItem(
                source=source,
                local_paths=paths,
                name=prefer_name or os.path.basename(source),
            )
            self._cache[source] = item
            return item

        # CivitAI integration: support plain download links and model/file ids
        if self._is_civitai(source):
            if source.startswith("civitai:"):
                local_path = self._download_from_civitai_spec(source)
            else:
                local_path = self._download_from_url(source, self.save_dir)
            paths = self._collect_lora_files(local_path)
            item = LoraItem(
                source=source,
                local_paths=paths,
                name=prefer_name or self._infer_name(source, local_path),
            )
            self._cache[source] = item
            return item

        # HuggingFace: repo id or specific file
        if self._is_huggingface_repo(source) or self._looks_like_hf_file(source):
            local_path = self._download(source, self.save_dir)
            paths = self._collect_lora_files(local_path)
            item = LoraItem(
                source=source,
                local_paths=paths,
                name=prefer_name or self._infer_name(source, local_path),
            )
            self._cache[source] = item
            return item

        # Fallback: generic URL
        if self._is_url(source):
            local_path = self._download_from_url(source, self.save_dir)
            paths = self._collect_lora_files(local_path)
            item = LoraItem(
                source=source,
                local_paths=paths,
                name=prefer_name or self._infer_name(source, local_path),
            )
            self._cache[source] = item
            return item

        # As a last resort, try to treat as local path string
        paths = self._collect_lora_files(source)
        item = LoraItem(
            source=source,
            local_paths=paths,
            name=prefer_name or os.path.basename(source),
        )
        self._cache[source] = item
        return item

    def _looks_like_hf_file(self, text: str) -> bool:
        # matches org/repo/…/file.safetensors or similar
        return bool(re.match(r"^[\w\-]+/[\w\-]+/.+\.[A-Za-z0-9]+$", text))

    def _is_civitai(self, url: str) -> bool:
        return "civitai.com" in url

    def _infer_name(self, source: str, local_path: str) -> str:
        if os.path.isdir(local_path):
            return os.path.basename(local_path.rstrip("/"))
        return os.path.splitext(os.path.basename(local_path))[0]

    def _collect_lora_files(self, path: str) -> List[str]:
        """Return a list of LoRA weight files for a given path (dir or file)."""
        if not path:
            return []
        if os.path.isdir(path):
            files = []
            for root, _dirs, fnames in os.walk(path):
                for fn in fnames:
                    if self._is_lora_file(fn):
                        files.append(os.path.join(root, fn))
            return sorted(files)
        if os.path.isfile(path) and self._is_lora_file(path):
            return [path]
        return []

    def _is_lora_file(self, filename: str) -> bool:
        lower = filename.lower()
        return lower.endswith((".safetensors", ".bin", ".pt", ".pth"))

    def load_into(
        self,
        model: Union[torch.nn.Module, PeftAdapterMixin],
        loras: List[Union[str, LoraItem, Tuple[Union[str, LoraItem], float]]],
        adapter_names: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
    ) -> List[LoraItem]:
        """
        Load multiple LoRAs into a PEFT-enabled model. Supports per-adapter scaling.
        - loras can be strings (sources), LoraItem, or tuples of (source|LoraItem, scale)
        - adapter_names optionally overrides adapter naming
        - scales optionally overrides per-adapter scale values
        Returns resolved LoraItem objects in load order.
        """
  
        if not hasattr(model, "set_adapters"):
            raise ValueError(
                "Model doesn't support PEFT/LoRA. Ensure transformer inherits PeftAdapterMixin."
            )

        # verify the state dict is correct or convert it to the correct format

        resolved: List[LoraItem] = []
        final_names: List[str] = []
        final_scales: List[float] = []
        for idx, entry in enumerate(loras):
            scale: float = 1.0
            if isinstance(entry, tuple):
                src_or_item, scale = entry
            else:
                src_or_item = entry

            if isinstance(src_or_item, LoraItem):
                item = src_or_item
            else:
                item = self.resolve(str(src_or_item))
            
            # override from global scales list if provided
            if scales is not None and idx < len(scales) and scales[idx] is not None:
                scale = float(scales[idx])
            item.scale = float(scale)
            resolved.append(item)

        for i, item in enumerate(resolved):
            adapter_name = (
                adapter_names[i]
                if adapter_names and i < len(adapter_names) and adapter_names[i]
                else item.name or f"lora_{self._hash(item.source)}"
            )
            final_names.append(adapter_name)
            final_scales.append(item.scale)
            # diffusers supports str or dict mapping for multiple files; we load one-by-one if multiple
            for local_path in item.local_paths:
                local_path_state_dict, converted = self.maybe_convert_state_dict(
                    local_path, model.config._class_name
                )
                
                if converted:
                    local_path_state_dict = strip_common_prefix(local_path_state_dict, model.state_dict())
                
                model.load_lora_adapter(
                    local_path_state_dict, adapter_name=adapter_name
                )
                
                logger.info(f"Loaded LoRA {adapter_name} from {local_path}")

        # Activate all adapters with their weights in one call
        try:
            model.set_adapters(final_names, weights=final_scales)
        except Exception as e:
            logger.warning(
                f"Failed to activate adapters {final_names} with scales {final_scales}: {e}"
            )
        return resolved

    def maybe_convert_state_dict(self, local_path: str, model_name: str) -> str:
        state_dict = self.load_file(local_path)
        converter = get_transformer_converter_by_model_name(model_name)
        converted = False
        if converter is not None:
            converter.convert(state_dict)
            converted = True
        return state_dict, converted

    def _download_from_civitai_spec(self, spec: str) -> str:
        """
        Support strings like:
          - "civitai:MODEL_ID" -> fetch model metadata, pick first LoRA SafeTensor file
          - "civitai-file:FILE_ID" -> download that specific file id
        Returns a local path (file) to the downloaded artifact.
        """
        import requests

        api_key = os.getenv("CIVITAI_API_KEY", None)
        headers = {"User-Agent": "apex-lora-manager/1.0"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def download_file_id(file_id: Union[int, str]) -> str:
            url = f"https://civitai.com/api/download/models/{file_id}"
            return self._download_from_url(url, self.save_dir)

        if spec.startswith("civitai-file:"):
            file_id = spec.split(":", 1)[1]
            return download_file_id(file_id)

        # civitai:MODEL_ID
        model_id = spec.split(":", 1)[1]
        meta_url = f"https://civitai.com/api/v1/models/{model_id}"
        resp = requests.get(meta_url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # Find a file that looks like a LoRA in SafeTensor format
        for version in data.get("modelVersions", []):
            for f in version.get("files", []):
                fname = f.get("name") or ""
                if fname.lower().endswith((".safetensors", ".pt", ".bin")):
                    file_id = f.get("id")
                    if file_id is not None:
                        return download_file_id(file_id)
        # Fallback to first file id
        for version in data.get("modelVersions", []):
            files = version.get("files", [])
            if files:
                file_id = files[0].get("id")
                if file_id is not None:
                    return download_file_id(file_id)
        raise RuntimeError(f"No downloadable files found for CivitAI model {model_id}")

    def load_file(self, local_path: str) -> str:
        if local_path.endswith(".safetensors"):
            return load_file(local_path)
        else:
            torch.load(local_path)
