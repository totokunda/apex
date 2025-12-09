import os
import re
import hashlib
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
from src.converters.convert import get_transformer_converter_by_model_name, strip_common_prefix
import torch
from loguru import logger
from diffusers.loaders import PeftAdapterMixin
from safetensors.torch import load_file
from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import DEFAULT_LORA_SAVE_PATH
from urllib.parse import urlencode
try: 
    from nunchaku.lora.flux.compose import compose_lora
except ImportError:
    compose_lora = None

@dataclass
class LoraItem:
    # A single LoRA entry that may consist of 1+ files (.safetensors/.bin and optional .json config)
    source: str
    local_paths: List[str]
    scale: float = 1.0
    name: Optional[str] = None
    component: Optional[str] = None

class LoraManager(DownloadMixin):
    def __init__(self, save_dir: str = DEFAULT_LORA_SAVE_PATH) -> None:
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # cache source->LoraItem to avoid repeated downloads
        self._cache: Dict[str, LoraItem] = {}

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _replace_up_down_to_AB_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Replace all keys that end with _up or _down with _A or _B
        # and make sure the LoRA rank is on the **second** dimension for lora_B
        # to match how diffusers / PEFT infer rank (see `peft.py` around rank[f"^{key}"] = val.shape[1]).
        new_state_dict: Dict[str, torch.Tensor] = {}

        # First, normalize key names `.lora_up` / `.lora_down` -> `.lora_A` / `.lora_B`
        for key, value in state_dict.items():
            if ".lora_up" in key:
                new_key = key.replace(".lora_up", ".lora_A")
            elif ".lora_down" in key:
                new_key = key.replace(".lora_down", ".lora_B")
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Heuristic: some LoRAs store matrices with rank on the **first** dim for `lora_B`
        # (shape `[r, out]`) instead of the second dim (`[out, r]`).
        # Since PEFT determines rank from `val.shape[1]` for `lora_B`, we transpose both
        # `lora_A` and `lora_B` for a pair when we detect that the smaller dimension of
        # `lora_B` is the first one.

        lora_pairs: Dict[str, Dict[str, str]] = {}
        for key in list(new_state_dict.keys()):
            if ".lora_A" in key:
                base = key.split(".lora_A", 1)[0]
                lora_pairs.setdefault(base, {})["A"] = key
            elif ".lora_B" in key:
                base = key.split(".lora_B", 1)[0]
                lora_pairs.setdefault(base, {})["B"] = key

        for base, pair in lora_pairs.items():
            a_key = pair.get("A")
            b_key = pair.get("B")
            if a_key is None or b_key is None:
                continue

            a_val = new_state_dict.get(a_key)
            b_val = new_state_dict.get(b_key)
            if not isinstance(a_val, torch.Tensor) or not isinstance(b_val, torch.Tensor):
                continue

            # Only handle simple linear-style LoRA weights
            if a_val.ndim != 2 or b_val.ndim != 2:
                continue

            b0, b1 = b_val.shape
            # If the smaller dimension (candidate rank) is on dim 0 for lora_B,
            # transpose both A and B so that rank becomes the second dim for lora_B.
            

        del state_dict
        return new_state_dict


    def resolve(
        self,
        source: str,
        prefer_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None,
    ) -> LoraItem:

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

        # Local path – nothing to download, so no progress_callback needed
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
            # Model-id style spec, delegate to helper with progress
            if source.startswith("civitai:"):
                local_path = self._download_from_civitai_spec(
                        source,
                        progress_callback=progress_callback,
                )
            else:
                # Direct CivitAI download URL
                local_path = self._download(
                    source,
                    self.save_dir,
                    progress_callback=progress_callback,
                )
            paths = self._collect_lora_files(local_path)
            item = LoraItem(
                source=source,
                local_paths=paths,
                name=prefer_name or self._infer_name(source, local_path),
            )
            self._cache[source] = item
            return item

        # Generic URL / HF / cloud / path handled by DownloadMixin with progress
        local_path = self._download(
            source,
            self.save_dir,
            progress_callback=progress_callback,
        )
        paths = self._collect_lora_files(local_path)
        item = LoraItem(
            source=source,
            local_paths=paths,
            name=prefer_name or self._infer_name(source, local_path),
        )
        self._cache[source] = item
        return item

    def _looks_like_hf_file(self, text: str) -> bool:
        # matches org/repo/…/file.safetensors or similar
        return bool(re.match(r"^[\w\-]+/[\w\-]+/.+\.[A-Za-z0-9]+$", text))

    def _is_civitai(self, url: str) -> bool:
        return "civitai.com" in url or url.startswith("civitai:")

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
                    path = os.path.join(root, fn)
                    if self._is_lora_file(path):
                        files.append(path)
            return sorted(files)
        if os.path.isfile(path) and self._is_lora_file(path):
            return [path]
        return []

    def _is_lora_file(self, filename: str) -> bool:
        """
        Heuristic check for whether a file is *likely* a LoRA weights file.
        Designed to be very cheap in CPU usage:
        - Only checks the extension.
        - Optionally sniffs a tiny header to rule out obvious HTML/text error pages.
        """
        try:
            with open(filename, "rb") as f:
                head = f.read(2048)
        except OSError:
            return False

        if not head:
            return False

        # If this decodes cleanly to mostly text and looks like HTML or an error,
        # treat it as "not a LoRA file". This keeps CPU usage tiny while avoiding
        # obviously-wrong files.
        text_sample = head.decode("utf-8", errors="ignore").lstrip().lower()
        html_markers = ("<!doctype html", "<html", "<head", "<body")
        if text_sample.startswith(html_markers):
            return False

        # Common text error indicators near the top of the file.
        for marker in ("error", "not found", "access denied"):
            if marker in text_sample[:512]:
                return False

        return True
    
    def _clean_adapter_name(self, name: str) -> str:
        if len(name) > 64:
            name = name[:64]
        if "." in name or "/" in name:
            name = name.replace(".", "_").replace("/", "_")
        return name

    def load_into(
        self,
        model: Union[torch.nn.Module, PeftAdapterMixin],
        loras: List[Union[str, LoraItem, Tuple[Union[str, LoraItem], float]]],
        adapter_names: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
        replace_keys: bool = True,
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
        loaded_resolved: List[LoraItem] = []
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
            
        
        if hasattr(model, "update_lora_params") and compose_lora is not None:
            composed_lora = []
            for i, item in enumerate(resolved):
                for local_path in item.local_paths:
                    composed_lora.append((local_path, item.scale))
            composed_lora = compose_lora(composed_lora)
            model.update_lora_params(composed_lora)
        else:
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
                    class_name = getattr(model.config, "_class_name", "lora")
                    local_path_state_dict, converted = self.maybe_convert_state_dict(
                        local_path, class_name
                    )

                    local_path_state_dict = strip_common_prefix(
                        local_path_state_dict, model.state_dict()
                    )

                    # Normalize keys that include an embedded adapter name, e.g.:
                    # "vace_blocks.0.attn2.to_k.lora_B.default.weight"
                    # becomes "vace_blocks.0.attn2.to_k.lora_B.weight"
                    local_path_state_dict = self._strip_adapter_name_from_keys(
                        local_path_state_dict
                    )
                    
                    keys = list(local_path_state_dict.keys())

                    prefix = None
                    if keys[0].startswith("transformer") and keys[-1].startswith(
                        "transformer"
                    ):
                        prefix = "transformer"
                    elif keys[0].startswith("diffusion_model") and keys[-1].startswith("diffusion_model"):
                        prefix = "diffusion_model"
                    elif keys[0].startswith("model") and keys[-1].startswith("model"):
                        prefix = "model"
                    
                    # ensure adapter name is not too long and does not have . or / in it if so remove it
                    adapter_name = self._clean_adapter_name(adapter_name)
                    model.load_lora_adapter(
                        local_path_state_dict, adapter_name=adapter_name, prefix=prefix
                    )
                    
                    

                    logger.info(f"Loaded LoRA {adapter_name} from {local_path}")

            # Activate all adapters with their weights in one call
            try:
                model.set_adapters(final_names, weights=final_scales)
                loaded_resolved.append(resolved[i])
            except Exception as e:
                raise e # For now 
                logger.warning(
                    f"Failed to activate adapters {final_names} with scales {final_scales}: {e}"
                )
        return loaded_resolved

    def maybe_convert_state_dict(self, local_path: str, model_name: str) -> str:
        state_dict = self.load_file(local_path)
        converter = get_transformer_converter_by_model_name(model_name)
        converted = False
        if converter is not None:
            converter.convert(state_dict)
            converted = True
        return state_dict, converted
    
    
    def _format_to_extension(self, format: str) -> str:
        format = format.lower()
        if format == "safetensors" or format == "safetensor":
            return "safetensors"
        elif format == "pickletensor" or format == "pickle" or format == "pt" or format == "pth":
            return "pt"
        else:
            return "safetensors" # default to safetensors

    def _download_from_civitai_spec(
        self,
        spec: str,
        progress_callback: Optional[Callable[[int, Optional[int], Optional[str]], None]] = None,
    ) -> str:
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

        def download_file_id(file_id: Union[int, str], format: Optional[str] = None) -> str:
            url = f"https://civitai.com/api/download/models/{file_id}"
            url_params = {}
            if format is not None and format == "safetensors" or format == "pt": 
                url_params["type"] = "Model"
                url_params["format"] = "SafeTensor" if format == "safetensors" or format == "safetensor" else "PickleTensor"
            api_key = os.getenv("CIVITAI_API_KEY", None)
            if api_key:
                url_params["token"] = api_key
            url_params = urlencode(url_params)
            url = f"{url}?{url_params}"
            local_path = self.download_from_url(url, self.save_dir, progress_callback=progress_callback, filename=f"{file_id}.{format}" if format else None) 
            return local_path

        if spec.startswith("civitai-file:"):
            file_id = spec.split(":", 1)[1]
            return download_file_id(file_id, format)

        # civitai:MODEL_ID
        model_id = spec.split(":", 1)[1]
        meta_url = f"https://civitai.com/api/v1/models/{model_id}"
        # Metadata fetch is typically small; no need to wire byte-level progress here.
        resp = requests.get(meta_url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # Find a file that looks like a LoRA in SafeTensor format
        format = None
        for version in data.get("modelVersions", []):
            file_id = version.get("id")
            for f in version.get("files", []):
                fname = f.get("name") or ""
                format = f.get("metadata", {}).get("format", "").lower()
                format = self._format_to_extension(format)
                if fname.lower().endswith((".safetensors", ".pt", ".bin")) and file_id is not None:
                    return download_file_id(file_id, format)
        
        raise RuntimeError(f"No downloadable files found for CivitAI model {model_id}")

    def load_file(self, local_path: str) -> str:
        if local_path.endswith(".safetensors"):
            return load_file(local_path)
        else:
            torch.load(local_path)

    def _strip_adapter_name_from_keys(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Some PEFT exports include the adapter name in the key, e.g.:
          "vace_blocks.0.attn2.to_k.lora_B.default.weight"
        where "default" (or any other string) is the adapter name.
        This method removes that adapter-name segment so the key becomes:
          "vace_blocks.0.attn2.to_k.lora_B.weight".
        """
        new_state: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            parts = key.split(".")
            # Look for the specific pattern: ... lora_A|lora_B.<adapter_name>.weight|bias|alpha
            if (
                len(parts) >= 3
                and parts[-3] in ("lora_A", "lora_B")
                and parts[-1] in ("weight", "bias", "alpha")
                and parts[-2] not in ("lora_A", "lora_B")
            ):
                # Drop the adapter name (the penultimate component)
                parts.pop(-2)
                key = ".".join(parts)
            new_state[key] = value
        
        del state_dict
        return new_state
