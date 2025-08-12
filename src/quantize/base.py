from abc import ABC, abstractmethod
from pathlib import Path
import yaml
from src.utils.yaml import LoaderWithInclude
from src.mixins.download_mixin import DownloadMixin

class BaseQuantizer(ABC, DownloadMixin):
    @abstractmethod
    def quantize(self, output_path: str = None, **kwargs):
        pass
    
    def _load_yaml(self, file_path: str | Path):
        file_path = Path(file_path)
        text = file_path.read_text()

        # --- PASS 1: extract your `shared:` list with a loader that skips !include tags ---
        prelim = yaml.load(text, Loader=yaml.FullLoader)
        # prelim.get("shared", [...]) is now a list of file-paths strings.

        # build alias → manifest Path
        shared_manifests = {}
        for entry in prelim.get("shared", []):
            p = (file_path.parent / entry).resolve()
            # assume e.g. "shared_wan.yml" → alias "wan"
            alias = p.stem.split("_", 1)[1]
            shared_manifests[alias] = p

        # attach it to our custom loader
        LoaderWithInclude.shared_manifests = shared_manifests

        # --- PASS 2: real load with !include expansion ---
        return yaml.load(text, Loader=LoaderWithInclude)