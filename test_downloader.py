import os
from dotenv import load_dotenv
load_dotenv()
from src.mixins.download_mixin import DownloadMixin

class _Downloader(DownloadMixin):
    """Concrete helper for tests."""
    pass

save_dir = "./"
dl = _Downloader()

url = "https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne/resolve/main/Mega-v9/wan2.2-rapid-mega-aio-nsfw-v9.safetensors"
progress_callback = lambda x, y, z: print(f"Downloaded {x} bytes of {y} bytes for {z}")
result_path = dl.download(url, save_dir, progress_callback)
detected = dl.is_downloaded(url, save_dir)
print(result_path)
print(detected)  