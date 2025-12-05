import os 
os.environ["CIVITAI_API_KEY"] = "79af640ad42efd36ff9f59a0d6c36ff4"
from src.lora.manager import LoraManager

lora_manager = LoraManager()
lora_item = lora_manager.resolve("civitai:1336683")
print(lora_item)

