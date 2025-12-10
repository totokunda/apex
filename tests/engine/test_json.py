from dotenv import load_dotenv
load_dotenv()
from diffusers.modular_pipelines import ModularPipeline

pipeline = ModularPipeline.from_pretrained("briaai/FIBO-VLM-prompt-to-JSON", trust_remote_code=True)
# Generate - short text to JSON
pipeline.to("cuda")
prompt="A woman with long, dark hair, dressed in a simple, dark gown. Her face is pale and her eyes are closed, conveying deep sorrow."
output = pipeline(prompt=prompt)
print(output)
