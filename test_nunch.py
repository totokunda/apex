import math
import os 
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
precision = get_precision() 

num_inference_steps = 8  # you can also use the 8-step model to improve the quality
rank = 128  # you can also use the rank=128 model to improve the quality
model_paths = {
    4: f"nunchaku-tech/nunchaku-qwen-image/svdq-{precision}_r{rank}-qwen-image-lightningv1.0-4steps.safetensors",
    8: f"nunchaku-tech/nunchaku-qwen-image/svdq-{precision}_r{rank}-qwen-image-lightningv1.1-8steps.safetensors",
}


# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_paths[num_inference_steps])
pipe = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image", transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
)

if get_gpu_memory() > 18:
    pipe.enable_model_cpu_offload()
else:
    # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=1
    )  # increase num_blocks_on_gpu if you have more VRAM
    pipe._exclude_from_cpu_offload.append("transformer")
    pipe.enable_sequential_cpu_offload()

prompt = """A highly realistic, NASA-style space photograph captured from low orbit. An astronaut in a modern EVA suit drifts slowly against the blackness of space, tethered to an unseen spacecraft. The suit is utilitarian and slightly worn, with visible fabric stitching, scuff marks, mission patches, and subtle dust particles. Earth curves below with accurate scale, thin atmospheric limb glowing faint blue, cloud systems and ocean patterns clearly visible. Sunlight is harsh and directional, creating strong contrast between bright highlights and deep shadow, with no exaggerated colors. The background is a sparse star field, dim and physically accurate. No nebulae or fantasy elements. Natural camera exposure, realistic reflections in the helmet visor, true-to-life color balance, documentary photography style, ISS EVA reference, ultra-high detail, photorealistic."""
image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    num_inference_steps=num_inference_steps,
    true_cfg_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save(f"qwen-image-lightning_r{rank}.png")