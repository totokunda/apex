from src.quantize import TransformerQuantizer

quantizer = TransformerQuantizer(
    output_path="./wan_1.3b.gguf",
    model_path='apex-diffusion/components/Wan-AI_Wan2.1-T2V-1.3B-Diffusers/transformer',
    architecture="wan",
    quantization="Q8_0",
)

quantizer.quantize()    