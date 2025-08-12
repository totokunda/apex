from src.quantize import TextEncoderQuantizer

quantizer = TextEncoderQuantizer(
    output_path="./umt5.gguf",
    model_path="hunyuanvideo-community/HunyuanVideo-I2V/text_encoder",
    tokenizer_path="hunyuanvideo-community/HunyuanVideo-I2V/tokenizer",
    file_type="f16",
    quantization="Q4_K",
)

quantizer.quantize()