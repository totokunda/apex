from src.quantize import TextEncoderQuantizer

quantizer = TextEncoderQuantizer(
    output_path="./step_llm.gguf",
    model_path="stepfun-ai/stepvideo-t2v/step_llm",
    tokenizer_path="stepfun-ai/stepvideo-t2v/step_llm",
    file_type="f16",
    quantization="Q3_K",
)

quantizer.quantize()