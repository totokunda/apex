from src.helpers import StepVideoTextEncoder
from src.utils.defaults import DEFAULT_COMPONENTS_PATH
import os

gguf_path = 'step_llm.Q3_K.gguf'

preprocessor = StepVideoTextEncoder(
    model_path=os.path.join(DEFAULT_COMPONENTS_PATH, "stepfun-ai_stepvideo-t2v/step_llm"),
    tokenizer_path=os.path.join(DEFAULT_COMPONENTS_PATH, "stepfun-ai_stepvideo-t2v/step_llm/step1_chat_tokenizer.model"),
    gguf_path=gguf_path
)