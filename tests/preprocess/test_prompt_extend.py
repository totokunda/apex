import os
import torch
from src.prompt_extension.prompt_extend import PromptExtendPreprocessor

os.environ["APEX_PROMPT_EXTEND_SMALL"] = "HuggingFaceTB/SmolLM2-360M-Instruct"
os.environ["APEX_LLAMA_8B_MODEL"] = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"

def test_prompt_extend_small_model_if_enabled():
    # Disabled by default to keep CI light. Enable by setting APEX_PROMPT_EXTEND_SMALL
    model_id = os.environ.get("APEX_PROMPT_EXTEND_SMALL", None)

    pre = PromptExtendPreprocessor(
        model_path=model_id,
        base="AutoModelForCausalLM",
        tokenizer_name=model_id,
        dtype=torch.float16,
    )

    out = pre(
        "A cat sitting on a windowsill.",
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    print(out.prompt)

    assert isinstance(out.prompt, str)
    assert len(out.prompt) > 0


def test_prompt_extend_llama8b_if_available():
    # Guarded by env var to avoid large downloads in CI.
    model_id = os.environ.get("APEX_LLAMA_8B_MODEL", None)
    pre = PromptExtendPreprocessor(
        model_path=model_id,
        base="AutoModelForCausalLM",
        tokenizer_name=model_id,
        dtype=torch.float16,
    )

    out = pre(
        "A cinematic futuristic city at dusk.",
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    print(out.prompt)

    assert isinstance(out.prompt, str)
    assert len(out.prompt) > 0


if __name__ == "__main__":
    test_prompt_extend_small_model_if_enabled()
    test_prompt_extend_llama8b_if_available()