import torch

from src.converters.transformer_converters import WanTransformerConverter


def test_wan_transformer_converter_noop_on_already_converted_state_dict():
    """
    Calling `convert()` on an already-converted WAN state_dict should be a no-op.

    This prevents accidental key corruption on repeated conversions (e.g. the
    norm2/norm3 swap hack in WAN converters).
    """
    converter = WanTransformerConverter()

    # Minimal already-converted keys (match diffusers-style WAN transformer keys).
    state_dict = {
        "blocks.0.attn1.to_q.weight": torch.zeros(1),
        "blocks.0.attn1.to_q.bias": torch.zeros(1),
        "condition_embedder.time_embedder.linear_1.weight": torch.zeros(1),
        "condition_embedder.text_embedder.linear_1.weight": torch.zeros(1),
    }

    before_keys = sorted(state_dict.keys())
    converter.convert(state_dict)
    after_keys = sorted(state_dict.keys())

    assert after_keys == before_keys


def test_wan_transformer_converter_still_converts_source_keys():
    """
    Ensure the early-exit heuristic doesn't block real conversions.
    """
    converter = WanTransformerConverter()

    state_dict = {
        "blocks.0.self_attn.q.weight": torch.zeros(1),
        "time_embedding.0.weight": torch.zeros(1),
    }

    converter.convert(state_dict)

    assert "blocks.0.attn1.to_q.weight" in state_dict
    assert "condition_embedder.time_embedder.linear_1.weight" in state_dict

