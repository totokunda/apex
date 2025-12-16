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


def test_wan_transformer_converter_does_not_swap_image_embedder_norm2():
    """
    Regression test:

    `img_emb.proj.4` maps to `condition_embedder.image_embedder.norm2`.
    The WAN norm2/norm3 swap hack must not rewrite this introduced `norm2`
    into `norm__placeholder`/`norm3`.
    """
    converter = WanTransformerConverter()

    state_dict = {
        # I2V image embedder path (introduces `.norm2` during conversion)
        "img_emb.proj.4.weight": torch.zeros(1),
        # Ensure the norm swap hack still applies to source block norms
        "blocks.0.norm2.weight": torch.zeros(1),
        "blocks.0.norm3.weight": torch.zeros(1),
    }

    converter.convert(state_dict)

    assert "condition_embedder.image_embedder.norm2.weight" in state_dict
    assert "condition_embedder.image_embedder.norm__placeholder.weight" not in state_dict
    assert "condition_embedder.image_embedder.norm3.weight" not in state_dict

    # Norm swap still works for block norms
    assert "blocks.0.norm3.weight" in state_dict
    assert "blocks.0.norm2.weight" in state_dict

