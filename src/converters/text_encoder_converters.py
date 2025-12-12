from typing import Dict, Any

from src.converters.transformer_converters import TransformerConverter


class T5TextEncoderConverter(TransformerConverter):
    """
    Converter for T5-style text encoder checkpoints.

    Uses the same rename/convert mechanics as `TransformerConverter`, with a
    `rename_dict` mirroring `T5_SD_MAP` from `src.quantize.load`.
    """

    def __init__(self):
        super().__init__()
        self.rename_dict = {
            # ---- Legacy quantized T5-style checkpoints (T5_SD_MAP) ----
            # Kept for backwards compatibility with older GGUF/SD mappings.
            "enc.": "encoder.",
            ".blk.": ".block.",
            "token_embd": "shared",
            "output_norm": "final_layer_norm",
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            "attn_norm": "layer.0.layer_norm",
            "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
            "ffn_up": "layer.1.DenseReluDense.wi_1",
            "ffn_down": "layer.1.DenseReluDense.wo",
            "ffn_gate": "layer.1.DenseReluDense.wi_0",
            "ffn_norm": "layer.1.layer_norm",
            # ---- New Wan 2.x text-encoder layout -> HF UMT5EncoderModel ----
            # Top-level embeddings / final norm
            "token_embedding": "shared",
            "norm.weight": "encoder.final_layer_norm.weight",
            "norm.bias": "encoder.final_layer_norm.bias",
            # Blocks prefix
            "blocks": "encoder.block",   # older style
            "block.": "encoder.block.",  # Wan 2.2 style: block.<idx>.*
            # Self-attention inside each block
            "attn.q": "layer.0.SelfAttention.q",
            "attn.k": "layer.0.SelfAttention.k",
            "attn.v": "layer.0.SelfAttention.v",
            "attn.o": "layer.0.SelfAttention.o",
            "attn.norm": "layer.0.layer_norm",
            "attn.rel_b": "layer.0.SelfAttention.relative_attention_bias",
            # Some checkpoints use sparse-dot style names without the "attn." prefix.
            "attn_q": "layer.0.SelfAttention.q",
            "attn_k": "layer.0.SelfAttention.k",
            "attn_v": "layer.0.SelfAttention.v",
            "attn_o": "layer.0.SelfAttention.o",
            # Per-block norms (Wan 2.2: norm1 / norm2)
            ".norm1": ".layer.0.layer_norm",
            ".norm2": ".layer.1.layer_norm",
            # Per-block position / relative bias
            "pos_embedding.embedding": "layer.0.SelfAttention.relative_attention_bias",
            # Feed-forward (Wan: ffn.fc1 / ffn.fc2 / ffn.gate.0)
            ".ffn.fc1": ".layer.1.DenseReluDense.wi_1",
            ".ffn.fc2": ".layer.1.DenseReluDense.wo",
            ".ffn.gate.0": ".layer.1.DenseReluDense.wi_0",
        }
        # No pre-special handling for now.
        self.pre_special_keys_map: Dict[str, Any] = {}
        # Special handling to mirror T5's tied input embeddings:
        # HF models expect both `shared.weight` and `encoder.embed_tokens.weight`.
        # The Wan checkpoints only provide a single token embedding weight tensor.
        self.special_keys_map: Dict[str, Any] = {
            "shared.weight": self._duplicate_shared_to_embed_tokens_inplace,
        }

    @staticmethod
    def _duplicate_shared_to_embed_tokens_inplace(key: str, state_dict: Dict[str, Any]):
        """
        Ensure that `encoder.embed_tokens.weight` exists and shares weights with `shared.weight`.

        UMT5-style encoders typically tie embeddings by setting:
          - `shared.weight`
          - `encoder.embed_tokens.weight` (same tensor)
        The Wan checkpoints only carry a single embedding matrix; we reuse it.
        """
        if key not in state_dict:
            return
        if "encoder.embed_tokens.weight" not in state_dict:
            state_dict["encoder.embed_tokens.weight"] = state_dict[key]

    def convert(self, state_dict: Dict[str, Any]):
        """
        Custom convert that allows multiple rename rules to apply to the same key.

        The base `TransformerConverter.convert` updates the state dict inside the
        inner rename loop, which effectively limits each key to one replacement.
        For T5/UMT5 we often need to rewrite both the block prefix and the
        inner-attention/FFN subkeys (e.g. `blocks.0.attn.q` ->
        `encoder.block.0.layer.0.SelfAttention.q`). This override applies all
        applicable replacements first, then performs a single in-place update.
        """

        # Keep the same ordering semantics as the base class.
        self._sort_rename_dict()

        # Apply any pre-special handlers first (these may drop or reshape keys).
        for key in list(state_dict.keys()):
            for (
                pre_special_key,
                handler_fn_inplace,
            ) in self.pre_special_keys_map.items():
                if pre_special_key in key:
                    handler_fn_inplace(key, state_dict)

        # Apply *all* rename_dict rules to each key before updating the dict.
        for key in list(state_dict.keys()):
            new_key = key
            for replace_key, rename_key in self.rename_dict.items():
                # Handle legacy "enc." prefix carefully: only normalize true
                # `enc.*` prefixes and avoid re-touching already-correct
                # `encoder.*` keys.
                if replace_key == "enc.":
                    if new_key.startswith("enc."):
                        new_key = new_key.replace("enc.", rename_key, 1)
                    continue

                if replace_key in new_key:
                    new_key = new_key.replace(replace_key, rename_key)

            # Collapse any accidental double "encoder.encoder." prefixes that
            # may be introduced when combining legacy `enc.` rules with newer
            # Wan-style `blocks` → `encoder.block` mappings.
            while "encoder.encoder." in new_key:
                new_key = new_key.replace("encoder.encoder.", "encoder.")

            if new_key != key and key in state_dict:
                state_dict[new_key] = state_dict.pop(key)

        # Finally, run any special key handlers (e.g., tying embeddings).
        for key in list(state_dict.keys()):
            for special_key, handler_fn_inplace in self.special_keys_map.items():
                if special_key in key:
                    handler_fn_inplace(key, state_dict)


class LlamaTextEncoderConverter(TransformerConverter):
    """
    Converter for LLaMA-style text encoder checkpoints.

    The mapping here mirrors `LLAMA_SD_MAP` from `src.quantize.load`.
    """

    def __init__(self):
        super().__init__()
        self.rename_dict = {
            "blk.": "model.layers.",
            "attn_norm": "input_layernorm",
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_norm": "post_attention_layernorm",
            "token_embd": "model.embed_tokens",
            "output_norm": "model.norm",
            "output.weight": "lm_head.weight",
        }
        self.pre_special_keys_map: Dict[str, Any] = {}
        self.special_keys_map: Dict[str, Any] = {}


class StepTextEncoderConverter(TransformerConverter):
    """
    Converter for STEP-style text encoder checkpoints.

    The mapping here mirrors `STEP_SD_MAP` from `src.quantize.load`.
    """

    def __init__(self):
        super().__init__()
        self.rename_dict = {
            # layers
            "blk.": "transformer.layers.",
            # attention norms
            "attn_norm": "attention_norm",
            # attention projections (unfused path for GGUF)
            "attn_q": "attention.wq",
            "attn_k": "attention.wk",
            "attn_v": "attention.wv",
            "attn_output": "attention.wo",
            # ffn norms
            "ffn_norm": "ffn_norm",
            # feed-forward weights (unfused path for GGUF)
            "ffn_gate": "feed_forward.ffn_gate",
            "ffn_up": "feed_forward.ffn_up",
            "ffn_down": "feed_forward.ffn_down",
            # embeddings
            "token_embd": "tok_embeddings.word_embeddings",
        }
        self.pre_special_keys_map: Dict[str, Any] = {}
        self.special_keys_map: Dict[str, Any] = {}


