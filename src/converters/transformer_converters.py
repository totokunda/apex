from typing import Dict, Any
from src.converters.utils import update_state_dict_, swap_proj_gate, swap_scale_shift
from src.quantize.ggml_ops import ggml_cat, ggml_chunk, ggml_split
import torch
import re

class TransformerConverter:
    def __init__(self):
        self.rename_dict = {}
        self.special_keys_map = {}
        self.pre_special_keys_map = {}

    @staticmethod
    def _is_specific_marker(s: str) -> bool:
        """
        Return True if `s` is a "specific" key fragment that can be used as a reliable
        signal when determining whether a checkpoint has already been converted.

        We intentionally ignore very generic fragments (e.g. "norm2") that may appear
        in *both* source and target key formats.
        """
        if not s:
            return False
        # Common ambiguous fragments that can exist in both source & target layouts.
        if s in {"norm", "norm1", "norm2", "norm3", "weight", "bias"}:
            return False
        # Prefer dotted/underscored fragments; otherwise require sufficient length.
        return ("." in s) or ("_" in s) or (len(s) >= 8)

    def _already_converted(self, state_dict: Dict[str, Any]) -> bool:
        """
        Best-effort heuristic to detect whether `state_dict` appears to already be in
        the *target* key format for this converter.

        This is intentionally conservative:
        - Requires *positive* evidence of target keys (target markers present)
        - Requires *absence* of source markers that strongly suggest an unconverted ckpt
        - Refuses to early-exit if we'd otherwise drop keys via pre/special handlers
        """
        if not state_dict:
            return True

        keys = list(state_dict.keys())

        # Guard against partially-converted states introduced by placeholder hacks.
        if any("norm__placeholder" in k for k in keys):
            return False

        # If we'd drop or synthesize keys, we are not "fully matching" yet.
        if self.pre_special_keys_map:
            for pre_special_key in self.pre_special_keys_map.keys():
                if any(pre_special_key in k for k in keys):
                    return False
        if self.special_keys_map:
            for special_key in self.special_keys_map.keys():
                if any(special_key in k for k in keys):
                    return False

        # Build conservative marker sets from the rename map.
        source_markers = [k for k in self.rename_dict.keys() if self._is_specific_marker(k)]
        target_markers = [v for v in self.rename_dict.values() if self._is_specific_marker(v)]

        # Without target markers we cannot safely assert the dict is already converted.
        if not target_markers:
            return False

        has_target = any(any(m in k for m in target_markers) for k in keys)
        if not has_target:
            return False

        has_source = any(any(m in k for m in source_markers) for k in keys)
        return not has_source

    @staticmethod
    def remove_keys_inplace(key: str, state_dict: Dict[str, Any]):
        state_dict.pop(key, None)

    def _sort_rename_dict(self):
        """Sort rename_dict by value length from longest to shortest to ensure proper replacement order."""
        self.rename_dict = dict(
            sorted(
                self.rename_dict.items(), key=lambda item: len(item[0]), reverse=True
            )
        )

    def convert(self, state_dict: Dict[str, Any]):
        self._sort_rename_dict()
        # If this looks like a checkpoint that already matches the target key layout,
        # exit early to keep conversion idempotent.
        if self._already_converted(state_dict):
            return state_dict
        # Apply pre-special keys map
        for key in list(state_dict.keys()):
            for (
                pre_special_key,
                handler_fn_inplace,
            ) in self.pre_special_keys_map.items():
                if pre_special_key in key:
                    handler_fn_inplace(key, state_dict)

        for key in list(state_dict.keys()):
            new_key = key
            for replace_key, rename_key in self.rename_dict.items():
                new_key = new_key.replace(replace_key, rename_key)
            update_state_dict_(state_dict, key, new_key)

        for key in list(state_dict.keys()):
            for special_key, handler_fn_inplace in self.special_keys_map.items():
                if special_key not in key:
                    continue
                handler_fn_inplace(key, state_dict)
        return state_dict

class WanTransformerConverter(TransformerConverter):
    def __init__(self):
        self.rename_dict = {
            "time_embedding.0": "condition_embedder.time_embedder.linear_1",
            "time_embedding.2": "condition_embedder.time_embedder.linear_2",
            "text_embedding.0": "condition_embedder.text_embedder.linear_1",
            "text_embedding.2": "condition_embedder.text_embedder.linear_2",
            "time_projection.1": "condition_embedder.time_proj",
            "head.modulation": "scale_shift_table",
            "head.head": "proj_out",
            "modulation": "scale_shift_table",
            "ffn.0": "ffn.net.0.proj",
            "ffn.2": "ffn.net.2",
            # Hack to swap the layer names
            # The original model calls the norms in following order: norm1, norm3, norm2
            # We convert it to: norm1, norm2, norm3
            "norm2": "norm__placeholder",
            "norm3": "norm2",
            "norm__placeholder": "norm3",
            # For the I2V model
            "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
            "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
            "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
            "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
            # for the FLF2V model
            "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
            # for the IP model
            "self_attn.q_loras": "attn1.add_q_lora",
            "self_attn.k_loras": "attn1.add_k_lora",
            "self_attn.v_loras": "attn1.add_v_lora",
            # Add attention component mappings
            # original keys
            "self_attn.q": "attn1.to_q",
            "self_attn.k": "attn1.to_k",
            "self_attn.v": "attn1.to_v",
            "self_attn.o": "attn1.to_out.0",
            "self_attn.norm_q": "attn1.norm_q",
            "self_attn.norm_k": "attn1.norm_k",
            "cross_attn.q": "attn2.to_q",
            "cross_attn.k": "attn2.to_k",
            "cross_attn.v": "attn2.to_v",
            "cross_attn.o": "attn2.to_out.0",
            "cross_attn.norm_q": "attn2.norm_q",
            "cross_attn.norm_k": "attn2.norm_k",
            "cross_attn.k_img": "attn2.add_k_proj",
            "cross_attn.v_img": "attn2.add_v_proj",
            "cross_attn.norm_k_img": "attn2.norm_added_k",
        }
        self.special_keys_map = {}
        # Drop auxiliary diff-bias / diff vectors that don't have a counterpart in the target Wan model.
        # Example keys:
        #   diffusion_model.blocks.0.cross_attn.k.diff_b
        #   diffusion_model.blocks.0.cross_attn.norm_k.diff
        # Also drop scaled-FP8 metadata tensors which don't exist in the target
        # diffusers WAN transformer:
        #   - a top-level "scaled_fp8" tensor
        #   - per-weight "*.scale_weight" tensors, e.g.:
        #       blocks.0.attn1.to_q.scale_weight
        #       blocks.0.ffn.net.0.proj.scale_weight
        self.pre_special_keys_map = {
            ".diff_b": self.remove_keys_inplace,
            ".diff": self.remove_keys_inplace,
            "scaled_fp8": self.remove_keys_inplace,
        }


class WanAnimateTransformerConverter(TransformerConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {
            "time_embedding.0": "condition_embedder.time_embedder.linear_1",
            "time_embedding.2": "condition_embedder.time_embedder.linear_2",
            "text_embedding.0": "condition_embedder.text_embedder.linear_1",
            "text_embedding.2": "condition_embedder.text_embedder.linear_2",
            "time_projection.1": "condition_embedder.time_proj",
            "head.modulation": "scale_shift_table",
            "head.head": "proj_out",
            "modulation": "scale_shift_table",
            "ffn.0": "ffn.net.0.proj",
            "ffn.2": "ffn.net.2",
            # Hack to swap the layer names
            # The original model calls the norms in following order: norm1, norm3, norm2
            # We convert it to: norm1, norm2, norm3
            "norm2": "norm__placeholder",
            "norm3": "norm2",
            "norm__placeholder": "norm3",
            "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
            "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
            "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
            "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
            # Add attention component mappings
            "self_attn.q": "attn1.to_q",
            "self_attn.k": "attn1.to_k",
            "self_attn.v": "attn1.to_v",
            "self_attn.o": "attn1.to_out.0",
            "self_attn.norm_q": "attn1.norm_q",
            "self_attn.norm_k": "attn1.norm_k",
            "cross_attn.q": "attn2.to_q",
            "cross_attn.k": "attn2.to_k",
            "cross_attn.v": "attn2.to_v",
            "cross_attn.o": "attn2.to_out.0",
            "cross_attn.norm_q": "attn2.norm_q",
            "cross_attn.norm_k": "attn2.norm_k",
            "cross_attn.k_img": "attn2.to_k_img",
            "cross_attn.v_img": "attn2.to_v_img",
            "cross_attn.norm_k_img": "attn2.norm_k_img",
            # After cross_attn -> attn2 rename, we need to rename the img keys
            "attn2.to_k_img": "attn2.add_k_proj",
            "attn2.to_v_img": "attn2.add_v_proj",
            "attn2.norm_k_img": "attn2.norm_added_k",
            # Wan Animate-specific mappings (motion encoder, face encoder, face adapter)
            # Motion encoder mappings
            # The name mapping is complicated for the convolutional part so we handle that in its own function
            "motion_encoder.enc.fc": "motion_encoder.motion_network",
            "motion_encoder.dec.direction.weight": "motion_encoder.motion_synthesis_weight",
            # Face encoder mappings - CausalConv1d has a .conv submodule that we need to flatten
            "face_encoder.conv1_local.conv": "face_encoder.conv1_local",
            "face_encoder.conv2.conv": "face_encoder.conv2",
            "face_encoder.conv3.conv": "face_encoder.conv3",
            # Face adapter mappings are handled in a separate function
        }

        # Special handling for Animate-only auxiliary modules.
        # - Motion encoder: rename sequential indices, drop blur-kernel buffers, fix bias shapes
        # - Face adapter: split fused KV projections into separate K/V projections
        self.special_keys_map = {
            "motion_encoder": convert_animate_motion_encoder_weights,
            "face_adapter": convert_animate_face_adapter_weights,
            ".diff_b": self.remove_keys_inplace,
            ".diff": self.remove_keys_inplace,
            "scaled_fp8": self.remove_keys_inplace,
        }

        
    def convert(self, state_dict: Dict[str, Any]):
        for key in list(state_dict.keys()):
            new_key = key[:]
            for replace_key, rename_key in self.rename_dict.items():
                new_key = new_key.replace(replace_key, rename_key)
            update_state_dict_(state_dict, key, new_key)
    
        for key in list(state_dict.keys()):
            for special_key, handler_fn_inplace in self.special_keys_map.items():
                if special_key not in key:
                    continue
                handler_fn_inplace(key, state_dict)

    
# TODO: Verify this and simplify if possible.
def convert_animate_motion_encoder_weights(key: str, state_dict: Dict[str, Any], final_conv_idx: int = 8) -> None:
    """
    Convert all motion encoder weights for Animate model.

    In the original model:
    - All Linear layers in fc use EqualLinear
    - All Conv2d layers in convs use EqualConv2d (except blur_conv which is initialized separately)
    - Blur kernels are stored as buffers in Sequential modules
    - ConvLayer is nn.Sequential with indices: [Blur (optional), EqualConv2d, FusedLeakyReLU (optional)]

    Conversion strategy:
    1. Drop .kernel buffers (blur kernels)
    2. Rename sequential indices to named components (e.g., 0 -> conv2d, 1 -> bias_leaky_relu)
    """
    # Skip if not a weight, bias, or kernel
    if ".weight" not in key and ".bias" not in key and ".kernel" not in key:
        return

    # Handle Blur kernel buffers from original implementation.
    # After renaming, these appear under: motion_encoder.res_blocks.*.conv{2,skip}.blur_kernel
    # Diffusers constructs blur kernels as a non-persistent buffer so we must drop these keys
    if ".kernel" in key and "motion_encoder" in key:
        # Remove unexpected blur kernel buffers to avoid strict load errors
        state_dict.pop(key, None)
        return

    # Rename Sequential indices to named components in ConvLayer and ResBlock
    if ".enc.net_app.convs." in key and (".weight" in key or ".bias" in key):
        parts = key.split(".")

        # Find the sequential index (digit) after convs or after conv1/conv2/skip
        # Examples:
        # - enc.net_app.convs.0.0.weight -> conv_in.weight (initial conv layer weight)
        # - enc.net_app.convs.0.1.bias -> conv_in.act_fn.bias (initial conv layer bias)
        # - enc.net_app.convs.{n:1-7}.conv1.0.weight -> res_blocks.{(n-1):0-6}.conv1.weight (conv1 weight)
        #     - e.g. enc.net_app.convs.1.conv1.0.weight -> res_blocks.0.conv1.weight
        # - enc.net_app.convs.{n:1-7}.conv1.1.bias -> res_blocks.{(n-1):0-6}.conv1.act_fn.bias (conv1 bias)
        #     - e.g. enc.net_app.convs.1.conv1.1.bias -> res_blocks.0.conv1.act_fn.bias
        # - enc.net_app.convs.{n:1-7}.conv2.1.weight -> res_blocks.{(n-1):0-6}.conv2.weight (conv2 weight)
        # - enc.net_app.convs.1.conv2.2.bias -> res_blocks.0.conv2.act_fn.bias (conv2 bias)
        # - enc.net_app.convs.{n:1-7}.skip.1.weight -> res_blocks.{(n-1):0-6}.conv_skip.weight (skip conv weight)
        # - enc.net_app.convs.8 -> conv_out (final conv layer)

        convs_idx = parts.index("convs") if "convs" in parts else -1
        if convs_idx >= 0 and len(parts) - convs_idx >= 2:
            bias = False
            # The nn.Sequential index will always follow convs
            sequential_idx = int(parts[convs_idx + 1])
            if sequential_idx == 0:
                if key.endswith(".weight"):
                    new_key = "motion_encoder.conv_in.weight"
                elif key.endswith(".bias"):
                    new_key = "motion_encoder.conv_in.act_fn.bias"
                    bias = True
            elif sequential_idx == final_conv_idx:
                if key.endswith(".weight"):
                    new_key = "motion_encoder.conv_out.weight"
            else:
                # Intermediate .convs. layers, which get mapped to .res_blocks.
                prefix = "motion_encoder.res_blocks."

                layer_name = parts[convs_idx + 2]
                if layer_name == "skip":
                    layer_name = "conv_skip"

                if key.endswith(".weight"):
                    param_name = "weight"
                elif key.endswith(".bias"):
                    param_name = "act_fn.bias"
                    bias = True

                suffix_parts = [str(sequential_idx - 1), layer_name, param_name]
                suffix = ".".join(suffix_parts)
                new_key = prefix + suffix

            param = state_dict.pop(key)
            if bias:
                param = param.squeeze()
            state_dict[new_key] = param

            return
        return
    return


def convert_animate_face_adapter_weights(key: str, state_dict: Dict[str, Any]) -> None:
    """
    Convert face adapter weights for the Animate model.

    The original model uses a fused KV projection but the diffusers models uses separate K and V projections.
    """
    # Skip if not a weight or bias
    if ".weight" not in key and ".bias" not in key:
        return

    prefix = "face_adapter."
    if ".fuser_blocks." in key:
        parts = key.split(".")

        module_list_idx = parts.index("fuser_blocks") if "fuser_blocks" in parts else -1
        if module_list_idx >= 0 and (len(parts) - 1) - module_list_idx == 3:
            block_idx = parts[module_list_idx + 1]
            layer_name = parts[module_list_idx + 2]
            param_name = parts[module_list_idx + 3]

            if layer_name == "linear1_kv":
                layer_name_k = "to_k"
                layer_name_v = "to_v"

                suffix_k = ".".join([block_idx, layer_name_k, param_name])
                suffix_v = ".".join([block_idx, layer_name_v, param_name])
                new_key_k = prefix + suffix_k
                new_key_v = prefix + suffix_v

                kv_proj = state_dict.pop(key)
                k_proj, v_proj = ggml_chunk(kv_proj, 2, dim=0)
                state_dict[new_key_k] = k_proj
                state_dict[new_key_v] = v_proj
                # check for scale_weight
                if ".weight" in key:
                    scale_weight_key = key.replace(".weight", ".scale_weight")
                    if scale_weight_key in state_dict:
                        scale_weight = state_dict.pop(scale_weight_key)
                        state_dict[new_key_k.replace(".weight", ".scale_weight")] = scale_weight
                        state_dict[new_key_v.replace(".weight", ".scale_weight")] = scale_weight
                return
            else:
                if layer_name == "q_norm":
                    new_layer_name = "norm_q"
                elif layer_name == "k_norm":
                    new_layer_name = "norm_k"
                elif layer_name == "linear1_q":
                    new_layer_name = "to_q"
                elif layer_name == "linear2":
                    new_layer_name = "to_out"

                suffix_parts = [block_idx, new_layer_name, param_name]
                suffix = ".".join(suffix_parts)
                new_key = prefix + suffix
                state_dict[new_key] = state_dict.pop(key)
                # check for scale_weight
                if ".weight" in key:
                    scale_weight_key = key.replace(".weight", ".scale_weight")
                    if scale_weight_key in state_dict:
                        scale_weight = state_dict.pop(scale_weight_key)
                        state_dict[new_key.replace(".weight", ".scale_weight")] = scale_weight
                return
    return

class SkyReelsTransformerConverter(WanTransformerConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict.update(
            {
                "fps_embedding": "condition_embedder.fps_embedding",
                "fps_projection": "condition_embedder.fps_projection",
            }
        )


class WanMultiTalkTransformerConverter(TransformerConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {
            "audio_cross_attn.q_linear": "audio_attn2.q_linear",
            "audio_cross_attn.q_norm": "audio_attn2.q_norm",
            "audio_cross_attn.k_norm": "audio_attn2.k_norm",
            "audio_cross_attn.v_norm": "audio_attn2.v_norm",
            "audio_cross_attn.kv_linear": "audio_attn2.kv_linear",
            "audio_cross_attn.add_q_norm": "audio_attn2.add_q_norm",
            "audio_cross_attn.add_k_norm": "audio_attn2.add_k_norm",
            "audio_cross_attn.proj": "audio_attn2.proj",
            "time_embedding.0": "condition_embedder.time_embedder.linear_1",
            "time_embedding.2": "condition_embedder.time_embedder.linear_2",
            "text_embedding.0": "condition_embedder.text_embedder.linear_1",
            "text_embedding.2": "condition_embedder.text_embedder.linear_2",
            "time_projection.1": "condition_embedder.time_proj",
            "head.modulation": "scale_shift_table",
            "head.head": "proj_out",
            "modulation": "scale_shift_table",
            "ffn.0": "ffn.net.0.proj",
            "ffn.2": "ffn.net.2",
            # Hack to swap the layer names
            # The original model calls the norms in following order: norm1, norm3, norm2
            # We convert it to: norm1, norm2, norm3
            "norm2": "norm__placeholder",
            "norm3": "norm2",
            "norm__placeholder": "norm3",
            # For the I2V model
            "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
            "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
            "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
            "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
            # for the FLF2V model
            "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
            # Add attention component mappings
            "self_attn.q": "attn1.to_q",
            "self_attn.k": "attn1.to_k",
            "self_attn.v": "attn1.to_v",
            "self_attn.o": "attn1.to_out.0",
            "self_attn.norm_q": "attn1.norm_q",
            "self_attn.norm_k": "attn1.norm_k",
            "cross_attn.q": "attn2.to_q",
            "cross_attn.k": "attn2.to_k",
            "cross_attn.v": "attn2.to_v",
            "cross_attn.o": "attn2.to_out.0",
            "cross_attn.norm_q": "attn2.norm_q",
            "cross_attn.norm_k": "attn2.norm_k",
            "cross_attn.k_img": "attn2.add_k_proj",
            "cross_attn.v_img": "attn2.add_v_proj",
            "cross_attn.norm_k_img": "attn2.norm_added_k",
        }

        self.special_keys_map = {}
        # WanMultiTalk checkpoints may also contain scaled-FP8 metadata tensors.
        # These do not have counterparts in the target diffusers architecture,
        # so we drop them early during conversion.
        self.pre_special_keys_map = {
            "scaled_fp8": self.remove_keys_inplace,
        }


class WanVaceTransformerConverter(WanTransformerConverter):
    def __init__(self):
        # Intentionally do not call super().__init__ here, since WanVace uses a
        # slightly different rename map than the base WAN converter.
        self.rename_dict = {
            "time_embedding.0": "condition_embedder.time_embedder.linear_1",
            "time_embedding.2": "condition_embedder.time_embedder.linear_2",
            "text_embedding.0": "condition_embedder.text_embedder.linear_1",
            "text_embedding.2": "condition_embedder.text_embedder.linear_2",
            "time_projection.1": "condition_embedder.time_proj",
            "head.modulation": "scale_shift_table",
            "head.head": "proj_out",
            "modulation": "scale_shift_table",
            "ffn.0": "ffn.net.0.proj",
            "ffn.2": "ffn.net.2",
            # Hack to swap the layer names
            # The original model calls the norms in following order: norm1, norm3, norm2
            # We convert it to: norm1, norm2, norm3
            "norm2": "norm__placeholder",
            "norm3": "norm2",
            "norm__placeholder": "norm3",
            # # For the I2V model
            # "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
            # "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
            # "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
            # "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
            # # for the FLF2V model
            # "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
            # Add attention component mappings
            "self_attn.q": "attn1.to_q",
            "self_attn.k": "attn1.to_k",
            "self_attn.v": "attn1.to_v",
            "self_attn.o": "attn1.to_out.0",
            "self_attn.norm_q": "attn1.norm_q",
            "self_attn.norm_k": "attn1.norm_k",
            "cross_attn.q": "attn2.to_q",
            "cross_attn.k": "attn2.to_k",
            "cross_attn.v": "attn2.to_v",
            "cross_attn.o": "attn2.to_out.0",
            "cross_attn.norm_q": "attn2.norm_q",
            "cross_attn.norm_k": "attn2.norm_k",
            "cross_attn.k_img": "attn2.add_k_proj",
            "cross_attn.v_img": "attn2.add_v_proj",
            "cross_attn.norm_k_img": "attn2.norm_added_k",
            "before_proj": "proj_in",
            "after_proj": "proj_out",
        }

        self.special_keys_map = {}
        # Mirror the WAN base converter behaviour: drop diff-bias helpers and
        # scaled-FP8 metadata tensors, which are only used by FP8 kernels and do
        # not correspond to any parameters in the target model.
        self.pre_special_keys_map = {
            ".diff_b": self.remove_keys_inplace,
            ".diff": self.remove_keys_inplace,
        }


class CogVideoXTransformerConverter(TransformerConverter):
    def __init__(self):
        self.rename_dict = {
            "transformer.final_layernorm": "norm_final",
            "transformer": "transformer_blocks",
            "attention": "attn1",
            "mlp": "ff.net",
            "dense_h_to_4h": "0.proj",
            "dense_4h_to_h": "2",
            ".layers": "",
            "dense": "to_out.0",
            "input_layernorm": "norm1.norm",
            "post_attn1_layernorm": "norm2.norm",
            "time_embed.0": "time_embedding.linear_1",
            "time_embed.2": "time_embedding.linear_2",
            "ofs_embed.0": "ofs_embedding.linear_1",
            "ofs_embed.2": "ofs_embedding.linear_2",
            "mixins.patch_embed": "patch_embed",
            "mixins.final_layer.norm_final": "norm_out.norm",
            "mixins.final_layer.linear": "proj_out",
            "mixins.final_layer.adaLN_modulation.1": "norm_out.linear",
            "mixins.pos_embed.pos_embedding": "patch_embed.pos_embedding",  # Specific to CogVideoX-5b-I2V
        }
        self.special_keys_map = {
            "query_key_value": self.reassign_query_key_value_inplace,
            "query_layernorm_list": self.reassign_query_key_layernorm_inplace,
            "key_layernorm_list": self.reassign_query_key_layernorm_inplace,
            "adaln_layer.adaLN_modulations": self.reassign_adaln_norm_inplace,
            "embed_tokens": self.remove_keys_inplace,
            "freqs_sin": self.remove_keys_inplace,
            "freqs_cos": self.remove_keys_inplace,
            "position_embedding": self.remove_keys_inplace,
        }
        self.pre_special_keys_map = {}

    @staticmethod
    def reassign_query_key_value_inplace(self, key: str, state_dict: Dict[str, Any]):
        to_q_key = key.replace("query_key_value", "to_q")
        to_k_key = key.replace("query_key_value", "to_k")
        to_v_key = key.replace("query_key_value", "to_v")
        to_q, to_k, to_v = ggml_chunk(state_dict[key], chunks=3, dim=0)
        state_dict[to_q_key] = to_q
        state_dict[to_k_key] = to_k
        state_dict[to_v_key] = to_v
        state_dict.pop(key)

    @staticmethod
    def reassign_query_key_layernorm_inplace(key: str, state_dict: Dict[str, Any]):
        layer_id, weight_or_bias = key.split(".")[-2:]

        if "query" in key:
            new_key = f"transformer_blocks.{layer_id}.attn1.norm_q.{weight_or_bias}"
        elif "key" in key:
            new_key = f"transformer_blocks.{layer_id}.attn1.norm_k.{weight_or_bias}"

        state_dict[new_key] = state_dict.pop(key)

    @staticmethod
    def reassign_adaln_norm_inplace(key: str, state_dict: Dict[str, Any]):
        layer_id, _, weight_or_bias = key.split(".")[-3:]

        weights_or_biases = state_dict[key].chunk(12, dim=0)
        norm1_weights_or_biases = ggml_cat(
            weights_or_biases[0:3] + weights_or_biases[6:9]
        )
        norm2_weights_or_biases = ggml_cat(
            weights_or_biases[3:6] + weights_or_biases[9:12]
        )

        norm1_key = f"transformer_blocks.{layer_id}.norm1.linear.{weight_or_bias}"
        state_dict[norm1_key] = norm1_weights_or_biases

        norm2_key = f"transformer_blocks.{layer_id}.norm2.linear.{weight_or_bias}"
        state_dict[norm2_key] = norm2_weights_or_biases

        state_dict.pop(key)

    @staticmethod
    def remove_keys_inplace(key: str, state_dict: Dict[str, Any]):
        state_dict.pop(key)

    @staticmethod
    def replace_up_keys_inplace(key: str, state_dict: Dict[str, Any]):
        key_split = key.split(".")
        layer_index = int(key_split[2])
        replace_layer_index = 4 - 1 - layer_index

        key_split[1] = "up_blocks"
        key_split[2] = str(replace_layer_index)
        new_key = ".".join(key_split)

        state_dict[new_key] = state_dict.pop(key)


class LTXTransformerConverter(TransformerConverter):
    def __init__(self):
        self.rename_dict = {
            "proj_in": "patchify_proj",
            "time_embed": "adaln_single",
            "attn1.norm_k.weight": "attn1.k_norm.weight",
            "attn1.norm_q.weight": "attn1.q_norm.weight",
            "attn2.norm_k.weight": "attn2.k_norm.weight",
            "attn2.norm_q.weight": "attn2.q_norm.weight",
        }
        self.special_keys_map = {
            "vae": self.remove_keys_inplace,
        }
        self.pre_special_keys_map = {}

    @staticmethod
    def remove_keys_inplace(key: str, state_dict: Dict[str, Any]):
        state_dict.pop(key)


class StepVideoTransformerConverter(TransformerConverter):
    def convert(self, state_dict: Dict[str, Any]):
        return state_dict


class HunyuanTransformerConverter(TransformerConverter):
    def __init__(self):
        self.rename_dict = {
            "img_in": "x_embedder",
            "time_in.mlp.0": "time_text_embed.timestep_embedder.linear_1",
            "time_in.mlp.2": "time_text_embed.timestep_embedder.linear_2",
            "guidance_in.mlp.0": "time_text_embed.guidance_embedder.linear_1",
            "guidance_in.mlp.2": "time_text_embed.guidance_embedder.linear_2",
            "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
            "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
            "double_blocks": "transformer_blocks",
            "img_attn_q_norm": "attn.norm_q",
            "img_attn_k_norm": "attn.norm_k",
            "img_attn_proj": "attn.to_out.0",
            "txt_attn_q_norm": "attn.norm_added_q",
            "txt_attn_k_norm": "attn.norm_added_k",
            "txt_attn_proj": "attn.to_add_out",
            "img_mod.linear": "norm1.linear",
            "img_norm1": "norm1.norm",
            "img_norm2": "norm2",
            "img_mlp": "ff",
            "txt_mod.linear": "norm1_context.linear",
            "txt_norm1": "norm1.norm",
            "txt_norm2": "norm2_context",
            "txt_mlp": "ff_context",
            "self_attn_proj": "attn.to_out.0",
            "modulation.linear": "norm.linear",
            "pre_norm": "norm.norm",
            "final_layer.norm_final": "norm_out.norm",
            "final_layer.linear": "proj_out",
            "fc1": "net.0.proj",
            "fc2": "net.2",
            "input_embedder": "proj_in",
        }

        self.special_keys_map = {
            "txt_in": self.remap_txt_in_,
            "img_attn_qkv": self.remap_img_attn_qkv_,
            "txt_attn_qkv": self.remap_txt_attn_qkv_,
            "single_blocks": self.remap_single_transformer_blocks_,
            "final_layer.adaLN_modulation.1": self.remap_norm_scale_shift_,
        }

        self.pre_special_keys_map = {}

    @staticmethod
    def remap_norm_scale_shift_(key, state_dict):
        weight = state_dict.pop(key)
        shift, scale = weight.chunk(2, dim=0)
        new_weight = ggml_cat([scale, shift], dim=0)
        state_dict[key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")] = (
            new_weight
        )

    @staticmethod
    def remap_txt_in_(key, state_dict):
        def rename_key(key):
            new_key = key.replace(
                "individual_token_refiner.blocks", "token_refiner.refiner_blocks"
            )
            new_key = new_key.replace("adaLN_modulation.1", "norm_out.linear")
            new_key = new_key.replace("txt_in", "context_embedder")
            new_key = new_key.replace(
                "t_embedder.mlp.0", "time_text_embed.timestep_embedder.linear_1"
            )
            new_key = new_key.replace(
                "t_embedder.mlp.2", "time_text_embed.timestep_embedder.linear_2"
            )
            new_key = new_key.replace("c_embedder", "time_text_embed.text_embedder")
            new_key = new_key.replace("mlp", "ff")
            return new_key

        if "self_attn_qkv" in key:
            weight = state_dict.pop(key)
            to_q, to_k, to_v = weight.chunk(3, dim=0)
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_q"))] = to_q
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_k"))] = to_k
            state_dict[rename_key(key.replace("self_attn_qkv", "attn.to_v"))] = to_v
        else:
            state_dict[rename_key(key)] = state_dict.pop(key)

    @staticmethod
    def remap_img_attn_qkv_(key, state_dict):
        weight = state_dict.pop(key)
        to_q, to_k, to_v = weight.chunk(3, dim=0)
        state_dict[key.replace("img_attn_qkv", "attn.to_q")] = to_q
        state_dict[key.replace("img_attn_qkv", "attn.to_k")] = to_k
        state_dict[key.replace("img_attn_qkv", "attn.to_v")] = to_v

    @staticmethod
    def remap_txt_attn_qkv_(key, state_dict):
        weight = state_dict.pop(key)
        to_q, to_k, to_v = weight.chunk(3, dim=0)
        state_dict[key.replace("txt_attn_qkv", "attn.add_q_proj")] = to_q
        state_dict[key.replace("txt_attn_qkv", "attn.add_k_proj")] = to_k
        state_dict[key.replace("txt_attn_qkv", "attn.add_v_proj")] = to_v

    @staticmethod
    def remap_single_transformer_blocks_(key, state_dict):
        hidden_size = 3072

        if "linear1.weight" in key:
            linear1_weight = state_dict.pop(key)
            split_size = (
                hidden_size,
                hidden_size,
                hidden_size,
                linear1_weight.size(0) - 3 * hidden_size,
            )
            q, k, v, mlp = ggml_split(linear1_weight, split_size, dim=0)
            new_key = key.replace(
                "single_blocks", "single_transformer_blocks"
            ).removesuffix(".linear1.weight")
            state_dict[f"{new_key}.attn.to_q.weight"] = q
            state_dict[f"{new_key}.attn.to_k.weight"] = k
            state_dict[f"{new_key}.attn.to_v.weight"] = v
            state_dict[f"{new_key}.proj_mlp.weight"] = mlp

        elif "linear1.bias" in key:
            linear1_bias = state_dict.pop(key)
            split_size = (
                hidden_size,
                hidden_size,
                hidden_size,
                linear1_bias.size(0) - 3 * hidden_size,
            )
            q_bias, k_bias, v_bias, mlp_bias = ggml_split(
                linear1_bias, split_size, dim=0
            )
            new_key = key.replace(
                "single_blocks", "single_transformer_blocks"
            ).removesuffix(".linear1.bias")
            state_dict[f"{new_key}.attn.to_q.bias"] = q_bias
            state_dict[f"{new_key}.attn.to_k.bias"] = k_bias
            state_dict[f"{new_key}.attn.to_v.bias"] = v_bias
            state_dict[f"{new_key}.proj_mlp.bias"] = mlp_bias

        else:
            new_key = key.replace("single_blocks", "single_transformer_blocks")
            new_key = new_key.replace("linear2", "proj_out")
            new_key = new_key.replace("q_norm", "attn.norm_q")
            new_key = new_key.replace("k_norm", "attn.norm_k")
            state_dict[new_key] = state_dict.pop(key)


class HunyuanAvatarTransformerConverter(HunyuanTransformerConverter):
    def __init__(self):
        super().__init__()

        self.rename_dict.update(
            {
                "audio_proj": "audio_projection",
                "motion_pose.mlp.0": "time_text_embed.motion_pose.linear_1",
                "motion_pose.mlp.2": "time_text_embed.motion_pose.linear_2",
                "motion_exp.mlp.0": "time_text_embed.motion_exp.linear_1",
                "motion_exp.mlp.2": "time_text_embed.motion_exp.linear_2",
                "fps_proj.mlp.0": "time_text_embed.fps_proj.linear_1",
                "fps_proj.mlp.2": "time_text_embed.fps_proj.linear_2",
                "ref_in": "ref_latents_embedder",
                "before_proj": "ref_latents_proj",
            }
        )


class MochiTransformerConverter(TransformerConverter):
    def __init__(self, num_layers: int = 48):
        super().__init__()
        self.num_layers = num_layers
        self.rename_dict = {
            # --- patch- & time-embed ------------------------------------------------
            "x_embedder.proj.": "patch_embed.proj.",
            "t_embedder.mlp.0.": "time_embed.timestep_embedder.linear_1.",
            "t_embedder.mlp.2.": "time_embed.timestep_embedder.linear_2.",
            "t5_y_embedder.to_kv.": "time_embed.pooler.to_kv.",
            "t5_y_embedder.to_q.": "time_embed.pooler.to_q.",
            "t5_y_embedder.to_out.": "time_embed.pooler.to_out.",
            "t5_yproj.": "time_embed.caption_proj.",
            # --- final linear -------------------------------------------------------
            "final_layer.linear.": "proj_out.",
            # --- blocks prefix ------------------------------------------------------
            "blocks.": "transformer_blocks.",  # keep the index!
            "mod_x.": "norm1.linear.",
            # *most* mod_y maps to norm1_context.linear – last block is handled in a
            # special handler because it becomes linear_1
            "mod_y.": "norm1_context.linear.",
            # mlp w2 is a direct rename, w1 handled specially for gate swap
            ".mlp_x.w2.": ".ff.net.2.",
            ".mlp_y.w2.": ".ff_context.net.2.",
            # q_norm/k_norm direct renames
            ".attn.q_norm_x.": ".attn1.norm_q.",
            ".attn.k_norm_x.": ".attn1.norm_k.",
            ".attn.q_norm_y.": ".attn1.norm_added_q.",
            ".attn.k_norm_y.": ".attn1.norm_added_k.",
            # proj_x / proj_y for attn1 outputs – proj_y only exists except last layer
            ".attn.proj_x.": ".attn1.to_out.0.",
            ".attn.proj_y.": ".attn1.to_add_out.",
        }
        self.pre_special_keys_map = {}

    def _handle_qkv_x(key: str, sd: Dict[str, Any]):
        """
        blocks.i.attn.qkv_x.weight   ->   transformer_blocks.i.attn1.{to_q,to_k,to_v}.weight
        """
        w = sd.pop(key)
        blk, _ = key.split(".attn.qkv_x.weight")
        blk = blk.replace("blocks.", "transformer_blocks.")
        q, k, v = w.chunk(3, dim=0)
        sd[f"{blk}.attn1.to_q.weight"] = q
        sd[f"{blk}.attn1.to_k.weight"] = k
        sd[f"{blk}.attn1.to_v.weight"] = v

    def _handle_qkv_y(key: str, sd: Dict[str, Any]):
        """
        blocks.i.attn.qkv_y.weight   ->   transformer_blocks.i.attn1.{add_q_proj,add_k_proj,add_v_proj}.weight
        """
        w = sd.pop(key)
        blk, _ = key.split(".attn.qkv_y.weight")
        blk = blk.replace("blocks.", "transformer_blocks.")
        q, k, v = w.chunk(3, dim=0)
        sd[f"{blk}.attn1.add_q_proj.weight"] = q
        sd[f"{blk}.attn1.add_k_proj.weight"] = k
        sd[f"{blk}.attn1.add_v_proj.weight"] = v

    def _handle_mlp_w1(key: str, sd: Dict[str, Any]):
        """
        *.mlp_[x|y].w1.weight  -> swap gate/proj and write to new location.
        """
        w = swap_proj_gate(sd.pop(key))
        is_context = ".mlp_y." in key
        blk, _ = key.split(".mlp_", maxsplit=1)
        blk = blk.replace("blocks.", "transformer_blocks.")
        if is_context:
            sd[f"{blk}.ff_context.net.0.proj.weight"] = w
        else:
            sd[f"{blk}.ff.net.0.proj.weight"] = w

    def _handle_final_layer_mod(key: str, sd: Dict[str, Any]):
        """
        final_layer.mod.[weight|bias] → norm_out.linear.[weight|bias] with swap of
        scale/shift halves.
        """
        is_weight = key.endswith(".weight")
        tensor = sd.pop(key)
        tensor = swap_scale_shift(tensor, dim=0)
        suffix = "weight" if is_weight else "bias"
        sd[f"norm_out.linear.{suffix}"] = tensor

    def _handle_mod_y_last_block(key: str, sd: Dict[str, Any]):
        """
        Last block’s mod_y is split into linear_1 instead of linear.
        """
        tensor = sd.pop(key)
        m = re.match(r"blocks\.(\d+)\.mod_y\.(weight|bias)", key)
        idx = int(m.group(1))
        suffix = m.group(2)
        new_key = f"transformer_blocks.{idx}.norm1_context.linear_1.{suffix}"
        sd[new_key] = tensor

    def convert(self, state_dict: Dict[str, Any]):
        for key in list(state_dict.keys()):
            # last layer’s mod_y handled separately first
            if f"blocks.{self.num_layers - 1}.mod_y." in key:
                self._handle_mod_y_last_block(key, state_dict)
                continue

            for pattern, handler in self.special_keys_map.items():
                if pattern in key:
                    handler(key, state_dict)
                    break  # key popped; go to next key

        # ---------- simple rename pass ------------------------------------------
        for key in list(state_dict.keys()):
            new_key = key
            for src, tgt in self.rename_dict.items():
                new_key = new_key.replace(src, tgt)
            update_state_dict_(state_dict, key, new_key)


class MagiTransformerConverter(TransformerConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {
            "videodit_blocks.final_layernorm": "norm_out",
            "final_linear": "proj_out",
            "videodit_blocks.layers": "blocks",
            "x_embedder": "patch_embedding",
            "t_embedder": "timestep_embedding",
            "y_embedder": "caption_embedding",
            "ada_modulate_layer": "adaln",
            "self_attention.linear_qkv.layer_norm": "norm1",
            "self_attn_post_norm": "norm2",
            "mlp_post_norm": "norm3",
            "mlp.layer_norm": "ffn.norm",
            "mlp.linear_fc1": "ffn.proj1",
            "mlp.linear_fc2": "ffn.proj2",
            "self_attention.linear_proj": "proj",
            "self_attention.k_layernorm_xattn.bias": "attn2.cross_k_norm.bias",
            "self_attention.k_layernorm_xattn.weight": "attn2.cross_k_norm.weight",
            "self_attention.k_layernorm.bias": "attn1.norm_k.bias",
            "self_attention.k_layernorm.weight": "attn1.norm_k.weight",
            "self_attention.q_layernorm_xattn.bias": "attn2.cross_q_norm.bias",
            "self_attention.q_layernorm_xattn.weight": "attn2.cross_q_norm.weight",
            "self_attention.q_layernorm.bias": "attn1.norm_q.bias",
            "self_attention.q_layernorm.weight": "attn1.norm_q.weight",
            "self_attention.linear_qkv.qx": "attn2.to_q",
            "self_attention.linear_qkv.k": "attn1.to_k",
            "self_attention.linear_qkv.q": "attn1.to_q",
            "self_attention.linear_qkv.v": "attn1.to_v",
            "self_attention.linear_kv_xattn": "attn2.to_kv",
        }
        self.pre_special_keys_map = {}


class FluxTransformerConverter(TransformerConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {
            "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
            "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
            "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
            "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
            "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
            "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
            "txt_in.weight": "context_embedder.weight",
            "txt_in.bias": "context_embedder.bias",
            "img_in.weight": "x_embedder.weight",
            "img_in.bias": "x_embedder.bias",
            "double_blocks": "transformer_blocks",
            "img_mod.lin": "norm1.linear",
            "txt_mod.lin": "norm1_context.linear",
            "img_attn.norm.query_norm.scale": "attn.norm_q.weight",
            "img_attn.norm.key_norm.scale": "attn.norm_k.weight",
            "txt_attn.norm.query_norm.scale": "attn.norm_added_q.weight",
            "txt_attn.norm.key_norm.scale": "attn.norm_added_k.weight",
            "img_mlp.0": "ff.net.0.proj",
            "img_mlp.2": "ff.net.2",
            "txt_mlp.0": "ff_context.net.0.proj",
            "txt_mlp.2": "ff_context.net.2",
            "img_attn.proj": "attn.to_out.0",
            "txt_attn.proj": "attn.to_add_out",
            "modulation.lin": "norm.linear",
            "linear2": "proj_out",
            "final_layer.linear": "proj_out",
        }
        # Not used directly in convert but kept for API parity
        self.pre_special_keys_map = {}
        self.special_keys_map = {}

        # Defaults taken from the Flux dev config used in `test_flux.py`.
        self.num_layers = 19
        self.num_single_layers = 38
        self.inner_dim = 3072
        self.mlp_ratio = 4.0

    def _infer_hyperparams_from_state_dict(self, sd: Dict[str, Any]):
        """
        Infer num_layers, num_single_layers, inner_dim and mlp_ratio directly from the
        checkpoint state_dict when possible, falling back to the defaults otherwise.
        """
        num_layers = None
        num_single_layers = None
        inner_dim = None
        mlp_ratio = None

        double_block_indices = set()
        single_block_indices = set()

        for key, tensor in sd.items():
            # Collect layer indices for double and single blocks
            if key.startswith("double_blocks."):
                m = re.match(r"double_blocks\.(\d+)\.", key)
                if m:
                    double_block_indices.add(int(m.group(1)))
            elif key.startswith("single_blocks."):
                m = re.match(r"single_blocks\.(\d+)\.", key)
                if m:
                    single_block_indices.add(int(m.group(1)))

            # Try to infer inner_dim and mlp_ratio from single_blocks linear1 weight
            if (
                (inner_dim is None or mlp_ratio is None)
                and "single_blocks." in key
                and key.endswith(".linear1.weight")
            ):
                # shape: (3 * inner_dim + mlp_hidden_dim, inner_dim)
                if hasattr(tensor, "shape") and len(tensor.shape) == 2:
                    out_features, in_features = tensor.shape
                    inferred_inner = in_features
                    mlp_hidden_dim = out_features - 3 * inferred_inner
                    if mlp_hidden_dim > 0:
                        inner_dim = inferred_inner
                        mlp_ratio = float(mlp_hidden_dim) / float(inferred_inner)

            # Fallback: infer inner_dim from qkv weights of double blocks
            if (
                inner_dim is None
                and "double_blocks." in key
                and "img_attn.qkv.weight" in key
            ):
                # shape: (3 * inner_dim, query_dim)
                if hasattr(tensor, "shape") and len(tensor.shape) == 2:
                    out_features, _ = tensor.shape
                    inferred_inner = out_features // 3
                    if inferred_inner > 0:
                        inner_dim = inferred_inner

            # Fallback for mlp_ratio using img_mlp weights from double blocks
            if (
                mlp_ratio is None
                and "double_blocks." in key
                and "img_mlp.0.weight" in key
            ):
                # shape: (mlp_hidden_dim, inner_dim)
                if hasattr(tensor, "shape") and len(tensor.shape) == 2:
                    mlp_hidden_dim, maybe_inner = tensor.shape
                    if inner_dim is None:
                        inner_dim = maybe_inner
                    if inner_dim and mlp_hidden_dim > 0:
                        mlp_ratio = float(mlp_hidden_dim) / float(inner_dim)

        if double_block_indices:
            num_layers = max(double_block_indices) + 1
        if single_block_indices:
            num_single_layers = max(single_block_indices) + 1

        # Fallbacks to defaults where inference failed
        if num_layers is None:
            num_layers = self.num_layers
        if num_single_layers is None:
            num_single_layers = self.num_single_layers
        if inner_dim is None:
            inner_dim = self.inner_dim
        if mlp_ratio is None:
            mlp_ratio = self.mlp_ratio

        # Persist inferred values
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.inner_dim = inner_dim
        self.mlp_ratio = mlp_ratio

        return num_layers, num_single_layers, inner_dim, mlp_ratio

    def convert(self, state_dict: Dict[str, Any]):
        """
        Convert a Flux transformer checkpoint to the diffusers format.

        - If a mapping exists and the source key is present, we rename it.
        - If a mapping exists but the source key is missing, we skip it.
        - Any keys that are never touched/mapped are kept with their original names.
        """

        def maybe_assign(
            dst: Dict[str, Any],
            dst_key: str,
            src: Dict[str, Any],
            src_key: str,
            transform=None,
        ):
            """Pop src[src_key] if present and assign to dst[dst_key]."""
            if src_key in src:
                val = src.pop(src_key)
                if transform is not None:
                    val = transform(val)
                dst[dst_key] = val

        def all_keys_present(src: Dict[str, Any], keys):
            return all(k in src for k in keys)

        # Work on a copy so we can freely pop while building a fresh dict.
        # IMPORTANT: Avoid unconditional .pop() to prevent KeyErrors on partial checkpoints.
        original_state_dict = dict(state_dict)
        converted_state_dict: Dict[str, Any] = {}

        num_layers, num_single_layers, inner_dim, mlp_ratio = (
            self._infer_hyperparams_from_state_dict(original_state_dict)
        )

        # ------------------------------------------------------------------
        # time_text_embed.timestep_embedder <- time_in (optional)
        # ------------------------------------------------------------------
        maybe_assign(
            converted_state_dict,
            "time_text_embed.timestep_embedder.linear_1.weight",
            original_state_dict,
            "time_in.in_layer.weight",
        )
        maybe_assign(
            converted_state_dict,
            "time_text_embed.timestep_embedder.linear_1.bias",
            original_state_dict,
            "time_in.in_layer.bias",
        )
        maybe_assign(
            converted_state_dict,
            "time_text_embed.timestep_embedder.linear_2.weight",
            original_state_dict,
            "time_in.out_layer.weight",
        )
        maybe_assign(
            converted_state_dict,
            "time_text_embed.timestep_embedder.linear_2.bias",
            original_state_dict,
            "time_in.out_layer.bias",
        )

        # time_text_embed.text_embedder <- vector_in (optional)
        maybe_assign(
            converted_state_dict,
            "time_text_embed.text_embedder.linear_1.weight",
            original_state_dict,
            "vector_in.in_layer.weight",
        )
        maybe_assign(
            converted_state_dict,
            "time_text_embed.text_embedder.linear_1.bias",
            original_state_dict,
            "vector_in.in_layer.bias",
        )
        maybe_assign(
            converted_state_dict,
            "time_text_embed.text_embedder.linear_2.weight",
            original_state_dict,
            "vector_in.out_layer.weight",
        )
        maybe_assign(
            converted_state_dict,
            "time_text_embed.text_embedder.linear_2.bias",
            original_state_dict,
            "vector_in.out_layer.bias",
        )

        # ------------------------------------------------------------------
        # guidance (optional)
        # ------------------------------------------------------------------
        guidance_keys = [
            "guidance_in.in_layer.weight",
            "guidance_in.in_layer.bias",
            "guidance_in.out_layer.weight",
            "guidance_in.out_layer.bias",
        ]
        if all_keys_present(original_state_dict, guidance_keys):
            maybe_assign(
                converted_state_dict,
                "time_text_embed.guidance_embedder.linear_1.weight",
                original_state_dict,
                "guidance_in.in_layer.weight",
            )
            maybe_assign(
                converted_state_dict,
                "time_text_embed.guidance_embedder.linear_1.bias",
                original_state_dict,
                "guidance_in.in_layer.bias",
            )
            maybe_assign(
                converted_state_dict,
                "time_text_embed.guidance_embedder.linear_2.weight",
                original_state_dict,
                "guidance_in.out_layer.weight",
            )
            maybe_assign(
                converted_state_dict,
                "time_text_embed.guidance_embedder.linear_2.bias",
                original_state_dict,
                "guidance_in.out_layer.bias",
            )

        # ------------------------------------------------------------------
        # context_embedder / x_embedder (optional)
        # ------------------------------------------------------------------
        maybe_assign(
            converted_state_dict,
            "context_embedder.weight",
            original_state_dict,
            "txt_in.weight",
        )
        maybe_assign(
            converted_state_dict,
            "context_embedder.bias",
            original_state_dict,
            "txt_in.bias",
        )
        maybe_assign(
            converted_state_dict,
            "x_embedder.weight",
            original_state_dict,
            "img_in.weight",
        )
        maybe_assign(
            converted_state_dict,
            "x_embedder.bias",
            original_state_dict,
            "img_in.bias",
        )

        # ------------------------------------------------------------------
        # double transformer blocks
        # ------------------------------------------------------------------
        for i in range(num_layers):
            block_prefix = f"transformer_blocks.{i}."

            # norms
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}norm1.linear.weight",
                original_state_dict,
                f"double_blocks.{i}.img_mod.lin.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}norm1.linear.bias",
                original_state_dict,
                f"double_blocks.{i}.img_mod.lin.bias",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}norm1_context.linear.weight",
                original_state_dict,
                f"double_blocks.{i}.txt_mod.lin.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}norm1_context.linear.bias",
                original_state_dict,
                f"double_blocks.{i}.txt_mod.lin.bias",
            )

            # QKV (sample)
            img_qkv_w_key = f"double_blocks.{i}.img_attn.qkv.weight"
            img_qkv_b_key = f"double_blocks.{i}.img_attn.qkv.bias"
            if img_qkv_w_key in original_state_dict:
                sample_q, sample_k, sample_v = ggml_chunk(
                    original_state_dict.pop(img_qkv_w_key),
                    3,
                    dim=0,
                )
                converted_state_dict[f"{block_prefix}attn.to_q.weight"] = ggml_cat(
                    [sample_q]
                )
                converted_state_dict[f"{block_prefix}attn.to_k.weight"] = ggml_cat(
                    [sample_k]
                )
                converted_state_dict[f"{block_prefix}attn.to_v.weight"] = ggml_cat(
                    [sample_v]
                )

            if img_qkv_b_key in original_state_dict:
                sample_q_bias, sample_k_bias, sample_v_bias = ggml_chunk(
                    original_state_dict.pop(img_qkv_b_key),
                    3,
                    dim=0,
                )
                converted_state_dict[f"{block_prefix}attn.to_q.bias"] = ggml_cat(
                    [sample_q_bias]
                )
                converted_state_dict[f"{block_prefix}attn.to_k.bias"] = ggml_cat(
                    [sample_k_bias]
                )
                converted_state_dict[f"{block_prefix}attn.to_v.bias"] = ggml_cat(
                    [sample_v_bias]
                )

            # QKV (context)
            txt_qkv_w_key = f"double_blocks.{i}.txt_attn.qkv.weight"
            txt_qkv_b_key = f"double_blocks.{i}.txt_attn.qkv.bias"
            if txt_qkv_w_key in original_state_dict:
                context_q, context_k, context_v = ggml_chunk(
                    original_state_dict.pop(txt_qkv_w_key),
                    3,
                    dim=0,
                )
                converted_state_dict[
                    f"{block_prefix}attn.add_q_proj.weight"
                ] = ggml_cat([context_q])
                converted_state_dict[
                    f"{block_prefix}attn.add_k_proj.weight"
                ] = ggml_cat([context_k])
                converted_state_dict[
                    f"{block_prefix}attn.add_v_proj.weight"
                ] = ggml_cat([context_v])

            if txt_qkv_b_key in original_state_dict:
                context_q_bias, context_k_bias, context_v_bias = ggml_chunk(
                    original_state_dict.pop(txt_qkv_b_key),
                    3,
                    dim=0,
                )
                converted_state_dict[
                    f"{block_prefix}attn.add_q_proj.bias"
                ] = ggml_cat([context_q_bias])
                converted_state_dict[
                    f"{block_prefix}attn.add_k_proj.bias"
                ] = ggml_cat([context_k_bias])
                converted_state_dict[
                    f"{block_prefix}attn.add_v_proj.bias"
                ] = ggml_cat([context_v_bias])

            # qk_norm
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.norm_q.weight",
                original_state_dict,
                f"double_blocks.{i}.img_attn.norm.query_norm.scale",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.norm_k.weight",
                original_state_dict,
                f"double_blocks.{i}.img_attn.norm.key_norm.scale",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.norm_added_q.weight",
                original_state_dict,
                f"double_blocks.{i}.txt_attn.norm.query_norm.scale",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.norm_added_k.weight",
                original_state_dict,
                f"double_blocks.{i}.txt_attn.norm.key_norm.scale",
            )

            # ff img_mlp
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}ff.net.0.proj.weight",
                original_state_dict,
                f"double_blocks.{i}.img_mlp.0.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}ff.net.0.proj.bias",
                original_state_dict,
                f"double_blocks.{i}.img_mlp.0.bias",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}ff.net.2.weight",
                original_state_dict,
                f"double_blocks.{i}.img_mlp.2.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}ff.net.2.bias",
                original_state_dict,
                f"double_blocks.{i}.img_mlp.2.bias",
            )

            maybe_assign(
                converted_state_dict,
                f"{block_prefix}ff_context.net.0.proj.weight",
                original_state_dict,
                f"double_blocks.{i}.txt_mlp.0.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}ff_context.net.0.proj.bias",
                original_state_dict,
                f"double_blocks.{i}.txt_mlp.0.bias",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}ff_context.net.2.weight",
                original_state_dict,
                f"double_blocks.{i}.txt_mlp.2.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}ff_context.net.2.bias",
                original_state_dict,
                f"double_blocks.{i}.txt_mlp.2.bias",
            )

            # output projections
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.to_out.0.weight",
                original_state_dict,
                f"double_blocks.{i}.img_attn.proj.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.to_out.0.bias",
                original_state_dict,
                f"double_blocks.{i}.img_attn.proj.bias",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.to_add_out.weight",
                original_state_dict,
                f"double_blocks.{i}.txt_attn.proj.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.to_add_out.bias",
                original_state_dict,
                f"double_blocks.{i}.txt_attn.proj.bias",
            )

        # ------------------------------------------------------------------
        # single transformer blocks
        # ------------------------------------------------------------------
        for i in range(num_single_layers):
            block_prefix = f"single_transformer_blocks.{i}."

            # norm.linear <- single_blocks.i.modulation.lin
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}norm.linear.weight",
                original_state_dict,
                f"single_blocks.{i}.modulation.lin.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}norm.linear.bias",
                original_state_dict,
                f"single_blocks.{i}.modulation.lin.bias",
            )

            # Q, K, V, mlp from linear1 (only if both weight & bias exist)
            lin1_w_key = f"single_blocks.{i}.linear1.weight"
            lin1_b_key = f"single_blocks.{i}.linear1.bias"
            if all_keys_present(original_state_dict, [lin1_w_key, lin1_b_key]):
                mlp_hidden_dim = int(inner_dim * mlp_ratio)
                split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)

                q, k, v, mlp = ggml_split(
                    original_state_dict.pop(lin1_w_key),
                    split_size,
                    dim=0,
                )
                q_bias, k_bias, v_bias, mlp_bias = ggml_split(
                    original_state_dict.pop(lin1_b_key),
                    split_size,
                    dim=0,
                )

                converted_state_dict[f"{block_prefix}attn.to_q.weight"] = ggml_cat([q])
                converted_state_dict[f"{block_prefix}attn.to_q.bias"] = ggml_cat(
                    [q_bias]
                )
                converted_state_dict[f"{block_prefix}attn.to_k.weight"] = ggml_cat([k])
                converted_state_dict[f"{block_prefix}attn.to_k.bias"] = ggml_cat(
                    [k_bias]
                )
                converted_state_dict[f"{block_prefix}attn.to_v.weight"] = ggml_cat([v])
                converted_state_dict[f"{block_prefix}attn.to_v.bias"] = ggml_cat(
                    [v_bias]
                )
                converted_state_dict[f"{block_prefix}proj_mlp.weight"] = ggml_cat([mlp])
                converted_state_dict[f"{block_prefix}proj_mlp.bias"] = ggml_cat(
                    [mlp_bias]
                )

            # qk norm
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.norm_q.weight",
                original_state_dict,
                f"single_blocks.{i}.norm.query_norm.scale",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}attn.norm_k.weight",
                original_state_dict,
                f"single_blocks.{i}.norm.key_norm.scale",
            )

            # output projections
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}proj_out.weight",
                original_state_dict,
                f"single_blocks.{i}.linear2.weight",
            )
            maybe_assign(
                converted_state_dict,
                f"{block_prefix}proj_out.bias",
                original_state_dict,
                f"single_blocks.{i}.linear2.bias",
            )

        # ------------------------------------------------------------------
        # final layer (optional)
        # ------------------------------------------------------------------
        maybe_assign(
            converted_state_dict,
            "proj_out.weight",
            original_state_dict,
            "final_layer.linear.weight",
        )
        maybe_assign(
            converted_state_dict,
            "proj_out.bias",
            original_state_dict,
            "final_layer.linear.bias",
        )
        maybe_assign(
            converted_state_dict,
            "norm_out.linear.weight",
            original_state_dict,
            "final_layer.adaLN_modulation.1.weight",
            transform=swap_scale_shift,
        )
        maybe_assign(
            converted_state_dict,
            "norm_out.linear.bias",
            original_state_dict,
            "final_layer.adaLN_modulation.1.bias",
            transform=swap_scale_shift,
        )

        # ------------------------------------------------------------------
        # Merge any remaining (unmapped) original keys, unchanged.
        # These are keys we never popped/renamed above.
        # ------------------------------------------------------------------
        for k, v in original_state_dict.items():
            converted_state_dict[k] = v

        # Mutate input dict in-place to follow the converter API
        state_dict.clear()
        state_dict.update(converted_state_dict)

class NoOpTransformerConverter(TransformerConverter):
    def __init__(self):
        super().__init__()
        self.rename_dict = {}
        self.pre_special_keys_map = {}
        self.special_keys_map = {}


class LoraTransformerConverter(TransformerConverter):
    """
    Thin adapter that delegates to `src.lora.lora_converter.LoraConverter` so it
    can be used from the generic converter utilities in `src.converters.convert`.

    This mirrors the simple `convert(state_dict)` API used by other
    `TransformerConverter` subclasses, but internally relies on the more
    specialized LoRA-to-PEFT conversion logic.
    """

    def __init__(self):
        super().__init__()
        # Import here to avoid any potential import cycles at module import time.
        from src.lora.lora_converter import LoraConverter

        self._lora_converter = LoraConverter()

    def convert(self, state_dict: Dict[str, Any]):
        # Delegate the heavy lifting to the dedicated LoRA converter.
        self._lora_converter.convert(state_dict)
        return state_dict
