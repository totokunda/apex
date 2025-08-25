from typing import Dict, Any
from src.converters.utils import update_state_dict_, swap_proj_gate, swap_scale_shift
import torch
import re


class TransformerConverter:
    def __init__(self):
        self.rename_dict = {}
        self.special_keys_map = {}
        self.pre_special_keys_map = {}

    def _sort_rename_dict(self):
        """Sort rename_dict by value length from longest to shortest to ensure proper replacement order."""
        self.rename_dict = dict(
            sorted(
                self.rename_dict.items(), key=lambda item: len(item[0]), reverse=True
            )
        )

    def convert(self, state_dict: Dict[str, Any]):
        self._sort_rename_dict()
        # Apply pre-special keys map
        for key in list(state_dict.keys()):
            for pre_special_key, handler_fn_inplace in self.pre_special_keys_map.items():
                if pre_special_key in key:
                    handler_fn_inplace(key, state_dict)
                    
                    
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


class WanVaceTransformerConverter(WanTransformerConverter):
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

    @staticmethod
    def reassign_query_key_value_inplace(self, key: str, state_dict: Dict[str, Any]):
        to_q_key = key.replace("query_key_value", "to_q")
        to_k_key = key.replace("query_key_value", "to_k")
        to_v_key = key.replace("query_key_value", "to_v")
        to_q, to_k, to_v = torch.chunk(state_dict[key], chunks=3, dim=0)
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
        norm1_weights_or_biases = torch.cat(
            weights_or_biases[0:3] + weights_or_biases[6:9]
        )
        norm2_weights_or_biases = torch.cat(
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

    @staticmethod
    def remap_norm_scale_shift_(key, state_dict):
        weight = state_dict.pop(key)
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
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
            q, k, v, mlp = torch.split(linear1_weight, split_size, dim=0)
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
            q_bias, k_bias, v_bias, mlp_bias = torch.split(
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
        
        self.pre_special_keys_map = {
            "lora_unet_single_blocks_": self.remap_single_transformer_blocks_lora,
            "lora_unet_double_blocks_": self.remap_double_transformer_blocks_lora,
        }
        
        self.special_keys_map = {
            "img_attn_qkv": self.remap_img_attn_qkv_,
            "txt_attn_qkv": self.remap_txt_attn_qkv_,
            "single_blocks": self.remap_single_transformer_blocks_,
            "final_layer.adaLN_modulation.1": self.remap_norm_scale_shift_,
        }
    
    @staticmethod
    def remap_norm_scale_shift_(key, state_dict):
        weight = state_dict.pop(key)
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        state_dict[key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")] = (
            new_weight
        )
    
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
    def remap_single_transformer_blocks_lora(key, state_dict):
        # Normalize odd markers like "(3)" that can appear in some dumps
        clean_key = re.sub(r"\(\d+\)", "", key)

        # Expected formats (examples):
        #   lora_unet_single_blocks_0_linear1.lora_down.weight
        #   lora_unet_single_blocks_0_linear2.lora_up.weight
        #   lora_unet_single_blocks_0_modulation_lin.alpha
        m = re.match(r"^lora_unet_single_blocks_(\d+)_(.+)$", clean_key)
        if not m:
            return

        block_idx, rest = m.groups()

        # Convert specific tokens, but keep component names intact
        # e.g. modulation_lin -> modulation.lin
        rest = re.sub(r"^modulation_lin", "modulation.lin", rest)

        new_key = f"single_blocks.{block_idx}.{rest}"
        state_dict[new_key] = state_dict.pop(key)
    
    @staticmethod
    def remap_double_transformer_blocks_lora(key, state_dict):
        # Normalize odd markers like "(3)" that can appear in some dumps
        clean_key = re.sub(r"\(\d+\)", "", key)

        # Examples to handle:
        #   lora_unet_double_blocks_0_img_attn_proj.lora_down.weight
        #   lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight
        #   lora_unet_double_blocks_0_img_mlp_0.lora_down.weight
        #   lora_unet_double_blocks_0_img_mlp_2.lora_up.weight
        #   lora_unet_double_blocks_0_txt_attn_proj.alpha
        #   lora_unet_double_blocks_0_txt_attn_qkv.lora_down.weight
        #   lora_unet_double_blocks_0_txt_mlp_0.lora_up.weight
        m = re.match(r"^lora_unet_double_blocks_(\d+)_(.+)$", clean_key)
        if not m:
            return

        block_idx, rest = m.groups()

        # Map the first token(s) precisely so rename_dict matches later:
        #   img_attn_proj -> img_attn.proj
        #   img_attn_qkv  -> img_attn.qkv
        #   txt_attn_proj -> txt_attn.proj
        #   txt_attn_qkv  -> txt_attn.qkv
        #   img_mlp_0     -> img_mlp.0
        #   img_mlp_2     -> img_mlp.2
        #   txt_mlp_0     -> txt_mlp.0
        #   txt_mlp_2     -> txt_mlp.2
        #   img_mod_lin   -> img_mod.lin
        #   txt_mod_lin   -> txt_mod.lin
        substitutions = [
            (r"^img_attn_proj", "img_attn.proj"),
            (r"^img_attn_qkv", "img_attn.qkv"),
            (r"^txt_attn_proj", "txt_attn.proj"),
            (r"^txt_attn_qkv", "txt_attn.qkv"),
            (r"^img_mlp_([02])", r"img_mlp.\1"),
            (r"^txt_mlp_([02])", r"txt_mlp.\1"),
            (r"^img_mod_lin", "img_mod.lin"),
            (r"^txt_mod_lin", "txt_mod.lin"),
        ]

        for pattern, repl in substitutions:
            rest = re.sub(pattern, repl, rest)

        new_key = f"double_blocks.{block_idx}.{rest}"
        state_dict[new_key] = state_dict.pop(key)

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
            q, k, v, mlp = torch.split(linear1_weight, split_size, dim=0)
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
            q_bias, k_bias, v_bias, mlp_bias = torch.split(
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
            new_key = new_key.replace("modulation.lin", "norm.linear")
            new_key = new_key.replace("norm.query_norm.scale", "attn.norm_q.weight")
            new_key = new_key.replace("norm.key_norm.scale", "attn.norm_k.weight")
            state_dict[new_key] = state_dict.pop(key)