from src.transformer.magi.base.model import MagiTransformer3DModel
import sys
import pickle
import torch
from src.attention import attention_register

from torch.distributed import destroy_process_group

sys.path.append("/workspace/apex/MAGI-1")
from inference.infra.distributed import dist_init
from inference.model.dit import get_dit
from einops import rearrange
from inference.common import MagiConfig
from inference.common import (
    InferenceParams,
    MagiConfig,
    ModelMetaArgs,
    PackedCoreAttnParams,
    PackedCrossAttnParams
)

config = MagiConfig.from_json(
    "/workspace/apex/MAGI-1/example/4.5B/4.5B_base_config.json"
)

dist_init(config)
attention_register.set_default('flash')

inputs = pickle.load(open("/workspace/apex/forward_3cfg.pkl", "rb"))
modela: MagiTransformer3DModel = (
    MagiTransformer3DModel.from_pretrained(
        "/workspace/models/components/sand-ai_MAGI-1_ckpt_magi_4.5B_base_inference_weight/ckpt/magi/4.5B_base/inference_weight/transformer",
        torch_dtype=torch.bfloat16,
    )
    .cuda()
    .eval()
)

for name, param in modela.named_parameters():
    if "attn2.norm_q" in name or "attn2.norm_k" in name:
        # make param bfloat16
        param.data = param.data.to(torch.bfloat16)

modelb = get_dit(config).eval()

for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.cuda()

out = inputs.pop("out_cond_pre_and_text")
x = inputs.pop("x")
t = inputs.pop("timestep")
y = inputs.pop("y")
caption_dropout_mask = inputs.pop("caption_dropout_mask")
xattn_mask = inputs.pop("xattn_mask")
kv_range = inputs.pop("kv_range")
inference_params: InferenceParams = inputs.pop("inference_params")

dtype = torch.bfloat16

kv_cache_params_a = {
    "max_sequence_length": inference_params.max_sequence_length,
    "max_batch_size": inference_params.max_batch_size,
    "key_value_memory_dict": inference_params.key_value_memory_dict,
    "sequence_length_offset": inference_params.sequence_len_offset,
}

with torch.no_grad():
    out_b = modelb(
        x=x,
        t=t,
        y=y,
        xattn_mask=xattn_mask,
        caption_dropout_mask=caption_dropout_mask,
        kv_range=kv_range,
        inference_params=inference_params,
        **inputs
    )
    
    out_a = modela(
        hidden_states=x,
        timestep=t,
        encoder_hidden_states=y,
        encoder_hidden_states_mask=xattn_mask,
        caption_dropout_mask=caption_dropout_mask,
        kv_cache_params=kv_cache_params_a,
        kv_range=kv_range,
        return_dict=False,
        **inputs
    )[0]
    
    torch.testing.assert_close(out_a, out_b, atol=1e-4, rtol=1e-1)
    exit()

    x = x * config.model_config.x_rescale_factor
    x = x.float()
    t = t.float()
    y = y.float()

    with torch.autocast(device_type="cuda", dtype=torch.float32):
        (
            hidden_states_a,
            condition_a,
            condition_map_a,
            rotary_pos_emb_a,
            encoder_hidden_states_flat_a,
            H_a,
            W_a,
            kv_cache_params_meta_a,
            cross_attn_params_a,
            self_attn_params_a,
        ) = modela.get_embedding_and_meta(
            x, t, y, caption_dropout_mask, xattn_mask, kv_range, **inputs
        )
    
    
    
    kv_cache_params_a.update(kv_cache_params_meta_a)
    hidden_states_a = hidden_states_a.to(dtype)
    hidden_states_a = rearrange(hidden_states_a, "N C T H W -> (T H W) N C").contiguous()  # (thw, N, D)
    # condition and y_xattn_flat will be downcast to bfloat16 in transformer block.
    condition_a = condition_a.to(dtype)
    encoder_hidden_states_flat_a = encoder_hidden_states_flat_a.to(dtype)
    block_hidden_states_a = hidden_states_a.clone()
    
    for block in modela.blocks:
        block_hidden_states_a = block(
            hidden_states=block_hidden_states_a,
            condition=condition_a,
            condition_map=condition_map_a,
            rotary_pos_emb=rotary_pos_emb_a,
            encoder_hidden_states=encoder_hidden_states_flat_a,
            self_attn_params=self_attn_params_a,
            cross_attn_params=cross_attn_params_a,
            kv_cache_params=kv_cache_params_a,
        )
        
    block_hidden_states_a = block_hidden_states_a.to(torch.float32)
    out_a = modela.norm_out(block_hidden_states_a)
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        out_a = modela.proj_out(out_a)
    

    with torch.autocast(device_type="cuda", dtype=torch.float32):
        (
            x_b,
            condition_b,
            condition_map_b,
            rotary_pos_emb_b,
            y_xattn_flat_b,
            xattn_mask_for_cuda_graph_b,
            H_b,
            W_b,
            ardf_meta_b,
            cross_attn_params_b,
        ) = modelb.get_embedding_and_meta(
            x, t, y, caption_dropout_mask, xattn_mask, kv_range, **inputs
        )
    
    x_b = x_b.to(dtype)
    x_b = rearrange(x_b, "N C T H W -> (T H W) N C").contiguous()  # (thw, N, D)
    # condition and y_xattn_flat will be downcast to bfloat16 in transformer block.
    condition_b = condition_b.to(dtype)
    y_xattn_flat_b = y_xattn_flat_b.to(dtype)
    core_attn_params = PackedCoreAttnParams(
        q_range=ardf_meta_b["q_range"],
        k_range=ardf_meta_b["k_range"],
        np_q_range=ardf_meta_b["q_range"].cpu().numpy(),
        np_k_range=ardf_meta_b["k_range"].cpu().numpy(),
        max_seqlen_q=ardf_meta_b["max_seqlen_q"],
        max_seqlen_k=ardf_meta_b["max_seqlen_k"],
    )
    
    meta_args = ModelMetaArgs(
        H=H_b,
        W=W_b,
        cp_pad_size=None,
        cp_split_sizes=None,
        slice_point=ardf_meta_b["slice_point"],
        denoising_range_num=ardf_meta_b["denoising_range_num"],
        range_num=ardf_meta_b["range_num"],
        extract_prefix_video_feature=inputs.get("extract_prefix_video_feature", False),
        fwd_extra_1st_chunk=inputs["fwd_extra_1st_chunk"],
        distill_nearly_clean_chunk=inputs.get("distill_nearly_clean_chunk", False),
        clip_token_nums=ardf_meta_b["clip_token_nums"],
        enable_cuda_graph=False,
        core_attn_params=core_attn_params,
        cross_attn_params=cross_attn_params_b,
    )
    
    out_b = modelb.videodit_blocks(
        hidden_states=x_b,
        condition=condition_b,
        condition_map=condition_map_b,
        rotary_pos_emb=rotary_pos_emb_b,
        y_xattn_flat=y_xattn_flat_b,
        meta_args=meta_args,
        inference_params=inference_params,
    )
    
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        out_b = modelb.final_linear(out_b)
    
    print(torch.allclose(hidden_states_a, x_b))
    print(torch.allclose(condition_a, condition_b))
    print(torch.allclose(condition_map_a, condition_map_b))
    print(torch.allclose(rotary_pos_emb_a, rotary_pos_emb_b))
    print(torch.allclose(encoder_hidden_states_flat_a, y_xattn_flat_b))
    torch.testing.assert_close(out_a, out_b, atol=1e-4, rtol=1e-1)
        

destroy_process_group()