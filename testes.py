import torch
import pickle
import sys
from src.transformer.hunyuan.avatar.model import HunyuanAvatarVideoTransformer3DModel
from torch.distributed import init_process_group, destroy_process_group

init_process_group(backend="nccl")
sys.path.append('/workspace/HunyuanVideo-Avatar')
from src.attention import attention_register
attention_register.set_default("flash")

from hymm_sp.modules.models_audio import HYVideoDiffusionTransformer
from hymm_sp.modules import load_model

print("Loading Module...")
module = torch.load("/workspace/apex/hy_debug.pt", map_location="cuda")


print("Loading Model A...")
model_a = HunyuanAvatarVideoTransformer3DModel.from_pretrained(
    "/workspace/models/components/tencent/HunyuanVideo-Avatar/resolve/main/ckpts/hunyuan-video-t2v-720p/transformers/transformer",
    torch_dtype=torch.bfloat16,
    device_map={
        "": "cuda"
    }
)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    print("Running Model A...")
    vec_aa = model_a(
        module["latent_model_input"],
        module["t_expand"],
        encoder_hidden_states=module["prompt_embeds_input"],
        encoder_attention_mask=module["prompt_mask_input"],
        pooled_projections=module["prompt_embeds_2_input"],
        ref_latents=module["ref_latents"],
        freqs_cos=module["freqs_cos"],
        freqs_sin=module["freqs_sin"],
        guidance=None,
        return_dict=True, 
        use_cache=module["is_cache"],
        encoder_hidden_states_motion=module["motion_exp_input"],
        encoder_hidden_states_pose=module["motion_pose_input"],
        encoder_hidden_states_fps=module["fps_input"],
        encoder_hidden_states_audio=module["audio_prompts_input"],
        encoder_hidden_states_face_mask=module["face_masks_input"],
    )
    print("Model A done!")
    # offload 
    model_a.to("cpu")
    del model_a
    torch.cuda.empty_cache()
    print("Model A memory freed!")


print("Loading Model B...")
with open('/workspace/apex/args.pkl', 'rb') as f:
    model_kwargs = pickle.load(f)
model_b = load_model(**model_kwargs)
model_b.load_state_dict(torch.load('/workspace/HunyuanVideo-Avatar/weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt')['module'])
model_b.to("cuda")

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    print("Running Model B...")
    vec_b = model_b(
        module["latent_model_input"],
        module["t_expand"],
        ref_latents=module["ref_latents"],
        text_states=module["prompt_embeds_input"],
        text_mask=module["prompt_mask_input"],
        text_states_2=module["prompt_embeds_2_input"],
        freqs_cos=module["freqs_cos"],
        freqs_sin=module["freqs_sin"],
        guidance=None,
        return_dict=True, 
        is_cache=module["is_cache"],
        motion_exp=module["motion_exp_input"],
        motion_pose=module["motion_pose_input"],
        fps=module["fps_input"],
        audio_prompts=module["audio_prompts_input"],
        face_mask=module["face_masks_input"],
    )

print("Comparing results...")
for va, vb in zip(vec_aa, vec_b):
    try:
        print("Testing", va.shape, vb.shape)
        torch.testing.assert_close(va, vb, atol=1e-4, rtol=1e-4)
    except Exception as e:
        print(e)


destroy_process_group()
print("Done!")