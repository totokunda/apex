import pickle 
import torch 
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

f1 = pickle.load(open("scheduler_debug.pkl", "rb"))
f2 = pickle.load(open("integrate_3cfg.pkl", "rb"))

m1 = f1['latents_chunk']
m2 = f2['x_chunk']

torch.testing.assert_close(m1, m2, atol=1e-4, rtol=1e-1)





