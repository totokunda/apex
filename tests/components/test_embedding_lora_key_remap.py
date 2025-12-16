import torch
import torch.nn as nn

from src.lora.key_remap import remap_embedding_lora_keys


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_type_embedding = nn.Embedding(3, 8)
        self.proj = nn.Linear(8, 8, bias=False)


def test_embedding_lora_keys_are_remapped_to_peft_convention_and_swapped():
    """
    Embedding LoRA in PEFT uses:
      - lora_embedding_A.<adapter_name>  (num_embeddings, r)
      - lora_embedding_B.<adapter_name>  (r, embed_dim)

    Many exporters treat embedding weights like linears and emit:
      - lora_A.weight (r, embed_dim)
      - lora_B.weight (num_embeddings, r)

    We remap keys and swap A/B so diffusers' adapter loader can insert the adapter
    name and load into PEFT's Embedding layer wrapper.
    """
    model = _DummyModel()

    emb_A = torch.randn(4, 8)  # (r, embed_dim)
    emb_B = torch.randn(3, 4)  # (num_embeddings, r)
    lin_A = torch.randn(4, 8)
    lin_B = torch.randn(8, 4)

    sd = {
        "cond_type_embedding.lora_A.weight": emb_A,
        "cond_type_embedding.lora_B.weight": emb_B,
        "proj.lora_A.weight": lin_A,
        "proj.lora_B.weight": lin_B,
    }

    out = remap_embedding_lora_keys(sd, model)

    # Embedding keys are rewritten and swapped
    assert "cond_type_embedding.lora_A.weight" not in out
    assert "cond_type_embedding.lora_B.weight" not in out
    assert torch.equal(out["cond_type_embedding.lora_embedding_A"], emb_B.transpose(0, 1))
    assert torch.equal(out["cond_type_embedding.lora_embedding_B"], emb_A.transpose(0, 1))

    # Non-embedding keys are left untouched
    assert torch.equal(out["proj.lora_A.weight"], lin_A)
    assert torch.equal(out["proj.lora_B.weight"], lin_B)


