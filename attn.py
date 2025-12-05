import math
import time

import torch
import torch.nn.functional as F

from sageattention import sageattn  # make sure you have the version you expect


def sdpa_ref(q, k, v, is_causal=False):
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )


def sdpa_sage(q, k, v, is_causal=False, tensor_layout="HND"):
    """
    SageAttention call.
    - v1.x:  sageattn(q, k, v, tensor_layout="HND", is_causal=False, smooth_k=True)
    - v2.2:  sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    We only pass the common args and leave smooth_k to default if present.
    """
    return sageattn(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal)


def max_abs_rel(a, b):
    diff = (a - b).float().abs()
    max_abs = diff.max().item()

    denom = b.float().abs().clamp_min(1e-8)
    max_rel = (diff / denom).max().item()
    return max_abs, max_rel


def cos_sim(a, b):
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return F.cosine_similarity(a_flat, b_flat, dim=0).item()


def run_forward_backward_test(
    batch_size=2,
    n_heads=8,
    seq_len=128,
    head_dim=64,
    dtype=torch.float16,
    device="cuda",
    is_causal=False,
):
    print("=== Forward / backward equivalence test ===")
    print(f"shape: (B={batch_size}, H={n_heads}, L={seq_len}, D={head_dim}), "
          f"dtype={dtype}, device={device}, causal={is_causal}")

    torch.manual_seed(0)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Clone for both paths
    q1 = q.clone().detach().requires_grad_(True)
    k1 = k.clone().detach().requires_grad_(True)
    v1 = v.clone().detach().requires_grad_(True)

    q2 = q.clone().detach().requires_grad_(True)
    k2 = k.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)

    out_ref = sdpa_ref(q1, k1, v1, is_causal=is_causal)
    out_sage = sdpa_sage(q2, k2, v2, is_causal=is_causal)

    max_abs, max_rel = max_abs_rel(out_ref, out_sage)
    cs = cos_sim(out_ref, out_sage)
    print(f"Forward max abs diff : {max_abs:.3e}")
    print(f"Forward max rel diff : {max_rel:.3e}")
    print(f"Forward cosine sim   : {cs:.6f}")



def benchmark(fn, iters=100, device="cuda", label="fn"):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.time() - t0) * 1000.0 / iters
    print(f"[{label}] {dt_ms:.3f} ms / call over {iters} iters")
    return dt_ms


def run_speed_test(
    batch_size=4,
    n_heads=16,
    seq_len=1024,
    head_dim=64,
    dtype=torch.float16,
    device="cuda",
    is_causal=False,
    iters=200,
):
    print("\n=== Speed test ===")
    print(f"shape: (B={batch_size}, H={n_heads}, L={seq_len}, D={head_dim}), "
          f"dtype={dtype}, device={device}, causal={is_causal}")

    torch.manual_seed(1)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def run_ref():
        with torch.no_grad():
            return sdpa_ref(q, k, v, is_causal=is_causal)

    def run_sage():
        with torch.no_grad():
            return sdpa_sage(q, k, v, is_causal=is_causal)

    # warmup
    for _ in range(10):
        run_ref()
        run_sage()
    if device == "cuda":
        torch.cuda.synchronize()

    benchmark(run_ref, iters=iters, device=device, label="torch SDPA")
    benchmark(run_sage, iters=iters, device=device, label="SageAttention")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Torch version:", torch.__version__)
    try:
        import sageattention as sa_mod
        print("SageAttention version:", getattr(sa_mod, "__version__", "unknown"))
    except Exception as e:
        print("Could not query SageAttention version:", e)

    run_forward_backward_test(
        batch_size=2,
        n_heads=8,
        seq_len=128,
        head_dim=64,
        dtype=dtype,
        device=device,
        is_causal=False,
    )

    run_speed_test(
        batch_size=4,
        n_heads=16,
        seq_len=1024,
        head_dim=64,
        dtype=dtype,
        device=device,
        is_causal=False,
        iters=200,
    )
