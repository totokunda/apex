# bench_qkv_fusion.py
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

def maybe_set_sdp(mode: str):
    if not torch.cuda.is_available():
        return
    try:
        if mode == "flash":
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        elif mode == "mem_efficient":
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
        elif mode == "math":
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        elif mode == "auto":
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
        else:
            raise ValueError(mode)
    except Exception as e:
        print(f"[warn] could not set sdp_kernel({mode}): {e}")

class MHA_FusedQKV(nn.Module):
    """x -> one Linear to (qkv) then split"""
    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.to_qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,S,C]
        B, S, C = x.shape
        qkv = self.to_qkv(x)  # [B,S,3C]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B,H,S,D]
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)  # [B,H,S,D]
        y = y.transpose(1, 2).contiguous().view(B, S, C)
        return self.out(y)

class MHA_SeparateQKV(nn.Module):
    """x -> three Linears: to_q, to_k, to_v"""
    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.to_q = nn.Linear(d_model, d_model, bias=bias)
        self.to_k = nn.Linear(d_model, d_model, bias=bias)
        self.to_v = nn.Linear(d_model, d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, S, C)
        return self.out(y)

def copy_weights_fused_to_separate(fused: MHA_FusedQKV, sep: MHA_SeparateQKV):
    """
    Make them numerically comparable:
    - fused.to_qkv weight is [3C, C], split into q/k/v
    - fused.to_qkv bias is [3C], split into q/k/v
    """
    with torch.no_grad():
        W = fused.to_qkv.weight  # [3C, C]
        b = fused.to_qkv.bias    # [3C] or None

        C = fused.d_model
        sep.to_q.weight.copy_(W[0:C, :])
        sep.to_k.weight.copy_(W[C:2*C, :])
        sep.to_v.weight.copy_(W[2*C:3*C, :])

        if b is not None:
            sep.to_q.bias.copy_(b[0:C])
            sep.to_k.bias.copy_(b[C:2*C])
            sep.to_v.bias.copy_(b[2*C:3*C])

        sep.out.weight.copy_(fused.out.weight)
        if fused.out.bias is not None:
            sep.out.bias.copy_(fused.out.bias)

@torch.no_grad()
def bench_forward(mod: nn.Module, x: torch.Tensor, iters: int, warmup: int) -> float:
    # warmup
    for _ in range(warmup):
        y = mod(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = mod(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters  # ms/iter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dmodel", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--minS", type=int, default=256)
    ap.add_argument("--maxS", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--sdp", default="auto", choices=["auto", "flash", "mem_efficient", "math"])
    ap.add_argument("--compile", action="store_true", help="torch.compile both modules (if available)")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        print("[warn] CPU with bf16/fp16 may be unsupported; using fp32")
        dtype = torch.float32

    maybe_set_sdp(args.sdp)

    fused = MHA_FusedQKV(args.dmodel, args.heads, bias=args.bias).to(device=device, dtype=dtype).eval()
    sep   = MHA_SeparateQKV(args.dmodel, args.heads, bias=args.bias).to(device=device, dtype=dtype).eval()
    copy_weights_fused_to_separate(fused, sep)

    if args.compile and hasattr(torch, "compile"):
        fused = torch.compile(fused)
        sep = torch.compile(sep)

    # log-spaced sequence lengths
    seqs = []
    if args.steps == 1:
        seqs = [args.maxS]
    else:
        for i in range(args.steps):
            t = i / (args.steps - 1)
            s = int(round(args.minS * ((args.maxS / args.minS) ** t)))
            s = max(args.minS, min(args.maxS, s))
            if not seqs or s != seqs[-1]:
                seqs.append(s)

    print(f"Device={device}, dtype={dtype}, B={args.batch}, C={args.dmodel}, H={args.heads}, SDPA={args.sdp}, compile={args.compile}")
    print(f"{'S':>6} | {'fused_qkv(ms)':>13} | {'separate(ms)':>12} | {'speedup':>8}")
    print("-" * 52)

    for S in seqs:
        x = torch.randn(args.batch, S, args.dmodel, device=device, dtype=dtype)

        t_fused = bench_forward(fused, x, iters=args.iters, warmup=args.warmup)
        t_sep   = bench_forward(sep,   x, iters=args.iters, warmup=args.warmup)

        speedup = t_sep / t_fused
        print(f"{S:6d} | {t_fused:13.3f} | {t_sep:12.3f} | {speedup:7.2f}x")

    if device.type == "cuda":
        try:
            print("\nSDPA kernel availability:")
            print("  flash:", torch.backends.cuda.flash_sdp_enabled())
            print("  mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
            print("  math:", torch.backends.cuda.math_sdp_enabled())
        except Exception as e:
            print("[warn] could not query SDPA kernel availability:", e)

if __name__ == "__main__":
    main()
