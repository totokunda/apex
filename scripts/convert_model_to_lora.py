import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple
from tqdm import tqdm
# Allow running this script directly without requiring manual PYTHONPATH setup.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu")

    # common patterns: raw state dict, {"state_dict": ...}, {"model": ...}
    if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
        return sd
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        return sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        return sd["model"]

    raise ValueError(f"Unsupported checkpoint format at {path!r}. Got type={type(sd)}")


def _strip_checkpoint_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace("._fsdp_wrapped_module", "")
        nk = nk.replace("model.", "")
        out[nk] = v
    return out


def _iter_linear_weight_keys(model: nn.Module) -> Iterable[str]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield f"{name}.weight"


def _lowrank_lora_factors(
    delta_w: torch.Tensor,
    rank: int,
    *,
    method: str = "auto",
    exact_max_elements: int = 2_000_000,
    exact_max_rank: int = 2048,
    oversample: int = 8,
    niter: int = 2,
    svd_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute LoRA factors A (rank x in) and B (out x rank) such that:
        delta_w ~= B @ A
    using either exact truncated SVD (optimal) or randomized low-rank SVD (fast).
    """
    if delta_w.ndim != 2:
        raise ValueError(f"delta_w must be 2D, got shape={tuple(delta_w.shape)}")

    out_dim, in_dim = delta_w.shape
    max_rank = min(out_dim, in_dim)
    r = int(min(rank, max_rank))
    if r <= 0:
        raise ValueError(f"rank must be >= 1, got {rank}")

    method = str(method).lower()
    if method not in {"auto", "exact", "randomized"}:
        raise ValueError("method must be one of: auto, exact, randomized")

    # Do the decomposition in a numerically-friendly dtype (often float64 on CPU)
    x = delta_w.to(dtype=svd_dtype)

    use_exact = method == "exact"
    if method == "auto":
        # Exact SVD is very expensive for large matrices; gate it by size.
        # (elements is a rough proxy; exact_max_rank limits the cubic term)
        elems = int(out_dim) * int(in_dim)
        use_exact = elems <= int(exact_max_elements) and max_rank <= int(exact_max_rank)

    if use_exact:
        # torch.linalg.svd returns U (m x k), S (k,), Vh (k x n)
        # where k = min(m, n). Truncation to top-r gives the optimal rank-r approx.
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]  # (r, in_dim)
        B = (U_r * S_r.unsqueeze(0)).to(dtype=torch.float32)  # (out_dim, r)
        A = Vh_r.to(dtype=torch.float32).contiguous()  # (r, in_dim)
    else:
        # torch.svd_lowrank returns U (m x q), S (q,), V (n x q) such that x ~= U diag(S) V^T
        q = min(max_rank, r + int(oversample))
        U, S, V = torch.svd_lowrank(x, q=q, niter=int(niter))
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = V[:, :r]  # (in_dim, r)
        B = (U_r * S_r.unsqueeze(0)).to(dtype=torch.float32)  # (out_dim, r)
        A = V_r.transpose(0, 1).to(dtype=torch.float32).contiguous()  # (r, in_dim)
    return A, B


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a LoRA state dict by factoring (new - base) deltas for all nn.Linear weights."
    )
    parser.add_argument(
        "--base_model",
        default="/home/tosin_coverquick_co/apex/Wan-22-TI2V-Base",
        help="Path or HF repo for the base model.",
    )
    parser.add_argument(
        "--base_subfolder",
        default="transformer",
        help="Subfolder under base_model containing the transformer weights/config.",
    )
    parser.add_argument(
        "--new_ckpt",
        default="/home/tosin_coverquick_co/apex-diffusion/components/af18e595fc128bba84f88a92bd6e26bff7fb6e27659ab1846547518b75d2d3bb_Wan2_2-TI2V-5B-Turbo_fp16.safetensors",
        help="Path to the fine-tuned/new checkpoint (pt/bin/safetensors).",
    )
    parser.add_argument(
        "--out",
        default="/home/tosin_coverquick_co/apex/lora_delta.safetensors",
        help="Output LoRA file path (.safetensors recommended).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help="LoRA rank to approximate each delta with.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="LoRA alpha. If omitted, defaults to rank (so scaling alpha/rank = 1).",
    )
    parser.add_argument(
        "--svd_method",
        choices=["auto", "exact", "randomized"],
        default="auto",
        help="SVD method for factoring deltas. exact is most accurate but can be very slow.",
    )
    parser.add_argument(
        "--svd_dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Dtype used during SVD computation (factors are saved as float32). float64 can improve accuracy.",
    )
    parser.add_argument(
        "--svd_oversample",
        type=int,
        default=8,
        help="Randomized SVD oversampling (only used for svd_method=randomized/auto-fallback).",
    )
    parser.add_argument(
        "--svd_niter",
        type=int,
        default=2,
        help="Randomized SVD power iterations (only used for svd_method=randomized/auto-fallback).",
    )
    parser.add_argument(
        "--svd_exact_max_elements",
        type=int,
        default=2_000_000,
        help="In auto mode, only use exact SVD if out_dim*in_dim <= this threshold.",
    )
    parser.add_argument(
        "--svd_exact_max_rank",
        type=int,
        default=2048,
        help="In auto mode, only use exact SVD if min(out_dim,in_dim) <= this threshold.",
    )
    parser.add_argument(
        "--skip_if_missing",
        action="store_true",
        help="Skip layers missing in new checkpoint instead of erroring.",
    )
    parser.add_argument(
        "--include_regex",
        default=None,
        help="Optional regex; only process linear weights whose key matches.",
    )
    parser.add_argument(
        "--exclude_regex",
        default=None,
        help="Optional regex; skip linear weights whose key matches.",
    )
    args = parser.parse_args()

    svd_dtype = torch.float64 if args.svd_dtype == "float64" else torch.float32

    # Imported lazily so `--help` can work even if optional deps aren't installed.
    from src.converters.transformer_converters import WanTransformerConverter
    from src.transformer.wan.base.model import WanTransformer3DModel

    # 1) Load base model (diffusers-format) so we can enumerate nn.Linear params reliably.
    base_model = WanTransformer3DModel.from_pretrained(
        args.base_model, subfolder=args.base_subfolder
    )
    base_sd = base_model.state_dict()

    # 2) Load and normalize new checkpoint keys; convert into the same key-space as base.
    raw_new_sd = _load_state_dict(args.new_ckpt)
    new_sd = _strip_checkpoint_prefixes(raw_new_sd)
    WanTransformerConverter().convert(new_sd)

    # 3) Build LoRA dict
    lora_sd: Dict[str, torch.Tensor] = {}
    include_re = None
    exclude_re = None
    if args.include_regex:
        import re

        include_re = re.compile(args.include_regex)
    if args.exclude_regex:
        import re

        exclude_re = re.compile(args.exclude_regex)

    processed = 0
    skipped_missing = 0
    skipped_shape = 0
    skipped_zero = 0
    max_rel_err = 0.0

    for w_key in tqdm(_iter_linear_weight_keys(base_model)):
        if include_re and not include_re.search(w_key):
            continue
        if exclude_re and exclude_re.search(w_key):
            continue

        if w_key not in base_sd:
            continue
        if w_key not in new_sd:
            if args.skip_if_missing:
                skipped_missing += 1
                continue
            raise KeyError(
                f"Missing {w_key!r} in new checkpoint after conversion. "
                f"Tip: use --skip_if_missing or adjust --include_regex/--exclude_regex."
            )

        w_base = base_sd[w_key]
        w_new = new_sd[w_key]
        if w_base.shape != w_new.shape or w_base.ndim != 2:
            skipped_shape += 1
            continue

        delta = (w_new.to(torch.float32) - w_base.to(torch.float32)).contiguous()
        if torch.count_nonzero(delta).item() == 0:
            skipped_zero += 1
            continue

        # Determine the effective rank for this layer (can't exceed min(out,in)).
        out_dim, in_dim = int(delta.shape[0]), int(delta.shape[1])
        eff_rank = int(min(args.rank, out_dim, in_dim))
        if eff_rank <= 0:
            skipped_shape += 1
            continue

        # The loader applies: w = w_base + (alpha/eff_rank) * (B @ A)
        # If alpha is omitted, we set alpha=eff_rank per-layer so scaling==1 and
        # the LoRA can best match the original delta at that rank.
        alpha_layer = float(args.alpha) if args.alpha is not None else float(eff_rank)
        scale = alpha_layer / float(eff_rank)
        target = delta if scale == 1.0 else (delta / scale)
        A, B = _lowrank_lora_factors(
            target,
            rank=args.rank,
            method=args.svd_method,
            exact_max_elements=args.svd_exact_max_elements,
            exact_max_rank=args.svd_exact_max_rank,
            oversample=args.svd_oversample,
            niter=args.svd_niter,
            svd_dtype=svd_dtype,
        )

        # store in the format expected by src/lora/manager.py:
        #   <param>.lora_A.weight  (rank, in)
        #   <param>.lora_B.weight  (out, rank)
        #   <param>.alpha          scalar
        lora_a_key = w_key.replace(".weight", ".lora_A.weight")
        lora_b_key = w_key.replace(".weight", ".lora_B.weight")
        lora_alpha_key = w_key.replace(".weight", ".alpha")

        lora_sd[lora_a_key] = A.to(torch.float32)
        lora_sd[lora_b_key] = B.to(torch.float32)
        lora_sd[lora_alpha_key] = torch.tensor(alpha_layer, dtype=torch.float32)

        # quick reconstruction check (scaled like loader does)
        scale = alpha_layer / float(A.shape[0])
        approx = scale * (B @ A)
        rel_err = (approx - delta).norm() / (delta.norm() + 1e-12)
        max_rel_err = max(max_rel_err, float(rel_err))

        processed += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.endswith(".safetensors"):
        save_file(lora_sd, args.out)
    else:
        torch.save(lora_sd, args.out)

    print(
        "Done.\n"
        f"- wrote: {args.out}\n"
        f"- processed linear weights: {processed}\n"
        f"- skipped (missing in new): {skipped_missing}\n"
        f"- skipped (shape/non-2d mismatch): {skipped_shape}\n"
        f"- skipped (zero delta): {skipped_zero}\n"
        f"- worst relative reconstruction error (scaled): {max_rel_err:.6f}\n"
        + (
            f"- alpha={float(args.alpha)} rank={args.rank} (scaling alpha/rank = {float(args.alpha)/float(args.rank):.6f})\n"
            if args.alpha is not None
            else f"- alpha=per-layer effective rank (scaling = 1.0)\n"
        )
    )


if __name__ == "__main__":
    main()
