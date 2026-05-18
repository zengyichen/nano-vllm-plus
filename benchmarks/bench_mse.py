"""Direct KV cache quantization error measurement (L1 benchmark).

Measures MSE, MAE, and cosine similarity between original and dequantized
K/V tensors. No model inference needed — uses random tensors with realistic shapes.
"""

import argparse
import json

import torch

from nanovllm.utils.quant import (
    TurboQuantMSEKVQuantizer,
    TurboQuantProdKVQuantizer,
    AsymTurboQuantKVQuantizer,
)

NUM_KV_HEADS = 8
HEAD_DIMS = [64, 96, 128]
NUM_TOKENS = [256, 1024, 4096]

# Dtype size used by the GPU allocator for KV cache metadata scales.
DTYPE_SIZE = 2  # fp16


def _theoretical_compression_ratio(quantizer, head_dim: int) -> float:
    """Compute compression ratio using bytes_per_token_head (GPU allocator formula).

    This is the compression the engine actually achieves on GPU, independent of
    whether the benchmark runs on CPU (where fallback paths may not pack bits
    as tightly as the Triton GPU kernels).
    """
    # Original FP16: 2 tensors (K+V) × head_dim elements × 2 bytes each
    orig_per_head = 4 * head_dim

    if isinstance(quantizer, AsymTurboQuantKVQuantizer):
        # bytes_per_token_head returns k_persistent + v_persistent
        persistent, _ = quantizer.bytes_per_token_head(head_dim, DTYPE_SIZE)
        return orig_per_head / persistent if persistent > 0 else float("inf")

    # For MSE and Prod quantizers, bytes_per_token_head returns persistent
    # for ONE tensor (K or V). Both use the same scheme, so total = 2×.
    persistent, _ = quantizer.bytes_per_token_head(head_dim, DTYPE_SIZE)
    total_per_head = 2 * persistent
    return orig_per_head / total_per_head if total_per_head > 0 else float("inf")


def random_kv_tensor(num_tokens: int, num_heads: int, head_dim: int) -> torch.Tensor:
    """Generate random floating-point K/V tensor (representative distribution)."""
    return torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16)


def compute_metrics(orig: torch.Tensor, recon: torch.Tensor) -> dict:
    """Compute L1 metrics between original and reconstructed tensors."""
    orig_f = orig.float()
    recon_f = recon.float()

    mse = float((orig_f - recon_f).square().mean().item())
    mae = float((orig_f - recon_f).abs().mean().item())

    # cosine similarity (per-vector mean)
    o = orig_f.reshape(-1, orig_f.shape[-1])
    r = recon_f.reshape(-1, recon_f.shape[-1])
    o_n = torch.nn.functional.normalize(o, dim=-1)
    r_n = torch.nn.functional.normalize(r, dim=-1)
    cos_sim = float((o_n * r_n).sum(dim=-1).mean().item())

    # relative MSE
    orig_var = float(o.var().item())
    rel_mse = float(mse / max(orig_var, 1e-8))

    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cos_sim,
        "relative_mse": rel_mse,
    }


def bench_asym_quantizer(device: str) -> list[dict]:
    """Benchmark AsymTurboQuantKVQuantizer (4-bit K + 4-bit grouped-linear V)."""
    results = []
    for head_dim in HEAD_DIMS:
        quantizer = AsymTurboQuantKVQuantizer(
            head_dim=head_dim, k_bits=4, v_bits=4, v_group_size=32, seed=42
        )
        # Initialize on the right device
        quantizer._ensure_algo(head_dim, torch.device(device))

        for num_tokens in NUM_TOKENS:
            k = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)
            v = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)

            # Quantize K
            k_packed, k_scales = quantizer.quantize_k(k)
            k_recon = quantizer.dequantize_k(k_packed, k_scales, k.dtype)

            # Quantize V
            v_packed, v_scales, v_zeros = quantizer.quantize_v(v)
            v_recon = quantizer.dequantize_v(v_packed, v_scales, v_zeros, v.dtype)

            k_metrics = compute_metrics(k, k_recon)
            v_metrics = compute_metrics(v, v_recon)

            results.append({
                "quantizer": "AsymTurboQuant_4bit",
                "head_dim": head_dim,
                "num_tokens": num_tokens,
                "num_heads": NUM_KV_HEADS,
                "mse": (k_metrics["mse"] + v_metrics["mse"]) / 2,
                "mae": (k_metrics["mae"] + v_metrics["mae"]) / 2,
                "cosine_similarity": (k_metrics["cosine_similarity"] + v_metrics["cosine_similarity"]) / 2,
                "relative_mse": (k_metrics["relative_mse"] + v_metrics["relative_mse"]) / 2,
                "k_mse": k_metrics["mse"],
                "k_mae": k_metrics["mae"],
                "k_cosine_similarity": k_metrics["cosine_similarity"],
                "k_relative_mse": k_metrics["relative_mse"],
                "v_mse": v_metrics["mse"],
                "v_mae": v_metrics["mae"],
                "v_cosine_similarity": v_metrics["cosine_similarity"],
                "v_relative_mse": v_metrics["relative_mse"],
                "compression_ratio": _theoretical_compression_ratio(quantizer, head_dim),
                "total_params": k.numel() + v.numel(),
            })

    return results


def bench_prod_quantizer(device: str, bits: int) -> list[dict]:
    """Benchmark TurboQuantProdKVQuantizer (3-4 bit prod quantization)."""
    results = []
    for head_dim in HEAD_DIMS:
        quantizer = TurboQuantProdKVQuantizer(bits=bits)
        quantizer._ensure_algo(head_dim, torch.device(device))

        for num_tokens in NUM_TOKENS:
            k = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)
            v = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)

            k_packed, k_scales = quantizer.quantize(k)
            k_recon = quantizer.dequantize(k_packed, k_scales, k.dtype)

            v_packed, v_scales = quantizer.quantize(v)
            v_recon = quantizer.dequantize(v_packed, v_scales, v.dtype)

            k_metrics = compute_metrics(k, k_recon)
            v_metrics = compute_metrics(v, v_recon)

            results.append({
                "quantizer": f"TurboQuantProd_{bits}bit",
                "head_dim": head_dim,
                "num_tokens": num_tokens,
                "num_heads": NUM_KV_HEADS,
                "mse": (k_metrics["mse"] + v_metrics["mse"]) / 2,
                "mae": (k_metrics["mae"] + v_metrics["mae"]) / 2,
                "cosine_similarity": (k_metrics["cosine_similarity"] + v_metrics["cosine_similarity"]) / 2,
                "relative_mse": (k_metrics["relative_mse"] + v_metrics["relative_mse"]) / 2,
                "k_mse": k_metrics["mse"],
                "k_cosine_similarity": k_metrics["cosine_similarity"],
                "v_mse": v_metrics["mse"],
                "v_cosine_similarity": v_metrics["cosine_similarity"],
                "compression_ratio": _theoretical_compression_ratio(quantizer, head_dim),
                "total_params": k.numel() + v.numel(),
            })

    return results


def bench_mse_quantizer(device: str, bits: int) -> list[dict]:
    """Benchmark TurboQuantMSEKVQuantizer (MSE-only quantization)."""
    results = []
    for head_dim in HEAD_DIMS:
        quantizer = TurboQuantMSEKVQuantizer(bits=bits)
        quantizer._ensure_algo(head_dim, torch.device(device))

        for num_tokens in NUM_TOKENS:
            k = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)
            v = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)

            k_packed, k_scales = quantizer.quantize(k)
            k_recon = quantizer.dequantize(k_packed, k_scales, k.dtype)

            v_packed, v_scales = quantizer.quantize(v)
            v_recon = quantizer.dequantize(v_packed, v_scales, v.dtype)

            k_metrics = compute_metrics(k, k_recon)
            v_metrics = compute_metrics(v, v_recon)

            avg_mse = (k_metrics["mse"] + v_metrics["mse"]) / 2
            avg_mae = (k_metrics["mae"] + v_metrics["mae"]) / 2
            avg_cos = (k_metrics["cosine_similarity"] + v_metrics["cosine_similarity"]) / 2
            avg_rel = (k_metrics["relative_mse"] + v_metrics["relative_mse"]) / 2

            results.append({
                "quantizer": f"TurboQuantMSE_{bits}bit",
                "head_dim": head_dim,
                "num_tokens": num_tokens,
                "num_heads": NUM_KV_HEADS,
                "mse": avg_mse,
                "mae": avg_mae,
                "cosine_similarity": avg_cos,
                "relative_mse": avg_rel,
                "k_mse": k_metrics["mse"],
                "k_cosine_similarity": k_metrics["cosine_similarity"],
                "v_mse": v_metrics["mse"],
                "v_cosine_similarity": v_metrics["cosine_similarity"],
                "compression_ratio": _theoretical_compression_ratio(quantizer, head_dim),
                "total_params": k.numel() + v.numel(),
            })

    return results


def print_table(results: list[dict]):
    """Print summary table with key metrics."""
    print("\n=== Quantization Error Summary ===")
    header = f"{'Quantizer':<26} {'hd':>4} {'tokens':>7} {'MSE':>10} {'MAE':>8} {'CosSim':>8} {'RelMSE':>8} {'CompRatio':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        name = r["quantizer"]
        hd = r["head_dim"]
        nt = r["num_tokens"]
        mse = r["mse"]
        mae = r["mae"]
        cos = r["cosine_similarity"]
        rel = r["relative_mse"]
        cr = r["compression_ratio"]
        print(f"{name:<26} {hd:>4} {nt:>7} {mse:>10.6f} {mae:>8.4f} {cos:>8.4f} {rel:>8.4f} {cr:>10.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Direct KV cache quantization error measurement (MSE/MAE/cosine similarity)."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    device = args.device
    print(f"[INFO] Running quantization error benchmarks on {device}")

    all_results = []
    all_results.extend(bench_mse_quantizer(device, bits=3))
    all_results.extend(bench_prod_quantizer(device, bits=3))
    all_results.extend(bench_prod_quantizer(device, bits=4))
    all_results.extend(bench_asym_quantizer(device))

    print_table(all_results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[INFO] Results saved to {args.output_json}")

    # Print final JSON to stdout for subprocess capture
    print(json.dumps({"mode": "all", "device": device, "results": all_results}))


if __name__ == "__main__":
    main()
