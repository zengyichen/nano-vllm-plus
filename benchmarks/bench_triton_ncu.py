"""Minimal kernel reproducer for NCU profiling.

Each kernel runs once with proper warm-up and sync.
Usage:
  ncu --set full -o benchmarks/results/store_kvcache_ncu --target-processes all \
      python benchmarks/bench_triton_ncu.py --kernel store_kvcache

  ncu --set full -o benchmarks/results/rmsnorm_ncu --target-processes all \
      python benchmarks/bench_triton_ncu.py --kernel rmsnorm

  ncu --set full -o benchmarks/results/rotary_ncu --target-processes all \
      python benchmarks/bench_triton_ncu.py --kernel rotary

  ncu --section SpeedOfLight,Occupancy,LaunchStats,MemoryWorkloadAnalysis \
      -o benchmarks/results/store_kvcache_speedoflight --target-processes all \
      python benchmarks/bench_triton_ncu.py --kernel store_kvcache
"""

import argparse
import os
import sys

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════
# Kernel 1: store_kvcache
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def run_store_kvcache(args):
    """Run store_kvcache once for NCU profiling."""
    N, H, D = args.num_tokens, 8, args.head_dim
    cache_size = N * 4

    device = "cuda"
    key = torch.randn(N, H, D, dtype=torch.float16, device=device)
    value = torch.randn(N, H, D, dtype=torch.float16, device=device)
    k_cache = torch.zeros(cache_size, H * D, dtype=torch.float16, device=device)
    v_cache = torch.zeros(cache_size, H * D, dtype=torch.float16, device=device)

    if args.slot_pattern == "random":
        slots = torch.randperm(cache_size, dtype=torch.int32, device=device)[:N]
    elif args.slot_pattern == "strided":
        slots = torch.arange(0, N * 3, 3, dtype=torch.int32, device=device)[:N]
    else:
        slots = torch.arange(N, dtype=torch.int32, device=device)

    # Warm-up: compile and cache
    _store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slots, D)
    torch.cuda.synchronize()

    # Execute once for NCU capture
    _store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slots, D)
    torch.cuda.synchronize()

    print(f"store_kvcache: N={N}, H={H}, D={D}, pattern={args.slot_pattern} — done")


# ═══════════════════════════════════════════════════════════════════
# Kernel 2: RMSNorm
# ═══════════════════════════════════════════════════════════════════

def run_rmsnorm(args):
    """Run RMSNorm once for NCU profiling."""
    import torch.nn as nn

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size, device="cuda"))
            self.variance_epsilon = eps

        @torch.compile(mode="reduce-overhead")
        def forward(self, x):
            x_f = x.float()
            variance = x_f.pow(2).mean(-1, keepdim=True)
            return (x_f * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype) * self.weight

    N, H = args.num_tokens, args.head_dim  # head_dim == hidden_dim for RMSNorm
    norm = RMSNorm(H)
    x = torch.randn(N, H, dtype=torch.float16, device="cuda")

    # Warm-up: compile + cache
    _ = norm.forward(x)
    torch.cuda.synchronize()

    # Execute once for NCU capture
    _ = norm.forward(x)
    torch.cuda.synchronize()

    print(f"rmsnorm: N={N}, H={H} — done")


# ═══════════════════════════════════════════════════════════════════
# Kernel 3: RotaryEmbedding
# ═══════════════════════════════════════════════════════════════════

def run_rotary(args):
    """Run RoPE once for NCU profiling."""

    @torch.compile(mode="reduce-overhead")
    def apply_rotary_emb_compiled(x, cos, sin):
        x_f = x.float()
        x1, x2 = torch.chunk(x_f, 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    N, H, D = args.num_tokens, 8, args.head_dim
    x = torch.randn(N, H, D, dtype=torch.float16, device="cuda")

    # Build cos/sin cache matching nanovllm RotaryEmbedding
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, D, 2, dtype=torch.float, device="cuda") / D))
    t = torch.arange(4096, dtype=torch.float, device="cuda")
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze(1)

    positions = torch.randint(0, 4096, (N,), device="cuda")
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    # Warm-up
    _ = apply_rotary_emb_compiled(x, cos, sin)
    torch.cuda.synchronize()

    # Execute once
    _ = apply_rotary_emb_compiled(x, cos, sin)
    torch.cuda.synchronize()

    print(f"rotary_emb: N={N}, H={H}, D={D} — done")


# ═══════════════════════════════════════════════════════════════════
# Kernel 4: Sampler (Gumbel-max)
# ═══════════════════════════════════════════════════════════════════

def run_sampler(args):
    """Run sampler once for NCU profiling."""

    @torch.compile(mode="reduce-overhead")
    def sampler_compiled(logits, temperatures):
        logits_f = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits_f, dim=-1)
        gumbel = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return probs.div_(gumbel).argmax(dim=-1)

    B, V = args.num_tokens, args.head_dim  # reuse: num_tokens=batch, head_dim=vocab
    logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
    temps = torch.full((B,), 0.6, dtype=torch.float32, device="cuda")

    # Warm-up
    _ = sampler_compiled(logits, temps)
    torch.cuda.synchronize()

    # Execute once
    _ = sampler_compiled(logits, temps)
    torch.cuda.synchronize()

    print(f"sampler: B={B}, V={V} — done")


# ═══════════════════════════════════════════════════════════════════
# Kernel 5: SiluAndMul
# ═══════════════════════════════════════════════════════════════════

def run_silu_and_mul(args):
    """Run SiluAndMul once for NCU profiling."""

    @torch.compile(mode="reduce-overhead")
    def silu_and_mul_compiled(gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    N, H = args.num_tokens, args.head_dim  # head_dim = intermediate_half
    gate_up = torch.randn(N, H * 2, dtype=torch.float16, device="cuda")

    # Warm-up
    _ = silu_and_mul_compiled(gate_up)
    torch.cuda.synchronize()

    # Execute once
    _ = silu_and_mul_compiled(gate_up)
    torch.cuda.synchronize()

    print(f"silu_and_mul: N={N}, H={H} — done")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

KERNELS = {
    "store_kvcache": run_store_kvcache,
    "rmsnorm": run_rmsnorm,
    "rotary": run_rotary,
    "sampler": run_sampler,
    "silu": run_silu_and_mul,
}


def main():
    parser = argparse.ArgumentParser(description="NCU profiling helper for nano-vLLM kernels")
    parser.add_argument("--kernel", choices=list(KERNELS.keys()), default="store_kvcache")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--slot-pattern", choices=["sequential", "random", "strided"],
                        default="random")
    args = parser.parse_args()

    print(f"[NCU profiling] kernel={args.kernel}, num_tokens={args.num_tokens}, head_dim={args.head_dim}")
    KERNELS[args.kernel](args)


if __name__ == "__main__":
    main()
