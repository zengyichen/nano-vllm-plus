"""Triton kernel benchmarks with warm-up, NVTX, and nsys/ncu profiling support.

Benchmarks the custom Triton kernels and torch.compile'd functions in nano-vLLM:
  1. store_kvcache_kernel — paged KV cache scatter
  2. apply_rotary_emb (torch.compile) — RoPE
  3. Sampler.forward (torch.compile) — Gumbel-max sampling
  4. RMSNorm.forward (torch.compile)
  5. SiluAndMul.forward (torch.compile)

Usage:
  # Quick benchmark (all kernels, default configs)
  python benchmarks/bench_triton.py

  # Benchmark specific kernel with custom sizes
  python benchmarks/bench_triton.py --kernel store_kvcache --head-dims 64 128 --num-tokens 256 1024

  # Profile with NVTX ranges (for nsys)
  python benchmarks/bench_triton.py --nvtx

  # Save results to JSON
  python benchmarks/bench_triton.py --output-json benchmarks/results/bench_triton.json

Profiling with nsys:
  nsys profile --trace=cuda,nvtx python benchmarks/bench_triton.py --nvtx --kernel store_kvcache

Profiling with ncu (single kernel, small config):
  ncu --set full python benchmarks/bench_triton.py --kernel store_kvcache --head-dims 128 --num-tokens 256
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import triton
import triton.language as tl

# ── NVTX helper ────────────────────────────────────────────────────
_nvtx_enabled = False

try:
    import nvtx
except ImportError:
    nvtx = None

def nvtx_range(name: str):
    """Context manager for NVTX range (no-op if nvtx unavailable)."""
    if _nvtx_enabled and nvtx is not None:
        return nvtx.annotate(name)
    else:
        from contextlib import nullcontext
        return nullcontext()


# ═══════════════════════════════════════════════════════════════════
# Kernel 1: store_kvcache — paged KV cache scatter
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


def triton_store_kvcache(key: torch.Tensor, value: torch.Tensor,
                         k_cache: torch.Tensor, v_cache: torch.Tensor,
                         slot_mapping: torch.Tensor):
    """Triton kernel for KV cache store."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    _store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                                  k_cache, v_cache, slot_mapping, D)


def torch_store_kvcache(key: torch.Tensor, value: torch.Tensor,
                         k_cache: torch.Tensor, v_cache: torch.Tensor,
                         slot_mapping: torch.Tensor):
    """PyTorch baseline: index_copy_ (flat tensors to match Triton's layout)."""
    N, H, D = key.shape
    key_flat = key.reshape(N, H * D)
    value_flat = value.reshape(N, H * D)
    idx = slot_mapping.long()
    k_cache.index_copy_(0, idx, key_flat)
    v_cache.index_copy_(0, idx, value_flat)


def torch_store_kvcache_slice(key: torch.Tensor, value: torch.Tensor,
                               k_cache: torch.Tensor, v_cache: torch.Tensor,
                               slot_mapping: torch.Tensor):
    """PyTorch baseline: direct slicing (fastest when slot_mapping is sequential)."""
    N, H, D = key.shape
    key_flat = key.reshape(N, H * D)
    value_flat = value.reshape(N, H * D)
    k_cache[slot_mapping] = key_flat
    v_cache[slot_mapping] = value_flat


# ═══════════════════════════════════════════════════════════════════
# Kernel 2: apply_rotary_emb — RoPE (torch.compile)
# ═══════════════════════════════════════════════════════════════════

def rotary_emb_eager(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Eager RoPE (no torch.compile)."""
    x_f = x.float()
    x1, x2 = torch.chunk(x_f, 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


@torch.compile
def rotary_emb_compiled(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """torch.compile'd RoPE."""
    x_f = x.float()
    x1, x2 = torch.chunk(x_f, 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════
# Kernel 3: RMSNorm (torch.compile)
# ═══════════════════════════════════════════════════════════════════

class RMSNormBench(nn.Module):
    """Replicate nanovllm RMSNorm for standalone benchmark."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_eager(self, x: torch.Tensor) -> torch.Tensor:
        """Eager RMSNorm."""
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        return (x_f * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype) * self.weight

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compiled RMSNorm."""
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        return (x_f * torch.rsqrt(variance + self.variance_epsilon)).to(x.dtype) * self.weight


# ═══════════════════════════════════════════════════════════════════
# Kernel 4: Sampler — Gumbel-max (torch.compile)
# ═══════════════════════════════════════════════════════════════════

def sampler_eager(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """Eager Gumbel-max sampling."""
    logits_f = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits_f, dim=-1)
    gumbel = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return probs.div_(gumbel).argmax(dim=-1)


@torch.compile
def sampler_compiled(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """Compiled Gumbel-max sampling."""
    logits_f = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits_f, dim=-1)
    gumbel = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return probs.div_(gumbel).argmax(dim=-1)


# ═══════════════════════════════════════════════════════════════════
# Kernel 5: SiluAndMul — activation (torch.compile)
# ═══════════════════════════════════════════════════════════════════

def silu_and_mul_eager(gate_up: torch.Tensor) -> torch.Tensor:
    """Eager SiLU + gate."""
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


@torch.compile
def silu_and_mul_compiled(gate_up: torch.Tensor) -> torch.Tensor:
    """Compiled SiLU + gate."""
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


# ═══════════════════════════════════════════════════════════════════
# Benchmark infrastructure
# ═══════════════════════════════════════════════════════════════════

def gpu_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "vram_gb": p.total_memory / 1e9,
        "sps": p.multi_processor_count,
        "max_threads_per_sm": p.max_threads_per_multi_processor,
        "compute_capability": f"{p.major}.{p.minor}",
        "memory_bus_bits": p.memory_bus_width,
        "memory_clock_mhz": p.memory_clock_rate / 1000,
        "peak_bw_gbs": (p.memory_bus_width / 8) * (p.memory_clock_rate / 1e6) * 2,  # GDDR6 DDR
    }


def time_kernel(fn, warmup: int = 10, repeat: int = 100,
                synchronize: bool = True) -> tuple[float, float]:
    """Time a callable with warm-up, returning (mean_ms, std_ms).

    Uses CUDA events for accurate GPU timing when synchronize=True.
    """
    # Warmup
    for _ in range(warmup):
        fn()

    if synchronize:
        # Use CUDA events for GPU-side timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(repeat):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std


def bench_store_kvcache(device, configs: list[dict]) -> list[dict]:
    """Benchmark store_kvcache_kernel vs PyTorch alternatives.

    Configs: list of {num_tokens, num_heads, head_dim, slot_pattern}
    slot_pattern: 'random' | 'sequential' | 'strided'
    """
    results = []
    for cfg in configs:
        N, H, D, pattern = cfg["num_tokens"], cfg["num_heads"], cfg["head_dim"], cfg["slot_pattern"]
        cache_size = max(N * 4, 4096)  # oversize cache

        key = torch.randn(N, H, D, dtype=torch.float16, device=device)
        value = torch.randn(N, H, D, dtype=torch.float16, device=device)

        if pattern == "sequential":
            slots = torch.arange(N, dtype=torch.int32, device=device)
        elif pattern == "random":
            slots = torch.randperm(cache_size, dtype=torch.int32, device=device)[:N]
        else:  # strided
            slots = torch.arange(0, N * 3, 3, dtype=torch.int32, device=device)[:N]

        k_cache_1 = torch.zeros(cache_size, H * D, dtype=torch.float16, device=device)
        v_cache_1 = torch.zeros(cache_size, H * D, dtype=torch.float16, device=device)
        k_cache_2 = k_cache_1.clone()
        v_cache_2 = v_cache_1.clone()
        k_cache_3 = k_cache_1.clone()
        v_cache_3 = v_cache_1.clone()

        memory_rw_gb = (key.numel() + value.numel()) * 2 * 2 / 1e9  # read+write, K+V, fp16

        # Triton kernel
        with nvtx_range(f"store_kvcache_triton_N{N}_H{H}_D{D}_{pattern}"):
            triton_ms, triton_std = time_kernel(
                lambda: triton_store_kvcache(key, value, k_cache_1, v_cache_1, slots),
                warmup=10, repeat=50)
        triton_bw = memory_rw_gb / (triton_ms / 1000)

        # PyTorch index_copy_
        with nvtx_range(f"store_kvcache_index_copy_N{N}_H{H}_D{D}_{pattern}"):
            torch_copy_ms, torch_copy_std = time_kernel(
                lambda: torch_store_kvcache(key, value, k_cache_2, v_cache_2, slots),
                warmup=10, repeat=50)

        # PyTorch slice assignment
        with nvtx_range(f"store_kvcache_slice_N{N}_H{H}_D{D}_{pattern}"):
            torch_slice_ms, torch_slice_std = time_kernel(
                lambda: torch_store_kvcache_slice(key, value, k_cache_3, v_cache_3, slots),
                warmup=10, repeat=50)

        # Verify correctness
        assert torch.allclose(k_cache_1, k_cache_2, atol=1e-3), f"Mismatch: triton vs index_copy"
        assert torch.allclose(k_cache_1, k_cache_3, atol=1e-3), f"Mismatch: triton vs slice"

        results.append({
            "kernel": "store_kvcache",
            "num_tokens": N, "num_heads": H, "head_dim": D,
            "slot_pattern": pattern, "memory_rw_gb": memory_rw_gb,
            "triton_ms": triton_ms, "triton_std_ms": triton_std,
            "triton_bandwidth_gbs": triton_bw,
            "torch_index_copy_ms": torch_copy_ms, "torch_slice_ms": torch_slice_ms,
        })

    return results


def bench_rotary_emb(device, configs: list[dict]) -> list[dict]:
    """Benchmark RoPE: eager vs torch.compile."""
    results = []
    for cfg in configs:
        N, H, D = cfg["num_tokens"], cfg["num_heads"], cfg["head_dim"]
        x = torch.randn(N, H, D, dtype=torch.float16, device=device)
        positions = torch.randint(0, 4096, (N,), device=device)

        # Pre-compute cos/sin like RotaryEmbedding does
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, D, 2, dtype=torch.float, device=device) / D))
        t = torch.arange(4096, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # Match nanovllm RotaryEmbedding shape: (max_pos, 1, head_dim)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze(1)
        cos_sin = cache[positions]  # (N, 1, head_dim) — broadcastable with (N, H, D//2)
        cos, sin = cos_sin.chunk(2, dim=-1)  # each: (N, 1, D//2)

        # Compile warm-up pass
        _ = rotary_emb_compiled(x, cos, sin)
        torch.cuda.synchronize()

        with nvtx_range(f"rotary_eager_N{N}_H{H}_D{D}"):
            eager_ms, eager_std = time_kernel(
                lambda: rotary_emb_eager(x, cos, sin), warmup=10, repeat=50)

        with nvtx_range(f"rotary_compiled_N{N}_H{H}_D{D}"):
            compiled_ms, compiled_std = time_kernel(
                lambda: rotary_emb_compiled(x, cos, sin), warmup=10, repeat=50)

        memory_rw_gb = (x.numel() * 6) * 2 / 1e9  # read x, cos, sin; write x; fp16 (with float cast)

        results.append({
            "kernel": "rotary_emb",
            "num_tokens": N, "num_heads": H, "head_dim": D,
            "eager_ms": eager_ms, "eager_std_ms": eager_std,
            "compiled_ms": compiled_ms, "compiled_std_ms": compiled_std,
            "speedup": eager_ms / compiled_ms,
            "memory_rw_gb": memory_rw_gb,
            "compiled_bandwidth_gbs": memory_rw_gb / (compiled_ms / 1000),
        })

    return results


def bench_rmsnorm(device, configs: list[dict]) -> list[dict]:
    """Benchmark RMSNorm: eager vs torch.compile."""
    results = []
    for cfg in configs:
        N, H = cfg["num_tokens"], cfg["hidden_dim"]
        x = torch.randn(N, H, dtype=torch.float16, device=device)
        norm = RMSNormBench(H).to(device)

        # Compile warm-up pass
        _ = norm.forward(x)
        torch.cuda.synchronize()

        with nvtx_range(f"rmsnorm_eager_N{N}_H{H}"):
            eager_ms, eager_std = time_kernel(
                lambda: norm.forward_eager(x), warmup=10, repeat=50)

        with nvtx_range(f"rmsnorm_compiled_N{N}_H{H}"):
            compiled_ms, compiled_std = time_kernel(
                lambda: norm.forward(x), warmup=10, repeat=50)

        results.append({
            "kernel": "rmsnorm",
            "num_tokens": N, "hidden_dim": H,
            "eager_ms": eager_ms, "eager_std_ms": eager_std,
            "compiled_ms": compiled_ms, "compiled_std_ms": compiled_std,
            "speedup": eager_ms / compiled_ms,
        })

    return results


def bench_sampler(device, configs: list[dict]) -> list[dict]:
    """Benchmark sampler: eager vs torch.compile."""
    results = []
    for cfg in configs:
        B, V = cfg["batch_size"], cfg["vocab_size"]
        logits = torch.randn(B, V, dtype=torch.float32, device=device)
        temps = torch.full((B,), 0.6, dtype=torch.float32, device=device)

        # Compile warm-up pass
        _ = sampler_compiled(logits, temps)
        torch.cuda.synchronize()

        with nvtx_range(f"sampler_eager_B{B}_V{V}"):
            eager_ms, eager_std = time_kernel(
                lambda: sampler_eager(logits, temps), warmup=10, repeat=50)

        with nvtx_range(f"sampler_compiled_B{B}_V{V}"):
            compiled_ms, compiled_std = time_kernel(
                lambda: sampler_compiled(logits, temps), warmup=10, repeat=50)

        results.append({
            "kernel": "sampler",
            "batch_size": B, "vocab_size": V,
            "eager_ms": eager_ms, "eager_std_ms": eager_std,
            "compiled_ms": compiled_ms, "compiled_std_ms": compiled_std,
            "speedup": eager_ms / compiled_ms,
        })

    return results


def bench_silu_and_mul(device, configs: list[dict]) -> list[dict]:
    """Benchmark SiluAndMul: eager vs torch.compile."""
    results = []
    for cfg in configs:
        N, H = cfg["num_tokens"], cfg["hidden_dim"]
        gate_up = torch.randn(N, H * 2, dtype=torch.float16, device=device)

        _ = silu_and_mul_compiled(gate_up)
        torch.cuda.synchronize()

        with nvtx_range(f"silu_eager_N{N}_H{H}"):
            eager_ms, eager_std = time_kernel(
                lambda: silu_and_mul_eager(gate_up), warmup=10, repeat=50)

        with nvtx_range(f"silu_compiled_N{N}_H{H}"):
            compiled_ms, compiled_std = time_kernel(
                lambda: silu_and_mul_compiled(gate_up), warmup=10, repeat=50)

        results.append({
            "kernel": "silu_and_mul",
            "num_tokens": N, "hidden_dim": H,
            "eager_ms": eager_ms, "eager_std_ms": eager_std,
            "compiled_ms": compiled_ms, "compiled_std_ms": compiled_std,
            "speedup": eager_ms / compiled_ms,
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════

def print_store_kvcache_table(results: list[dict]):
    print("\n" + "=" * 120)
    print("1. store_kvcache — Paged KV Cache Scatter")
    print("   ⚠ Kernel times < 50μs are limited by CUDA event resolution (~5μs).")
    print("     For accurate small-kernel timings, use: ncu --set full python benchmarks/bench_triton.py --kernel store_kvcache")
    print("=" * 120)
    hdr = (f"  {'N':>5} {'H':>3} {'D':>4} {'pattern':<12} "
           f"{'Triton(μs)':>10} {'IdxCopy(μs)':>11} {'Slice(μs)':>10} "
           f"{'Winner':>10} {'BW(GB/s)':>10}")
    print(hdr)
    print("  " + "-" * 110)
    for r in results:
        fastest = min(r["triton_ms"], r["torch_index_copy_ms"], r["torch_slice_ms"])
        winner = ("Triton" if r["triton_ms"] <= fastest * 1.05 else
                  "IdxCopy" if r["torch_index_copy_ms"] <= fastest * 1.05 else "Slice")
        # Scale: use μs when values are small, ms when large
        if r["num_tokens"] >= 4096:
            print(f"  {r['num_tokens']:>5} {r['num_heads']:>3} {r['head_dim']:>4} "
                  f"{r['slot_pattern']:<12} "
                  f"{r['triton_ms']*1000:>10.1f} {r['torch_index_copy_ms']*1000:>11.1f} "
                  f"{r['torch_slice_ms']*1000:>10.1f} {winner:>10} {r['triton_bandwidth_gbs']:>9.1f}")
        else:
            print(f"  {r['num_tokens']:>5} {r['num_heads']:>3} {r['head_dim']:>4} "
                  f"{r['slot_pattern']:<12} "
                  f"{r['triton_ms']*1000:>10.1f} {r['torch_index_copy_ms']*1000:>11.1f} "
                  f"{r['torch_slice_ms']*1000:>10.1f} {winner:>10} {'(n/a)':>10}")


def print_compile_table(title: str, results: list[dict], keys: list[str]):
    print("\n" + "=" * 110)
    print(f"{title} — eager vs torch.compile")
    print("=" * 110)
    parts = ["  "]
    for k in keys:
        parts.append(f"{k:>12}")
    parts.extend([f"{'Eager(ms)':>10}", f"{'Comp(ms)':>10}", f"{'Speedup':>8}"])
    hdr = " ".join(parts)
    print(hdr)
    print("  " + "-" * 100)
    for r in results:
        vals = " ".join(f"{r[k]:>12}" for k in keys)
        print(f"  {vals} {r['eager_ms']:>10.4f} {r['compiled_ms']:>10.4f} {r['speedup']:>7.2f}x")


def print_bandwidth_summary(results: list[dict]):
    """Print bandwidth utilization summary for store_kvcache kernel."""
    store_results = [r for r in results if r.get("kernel") == "store_kvcache"]
    if not store_results:
        return

    peak_bw = gpu_info()["peak_bw_gbs"]
    print("\n" + "=" * 110)
    print("BANDWIDTH UTILIZATION — store_kvcache")
    print(f"  Peak GPU bandwidth: {peak_bw:.0f} GB/s")
    print("=" * 110)
    for r in store_results:
        pct = min(r["triton_bandwidth_gbs"] / peak_bw * 100, 100)
        bar = "#" * int(pct / 5)
        note = " (sub-μs — CUDA event resolution limited)" if r["num_tokens"] <= 256 else ""
        print(f"  N={r['num_tokens']:>5} H={r['num_heads']:>3} D={r['head_dim']:>4} "
              f"{r['slot_pattern']:<12}: {r['triton_bandwidth_gbs']:>6.1f} GB/s "
              f"({pct:>5.1f}%) {bar}{note}")


def analyze_roofline(results: list[dict]):
    """Simple roofline analysis for each kernel."""
    store_r = [r for r in results if r.get("kernel") == "store_kvcache"]
    if not store_r:
        return

    peak_bw = gpu_info()["peak_bw_gbs"]

    print("\n" + "=" * 110)
    print("ROOFLINE ANALYSIS")
    print(f"  Peak BW: {peak_bw:.0f} GB/s  |  Peak Compute (FP16): ~90 TFLOPS")
    print("=" * 110)
    print(f"  {'Kernel':<22} {'Config':<30} {'BW(GB/s)':>8} {'BW Util':>8} {'Bottleneck':>14}")
    print("  " + "-" * 90)

    for r in store_r:
        cfg = f"N={r['num_tokens']} H={r['num_heads']} D={r['head_dim']} {r['slot_pattern']}"
        bw = r["triton_bandwidth_gbs"]
        util = bw / peak_bw * 100
        # store_kvcache: read 2×(H×D×2) + write 2×(H×D×2) per token, zero compute → always BW-bound
        bottleneck = "Memory Bandwidth"
        print(f"  {'store_kvcache':<22} {cfg:<30} {bw:>8.1f} {util:>7.1f}% {bottleneck:>14}")

    for r in [x for x in results if x.get("kernel") in ("rotary_emb", "rmsnorm", "silu_and_mul")]:
        # Estimate FLOPs for these kernels
        if r["kernel"] == "rotary_emb":
            flops = r["num_tokens"] * r["num_heads"] * r["head_dim"] * 4  # 4 mul+add per elem
        elif r["kernel"] == "rmsnorm":
            flops = r["num_tokens"] * r["hidden_dim"] * 2  # square + multiply
        elif r["kernel"] == "silu_and_mul":
            flops = r["num_tokens"] * r["hidden_dim"] * 2  # SiLU + multiply
        else:
            flops = 0

        if flops > 0 and "compiled_bandwidth_gbs" in r:
            ai = flops / (r.get("memory_rw_gb", 1e-6) * 1e9)  # FLOP/byte
            bottleneck = "Memory BW" if ai < 50 else "Compute" if ai > 200 else "Mixed"
            cfg = f"N={r['num_tokens']}"
            print(f"  {r['kernel']:<22} {cfg:<30} "
                  f"{r.get('compiled_bandwidth_gbs', 0):>8.1f} {'N/A':>7} "
                  f"AI={ai:.1f} FLOP/byte {bottleneck:>6}")

    # Sampler
    for r in [x for x in results if x.get("kernel") == "sampler"]:
        flops = r["batch_size"] * r["vocab_size"] * 2  # softmax + div
        print(f"  {'sampler':<22} {'B=' + str(r['batch_size']) + ' V=' + str(r['vocab_size']):<30} "
              f"{'N/A':>8} {'N/A':>7} Softmax O(V) Compute{'':>6}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    global _nvtx_enabled

    parser = argparse.ArgumentParser(description="Triton kernel benchmarks for nano-vLLM")
    parser.add_argument("--kernel", choices=["all", "store_kvcache", "rotary", "rmsnorm",
                                              "sampler", "silu"], default="all")
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--patterns", nargs="+", default=["sequential", "random", "strided"])
    parser.add_argument("--nvtx", action="store_true", help="Enable NVTX ranges for nsys profiling")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    _nvtx_enabled = args.nvtx

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    device = "cuda"
    info = gpu_info()
    print(f"GPU: {info['name']} ({info['vram_gb']:.1f} GB, {info['sps']} SMs, "
          f"CC {info['compute_capability']}, {info['peak_bw_gbs']:.0f} GB/s peak BW)")
    if args.nvtx:
        print(f"[NVTX enabled — use with: nsys profile --trace=cuda,nvtx python benchmarks/bench_triton.py --nvtx]")
    else:
        print(f"[NVTX disabled — use --nvtx flag for nsys profiling]")

    all_results = []
    num_heads = 8  # standard for Qwen3-0.6B

    # ── store_kvcache ──
    if args.kernel in ("all", "store_kvcache"):
        configs = []
        for nt in args.num_tokens:
            for d in args.head_dims:
                for pat in args.patterns:
                    configs.append({"num_tokens": nt, "num_heads": num_heads,
                                    "head_dim": d, "slot_pattern": pat})
        print(f"\nBenchmarking store_kvcache ({len(configs)} configs)...")
        r = bench_store_kvcache(device, configs)
        all_results.extend(r)
        print_store_kvcache_table(r)
        print_bandwidth_summary(all_results)

    # ── rotary_emb ──
    if args.kernel in ("all", "rotary"):
        configs = []
        for nt in args.num_tokens:
            for d in args.head_dims:
                configs.append({"num_tokens": nt, "num_heads": num_heads, "head_dim": d})
        print(f"\nBenchmarking rotary_emb ({len(configs)} configs)...")
        r = bench_rotary_emb(device, configs)
        all_results.extend(r)
        print_compile_table("2. apply_rotary_emb (RoPE)", r,
                            ["num_tokens", "num_heads", "head_dim"])

    # ── rmsnorm ──
    if args.kernel in ("all", "rmsnorm"):
        configs = [{"num_tokens": nt, "hidden_dim": d}
                   for nt in args.num_tokens for d in [1024, 4096]]
        print(f"\nBenchmarking RMSNorm ({len(configs)} configs)...")
        r = bench_rmsnorm(device, configs)
        all_results.extend(r)
        print_compile_table("3. RMSNorm", r, ["num_tokens", "hidden_dim"])

    # ── sampler ──
    if args.kernel in ("all", "sampler"):
        configs = [{"batch_size": bs, "vocab_size": vs}
                   for bs in [1, 8, 32] for vs in [32000, 151936]]
        print(f"\nBenchmarking Sampler ({len(configs)} configs)...")
        r = bench_sampler(device, configs)
        all_results.extend(r)
        print_compile_table("4. Sampler (Gumbel-max)", r, ["batch_size", "vocab_size"])

    # ── silu_and_mul ──
    if args.kernel in ("all", "silu"):
        configs = [{"num_tokens": nt, "hidden_dim": d}
                   for nt in args.num_tokens for d in [1024, 3072]]
        print(f"\nBenchmarking SiluAndMul ({len(configs)} configs)...")
        r = bench_silu_and_mul(device, configs)
        all_results.extend(r)
        print_compile_table("5. SiluAndMul (activation)", r, ["num_tokens", "hidden_dim"])

    # ── Roofline ──
    analyze_roofline(all_results)

    # ── Summary ──
    print("\n" + "=" * 110)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 110)

    for kernel_name in sorted(set(r["kernel"] for r in all_results)):
        kernel_results = [r for r in all_results if r["kernel"] == kernel_name]
        if kernel_name == "store_kvcache":
            avg_bw = sum(r["triton_bandwidth_gbs"] for r in kernel_results) / len(kernel_results)
            best_bw = max(r["triton_bandwidth_gbs"] for r in kernel_results)
            peak = info["peak_bw_gbs"]
            print(f"  store_kvcache:   avg BW={avg_bw:.0f} GB/s, peak={best_bw:.0f} GB/s "
                  f"({best_bw/peak*100:.0f}% of {peak:.0f} GB/s peak)")
            # Is the Triton kernel better than PyTorch?
            triton_wins = sum(1 for r in kernel_results
                              if r["triton_ms"] < min(r["torch_index_copy_ms"], r["torch_slice_ms"]))
            print(f"                  Triton wins {triton_wins}/{len(kernel_results)} configs "
                  f"vs PyTorch index_copy_/slice")
        elif kernel_name in ("rotary_emb", "rmsnorm", "sampler", "silu_and_mul"):
            avg_speedup = sum(r["speedup"] for r in kernel_results) / len(kernel_results)
            best_speedup = max(r["speedup"] for r in kernel_results)
            print(f"  {kernel_name:<15}: torch.compile avg {avg_speedup:.2f}x faster, "
                  f"best {best_speedup:.2f}x")

    print()
    print("  Profiling commands:")
    print(f"    nsys profile --trace=cuda,nvtx python {__file__} --nvtx --kernel store_kvcache")
    print(f"    ncu --set full python {__file__} --kernel store_kvcache --head-dims 128 --num-tokens 256")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        payload = {"gpu": info, "results": all_results}
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
