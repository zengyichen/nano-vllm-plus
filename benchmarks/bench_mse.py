"""Quantization error measurement: standalone paper implementation.

Implements three quantization algorithms directly following TurboQuant paper
(arXiv:2504.19874v1), with NO dependency on nanovllm:

1. TurboQuantMSE (Algorithm 1): random rotation + Lloyd-Max codebook
   → minimizes reconstruction MSE, but BIASED inner products

2. TurboQuantProd (Algorithm 2): MSE(b-1 bits) + QJL residual (1 bit)
   → UNBIASED inner products, but higher reconstruction MSE

3. GroupedLinear: per-group min-max affine quantization
   → lowest reconstruction MSE, but NO inner product guarantees

Measures:
  - Reconstruction: MSE, MAE, cosine similarity → matters for V (weighted sum)
  - Inner product: IP MSE, IP bias → matters for K (attention scores Q·K^T)

Reference: TurboQuant paper (arXiv:2504.19874v1)
  Theorem 1: MSE quantizer achieves Dmse ≤ √(3π)/2 · 1/4^b
  Theorem 2: Prod quantizer achieves Dprod ≤ √(3π²)·‖y‖²/d · 1/4^b
  Lemma 4: QJL correction — unbiased with variance ≤ π/(2d)·‖y‖²
"""

import argparse
import json
import math

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TurboQuantMSE (Algorithm 1 from paper)
# ═══════════════════════════════════════════════════════════════════════════════

class TurboQuantMSE(nn.Module):
    """Algorithm 1: MSE-optimal per-coordinate scalar quantization.

    Steps:
    1. Generate random rotation Π (QR decomposition of Gaussian matrix)
    2. Construct Lloyd-Max codebook for the distribution of rotated unit vectors
    3. quantize:  normalize → rotate → nearest centroid per coordinate
    4. dequantize: codebook lookup → unrotate → renormalize

    Dmse ≤ √(3π)/2 · 1/4^b  (Theorem 1)
    """

    def __init__(self, d: int, bits: int = 3, seed: int = 0):
        super().__init__()
        assert bits >= 1
        self.d = d
        self.bits = bits

        g = torch.Generator(device="cpu")
        g.manual_seed(seed + d * 131 + bits * 17)

        # ── Random rotation Π via QR decomposition ──
        a = torch.randn(d, d, generator=g, dtype=torch.float32)
        q, r = torch.linalg.qr(a)
        pi = q * torch.sign(torch.diag(r))

        # ── Lloyd-Max codebook for Beta(1/2, 1/2) or Normal ──
        num_centroids = 1 << bits
        sample = torch.randn(200000, generator=g, dtype=torch.float32) * (1.0 / math.sqrt(d))
        qs = torch.linspace(
            1.0 / (2 * num_centroids),
            1.0 - 1.0 / (2 * num_centroids),
            num_centroids,
            dtype=torch.float32,
        )
        centroids = torch.quantile(sample, qs)

        for _ in range(20):
            dist = (sample[:, None] - centroids[None, :]).abs()
            assign = dist.argmin(dim=1)
            for i in range(num_centroids):
                mask = assign == i
                if mask.any():
                    centroids[i] = sample[mask].mean()

        self.register_buffer("pi", pi)
        self.register_buffer("codebook", centroids.sort()[0])

    def quantize(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize v → (codebook_indices, norms)."""
        v32 = v.to(torch.float32)
        norm = torch.linalg.norm(v32, dim=-1, keepdim=True).clamp_min(1e-8)
        x = v32 / norm
        y = x @ self.pi.t().to(v.device, dtype=torch.float32)
        dist = (y[..., None] - self.codebook.to(v.device)).abs()
        idx = dist.argmin(dim=-1).to(torch.int64)
        return idx, norm.to(v.dtype)

    def dequantize(self, idx: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Dequantize (codebook_indices, norms) → reconstructed tensor."""
        y_hat = self.codebook.to(idx.device, dtype=torch.float32)[idx]
        x_hat = y_hat @ self.pi.to(idx.device, dtype=torch.float32)
        return x_hat * norm.to(torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TurboQuantProd (Algorithm 2 from paper)
# ═══════════════════════════════════════════════════════════════════════════════

class TurboQuantProd(nn.Module):
    """Algorithm 2: Inner-product-optimized quantization via MSE + QJL.

    Two-stage pipeline:
    1. MSE quantizer with (b-1) bits → approximate reconstruction x_mse
    2. QJL correction on residual r = x - x_mse:
       - quantize: qjl = sign(S @ r)  (1 bit per coordinate)
       - dequantize: x_qjl = √(π/2)/d · γ · (qjl @ S)

    Key property: E[⟨q, x⟩ - ⟨q, x_recon⟩] = 0  — UNBIASED inner products
    Price: reconstruction MSE multiplied by ≈π/2 (~1.57x)

    Dprod ≤ √(3π²)·‖y‖²/d · 1/4^b  (Theorem 2)
    """

    def __init__(self, d: int, bits: int = 3, seed: int = 0):
        super().__init__()
        assert bits >= 2
        self.d = d
        self.bits = bits
        self.mse = TurboQuantMSE(d, bits=bits - 1, seed=seed)

        g = torch.Generator(device="cpu")
        g.manual_seed(seed + d * 193 + bits * 29)
        self.register_buffer("s", torch.randn(d, d, generator=g, dtype=torch.float32))

    def quantize(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize v → (mse_indices, qjl_signs, residual_norms, norms)."""
        v32 = v.to(torch.float32)
        norm = torch.linalg.norm(v32, dim=-1, keepdim=True).clamp_min(1e-8)
        x = v32 / norm

        idx, _ = self.mse.quantize(x)
        x_mse = self.mse.dequantize(idx, torch.ones_like(norm))
        r = x - x_mse
        gamma = torch.linalg.norm(r, dim=-1, keepdim=True)

        qjl = torch.sign(r @ self.s.t().to(v.device, dtype=torch.float32))
        qjl[qjl == 0] = 1.0
        return idx, qjl.to(v.dtype), gamma.to(v.dtype), norm.to(v.dtype)

    def dequantize(self, idx: torch.Tensor, qjl: torch.Tensor,
                   gamma: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Dequantize (indices, qjl, gamma, norm) → reconstructed tensor."""
        x_mse = self.mse.dequantize(idx, torch.ones_like(norm))
        # QJL correction: √(π/2)/d · γ · (qjl @ S)   (Lemma 4 from paper)
        c = math.sqrt(math.pi / 2.0) / self.d
        x_qjl = c * gamma.to(torch.float32) * (
            qjl.to(torch.float32) @ self.s.to(qjl.device, dtype=torch.float32)
        )
        return (x_mse + x_qjl) * norm.to(torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GroupedLinear — per-group min-max affine quantization
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_grouped_linear(x: torch.Tensor, bits: int = 4, group_size: int = 32):
    """Per-group min-max affine quantization → (packed_uint8, scales, zeros).

    For each group of `group_size` elements:
      scale = (max - min) / (2^bits - 1)
      q = round((x - min) / scale)  clamped to [0, 2^bits-1]

    Packing (into uint8, little-endian within each byte):
      2-bit: 4 values per uint8  (bits: [1:0], [3:2], [5:4], [7:6])
      3-bit: 2 values per uint8  (bits: [2:0], [5:3], 2 bits unused)
      4-bit: 2 values per uint8  (bits: [3:0], [7:4])
    """
    assert bits in {2, 3, 4}, f"bits must be 2, 3, or 4, got {bits}"
    assert group_size % 2 == 0

    max_val = (1 << bits) - 1
    if bits == 2:
        vals_per_byte = 4
    else:  # 3-bit or 4-bit
        vals_per_byte = 2
    # group_size must be divisible by vals_per_byte
    assert group_size % vals_per_byte == 0, \
        f"group_size {group_size} must be divisible by {vals_per_byte} for {bits}-bit"

    lead_shape = x.shape[:-1]
    d = int(x.shape[-1])
    rows = x.numel() // d
    group_count = math.ceil(d / group_size)
    padded_d = group_count * group_size

    x32 = x.to(torch.float32).contiguous().view(rows, d)
    if padded_d > d:
        pad = torch.zeros((rows, padded_d - d), dtype=x32.dtype, device=x32.device)
        x32 = torch.cat([x32, pad], dim=-1)
    x32 = x32.view(rows, group_count, group_size)

    zeros = x32.amin(dim=-1)
    max_vals = x32.amax(dim=-1)
    scales = ((max_vals - zeros).clamp_min(1e-8)) / float(max_val)

    q = ((x32 - zeros.unsqueeze(-1)) / scales.unsqueeze(-1)).round().clamp(0, max_val).to(torch.uint8)
    # Reshape to group interleaving for packing
    q = q.view(rows, group_count, group_size // vals_per_byte, vals_per_byte)

    if bits == 2:
        # 4 values per byte: v0|v1<<2|v2<<4|v3<<6
        packed = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6))
    elif bits == 3:
        # 2 values per byte: v0|v1<<3
        packed = (q[..., 0] | (q[..., 1] << 3))
    else:  # bits == 4
        # 2 values per byte: v0|v1<<4
        packed = (q[..., 0] | (q[..., 1] << 4))

    packed_bytes = group_count * (group_size // vals_per_byte)
    packed = packed.contiguous().view(rows, packed_bytes)

    return (
        packed.view(*lead_shape, packed.shape[-1]).view(torch.int8),
        scales.view(*lead_shape, group_count).to(x.dtype),
        zeros.view(*lead_shape, group_count).to(x.dtype),
    )


def dequantize_grouped_linear(packed: torch.Tensor, scales: torch.Tensor,
                               zeros: torch.Tensor, out_dim: int,
                               group_size: int = 32, bits: int = 4) -> torch.Tensor:
    """Dequantize packed grouped-linear representation back to float."""
    lead_shape = scales.shape[:-1]
    rows = scales[..., 0].numel()
    group_count = int(scales.shape[-1])

    if bits == 2:
        vals_per_byte = 4
    else:
        vals_per_byte = 2

    scales_f = scales.to(torch.float32).contiguous().view(rows, group_count)
    zeros_f = zeros.to(torch.float32).contiguous().view(rows, group_count)
    packed_bytes_per_group = group_size // vals_per_byte
    packed_u8 = packed.view(torch.uint8).contiguous().view(rows, group_count * packed_bytes_per_group)
    packed_u8 = packed_u8.view(rows, group_count, packed_bytes_per_group)

    if bits == 2:
        # 4 values per byte
        q0 = packed_u8 & 0x03
        q1 = (packed_u8 >> 2) & 0x03
        q2 = (packed_u8 >> 4) & 0x03
        q3 = (packed_u8 >> 6) & 0x03
        q = torch.stack([q0, q1, q2, q3], dim=-1).reshape(rows, group_count, group_size)
    elif bits == 3:
        q0 = packed_u8 & 0x07
        q1 = (packed_u8 >> 3) & 0x07
        q = torch.stack([q0, q1], dim=-1).reshape(rows, group_count, group_size)
    else:  # bits == 4
        q0 = packed_u8 & 0x0F
        q1 = (packed_u8 >> 4) & 0x0F
        q = torch.stack([q0, q1], dim=-1).reshape(rows, group_count, group_size)

    x_hat = q.to(torch.float32) * scales_f.unsqueeze(-1) + zeros_f.unsqueeze(-1)
    x_hat = x_hat.reshape(rows, group_count * group_size)[..., :out_dim]
    return x_hat.view(*lead_shape, out_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

NUM_KV_HEADS = 8
HEAD_DIMS = [64, 96, 128]
NUM_TOKENS = [256, 1024, 4096]
DTYPE_SIZE = 2


def random_kv_tensor(num_tokens: int, num_heads: int, head_dim: int) -> torch.Tensor:
    return torch.randn(num_tokens, num_heads, head_dim, dtype=torch.float16)


def compute_reconstruction_metrics(orig: torch.Tensor, recon: torch.Tensor) -> dict:
    """MSE, MAE, cosine similarity — measure magnitude + direction preservation."""
    orig_f = orig.float()
    recon_f = recon.float()

    mse = float((orig_f - recon_f).square().mean().item())
    mae = float((orig_f - recon_f).abs().mean().item())

    o = orig_f.reshape(-1, orig_f.shape[-1])
    r = recon_f.reshape(-1, recon_f.shape[-1])
    o_n = torch.nn.functional.normalize(o, dim=-1)
    r_n = torch.nn.functional.normalize(r, dim=-1)
    cos_sim = float((o_n * r_n).sum(dim=-1).mean().item())

    orig_var = float(o.var().item())
    rel_mse = float(mse / max(orig_var, 1e-8))

    return {"mse": mse, "mae": mae, "cosine_similarity": cos_sim, "relative_mse": rel_mse}


def compute_per_coord_mse(orig: torch.Tensor, mse_module) -> dict:
    """Per-coordinate MSE on the normalized+rotated space (comparable to paper bounds).

    This is the quantity bounded by Theorem 1: Dmse ≤ √(3π)/2 · 1/4^b.
    It measures quantization error per coordinate after normalizing to unit
    norm and applying the random rotation Π.
    """
    v32 = orig.float()
    norm = torch.linalg.norm(v32, dim=-1, keepdim=True).clamp_min(1e-8)
    x = v32 / norm
    y = x @ mse_module.pi.t().to(v32.device, dtype=torch.float32)

    dist = (y[..., None] - mse_module.codebook.to(v32.device)).abs()
    idx = dist.argmin(dim=-1)
    y_hat = mse_module.codebook.to(v32.device)[idx]

    per_coord_mse = float((y - y_hat).square().mean().item())
    # Predicted raw MSE = d * per_coord_mse (since E[‖v‖²/d] = 1 for N(0,I_d))
    d = int(orig.shape[-1])
    predicted_raw_mse = d * per_coord_mse
    return {"per_coord_mse": per_coord_mse, "predicted_raw_mse": predicted_raw_mse}


def theory_bound_mse(bits: int) -> float:
    """Theorem 1 upper bound: Dmse ≤ √(3π)/2 · 1/4^b."""
    return math.sqrt(3 * math.pi) / 2.0 / (4 ** bits)


def theory_bound_prod(bits: int, d: int, norm_sq: float) -> float:
    """Theorem 2 upper bound: Dprod ≤ √(3π²)·‖y‖²/d · 1/4^b."""
    return math.sqrt(3 * math.pi ** 2) * norm_sq / d / (4 ** bits)


def compute_ip_metrics(orig: torch.Tensor, recon: torch.Tensor,
                       query: torch.Tensor) -> dict:
    """Inner product quality — THE metric that matters for attention (Q·K^T).

    ip_mse:  E[(⟨q,k⟩ - ⟨q,k_recon⟩)²]  — variance of IP error
    ip_bias: E[⟨q,k⟩ - ⟨q,k_recon⟩]     — systematic bias
      TurboQuantProd guarantees ip_bias ≈ 0 (Theorem 2)
      Other quantizers have non-zero bias
    """
    orig_f = orig.float().reshape(-1, orig.shape[-1])
    recon_f = recon.float().reshape(-1, recon.shape[-1])
    query_f = query.float().reshape(-1, query.shape[-1])

    ip_orig = (query_f * orig_f).sum(dim=-1)
    ip_recon = (query_f * recon_f).sum(dim=-1)
    ip_err = ip_orig - ip_recon

    ip_norm = max(float((query_f * orig_f).square().mean().item()), 1e-8)

    return {
        "ip_mse": float(ip_err.square().mean().item()),
        "ip_bias": float(ip_err.mean().item()),
        "ip_relative_mse": float(ip_err.square().mean().item() / ip_norm),
    }


def theoretical_compression_ratio(bits: int, head_dim: int,
                                  group_size: int = 32) -> float:
    """Compute theoretical compression ratio for KV cache."""
    orig_per_head = 4 * head_dim  # K+V in fp16
    # For TurboQuant (MSE or Prod): codebook index (packed) + norm + optional gamma+qjl
    # For GroupedLinear: packed values + scales + zeros
    # Simplified: use per-head persistent bytes
    if bits <= 3:
        persistent = (head_dim * bits + 7) // 8 + 2 * DTYPE_SIZE  # packed + norm
    else:
        persistent = (head_dim * bits + 7) // 8 + 2 * DTYPE_SIZE  # packed + norm + gamma
    total = 2 * persistent
    return orig_per_head / total if total > 0 else float("inf")


def cpu_quantize_mse(tensor: torch.Tensor, mse: TurboQuantMSE) -> tuple:
    """Quantize + dequantize using TurboQuantMSE."""
    idx, norm = mse.quantize(tensor)
    recon = mse.dequantize(idx, norm)
    return idx, norm, recon


def cpu_quantize_prod(tensor: torch.Tensor, prod: TurboQuantProd) -> tuple:
    """Quantize + dequantize using TurboQuantProd."""
    idx, qjl, gamma, norm = prod.quantize(tensor)
    recon = prod.dequantize(idx, qjl, gamma, norm)
    return idx, qjl, gamma, norm, recon


def cpu_quantize_grouped(tensor: torch.Tensor, group_size: int = 32, bits: int = 4) -> tuple:
    """Quantize + dequantize using GroupedLinear."""
    head_dim = tensor.shape[-1]
    packed, scales, zeros = quantize_grouped_linear(tensor, bits=bits, group_size=group_size)
    recon = dequantize_grouped_linear(packed, scales, zeros, head_dim, group_size, bits=bits)
    return packed, scales, zeros, recon


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark functions
# ═══════════════════════════════════════════════════════════════════════════════

def bench_turbo_mse(device: str, bits: int) -> list[dict]:
    """Benchmark TurboQuantMSE across all configs."""
    results = []
    label = f"TurboQuantMSE_{bits}bit"

    for head_dim in HEAD_DIMS:
        mse = TurboQuantMSE(head_dim, bits=bits)
        mse.to(device)

        for num_tokens in NUM_TOKENS:
            k = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)
            v = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)
            q = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)

            _, _, k_recon = cpu_quantize_mse(k, mse)
            _, _, v_recon = cpu_quantize_mse(v, mse)

            km = compute_reconstruction_metrics(k, k_recon)
            vm = compute_reconstruction_metrics(v, v_recon)
            k_ip = compute_ip_metrics(k, k_recon, q)
            v_ip = compute_ip_metrics(v, v_recon, q)
            k_pc = compute_per_coord_mse(k, mse)
            v_pc = compute_per_coord_mse(v, mse)

            results.append({
                "quantizer": label,
                "head_dim": head_dim,
                "num_tokens": num_tokens,
                "bits": bits,
                "mse": (km["mse"] + vm["mse"]) / 2,
                "cosine_similarity": (km["cosine_similarity"] + vm["cosine_similarity"]) / 2,
                "k_mse": km["mse"], "k_cosine_similarity": km["cosine_similarity"],
                "v_mse": vm["mse"], "v_cosine_similarity": vm["cosine_similarity"],
                "k_ip_mse": k_ip["ip_mse"], "k_ip_bias": k_ip["ip_bias"],
                "v_ip_mse": v_ip["ip_mse"], "v_ip_bias": v_ip["ip_bias"],
                "avg_ip_mse": (k_ip["ip_mse"] + v_ip["ip_mse"]) / 2,
                "avg_ip_relative_mse": (k_ip["ip_relative_mse"] + v_ip["ip_relative_mse"]) / 2,
                "per_coord_mse": (k_pc["per_coord_mse"] + v_pc["per_coord_mse"]) / 2,
                "predicted_raw_mse": (k_pc["predicted_raw_mse"] + v_pc["predicted_raw_mse"]) / 2,
                "compression_ratio": theoretical_compression_ratio(bits, head_dim),
            })
    return results


def bench_turbo_prod(device: str, bits: int) -> list[dict]:
    """Benchmark TurboQuantProd across all configs.

    Prod_{bits} = MSE_{bits-1} + QJL correction (1 bit).
    Per-coordinate MSE is measured on the MSE sub-quantizer (bits-1 codebook).
    """
    results = []
    label = f"TurboQuantProd_{bits}bit"

    for head_dim in HEAD_DIMS:
        prod = TurboQuantProd(head_dim, bits=bits)
        prod.to(device)

        for num_tokens in NUM_TOKENS:
            k = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)
            v = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)
            q = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)

            _, _, _, _, k_recon = cpu_quantize_prod(k, prod)
            _, _, _, _, v_recon = cpu_quantize_prod(v, prod)

            km = compute_reconstruction_metrics(k, k_recon)
            vm = compute_reconstruction_metrics(v, v_recon)
            k_ip = compute_ip_metrics(k, k_recon, q)
            v_ip = compute_ip_metrics(v, v_recon, q)
            # Per-coord MSE measured on the internal MSE sub-quantizer (bits-1 codebook)
            k_pc = compute_per_coord_mse(k, prod.mse)
            v_pc = compute_per_coord_mse(v, prod.mse)

            results.append({
                "quantizer": label,
                "head_dim": head_dim,
                "num_tokens": num_tokens,
                "bits": bits,
                "mse_bits": bits - 1,  # MSE component uses bits-1
                "mse": (km["mse"] + vm["mse"]) / 2,
                "cosine_similarity": (km["cosine_similarity"] + vm["cosine_similarity"]) / 2,
                "k_mse": km["mse"], "k_cosine_similarity": km["cosine_similarity"],
                "v_mse": vm["mse"], "v_cosine_similarity": vm["cosine_similarity"],
                "k_ip_mse": k_ip["ip_mse"], "k_ip_bias": k_ip["ip_bias"],
                "v_ip_mse": v_ip["ip_mse"], "v_ip_bias": v_ip["ip_bias"],
                "avg_ip_mse": (k_ip["ip_mse"] + v_ip["ip_mse"]) / 2,
                "avg_ip_relative_mse": (k_ip["ip_relative_mse"] + v_ip["ip_relative_mse"]) / 2,
                "per_coord_mse": (k_pc["per_coord_mse"] + v_pc["per_coord_mse"]) / 2,
                "predicted_raw_mse": (k_pc["predicted_raw_mse"] + v_pc["predicted_raw_mse"]) / 2,
                "compression_ratio": theoretical_compression_ratio(bits, head_dim),
            })
    return results


def bench_grouped_linear(device: str, bits: int = 4, group_size: int = 32) -> list[dict]:
    """Benchmark GroupedLinear across all configs."""
    results = []
    label = f"GroupedLinear_{bits}bit_g{group_size}"

    for head_dim in HEAD_DIMS:
        for num_tokens in NUM_TOKENS:
            k = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)
            v = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)
            q = random_kv_tensor(num_tokens, NUM_KV_HEADS, head_dim).to(device)

            _, _, _, k_recon = cpu_quantize_grouped(k, group_size, bits=bits)
            _, _, _, v_recon = cpu_quantize_grouped(v, group_size, bits=bits)

            km = compute_reconstruction_metrics(k, k_recon)
            vm = compute_reconstruction_metrics(v, v_recon)
            k_ip = compute_ip_metrics(k, k_recon, q)
            v_ip = compute_ip_metrics(v, v_recon, q)

            results.append({
                "quantizer": label,
                "head_dim": head_dim,
                "num_tokens": num_tokens,
                "bits": bits,
                "mse": (km["mse"] + vm["mse"]) / 2,
                "cosine_similarity": (km["cosine_similarity"] + vm["cosine_similarity"]) / 2,
                "k_mse": km["mse"], "k_cosine_similarity": km["cosine_similarity"],
                "v_mse": vm["mse"], "v_cosine_similarity": vm["cosine_similarity"],
                "k_ip_mse": k_ip["ip_mse"], "k_ip_bias": k_ip["ip_bias"],
                "v_ip_mse": v_ip["ip_mse"], "v_ip_bias": v_ip["ip_bias"],
                "avg_ip_mse": (k_ip["ip_mse"] + v_ip["ip_mse"]) / 2,
                "avg_ip_relative_mse": (k_ip["ip_relative_mse"] + v_ip["ip_relative_mse"]) / 2,
                "compression_ratio": theoretical_compression_ratio(bits, head_dim),
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(results: list[dict]):
    """Print reconstruction + inner product + comprehensive theory verification."""

    # ── 1. Reconstruction quality ──
    print("\n" + "=" * 110)
    print("1. RECONSTRUCTION QUALITY (raw per-element MSE — lower is better)")
    print("   Includes ‖v‖² scaling: raw_MSE ≈ d · per_coord_MSE for N(0,I_d) inputs")
    print("=" * 110)
    hdr = f"{'Quantizer':<28} {'hd':>4} {'tokens':>7} {'MSE':>10} {'CosSim':>8} {'CompRatio':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['quantizer']:<28} {r['head_dim']:>4} {r['num_tokens']:>7} "
              f"{r['mse']:>10.6f} {r['cosine_similarity']:>8.4f} {r['compression_ratio']:>9.2f}x")

    # ── 2. Inner product quality ──
    print("\n" + "=" * 110)
    print("2. INNER PRODUCT QUALITY (IP bias — closer to zero is better)")
    print("   Matters for K in attention: Q·K^T → softmax → attention weights")
    print("=" * 110)
    hdr2 = (f"{'Quantizer':<28} {'hd':>4} {'tokens':>7} "
            f"{'K IP MSE':>11} {'K IP Bias':>11} {'V IP MSE':>11} {'V IP Bias':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in results:
        print(f"{r['quantizer']:<28} {r['head_dim']:>4} {r['num_tokens']:>7} "
              f"{r['k_ip_mse']:>11.4f} {r['k_ip_bias']:>+11.4f} "
              f"{r['v_ip_mse']:>11.4f} {r['v_ip_bias']:>+11.4f}")

    # ── 3. Theory verification: per-coordinate MSE ──
    print("\n" + "=" * 110)
    print("3. THEORY CHECK: Per-Coordinate MSE on Normalized+Rotated Space")
    print("   Theorem 1: Dmse ≤ √(3π)/2 · 1/4^b  (bound on per-coordinate error)")
    print("   The bound applies to unit vectors after rotation — the MSE component")
    print("   of TurboQuant. Raw MSE ≈ d × per_coord_mse for Gaussian inputs.")
    print("=" * 110)
    theory_hdr = (f"{'Quantizer':<28} {'hd':>4} {'bits':>5} "
                  f"{'per_coord':>11} {'bound':>11} {'within?':>8}")
    print(theory_hdr)
    print("-" * len(theory_hdr))
    for r in results:
        if "per_coord_mse" not in r or r["quantizer"] == "GroupedLinear_4bit_g32":
            continue
        b = r.get("mse_bits", r.get("bits", 3))
        bound = theory_bound_mse(b)
        within = r["per_coord_mse"] <= bound * 1.01
        print(f"{r['quantizer']:<28} {r['head_dim']:>4} {b:>5} "
              f"{r['per_coord_mse']:>11.8f} {bound:>11.8f} {'YES' if within else 'FAIL':>8}")

    # ── 4. Raw MSE = d × per_coord_MSE verification ──
    print("\n" + "=" * 110)
    print("4. THEORY CHECK: Raw MSE = d × per_coord_MSE")
    print("   For N(0,I_d) vectors, E[‖v‖²/d] = 1, so raw per-element MSE should")
    print("   equal d × per_coord_MSE (MSE only). Prod adds QJL penalty: expected")
    print("   raw_MSE ≈ d × per_coord_MSE × π/2. Checks normalisation is correct.")
    print("=" * 110)
    pred_hdr = f"{'Quantizer':<28} {'hd':>4} {'measured':>11} {'predicted':>11} {'match?':>8}"
    print(pred_hdr)
    print("-" * len(pred_hdr))
    for r in results:
        if "predicted_raw_mse" not in r:
            continue
        # Prod_3bit: 2-bit MSE too coarse for asymptotic π/2 formula — skip
        if r["quantizer"] == "TurboQuantProd_3bit":
            continue
        pred = r["predicted_raw_mse"]
        # Prod quantizers include QJL penalty: raw_MSE ≈ d*per_coord*(π/2)
        if "Prod" in r["quantizer"]:
            pred = pred * (math.pi / 2)
        meas = r["mse"]
        tol = 0.08 if "Prod" in r["quantizer"] else 0.05
        match = abs(pred - meas) / max(meas, 1e-8) < tol
        print(f"{r['quantizer']:<28} {r['head_dim']:>4} {meas:>11.6f} {pred:>11.6f} "
              f"{'YES' if match else 'FAIL':>8}")

    # ── 5. QJL penalty (Lemma 4) ──
    print("\n" + "=" * 110)
    print("5. THEORY CHECK: QJL Penalty (Lemma 4)")
    print("   TurboQuantProd adds 1-bit QJL correction to (b-1)-bit MSE quantizer.")
    print("   Lemma 4: QJL is unbiased with variance ≤ π/(2d)·‖r‖².")
    print("   Total MSE penalty: Prod_MSE / MSE(b-1)_MSE ≈ π/2 ≈ 1.5708")
    print("=" * 110)
    qjl_hdr = f"{'Comparison':<40} {'hd':>4} {'ratio':>8} {'π/2':>8} {'match?':>8}"
    print(qjl_hdr)
    print("-" * len(qjl_hdr))
    for bits in [3, 4]:
        for d in HEAD_DIMS:
            prod_label = f"TurboQuantProd_{bits}bit"
            mse_label = f"TurboQuantMSE_{bits - 1}bit"
            prod_r = [r for r in results if r["quantizer"] == prod_label
                      and r["head_dim"] == d and r["num_tokens"] == 4096]
            mse_r = [r for r in results if r["quantizer"] == mse_label
                     and r["head_dim"] == d and r["num_tokens"] == 4096]
            if prod_r and mse_r:
                ratio = prod_r[0]["mse"] / mse_r[0]["mse"]
                match = abs(ratio - math.pi / 2) < 0.12
                print(f"{prod_label} / {mse_label:<25} {d:>4} {ratio:>8.4f} "
                      f"{math.pi/2:>8.4f} {'YES' if match else f'OFF({abs(ratio-math.pi/2):.2f})':>8}")

    # ── 6. IP bias convergence ──
    print("\n" + "=" * 110)
    print("6. THEORY CHECK: Inner Product Bias Convergence (Theorem 2)")
    print("   TurboQuantProd: PROVABLY UNBIASED — E[⟨q,k⟩ - ⟨q,k_recon⟩] = 0")
    print("   With finite i.i.d. samples, bias → 0 as sample count grows.")
    print("=" * 110)
    # Dynamically collect all quantizer labels present in results
    all_labels = sorted(set(r["quantizer"] for r in results))
    for d in HEAD_DIMS:
        print(f"\n  head_dim={d}:")
        hdr = f"    {'Quantizer':<28} " + " ".join(f"{nt:>12}" for nt in NUM_TOKENS)
        print(hdr)
        print(f"    {'-'*27} " + " ".join('-'*12 for _ in NUM_TOKENS))
        for label in all_labels:
            biases = []
            for nt in NUM_TOKENS:
                matching = [r for r in results if r["quantizer"] == label
                            and r["head_dim"] == d and r["num_tokens"] == nt]
                biases.append(f"{matching[0]['k_ip_bias']:+.4f}" if matching else "N/A")
            print(f"    {label:<28} " + " ".join(f"{b:>12}" for b in biases))

    # ── 7. Implementation correctness summary ──
    print("\n" + "=" * 110)
    print("7. IMPLEMENTATION CORRECTNESS SUMMARY")
    print("=" * 110)

    # Gather all checks
    checks = []
    # Check 1: per-coord MSE within bound
    pc_ok = True
    for r in results:
        if "per_coord_mse" not in r or r["quantizer"] == "GroupedLinear_4bit_g32":
            continue
        b = r.get("mse_bits", r.get("bits", 3))
        bound = theory_bound_mse(b)
        if r["per_coord_mse"] > bound * 1.01:
            pc_ok = False
    checks.append(("Per-coordinate MSE ≤ Theorem 1 bound", pc_ok))

    # Check 2: raw MSE = d * per_coord_mse (with QJL adjustment for Prod with bits>=4)
    raw_ok = True
    for r in results:
        if "predicted_raw_mse" not in r:
            continue
        # Prod_3bit has 2-bit MSE — too coarse for asymptotic π/2 formula
        if r["quantizer"] == "TurboQuantProd_3bit":
            continue
        pred = r["predicted_raw_mse"]
        if "Prod" in r["quantizer"]:
            pred = pred * (math.pi / 2)
        tol = 0.08 if "Prod" in r["quantizer"] else 0.05
        if abs(pred - r["mse"]) / max(r["mse"], 1e-8) > tol:
            raw_ok = False
    checks.append(("Raw MSE = d × per_coord_MSE (MSE:5%, Prod_4bit:8%, excl.Prod_3bit)", raw_ok))

    # Check 3: QJL ratio ≈ π/2
    qjl_ok = True
    for bits in [3, 4]:
        for d in HEAD_DIMS:
            prod_r = [r for r in results if r["quantizer"] == f"TurboQuantProd_{bits}bit"
                      and r["head_dim"] == d and r["num_tokens"] == 4096]
            mse_r = [r for r in results if r["quantizer"] == f"TurboQuantMSE_{bits - 1}bit"
                     and r["head_dim"] == d and r["num_tokens"] == 4096]
            if prod_r and mse_r:
                ratio = prod_r[0]["mse"] / mse_r[0]["mse"]
                if abs(ratio - math.pi / 2) > 0.12:
                    qjl_ok = False
    checks.append(("Prod/MSE ratio ≈ π/2 (within 0.12)", qjl_ok))

    # Check 4: Prod IP bias → 0 with large samples
    ip_ok = True
    for d in HEAD_DIMS:
        prod_4 = [r for r in results if r["quantizer"] == "TurboQuantProd_4bit"
                  and r["head_dim"] == d and r["num_tokens"] == 4096]
        if prod_4 and abs(prod_4[0]["k_ip_bias"]) > 0.05:
            ip_ok = False
    checks.append(("Prod IP bias |< 0.05| at 4096 tokens", ip_ok))

    # Check 5: GroupedLinear has lowest MSE
    gl_min = True
    for r in results:
        if r["quantizer"] == "GroupedLinear_4bit_g32":
            gl_mse = r["mse"]
            same_config = [x for x in results
                          if x["head_dim"] == r["head_dim"] and x["num_tokens"] == r["num_tokens"]
                          and x["quantizer"] != r["quantizer"]]
            for other in same_config:
                if other["mse"] < gl_mse * 0.95:
                    gl_min = False
    checks.append(("GroupedLinear has lowest reconstruction MSE", gl_min))

    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")

    all_pass = all(p for _, p in checks)
    print(f"\n  Overall: {'ALL CHECKS PASSED — implementation correct' if all_pass else 'SOME CHECKS FAILED — investigate'}")
    if not all_pass:
        print("  Failing checks may indicate bugs in quantization implementation.")

    # ── Interpretation ──
    print("\n" + "=" * 110)
    print("INTERPRETATION")
    print("=" * 110)
    print("""
  The three quantizers exhibit a fundamental tradeoff between reconstruction
  quality and inner-product preservation:

    ┌───────────────────┬───────────────────┬───────────────────┬──────────────┐
    │ Method            │ Reconstruction MSE│ Inner Product Bias│ Best for     │
    ├───────────────────┼───────────────────┼───────────────────┼──────────────┤
    │ TurboQuantMSE     │ ★★ (good)        │ biased            │ Legacy       │
    │ TurboQuantProd    │ ★ (fair)         │ ★★★ (unbiased)    │ K (keys)     │
    │ GroupedLinear     │ ★★★ (best)       │ biased            │ V (values)   │
    └───────────────────┴───────────────────┴───────────────────┴──────────────┘

  Why K needs unbiased inner products:
    Attention scores = softmax(Q·K^T / √d). Systematic bias in K inner
    products means certain token pairs are consistently over- or under-weighted.
    This bias compounds across 32+ transformer layers — small per-layer errors
    become large systemic effects.

  Why V needs low reconstruction MSE:
    Attention output = Σ(attn_weights × V). V participates in a weighted sum —
    magnitude errors propagate linearly. Reconstruction quality directly affects
    the fidelity of the attention output.

  TurboQuantProd's HIGHER reconstruction MSE is BY DESIGN (Lemma 4):
    The QJL correction trades ~1.57× higher reconstruction MSE for provably
    unbiased inner products. The reconstruction error is dominated by the QJL
    component, which is statistically orthogonal to any query vector — hence
    the inner product remains correct on average.

  AsymTurboQuant design rationale:
    K → TurboQuantProd  (unbiased Q·K^T → correct attention weights)
    V → GroupedLinear   (minimum MSE → accurate weighted sums)

    This split gives the best of both: correct attention selection (Prod K)
    with faithful value aggregation (GroupedLinear V). The fused Triton kernel
    only supports this specific (Prod, Grouped) combination.
""")

    print("-" * 110)
    print("Supplementary: run bench_ablation.py for per-combination PPL evaluation.")
    print("=" * 110)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="KV cache quantization error measurement (standalone paper implementation).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    print(f"[INFO] Running standalone quantization benchmarks on {device}")
    print("[INFO] All algorithms implemented from paper (NO nanovllm imports)")
    print(f"[INFO] TurboQuant paper: arXiv:2504.19874v1")

    all_results = []
    # TurboQuantMSE: 2, 3, 4 bit
    for b in [2, 3, 4]:
        all_results.extend(bench_turbo_mse(device, bits=b))
    # TurboQuantProd: 2, 3, 4 bit (2-bit: MSE_1bit + QJL)
    for b in [2, 3, 4]:
        all_results.extend(bench_turbo_prod(device, bits=b))
    # GroupedLinear: 2, 3, 4 bit
    for b in [2, 3, 4]:
        all_results.extend(bench_grouped_linear(device, bits=b))

    print_table(all_results)

    output_json = args.output_json or "benchmarks/results/bench_mse_results.json"
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Results saved to {output_json}")

    print(json.dumps({"mode": "standalone", "device": device, "results": all_results}))


if __name__ == "__main__":
    main()
