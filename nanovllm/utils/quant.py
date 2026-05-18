import math
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn


class MSEQuantized(NamedTuple):
    """Output of TurboQuant MSE quantization."""
    indices: torch.Tensor
    norms: torch.Tensor
    bits: int


class ProdQuantized(NamedTuple):
    """Output of TurboQuant inner-product quantization."""
    mse_indices: torch.Tensor
    qjl_signs: torch.Tensor
    residual_norms: torch.Tensor
    norms: torch.Tensor
    mse_bits: int


class AsymQuantizedV(NamedTuple):
    """Output of grouped-linear value quantization."""
    packed: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int
    group_size: int

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

try:
    from nanovllm.layers.fused_quant_attn import fused_compress_mse
except Exception:
    fused_compress_mse = None


if _TRITON_AVAILABLE:

    @triton.jit
    def _nearest_codebook_3bit_kernel(
        y_ptr,
        idx_ptr,
        yhat_ptr,
        codebook_ptr,
        stride_y_m,
        stride_y_d,
        stride_idx_m,
        stride_idx_d,
        stride_yhat_m,
        stride_yhat_d,
        d,
        BLOCK_D: tl.constexpr,  # type: ignore[valid-type]
    ):
        pid_m = tl.program_id(0)
        pid_blk = tl.program_id(1)
        offs = pid_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < d

        y = tl.load(y_ptr + pid_m * stride_y_m + offs * stride_y_d, mask=mask, other=0.0)

        c0 = tl.load(codebook_ptr + 0)
        c1 = tl.load(codebook_ptr + 1)
        c2 = tl.load(codebook_ptr + 2)
        c3 = tl.load(codebook_ptr + 3)
        c4 = tl.load(codebook_ptr + 4)
        c5 = tl.load(codebook_ptr + 5)
        c6 = tl.load(codebook_ptr + 6)
        c7 = tl.load(codebook_ptr + 7)

        best_dist = tl.abs(y - c0)
        best_idx = tl.zeros([BLOCK_D], dtype=tl.int32)
        best_val = c0 + tl.zeros([BLOCK_D], dtype=tl.float32)

        d1 = tl.abs(y - c1)
        take = d1 < best_dist
        best_dist = tl.where(take, d1, best_dist)
        best_idx = tl.where(take, 1, best_idx)
        best_val = tl.where(take, c1, best_val)

        d2 = tl.abs(y - c2)
        take = d2 < best_dist
        best_dist = tl.where(take, d2, best_dist)
        best_idx = tl.where(take, 2, best_idx)
        best_val = tl.where(take, c2, best_val)

        d3 = tl.abs(y - c3)
        take = d3 < best_dist
        best_dist = tl.where(take, d3, best_dist)
        best_idx = tl.where(take, 3, best_idx)
        best_val = tl.where(take, c3, best_val)

        d4 = tl.abs(y - c4)
        take = d4 < best_dist
        best_dist = tl.where(take, d4, best_dist)
        best_idx = tl.where(take, 4, best_idx)
        best_val = tl.where(take, c4, best_val)

        d5 = tl.abs(y - c5)
        take = d5 < best_dist
        best_dist = tl.where(take, d5, best_dist)
        best_idx = tl.where(take, 5, best_idx)
        best_val = tl.where(take, c5, best_val)

        d6 = tl.abs(y - c6)
        take = d6 < best_dist
        best_dist = tl.where(take, d6, best_dist)
        best_idx = tl.where(take, 6, best_idx)
        best_val = tl.where(take, c6, best_val)

        d7 = tl.abs(y - c7)
        take = d7 < best_dist
        best_idx = tl.where(take, 7, best_idx)
        best_val = tl.where(take, c7, best_val)

        tl.store(idx_ptr + pid_m * stride_idx_m + offs * stride_idx_d, best_idx.to(tl.uint8), mask=mask)
        tl.store(yhat_ptr + pid_m * stride_yhat_m + offs * stride_yhat_d, best_val, mask=mask)


    @triton.jit
    def _pack_idx_qjl_4bit_kernel(
        idx_ptr,
        qjl_ptr,
        out_ptr,
        stride_idx_m,
        stride_idx_d,
        stride_qjl_m,
        stride_qjl_d,
        stride_out_m,
        stride_out_d,
        d,
        packed_d,
        BLOCK_P: tl.constexpr,  # type: ignore[valid-type]
    ):
        pid_m = tl.program_id(0)
        pid_blk = tl.program_id(1)
        offs_p = pid_blk * BLOCK_P + tl.arange(0, BLOCK_P)
        mask_p = offs_p < packed_d

        d0 = offs_p * 2
        d1 = d0 + 1
        mask0 = mask_p & (d0 < d)
        mask1 = mask_p & (d1 < d)

        idx0 = tl.load(idx_ptr + pid_m * stride_idx_m + d0 * stride_idx_d, mask=mask0, other=0).to(tl.int32)
        idx1 = tl.load(idx_ptr + pid_m * stride_idx_m + d1 * stride_idx_d, mask=mask1, other=0).to(tl.int32)

        q0 = tl.load(qjl_ptr + pid_m * stride_qjl_m + d0 * stride_qjl_d, mask=mask0, other=1.0)
        q1 = tl.load(qjl_ptr + pid_m * stride_qjl_m + d1 * stride_qjl_d, mask=mask1, other=1.0)

        b0 = tl.where(q0 > 0, 1, 0)
        b1 = tl.where(q1 > 0, 1, 0)
        nib0 = ((idx0 << 1) | b0) & 0x0F
        nib1 = ((idx1 << 1) | b1) & 0x0F
        packed = (nib0 | (nib1 << 4)).to(tl.uint8)

        tl.store(out_ptr + pid_m * stride_out_m + offs_p * stride_out_d, packed, mask=mask_p)


    @triton.jit
    def _unpack_prod4_kernel(
        packed_ptr,
        yhat_ptr,
        qjl_ptr,
        codebook_ptr,
        stride_packed_m,
        stride_packed_d,
        stride_yhat_m,
        stride_yhat_d,
        stride_qjl_m,
        stride_qjl_d,
        d,
        packed_d,
        BLOCK_P: tl.constexpr,  # type: ignore[valid-type]
    ):
        pid_m = tl.program_id(0)
        pid_blk = tl.program_id(1)
        offs_p = pid_blk * BLOCK_P + tl.arange(0, BLOCK_P)
        mask_p = offs_p < packed_d

        packed = tl.load(packed_ptr + pid_m * stride_packed_m + offs_p * stride_packed_d, mask=mask_p, other=0).to(tl.int32)

        nib0 = packed & 0x0F
        nib1 = (packed >> 4) & 0x0F
        idx0 = nib0 >> 1
        idx1 = nib1 >> 1
        bit0 = nib0 & 1
        bit1 = nib1 & 1

        y0 = tl.load(codebook_ptr + idx0, mask=mask_p, other=0.0)
        y1 = tl.load(codebook_ptr + idx1, mask=mask_p, other=0.0)
        q0 = tl.where(bit0 > 0, 1.0, -1.0)
        q1 = tl.where(bit1 > 0, 1.0, -1.0)

        d0 = offs_p * 2
        d1 = d0 + 1
        mask0 = mask_p & (d0 < d)
        mask1 = mask_p & (d1 < d)

        tl.store(yhat_ptr + pid_m * stride_yhat_m + d0 * stride_yhat_d, y0, mask=mask0)
        tl.store(qjl_ptr + pid_m * stride_qjl_m + d0 * stride_qjl_d, q0, mask=mask0)
        tl.store(yhat_ptr + pid_m * stride_yhat_m + d1 * stride_yhat_d, y1, mask=mask1)
        tl.store(qjl_ptr + pid_m * stride_qjl_m + d1 * stride_qjl_d, q1, mask=mask1)


def _triton_prod4_enabled(bits: int, device: torch.device) -> bool:
    return bits == 4 and _TRITON_AVAILABLE and device.type == "cuda"


def _triton_find_codebook_indices_3bit(y: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, d = y.shape
    idx = torch.empty((rows, d), dtype=torch.uint8, device=y.device)
    y_hat = torch.empty_like(y)
    block_d = 128 if d >= 128 else 64
    grid = (rows, triton.cdiv(d, block_d))
    _nearest_codebook_3bit_kernel[grid](
        y,
        idx,
        y_hat,
        codebook,
        y.stride(0),
        y.stride(1),
        idx.stride(0),
        idx.stride(1),
        y_hat.stride(0),
        y_hat.stride(1),
        d,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return idx, y_hat


def _triton_pack_idx_qjl_4bit(idx: torch.Tensor, qjl_logits: torch.Tensor) -> torch.Tensor:
    rows, d = idx.shape
    packed_d = (d + 1) // 2
    packed = torch.empty((rows, packed_d), dtype=torch.uint8, device=idx.device)
    block_p = 128
    grid = (rows, triton.cdiv(packed_d, block_p))
    _pack_idx_qjl_4bit_kernel[grid](
        idx,
        qjl_logits,
        packed,
        idx.stride(0),
        idx.stride(1),
        qjl_logits.stride(0),
        qjl_logits.stride(1),
        packed.stride(0),
        packed.stride(1),
        d,
        packed_d,
        BLOCK_P=block_p,
        num_warps=4,
    )
    return packed


def _triton_unpack_prod4(packed: torch.Tensor, codebook: torch.Tensor, out_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows, packed_d = packed.shape
    y_hat = torch.empty((rows, out_dim), dtype=torch.float32, device=packed.device)
    qjl = torch.empty((rows, out_dim), dtype=torch.float32, device=packed.device)
    block_p = 128
    grid = (rows, triton.cdiv(packed_d, block_p))
    _unpack_prod4_kernel[grid](
        packed,
        y_hat,
        qjl,
        codebook,
        packed.stride(0),
        packed.stride(1),
        y_hat.stride(0),
        y_hat.stride(1),
        qjl.stride(0),
        qjl.stride(1),
        out_dim,
        packed_d,
        BLOCK_P=block_p,
        num_warps=4,
    )
    return y_hat, qjl


def _triton_quantize_prod4(
    tensor: torch.Tensor,
    pi: torch.Tensor,
    codebook: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lead_shape = tensor.shape[:-1]
    d = tensor.shape[-1]
    rows = tensor.numel() // d

    v = tensor.to(torch.float32).contiguous().view(rows, d)
    pi32 = pi.to(tensor.device, dtype=torch.float32)
    codebook32 = codebook.to(tensor.device, dtype=torch.float32).contiguous()
    s32 = s.to(tensor.device, dtype=torch.float32)

    norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
    x = v / norm
    y = x @ pi32.t()

    idx, y_hat = _triton_find_codebook_indices_3bit(y, codebook32)
    x_mse = y_hat @ pi32
    r = x - x_mse
    gamma = torch.linalg.norm(r, dim=-1, keepdim=True)
    qjl_logits = r @ s32.t()

    packed = _triton_pack_idx_qjl_4bit(idx, qjl_logits)
    packed = packed.view(*lead_shape, packed.shape[-1]).view(torch.int8)
    gamma = gamma.view(*lead_shape, 1)
    norm = norm.view(*lead_shape, 1)
    return packed, gamma, norm


def _fused_turboquant_and_cache_kernel(
    tensor: torch.Tensor,
    pi: torch.Tensor,
    codebook: torch.Tensor,
    s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Single entrypoint for K quantization + packing used by KV cache writers.
    return _triton_quantize_prod4(tensor, pi, codebook, s)


def _triton_dequantize_prod4(
    q_tensor: torch.Tensor,
    scales: torch.Tensor,
    pi: torch.Tensor,
    codebook: torch.Tensor,
    s: torch.Tensor,
    out_dim: int,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    lead_shape = scales.shape[:-1]
    rows = scales[..., 0].numel()

    packed = q_tensor.view(torch.uint8).contiguous().view(rows, q_tensor.shape[-1])
    pi32 = pi.to(q_tensor.device, dtype=torch.float32)
    codebook32 = codebook.to(q_tensor.device, dtype=torch.float32).contiguous()
    s32 = s.to(q_tensor.device, dtype=torch.float32)

    y_hat, qjl = _triton_unpack_prod4(packed, codebook32, out_dim)
    x_mse = y_hat @ pi32

    gamma = scales[..., 0:1].to(torch.float32).contiguous().view(rows, 1)
    norm = scales[..., 1:2].to(torch.float32).contiguous().view(rows, 1)
    c = math.sqrt(math.pi / 2.0) / float(out_dim)
    x_qjl = c * gamma * (qjl @ s32)
    x_hat = (x_mse + x_qjl) * norm
    x_hat = x_hat.view(*lead_shape, out_dim)
    if out is not None:
        if tuple(out.shape) != tuple(x_hat.shape):
            raise ValueError(f"dequant out shape mismatch: expected {tuple(x_hat.shape)}, got {tuple(out.shape)}")
        out.copy_(x_hat.to(out.dtype))
        return out
    return x_hat.to(out_dtype)


def _quantize_grouped_linear_4bit(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(f"group_size must be positive and even, got {group_size}")

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
    scales = ((max_vals - zeros).clamp_min(1e-8)) / 15.0

    q = ((x32 - zeros.unsqueeze(-1)) / scales.unsqueeze(-1)).round().clamp(0, 15).to(torch.uint8)
    q0 = q[..., 0::2]
    q1 = q[..., 1::2]
    packed = (q0 | (q1 << 4)).contiguous().view(rows, group_count * (group_size // 2))

    return (
        packed.view(*lead_shape, packed.shape[-1]).view(torch.int8),
        scales.view(*lead_shape, group_count).to(x.dtype),
        zeros.view(*lead_shape, group_count).to(x.dtype),
    )


def _dequantize_grouped_linear_4bit(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    out_dim: int,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError(f"group_size must be positive and even, got {group_size}")

    lead_shape = scales.shape[:-1]
    rows = scales[..., 0].numel()
    group_count = int(scales.shape[-1])
    scales_flat = scales.to(torch.float32).contiguous().view(rows, group_count)
    zeros_flat = zeros.to(torch.float32).contiguous().view(rows, group_count)
    packed_u8 = packed.view(torch.uint8).contiguous().view(rows, group_count * (group_size // 2))
    packed_u8 = packed_u8.view(rows, group_count, group_size // 2)

    q0 = packed_u8 & 0x0F
    q1 = (packed_u8 >> 4) & 0x0F
    q = torch.stack([q0, q1], dim=-1).reshape(rows, group_count, group_size)

    x_hat = q.to(torch.float32) * scales_flat.unsqueeze(-1) + zeros_flat.unsqueeze(-1)
    x_hat = x_hat.reshape(rows, group_count * group_size)[..., :out_dim]
    x_hat = x_hat.view(*lead_shape, out_dim)
    if out is not None:
        if tuple(out.shape) != tuple(x_hat.shape):
            raise ValueError(f"dequant out shape mismatch: expected {tuple(x_hat.shape)}, got {tuple(out.shape)}")
        out.copy_(x_hat.to(out.dtype))
        return out
    return x_hat.to(out_dtype)


class TurboQuantMSE(nn.Module):
    def __init__(self, d: int, bits: int = 3, seed: int = 0):
        super().__init__()
        assert bits >= 1
        self.d = int(d)
        self.bits = int(bits)

        g = torch.Generator(device="cpu")
        g.manual_seed(seed + self.d * 131 + self.bits * 17)

        a = torch.randn(self.d, self.d, generator=g, device="cpu", dtype=torch.float32)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diag(r))
        self.register_buffer("pi", q)

        num_centroids = 1 << self.bits
        sample = torch.randn(200000, generator=g, device="cpu", dtype=torch.float32) * (1.0 / math.sqrt(self.d))
        qs = torch.linspace(
            1.0 / (2 * num_centroids),
            1.0 - 1.0 / (2 * num_centroids),
            num_centroids,
            device="cpu",
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
        self.register_buffer("codebook", centroids.sort()[0])

    def quantize(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: fused_compress_mse returns packed uint8 (byte-pair), but dequantize
        # and callers expect unpacked int64 codebook indices. Skip fused path for now.
        v32 = v.to(torch.float32)
        norm = torch.linalg.norm(v32, dim=-1, keepdim=True).clamp_min(1e-8)
        x = v32 / norm
        y = x @ self.pi.t().to(v.device, dtype=torch.float32)
        dist = (y[..., None] - self.codebook.to(v.device)).abs()
        idx = dist.argmin(dim=-1).to(torch.int64)
        return idx, norm.to(v.dtype)

    def dequantize(self, idx: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        y_hat = self.codebook.to(idx.device, dtype=torch.float32)[idx]
        x_hat = y_hat @ self.pi.to(idx.device, dtype=torch.float32)
        return x_hat * norm


class TurboQuantProd(nn.Module):
    def __init__(self, d: int, bits: int = 3, seed: int = 0):
        super().__init__()
        assert bits >= 2
        self.d = int(d)
        self.bits = int(bits)
        self.mse = TurboQuantMSE(self.d, bits=self.bits - 1, seed=seed)

        g = torch.Generator(device="cpu")
        g.manual_seed(seed + self.d * 193 + self.bits * 29)
        self.register_buffer("s", torch.randn(self.d, self.d, generator=g, device="cpu", dtype=torch.float32))

    def quantize(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def dequantize(self, idx: torch.Tensor, qjl: torch.Tensor, gamma: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        x_mse = self.mse.dequantize(idx, torch.ones_like(norm))
        c = math.sqrt(math.pi / 2.0) / self.d
        x_qjl = c * gamma.to(torch.float32) * (qjl.to(torch.float32) @ self.s.to(qjl.device, dtype=torch.float32))
        x_hat = x_mse + x_qjl
        return x_hat * norm


class AsymTurboQuantKVQuantizer:
    def __init__(self, head_dim: int | None = None, k_bits: int = 4, v_bits: int = 4, v_group_size: int = 32, seed: int = 0):
        self.bits = int(k_bits)
        self.scale_dim = 1
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.v_bits = int(v_bits)
        self.v_group_size = int(v_group_size)
        self.seed = int(seed)
        self.k_algo: TurboQuantProd | None = None
        self.v_scale_dim = 2
        if self.bits != 4:
            raise ValueError("AsymTurboQuantKVQuantizer currently implements 4-bit K only")
        if self.v_bits != 4:
            raise ValueError("AsymTurboQuantKVQuantizer currently implements 4-bit grouped-linear V only")
        if self.v_group_size <= 0 or self.v_group_size % 2 != 0:
            raise ValueError(f"v_group_size must be positive and even, got {self.v_group_size}")
        if head_dim is not None:
            self._ensure_algo(int(head_dim), torch.device("cpu"))

    def _ensure_algo(self, head_dim: int, device: torch.device):
        if self.k_algo is None or self.k_algo.d != head_dim:
            self.k_algo = TurboQuantProd(head_dim, bits=self.bits, seed=self.seed)
        self.k_algo = self.k_algo.to(device)
        self.head_dim = int(head_dim)

    def allocate_cache(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device):
        raise RuntimeError("use allocate_cache_split for AsymTurboQuantKVQuantizer")

    def allocate_cache_split(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device):
        self._ensure_algo(int(head_dim), torch.device(device))
        k_cache_dim = (int(head_dim) + 1) // 2 if self.bits == 4 and _TRITON_AVAILABLE else int(head_dim)
        v_cache_dim = (int(head_dim) + 1) // 2
        group_count = math.ceil(int(head_dim) / self.v_group_size)

        k_cache = torch.empty(int(num_layers), int(num_blocks), int(block_size), int(num_kv_heads), k_cache_dim, dtype=torch.int8, device=device)
        k_scales = torch.empty(int(num_layers), int(num_blocks), int(block_size), int(num_kv_heads), 2, dtype=dtype, device=device)
        v_cache = torch.empty(int(num_layers), int(num_blocks), int(block_size), int(num_kv_heads), v_cache_dim, dtype=torch.int8, device=device)
        v_scales = torch.empty(int(num_layers), int(num_blocks), int(block_size), int(num_kv_heads), group_count, dtype=dtype, device=device)
        v_zeros = torch.empty_like(v_scales)
        return k_cache, k_scales, v_cache, v_scales, v_zeros

    def quantize_k(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_algo(tensor.shape[-1], tensor.device)
        assert self.k_algo is not None

        if _triton_prod4_enabled(self.bits, tensor.device):
            packed, gamma, norm = _fused_turboquant_and_cache_kernel(
                tensor,
                self.k_algo.mse.pi,
                self.k_algo.mse.codebook,
                self.k_algo.s,
            )
            scales = torch.cat([gamma, norm], dim=-1).to(tensor.dtype)
            return packed, scales

        idx, qjl, gamma, norm = self.k_algo.quantize(tensor)
        qjl_bit = (qjl > 0).to(torch.uint8)
        packed = ((idx.to(torch.uint8) << 1) | qjl_bit).view(torch.int8)
        scales = torch.cat([gamma, norm], dim=-1).to(tensor.dtype)
        return packed, scales

    def quantize_v(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.head_dim is None:
            self._ensure_algo(tensor.shape[-1], tensor.device)
        packed, scales, zeros = _quantize_grouped_linear_4bit(tensor, self.v_group_size)
        return packed, scales, zeros

    def quantize_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.head_dim is None:
            self._ensure_algo(k.shape[-1], k.device)
        assert self.k_algo is not None

        if _triton_prod4_enabled(self.bits, k.device):
            q_k, gamma, norm = _fused_turboquant_and_cache_kernel(
                k,
                self.k_algo.mse.pi,
                self.k_algo.mse.codebook,
                self.k_algo.s,
            )
            s_k = torch.cat([gamma, norm], dim=-1).to(k.dtype)
        else:
            q_k, s_k = self.quantize_k(k)

        q_v, s_v, z_v = self.quantize_v(v)
        return q_k, s_k, q_v, s_v, z_v

    def dequantize_k(self, q_tensor: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype, out: torch.Tensor | None = None) -> torch.Tensor:
        if self.k_algo is None:
            self._ensure_algo(self.head_dim or q_tensor.shape[-1], q_tensor.device)
        assert self.k_algo is not None

        if _triton_prod4_enabled(self.bits, q_tensor.device):
            return _triton_dequantize_prod4(
                q_tensor,
                scales,
                self.k_algo.mse.pi,
                self.k_algo.mse.codebook,
                self.k_algo.s,
                self.k_algo.d,
                dtype,
                out=out,
            )

        packed_u8 = q_tensor.view(torch.uint8).to(torch.int64)
        idx = packed_u8 >> 1
        qjl_bit = packed_u8 & 1
        qjl = torch.where(qjl_bit > 0, torch.ones_like(q_tensor, dtype=dtype), -torch.ones_like(q_tensor, dtype=dtype))
        gamma = scales[..., 0:1]
        norm = scales[..., 1:2]
        out_tensor = self.k_algo.dequantize(idx, qjl, gamma, norm).to(dtype)
        if out is not None:
            if tuple(out.shape) != tuple(out_tensor.shape):
                raise ValueError(f"dequant out shape mismatch: expected {tuple(out_tensor.shape)}, got {tuple(out.shape)}")
            out.copy_(out_tensor.to(out.dtype))
            return out
        return out_tensor

    def dequantize_v(
        self,
        q_tensor: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if zeros is None:
            raise ValueError("zeros tensor is required for asym grouped-linear V dequantization")
        head_dim = self.head_dim or int(scales.shape[-1]) * self.v_group_size
        return _dequantize_grouped_linear_4bit(q_tensor, scales, zeros, self.v_group_size, head_dim, dtype, out=out)

    def rotate_query(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k_algo is None:
            self._ensure_algo(q.shape[-1], q.device)
        assert self.k_algo is not None and self.k_algo.mse is not None
        qf = q.to(torch.float32).contiguous()
        pi = self.k_algo.mse.pi.to(q.device, dtype=qf.dtype)
        s = self.k_algo.s.to(q.device, dtype=qf.dtype)
        return (qf @ pi.t()).contiguous(), (qf @ s.t()).contiguous()

    def get_codebook(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self.k_algo is None:
            if self.head_dim is None:
                raise ValueError("head_dim is not initialized for asym quantizer")
            self._ensure_algo(self.head_dim, device)
        assert self.k_algo is not None
        return self.k_algo.mse.codebook.to(device=device, dtype=dtype)

    def supports_fused_decode(self, device: torch.device) -> bool:
        return device.type == "cuda" and _TRITON_AVAILABLE

    def dequantize(self, q_tensor: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.dequantize_k(q_tensor, scales, dtype, out=out)

    def bytes_per_token_head(self, head_dim: int, dtype_size: int) -> tuple[int, int]:
        k_persistent = (int(head_dim) + 1) // 2 + dtype_size * 2 if self.bits == 4 and _TRITON_AVAILABLE else int(head_dim) + dtype_size * 2
        group_count = math.ceil(int(head_dim) / self.v_group_size)
        v_persistent = (int(head_dim) + 1) // 2 + dtype_size * group_count * 2
        transient = 0
        return k_persistent + v_persistent, transient


class BaseKVQuantizer(ABC):
    def __init__(self, bits: int):
        self.bits = int(bits)
        self.scale_dim = 1

    @abstractmethod
    def allocate_cache(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device):
        pass

    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def dequantize(
        self,
        q_tensor: torch.Tensor,
        scales: torch.Tensor,
        dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass

    def bytes_per_token_head(self, head_dim: int, dtype_size: int) -> tuple[int, int]:
        """
        Returns (persistent, transient) bytes per token per KV head.
        - persistent: bytes stored in kv_cache + kv_scales
        - transient: temporary bytes needed during runtime dequantization
        """
        raise NotImplementedError


class TurboQuantMSEKVQuantizer(BaseKVQuantizer):
    def __init__(self, bits: int = 3):
        super().__init__(bits)
        self.scale_dim = 1
        self.algo = None

    def _ensure_algo(self, head_dim: int, device: torch.device):
        if self.algo is None or self.algo.d != head_dim:
            self.algo = TurboQuantMSE(head_dim, bits=self.bits)
        self.algo = self.algo.to(device)

    def allocate_cache(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device):
        self._ensure_algo(int(head_dim), torch.device(device))
        kv_cache = torch.empty(
            2,
            int(num_layers),
            int(num_blocks),
            int(block_size),
            int(num_kv_heads),
            int(head_dim),
            dtype=torch.int8,
            device=device,
        )
        kv_scales = torch.empty(
            2,
            int(num_layers),
            int(num_blocks),
            int(block_size),
            int(num_kv_heads),
            self.scale_dim,
            dtype=dtype,
            device=device,
        )
        return kv_cache, kv_scales

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_algo(tensor.shape[-1], tensor.device)
        idx, norm = self.algo.quantize(tensor)
        q = idx.to(torch.uint8).view(torch.int8)
        return q, norm

    def dequantize(
        self,
        q_tensor: torch.Tensor,
        scales: torch.Tensor,
        dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._ensure_algo(q_tensor.shape[-1], q_tensor.device)
        idx = q_tensor.view(torch.uint8).to(torch.int64)
        norm = scales[..., :1]
        out_tensor = self.algo.dequantize(idx, norm).to(dtype)
        if out is not None:
            if tuple(out.shape) != tuple(out_tensor.shape):
                raise ValueError(f"dequant out shape mismatch: expected {tuple(out_tensor.shape)}, got {tuple(out.shape)}")
            out.copy_(out_tensor.to(out.dtype))
            return out
        return out_tensor

    def bytes_per_token_head(self, head_dim: int, dtype_size: int) -> tuple[int, int]:
        # Persistent: int8 code per dim + norm scale
        persistent = int(head_dim) + dtype_size * self.scale_dim
        # Runtime: full fp reconstruction per dim during dequantize
        transient = int(head_dim) * dtype_size
        return persistent, transient


class TurboQuantProdKVQuantizer(BaseKVQuantizer):
    def __init__(self, bits: int = 3):
        super().__init__(bits)
        assert self.bits >= 2
        self.scale_dim = 2  # [gamma, norm]
        self.algo = None
        self.head_dim = None

    def _ensure_algo(self, head_dim: int, device: torch.device):
        if self.algo is None or self.algo.d != head_dim:
            self.algo = TurboQuantProd(head_dim, bits=self.bits)
        self.algo = self.algo.to(device)
        self.head_dim = int(head_dim)

    def _use_triton_prod4(self, device: torch.device) -> bool:
        return _triton_prod4_enabled(self.bits, device)

    def supports_fused_decode(self, device: torch.device) -> bool:
        return self._use_triton_prod4(device)

    def allocate_cache(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device):
        dev = torch.device(device)
        self._ensure_algo(int(head_dim), dev)
        cache_dim = (int(head_dim) + 1) // 2 if self._use_triton_prod4(dev) else int(head_dim)
        kv_cache = torch.empty(
            2,
            int(num_layers),
            int(num_blocks),
            int(block_size),
            int(num_kv_heads),
            cache_dim,
            dtype=torch.int8,
            device=device,
        )
        kv_scales = torch.empty(
            2,
            int(num_layers),
            int(num_blocks),
            int(block_size),
            int(num_kv_heads),
            self.scale_dim,
            dtype=dtype,
            device=device,
        )
        return kv_cache, kv_scales

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_algo(tensor.shape[-1], tensor.device)

        if self._use_triton_prod4(tensor.device):
            packed, gamma, norm = _fused_turboquant_and_cache_kernel(
                tensor,
                self.algo.mse.pi,
                self.algo.mse.codebook,
                self.algo.s,
            )
            scales = torch.cat([gamma, norm], dim=-1).to(tensor.dtype)
            return packed, scales

        idx, qjl, gamma, norm = self.algo.quantize(tensor)

        qjl_bit = (qjl > 0).to(torch.uint8)
        packed = ((idx.to(torch.uint8) << 1) | qjl_bit).view(torch.int8)
        scales = torch.cat([gamma, norm], dim=-1)
        return packed, scales

    def dequantize(
        self,
        q_tensor: torch.Tensor,
        scales: torch.Tensor,
        dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.algo is None:
            self._ensure_algo(self.head_dim or q_tensor.shape[-1], q_tensor.device)

        if self._use_triton_prod4(q_tensor.device):
            return _triton_dequantize_prod4(
                q_tensor,
                scales,
                self.algo.mse.pi,
                self.algo.mse.codebook,
                self.algo.s,
                self.algo.d,
                dtype,
                out=out,
            )

        packed_u8 = q_tensor.view(torch.uint8).to(torch.int64)
        idx = packed_u8 >> 1
        qjl_bit = packed_u8 & 1
        qjl = torch.where(qjl_bit > 0, torch.ones_like(q_tensor, dtype=dtype), -torch.ones_like(q_tensor, dtype=dtype))

        gamma = scales[..., 0:1]
        norm = scales[..., 1:2]
        out_tensor = self.algo.dequantize(idx, qjl, gamma, norm).to(dtype)
        if out is not None:
            if tuple(out.shape) != tuple(out_tensor.shape):
                raise ValueError(f"dequant out shape mismatch: expected {tuple(out_tensor.shape)}, got {tuple(out.shape)}")
            out.copy_(out_tensor.to(out.dtype))
            return out
        return out_tensor

    def bytes_per_token_head(self, head_dim: int, dtype_size: int) -> tuple[int, int]:
        # For Triton 4-bit prod path we pack two dims per byte.
        if self.bits == 4 and _TRITON_AVAILABLE:
            persistent = (int(head_dim) + 1) // 2 + dtype_size * self.scale_dim
        else:
            persistent = int(head_dim) + dtype_size * self.scale_dim
        # Decode now uses the fused score path, so no persistent fp dequant workspace is reserved.
        transient = 0
        return persistent, transient


def get_kv_quantizer(config) -> BaseKVQuantizer | None:
    algo = (config.kv_quant_algo or "").lower()
    bits = config.kv_quant_bits if config.kv_quant_bits is not None else 3
    if algo in {"turboquant", "turboquant_prod", "turboquant-prod"}:
        return TurboQuantProdKVQuantizer(bits)
    if algo in {"asym_turboquant", "asym-turboquant", "asym"}:
        return AsymTurboQuantKVQuantizer(k_bits=bits, v_bits=config.kv_v_bits, v_group_size=config.kv_v_group_size)
    if algo in {"turboquant_mse", "turboquant-mse"}:
        return TurboQuantMSEKVQuantizer(bits)
    return None
