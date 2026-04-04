import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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
    def dequantize(self, q_tensor: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
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

    def dequantize(self, q_tensor: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        self._ensure_algo(q_tensor.shape[-1], q_tensor.device)
        idx = q_tensor.view(torch.uint8).to(torch.int64)
        norm = scales[..., :1]
        return self.algo.dequantize(idx, norm).to(dtype)

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

    def _ensure_algo(self, head_dim: int, device: torch.device):
        if self.algo is None or self.algo.d != head_dim:
            self.algo = TurboQuantProd(head_dim, bits=self.bits)
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
        idx, qjl, gamma, norm = self.algo.quantize(tensor)

        qjl_bit = (qjl > 0).to(torch.uint8)
        packed = ((idx.to(torch.uint8) << 1) | qjl_bit).view(torch.int8)
        scales = torch.cat([gamma, norm], dim=-1)
        return packed, scales

    def dequantize(self, q_tensor: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        self._ensure_algo(q_tensor.shape[-1], q_tensor.device)

        packed_u8 = q_tensor.view(torch.uint8).to(torch.int64)
        idx = packed_u8 >> 1
        qjl_bit = packed_u8 & 1
        qjl = torch.where(qjl_bit > 0, torch.ones_like(q_tensor, dtype=dtype), -torch.ones_like(q_tensor, dtype=dtype))

        gamma = scales[..., 0:1]
        norm = scales[..., 1:2]
        return self.algo.dequantize(idx, qjl, gamma, norm).to(dtype)

    def bytes_per_token_head(self, head_dim: int, dtype_size: int) -> tuple[int, int]:
        # Persistent: packed (idx + qjl bit) as uint8/int8 per dim + [gamma, norm]
        persistent = int(head_dim) + dtype_size * self.scale_dim
        # Runtime: full fp reconstruction per dim during dequantize
        transient = int(head_dim) * dtype_size
        return persistent, transient


def get_kv_quantizer(config) -> BaseKVQuantizer | None:
    algo = (config.kv_quant_algo or "").lower()
    bits = config.kv_quant_bits if config.kv_quant_bits is not None else 3
    if algo in {"turboquant", "turboquant_prod", "turboquant-prod"}:
        return TurboQuantProdKVQuantizer(bits)
    if algo in {"turboquant_mse", "turboquant-mse"}:
        return TurboQuantMSEKVQuantizer(bits)
    return None
