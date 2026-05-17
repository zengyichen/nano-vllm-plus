import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.fused_quant_attn import (
    fused_asym_quantized_decode_attention,
    fused_quantized_decode_attention,
)
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
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


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_scales(k_scale: torch.Tensor, v_scale: torch.Tensor, k_cache_scale: torch.Tensor, v_cache_scale: torch.Tensor, slot_mapping: torch.Tensor):
    # k_scale: [N, num_heads, scale_dim], k_cache_scale: [num_blocks, block_size, num_heads, scale_dim]
    N, num_heads, scale_dim = k_scale.shape
    D = num_heads * scale_dim
    assert k_scale.stride(-1) == 1 and v_scale.stride(-1) == 1
    assert k_scale.stride(1) == scale_dim and v_scale.stride(1) == scale_dim
    assert k_cache_scale.stride(1) == D and v_cache_scale.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](k_scale, k_scale.stride(0), v_scale, v_scale.stride(0), k_cache_scale, v_cache_scale, slot_mapping, D)


@triton.jit
def store_tensor_kernel(
    src_ptr,
    src_stride,
    dst_ptr,
    dst_stride,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    src_offsets = idx * src_stride + tl.arange(0, D)
    dst_offsets = slot * D + tl.arange(0, D)
    value = tl.load(src_ptr + src_offsets)
    tl.store(dst_ptr + dst_offsets, value)


def store_tensor(src: torch.Tensor, dst: torch.Tensor, slot_mapping: torch.Tensor):
    N = src.shape[0]
    D = src.numel() // N
    assert src.stride(-1) == 1 and dst.stride(-1) == 1
    assert slot_mapping.numel() == N
    store_tensor_kernel[(N,)](src, src.stride(0), dst, dst.stride(0), slot_mapping, D)


def _build_slot_mapping_from_block_tables(
    block_tables: torch.Tensor,
    seqlens: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = block_tables.device

    seqlens_i32 = seqlens.to(dtype=torch.int32)
    cu_seqlens_k = torch.empty(seqlens_i32.numel() + 1, device=device, dtype=torch.int32)
    cu_seqlens_k[0] = 0
    cu_seqlens_k[1:] = torch.cumsum(seqlens_i32, dim=0)

    max_blocks = int(block_tables.shape[1])
    if max_blocks == 0:
        return torch.empty(0, device=device, dtype=torch.int64), cu_seqlens_k

    max_seqlen_upper = max_blocks * int(block_size)
    pos = torch.arange(max_seqlen_upper, device=device, dtype=torch.int64)
    block_idx = torch.div(pos, block_size, rounding_mode="floor")
    block_off = pos - block_idx * block_size

    # Expand each sequence's block table to token slots, then mask out padding.
    block_ids = block_tables.to(torch.int64).index_select(1, block_idx)
    slots = block_ids * block_size + block_off
    valid = pos.unsqueeze(0) < seqlens_i32.to(torch.int64).unsqueeze(1)
    slot_mapping = slots[valid]
    return slot_mapping, cu_seqlens_k


def _gather_quantized_cache(
    cache: torch.Tensor,
    scales: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cache: [num_blocks, block_size, num_kv_heads, dim_or_packed]
    # scales: [num_blocks, block_size, num_kv_heads, scale_dim]
    cache_flat = cache.reshape(-1, cache.shape[-2], cache.shape[-1])
    scales_flat = scales.reshape(-1, scales.shape[-2], scales.shape[-1])
    gathered_cache = cache_flat.index_select(0, slot_mapping)
    gathered_scales = scales_flat.index_select(0, slot_mapping)
    return gathered_cache, gathered_scales


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.quantizer = None
        self.k_scales = self.v_scales = None
        self.v_zeros = None
        self.quant_decode_backend = "auto"
        self.decode_k_workspace = None
        self.decode_v_workspace = None
        self.decode_workspace_tokens = 0

    def _rotate_query(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.quantizer, "rotate_query"):
            return self.quantizer.rotate_query(q)
        algo = getattr(self.quantizer, "algo", None)
        if algo is None:
            return q.float().contiguous(), q.float().contiguous()
        qf = q.to(torch.float32).contiguous()
        pi = algo.mse.pi.to(q.device, dtype=qf.dtype)
        s = algo.s.to(q.device, dtype=qf.dtype)
        return (qf @ pi.t()).contiguous(), (qf @ s.t()).contiguous()

    def _dequantize_k(self, q_tensor: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype, out: torch.Tensor | None = None) -> torch.Tensor:
        if hasattr(self.quantizer, "dequantize_k"):
            return self.quantizer.dequantize_k(q_tensor, scales, dtype, out=out)
        return self.quantizer.dequantize(q_tensor, scales, dtype, out=out)

    def _dequantize_v(
        self,
        q_tensor: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(self.quantizer, "dequantize_v"):
            return self.quantizer.dequantize_v(q_tensor, scales, zeros, dtype, out=out)
        return self.quantizer.dequantize(q_tensor, scales, dtype, out=out)

    def _run_quantized_varlen_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        block_tables: torch.Tensor,
        seqlens_k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dtype: torch.dtype,
        pre_slot_mapping: torch.Tensor | None = None,
        pre_cu_seqlens_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pre_slot_mapping is None or pre_cu_seqlens_k is None:
            block_size = int(k_cache.shape[1])
            slot_mapping, cu_seqlens_k = _build_slot_mapping_from_block_tables(
                block_tables,
                seqlens_k,
                block_size,
            )
        else:
            slot_mapping = pre_slot_mapping
            cu_seqlens_k = pre_cu_seqlens_k

        k_q, k_s = _gather_quantized_cache(k_cache, k_scales, slot_mapping)
        v_q, v_s = _gather_quantized_cache(v_cache, v_scales, slot_mapping)
        v_z = None
        if self.v_zeros is not None:
            v_z, _ = _gather_quantized_cache(self.v_zeros, self.v_zeros, slot_mapping)
        k = self._dequantize_k(k_q, k_s, dtype)
        v = self._dequantize_v(v_q, v_s, v_z, dtype)

        return flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
        )

    def _run_quantized_decode_dequant_flash_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        dtype: torch.dtype,
        pre_slot_mapping: torch.Tensor | None = None,
        pre_cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_k: int = 0,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.decode_k_workspace is None or self.decode_v_workspace is None:
            return None

        if pre_slot_mapping is None or pre_cu_seqlens_k is None:
            block_size = int(k_cache.shape[1])
            slot_mapping, cu_seqlens_k = _build_slot_mapping_from_block_tables(
                block_tables,
                context_lens,
                block_size,
            )
        else:
            slot_mapping = pre_slot_mapping
            cu_seqlens_k = pre_cu_seqlens_k

        total_k_tokens = int(slot_mapping.shape[0])
        if total_k_tokens <= 0 or total_k_tokens > int(self.decode_workspace_tokens):
            return None

        k_q, k_s = _gather_quantized_cache(k_cache, k_scales, slot_mapping)
        v_q, v_s = _gather_quantized_cache(v_cache, v_scales, slot_mapping)
        v_z = None
        if self.v_zeros is not None:
            v_z, _ = _gather_quantized_cache(self.v_zeros, self.v_zeros, slot_mapping)

        k_buf = self.decode_k_workspace[:total_k_tokens]
        v_buf = self.decode_v_workspace[:total_k_tokens]
        self._dequantize_k(k_q, k_s, dtype, out=k_buf)
        self._dequantize_v(v_q, v_s, v_z, dtype, out=v_buf)

        if cu_seqlens_q is None:
            cu_seqlens_q = torch.arange(0, q.shape[0] + 1, device=q.device, dtype=torch.int32)
        if max_seqlen_k <= 0:
            max_seqlen_k = int(block_tables.shape[1]) * int(k_cache.shape[1])

        return flash_attn_varlen_func(
            q,
            k_buf,
            v_buf,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
        )

    def _run_asym_turboquant_decode_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        v_zeros: torch.Tensor | None,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        dtype: torch.dtype,
        pre_slot_mapping: torch.Tensor | None = None,
        pre_cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_k: int = 0,
        cu_seqlens_q: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        q_rot, q_sketch = self._rotate_query(q)
        context = get_context()
        context.quant_q_rot = q_rot
        context.quant_q_sketch = q_sketch

        if v_zeros is None:
            return None

        if q.device.type == "cuda":
            codebook = None
            if hasattr(self.quantizer, "get_codebook"):
                codebook = self.quantizer.get_codebook(q.device, dtype=torch.float32)
            else:
                algo = getattr(self.quantizer, "algo", None)
                if algo is not None and hasattr(algo, "mse"):
                    codebook = algo.mse.codebook.to(q.device, dtype=torch.float32)

            if codebook is not None:
                return fused_asym_quantized_decode_attention(
                    q_rot,
                    q_sketch,
                    k_cache,
                    v_cache,
                    k_scales,
                    v_scales,
                    v_zeros,
                    block_tables,
                    context_lens,
                    self.scale,
                    self.num_kv_heads,
                    codebook,
                    int(getattr(self.quantizer, "v_group_size", 32)),
                    dtype,
                )

        if self.decode_k_workspace is None or self.decode_v_workspace is None:
            return None

        if pre_slot_mapping is None or pre_cu_seqlens_k is None:
            block_size = int(k_cache.shape[1])
            slot_mapping, cu_seqlens_k = _build_slot_mapping_from_block_tables(
                block_tables,
                context_lens,
                block_size,
            )
        else:
            slot_mapping = pre_slot_mapping
            cu_seqlens_k = pre_cu_seqlens_k

        total_k_tokens = int(slot_mapping.shape[0])
        if total_k_tokens <= 0 or total_k_tokens > int(self.decode_workspace_tokens):
            return None

        k_q, k_s = _gather_quantized_cache(k_cache, k_scales, slot_mapping)
        v_q, v_s = _gather_quantized_cache(v_cache, v_scales, slot_mapping)
        v_z = None if v_zeros is None else _gather_quantized_cache(v_zeros, v_zeros, slot_mapping)[0]

        k_buf = self.decode_k_workspace[:total_k_tokens]
        v_buf = self.decode_v_workspace[:total_k_tokens]
        self._dequantize_k(k_q, k_s, dtype, out=k_buf)
        self._dequantize_v(v_q, v_s, v_z, dtype, out=v_buf)

        if cu_seqlens_q is None:
            cu_seqlens_q = torch.arange(0, q.shape[0] + 1, device=q.device, dtype=torch.int32)
        if max_seqlen_k <= 0:
            max_seqlen_k = int(block_tables.shape[1]) * int(k_cache.shape[1])

        return flash_attn_varlen_func(
            q,
            k_buf,
            v_buf,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        k_scales, v_scales = self.k_scales, self.v_scales
        dtype = k.dtype

        if k_cache.numel() and v_cache.numel():
            if self.quantizer:
                if hasattr(self.quantizer, "quantize_k") and hasattr(self.quantizer, "quantize_v"):
                    if hasattr(self.quantizer, "quantize_kv"):
                        q_k, scale_k, q_v, scale_v, zero_v = self.quantizer.quantize_kv(k, v)
                    else:
                        q_k, scale_k = self.quantizer.quantize_k(k)
                        q_v, scale_v, zero_v = self.quantizer.quantize_v(v)
                    store_kvcache(q_k, q_v, k_cache, v_cache, context.slot_mapping)
                    store_tensor(scale_k, k_scales, context.slot_mapping)
                    if self.v_zeros is not None:
                        store_tensor(scale_v, v_scales, context.slot_mapping)
                        store_tensor(zero_v, self.v_zeros, context.slot_mapping)
                else:
                    q_k, scale_k = self.quantizer.quantize(k)
                    q_v, scale_v = self.quantizer.quantize(v)
                    store_kvcache(q_k, q_v, k_cache, v_cache, context.slot_mapping)
                    store_kvcache_scales(scale_k, scale_v, k_scales, v_scales, context.slot_mapping)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if self.quantizer and context.block_tables is not None:
                # Prefix-cache prefill: gather only active cached slots, then dequantize.
                seqlens_k = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
                o = self._run_quantized_varlen_attention(
                    q,
                    k_cache,
                    v_cache,
                    k_scales,
                    v_scales,
                    context.block_tables,
                    seqlens_k,
                    context.cu_seqlens_q,
                    context.max_seqlen_q,
                    context.max_seqlen_k,
                    dtype,
                    pre_slot_mapping=context.quant_slot_mapping,
                    pre_cu_seqlens_k=context.quant_cu_seqlens_k,
                )
            else:
                if context.block_tables is not None:    # prefix cache (non-quantized path)
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
        else:    # decode
            if self.quantizer and context.block_tables is not None and context.context_lens is not None:
                backend = getattr(self, "quant_decode_backend", "auto")

                if backend == "asym_turboquant":
                    o = self._run_asym_turboquant_decode_attention(
                        q,
                        k_cache,
                        v_cache,
                        k_scales,
                        v_scales,
                        self.v_zeros,
                        context.block_tables,
                        context.context_lens,
                        dtype,
                        pre_slot_mapping=context.quant_slot_mapping,
                        pre_cu_seqlens_k=context.quant_cu_seqlens_k,
                        max_seqlen_k=context.quant_max_seqlen_k,
                        cu_seqlens_q=context.cu_seqlens_q,
                    )
                    if o is not None:
                        return o

                if backend in {"dequant_flash", "auto"}:
                    o = self._run_quantized_decode_dequant_flash_attention(
                        q,
                        k_cache,
                        v_cache,
                        k_scales,
                        v_scales,
                        context.block_tables,
                        context.context_lens,
                        dtype,
                        pre_slot_mapping=context.quant_slot_mapping,
                        pre_cu_seqlens_k=context.quant_cu_seqlens_k,
                        max_seqlen_k=context.quant_max_seqlen_k,
                        cu_seqlens_q=context.cu_seqlens_q,
                    )
                    if o is not None:
                        return o

                if (
                    backend in {"fused", "auto"}
                    and hasattr(self.quantizer, "supports_fused_decode")
                    and self.quantizer.supports_fused_decode(q.device)
                ):
                    algo = self.quantizer.algo
                    return fused_quantized_decode_attention(
                        q,
                        k_cache,
                        v_cache,
                        k_scales,
                        v_scales,
                        context.block_tables,
                        context.context_lens,
                        self.scale,
                        self.num_kv_heads,
                        self.quantizer,
                        algo.mse.pi,
                        algo.mse.codebook,
                        algo.s,
                    )

                # Fallback path for non-fused quantization variants.
                cu_seqlens_q = context.cu_seqlens_q
                if cu_seqlens_q is None:
                    cu_seqlens_q = torch.arange(0, q.shape[0] + 1, device=q.device, dtype=torch.int32)
                max_seqlen_k = context.quant_max_seqlen_k
                if max_seqlen_k <= 0:
                    max_seqlen_k = int(context.block_tables.shape[1]) * int(k_cache.shape[1])
                o = self._run_quantized_varlen_attention(
                    q,
                    k_cache,
                    v_cache,
                    k_scales,
                    v_scales,
                    context.block_tables,
                    context.context_lens,
                    cu_seqlens_q,
                    1,
                    max_seqlen_k,
                    dtype,
                    pre_slot_mapping=context.quant_slot_mapping,
                    pre_cu_seqlens_k=context.quant_cu_seqlens_k,
                )
            else:
                if self.quantizer:
                    # Safety fallback when decode metadata is incomplete.
                    k_cache = self._dequantize_k(k_cache, k_scales, dtype)
                    v_cache = self._dequantize_v(v_cache, v_scales, self.v_zeros, dtype)
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
        return o
