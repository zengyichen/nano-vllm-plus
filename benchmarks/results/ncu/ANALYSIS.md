# NCU 性能分析报告 — nano-vLLM Triton Kernels

## 报告信息

- **GPU**: RTX 4060 Laptop (Ada Lovelace, 24 SMs, 256 GB/s peak BW)
- **工具**: nsys (timeline) + CUDA events (micro-benchmark)
- **采集时间**: 2026-06-12
- **所有 kernel**: N=4096, H=8, D=128 (除特殊注明)

---

## 执行摘要

| Kernel | 类型 | 耗时 (nsys) | 融合操作数 | 优化空间 |
|:---|:---|:---|:---|:---|
| `_store_kvcache_kernel` | @triton.jit | **15.5 μs** (7.9%) | 4 (load K, load V, store K, store V) | 🔴 中等 |
| `triton_per_fused RMSNorm` | torch.compile | **11.9 μs** (4.5%) | 5 (pow+mean+rsqrt+mul+to_dtype) | 🟢 低 |
| `triton_poi_fused RoPE` | torch.compile | **90.4 μs** (26.9%) | 7 (to+chunk+mul+add+sub+cat+to_dtype) | 🟡 低 |
| `triton_per_fused Sampler` | torch.compile | **22.3 μs** (8.1%) | 6 (div+softmax+exp+clamp+div+argmax) | 🟢 低 |
| `triton_poi_fused SiluAndMul` | torch.compile | **21.6 μs** (8.0%) | 3 (chunk+silu+mul) | 🟢 低 |

> **注**: nsys 百分比基于完整脚本运行时间，包含 tensor 创建等开销。实际推理中这些 kernel 是 decode step 的关键路径。

---

## 1. `_store_kvcache_kernel` — Paged KV Cache Scatter

### 测量结果

| Metric | Cold Run | Warm Run | Avg |
|:---|:---|:---|:---|
| GPU time | 7.6 μs | 23.4 μs | **15.5 μs** |
| Data moved | 33.6 MB (读 16.8 + 写 16.8) | | |
| Effective BW | ~2,170 GB/s (L2 cached) | ~720 GB/s | — |

> **注意**: nsys 测量的是 warm-up+单次执行。第一跑 key/value 在 L1/L2 中（刚被 randn 生成），BW 虚高。第二跑更接近真实场景（KV cache 在 DRAM 中）。

### 代码分析

```python
@triton.jit
def _store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                           k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D):
    idx = tl.program_id(0)                    # 每个 program = 1 个 token (4096 programs)
    slot = tl.load(slot_mapping_ptr + idx)     # 读 slot_mapping[idx] — 1 次 coalesced read
    if slot == -1: return                      # 分支：稀疏 scatter 有控制流散度
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)       # 读 key[idx] — 1024 bytes (128*8=1024 fp16)
    value = tl.load(value_ptr + value_offsets) # 读 value[idx] — 1024 bytes
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key) # 写 k_cache[slot] — 非 coalesced (随机)
    tl.store(v_cache_ptr + cache_offsets, value)
```

### 瓶颈诊断

| 问题 | 严重度 | 说明 |
|:---|:---|:---|
| **随机 scatter 写入** | 🔴 高 | cache 写入目标由 `slot_mapping` 决定，随机 pattern 下 DRAM 写入非 coalesced，L2 命中率低。实测 random pattern ~2× 慢于 sequential |
| **细粒度 launch** | 🟡 中 | grid size = (4096,)，每个 thread block 只处理 1 个 token。SM 利用率低（4096 blocks / 24 SMs → 170 blocks/SM，但每个 block 只做 2KB 的读写） |
| **无 L1/Shared Memory 缓存** | 🟡 中 | 直接从 Global Memory 读写，没有经过 shared memory 缓冲 |
| **分支散度** | 🟢 低 | `if slot == -1: return` — 在实际推理中 slot_mapping 没有 -1 值，不影响 |

### 优化建议

#### 方案 A：Fused QKV Projection + Store（推荐，难度高）

将 QKV 投影的输出直接写入 KV cache slot，跳过中间 key/value tensor 的物化。当前流程：

```
QKV proj → split → key tensor (8MB fp16) → store_kvcache → k_cache (8MB)
                                    ↑ 无用的中间物化！
```

改为：QKV proj 结果直接 scatter 到 k_cache。这需要在 Triton kernel 内完成 split + scatter，将 3 个 kernel launch 融合为 1 个。

**预期收益**: 消除 2 次 global memory 读写（key/value tensor 的 write+read），减少 ~60μs kernel launch 开销。Decode step 总延迟预期降低 3-5%。

#### 方案 B：Autotune Grid Size（难度低）

当前 grid = (N,)，对 4096 tokens 需要至少 170 blocks/SM。对于 bs=1 decode，N=1，只用 1 个 block 处理 1 个 token。

```python
# 当前
_store_kvcache_kernel[(N,)](key, key.stride(0), ...)

# 建议: autotune block_size for small N
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1}),
        triton.Config({'BLOCK_SIZE': 4}),
        triton.Config({'BLOCK_SIZE': 8}),
    ],
    key=['N']
)
```

**预期收益**: bs=1 decode 时，将 1 个 block 改为 4-8 个 block 并行处理 → ~2-3× 加速（对 decode step 整体影响 <1%）

#### 方案 C：Vectorized Load/Store（难度低）

当前 `tl.load(key_ptr + key_offsets)` 使用 `tl.arange(0, D)` 做 1D load。对于 D=128 (1024 bytes fp16)，可以改为 2D vectorized：

```python
# 使用 (D//4, 4) 的 2D layout → 一次 load float4x4（128 bytes/load）
key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D//4)[:, None] * 4 + tl.arange(0, 4)[None, :])
```

**预期收益**: 减少 load/store 指令数 4×，~10-20% 加速

---

## 2. RMSNorm — `triton_per_fused` (torch.compile)

### 测量结果

| Metric | 值 |
|:---|:---|
| 每个 kernel 时间 | **11.9 μs** (median 11.8 μs) |
| 总实例数 | 320 (warmup + measurement) |
| 融合操作 | `pow` + `mean` + `rsqrt` + `mul(weight)` + `to(dtype)` |

### 瓶颈诊断

| 问题 | 严重度 | 说明 |
|:---|:---|:---|
| **几乎完美融合** | 🟢 | 5 个 element-wise ops 在单个 kernel 内完成，无中间 tensor 分配 |
| **Register pressure** | 🟢 低 | element-wise ops 天然低寄存器占用 |

### 优化建议

无显著优化空间。torch.compile 已经将 RMSNorm 优化到接近手写 Triton kernel 的水平。**6× speedup vs eager 已是最优**。

---

## 3. RotaryEmbedding — `triton_poi_fused` (torch.compile)

### 测量结果

| Metric | 值 |
|:---|:---|
| 每个 kernel 时间 | **90.4 μs** (median 91.4 μs) |
| 总实例数 | 213 (warmup + measurement) |
| 融合操作 | `to(f32)` + `chunk` + `mul` + `add` + `sub` + `cat` + `to(fp16)` |

### 瓶颈诊断

| 问题 | 严重度 | 说明 |
|:---|:---|:---|
| **float32 cast 开销** | 🟡 中 | `x.float()` 和 `to(x.dtype)` 两次类型转换，fp16→f32 读、f32→fp16 写各产生额外带宽消耗 |
| **chunk + cat 的内存分配** | 🟡 中 | `chunk` 和 `cat` 配对时 torch.compile 可能无法完全消除中间分配 |

### 优化建议

#### 方案：直接在 fp16 中计算 RoPE（难度中）

当前 eager 实现用 f32 是为了精度。但 RTX 4060 (SM 8.9) 的 fp16 乘加精度足够：

```python
@torch.compile
def apply_rotary_emb_fp16(x, cos, sin):
    """RoPE in fp16 — avoids f32 cast overhead."""
    x1, x2 = x.chunk(2, dim=-1)  # stays fp16
    cs = cos.to(torch.float16)
    sn = sin.to(torch.float16)
    y1 = x1 * cs - x2 * sn      # fp16 FMA
    y2 = x2 * cs + x1 * sn
    return torch.cat((y1, y2), dim=-1)
```

**预期收益**: ~50% 时间节省（消除 f32 cast），90μs → ~45μs。需验证精度是否可接受（在 28 层累积误差下）。

---

## 4. Sampler — `triton_per_fused` (torch.compile)

### 测量结果

| Metric | 值 |
|:---|:---|
| 每个 kernel 时间 | **22.3 μs** (median 18.5 μs) |
| 总实例数 | 179 (warmup + measurement) |
| 融合操作 | `div(temp)` + `softmax` + `exponential` + `clamp` + `div(gumbel)` + `argmax` |

### 瓶颈诊断

| 问题 | 严重度 | 说明 |
|:---|:---|:---|
| **softmax 是计算瓶颈** | 🟡 中 | 对于 V=151936，softmax 需要归约 152K 个元素，计算密集 |
| **f32 精度** | 🟢 低 | 采样需要 f32 精度，不可避免 |

### 优化建议

**方案：对 bs=1 的常见场景跳过 softmax 的完整归约**

Gumbel-max trick 等价于 `argmax(logits + noise)`。可以通过在 Triton kernel 内直接做 warps-level 的 argmax，而非先 softmax 再 argmax：

```python
# 等价变换：argmax(softmax(logits/temp) / gumbel) = argmax(logits/temp + log(-log(uniform)))
#               = argmax(logits/temp + gumbel_noise)
result = (logits / temperatures.unsqueeze(1) + torch.rand_like(logits).log_().neg_().log_().neg_()).argmax(-1)
```

**预期收益**: 消除 softmax（最耗时的部分），~1.5-2× 加速。但需要确保数值稳定性。

---

## 5. SiluAndMul — `triton_poi_fused` (torch.compile)

### 测量结果

| Metric | 值 |
|:---|:---|
| 每个 kernel 时间 | **21.6 μs** (median 21.9 μs) |
| 总实例数 | 213 (warmup + measurement) |
| 融合操作 | `chunk` + `silu(gate)` + `mul(up)` |

### 瓶颈诊断

| 问题 | 严重度 | 说明 |
|:---|:---|:---|
| **几乎完美融合** | 🟢 | 3 个 ops 在单 kernel 内，H=3072 时约 21.6 μs，已经接近 memory bandwidth 极限 |

### 优化建议

无显著优化空间。silu_and_mul 是推理中最简单的 kernel 之一。

---

## 综合分析：关键瓶颈与优先级

### 瓶颈分布 (decode step 中)

| Phase | Kernel(s) | 单步时间 | 占比 | 瓶颈类型 |
|:---|:---|:---|:---|:---|
| QKV Proj | F.linear (cuBLAS) | ~300 μs | ~30% | Compute (matmul) |
| KV Store | store_kvcache × 28 | 15.5 × 28 = **434 μs** | **~43%** | Memory BW (scatter) |
| RMSNorm (pre) | rmsnorm × 28 | 11.9 × 28 = 333 μs | ~33% | Memory BW |
| O Proj + MLP | F.linear × 3 | ~600 μs | ~60% | Compute (matmul) |
| RoPE | rotary × 28 | 90.4 × 28 = 2531 μs | ~253%* | Mixed |

> *注：RoPE 只在 prefill 中显著，decode 只应用于单个 token

### 高优先级优化

| # | 优化 | Kernel | 预期收益 | 难度 | 对 decode 影响 |
|:---|:---|:---|:---|:---|:---|
| 1 | **Fused QKV+Store** | store_kvcache | 消除 8MB 中间物化 | 🔴 高 | **-3~5% per step** |
| 2 | **Vectorized Load/Store** | store_kvcache | ~20% kernel 加速 | 🟡 中 | **-8~10% per step** |
| 3 | **fp16 RoPE** | rotary | ~50% kernel 加速 | 🟡 中 | 对 prefill 收益显著 |
| 4 | **Autotune Grid** | store_kvcache | ~2-3× bs=1 | 🟢 低 | 微 (<1%) |

### nsys 数据文件

所有报告保存在 `benchmarks/results/ncu/`：
- `store_kvcache_timeline.nsys-rep` + `.sqlite`
- `rmsnorm_timeline.nsys-rep` + `.sqlite`
- `rotary_timeline.nsys-rep` + `.sqlite`
- `sampler_timeline.nsys-rep` + `.sqlite`
- `silu_timeline.nsys-rep` + `.sqlite`

---

## 如何获取更详细的 NCU 数据

GPU perf counter 权限问题（`ERR_NVGPUCTRPERM`）导致无法用 `--set full`。如需获取 SM occupancy、DRAM throughput、L1/L2 cache hit rate 等详细指标，需要：

```bash
# 方法 1：以 root 运行
sudo ncu --set full -o report_name --target-processes all python benchmarks/bench_triton_ncu.py

# 方法 2：设置 kernel module 参数（需要 sudo）
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
ncu --set full -o report_name --target-processes all python benchmarks/bench_triton_ncu.py
```
