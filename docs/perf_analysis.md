# nano-vLLM (main) 性能瓶颈分析与改进方案

## 硬件环境

| 参数 | 值 | 含义 |
|------|-----|------|
| GPU | RTX 4060 Laptop (Ada Lovelace) | SM 8.9 |
| VRAM | 8 GB | **最大约束** |
| 显存带宽 | 128-bit × 8001 MHz = ~256 GB/s | **decode 瓶颈** |
| SMs | 24 | 并行计算单元 |
| L2 Cache | 32 MB | 缓解带宽压力 |
| Max Shared Mem | 99 KB/block | Triton kernel 约束 |
| Max Threads/SM | 1536 | 48 warps/SM |
| CPU | i9-12900HX (16C/24T) | 调度/哈希计算 |
| RAM | 16 GB | Block 管理元数据 |

---

## 模块一：权重量化（致命瓶颈）

### 现状
main 分支完全没有权重量化支持。模型以 FP16 加载，Qwen3-8B 约 16 GB 权重，**在 8 GB GPU 上直接 OOM，无法运行**。

### 瓶颈原因
- `linear.py` 只实现了标准 FP16 `F.linear`，无任何量化路径
- `qwen3.py` 的 `packed_modules_mapping` 已定义，但 `loader.py` 和模型层没有对应的解包逻辑
- AWQ 格式的 safetensors 中 q/k/v/gate/up 权重被 pack 为 int32，需要 fused dequant + matmul

### 改进方案
**方案：实现 AWQ 线性层（已在 feature/quant 分支完成）**

核心思路：
1. 新增 `awq_linear.py`，实现 `torch.float32` 存储的 scale+zero，`int32` 存储的 packed weight
2. Triton kernel 实现融合反量化+矩阵乘：`dequantize_weight` (从 int32 解包 4-bit) → `tl.dot` → `scale * (accum - zero_sum * input_sum)`
3. 关键优化：Split-K 减少寄存器压力，autotune 选择最优 tile 配置，int32 dequant 减少指令数
4. 预期效果：模型权重 ~4 GB（压缩 4×），8 GB GPU 可运行 8B 模型

---

## 模块二：KV Cache 显存（关键瓶颈）

### 现状
KV Cache 以 FP16 存储，无量化。

### 瓶颈分析
以 Qwen3-8B（32 层，8 KV heads，128 head_dim）为例：

```
单 token KV: 2 × 32 × 8 × 128 × 2 bytes = 128 KB
512 token 上下文: 128 KB × 512 = 64 MB per sequence
batch=8, 512 token: 512 MB KV cache
```

对于 8 GB GPU，扣除模型权重 ~4 GB（AWQ 后），剩余 ~3.5 GB 用于 KV cache 和其他。在 512 token 上下文下约可支持 50+ concurrent sequences，但在长上下文（4096+）下 KV cache 迅速成为瓶颈：

```
4096 token: 128 KB × 4096 ≈ 512 MB per sequence
```

### 改进方案
**方案 A：FP8 KV Cache**
- K/V 存储为 FP8（`torch.float8_e4m3fn`），节省 50% 显存
- 改动小：只需在 `store_kvcache` 时 cast 为 FP8，attention 前 cast 回 FP16
- 风险：FP8 精度有限，长序列可能累积误差
- 预期：KV cache 从 128 KB/token → 64 KB/token

**方案 B：4-bit KV Cache 量化（已在 feature/quant 完成）**
- K 用 TurboQuantProd（旋转+码本+QJL），V 用 GroupedLinear（per-group affine）
- 压缩比 ~4×，KV cache 从 128 KB/token → ~36 KB/token（含 scale/zero 开销）
- 需要对应的 fused decode kernel 来保持速度

---

## 模块三：Attention Decode — 显存带宽瓶颈

### 现状
`attention.py` decode 路径调用 `flash_attn_with_kvcache`。每次 decode step：
1. 从 KV cache 加载当前 token 之前的所有 K/V（FP16）
2. 计算 Q·K^T → softmax → weighted sum V
3. 写入当前 token 的 K/V 到 cache

### 瓶颈分析
Decode 是 **memory-bound** 的。RTX 4060 带宽 ~256 GB/s。

对于 512 token 上下文、单层 attention：

```
加载 KV: 512 × 8 heads × 128 dim × 2 bytes × 2 (K+V) = 2 MB
计算量: Q(1×8×128) · K^T(512×8×128) ≈ 1M FLOPs
算术强度: 1M FLOPs / 2 MB = 0.5 FLOP/byte
```

0.5 FLOP/byte 远低于 GPU 的计算/带宽比（~50 FLOP/byte for FP16 tensor cores），说明 **100% 带宽瓶颈**。

理论最大 decode throughput：

```
每 token 每层需加载: context_len × num_kv_heads × head_dim × 2(K+V) × 2 bytes
512 token: 512 × 8 × 128 × 4 = 2 MB/layer
32 层: 64 MB/token
理论 throughput: 256 GB/s / 64 MB/token ≈ 4000 token/s (理论极限)
```

实际远低于此，因为 FlashAttention 的 tile 加载存在开销，且 Python 调度、小 batch 无法充分利用带宽。

### 改进方案
**方案 A：增大 decode batch size**
- 当前 `max_num_seqs` 限制了并发 decode 数
- decode 阶段 batch 越大，带宽利用率越高（多个 query 共享同一份 KV cache 加载）
- 预期：batch=8 → 16 可将 decode throughput 提升 30-50%

**方案 B：Quantized KV Decode with Fused Kernel**
- 量化 KV cache 减少加载量（4-bit vs 16-bit = 4×）
- 融合反量化+attention 计算避免物化 FP16 tensor
- 已在 feature/quant 分支的 `fused_quant_attn.py` 中实现
- 预期：decode 延迟降低 30-50%，同时节省 KV cache 显存

**方案 C：FlashInfer / FlashAttention-3 升级**
- 当前使用 FlashAttention-2 (`flash_attn_with_kvcache`)，未针对 Ada Lovelace 优化
- FlashAttention-3 利用 Hopper+ 的 FP8 tensor core 和 TMA 异步拷贝
- Ada Lovelace (SM 8.9) 支持 FP8，但缺少 TMA
- 预期收益有限：~10-15%，但代码改动较大

---

## 模块四：连续批处理 (Continuous Batching)

### 现状
`Scheduler.schedule()` 采用先 prefill 后 decode 的严格分离策略：**同一 step 中不能同时处理 prefill 和 decode**。当前无 prefill 请求时才会调度 decode。

### 瓶颈分析
这导致两个问题：
1. **Prefill 阻塞 decode**：当一个长 prompt 进入 prefill 时，所有 running 的 decode 序列必须等待
2. **吞吐抖动**：prefill step 的计算量远大于 decode step（500 token prefill vs 1 token decode），导致 GPU 利用率在时间轴上不均匀

### 改进方案
**方案：实现 Chunked Prefill + Mixed Batch**

核心思路：
1. 将长 prefill 拆成多个 chunk（如每 chunk 256 token），与 decode step 交替调度
2. 一个 batch 内可以同时包含 prefill 序列和 decode 序列
3. 实现要点：
   - `prepare_prefill` 和 `prepare_decode` 合并为统一的 `prepare_batch`
   - attention kernel 需要处理混合的 query 长度（FlashAttention varlen 已支持）
   - scheduler 中添加优先级：decode > prefill chunk（保证 decode 低延迟）
4. 预期：消除 prefill 对 decode 的阻塞，decode 延迟从 ~50ms 降至 ~10ms（P99）

---

## 模块五：CUDA Graph 优化

### 现状
`capture_cudagraph()` 在 decode 路径录制了 [1, 2, 4, 8, 16, 32, 48, ...] 共 ~36 个 graph。prefill 路径未录制 graph。

### 瓶颈分析
1. **Graph 录制时间**：36 个 graph 录制耗时较长（~10-20 秒），且占用显存（每个 graph 存储中间激活）
2. **Graph 内存碎片**：大量 graph 导致 CUDA graph pool 内存碎片化
3. **不必要的 graph**：batch=1 和 batch=2 的 graph 收益极小（decode 本身是 memory-bound，kernel launch overhead 只占总延迟的 <5%）
4. **Graph padding**：当实际 batch 为 3 时，使用 graph_bs=4 的 graph，浪费 25% 的无效计算

### 改进方案
**方案：减少 graph 数量 + 动态 padding 策略**

```python
# 当前: [1, 2, 4, 8, 16, 32, 48, 64, ..., 512]  → 36 个 graph
# 改进: [4, 8, 16, 32, 64, 128, 256, 512]          → 8 个 graph
# 小 batch (<4) 用 eager mode，kernel launch overhead 可忽略
```

额外优化：`torch.compile` + CUDA graph 组合。PyTorch 2.0+ 的 `torch.compile` 可以大幅减少小 batch eager mode 的 kernel launch 开销。当前 `rotary_embedding.py` 和 `sampler.py` 已使用 `@torch.compile`，可以扩展到整个 decode 路径，从而进一步减少对 CUDA graph 的依赖。

---

## 模块六：Prefix Caching 实现效率

### 现状
`block_manager.py` 在 CPU 端用 xxhash 计算每个 block 的哈希，并维护 `hash_to_block_id` 字典。

### 瓶颈分析
1. **CPU 哈希开销**：每个 256-token block 需要 xxhash + numpy array conversion。对于长 prompt（4096 token = 16 blocks），这部分开销约 0.5-1ms
2. **O(N²) 哈希链**：`compute_hash(token_ids, prefix_hash)` 的链式设计是正确的，但如果 prefix_hash 碰撞，需要 fallback 到 token 级比较
3. **无 LRU 淘汰**：当 cache 满时，当前实现直接拒绝分配（`can_allocate` 返回 False），而非淘汰旧 block。这导致长 prompt 在 cache 满时完全无法分配

### 改进方案
**方案 A：GPU 哈希**
- 用 Triton kernel 在 GPU 上并行计算 block 哈希（类似 xxhash 的 GPU 实现）
- 避免 CPU-GPU 同步和数据传输
- 预期：消除 0.5-1ms 的 CPU 哈希延迟

**方案 B：LRU Eviction**
- 为每个 block 维护最后访问时间戳
- `can_allocate` 返回 False 时，淘汰最久未访问的 block（ref_count == 0 且非当前使用中）
- 允许长 prompt 在缓存压力下仍然可以运行

---

## 模块七：PreSchduling 中的 Python 开销

### 现状
每个 step 的 Python 路径：
```
Scheduler.schedule()    → O(N) 遍历 waiting+running 队列
ModelRunner.prepare_*() → Python list → CPU tensor → GPU (non_blocking)
ModelRunner.run_model() → GPU compute
Scheduler.postprocess() → Sequence.append_token()
```

### 瓶颈分析
以 decode step、batch=8 为例，每个 step 的 Python 开销：

| 操作 | 时间 |
|------|------|
| schedule() 队列遍历 | ~10μs |
| prepare_decode() tensor 构造 | ~50μs |
| run_model() CUDA launch | ~5μs |
| postprocess() | ~10μs |
| **GPU compute（主要）** | **~5-15ms** |

Python 开销 < 100μs vs GPU 5-15ms，Python 不是主要瓶颈。但以下情况会变严重：

1. **长 waiting 队列**：O(N) 的 `schedule()` 遍历可达到 ~1ms
2. **GC 触发**：Python GC 在 decode loop 中触发可导致 10-50ms 的卡顿
3. **SharedMemory pickle**：TP > 1 时，每个 method call 通过 pickle + shared memory 传递，开销 ~100-200μs per call

### 改进方案
**方案：Persistent Input Buffers**
- 预先分配 GPU tensor 作为输入缓冲区（类似 CUDA graph variables）
- `prepare_decode` 直接写入预分配的 tensor，避免每次 step 创建新 tensor
- 预期：消除 ~50μs per step 的 tensor 创建开销（收益极小，但降低 CPU-GPU 同步频率）

---

## 模块八：Prefill 计算效率

### 现状
Prefill 使用 `flash_attn_varlen_func`，QKV 投影为单一 `F.linear`，未分块处理。

### 瓶颈分析
Qwen3-8B 的模型参数：

| 参数 | 值 |
|------|-----|
| hidden_size | 4096 |
| intermediate_size | 12288 (4096×3) |
| num_layers | 32 |
| num_q_heads | 32 |
| num_kv_heads | 8 |
| head_dim | 128 |

Prefill 计算量最大的部分是 QKV 投影和 Attention：

```
单层 QKV 投影: 4096 × (4096+1024+1024) = 25M FLOPs per token
单层 MLP: 4096 × (12288+12288) ≈ 100M FLOPs per token
单层 Attention: 2 × 128 × N² ≈ 0.25M × N FLOPs (N=seq_len)
```

对于 512 token prefill，总计算量约 512 × 32 × 130M ≈ 2.1T FLOPs。

RTX 4060 FP16 tensor core 峰值 ~90 TFLOPS（理论），实际可达 ~40-50 TFLOPS。所以 512 token prefill 理论上只需 ~50ms。

### 改进方案
**方案：Chunked Prefill**
- 将 512 token prefill 拆成 2×256 的 chunk
- 第一个 chunk 的 attention 是 causal（256×256），第二个也是 causal 但需加载前一 chunk 的 KV
- 好处：prefill 不阻塞 decode，且减少单次 prefill 的峰值显存

---

## 总结：优先级排序

| 优先级 | 模块 | 问题 | 改进 | 预期收益 |
|--------|------|------|------|---------|
| **P0** | 权重量化 | FP16 模型 OOM | AWQ Triton kernel | **模型可运行** |
| **P0** | KV Cache | FP16 KV 占 64MB@512tok | 4-bit 量化 | **4× 更多并发** |
| **P1** | Continuous Batching | Prefill 阻塞 decode | Chunked prefill | **P99 延迟降 5×** |
| **P1** | Decode 带宽 | Memory-bound decode | 量化 KV fused kernel | **吞吐 +30-50%** |
| **P2** | CUDA Graph | 36 个 graph 浪费显存 | 减少到 8 个 | 显存节省 ~200MB |
| **P2** | Prefix Cache | CPU 哈希，无 LRU | GPU 哈希 + 淘汰 | 长 prompt 可用性 |
| **P3** | Python 开销 | 每 step 创建 tensor | Persistent buffers | 微优化（~0.5%） |

P0 项已在 `feature/quant` 分支中实现。P1 是实现差异化价值的关键方向。
