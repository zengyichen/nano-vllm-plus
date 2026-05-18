# Nano-vLLM Benchmarks

本目录包含 nano-vllm 的完整 benchmark 套件，覆盖**吞吐量**、**显存占用**、**KV cache 压缩率**和**量化质量（perplexity）**四个维度。

所有 benchmark 均支持 `--help` 查看参数说明。

## 依赖

```bash
# 在 nano-vllm 项目根目录下
uv pip install -e .
```

所有 benchmark 通过 `nanovllm` 包导入，无需额外设置 `PYTHONPATH`。

## Benchmark 概览

| 文件 | 测试内容 | 适用场景 |
|------|---------|---------|
| `bench.py` | 纯 decode 吞吐量 | 快速验证推理速度 |
| `bench_quant.py` | 量化吞吐量 + OOM 边界 | 对比不同 KV 量化方案的性能和显存极限 |
| `bench_kv.py` | KV cache 显存压缩率 | 测量量化对 KV cache 的压缩效果 |
| `bench_ppl.py` | 量化质量（perplexity） | 评估量化引入的精度损失 |
| `bench_memory.py` | 显存分配压力测试 | 验证大 batch/长序列下的显存利用率 |

---

## 1. bench.py — 基础吞吐量

最简单的 benchmark，用随机 token 序列测试纯 decode 吞吐量。

```bash
# 使用默认参数（Qwen3-0.6B, 4条序列）
python benchmarks/bench.py

# 自定义模型和参数
python benchmarks/bench.py \
  --model ~/huggingface/Qwen3-8B-AWQ/ \
  --num-seqs 8 \
  --max-input-len 512 \
  --max-output-len 1024

# 开启 CUDA graph
python benchmarks/bench.py --enforce-eager=False
```

**输出示例**：
```
Total: 4096 tok, Time: 12.34s, Throughput: 332.01 tok/s
```

---

## 2. bench_quant.py — 量化吞吐量 + OOM 边界

核心 benchmark，对比三种模式：
- **baseline**：无 KV 量化
- **quant**：TurboQuant Prod 4-bit KV 量化
- **asym**：AsymTurboQuant 4-bit KV 量化

### 2a. 吞吐量对比（所有模式）

```bash
# 一次运行对比所有模式（每个模式独立子进程，互不干扰）
python benchmarks/bench_quant.py

# 单独运行某一模式（结果以 JSON 输出到最后一行）
python benchmarks/bench_quant.py --mode baseline
python benchmarks/bench_quant.py --mode quant
python benchmarks/bench_quant.py --mode asym

# 指定模型
python benchmarks/bench_quant.py --model ~/huggingface/Qwen3-8B-AWQ/
```

**输出示例**：
```
Baseline Throughput : 45.23 tok/s
TurboQuant 4-bit    : 38.17 tok/s
AsymTurboQuant 4-bit: 48.91 tok/s
Throughput Impact   : -15.61%
Asym Impact         : +8.14%
```

### 2b. OOM 边界测试

测试各模式在递增上下文长度下的显存极限。

```bash
# 对比 baseline 和 asym 两种模式的 OOM 边界
python benchmarks/bench_quant.py --mode oom

# 单独测试某种模式
python benchmarks/bench_quant.py --mode oom --oom-submode baseline
python benchmarks/bench_quant.py --mode oom --oom-submode asym
```

测试从 5000 到 16000 tokens（步长 1000），输出每一步的 PASS/FAIL 状态。

---

## 3. bench_kv.py — KV Cache 压缩率

在固定上下文长度下，测量三种模式的 KV cache 实际显存占用。

```bash
# 默认 2048 tokens 上下文
python benchmarks/bench_kv.py

# 自定义上下文长度
python benchmarks/bench_kv.py --context-len 4096

# 单独测试某模式
python benchmarks/bench_kv.py --mode noquant
python benchmarks/bench_kv.py --mode asym
```

**输出示例**：
```
=== KV Compression Summary ===
mode         success  peak_used_GB  bytes/token  compression_vs_noquant  saving_vs_noquant(%)
noquant      True    0.4821        246.78       1.0000                  0.00
kvquant      True    0.1873        95.87        2.5734                  61.14
asym         True    0.1402        71.76        3.4389                  70.92
```

- `compression_rate_vs_noquant`：相对未量化的压缩倍率（越高越好）
- `saving_pct_vs_noquant`：显存节省百分比
- `bytes_per_token`：每 token 实际 KV cache 占用（含 scale/zero）

---

## 4. bench_ppl.py — 量化质量（Perplexity）

使用固定文本，在三种 KV 量化模式下分别计算 perplexity。采用**逐 token 评估**的方式，每步创建新 Sequence 并走完整的 prefill + prefix-cache 路径，真实反映量化对生成质量的影响。

```bash
# 默认 1000 词的英语文本
python benchmarks/bench_ppl.py

# 自定义文本长度
python benchmarks/bench_ppl.py --sample-words 500

# 单独测试某模式
python benchmarks/bench_ppl.py --mode noquant
python benchmarks/bench_ppl.py --mode asym
```

**输出示例**：
```
=== Perplexity Summary ===
mode      success  perplexity  mean_nll  eval_tokens  ppl_ratio_vs_noquant  ppl_delta_pct
noquant   True    5.234567    1.655234  1456         1.000000              +0.00
kvquant   True    5.312456    1.670012  1456         1.014879              +1.49
asym      True    5.289123    1.665601  1456         1.010421              +1.04
```

- `ppl_ratio_vs_noquant`：相对于未量化的 perplexity 比率（越接近 1.0 越好）
- `ppl_delta_pct`：perplexity 变化百分比

---

## 5. bench_memory.py — 显存分配压力测试

通过配置极限 batch size 和序列长度，验证 VRAM 分配器是否能正确利用可用显存。

```bash
# 默认参数：128 batch, 1024 seq_len
python benchmarks/bench_memory.py

# 极限压力测试
python benchmarks/bench_memory.py \
  --model ~/huggingface/Qwen3-8B-AWQ/ \
  --max-batch 256 \
  --seq-len 2048 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95
```

**输出示例**：
```
=== Final Memory Status ===
KV Cache Total Blocks : 450
KV Cache Peak Used    : 432 (96.0%)
KV Cache Currently Used: 0 (0.0%)
```

关注 `peak_usage_percent`：接近 100% 说明 VRAM 分配器充分利用了可用显存。

---

## 典型使用流程

### 1. 快速验证（开发阶段）

```bash
# 验证推理管道正常工作
python benchmarks/bench.py --num-seqs 1 --max-input-len 128 --max-output-len 64
```

### 2. 全面对比（发布前）

```bash
# 步骤 1：吞吐量对比
python benchmarks/bench_quant.py --model ~/huggingface/Qwen3-8B-AWQ/

# 步骤 2：KV cache 压缩率
python benchmarks/bench_kv.py --model ~/huggingface/Qwen3-8B-AWQ/ --context-len 4096

# 步骤 3：质量评估（perplexity）
python benchmarks/bench_ppl.py --model ~/huggingface/Qwen3-8B-AWQ/ --sample-words 2000

# 步骤 4：OOM 边界
python benchmarks/bench_quant.py --mode oom --model ~/huggingface/Qwen3-8B-AWQ/
```

### 3. 显存压力测试

```bash
python benchmarks/bench_memory.py \
  --model ~/huggingface/Qwen3-8B-AWQ/ \
  --max-batch 512 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95
```

---

## 注意事项

- **默认模型路径**：benchmark 默认使用 `~/huggingface/Qwen3-8B-AWQ/` 或 `~/huggingface/Qwen3-0.6B/`，请通过 `--model` 参数指定实际路径
- **子进程隔离**：`bench_quant.py`、`bench_kv.py`、`bench_ppl.py` 的 `all` 模式会将每个方案放在独立子进程中运行，避免 CUDA 显存碎片影响公平对比
- **Perplexity 运行时间**：`bench_ppl.py` 采用逐 token 评估，运行时间与 token 数的平方成正比。1000 词约 1500 tokens，需要约 1500 次 prefill 调用，请耐心等待
- **OOM 测试**：OOM 测试会触发 CUDA OOM（这是预期行为），子进程会优雅捕获异常，不影响主进程
