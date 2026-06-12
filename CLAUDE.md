# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

nano-vLLM 是一个从零构建的轻量级 vLLM 实现，专注于可读性和 LLM 推理引擎的核心原理演示。当前支持 Qwen3-0.6B 模型（FP16）。

**目标**：设计并扩展框架，支持更多功能、更高性能（高吞吐、小占用、低延迟）。

**分支**：
- `main` — 核心推理引擎，支持 FP16 模型（Qwen3-0.6B），无量化
- `feature/quant` — 量化分支，实现 AWQ 权重量化 + TurboQuant KV Cache 量化，支持 Qwen3-8B-AWQ

## 构建与安装

```bash
# 一键安装
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FLASH_ATTENTION_FORCE_BUILD=1 MAX_JOBS=4
uv pip install -e . --no-build-isolation
```

**依赖**：torch>=2.4.0, triton>=3.0.0, transformers>=4.51.0, flash-attn, xxhash。

```bash
# 运行
export CUDA_HOME=/usr/local/cuda
.venv/bin/python example.py
```

## 架构

`LLM`（`nanovllm/llm.py`，继承 `LLMEngine`）是用户入口。整个框架约 1200 行 Python 代码。

### 引擎流水线（`nanovllm/engine/`）

`LLMEngine.generate()` 驱动主循环：添加请求 → `step()` → 收集输出。

`step()` 调用：`Scheduler.schedule()` → `ModelRunner.run()` → `Scheduler.postprocess()`。

- **Scheduler**（`scheduler.py`）：先 prefill 后 decode 的调度策略。prefill 阶段贪心打包 waiting 队列序列至 `max_num_batched_tokens` 上限。decode 阶段轮转 running 序列。抢占通过释放 KV cache block 并重新入队。
- **BlockManager**（`block_manager.py`）：PagedAttention 风格的 KV cache 分配器，支持基于 xxhash 的前缀缓存。block 使用引用计数管理，哈希链支持跨序列前缀匹配。
- **Sequence**（`sequence.py`）：追踪 token ID、KV cache block 表、状态（WAITING/RUNNING/FINISHED）。自定义 pickle 支持跨进程传输。
- **ModelRunner**（`model_runner.py`）：核心模块，负责模型加载、KV cache 分配/计算、prefill/decode 的前处理、模型执行、CUDA graph 录制/回放。

### 模型定义（`nanovllm/models/qwen3.py`）

`Qwen3ForCausalLM` 是当前唯一支持的模型。`packed_modules_mapping` 字典将 HF 权重名称映射到融合层参数。支持张量并行和 Q/KV 头的自动分裂。

### 层（`nanovllm/layers/`）

- **attention.py**：最复杂的层。prefill 调用 `flash_attn_varlen_func`。decode 调用 `flash_attn_with_kvcache`。
- **linear.py**：标准 FP16 线性层，支持张量并行切分（Column/Row/QKV/Merged 四种）。每个参数有 `weight_loader` 属性给加载器使用。
- **rotary_embedding.py**：RoPE 嵌入，`@torch.compile` 优化。
- **sampler.py**：采样器，`@torch.compile` 优化，Gumbel-max 实现。
- **embed_head.py**：词嵌入和语言模型头的张量并行实现。

### 工具（`nanovllm/utils/`）

- **context.py**：全局 `Context` 数据类，通过 `get_context()`/`set_context()` 访问。传递 slot mapping、block table、序列长度等元数据。
- **loader.py**：遍历 `.safetensors`，解析 packed module mapping，分发到 `weight_loader`。

### 硬件目标

- GPU：RTX 4060 Laptop (Ada Lovelace, 8GB VRAM, 24 SMs)
- 显存带宽：~256 GB/s (128-bit × 8001 MHz)
- Decode 是 memory-bound（算术强度 ~0.5 FLOP/byte）

## 开发流程

1. 实现功能
2. 验证正确性和性能（使用 `benchmarks/` 中的脚本）
3. 写详细文档（原理 + 实现，放在 `docs/`）
4. `git commit` + `git push`

## 性能优化路线图

详见 `docs/perf_analysis.md`。核心优先级：
- **P0** — AWQ 权重量化（`feature/quant`） + KV Cache 量化
- **P1** — Continuous batching（chunked prefill + mixed batch）
- **P1** — decode 带宽优化（量化 fused kernel）
- **P2** — CUDA graph 精简、前缀缓存 LRU、GPU 哈希

## 代码规范

- 类型提示：全程使用类型注解
- 注释：中英文均可，关键数学推导和优化原理必须有中文说明
- 测试：`benchmarks/bench_thruput.py` 是性能测试入口
- 文档：设计文档放在 `docs/`，benchmark 结果放在 `benchmarks/results/`
- 新特征分支命名：`feature/<name>`
