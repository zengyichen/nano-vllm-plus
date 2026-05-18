# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 构建与安装

```bash
# 一键安装（来自 install.sh）
export CUDA_HOME=/usr/local/cuda
export FLASH_ATTENTION_FORCE_BUILD=1 MAX_JOBS=4
uv pip install -e . --no-build-isolation
```

依赖：torch>=2.4.0, triton>=3.0.0, transformers>=4.51.0, flash-attn, xxhash。

## 架构

Nano-vLLM 是一个轻量级 vLLM 重实现（约 1,200 行 Python），API 与 vLLM 基本一致。

**入口点**：`LLM`（`nanovllm/llm.py`）继承自 `LLMEngine`，是用户直接使用的唯一类。

### 引擎流水线（`nanovllm/engine/`）

`LLMEngine.generate()` 驱动整个循环：添加请求 → `step()` → 收集输出。

`step()` 调用 `scheduler.schedule()` → `model_runner.call("run", seqs, is_prefill)` → `scheduler.postprocess()`。

- **Scheduler**（`scheduler.py`）：先 prefill 后 decode 的调度策略。prefill 阶段贪心打包等待队列中的序列，直到达到 `max_num_batched_tokens` 上限。decode 阶段轮转运行中的序列。抢占方式为释放 KV cache block 并重新入队。
- **BlockManager**（`block_manager.py`）：PagedAttention 风格的 KV cache 分配器，支持基于 xxhash 内容哈希的**前缀缓存**。block 使用引用计数管理。哈希链：每个 block 的哈希值包含前一个 block 的哈希，从而支持跨序列的前缀匹配。
- **Sequence**（`sequence.py`）：追踪 token ID、KV cache block 表、已生成 token 数量、状态（WAITING/RUNNING/FINISHED）。自定义 pickle 以优化跨进程传输。
- **ModelRunner**（`model_runner.py`）：核心模块。负责：
  1. 模型预热（两次 prefill 以测量激活内存峰值）
  2. KV cache 显存预算计算与分配（量化或 FP）
  3. `prepare_prefill()` / `prepare_decode()` — 构建输入张量和全局 `Context`
  4. `run_model()` — eager prefill、eager decode 或 CUDA graph 回放
  5. decode 阶段多种 batch size 的 CUDA graph 录制
  6. 通过共享内存协调 TP（rank 0 写入方法调用；worker rank 循环读取执行）

### 模型定义（`nanovllm/models/qwen3.py`）

`Qwen3ForCausalLM` 是唯一支持的模型。自动从 HF `config.json` 检测 AWQ 量化（`quantization_config.quant_method == "awq"`），并切换到 AWQ 线性层。`packed_modules_mapping` 字典将 HF 权重名称（q_proj、k_proj、v_proj、gate_proj、up_proj）映射到融合层的参数名和 shard ID。

### 层（`nanovllm/layers/`）

- **attention.py**：最复杂的层。**prefill** 阶段：将 K/V 存入缓存（量化或 FP），调用 `flash_attn_varlen_func`。**decode** 阶段：根据 `quant_decode_backend` 路由到不同量化后端，或对 FP 使用 `flash_attn_with_kvcache`。路由逻辑会逐级回退。
- **awq_linear.py**：AWQ 权重量化（4-bit）。使用 Triton autotune 的融合反量化+矩阵乘法内核，支持 Split-K 和 atomic-add 规约。Triton 前置条件不满足时回退到分块 PyTorch 反量化。
- **fused_quant_attn.py**：融合注意力分数计算与即时 KV 反量化的 Triton 内核，避免物化 KV 张量。三个变体：`fused_score`（仅 K 分数）、`fused_attention`（TurboQuant Prod 路径）、`fused_asym_quantized_decode_attention`（AsymTurboQuant 路径，向量化 V 反量化 + `tl.dot` 累积）。
- **linear.py**：标准线性层，支持张量并行切分 — `ColumnParallelLinear`、`RowParallelLinear`（含 all-reduce）、`QKVParallelLinear`、`MergedColumnParallelLinear`。每个参数都有 `weight_loader` 属性供模型加载器使用。

### 工具（`nanovllm/utils/`）

- **context.py**：全局可变 `Context` 数据类，通过 `get_context()` / `set_context()` 访问。携带 slot mapping、block table、序列长度和量化元数据贯穿整个前向传播，避免逐层传参。每次 step 后必须调用 `reset_context()`。
- **loader.py**：遍历 `.safetensors` 文件，解析 packed module mapping，分发到各个参数的 `weight_loader` 可调用对象。
- **quant.py**：KV cache 量化子系统。三个量化器类实现 `BaseKVQuantizer` 接口：
  - `TurboQuantMSEKVQuantizer`（3-bit）：基于 MSE 的向量量化，含学习的旋转矩阵和码本
  - `TurboQuantProdKVQuantizer`（3-4 bit）：在 MSE 基础上扩展内积残差（QJL）
  - `AsymTurboQuantKVQuantizer`（4-bit K + 4-bit 分组线性 V）：分离式 K/V 缓存布局。K 使用 TurboQuantProd；V 使用逐组仿射量化（scales + zeros）

  通过 `get_kv_quantizer(config)` 根据 `kv_quant_algo` 选择量化器。

### Decode 后端路由

`Attention.forward()` 的 decode 路径按以下优先级选择后端：
1. `asym_turboquant` — 若码本可用则使用融合内核，否则反量化到 workspace → FlashAttention
2. `dequant_flash` — 从缓存中 gather 量化 KV → 反量化到预分配 workspace → FlashAttention varlen
3. `fused` — TurboQuant 融合分数+注意力 Triton 内核（不物化 KV 张量）
4. 回退 — 反量化整个 KV cache → `flash_attn_with_kvcache`

## 核心概念

- **KV cache 以 block 为单位**（默认 256 token）。block table 将逻辑位置映射到物理 block ID。
- **张量并行**使用 `multiprocessing.spawn` + 共享内存。rank 0 上的 `ModelRunner.call()` 将方法名和参数 pickle 到共享内存；worker rank 反序列化并执行。GPU 集合通信使用 NCCL。
- **CUDA graph**仅在 decode 阶段录制，batch size 为 [1, 2, 4, 8, 16, 32, ...]，最大不超过 `cuda_graph_max_bs`（上限 512，≤10GB 显存的 GPU 上限 32）。`run_model()` 中用 graph 回放替代 eager 执行。
- **量化 decode graph**通过 `kv_decode_graph=True` 显式启用，仅对 `dequant_flash` 和 `asym_turboquant` 后端有效。graph 变量包含量化专用张量（`quant_slot_mapping`、`quant_cu_seqlens_k`）。
- **显存预算**在 `allocate_kv_cache()` 中计算：从 `gpu_memory_utilization * 总显存` 中扣除模型权重、激活峰值（预热阶段测量）、安全边距和可选的 decode workspace，剩余部分除以每 block 字节数得到 `num_kvcache_blocks`。
