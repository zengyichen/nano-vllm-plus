# 基于 nano-vLLM 的 KV Cache 量化策略的设计与评估

#### 2354280 曾熠辰

#### 导师：冯恺睿

---


## TurboQuant

https://arxiv.org/pdf/2504.19874

提出了两种 KV Cache 量化方法：
1. TurboQuantMSE: 随机旋转后，每位存储其“最近质心”的 index；
  质心（共 $2^b$ 个）在旋转后分布上用 Lloyd–Max 优化求出。
2. TurboQuantProd：在（1）基础上，分出一位用于修正结果，使内积无偏。

---

- **Theorem 1**：$D_{\text{mse}} \leq d\cdot \frac{\sqrt{3\pi}}{2} \cdot \frac{1}{4^b}$（每坐标）
- **Theorem 2**：Prod 内积无偏，$\mathbb{E}[\hat{x} \cdot \hat{y}] = x \cdot y$
- **Lemma 4**：QJL 代价 → $D_{\text{prod}} / D_{\text{mse}} \approx \pi/2$

---

### 三种候选算法

| 算法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| TurboQuantMSE | 随机旋转 + Lloyd-Max 码本量化 | 理论保界（Theorem 1） | 内积有偏，MSE 受维度影响 |
| TurboQuantProd | MSE + QJL 内积残差修正 | 内积无偏（Theorem 2） | π/2 的 MSE 惩罚 |
| GroupedLinear | 逐组仿射量化 (scales + zeros) | 重建精度高，实现简单 | 需存储 scale/zero 元数据 |

---

### 原始设计
1. 方案 A：K=TurboQuantProd + V=TurboQuantProd（全 TurboQuant）
2. 方案 B：K=TurboQuantProd + V=GroupedLinear（AsymTurboQuant）

初步发现
- 方案 B 的 PPL 优于方案 A
- V 使用 GroupedLinear 对重建精度提升显著

---

### 存在的问题

1. PPL 测试采用续写方式，不够全面、准确
2. 只测试了 2 种组合，遗漏了另外 2 种：
   - K=GroupedLinear + V=TurboQuantProd（反转）
   - K=GroupedLinear + V=GroupedLinear（全 GroupedLinear）
3. 缺乏系统的数学分析来解释各组合的表现差异

---

### 设计重构

1. `nanovllm/config.py`：新增 `k_quant_algo`、`v_quant_algo` 字段，保留 `kv_quant_algo` 向后兼容
2. `nanovllm/utils/quant.py`：重构 `AsymTurboQuantKVQuantizer`，K 和 V 独立选择量化方法
3. `nanovllm/engine/model_runner.py`：根据 K/V 组合自动选择 decode 后端
4. `nanovllm/layers/attention.py`：统一 split cache 路径，支持所有组合的反量化

使用命令行分别指定 K/V 的量化方案。

---

### 实验

- 模型：Qwen3-8B-AWQ (4-bit 权重)
- 数据集：WikiText-2 test set
- prefix_len: 256 tokens (prefill)
- max_eval_tokens: 512 tokens (decode 逐个评估)
- 量化位宽：K=4-bit, V=4-bit (group_size=32 for GroupedLinear)

---

### 五种测试模式

| Mode | K 算法 | V 算法 | Decode 后端 |
|------|--------|--------|-------------|
| NoQuant | FP16 | FP16 | FlashAttention |
| K=Prod_V=Prod | TurboQuantProd | TurboQuantProd | dequant_flash |
| K=Prod_V=Grouped | TurboQuantProd | GroupedLinear | asym_turboquant (fused) |
| K=Grouped_V=Prod | GroupedLinear | TurboQuantProd | dequant_flash |
| K=Grouped_V=Grouped | GroupedLinear | GroupedLinear | dequant_flash |

测试 PPL。

---

### 核心结果

```
mode                   success ppl         mean_nll  tokens  ppl_ratio  delta_pct
---------------------------------------------------------------------------------
NoQuant                True     9.987      2.301     512     1.000      +0.00%
K=Prod_V=Prod          True    12.121      2.495     512     1.214     +21.37%
K=Prod_V=Grouped       True    10.418      2.343     512     1.043      +4.31%
K=Grouped_V=Prod       True    10.201      2.323     512     1.021      +2.14%
K=Grouped_V=Grouped    True    10.131      2.316     512     1.014      +1.44%
```

1. 使用 GroupedLinear 相比 TurboQuant 有巨大优势
2. V 使用 GroupedLinear 比 K 使用 GroupedLinear 对 PPL 的提升更大

---

### K=Grouped_V=Prod 优于 K=Prod_V=Grouped

按照原始设计，K=Prod + V=Grouped 被认为是"最优"的非对称组合。

但实验显示反转组合 (K=Grouped, V=Prod) 反而更好。

---

### 从 Attention 机制分析
Attention 的计算流程：
```
S = Q @ K^T          ← K 的量化误差直接影响 attention 分数
A = softmax(S)       ← softmax 可能放大某些维度的偏差
O = A @ V            ← V 的量化误差线性传播到输出
```
---

### K 的精度比 V 更重要

1. K 的误差经 softmax 非线性放大：
   - softmax 是高度非线性的 (指数函数)
   - K 的量化误差 → 内积偏差 → softmax 后某些 token 的权重被指数级放大或抑制
   - 设 ΔS = S_quant - S_fp，则 softmax 偏差 ∝ exp(ΔS) - 1（非对称，正向偏差放大更多）

2. V 的误差是线性传播的：
   - O = Σ_i A_i · V_i，V 的量化误差被 attention 权重加权平均
   - 加权平均是一种平滑操作，误差被分散到多个 token

---

### TurboQuant 优势

- 内积是无偏的（Theorem 2）。
  在长序列上，无偏性保证 attention 分数不会系统性偏离
- 在低位宽下表现优秀

---

### 下阶段研究计划

1. 测试更大模型、更长上下文下的量化表现
2. 探索更低位宽、混合位宽量化
3. 进一步学习 vLLM 框架，开展算子优化、推理加速等研究

