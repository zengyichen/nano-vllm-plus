# 汇报讲稿：基于 nano-vLLM 的 KV Cache 量化策略的设计与评估

---

## 封面

> 大家好，我今天汇报的题目是"基于 nano-vLLM 的 KV Cache 量化策略的设计与评估"。指导教师冯恺睿老师。
>
> 本次工作的核心问题是：在 LLM 推理中，KV Cache 的显存占用随序列长度线性增长，成为吞吐瓶颈。量化是自然的解决方案——将 KV Cache 从 FP16 压缩到 4-bit，理论减少约 4 倍显存。但量化引入的误差会影响 attention 计算，进而损害模型输出质量。
>
> 我们的目标是：理解不同量化算法在 K 和 V 上的实际表现差异，通过系统实验和理论分析找到最优的 K/V 组合策略。

---

## TurboQuant 论文简介

> 我们的工作基于 TurboQuant 这篇论文（arXiv:2504.19874）。
>
> 论文提出了两种 KV Cache 量化方法。
>
> 第一种是 **TurboQuantMSE**：先对向量做随机旋转，将能量均匀分布到各维度，然后每位存储其最近质心的索引。质心共有 2 的 b 次方个，在旋转后的分布上通过 Lloyd-Max 算法迭代优化求出。这是一种 MSE 最优的标量量化。
>
> 第二种是 **TurboQuantProd**：在 MSE 的基础上，分出一位用于 QJL 内积残差修正。QJL 是 Johnson-Lindenstrauss 引理的量化变体——核心思想是，量化残差在与随机投影的内积中趋于零均值，因此可以用它来修正内积偏差。这使得最终的内积估计在理论上无偏。

---

## 关键定理

> 论文有三个核心理论结果。
>
> **Theorem 1**：TurboQuantMSE 的每坐标 MSE 有上界——D_mse 小于等于 d 乘以根号 3π 除以 2，再乘以 1/4^b。这意味着每增加 1 比特，误差减少约 4 倍。注意这是每坐标的 bound，在归一化和旋转后的空间上成立。
>
> **Theorem 2**：TurboQuantProd 的内积是无偏的——量化后内积的期望等于真实内积。这是 Prod 区别于所有其他量化方法的根本性质。它不是"平均意义上"的无偏——是对任意固定的 query 和 key 对，逐对保证的。
>
> **Lemma 4**：无偏性的代价——QJL 修正使得重建 MSE 与纯 MSE 量化的比值约为 π/2，约等于 1.57 倍。这个比值是渐进的，在我们的实验中得到了精确验证。

---

## 三种候选算法

> 我们从论文中选取了三种候选算法进行实现和对比。
>
> **TurboQuantMSE**：随机旋转加 Lloyd-Max 码本量化。优点是 MSE 有理论上界保证，缺点是对内积估计有偏。
>
> **TurboQuantProd**：在 MSE 基础上增加 QJL 内积残差修正。优点是内积无偏（Theorem 2），缺点是 MSE 被放大 π/2 倍——用重建精度换取内积保真度。
>
> **GroupedLinear**：逐组仿射量化。把向量按维度分组，每组独立计算 scale 和 zero，做均匀量化。优点是重建精度高、实现简单，缺点是需要额外存储 scale 和 zero 元数据，且没有内积无偏性保证。
>
> 三种算法之间的核心权衡是：重建精度 vs 内积无偏性。接下来我们会看到这个权衡在当前配置下如何表现。

---

## 原始设计

> 我们的原始设计方案只有两种组合。
>
> 方案 A：K 和 V 都用 TurboQuantProd——全 TurboQuant 方案。
> 方案 B：K 用 TurboQuantProd，V 用 GroupedLinear——称为 AsymTurboQuant。
>
> 初步测试发现方案 B 的 PPL 优于方案 A。这让我们认为 V 使用 GroupedLinear 对重建精度的提升是显著的、有效的。

---

## 存在的问题

> 但这个结论不够完整，存在三个问题。
>
> 第一，之前的 PPL 测试采用续写方式，不够系统和准确——没有区分 prefill 和 decode 阶段的评估。
>
> 第二，只测试了 2 种 K/V 组合，但完整的组合空间应该有 4 种：K 可以选 TurboQuant 或 GroupedLinear，V 同理。我们遗漏了反转组合 K=Grouped + V=Prod，以及全 GroupedLinear 组合。这两个遗漏的组合恰好后来被证明是最优的。
>
> 第三，缺乏系统的数学分析来解释各组合的表现差异——为什么 V 的 GroupedLinear 效果好？K 换成 GroupedLinear 会不会更好？这些问题没有从 attention 机制的角度得到回答。

---

## 设计重构

> 为了解决上述问题，我们首先需要改造 nano-vLLM 的代码，实现 K 和 V 量化策略的完全分离。我们修改了四个核心文件。
>
> 第一，`config.py`：新增 `k_quant_algo` 和 `v_quant_algo` 两个独立字段，同时保留 `kv_quant_algo` 用于向后兼容。旧参数自动映射到新字段。
>
> 第二，`quant.py`：重构 `AsymTurboQuantKVQuantizer`，K 和 V 各自独立维护算法选择和工厂方法。关键修复是分离 `_ensure_algo()` 和 `_ensure_v_algo()`——确保当 K 用 GroupedLinear 而 V 用 TurboQuantProd 时，两者的码本和量化参数互不干扰。
>
> 第三，`model_runner.py`：根据 K/V 组合自动选择 decode 后端——K=Prod + V=Grouped 走 asym_turboquant 融合 kernel，其他组合走 dequant_flash。
>
> 第四，`attention.py`：统一 split cache 路径，支持所有组合的反量化存储和加载。
>
> 现在用户可以通过命令行分别指定 K 和 V 的量化方案。

---

## 实验设置

> 我们使用 Qwen3-8B-AWQ 模型，权重已经是 4-bit AWQ 量化的。数据集是 WikiText-2 测试集。
>
> 评估方式：前 256 个 token 作为 prefill（prefix），在 prefill 阶段评估 token [1:256] 的 NLL。之后逐个 token 做 decode，最多评估 512 步，每一步用模型预测的下一个 token 的概率与真实下一个 token 对比计算 NLL。这种逐 token 评估方式比续写方式更准确——它测量的是模型在所有位置的预测质量，而不仅仅是续写结果的流畅度。
>
> 量化参数：K 统一 4-bit，V 统一 4-bit。GroupedLinear 的 V 使用 group_size=32，即把 128 维的 head_dim 切分成 4 组。

---

## 五种测试模式

> 我们测试了五种模式，覆盖全部可能的 K/V 组合空间。
>
> NoQuant：FP16 KV Cache，作为精度的理论上界。
>
> K=Prod_V=Prod：K 和 V 都使用 TurboQuantProd，走 dequant_flash 后端——全 TurboQuant 方案。
>
> K=Prod_V=Grouped：K 用 TurboQuantProd，V 用 GroupedLinear，这是唯一能走 asym_turboquant 融合 kernel 的组合——即原始的 AsymTurboQuant 方案。
>
> K=Grouped_V=Prod：K 用 GroupedLinear，V 用 TurboQuantProd——反转组合，走 dequant_flash 后端。这个组合在原始设计中是被遗漏的。
>
> K=Grouped_V=Grouped：全部使用 GroupedLinear——全分组量化方案，同样在原始设计中被遗漏。
>
> 所有模式的评估方式完全一致：prefill 计算 prefix 的 NLL + decode 逐 token 计算 NLL。

---

## 核心结果

> 这是我们的核心实验结果。
>
> NoQuant baseline 的 PPL 为 9.987，这是精度上界。
>
> 全 TurboQuant（K=Prod_V=Prod）的 PPL 为 12.121，相对 NoQuant 退化 +21.37%。这是一个不可接受的退化——全 TurboQuant 在 d=128 的配置下完全失败。
>
> K=Prod_V=Grouped（AsymTurboQuant）的 PPL 为 10.418，退化 +4.31%。可接受，且能使用融合 kernel。
>
> K=Grouped_V=Prod（反转组合）的 PPL 为 10.201，退化仅 +2.14%。比 AsymTurboQuant 更好，尽管 V 用的是 MSE 更差的 TurboQuant——这在直觉上是反过来的。
>
> K=Grouped_V=Grouped 的 PPL 为 10.131，退化仅 +1.44%，几乎是所有量化方案中的最优结果，与 NoQuant 的差距极小。
>
> 从这张表可以读出两个关键发现。第一，使用 GroupedLinear 相比 TurboQuant 有巨大优势——全 TurboQuant +21% 的退化是完全不可接受的。第二，V 使用 GroupedLinear 比 K 使用 GroupedLinear 对 PPL 的提升更大——对比 K=Prod 行的两个组合，V 从 Prod 换成 Grouped 改善 17 个百分点；而对比 V=Prod 列的两个组合，K 从 Prod 换成 Grouped 改善 19 个百分点。但绝对值上，K 的改善空间更大。

---

## K=Grouped_V=Prod 优于 K=Prod_V=Grouped

> 这是整个实验中最反直觉的发现。
>
> 按照原始设计，K=Prod + V=Grouped 被认为是非对称组合中的最优方案——K 享受 TurboQuant 的内积无偏性，V 享受 GroupedLinear 的高重建精度。但实验结果恰恰相反：K=Grouped + V=Prod 的 PPL 退化 +2.14%，而 K=Prod + V=Grouped 退化 +4.31%。反转组合反而更好。
>
> 为什么会这样？要解释这个现象，需要分析 K 和 V 的量化误差在 Attention 计算中的传播路径有何不同。

---

## 从 Attention 机制分析

> Attention 计算分为三步。
>
> 第一步 S = Q 乘 K 的转置——K 的量化误差在这里直接影响 attention 分数。设 ΔK = K_quantized - K_fp，则 attention score 的误差 ΔS = Q × ΔK^T。这是 query 向量和 K 误差向量的内积。
>
> 第二步 A = softmax(S)——softmax 可能放大某些维度的偏差。注意 softmax 是指数函数。
>
> 第三步 O = A × V——V 的量化误差在这里线性传播到输出。ΔO = Σ A_i × ΔV_i，即 V 的误差被 attention 权重加权求和。
>
> 关键区别在于：K 走 softmax（非线性），V 走加权平均（线性）。

---

## K 的精度比 V 更重要

> 为什么 K 的精度比 V 的精度更重要？从两个方面论证。
>
> **第一，K 的误差经 softmax 非线性放大。**
>
> softmax 是高度非线性的指数函数：A_i = exp(S_i) / Σ exp(S_j)。当某个位置的 attention score 被量化噪声放大 ΔS 时，该位置的 attention 权重被放大 exp(ΔS) 倍。这是一个非对称的放大效应——正向偏差（指数放大）的影响远大于负向偏差（指数缩小）。
>
> 考虑 K 用 TurboQuant 的情况：MSE 约 0.65，经过内积计算后，S 中多个位置的噪声幅度达到 0.5-1.0。softmax 将这些噪声指数级放大。即使 V 是完美的 FP16，错误的 attention 权重必然导致错误的输出。
>
> 反之，K 用 GroupedLinear（MSE 约 0.15）时，S 的噪声很小，softmax 分布接近真实。即使 V 有较大误差，加权平均也会将其分散。
>
> **第二，V 的误差是线性传播的。**
>
> V 的误差传播 ΔO = Σ A_i × ΔV_i 是 attention 权重对 V 误差的加权平均。加权平均天然是一种平滑算子——多个带有独立噪声的 V 向量被平均后，噪声方差被大幅降低，缩小约 1/N 倍。
>
> **定量对比**：K 的 MSE=0.65 经 softmax 放大后导致 attention 分布 KL 散度约 0.025；而 V 的 MSE=0.65 经加权平均后对输出的影响被显著衰减。K 的精度优先级大于 V 的精度优先级——这一发现在 PPL 表中得到完美验证：K=Grouped 的两个组合无论 V 用什么，都优于 K=Prod 的对应组合。

---

## TurboQuant 的优势

> 尽管 K=Grouped_V=Grouped 在精度上最优，TurboQuant 仍有其不可替代的价值，体现在两个方面。
>
> **第一，内积无偏性（Theorem 2）。** 对于任何固定的 query 向量，E[q^T × K_quantized] = q^T × K_real。这不是平均意义上的性质——是逐 query 逐 key 对的严格保证。在极长序列上（超过 16K tokens），attention 分布依赖大量 key 的内积比较。无偏性保证：即使单对内积有方差，大量比较的排序不会被系统性改变。GroupedLinear 虽然 MSE 低，但分组的 per-group scale/zero 在特定数据分布下可能产生系统性偏差。
>
> **第二，在低位宽下表现优秀。** 我们额外在 bench_mse 中对 2/3/4 bit 三种位宽做了全面对比。在 2-bit 时，TurboQuantMSE 的 MSE 为 0.115，反而优于 GroupedLinear 的 0.154。因为极低位宽下 Lloyd-Max 优化的码本（4 个质心）比均匀量化的 4 个等间距级别更高效。而在 3-bit 和 4-bit 下，均匀量化的级别数（8 和 16）足够密集，分组策略的优势才开始显现。这意味着 TurboQuant 是低位宽场景下的首选——3-bit TurboQuant 可能以更少的存储达到与 4-bit GroupedLinear 相似的精度。

---

## 下阶段研究计划

> 基于当前发现，后续研究分为三个方向。
>
> 第一，测试更大模型和更长上下文下的量化表现。当前实验限于 Qwen3-8B、512 token。需要验证结论在 14B/32B 模型、4K/32K 上下文上是否仍然成立。特别关注 TurboQuant 的无偏性在长序列上能否弥补其 MSE 劣势。
>
> 第二，探索更低位宽和混合位宽量化。3-bit TurboQuant 在 MSE benchmark 中已展现出潜力——它的 MSE 介于 4-bit Prod 和 4-bit GroupedLinear 之间，但存储更少。K=3-bit TurboQuant + V=4-bit GroupedLinear 的混合方案可能成为精度和压缩比的最优平衡点。
>
> 第三，进一步学习 vLLM 框架的 KV Cache 量化实现，开展算子优化和推理加速研究。特别是 fused kernel 的泛化——目前只支持 K=Prod + V=Grouped，能否扩展到其他组合？这需要深入研究 Triton 的 tile-level 计算与反量化操作的融合策略。

---

## 总结

> 最后总结本次工作的核心发现。
>
> 我们实现了 nano-vLLM 中 K/V 量化策略的完全分离，并在 WikiText-2 上系统测试了四种 K/V 组合的 PPL。实验结果揭示了三个关键洞察：
>
> 第一，全 TurboQuant 在 d=128 下不可用（+21.37% PPL），根本原因是维度放大（d=128 × 每坐标误差）和 QJL 惩罚（× π/2）的双重效应。
>
> 第二，K=Grouped_V=Grouped 精度最优（+1.44%），K=Prod_V=Grouped 是工程最优（+4.31% 但有 fused kernel 加速）。两者之间存在清晰的精度-速度权衡。
>
> 第三，也是最重要的发现：K 的精度优先级高于 V 的精度。这是因为 softmax 的非线性放大效应远强于加权平均的线性衰减效应。这一结论从 attention 机制的数学结构中直接推导出来，并得到了 PPL 实验的完美验证。
>
> 谢谢大家。

---

## Q&A 预期问题

### Q1: 为什么不在更大的模型上测试？
> 当前受限于 GPU 显存（8GB），Qwen3-8B-AWQ 是最大可运行的模型。核心结论（K 精度 > V 精度、维度效应）在 128 维 head_dim 上成立，需要更大模型验证泛化性。

### Q2: 2-bit 测试中 TurboQuantMSE 为什么优于 GroupedLinear？
> 极低位宽下 Lloyd-Max 优化码本（4 个非均匀质心）比均匀量化（4 个等间距级别）表达能力强。3-bit 及以上（8/16 级别），均匀量化的级别密度足够，分组策略的维度缩减优势才开始主导。

### Q3: 融合 kernel 为什么只支持 K=Prod + V=Grouped？
> `fused_asym_quantized_decode_attention` 的设计假设：K 使用 TurboQuant 码本+索引格式（QJL 可融合修正内积），V 使用 per-group affine 格式（scale/zero 可向量化）。其他组合需要在 kernel 内做旋转逆变换（TurboQuant V）或分组查表（GroupedLinear K），与 tile-level 的 tl.dot 操作无法高效融合。

### Q4: 实验结果是否可以推广到其他模型架构？
> 核心结论（K 精度 > V 精度）在所有使用标准 multi-head attention 的模型上成立。对于 GQA（Grouped Query Attention），KV 头数更少，每个 KV 头的量化精度影响更大，结论更加显著。对于 MLA（Multi-head Latent Attention），需要单独分析因为 K/V 的维度结构完全不同。
