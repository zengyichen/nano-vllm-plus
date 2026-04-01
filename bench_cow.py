import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams

def main():
    seed(0)
    # 设置测试参数
    num_seqs = 128            # 平行采样的分支数量（并发数）
    prompt_len = 1024         # 共享的 Prompt 长度
    max_output_len = 256      # 每个分支生成的最大 Token 数

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    print(f"Loading model from {path}...")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 1. 创建共享请求 (Shared Prompt)
    shared_prompt_ids = [randint(0, 10000) for _ in range(prompt_len)]
    
    # 2. 构造具有完全相同 prompt 的多个请求
    # 这在底层会极大地考验 Prefix Caching 和 CoW 机制：
    # 前 1024 个 token 如果能完美散列命中或引用计数复用，显存和 Prefill 时间将极低
    prompt_token_ids = [shared_prompt_ids for _ in range(num_seqs)]
    
    # 设置略带随机性的采样参数（保证即使共享前缀，后期的自回归输出也可能不同）
    sampling_params = [
        SamplingParams(temperature=0.8, ignore_eos=True, max_tokens=max_output_len) 
        for _ in range(num_seqs)
    ]

    print("Warming up...")
    llm.generate(["Benchmark: "], SamplingParams())

    # 4. Generate & Benchmark
    print("\n" + "="*50)
    print(f"🚀 运行 CoW & Prefix Caching Benchmark (Parallel Sampling)")
    print(f"🔗 共享 Prompt 长度: {prompt_len}")
    print(f"🌿 平行分支 (num_seqs): {num_seqs}")
    print(f"✍️ 生成步数 (max_tokens): {max_output_len}")
    print("="*50)

    t0 = time.time()
    # 核心推理阶段
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed = time.time() - t0
    
    # 计算评估指标
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_output_tokens / elapsed

    print("\n" + "="*50)
    print(f"⏱️  总耗时: {elapsed:.4f} 秒")
    print(f"📊 总生成 Output Token: {total_output_tokens}")
    print(f"⚡ 吞吐量 (Throughput): {throughput:.2f} tokens/s")
    print(f"🎯 若你的 CoW 优化成功，此处的吞吐量应当有显著提升！")
    print("="*50)


if __name__ == "__main__":
    main()
