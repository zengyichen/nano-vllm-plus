"""Memory usage benchmark.

Validates VRAM allocation by pushing batch size and sequence length
to the configured limits, then reporting peak KV cache utilization.
"""

import argparse
import os
from random import randint, seed

from nanovllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Benchmark nano-vllm memory usage.")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"), help="Path or name of the model")
    parser.add_argument("--max-batch", type=int, default=128, help="Maximum batch size to benchmark")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length per request")
    parser.add_argument("--max-model-len", type=int, default=1024, help="Max model sequence length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    args = parser.parse_args()

    print("=== Configuration ===")
    print(f"Model: {args.model}")
    print(f"Max Batch Size: {args.max_batch}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"Sequence Length (prompt): {args.seq_len}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print("=====================\n")

    print("Initializing LLM Engine... (Watch for [VRAM] tags below)")
    
    try:
        # 1. Provide settings that push the limits to see the memory footprint correctly allocated
        llm = LLM(
            model=args.model,
            max_num_seqs=args.max_batch,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_batch * args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    print("\nEngine Initialized Successfully! Testing dynamic generation usage...")

    # 2. Run an example batch
    seed(0)
    
    num_seqs = args.max_batch
    max_input_len = min(args.seq_len, args.max_model_len // 2)
    max_output_len = args.max_model_len - max_input_len
    
    print(f"Generating {num_seqs} random sequences (len ~100 to {max_input_len}) and decoding to ~100-{max_output_len} tokens")

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(max_tokens=randint(100, max_output_len), ignore_eos=True) for _ in range(num_seqs)]

    # Warmup specific trace
    llm.generate(["Warmup query"], SamplingParams(max_tokens=10), use_tqdm=False)

    print("\nStarting memory benchmark trace...")
    llm.generate(prompt_token_ids, sampling_params)

    # 3. Final summary on actual usage ratio
    stats = llm.scheduler.block_manager.get_usage_stats()
    print("\n=== Final Memory Status ===")
    print(f"KV Cache Total Blocks : {stats['total_blocks']}")
    print(f"KV Cache Peak Used    : {stats['peak_used_blocks']} ({stats['peak_usage_percent']:.1f}%) <--- Max usage hit during generation")
    print(f"KV Cache Currently Used: {stats['used_blocks']} ({stats['usage_percent']:.1f}%) <--- Expected 0% because tasks finished")
    print("===========================")

if __name__ == "__main__":
    main()
