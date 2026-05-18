"""Decode throughput benchmark (single mode, no quantization).

Measures pure decode throughput with random token sequences.
Simplest benchmark in the suite — good for quick sanity checks.
"""

import argparse
import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Simple decode throughput benchmark")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    args = parser.parse_args()

    seed(0)
    path = os.path.expanduser(args.model)
    llm = LLM(path, enforce_eager=args.enforce_eager, max_model_len=args.max_model_len)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len))
        for _ in range(args.num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens} tok, Time: {t:.2f}s, Throughput: {throughput:.2f} tok/s")


if __name__ == "__main__":
    main()
