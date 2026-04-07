import argparse
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def is_llama3_bnb_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "llama-3" in normalized and "bnb" in normalized and "4bit" in normalized


def main():
    parser = argparse.ArgumentParser(description="Nano-vLLM benchmark runner")
    parser.add_argument(
        "model",
        nargs="?",
        default="~/huggingface/Qwen3-0.6B/",
        help="Model path or Hugging Face model ID to benchmark",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=8,
        help="Number of sequences to benchmark",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=128,
        help="Maximum input prompt length",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=1024,
        help="Maximum generated output length",
    )
    args = parser.parse_args()

    seed(0)
    num_seqs = args.num_seqs
    max_input_len = args.max_input_len
    max_ouput_len = args.max_output_len

    model_id = os.path.expanduser(args.model)
    llm_kwargs = {
        "enforce_eager": False,
        "max_model_len": 4096,
    }
    if is_llama3_bnb_model(model_id):
        llm_kwargs.update(
            {
                "tensor_parallel_size": 1,
                "max_num_seqs": 4,
                "max_num_batched_tokens": 4096,
            }
        )

    llm = LLM(model_id, **llm_kwargs)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, max_ouput_len),
        )
        for _ in range(num_seqs)
    ]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
