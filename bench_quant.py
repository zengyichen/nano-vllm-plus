import argparse
import json
import os
import subprocess
import sys
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def run_benchmark(llm_name, llm, num_seqs=1, max_input_len=128, max_output_len=1024):
    print(f"\n--- Running benchmark: {llm_name} ---")
    seed(0)
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(8, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(8, max_output_len))
        for _ in range(num_seqs)
    ]

    llm.generate(["Warmup max tokens: "], [SamplingParams(max_tokens=2)])
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"[{llm_name}] Total: {total_tokens} tokens, Time: {t:.2f}s, Throughput: {throughput:.2f} tok/s\n")
    return throughput


def run_single_mode(mode: str, model_path: str) -> float:
    kwargs = dict(
        enforce_eager=True,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
    )
    name = "Baseline"
    if mode == "quant":
        kwargs.update(kv_quant_algo="turboquant", kv_quant_bits=4)
        name = "TurboQuant_4bit_prod"
    elif mode == "asym":
        kwargs.update(
            kv_quant_algo="asym_turboquant",
            kv_quant_bits=4,
            kv_decode_backend="asym_turboquant",
            kv_v_bits=4,
            kv_v_group_size=32,
        )
        name = "AsymTurboQuant_4bit"

    llm = None
    try:
        llm = LLM(model_path, **kwargs)
        throughput = run_benchmark(name, llm)
    finally:
        if llm is not None:
            llm.exit()
    return throughput


def run_mode_subprocess(mode: str, model_path: str) -> float:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode",
        mode,
        "--model",
        model_path,
    ]
    out = subprocess.check_output(cmd, text=True)
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    result = json.loads(lines[-1])
    return float(result["throughput"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "baseline", "quant", "asym"], default="all")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-8B-AWQ/"))
    args = parser.parse_args()

    if args.mode in {"baseline", "quant", "asym"}:
        throughput = run_single_mode(args.mode, args.model)
        print(json.dumps({"mode": args.mode, "throughput": throughput}))
        return

    print("========================================")
    print("Baseline (No Quantization)")
    print("========================================")
    throughput_baseline = run_mode_subprocess("baseline", args.model)

    print("========================================")
    print("TurboQuant Prod 4-bit Quantization")
    print("========================================")
    throughput_quant = run_mode_subprocess("quant", args.model)

    print("========================================")
    print("Asym TurboQuant 4-bit Quantization")
    print("========================================")
    throughput_asym = run_mode_subprocess("asym", args.model)

    print("========================================")
    print("Benchmark Comparison Summary")
    print("========================================")
    print(f"Baseline Throughput : {throughput_baseline:.2f} tok/s")
    print(f"TurboQuant 4-bit    : {throughput_quant:.2f} tok/s")
    print(f"AsymTurboQuant 4-bit: {throughput_asym:.2f} tok/s")

    if throughput_baseline > 0 and throughput_quant > 0:
        ratio = throughput_quant / throughput_baseline
        diff = (ratio - 1.0) * 100.0
        print(f"Throughput Impact   : {diff:+.2f}%")
    if throughput_baseline > 0 and throughput_asym > 0:
        ratio = throughput_asym / throughput_baseline
        diff = (ratio - 1.0) * 100.0
        print(f"Asym Impact         : {diff:+.2f}%")


if __name__ == "__main__":
    main()
