import argparse
import gc
import json
import os
import subprocess
import sys
import time
from random import randint, seed

import torch
import torch.distributed as dist

from nanovllm import LLM, SamplingParams


def _cleanup_runtime_state(llm=None):
    if llm is not None:
        llm.exit()
        del llm
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _parse_last_json_line(lines: list[str]):
    for i in range(len(lines) - 1, -1, -1):
        try:
            return json.loads(lines[i]), i
        except json.JSONDecodeError:
            continue
    return None, -1


def run_benchmark(llm_name, llm, num_seqs=4, max_input_len=128, max_output_len=128):
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


def run_oom_test(mode: str, model_path: str):
    target_lengths = list(range(5000, 16001, 1000))
    kwargs = dict(
        enforce_eager=True,
        max_num_seqs=1,
        # Keep headroom for prefill-time activation spikes and allocator fragmentation.
        gpu_memory_utilization=0.95,
        kv_allocator_safety_margin_mb=128,
        kv_activation_peak_reserve_mb=512,
    )
    name = "Baseline"
    if mode == "asym":
        kwargs.update(
            kv_quant_algo="asym_turboquant",
            kv_quant_bits=4,
            kv_decode_backend="asym_turboquant",
            kv_decode_workspace_mb=0,
            kv_v_bits=4,
            kv_v_group_size=32,
        )
        name = "AsymTurboQuant_4bit"

    results = []
    for max_input_len in target_lengths:
        kwargs["max_model_len"] = max_input_len
        kwargs["max_num_batched_tokens"] = max_input_len
        print(f"\n--- {name} OOM test: max_input_len={max_input_len} ---")

        llm = None
        try:
            llm = LLM(model_path, **kwargs)
            effective_max_tokens = int(llm.scheduler.max_num_batched_tokens)
            if max_input_len > effective_max_tokens:
                error_msg = (
                    f"Requested input length {max_input_len} exceeds effective KV capacity {effective_max_tokens}. "
                    "Allocator clamped max_model_len/max_num_batched_tokens for this run."
                )
                print(f"[ERROR] capacity failed for {max_input_len}: {error_msg}")
                results.append({
                    "max_input_len": max_input_len,
                    "success": False,
                    "phase": "capacity",
                    "error": error_msg,
                })
                break

            prompt_tokens = [0] * max_input_len
            llm.add_request(prompt_tokens, SamplingParams(temperature=1e-5, ignore_eos=True, max_tokens=0))
            llm.step()
            print(f"[OK] prefill succeeded for max_input_len={max_input_len}")
            results.append({
                "max_input_len": max_input_len,
                "success": True,
                "phase": "prefill",
                "error": None,
            })
        except Exception as e:
            phase = "init" if llm is None else "prefill"
            print(f"[ERROR] {phase} failed for {max_input_len}: {type(e).__name__}: {e}")
            results.append({
                "max_input_len": max_input_len,
                "success": False,
                "phase": phase,
                "error": repr(e),
            })
            break
        finally:
            _cleanup_runtime_state(llm)

    print(f"\n=== {name} OOM summary ===")
    for entry in results:
        status = "PASS" if entry["success"] else "FAIL"
        print(f"{entry['max_input_len']:>6}: {status} ({entry['phase']})")
    return {"mode": mode, "results": results}


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


def run_oom_mode_subprocess(mode: str, model_path: str):
    run_label = f"OOM/{mode}"
    print(f"\n[RUNNING] {run_label}")
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode",
        "oom",
        "--oom-submode",
        mode,
        "--model",
        model_path,
    ]
    env = os.environ.copy()
    alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{alloc_conf},expandable_segments:True" if alloc_conf else "expandable_segments:True"
        )

    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    merged = ""
    if proc.stdout:
        merged += proc.stdout
    if proc.stderr:
        if merged and not merged.endswith("\n"):
            merged += "\n"
        merged += proc.stderr

    lines = [line.strip() for line in merged.splitlines() if line.strip()]
    payload, payload_idx = _parse_last_json_line(lines)

    for i, line in enumerate(lines):
        if i == payload_idx:
            continue
        print(f"[{run_label}] {line}")

    if payload is None:
        error_msg = f"subprocess failed (code={proc.returncode}) with no JSON output"
        print(f"[{run_label}] [ERROR] {error_msg}")
        return {
            "mode": mode,
            "results": [{
                "max_input_len": None,
                "success": False,
                "phase": "subprocess",
                "error": error_msg,
            }],
        }

    if proc.returncode != 0:
        error_msg = f"subprocess exited with non-zero code={proc.returncode}"
        print(f"[{run_label}] [WARN] {error_msg}")
        if isinstance(payload, dict):
            payload["subprocess_error"] = error_msg
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "baseline", "quant", "asym", "oom"], default="all")
    parser.add_argument("--oom-submode", choices=["baseline", "asym"], default=None)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-8B-AWQ/"))
    args = parser.parse_args()

    if args.mode in {"baseline", "quant", "asym"}:
        throughput = run_single_mode(args.mode, args.model)
        print(json.dumps({"mode": args.mode, "throughput": throughput}))
        return

    if args.mode == "oom" and args.oom_submode is not None:
        oom_result = run_oom_test(args.oom_submode, args.model)
        print(json.dumps(oom_result))
        return

    if args.mode == "oom":
        oom_baseline = run_oom_mode_subprocess("baseline", args.model)
        oom_asym = run_oom_mode_subprocess("asym", args.model)
        print(json.dumps({"mode": "oom", "results": {"baseline": oom_baseline, "asym": oom_asym}}))
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
