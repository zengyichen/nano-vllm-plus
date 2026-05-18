import argparse
import gc
import json
import os
import subprocess
import sys

import torch
import torch.distributed as dist

from nanovllm import LLM, SamplingParams


MODE_NAME = {
    "noquant": "NoQuant",
    "kvquant": "KVQuant_4bit",
    "asym": "AsymTurboQuant_4bit",
}


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


def _tensor_nbytes(tensor) -> int:
    if not isinstance(tensor, torch.Tensor):
        return 0
    return int(tensor.numel() * tensor.element_size())


def _collect_kv_cache_bytes(model_runner) -> tuple[int, dict[str, int]]:
    cache_fields = [
        "kv_cache",
        "kv_scales",
        "kv_k_cache",
        "kv_k_scales",
        "kv_v_cache",
        "kv_v_scales",
        "kv_v_zeros",
    ]
    details = {}
    total = 0
    for field in cache_fields:
        nbytes = _tensor_nbytes(getattr(model_runner, field, None))
        if nbytes > 0:
            details[field] = nbytes
            total += nbytes
    return total, details


def run_kv_benchmark_mode(mode: str, model_path: str, context_len: int):
    kwargs = {
        "enforce_eager": True,
        "max_num_seqs": 1,
        "max_model_len": int(context_len),
        "max_num_batched_tokens": int(context_len),
    }
    if mode == "kvquant":
        kwargs.update(kv_quant_algo="turboquant", kv_quant_bits=4)
    elif mode == "asym":
        kwargs.update(
            kv_quant_algo="asym_turboquant",
            kv_quant_bits=4,
            kv_decode_backend="asym_turboquant",
            kv_v_bits=4,
            kv_v_group_size=32,
        )

    name = MODE_NAME[mode]
    llm = None
    result = {
        "mode": mode,
        "name": name,
        "context_len": int(context_len),
        "success": False,
        "error": None,
    }

    try:
        print(f"\n--- {name} KV benchmark: context_len={context_len} ---")
        llm = LLM(model_path, **kwargs)

        effective_len = int(llm.scheduler.max_num_batched_tokens)
        result["effective_max_num_batched_tokens"] = effective_len
        if effective_len < int(context_len):
            raise RuntimeError(
                f"requested context_len={context_len} exceeds effective allocator cap={effective_len}"
            )

        prompt_token_ids = [[0] * int(context_len)]
        sampling_params = [SamplingParams(temperature=1e-5, ignore_eos=True, max_tokens=1)]
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

        usage_stats = llm.scheduler.block_manager.get_usage_stats()
        total_blocks = int(usage_stats["total_blocks"])
        peak_blocks = int(usage_stats["peak_used_blocks"])
        block_size = int(llm.model_runner.block_size)

        kv_total_bytes, kv_components = _collect_kv_cache_bytes(llm.model_runner)
        kv_bytes_per_block = (kv_total_bytes / total_blocks) if total_blocks > 0 else 0.0
        kv_peak_used_bytes = kv_bytes_per_block * peak_blocks

        # With max_tokens=1, generation completes in prefill and appends one completion token.
        measured_tokens = int(context_len) + 1
        kv_peak_used_bytes_per_token = (
            kv_peak_used_bytes / measured_tokens if measured_tokens > 0 else 0.0
        )

        result.update(
            {
                "success": True,
                "kv_cache_total_bytes": int(kv_total_bytes),
                "kv_cache_total_gb": float(kv_total_bytes / 1024**3),
                "kv_cache_component_bytes": kv_components,
                "kv_total_blocks": total_blocks,
                "kv_peak_used_blocks": peak_blocks,
                "kv_block_size": block_size,
                "kv_bytes_per_block": float(kv_bytes_per_block),
                "kv_peak_used_bytes": float(kv_peak_used_bytes),
                "kv_peak_used_gb": float(kv_peak_used_bytes / 1024**3),
                "measured_tokens": measured_tokens,
                "kv_peak_used_bytes_per_token": float(kv_peak_used_bytes_per_token),
            }
        )
    except Exception as e:
        result["error"] = repr(e)
    finally:
        _cleanup_runtime_state(llm)

    return result


def add_compression_summary(results: dict[str, dict]):
    base = results.get("noquant")
    if not base or not base.get("success"):
        return
    base_bytes = float(base.get("kv_peak_used_bytes", 0.0))
    if base_bytes <= 0:
        return

    for mode, entry in results.items():
        if not entry.get("success"):
            continue
        mode_bytes = float(entry.get("kv_peak_used_bytes", 0.0))
        if mode_bytes <= 0:
            continue
        ratio = mode_bytes / base_bytes
        entry["compression_rate_vs_noquant"] = float(base_bytes / mode_bytes)
        entry["size_ratio_vs_noquant"] = float(ratio)
        entry["saving_pct_vs_noquant"] = float((1.0 - ratio) * 100.0)


def run_mode_subprocess(mode: str, model_path: str, context_len: int):
    run_label = f"KV/{mode}"
    print(f"\n[RUNNING] {run_label}")
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode",
        mode,
        "--model",
        model_path,
        "--context-len",
        str(context_len),
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
        return {
            "mode": mode,
            "name": MODE_NAME[mode],
            "context_len": int(context_len),
            "success": False,
            "error": f"subprocess failed (code={proc.returncode}) with no JSON output",
        }

    if proc.returncode != 0 and isinstance(payload, dict):
        payload["subprocess_error"] = f"subprocess exited with non-zero code={proc.returncode}"
    return payload


def print_summary_table(results: dict[str, dict]):
    print("\n=== KV Compression Summary ===")
    print(
        "mode         success  peak_used_GB  bytes/token  compression_vs_noquant  saving_vs_noquant(%)"
    )
    for mode in ["noquant", "kvquant", "asym"]:
        entry = results.get(mode, {})
        success = str(bool(entry.get("success", False)))
        peak_gb = entry.get("kv_peak_used_gb")
        bpt = entry.get("kv_peak_used_bytes_per_token")
        comp = entry.get("compression_rate_vs_noquant")
        saving = entry.get("saving_pct_vs_noquant")

        peak_text = f"{peak_gb:.4f}" if isinstance(peak_gb, (int, float)) else "-"
        bpt_text = f"{bpt:.2f}" if isinstance(bpt, (int, float)) else "-"
        comp_text = f"{comp:.4f}" if isinstance(comp, (int, float)) else "-"
        saving_text = f"{saving:.2f}" if isinstance(saving, (int, float)) else "-"

        print(
            f"{mode:<12} {success:<7} {peak_text:<13} {bpt_text:<12} {comp_text:<23} {saving_text:<21}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache compression at fixed context length.")
    parser.add_argument("--mode", choices=["all", "noquant", "kvquant", "asym"], default="all")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-8B-AWQ/"))
    parser.add_argument("--context-len", type=int, default=2048)
    args = parser.parse_args()

    if args.mode in {"noquant", "kvquant", "asym"}:
        result = run_kv_benchmark_mode(args.mode, args.model, args.context_len)
        print(json.dumps(result))
        return

    results = {}
    for mode in ["noquant", "kvquant", "asym"]:
        results[mode] = run_mode_subprocess(mode, args.model, args.context_len)

    add_compression_summary(results)
    print_summary_table(results)
    print(
        json.dumps(
            {
                "mode": "all",
                "model": args.model,
                "context_len": int(args.context_len),
                "results": results,
            }
        )
    )


if __name__ == "__main__":
    main()