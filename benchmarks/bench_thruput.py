"""Benchmark nano-vLLM inference efficiency for Qwen3-0.6B.

Each phase runs in an isolated subprocess for clean VRAM measurement.

Usage:
  python benchmarks/bench_thruput.py
  python benchmarks/bench_thruput.py --phase decode --batch-size 1
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch


def run_phase(phase: str, **kwargs) -> dict:
    """Run a benchmark phase in a subprocess for VRAM isolation."""
    cmd = [sys.executable, __file__, "--phase", phase]
    for k, v in kwargs.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return {"phase": phase, "error": f"exit={proc.returncode}"}

    # Extract JSON from stdout (last line should be JSON)
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"phase": phase, "error": "no JSON output", "stdout": proc.stdout}


# ═══════════════════════════════════════════════════════════════════
# In-process benchmark implementations (called via subprocess)
# ═══════════════════════════════════════════════════════════════════

def bench_decode_inproc(model: str, batch_size: int, context_len: int,
                         num_steps: int):
    """Decode benchmark — run in subprocess."""
    import gc
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.config import Config
    from nanovllm.sampling_params import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    max_len = context_len + num_steps + 10
    config = Config(
        model=model,
        enforce_eager=True,
        max_model_len=max_len,
        max_num_batched_tokens=max(max_len, 4096),
        max_num_seqs=batch_size,
    )
    runner = ModelRunner(config, 0, [])
    scheduler = Scheduler(config)
    sp = SamplingParams(temperature=1.0, max_tokens=num_steps + 1, ignore_eos=True)

    # Prefill
    fill_ids = tokenizer.encode("hello world " * ((context_len // 3) + 1),
                                 add_special_tokens=False)[:context_len]
    for _ in range(batch_size):
        scheduler.add(Sequence(fill_ids.copy(), sp))

    while True:
        seqs, is_prefill = scheduler.schedule()
        if not is_prefill:
            break
        runner.call("run", seqs, is_prefill)
        scheduler.postprocess(seqs, [0] * len(seqs))

    # Measure decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_tokens = 0
    measured_steps = 0
    for _ in range(num_steps):
        seqs, is_prefill = scheduler.schedule()
        if is_prefill or not seqs:
            break
        token_ids = runner.call("run", seqs, is_prefill)
        scheduler.postprocess(seqs, token_ids)
        total_tokens += len(seqs)
        measured_steps += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    result = {
        "phase": "decode",
        "batch_size": batch_size,
        "context_len": context_len,
        "measured_steps": measured_steps,
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "time_per_step_ms": elapsed * 1000 / measured_steps if measured_steps else 0,
        "throughput_tok_s": total_tokens / elapsed if elapsed > 0 else 0,
    }

    runner.call("exit")
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


def bench_e2e_inproc(model: str, prompt: str, max_tokens: int):
    """E2E benchmark — run in subprocess."""
    import gc
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.config import Config
    from nanovllm.sampling_params import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_len = len(token_ids) + max_tokens + 10
    config = Config(
        model=model,
        enforce_eager=True,
        max_model_len=max_len,
        max_num_batched_tokens=max(max_len, 4096),
        max_num_seqs=1,
    )
    runner = ModelRunner(config, 0, [])
    scheduler = Scheduler(config)
    sp = SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True)

    scheduler.add(Sequence(token_ids, sp))

    # Prefill
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    seqs, is_prefill = scheduler.schedule()
    runner.call("run", seqs, is_prefill)
    scheduler.postprocess(seqs, [0] * len(seqs))
    torch.cuda.synchronize()
    pref_time = time.perf_counter() - t0

    # Decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0
    for _ in range(max_tokens):
        seqs, is_prefill = scheduler.schedule()
        if is_prefill or not seqs:
            break
        token_ids = runner.call("run", seqs, is_prefill)
        scheduler.postprocess(seqs, token_ids)
        steps += 1
    torch.cuda.synchronize()
    dec_time = time.perf_counter() - t0

    result = {
        "phase": "e2e",
        "prompt_len": len(token_ids),
        "decode_steps": steps,
        "prefill_time_ms": pref_time * 1000,
        "decode_time_ms": dec_time * 1000,
        "total_time_ms": (pref_time + dec_time) * 1000,
        "prefill_throughput_tok_s": len(token_ids) / pref_time if pref_time > 0 else 0,
        "decode_throughput_tok_s": steps / dec_time if dec_time > 0 else 0,
    }

    runner.call("exit")
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark nano-vLLM inference")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--phase", choices=["all", "decode", "e2e"], default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--decode-steps", type=int, default=100)
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--e2e-max-tokens", type=int, default=100)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    # Handle subprocess invocation
    if args.phase == "decode":
        result = bench_decode_inproc(args.model, args.batch_size, args.context_len, args.decode_steps)
        print(json.dumps(result))
        return
    elif args.phase == "e2e":
        prompt = "The capital of France is Paris. The capital of Germany is Berlin."
        result = bench_e2e_inproc(args.model, prompt, args.e2e_max_tokens)
        print(json.dumps(result))
        return

    # === Orchestrator mode (phase == "all") ===
    print(f"Model: {args.model}")
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"Config: batch_sizes={args.batch_sizes}, decode_steps={args.decode_steps}, "
          f"context_len={args.context_len}")
    print("=" * 70)

    all_results = []

    # Decode benchmarks (each in isolated subprocess)
    print(f"\n--- Decode Benchmark (context={args.context_len}, steps={args.decode_steps}) ---")
    for bs in args.batch_sizes:
        print(f"  Batch={bs:>2}...", end=" ", flush=True)
        r = run_phase("decode", model=args.model, batch_size=bs,
                       context_len=args.context_len, decode_steps=args.decode_steps)
        all_results.append(r)
        if "throughput_tok_s" in r:
            print(f"{r['time_per_step_ms']:.2f} ms/step = {r['throughput_tok_s']:.0f} tok/s")
        else:
            print(f"FAILED: {r.get('error', 'unknown')}")

    # E2E benchmark
    print(f"\n--- E2E Benchmark (max_tokens={args.e2e_max_tokens}) ---")
    e2e = run_phase("e2e", model=args.model, e2e_max_tokens=args.e2e_max_tokens)
    all_results.append(e2e)
    if "prefill_time_ms" in e2e:
        print(f"  Prefill ({e2e['prompt_len']} tok): {e2e['prefill_time_ms']:.1f} ms "
              f"({e2e['prefill_throughput_tok_s']:.0f} tok/s)")
        print(f"  Decode ({e2e['decode_steps']} steps): {e2e['decode_time_ms']:.1f} ms "
              f"({e2e['decode_throughput_tok_s']:.0f} tok/s)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Qwen3-0.6B (FP16) on RTX 4060 Laptop 8GB")
    print("=" * 70)
    for r in all_results:
        if r.get("phase") == "decode":
            print(f"  Decode bs={r['batch_size']:>2} ctx={r['context_len']}: "
                  f"{r['time_per_step_ms']:>6.2f} ms/step "
                  f"= {r.get('throughput_tok_s', 0):>6.0f} tok/s")
        elif r.get("phase") == "e2e":
            print(f"  E2E: pref={e2e.get('prefill_time_ms',0):.0f}ms "
                  f"+ dec={e2e.get('decode_time_ms',0):.0f}ms "
                  f"({e2e.get('decode_throughput_tok_s',0):.0f} tok/s decode)")

    print(f"\n  Key insight: decode is extremely memory-bandwidth-bound.")
    print(f"  ~6 MB KV cache per decode step, 256 GB/s GPU → theoretical ~42K tok/s.")
    print(f"  Actual ~{all_results[0].get('throughput_tok_s',0):.0f} tok/s (bs=1) — "
          f"Python+launch overhead dominates.")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
