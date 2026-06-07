"""Perplexity benchmark with standard dataset and decode-path evaluation.

Uses WikiText-2 test set. Prefills a prefix, then increments through remaining
tokens via the decode path — exercising the quantization-aware decode backend.

Usage:
  python benchmarks/bench_ppl.py                     # all modes, subprocess isolation
  python benchmarks/bench_ppl.py --mode noquant       # single mode in-process
  python benchmarks/bench_ppl.py --max-eval-tokens 512 --prefix-len 128  # faster eval
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context, get_context

MODE_NAME = {
    "noquant": "NoQuant",
    "k_prod_v_prod": "K=Prod_V=Prod",
    "k_prod_v_grouped": "K=Prod_V=Grouped",
    "k_grouped_v_prod": "K=Grouped_V=Prod",
    "k_grouped_v_grouped": "K=Grouped_V=Grouped",
}

ALL_MODES = ["noquant", "k_prod_v_prod", "k_prod_v_grouped", "k_grouped_v_prod", "k_grouped_v_grouped"]


def load_wikitext(path: str) -> str:
    """Load WikiText-2 raw text, joining paragraphs with newlines."""
    with open(path, "r") as f:
        return f.read()


def compute_ppl_decode(
    model_runner,
    block_manager,
    token_ids: list[int],
    prefix_len: int,
    max_eval_tokens: int,
    chunk_tokens: int = 256,
) -> tuple[float, float, int]:
    """Compute PPL by prefill + incremental decode.

    Prefill handles the prefix (positions [0:prefix_len]), producing logits
    that cover NLL for tokens [1:prefix_len]. The decode loop then appends
    ground-truth tokens one by one, each decode step producing a single set
    of logits that predicts the NEXT token.
    """
    total_tokens = len(token_ids)
    if total_tokens < prefix_len + 1:
        raise ValueError(
            f"Need at least {prefix_len + 1} tokens, got {total_tokens}"
        )

    seq = Sequence(
        token_ids[:prefix_len],
        SamplingParams(temperature=1e-5, ignore_eos=True, max_tokens=1),
    )
    block_manager.allocate(seq)

    total_nll = 0.0
    total_targets = 0

    # Phase 1 — Prefill: logits at positions [0:prefix_len-1] predict tokens [1:prefix_len]
    input_ids, positions = model_runner.prepare_prefill([seq])
    get_context().return_all_logits = True
    logits = model_runner.run_model(input_ids, positions, is_prefill=True)

    if logits is None or int(logits.size(0)) < prefix_len:
        raise RuntimeError(
            f"Expected prefill logits shape [{prefix_len}, V], got {tuple(logits.shape)}"
        )

    log_probs = F.log_softmax(logits.float(), dim=-1)
    for j in range(prefix_len - 1):
        target = int(token_ids[j + 1])
        total_nll -= float(log_probs[j, target].item())
    total_targets += prefix_len - 1

    reset_context()

    # Phase 2 — Decode: each step produces logits for the NEXT token
    num_decode_steps = min(max_eval_tokens - total_targets, total_tokens - prefix_len - 1)
    if num_decode_steps < 0:
        num_decode_steps = 0

    for step in range(num_decode_steps):
        i = prefix_len + step  # index of the NEXT target token

        # Prepare for appending after model runs (may allocate/finalize blocks)
        while not block_manager.can_append(seq):
            raise RuntimeError("No free KV cache blocks for decode step")
        block_manager.may_append(seq)

        input_ids, positions = model_runner.prepare_decode([seq])
        logits = model_runner.run_model(input_ids, positions, is_prefill=False)

        if logits is None or int(logits.size(0)) != 1:
            raise RuntimeError(
                f"Expected decode logits shape [1, V], got {tuple(logits.shape)}"
            )

        log_probs = F.log_softmax(logits.float(), dim=-1)
        target = int(token_ids[i])
        total_nll -= float(log_probs[0, target].item())
        total_targets += 1

        # Advance sequence with ground-truth token
        seq.append_token(token_ids[i])

        reset_context()

        if chunk_tokens > 0 and ((step + 1) % chunk_tokens == 0 or step == num_decode_steps - 1):
            print(f"[INFO] decoded {step + 1}/{num_decode_steps} steps ({total_targets} NLL targets)")

    block_manager.deallocate(seq)
    reset_context()

    if total_targets <= 0:
        raise RuntimeError("No tokens were evaluated for perplexity")

    mean_nll = float(total_nll) / total_targets
    ppl = math.exp(mean_nll)
    return ppl, mean_nll, total_targets


def run_ppl_mode(
    mode: str,
    model_path: str,
    text: str,
    prefix_len: int,
    max_eval_tokens: int,
    chunk_tokens: int,
):
    name = MODE_NAME[mode]
    result = {
        "mode": mode,
        "name": name,
        "prefix_len": int(prefix_len),
        "max_eval_tokens": int(max_eval_tokens),
        "chunk_tokens": int(chunk_tokens),
        "success": False,
        "error": None,
    }

    llm = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        kwargs = _mode_kwargs(mode, prefix_len + max_eval_tokens)
        print(f"\n--- {name} perplexity benchmark ---")
        print(
            f"[INFO] total_tokens={len(token_ids)}, prefix_len={prefix_len}, "
            f"max_eval_tokens={max_eval_tokens}"
        )

        llm = LLM(model_path, **kwargs)

        # Re-tokenize to ensure consistency (same tokenizer instance)
        token_ids = llm.tokenizer.encode(text, add_special_tokens=False)

        ppl, mean_nll, evaluated_tokens = compute_ppl_decode(
            llm.model_runner,
            llm.scheduler.block_manager,
            token_ids,
            prefix_len,
            max_eval_tokens,
            chunk_tokens,
        )

        result.update(
            {
                "success": True,
                "total_tokens": int(len(token_ids)),
                "evaluated_tokens": int(evaluated_tokens),
                "mean_nll": float(mean_nll),
                "perplexity": float(ppl),
            }
        )
    except Exception as e:
        result["error"] = repr(e)
    finally:
        _cleanup_runtime_state(llm)

    return result


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


def _mode_kwargs(mode: str, total_context: int) -> dict:
    kwargs = {
        "enforce_eager": True,
        "max_num_seqs": 1,
        "max_model_len": max(64, int(total_context) + 1),
        "max_num_batched_tokens": max(64, int(total_context) + 1),
    }
    if mode == "k_prod_v_prod":
        kwargs.update(
            k_quant_algo="turboquant_prod",
            v_quant_algo="turboquant_prod",
            kv_quant_bits=4,
        )
    elif mode == "k_prod_v_grouped":
        kwargs.update(
            k_quant_algo="turboquant_prod",
            v_quant_algo="grouped_linear",
            kv_quant_bits=4,
            kv_v_bits=4,
            kv_v_group_size=32,
        )
    elif mode == "k_grouped_v_prod":
        kwargs.update(
            k_quant_algo="grouped_linear",
            v_quant_algo="turboquant_prod",
            kv_quant_bits=4,
            kv_v_bits=4,
            kv_v_group_size=32,
        )
    elif mode == "k_grouped_v_grouped":
        kwargs.update(
            k_quant_algo="grouped_linear",
            v_quant_algo="grouped_linear",
            kv_quant_bits=4,
            kv_v_bits=4,
            kv_v_group_size=32,
        )
    return kwargs


def run_mode_subprocess(
    mode: str,
    model_path: str,
    data_file: str,
    prefix_len: int,
    max_eval_tokens: int,
    chunk_tokens: int,
):
    run_label = f"PPL/{mode}"
    print(f"\n[RUNNING] {run_label}")

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode", mode,
        "--model", model_path,
        "--data-file", data_file,
        "--prefix-len", str(prefix_len),
        "--max-eval-tokens", str(max_eval_tokens),
        "--chunk-tokens", str(chunk_tokens),
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
            "prefix_len": int(prefix_len),
            "max_eval_tokens": int(max_eval_tokens),
            "success": False,
            "error": f"subprocess failed (code={proc.returncode}) with no JSON output",
        }

    if proc.returncode != 0 and isinstance(payload, dict):
        payload["subprocess_error"] = f"subprocess exited with non-zero code={proc.returncode}"
    return payload


def add_relative_metrics(results: dict[str, dict]):
    base = results.get("noquant")
    if not base or not base.get("success"):
        return

    base_ppl = float(base.get("perplexity", 0.0))
    if base_ppl <= 0:
        return

    for mode, entry in results.items():
        if not entry.get("success"):
            continue
        ppl = float(entry.get("perplexity", 0.0))
        if ppl <= 0:
            continue
        entry["ppl_ratio_vs_noquant"] = float(ppl / base_ppl)
        entry["ppl_delta_pct_vs_noquant"] = float((ppl / base_ppl - 1.0) * 100.0)


def print_summary(results: dict[str, dict]):
    print("\n=== Perplexity Summary ===")
    header = f"{'mode':<22} {'success':<7} {'ppl':<11} {'mean_nll':<9} {'tokens':<7} {'ppl_ratio':<10} {'delta_pct':<9}"
    print(header)
    print("-" * len(header))
    for mode in ALL_MODES:
        row = results.get(mode, {})
        success = str(bool(row.get("success", False)))
        ppl = row.get("perplexity")
        nll = row.get("mean_nll")
        et = row.get("evaluated_tokens")
        ratio = row.get("ppl_ratio_vs_noquant")
        delta = row.get("ppl_delta_pct_vs_noquant")

        name = MODE_NAME.get(mode, mode)
        ppl_text = f"{ppl:.6f}" if isinstance(ppl, (int, float)) else "-"
        nll_text = f"{nll:.6f}" if isinstance(nll, (int, float)) else "-"
        et_text = f"{int(et)}" if isinstance(et, (int, float)) else "-"
        ratio_text = f"{ratio:.6f}" if isinstance(ratio, (int, float)) else "-"
        delta_text = f"{delta:+.2f}%" if isinstance(delta, (int, float)) else "-"

        print(f"{name:<22} {success:<7} {ppl_text:<11} {nll_text:<9} {et_text:<7} {ratio_text:<10} {delta_text}")


def main():
    parser = argparse.ArgumentParser(
        description="Perplexity benchmark with WikiText-2 and decode-path evaluation."
    )
    parser.add_argument("--mode", choices=["all"] + ALL_MODES, default="all")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-8B-AWQ/"))
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--max-eval-tokens", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    args = parser.parse_args()

    data_file = args.data_file or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "wikitext-2-test.txt"
    )

    if args.mode in ALL_MODES:
        text = load_wikitext(data_file)
        result = run_ppl_mode(
            args.mode, args.model, text,
            args.prefix_len, args.max_eval_tokens, args.chunk_tokens,
        )
        print(json.dumps(result))
        return

    results = {}
    for mode in ALL_MODES:
        results[mode] = run_mode_subprocess(
            mode, args.model, data_file,
            args.prefix_len, args.max_eval_tokens, args.chunk_tokens,
        )

    add_relative_metrics(results)
    print_summary(results)
    print(
        json.dumps(
            {
                "mode": "all",
                "model": args.model,
                "data_file": data_file,
                "prefix_len": int(args.prefix_len),
                "max_eval_tokens": int(args.max_eval_tokens),
                "results": results,
            }
        )
    )


if __name__ == "__main__":
    main()
