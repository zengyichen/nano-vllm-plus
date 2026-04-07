import argparse
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def is_llama3_bnb_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "llama-3" in normalized and "bnb" in normalized and "4bit" in normalized


def main():
    parser = argparse.ArgumentParser(description="Nano-vLLM example runner")
    parser.add_argument(
        "model",
        nargs="?",
        default="~/huggingface/Qwen3-0.6B/",
        help="Model path or Hugging Face model ID to load",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Do not apply chat template to prompts",
    )
    args = parser.parse_args()

    model_id = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm_kwargs = {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
    }
    if is_llama3_bnb_model(model_id):
        llm_kwargs.update(
            {
                "max_num_seqs": 4,
                "max_model_len": 4096,
                "max_num_batched_tokens": 4096,
            }
        )

    llm = LLM(model_id, **llm_kwargs)

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    if not args.no_chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
