import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-8B-AWQ/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # Keep example defaults conservative for 8GB GPUs.
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        kv_quant_algo="turboquant",
        kv_quant_bits=4,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
    user_prompt = "Give me a short introduction to large language models."

    fallback_chat_template = '''{%- for message in messages %}
{%- if message.role == 'system' %}
{{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}
{%- elif message.role == 'user' %}
{{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
{%- elif message.role == 'assistant' %}
{{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\n\n</think>\n\n' }}
{%- endif %}
{%- endif %}'''
    chat_template = tokenizer.chat_template or fallback_chat_template
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        chat_template=chat_template,
    )

    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]
    token_ids = output["token_ids"]

    try:
        index = len(token_ids) - token_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(token_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(token_ids[index:], skip_special_tokens=True).strip("\n")
    print("\nUser prompt:", user_prompt)
    print("Formatted prompt:", prompt)
    print("Thinking content:", thinking_content)
    print("Content:", content)


if __name__ == "__main__":
    main()
