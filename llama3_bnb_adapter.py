import torch

from nanovllm.models.llama3_bnb_adapter import inject_nano_vllm_backend, load_llama3_bnb


def main(model_id: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"):
    print(f"[*] Loading base model: {model_id}")
    model = load_llama3_bnb(model_id=model_id, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[*] Base model loaded. CUDA memory: {allocated:.2f} GB")
    else:
        print("[*] Base model loaded on CPU.")

    inject_nano_vllm_backend(model)
    print(f"[*] Monkey patch complete. Injected {len(model.model.layers)} attention hooks.")


if __name__ == "__main__":
    main()
