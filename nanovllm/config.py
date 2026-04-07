from dataclasses import dataclass
import torch
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def _is_llama3_bnb_model(self) -> bool:
        model_name = self.model.lower()
        return self.hf_config.model_type == "llama" and "llama-3" in model_name and "bnb" in model_name and "4bit" in model_name

    def __post_init__(self):
        assert self.kvcache_block_size > 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        torch_dtype = getattr(self.hf_config, "torch_dtype", torch.float16)
        if isinstance(torch_dtype, str):
            self.hf_config.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        elif torch_dtype is None:
            self.hf_config.torch_dtype = torch.float16
        self.hf_config.rope_scaling = None

        if self._is_llama3_bnb_model():
            self.max_num_seqs = min(self.max_num_seqs, 4)
            self.max_model_len = min(self.max_model_len, 4096)
            self.max_num_batched_tokens = min(self.max_num_batched_tokens, 4096)
            if self.tensor_parallel_size != 1:
                raise ValueError("Llama-3 BNB-4bit path currently supports tensor_parallel_size=1 only.")
            if self.kvcache_block_size % 256 != 0:
                # Current flash-attn paged cache kernels require block size divisible by 256.
                self.kvcache_block_size = 256

        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.kvcache_block_size % 256 == 0
