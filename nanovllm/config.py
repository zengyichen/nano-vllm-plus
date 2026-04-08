import os
from dataclasses import dataclass
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
    kv_quant_algo: str = None
    kv_quant_bits: int = 8
    kv_decode_backend: str = "auto"
    kv_decode_workspace_mb: int = 64
    kv_decode_graph: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.kv_decode_backend = (self.kv_decode_backend or "auto").lower()
        assert self.kv_decode_backend in {"auto", "dequant_flash", "fused"}
        assert self.kv_decode_workspace_mb >= 0
        self.kv_decode_graph = bool(self.kv_decode_graph)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_config.rope_scaling = None
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
