from __future__ import annotations

from typing import Any

import torch
from torch import nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from nanovllm.layers.attention import store_kvcache
from nanovllm.utils.context import get_context


def is_llama3_bnb_model(model_name: str, model_type: str | None = None) -> bool:
    normalized_name = model_name.lower()
    model_type_ok = model_type in (None, "llama")
    return model_type_ok and "llama-3" in normalized_name and "bnb" in normalized_name and "4bit" in normalized_name


def load_llama3_bnb(model_id: str, torch_dtype: torch.dtype = torch.float16):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    device_map: dict[str, int | str]
    if torch.cuda.is_available():
        device_map = {"": torch.cuda.current_device()}
    else:
        device_map = {"": "cpu"}
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )


def _flatten_heads(x: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = x.shape
    return x.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_heads, head_dim)


def custom_llama3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    del attention_mask, past_key_values, cache_position
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if position_embeddings is None:
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(hidden_states.size(1), device=hidden_states.device, dtype=torch.long).unsqueeze(0)
        position_embeddings = self.rotary_emb(value_states, position_ids=position_ids)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = _flatten_heads(query_states)
    key_states = _flatten_heads(key_states)
    value_states = _flatten_heads(value_states)

    context = get_context()
    if self.k_cache.numel() and self.v_cache.numel():
        store_kvcache(key_states, value_states, self.k_cache, self.v_cache, context.slot_mapping)

    if context.is_prefill:
        k, v = (self.k_cache, self.v_cache) if context.block_tables is not None else (key_states, value_states)
        attn_output = flash_attn_varlen_func(
            query_states,
            k,
            v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scaling,
            causal=True,
            block_table=context.block_tables,
        )
    else:
        attn_output = flash_attn_with_kvcache(
            query_states.unsqueeze(1),
            self.k_cache,
            self.v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scaling,
            causal=True,
        ).squeeze(1)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def inject_nano_vllm_backend(model):
    for layer in model.model.layers:
        attn_module = layer.self_attn
        attn_module.forward = custom_llama3_attention_forward.__get__(attn_module, attn_module.__class__)
        device = next(attn_module.parameters()).device
        attn_module.k_cache = torch.empty(0, device=device)
        attn_module.v_cache = torch.empty(0, device=device)
    return model


class Llama3BNBForCausalLM(nn.Module):

    def __init__(
        self,
        model_id: str,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        hf_model = load_llama3_bnb(model_id, torch_dtype=torch_dtype)
        hf_model = inject_nano_vllm_backend(hf_model)
        self.hf_model = hf_model
        self.model = hf_model.model
        self.lm_head = hf_model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            position_ids=positions.unsqueeze(0),
            use_cache=False,
        )
        return outputs.last_hidden_state.squeeze(0)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            last_indices = context.cu_seqlens_q[1:] - 1
            hidden_states = hidden_states[last_indices].contiguous()
        return self.lm_head(hidden_states)
