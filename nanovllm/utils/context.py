from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    quant_slot_mapping: torch.Tensor | None = None
    quant_cu_seqlens_k: torch.Tensor | None = None
    quant_max_seqlen_k: int = 0
    quant_q_rot: torch.Tensor | None = None
    quant_q_sketch: torch.Tensor | None = None
    quant_v_group_size: int = 0
    return_all_logits: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    quant_slot_mapping=None,
    quant_cu_seqlens_k=None,
    quant_max_seqlen_k=0,
    quant_q_rot=None,
    quant_q_sketch=None,
    quant_v_group_size=0,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        quant_slot_mapping=quant_slot_mapping,
        quant_cu_seqlens_k=quant_cu_seqlens_k,
        quant_max_seqlen_k=quant_max_seqlen_k,
        quant_q_rot=quant_q_rot,
        quant_q_sketch=quant_q_sketch,
        quant_v_group_size=quant_v_group_size,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
