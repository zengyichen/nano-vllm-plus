import math
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.quant import get_kv_quantizer


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.quantizer = get_kv_quantizer(config)
        self.quant_decode_backend = config.kv_decode_backend
        self.quant_decode_graph = bool(config.kv_decode_graph)
        self.decode_workspace_tokens = 0
        self.decode_k_workspace = None
        self.decode_v_workspace = None
        self.quant_graph_max_bs = 0
        self.graph_quant_tokens = {}
        self.split_kv_cache = False
        self.kv_k_cache = None
        self.kv_k_scales = None
        self.kv_v_cache = None
        self.kv_v_scales = None
        self.kv_k_zeros = None
        self.kv_v_zeros = None
        self.warmup_activation_bytes = 0
        self.warmup_peak_bytes = 0
        self.warmup_current_bytes = 0
        if self.quantizer is not None and self.quant_decode_backend == "auto":
            k_m = getattr(self.quantizer, "k_method", "")
            v_m = getattr(self.quantizer, "v_method", "")
            if k_m == "turboquant_prod" and v_m == "grouped_linear":
                self.quant_decode_backend = "asym_turboquant"
            else:
                self.quant_decode_backend = "dequant_flash"
        if self.quantizer is not None and not self.enforce_eager:
            # Keep eager by default; quant graph replay is opt-in per backend.
            graph_ready_backend = self.quant_decode_backend in {"dequant_flash", "asym_turboquant"}
            if not (self.quant_decode_graph and graph_ready_backend):
                self.enforce_eager = True

        device = "cuda" if torch.cuda.is_available() else "cpu"
        backend = "nccl" if device == "cuda" else "gloo"
        dist.init_process_group(backend, "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        if device == "cuda":
            torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(device)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)

        # Adaptive warmup: when VRAM is tight (model >70% total), reduce warmup
        # batch size to avoid OOM during the activation peak measurement pass.
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_bytes = total_bytes - free_bytes
        if used_bytes > int(total_bytes * 0.70):
            num_seqs = 1

        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]

        # First pass triggers one-off compile/autotune paths.
        torch.cuda.reset_peak_memory_stats()
        self.run(seqs, True)
        torch.cuda.empty_cache()

        # Second pass approximates steady-state activation peak for allocator reserve.
        torch.cuda.reset_peak_memory_stats()
        self.run(seqs, True)
        stats = torch.cuda.memory_stats()
        self.warmup_peak_bytes = int(stats["allocated_bytes.all.peak"])
        self.warmup_current_bytes = int(stats["allocated_bytes.all.current"])
        self.warmup_activation_bytes = max(0, self.warmup_peak_bytes - self.warmup_current_bytes)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        model_allocated = int(torch.cuda.memory_stats()["allocated_bytes.all.current"])
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        dtype_size = torch.empty((), dtype=hf_config.torch_dtype).element_size()
        split_quant_layout = self.quantizer is not None and hasattr(self.quantizer, "allocate_cache_split")

        # Handle block size calculation with quantization
        if self.quantizer:
            persistent_per_token_head, transient_per_token_head = self.quantizer.bytes_per_token_head(head_dim, dtype_size)
            layout_factor = 1 if split_quant_layout else 2
            block_persistent_bytes = (
                layout_factor
                * hf_config.num_hidden_layers
                * self.block_size
                * num_kv_heads
                * persistent_per_token_head
            )
            # attention.py currently dequantizes cache tensors into fp tensors before attention,
            # so reserve transient fp workspace proportional to cache footprint.
            block_transient_bytes = (
                layout_factor
                * hf_config.num_hidden_layers
                * self.block_size
                * num_kv_heads
                * transient_per_token_head
            )
            block_bytes = block_persistent_bytes + block_transient_bytes
        else:
            block_persistent_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * dtype_size
            block_transient_bytes = 0
            block_bytes = block_persistent_bytes

        workspace_reserve_bytes = 0
        if self.quantizer and self.quant_decode_backend in {"dequant_flash", "asym_turboquant"}:
            workspace_bytes = int(config.kv_decode_workspace_mb) * 1024 * 1024
            per_token_bytes = 2 * int(num_kv_heads) * int(head_dim) * int(dtype_size)
            max_tokens = int(config.max_num_seqs) * int(config.max_model_len)
            if workspace_bytes > 0:
                workspace_reserve_bytes = min(workspace_bytes, max_tokens * max(1, per_token_bytes))

        safety_margin_bytes = int(config.kv_allocator_safety_margin_mb) * 1024 * 1024
        activation_cap_bytes = int(config.kv_activation_peak_reserve_mb) * 1024 * 1024
        activation_reserve_bytes = min(int(self.warmup_activation_bytes), activation_cap_bytes)

        available_bytes = int(total * config.gpu_memory_utilization)
        available_bytes -= int(used)
        available_bytes -= safety_margin_bytes
        available_bytes -= activation_reserve_bytes
        available_bytes -= workspace_reserve_bytes
        config.num_kvcache_blocks = available_bytes // int(block_bytes)
        assert config.num_kvcache_blocks > 0, (
            f"No VRAM available for KV Cache: available={available_bytes / 1024**2:.1f}MB, "
            f"block={block_bytes / 1024**2:.2f}MB"
        )

        # When KV cache blocks are scarce, reduce max_num_seqs and max_num_batched_tokens
        # so that max_model_len is not starved. Aim for at least 1024-token contexts.
        min_desired_model_len = min(1024, int(config.max_model_len))
        min_blocks_per_seq = math.ceil(min_desired_model_len / int(self.block_size))
        max_seqs_by_blocks = max(1, int(config.num_kvcache_blocks) // min_blocks_per_seq)
        if int(config.max_num_seqs) > max_seqs_by_blocks:
            if self.rank == 0:
                print(
                    f"[Warning] Requested max_num_seqs={config.max_num_seqs} too large for "
                    f"{config.num_kvcache_blocks} KV blocks. Reducing to {max_seqs_by_blocks}."
                )
            config.max_num_seqs = max_seqs_by_blocks
            mint = int(config.max_num_seqs) * int(config.max_model_len)
            if int(config.max_num_batched_tokens) > mint:
                config.max_num_batched_tokens = mint

        # Keep runtime scheduling self-consistent: requested max_model_len must fit KV blocks.
        effective_num_seqs = max(
            1,
            min(
                int(config.max_num_seqs),
                int(config.max_num_batched_tokens) // max(1, int(config.max_model_len)),
            ),
        )
        min_blocks_required = effective_num_seqs * math.ceil(int(config.max_model_len) / int(self.block_size))
        if config.num_kvcache_blocks < min_blocks_required:
            max_len_by_blocks = (config.num_kvcache_blocks // effective_num_seqs) * int(self.block_size)
            assert max_len_by_blocks > 0, "No KV blocks left after VRAM reservation"
            if self.rank == 0:
                print(
                    f"[Warning] Requested max_model_len={config.max_model_len} exceeds KV capacity. "
                    f"Clamping to {max_len_by_blocks}."
                )
            config.max_model_len = int(max_len_by_blocks)

        max_batched_tokens_by_blocks = (config.num_kvcache_blocks // effective_num_seqs) * int(self.block_size) * effective_num_seqs
        if int(config.max_num_batched_tokens) > int(max_batched_tokens_by_blocks):
            if self.rank == 0:
                print(
                    f"[Warning] Requested max_num_batched_tokens={config.max_num_batched_tokens} exceeds KV capacity. "
                    f"Clamping to {max_batched_tokens_by_blocks}."
                )
            config.max_num_batched_tokens = int(max_batched_tokens_by_blocks)
        
        # Allocate KV Cache
        if self.quantizer and hasattr(self.quantizer, "allocate_cache_split"):
            self.split_kv_cache = True
            (
                self.kv_k_cache,
                self.kv_k_scales,
                self.kv_k_zeros,
                self.kv_v_cache,
                self.kv_v_scales,
                self.kv_v_zeros,
            ) = self.quantizer.allocate_cache_split(
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                hf_config.torch_dtype,
                "cuda",
            )
            self.kv_cache = None
            self.kv_scales = None
        elif self.quantizer:
            self.kv_cache, self.kv_scales = self.quantizer.allocate_cache(
                hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, hf_config.torch_dtype, "cuda"
            )
        else:
            self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
            self.kv_scales = None

        if self.quantizer and self.quant_decode_backend in {"dequant_flash", "asym_turboquant"}:
            workspace_bytes = int(config.kv_decode_workspace_mb) * 1024 * 1024
            per_token_bytes = 2 * int(num_kv_heads) * int(head_dim) * int(dtype_size)
            max_tokens = int(config.max_num_seqs) * int(config.max_model_len)
            self.decode_workspace_tokens = min(max_tokens, workspace_bytes // max(1, per_token_bytes)) if workspace_bytes > 0 else 0
            if self.decode_workspace_tokens > 0:
                self.decode_k_workspace = torch.empty(
                    self.decode_workspace_tokens,
                    int(num_kv_heads),
                    int(head_dim),
                    dtype=hf_config.torch_dtype,
                    device="cuda",
                )
                self.decode_v_workspace = torch.empty_like(self.decode_k_workspace)
            else:
                self.decode_k_workspace = None
                self.decode_v_workspace = None
            if self.quant_decode_graph:
                if self.quant_decode_backend == "dequant_flash" and self.decode_workspace_tokens > 0:
                    self.quant_graph_max_bs = max(
                        1,
                        min(int(config.max_num_seqs), self.decode_workspace_tokens // max(1, int(config.max_model_len))),
                    )
                elif self.quant_decode_backend == "asym_turboquant":
                    self.quant_graph_max_bs = int(config.max_num_seqs)
                else:
                    self.quant_graph_max_bs = 0
            else:
                self.quant_graph_max_bs = 0
        else:
            self.decode_k_workspace = None
            self.decode_v_workspace = None
            self.quant_graph_max_bs = 0
            
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                if self.split_kv_cache:
                    module.k_cache = self.kv_k_cache[layer_id]
                    module.v_cache = self.kv_v_cache[layer_id]
                else:
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                module.quantizer = self.quantizer
                module.quant_decode_backend = self.quant_decode_backend
                module.decode_k_workspace = self.decode_k_workspace
                module.decode_v_workspace = self.decode_v_workspace
                module.decode_workspace_tokens = self.decode_workspace_tokens
                if self.quantizer:
                    if self.split_kv_cache:
                        module.k_scales = self.kv_k_scales[layer_id]
                        module.v_scales = self.kv_v_scales[layer_id]
                        module.k_zeros = self.kv_k_zeros[layer_id] if self.kv_k_zeros is not None else None
                        module.v_zeros = self.kv_v_zeros[layer_id] if self.kv_v_zeros is not None else None
                    else:
                        module.k_scales = self.kv_scales[0, layer_id]
                        module.v_scales = self.kv_scales[1, layer_id]
                        module.v_zeros = None
                layer_id += 1

        if self.rank == 0:
            print(f"[VRAM] Model Weights: {model_allocated / 1024**3:.2f} GB")
            print(f"[VRAM] Activations (Warmup Peak): {self.warmup_activation_bytes / 1024**3:.2f} GB")
            print(f"[VRAM] Safety Margin: {safety_margin_bytes / 1024**3:.2f} GB")
            if activation_reserve_bytes > 0:
                print(f"[VRAM] Activation Reserve: {activation_reserve_bytes / 1024**3:.2f} GB")
            if workspace_reserve_bytes > 0:
                print(f"[VRAM] Decode Workspace Reserve: {workspace_reserve_bytes / 1024**3:.2f} GB")
            
            if self.split_kv_cache:
                kv_k_size = self.kv_k_cache.element_size() * self.kv_k_cache.nelement()
                kv_k_size += self.kv_k_scales.element_size() * self.kv_k_scales.nelement()
                if self.kv_k_zeros is not None:
                    kv_k_size += self.kv_k_zeros.element_size() * self.kv_k_zeros.nelement()
                kv_v_size = self.kv_v_cache.element_size() * self.kv_v_cache.nelement()
                kv_v_size += self.kv_v_scales.element_size() * self.kv_v_scales.nelement()
                if self.kv_v_zeros is not None:
                    kv_v_size += self.kv_v_zeros.element_size() * self.kv_v_zeros.nelement()
                print(f"[VRAM] K Cache: {kv_k_size / 1024**3:.2f} GB")
                print(f"[VRAM] V Cache: {kv_v_size / 1024**3:.2f} GB")
            else:
                kv_size = self.kv_cache.element_size() * self.kv_cache.nelement()
                if self.quantizer:
                    kv_size += self.kv_scales.element_size() * self.kv_scales.nelement()
                print(f"[VRAM] KV Cache: {kv_size / 1024**3:.2f} GB")
            if self.decode_k_workspace is not None and self.decode_v_workspace is not None:
                ws_size = (
                    self.decode_k_workspace.element_size() * self.decode_k_workspace.nelement()
                    + self.decode_v_workspace.element_size() * self.decode_v_workspace.nelement()
                )
                print(
                    f"[VRAM] KV Decode Workspace: {ws_size / 1024**3:.2f} GB "
                    f"({self.decode_workspace_tokens} tokens)"
                )
                if self.quant_decode_graph and self.quant_graph_max_bs > 0:
                    print(f"[Graph] Quant Decode Max BS: {self.quant_graph_max_bs}")
            if self.quantizer and block_transient_bytes > 0:
                reserved_tmp = config.num_kvcache_blocks * block_transient_bytes
                print(f"[VRAM] KV Dequant Workspace(Est.): {reserved_tmp / 1024**3:.2f} GB")
            
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def _build_quant_slot_mapping(
        self,
        block_tables: torch.Tensor,
        seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seqlens_i32 = seqlens.to(dtype=torch.int32)
        cu_seqlens_k = torch.empty(seqlens_i32.numel() + 1, device=block_tables.device, dtype=torch.int32)
        cu_seqlens_k[0] = 0
        cu_seqlens_k[1:] = torch.cumsum(seqlens_i32, dim=0)

        max_blocks = int(block_tables.shape[1])
        if max_blocks == 0:
            return torch.empty(0, device=block_tables.device, dtype=torch.int64), cu_seqlens_k

        max_seqlen_upper = max_blocks * int(self.block_size)
        pos = torch.arange(max_seqlen_upper, device=block_tables.device, dtype=torch.int64)
        block_idx = torch.div(pos, self.block_size, rounding_mode="floor")
        block_off = pos - block_idx * self.block_size

        block_ids = block_tables.to(torch.int64).index_select(1, block_idx)
        slots = block_ids * self.block_size + block_off
        valid = pos.unsqueeze(0) < seqlens_i32.to(torch.int64).unsqueeze(1)
        slot_mapping = slots[valid]
        return slot_mapping, cu_seqlens_k

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        quant_slot_mapping = None
        quant_cu_seqlens_k = None
        quant_max_seqlen_k = 0
        if self.quantizer and block_tables is not None:
            seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            quant_slot_mapping, quant_cu_seqlens_k = self._build_quant_slot_mapping(block_tables, seqlens_k)
            quant_max_seqlen_k = max_seqlen_k
        quant_v_group_size = int(getattr(self.quantizer, "v_group_size", 0)) if self.quantizer is not None else 0

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
            quant_slot_mapping=quant_slot_mapping,
            quant_cu_seqlens_k=quant_cu_seqlens_k,
            quant_max_seqlen_k=quant_max_seqlen_k,
            quant_v_group_size=quant_v_group_size,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.arange(0, input_ids.numel() + 1, device=input_ids.device, dtype=torch.int32)
        block_tables = self.prepare_block_tables(seqs)

        quant_slot_mapping = None
        quant_cu_seqlens_k = None
        quant_max_seqlen_k = 0
        if self.quantizer:
            quant_slot_mapping, quant_cu_seqlens_k = self._build_quant_slot_mapping(block_tables, context_lens)
            quant_max_seqlen_k = int(block_tables.shape[1]) * int(self.block_size)
        quant_v_group_size = int(getattr(self.quantizer, "v_group_size", 0)) if self.quantizer is not None else 0

        set_context(
            False,
            cu_seqlens_q=cu_seqlens_q,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            quant_slot_mapping=quant_slot_mapping,
            quant_cu_seqlens_k=quant_cu_seqlens_k,
            quant_max_seqlen_k=quant_max_seqlen_k,
            quant_v_group_size=quant_v_group_size,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph_bs = next((x for x in self.graph_bs if x >= bs), None)
            if graph_bs is None:
                return self.model.compute_logits(self.model(input_ids, positions))

            quant_graph_enabled = (
                self.quantizer is not None
                and self.quant_decode_graph
                and self.quant_decode_backend in {"dequant_flash", "asym_turboquant"}
            )
            if quant_graph_enabled:
                if (
                    self.quant_graph_max_bs <= 0
                    or bs > self.quant_graph_max_bs
                ):
                    return self.model.compute_logits(self.model(input_ids, positions))
                if self.quant_decode_backend == "dequant_flash":
                    if (
                        context.quant_slot_mapping is None
                        or context.quant_cu_seqlens_k is None
                        or "quant_slot_mapping" not in self.graph_vars
                        or "quant_cu_seqlens_k" not in self.graph_vars
                    ):
                        return self.model.compute_logits(self.model(input_ids, positions))
                    qsm_len = int(context.quant_slot_mapping.numel())
                    graph_qsm_cap = int(self.graph_quant_tokens.get(graph_bs, 0))
                    if qsm_len <= 0 or qsm_len > graph_qsm_cap:
                        return self.model.compute_logits(self.model(input_ids, positions))

            graph = self.graphs[graph_bs]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            if self.quantizer and self.quant_decode_backend == "dequant_flash" and self.quant_decode_graph:
                qsm = context.quant_slot_mapping
                qcu = context.quant_cu_seqlens_k
                qsm_len = int(qsm.numel())
                qcu_len = int(qcu.numel())
                graph_vars["quant_slot_mapping"].zero_()
                graph_vars["quant_slot_mapping"][:qsm_len] = qsm
                graph_vars["quant_cu_seqlens_k"].zero_()
                graph_vars["quant_cu_seqlens_k"][:qcu_len] = qcu
                if qcu_len < graph_vars["quant_cu_seqlens_k"].numel():
                    graph_vars["quant_cu_seqlens_k"][qcu_len:] = qcu[-1]

            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(int(self.config.max_num_seqs), int(self.config.cuda_graph_max_bs), 512)
        if int(config.max_model_len) >= 4096:
            total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            # On low-VRAM cards, large graph captures waste memory for long-context runs.
            if total_memory <= 10 * 1024**3:
                max_bs = min(max_bs, 32)
        quant_graph_enabled = (
            self.quantizer is not None
            and self.quant_decode_backend in {"dequant_flash", "asym_turboquant"}
            and self.quant_decode_graph
        )
        if quant_graph_enabled:
            max_bs = min(max_bs, self.quant_graph_max_bs)
            if max_bs <= 0:
                self.enforce_eager = True
                return

        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        dequant_graph_enabled = quant_graph_enabled and self.quant_decode_backend == "dequant_flash"
        quant_slot_mapping = None
        quant_cu_seqlens_k = None
        max_quant_tokens = 0
        if dequant_graph_enabled:
            max_quant_tokens = min(int(self.decode_workspace_tokens), max_bs * int(config.max_model_len))
            if max_quant_tokens <= 0:
                self.enforce_eager = True
                return
            quant_slot_mapping = torch.zeros(max_quant_tokens, dtype=torch.int64)
            quant_cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32)

        graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        graph_bs = sorted({x for x in graph_bs if x <= max_bs})
        if max_bs not in graph_bs:
            graph_bs.append(max_bs)
        self.graph_bs = graph_bs
        self.graphs = {}
        self.graph_pool = None
        self.graph_quant_tokens = {}

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            if dequant_graph_enabled:
                cap_tokens = min(max_quant_tokens, bs * int(config.max_model_len))
                if cap_tokens <= 0:
                    continue
                if bs == 1:
                    quant_cu_seqlens_k[0] = 0
                    quant_cu_seqlens_k[1] = cap_tokens
                else:
                    quant_cu = torch.div(
                        torch.arange(0, bs + 1, dtype=torch.int64, device=quant_cu_seqlens_k.device) * cap_tokens,
                        bs,
                        rounding_mode="floor",
                    ).to(torch.int32)
                    quant_cu_seqlens_k[: bs + 1].copy_(quant_cu)
                set_context(
                    False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                    quant_slot_mapping=quant_slot_mapping[:cap_tokens],
                    quant_cu_seqlens_k=quant_cu_seqlens_k[: bs + 1],
                    quant_max_seqlen_k=int(config.max_model_len),
                    quant_v_group_size=int(getattr(self.quantizer, "v_group_size", 0)),
                )
            else:
                set_context(
                    False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                    quant_v_group_size=int(getattr(self.quantizer, "v_group_size", 0)) if self.quantizer is not None else 0,
                )

            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            if dequant_graph_enabled:
                self.graph_quant_tokens[bs] = cap_tokens
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        if dequant_graph_enabled:
            self.graph_vars["quant_slot_mapping"] = quant_slot_mapping
            self.graph_vars["quant_cu_seqlens_k"] = quant_cu_seqlens_k
