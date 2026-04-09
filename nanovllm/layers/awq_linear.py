import torch
import torch.distributed as dist
from torch import nn


AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _pack_num(bits: int = 4) -> int:
    return 32 // bits


def _unpack_awq_matrix(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
    if packed.dim() != 2:
        raise ValueError(f"expected a 2D packed tensor, got shape {tuple(packed.shape)}")
    shifts = torch.arange(0, 32, bits, device=packed.device, dtype=torch.int32)
    unpacked = torch.bitwise_right_shift(packed.to(torch.int32).unsqueeze(-1), shifts).to(torch.int8)
    return unpacked.reshape(packed.shape[0], -1)


def _reverse_awq_order(values: torch.Tensor) -> torch.Tensor:
    if values.shape[1] % len(AWQ_REVERSE_ORDER) != 0:
        raise ValueError(f"AWQ packed width must be divisible by {len(AWQ_REVERSE_ORDER)}")
    order = torch.arange(values.shape[1], device=values.device, dtype=torch.int64)
    order = order.view(-1, len(AWQ_REVERSE_ORDER))[:, AWQ_REVERSE_ORDER].reshape(-1)
    return values.index_select(1, order)


def dequantize_awq_gemm(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    qweight_unpacked = _reverse_awq_order(_unpack_awq_matrix(qweight))
    qzeros_unpacked = _reverse_awq_order(_unpack_awq_matrix(qzeros))

    max_value = (1 << 4) - 1
    qweight_unpacked = torch.bitwise_and(qweight_unpacked, max_value).to(torch.float32)
    qzeros_unpacked = torch.bitwise_and(qzeros_unpacked, max_value).to(torch.float32)

    scales = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    qzeros_unpacked = qzeros_unpacked.repeat_interleave(group_size, dim=0)
    return ((qweight_unpacked - qzeros_unpacked) * scales).to(dtype)


def _awq_matmul_chunked(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
    chunk_size: int = 1024,
) -> torch.Tensor:
    outputs = []
    total_output_size = int(scales.shape[1])
    if total_output_size % _pack_num() != 0:
        raise ValueError(f"AWQ output width must be divisible by {_pack_num()}")

    for start in range(0, total_output_size, chunk_size):
        end = min(start + chunk_size, total_output_size)
        packed_start = start // _pack_num()
        packed_end = end // _pack_num()
        weight_chunk = dequantize_awq_gemm(
            qweight[:, packed_start:packed_end],
            qzeros[:, packed_start:packed_end],
            scales[:, start:end],
            group_size,
            dtype=dtype,
        )
        outputs.append(torch.matmul(x, weight_chunk))
        del weight_chunk
    return torch.cat(outputs, dim=-1)


def _copy_sharded_columns(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_offset: int,
    shard_size: int,
    tp_rank: int,
    tp_size: int,
) -> None:
    shard = loaded_weight if tp_size == 1 else loaded_weight.chunk(tp_size, dim=1)[tp_rank]
    shard = shard.to(dtype=param.dtype, device=param.device)
    param.data.narrow(1, shard_offset, shard_size).copy_(shard)


def _copy_sharded_rows(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> None:
    shard = loaded_weight if tp_size == 1 else loaded_weight.chunk(tp_size, dim=0)[tp_rank]
    shard = shard.to(dtype=param.dtype, device=param.device)
    param.data.copy_(shard)


def _copy_sharded_vector(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    tp_rank: int,
    tp_size: int,
) -> None:
    shard = loaded_weight if tp_size == 1 else loaded_weight.chunk(tp_size, dim=0)[tp_rank]
    shard = shard.to(dtype=param.dtype, device=param.device)
    param.data.copy_(shard)


class AWQFusedColumnParallelLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        shard_ids: list[str | int],
        group_size: int,
        bias: bool,
    ) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.shard_ids = shard_ids
        self.group_size = group_size
        self.pack_num = _pack_num()
        self.local_output_sizes = []
        for size in output_sizes:
            assert size % self.tp_size == 0
            self.local_output_sizes.append(size // self.tp_size)
        self.out_features = sum(output_sizes)
        self.local_out_features = sum(self.local_output_sizes)
        assert self.input_size % self.group_size == 0
        assert self.local_out_features % self.pack_num == 0
        assert self.out_features % self.pack_num == 0

        self.qweight = nn.Parameter(
            torch.empty(self.input_size, self.local_out_features // self.pack_num, dtype=torch.int32),
            requires_grad=False,
        )
        self.qzeros = nn.Parameter(
            torch.empty(self.input_size // self.group_size, self.local_out_features // self.pack_num, dtype=torch.int32),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.empty(self.input_size // self.group_size, self.local_out_features, dtype=torch.float16),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out_features, dtype=torch.float16), requires_grad=False)
        else:
            self.bias = None

        self.qweight.weight_loader = self._load_qweight
        self.qzeros.weight_loader = self._load_qzeros
        self.scales.weight_loader = self._load_scales
        if self.bias is not None:
            self.bias.weight_loader = self._load_bias

    def _resolve_shard(self, loaded_shard_id: str | int) -> tuple[int, int]:
        shard_index = self.shard_ids.index(loaded_shard_id)
        shard_offset = sum(self.local_output_sizes[:shard_index])
        shard_size = self.local_output_sizes[shard_index]
        return shard_offset, shard_size

    def _load_qweight(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | int) -> None:
        shard_offset, shard_size = self._resolve_shard(loaded_shard_id)
        _copy_sharded_columns(param, loaded_weight, shard_offset // self.pack_num, shard_size // self.pack_num, self.tp_rank, self.tp_size)

    def _load_qzeros(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | int) -> None:
        shard_offset, shard_size = self._resolve_shard(loaded_shard_id)
        _copy_sharded_columns(param, loaded_weight, shard_offset // self.pack_num, shard_size // self.pack_num, self.tp_rank, self.tp_size)

    def _load_scales(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | int) -> None:
        shard_offset, shard_size = self._resolve_shard(loaded_shard_id)
        _copy_sharded_columns(param, loaded_weight, shard_offset, shard_size, self.tp_rank, self.tp_size)

    def _load_bias(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | int) -> None:
        shard_offset, shard_size = self._resolve_shard(loaded_shard_id)
        shard = loaded_weight if self.tp_size == 1 else loaded_weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        shard = shard.narrow(0, shard_offset, shard_size).to(dtype=param.dtype, device=param.device)
        param.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = _awq_matmul_chunked(x, self.qweight, self.qzeros, self.scales, self.group_size, dtype=x.dtype)
        if self.bias is not None:
            output = output + self.bias
        return output


class AWQRowParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, group_size: int, bias: bool) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert input_size % self.tp_size == 0
        self.input_size = input_size
        self.local_input_size = input_size // self.tp_size
        self.output_size = output_size
        self.group_size = group_size
        self.pack_num = _pack_num()
        assert self.local_input_size % self.group_size == 0
        assert self.output_size % self.pack_num == 0

        self.qweight = nn.Parameter(
            torch.empty(self.local_input_size, self.output_size // self.pack_num, dtype=torch.int32),
            requires_grad=False,
        )
        self.qzeros = nn.Parameter(
            torch.empty(self.local_input_size // self.group_size, self.output_size // self.pack_num, dtype=torch.int32),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.empty(self.local_input_size // self.group_size, self.output_size, dtype=torch.float16),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size, dtype=torch.float16), requires_grad=False)
        else:
            self.bias = None

        self.qweight.weight_loader = self._load_qweight
        self.qzeros.weight_loader = self._load_qzeros
        self.scales.weight_loader = self._load_scales
        if self.bias is not None:
            self.bias.weight_loader = self._load_bias

    def _load_qweight(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        _copy_sharded_rows(param, loaded_weight, self.tp_rank, self.tp_size)

    def _load_qzeros(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        _copy_sharded_rows(param, loaded_weight, self.tp_rank, self.tp_size)

    def _load_scales(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        _copy_sharded_rows(param, loaded_weight, self.tp_rank, self.tp_size)

    def _load_bias(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight.to(dtype=param.dtype, device=param.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = _awq_matmul_chunked(x, self.qweight, self.qzeros, self.scales, self.group_size, dtype=x.dtype)
        if self.bias is not None and self.tp_rank == 0:
            output = output + self.bias
        if self.tp_size > 1:
            dist.all_reduce(output)
        return output


class AWQQKVParallelLinear(AWQFusedColumnParallelLinear):
    def __init__(self, input_size: int, q_size: int, kv_size: int, group_size: int, bias: bool) -> None:
        super().__init__(input_size, [q_size, kv_size, kv_size], ["q", "k", "v"], group_size, bias)


class AWQMergedColumnParallelLinear(AWQFusedColumnParallelLinear):
    def __init__(self, input_size: int, intermediate_size: int, group_size: int, bias: bool) -> None:
        super().__init__(input_size, [intermediate_size, intermediate_size], [0, 1], group_size, bias)