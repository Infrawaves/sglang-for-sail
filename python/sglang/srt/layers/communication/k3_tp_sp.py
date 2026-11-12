"""Contiguous token shards for Kimi-K3/K3.1 eager TP-SP."""

from contextlib import nullcontext

import torch

from sglang.srt.runtime_context import get_parallel


class TPTokenShard:
    def __init__(self, num_tokens: int):
        self.group = get_parallel().tp_group
        self.num_tokens = num_tokens
        self.shard_size = (
            num_tokens + self.group.world_size - 1
        ) // self.group.world_size
        self.padded_size = self.shard_size * self.group.world_size
        start = min(self.group.rank_in_group * self.shard_size, num_tokens)
        stop = min(start + self.shard_size, num_tokens)
        self.rows = slice(start, stop)
        self.local_size = stop - start
        self.valid_size = self.local_size
        self.has_padding = False

    @classmethod
    def interleave(cls, padded_rows: int, valid_rows: int, *, has_padding: bool = True):
        """Keep padded rank-major CP rows through both AG and RS."""
        shard = cls.__new__(cls)
        shard.group = get_parallel().attn_cp_group
        assert 0 <= valid_rows <= padded_rows
        shard.shard_size = padded_rows
        shard.padded_size = shard.num_tokens = padded_rows * shard.group.world_size
        shard.local_size = padded_rows
        shard.valid_size = valid_rows
        shard.has_padding = has_padding
        start = shard.group.rank_in_group * padded_rows
        shard.rows = slice(start, start + padded_rows)
        return shard

    @property
    def pp_layout(self) -> tuple[int, int, int]:
        return self.num_tokens, self.rows.start, self.group.world_size

    @staticmethod
    def _pad(x: torch.Tensor, size: int) -> torch.Tensor:
        if x.shape[0] == size:
            return x.contiguous()
        padding = (0, 0) * (x.ndim - 1) + (0, size - x.shape[0])
        return torch.nn.functional.pad(x, padding)

    @staticmethod
    def _comm(group):
        # Eager c10d uses its own stream and adds host dispatch overhead.
        # PyNCCL enqueues directly on the caller's main/alt stream.
        comm = group.pynccl_comm
        return nullcontext() if comm is None else comm.change_state(enable=True)

    def all_gather(self, local: torch.Tensor, group=None) -> torch.Tensor:
        if self.num_tokens == 0:
            return local
        group = self.group if group is None else group
        full = local.new_empty((self.padded_size, *local.shape[1:]))
        with self._comm(group):
            group.all_gather_into_tensor(full, self._pad(local, self.shard_size))
        return full[: self.num_tokens]

    def reduce_scatter(self, partial: torch.Tensor, group=None) -> torch.Tensor:
        if self.num_tokens == 0:
            return partial
        group = self.group if group is None else group
        local = partial.new_empty((self.shard_size, *partial.shape[1:]))
        with self._comm(group):
            group.reduce_scatter_tensor(local, self._pad(partial, self.padded_size))
        return local[: self.local_size]

    @staticmethod
    def _use_triton_packing(tensors) -> bool:
        return (
            tensors[0].is_cuda
            and len(tensors) in (3, 4)
            and all(t.element_size() in (1, 2, 4, 8) for t in tensors)
        )

    def pack_moe_inputs(self, *tensors: torch.Tensor) -> torch.Tensor:
        if self._use_triton_packing(tensors):
            from sglang.kernels.ops.communication.k3_moe_pack import pack_moe_inputs

            return pack_moe_inputs(tensors, self.shard_size)

        alignment = max(t.element_size() for t in tensors)
        widths = [t.shape[1] * t.element_size() for t in tensors]
        strides = [(width + alignment - 1) // alignment * alignment for width in widths]
        packed = torch.zeros(
            (self.shard_size, sum(strides)), dtype=torch.uint8, device=tensors[0].device
        )
        offset = 0
        for tensor, width, stride in zip(tensors, widths, strides):
            packed[: tensor.shape[0], offset : offset + width].view(tensor.dtype).copy_(
                tensor
            )
            offset += stride
        return packed

    def unpack_moe_inputs(
        self, gathered: torch.Tensor, *tensors: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        if self._use_triton_packing(tensors):
            from sglang.kernels.ops.communication.k3_moe_pack import unpack_moe_inputs

            return unpack_moe_inputs(gathered, tensors)

        alignment = max(t.element_size() for t in tensors)
        outputs = []
        offset = 0
        for tensor in tensors:
            width = tensor.shape[1] * tensor.element_size()
            outputs.append(
                gathered[:, offset : offset + width].contiguous().view(tensor.dtype)
            )
            offset += (width + alignment - 1) // alignment * alignment
        return tuple(outputs)
