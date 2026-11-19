"""Persistent symmetric receive storage for K3.1 shared-expert CE AllGather."""

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    SymmetricMemoryContext,
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.runtime_context import get_buffer


class _SharedGatherWorkspace:
    def __init__(self):
        self.output = None

    def reserve(self, local: torch.Tensor, rows: int, group) -> torch.Tensor:
        if self.output is None or self.output.shape[0] < rows:
            if self.output is not None:
                torch.cuda.current_stream().synchronize()
            capacity = 1 << (rows - 1).bit_length()
            saved_context = disable_symmetric_memory_context()
            try:
                with SymmetricMemoryContext(group):
                    self.output = local.new_empty((capacity, *local.shape[1:]))
            finally:
                restore_symmetric_memory_context(saved_context)
        return self.output[:rows]


def prepare_shared_ce_gather(local: torch.Tensor, shard, group) -> torch.Tensor:
    """Register on main before the fork; reuse is ordered by the shared tail join."""
    if group is None or group.ranks != shard.group.ranks:
        raise ValueError("Shared-expert CE AllGather requires full shared TP")
    comm = group.pynccl_comm
    if comm is None or not comm.available:
        raise RuntimeError("Shared-expert CE AllGather requires a PCCL communicator")
    key = (
        f"k3_shared_ce_ag:{group.unique_name}:{local.device}:"
        f"{local.dtype}:{tuple(local.shape[1:])}"
    )
    workspace = get_buffer(key, _SharedGatherWorkspace)
    return workspace.reserve(local, shard.padded_size, group)
