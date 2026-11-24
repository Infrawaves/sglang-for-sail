"""Persistent symmetric receive storage for K3.1 shared-expert CE AllGather."""

import logging

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    SymmetricMemoryContext,
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.runtime_context import get_buffer, max_prefill_buffer_tokens

logger = logging.getLogger(__name__)


class _SharedGatherWorkspace:
    def __init__(self):
        self.output = None
        self.captured_outputs: dict[int, torch.Tensor] = {}

    def get_symm_output(
        self, local: torch.Tensor, rows: int, group, *, capturing: bool
    ) -> torch.Tensor:
        """Reserve for eager prefill before capture pins the receive address."""
        if self.output is None or self.output.shape[0] < rows:
            if capturing:
                raise RuntimeError(
                    f"Shared-expert CE AllGather warmup did not reserve {rows} rows"
                )
            previous_rows = 0 if self.output is None else self.output.shape[0]
            if self.output is not None:
                torch.cuda.current_stream().synchronize()
            reserve_rows = rows
            if self.output is None:
                reserve_rows = max(rows, max_prefill_buffer_tokens())
            capacity = 1 << (reserve_rows - 1).bit_length()
            saved_context = disable_symmetric_memory_context()
            try:
                with SymmetricMemoryContext(group):
                    self.output = local.new_empty((capacity, *local.shape[1:]))
            finally:
                restore_symmetric_memory_context(saved_context)
            logger.info(
                "K3 shared-expert CE AG symm buffer on %s: %d rows x %s %s, "
                "%.1f MB (requested %d rows, previous capacity %d rows)",
                group.unique_name,
                capacity,
                tuple(local.shape[1:]),
                local.dtype,
                self.output.numel() * self.output.element_size() / (1024 * 1024),
                rows,
                previous_rows,
            )
        if capturing:
            self.captured_outputs[self.output.shape[0]] = self.output
        return self.output[:rows]


def prepare_shared_ce_gather(
    local: torch.Tensor, shard, group, *, capturing: bool = False
) -> torch.Tensor:
    """Register before the fork in eager mode; reuse registered storage in capture."""
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
    return workspace.get_symm_output(
        local, shard.padded_size, group, capturing=capturing
    )
