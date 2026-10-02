# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-node token transport, extracted from the original draft IPC ring.

ZMQ exchanges handles once. GPU payload copies precede doorbells on the
same stream. Slot ownership lasts until the target consumes the response.
"""

import time
from dataclasses import dataclass

import msgspec
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

NUM_SLOTS = 16


class CudaTensorHandle(msgspec.Struct):
    size: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int
    device: int
    storage_handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    requires_grad: bool
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool


def export_tensor(tensor: torch.Tensor) -> CudaTensorHandle:
    if tensor.device.type != "cuda" or tensor.dtype != torch.int64:
        raise ValueError("Draft IPC requires CUDA int64 tensors")
    rebuild, args = reduce_tensor(tensor)
    assert rebuild is rebuild_cuda_tensor
    return CudaTensorHandle(tuple(args[1]), tuple(args[2]), args[3], *args[6:])


def rebuild_tensors(handles: bytes, role: str) -> dict[str, torch.Tensor]:
    peers = msgspec.msgpack.decode(handles, type=dict[str, dict[str, CudaTensorHandle]])
    return {
        name: rebuild_cuda_tensor(
            torch.Tensor,
            handle.size,
            handle.stride,
            handle.offset,
            torch.storage.TypedStorage,
            torch.int64,
            handle.device,
            handle.storage_handle,
            handle.storage_size_bytes,
            handle.storage_offset_bytes,
            handle.requires_grad,
            handle.ref_counter_handle,
            handle.ref_counter_offset,
            handle.event_handle,
            handle.event_sync_required,
        )
        for name, handle in peers[role].items()
    }


@dataclass
class PendingDraft:
    slot: int
    sequence: int
    batch_size: int


class IpcDraftRing:
    """Target-owned buffers shared with the service and draft GPU worker."""

    def __init__(self, device: torch.device, max_batch: int, k: int, timeout_ms: int):
        self.device = device
        self.max_batch = max_batch
        self.k = k
        self.timeout_ms = timeout_ms
        self.requests = torch.zeros(
            NUM_SLOTS, max_batch, 4, dtype=torch.int64, device=device
        )
        self.proposals = torch.zeros(
            NUM_SLOTS, max_batch, k, dtype=torch.int64, device=device
        )
        self.request_doorbells = torch.zeros(
            NUM_SLOTS, dtype=torch.int64, device=device
        )
        self.response_doorbells = torch.zeros_like(self.request_doorbells)
        self.next_slot = 0
        self.sequences = [0] * NUM_SLOTS
        self.pending: set[int] = set()

    def export(self) -> bytes:
        # Each exported tensor has one importing process/refcount owner.
        return msgspec.msgpack.encode(
            {
                "service": {
                    "requests": export_tensor(self.requests),
                    "doorbells": export_tensor(self.request_doorbells),
                },
                "worker": {
                    "proposals": export_tensor(self.proposals),
                    "doorbells": export_tensor(self.response_doorbells),
                },
            }
        )

    def dispatch(self, metadata: torch.Tensor) -> PendingDraft:
        batch_size = metadata.shape[0]
        assert batch_size <= self.max_batch and batch_size < (1 << 16)
        slot = self.next_slot
        assert slot not in self.pending
        self.pending.add(slot)
        self.next_slot = (slot + 1) % NUM_SLOTS
        self.sequences[slot] += 1
        sequence = self.sequences[slot]
        self.requests[slot, :batch_size].copy_(metadata, non_blocking=True)
        self.request_doorbells[slot].fill_((sequence << 16) | batch_size)
        return PendingDraft(slot, sequence, batch_size)

    def await_response(self, pending: PendingDraft) -> torch.Tensor:
        deadline = time.monotonic() + self.timeout_ms / 1000
        while self.response_doorbells[pending.slot].item() != pending.sequence:
            if time.monotonic() > deadline:
                raise TimeoutError("CUDA IPC draft response timed out")
        # Clone before the slot can be reused by another round.
        result = self.proposals[pending.slot, : pending.batch_size].clone()
        self.pending.remove(pending.slot)
        return result


class IpcDraftPeer:
    """Service-side polling of bounded outcome metadata, without model state."""

    def __init__(self, handles: bytes, k: int):
        self.k = k
        self.buffers = rebuild_tensors(handles, "service")
        self.device = self.buffers["requests"].device
        self.last_seen = [0] * NUM_SLOTS
        self.doorbells_cpu = torch.zeros(NUM_SLOTS, dtype=torch.int64, pin_memory=True)
        self.requests_cpu = torch.empty_like(
            self.buffers["requests"], device="cpu", pin_memory=True
        )
        self.stream = torch.cuda.Stream(self.device)

    def poll(self):
        with torch.cuda.stream(self.stream):
            self.doorbells_cpu.copy_(self.buffers["doorbells"], non_blocking=True)
        self.stream.synchronize()
        pending = []
        for slot, encoded in enumerate(self.doorbells_cpu.tolist()):
            sequence, batch_size = encoded >> 16, encoded & 0xFFFF
            if sequence == self.last_seen[slot]:
                continue
            with torch.cuda.stream(self.stream):
                self.requests_cpu[slot, :batch_size].copy_(
                    self.buffers["requests"][slot, :batch_size], non_blocking=True
                )
            self.stream.synchronize()
            self.last_seen[slot] = sequence
            pending.append(
                (slot, sequence, self.requests_cpu[slot, :batch_size].numpy().copy())
            )
        return pending


def attach_draft_ipc(worker, peer_id: str, handles: bytes) -> None:
    """Existing collective RPC installs transport buffers in the GPU worker."""
    runner = worker.model_runner
    with torch.accelerator.device_index(runner.device.index):
        runner.external_draft_ipc[peer_id] = rebuild_tensors(handles, "worker")


def publish_draft_ipc(worker, peer_id: str, slot: int, sequence: int) -> None:
    """Signal only after the scheduler has paused all rows in this round."""
    buffers = worker.model_runner.external_draft_ipc[peer_id]
    with torch.accelerator.device_index(buffers["proposals"].device.index):
        # The worker's proposal copies also use this device's current stream.
        buffers["doorbells"][slot].fill_(sequence)
