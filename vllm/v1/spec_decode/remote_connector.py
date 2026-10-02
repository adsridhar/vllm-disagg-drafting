# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid
from typing import TYPE_CHECKING

import numpy as np
import zmq

from vllm.v1.spec_decode.remote_protocol import (
    ProposalRequest,
    decode_response,
    encode_request,
)

if TYPE_CHECKING:
    from vllm.v1.spec_decode.remote_ipc import IpcDraftRing


class RemoteDraftConnector:
    """Blocking target-side connection to a stateless draft service."""

    def __init__(self, address: str, timeout_ms: int):
        self.socket = zmq.Context.instance().socket(zmq.DEALER)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(address)
        self.next_request_id = 0

    def propose(
        self,
        prefixes: list[list[int]],
        num_speculative_tokens: int,
    ) -> list[list[int]]:
        request_id = self.next_request_id
        self.next_request_id += 1
        lengths = [len(prefix) for prefix in prefixes]
        flat_tokens = np.asarray(
            [token for prefix in prefixes for token in prefix], dtype=np.int32
        )
        request = ProposalRequest(request_id, lengths, num_speculative_tokens)
        self.socket.send_multipart(encode_request(request, flat_tokens), copy=False)
        response, tokens = decode_response(self.socket.recv_multipart())
        if response.request_id != request_id:
            raise RuntimeError("Draft response does not match the request")

        proposals = []
        offset = 0
        for length in response.proposal_lengths:
            proposals.append(tokens[offset : offset + length].tolist())
            offset += length
        return proposals

    def close(self) -> None:
        self.socket.close(linger=0)


class ResumableDraftConnector(RemoteDraftConnector):
    """Persistent request control with ZMQ or CUDA IPC round payloads."""

    def __init__(self, address: str, timeout_ms: int):
        super().__init__(address, timeout_ms)
        self.peer_id = uuid.uuid4().hex
        self.generations: dict[int, int] = {}
        self.ring: IpcDraftRing | None = None

    def request(
        self,
        operation: str,
        handles: list[int],
        k: int = 0,
        prefixes: list[list[int]] | None = None,
        accepted: list[int] | None = None,
        bonuses: list[int] | None = None,
        ipc_handles: bytes = b"",
    ):
        prefixes = prefixes or []
        request = ProposalRequest(
            self.next_request_id,
            [len(p) for p in prefixes],
            k,
            operation=operation,
            peer_id=self.peer_id,
            handles=handles,
            generations=[self.generations[h] for h in handles]
            if operation == "commit"
            else [],
            accepted_counts=accepted or [],
            bonus_tokens=bonuses or [],
            ipc_handles=ipc_handles,
        )
        self.next_request_id += 1
        tokens = np.asarray([t for p in prefixes for t in p], dtype=np.int32)
        self.socket.send_multipart(encode_request(request, tokens), copy=False)
        response, tokens = decode_response(self.socket.recv_multipart())
        if response.request_id != request.request_id:
            raise RuntimeError("Draft response does not match the request")
        if response.generations:
            self.generations.update(zip(handles, response.generations, strict=True))
        proposals = []
        offset = 0
        for length in response.proposal_lengths:
            proposals.append(tokens[offset : offset + length].tolist())
            offset += length
        return proposals

    def open_requests(self, handles: list[int], prefixes: list[list[int]], k: int):
        return self.request("open", handles, k, prefixes)

    def establish_ipc(self, device, max_batch: int, k: int) -> None:
        from vllm.v1.spec_decode.remote_ipc import IpcDraftRing

        ring = IpcDraftRing(device, max_batch, k, self.socket.getsockopt(zmq.RCVTIMEO))
        self.request("ipc", [], k, ipc_handles=ring.export())
        self.ring = ring

    def commit_and_propose(self, handles: list[int], accepted, bonuses):
        proposals = self.request("commit", handles, accepted=accepted, bonuses=bonuses)
        return proposals

    def dispatch(self, handles: list[int], accepted, bonuses):
        import torch

        assert self.ring is not None
        identifiers = torch.tensor(
            [[h, self.generations[h]] for h in handles],
            dtype=torch.int64,
            device=self.ring.device,
        )
        metadata = torch.cat((identifiers, accepted[:, None], bonuses[:, None]), dim=1)
        return self.ring.dispatch(metadata)

    def await_proposals(self, handles: list[int], pending):
        assert self.ring is not None
        result = self.ring.await_response(pending)
        for handle in handles:
            self.generations[handle] += 1
        return result

    def close_requests(self, handles: list[int]) -> None:
        if handles:
            self.request("close", handles)
            for handle in handles:
                del self.generations[handle]

    def close(self) -> None:
        self.close_requests(list(self.generations))
        if self.ring is not None:
            self.request("disconnect", [])
            self.ring = None
        super().close()
