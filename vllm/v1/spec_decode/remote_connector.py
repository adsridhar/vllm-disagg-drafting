# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import zmq

from vllm.v1.spec_decode.remote_protocol import (
    ProposalRequest,
    decode_response,
    encode_request,
)


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
