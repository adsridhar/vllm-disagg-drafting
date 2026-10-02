# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import numpy as np


class ProposalRequest(msgspec.Struct):
    request_id: int
    prefix_lengths: list[int]
    num_speculative_tokens: int
    operation: str = "propose"
    peer_id: str = ""
    handles: list[int] = []
    generations: list[int] = []
    accepted_counts: list[int] = []
    bonus_tokens: list[int] = []
    ipc_handles: bytes = b""


class ProposalResponse(msgspec.Struct):
    request_id: int
    proposal_lengths: list[int]
    generations: list[int] = []


_request_encoder = msgspec.msgpack.Encoder()
_request_decoder = msgspec.msgpack.Decoder(ProposalRequest)
_response_encoder = msgspec.msgpack.Encoder()
_response_decoder = msgspec.msgpack.Decoder(ProposalResponse)


def encode_request(
    request: ProposalRequest, token_ids: np.ndarray
) -> list[bytes | memoryview]:
    return [_request_encoder.encode(request), memoryview(token_ids)]


def decode_request(frames: list[bytes]) -> tuple[ProposalRequest, np.ndarray]:
    request = _request_decoder.decode(frames[0])
    token_ids = np.frombuffer(frames[1], dtype=np.int32)
    return request, token_ids


def encode_response(
    response: ProposalResponse, token_ids: np.ndarray
) -> list[bytes | memoryview]:
    return [_response_encoder.encode(response), memoryview(token_ids)]


def decode_response(frames: list[bytes]) -> tuple[ProposalResponse, np.ndarray]:
    response = _response_decoder.decode(frames[0])
    token_ids = np.frombuffer(frames[1], dtype=np.int32)
    return response, token_ids
