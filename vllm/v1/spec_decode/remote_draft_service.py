# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Protocol

import numpy as np
import uvloop
import zmq
import zmq.asyncio

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.spec_decode.remote_protocol import (
    ProposalRequest,
    ProposalResponse,
    decode_request,
    encode_response,
)

logger = init_logger(__name__)


class DraftEngine(Protocol):
    def generate(
        self,
        prompt: TokensPrompt,
        sampling_params: SamplingParams,
        request_id: str,
    ) -> AsyncIterator[RequestOutput]: ...


class RemoteDraftService:
    """Batches stateless proposal requests through a normal AsyncLLM."""

    def __init__(self, engine: DraftEngine):
        self.engine = engine

    async def propose(
        self, request: ProposalRequest, flat_tokens: np.ndarray
    ) -> list[list[int]]:
        prefixes = []
        offset = 0
        for length in request.prefix_lengths:
            prefixes.append(flat_tokens[offset : offset + length].tolist())
            offset += length

        params = SamplingParams(
            temperature=0,
            max_tokens=request.num_speculative_tokens,
            ignore_eos=True,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        async def generate(row: int, prefix: list[int]) -> list[int]:
            output = None
            request_id = f"remote-draft-{request.request_id}-{row}"
            async for output in self.engine.generate(
                TokensPrompt(prompt_token_ids=prefix), params, request_id
            ):
                pass
            assert output is not None
            return list(output.outputs[0].token_ids)

        return await asyncio.gather(
            *(generate(row, prefix) for row, prefix in enumerate(prefixes))
        )


async def serve_draft_requests(
    engine: DraftEngine, port: int, ready: Callable[[], None] | None = None
) -> None:
    socket = zmq.asyncio.Context.instance().socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{port}")
    logger.info("Remote draft service listening on port %d", port)
    if ready is not None:
        ready()
    service = RemoteDraftService(engine)
    try:
        while True:
            identity, *frames = await socket.recv_multipart()
            request, tokens = decode_request(frames)
            proposals = await service.propose(request, tokens)
            lengths = [len(proposal) for proposal in proposals]
            flat = np.asarray(
                [token for proposal in proposals for token in proposal],
                dtype=np.int32,
            )
            response = ProposalResponse(request.request_id, lengths)
            await socket.send_multipart(
                [identity, *encode_response(response, flat)], copy=False
            )
    finally:
        socket.close(linger=0)


def run_draft_server(args) -> None:
    async def run() -> None:
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine = AsyncLLM.from_engine_args(engine_args)
        try:
            await serve_draft_requests(engine, args.draft_server_port)
        finally:
            engine.shutdown()

    uvloop.run(run())
