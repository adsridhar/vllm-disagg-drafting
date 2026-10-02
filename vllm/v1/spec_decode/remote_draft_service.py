# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Protocol

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
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.spec_decode.remote_protocol import (
    ProposalRequest,
    ProposalResponse,
    decode_request,
    encode_response,
)

if TYPE_CHECKING:
    from vllm.v1.spec_decode.remote_ipc import IpcDraftPeer

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


class ResumableDraftAdapter:
    """Owns collectors for normal engine requests awaiting verification."""

    def __init__(self, engine: AsyncLLM):
        self.engine = engine
        self.collectors: dict[str, RequestOutputCollector] = {}

    async def open(self, handle: str, prefix: list[int], k: int) -> list[int]:
        params = SamplingParams(
            temperature=0,
            max_tokens=k,
            ignore_eos=True,
            detokenize=False,
            output_kind=RequestOutputKind.DELTA,
            extra_args={"external_draft": True},
        )
        request = self.engine.input_processor.process_inputs(
            handle,
            TokensPrompt(prompt_token_ids=prefix),
            params,
            supported_tasks=await self.engine.get_supported_tasks(),
            resumable=True,
        )
        collector = await self.engine.add_request(handle, request, params)
        self.collectors[handle] = collector
        return await self.collect(handle)

    async def collect(self, handle: str) -> list[int]:
        tokens: list[int] = []
        collector = self.collectors[handle]
        while True:
            output = collector.get_nowait() or await collector.get()
            assert isinstance(output, RequestOutput)
            completion = output.outputs[0]
            tokens.extend(completion.token_ids)
            if completion.finish_reason is not None:
                if completion.finish_reason != "length":
                    raise RuntimeError(
                        f"Draft round stopped: {completion.finish_reason}"
                    )
                return tokens

    async def commit(
        self,
        handles: list[str],
        generations: list[int],
        accepted: list[int],
        bonuses: list[int],
        ipc_routes: list[tuple[str, int, int]] | None = None,
    ) -> list[list[int]]:
        commits = []
        for row, (handle, generation, count, bonus) in enumerate(
            zip(handles, generations, accepted, bonuses, strict=True)
        ):
            internal_id = self.collectors[handle].request_id
            state = self.engine.output_processor.request_states[internal_id]
            assert state.input_chunk_queue is None
            assert state.detokenizer is not None
            proposal = state.detokenizer.output_token_ids
            assert state.prompt_token_ids is not None
            suffix = proposal[:count] + [bonus]
            route = ipc_routes[row] if ipc_routes else None
            commits.append((internal_id, generation, count, bonus, route))
            state.prompt_token_ids.extend(suffix)
            state.prompt_len = len(state.prompt_token_ids)
            state.detokenizer.token_ids.clear()
            state.sent_tokens_offset = 0
            state.is_prefilling = True
            state.input_chunk_queue = deque()
        await self.engine.engine_core.call_utility_async(
            "commit_external_verification", commits
        )
        return await asyncio.gather(*(self.collect(handle) for handle in handles))

    async def close(self, handles: list[str]) -> None:
        collectors = [self.collectors.pop(handle) for handle in handles]
        await self.engine.abort([c.request_id for c in collectors], internal=True)
        for collector in collectors:
            collector.close()


class ResumableDraftService:
    """Persistent proposals using the engine's resumable request lifecycle."""

    def __init__(self, engine: AsyncLLM):
        self.engine = engine
        self.adapter = ResumableDraftAdapter(engine)
        self.generations: dict[str, int] = {}
        self.ipc_peers: dict[str, IpcDraftPeer] = {}

    async def handle(self, request: ProposalRequest, flat_tokens: np.ndarray):
        handles = [f"{request.peer_id}:{h}" for h in request.handles]
        if request.operation == "open":
            offset = 0
            prefixes = []
            for length in request.prefix_lengths:
                prefixes.append(flat_tokens[offset : offset + length].tolist())
                offset += length
            if any(handle in self.generations for handle in handles):
                raise ValueError("Draft handle already exists")
            proposals = await asyncio.gather(
                *(
                    self.adapter.open(h, p, request.num_speculative_tokens)
                    for h, p in zip(handles, prefixes, strict=True)
                )
            )
            for handle in handles:
                self.generations[handle] = 0
        elif request.operation == "commit":
            proposals = await self.commit(request)
        elif request.operation == "close":
            await self.adapter.close(handles)
            for handle in handles:
                del self.generations[handle]
            proposals = []
        elif request.operation == "disconnect":
            await self.engine.collective_rpc(
                "detach_remote_draft_ipc", args=(request.peer_id,)
            )
            peer = self.ipc_peers.pop(request.peer_id)
            peer.buffers.clear()
            proposals = []
        elif request.operation == "ipc":
            from vllm.v1.spec_decode.remote_ipc import IpcDraftPeer

            peer = IpcDraftPeer(request.ipc_handles, request.num_speculative_tokens)
            await self.engine.collective_rpc(
                "attach_remote_draft_ipc", args=(request.peer_id, request.ipc_handles)
            )
            self.ipc_peers[request.peer_id] = peer
            proposals = []
        else:
            raise ValueError(f"Unsupported draft operation: {request.operation}")
        generations = [self.generations[h] for h in handles] if proposals else []
        return proposals, generations

    async def commit(self, request: ProposalRequest, slot: int | None = None):
        handles = [f"{request.peer_id}:{h}" for h in request.handles]
        for handle, generation in zip(handles, request.generations, strict=True):
            if self.generations[handle] != generation:
                raise ValueError("Stale draft proposal generation")
        routes = (
            [(request.peer_id, slot, row) for row in range(len(handles))]
            if slot is not None
            else None
        )
        proposals = await self.adapter.commit(
            handles,
            request.generations,
            request.accepted_counts,
            request.bonus_tokens,
            routes,
        )
        for handle in handles:
            self.generations[handle] += 1
        return proposals

    async def serve_ipc(self) -> None:
        while True:
            for peer_id, peer in list(self.ipc_peers.items()):
                for slot, sequence, metadata in peer.poll():
                    request = ProposalRequest(
                        0,
                        [],
                        peer.k,
                        operation="commit",
                        peer_id=peer_id,
                        handles=metadata[:, 0].tolist(),
                        generations=metadata[:, 1].tolist(),
                        accepted_counts=metadata[:, 2].tolist(),
                        bonus_tokens=metadata[:, 3].tolist(),
                    )
                    proposals = await self.commit(request, slot)
                    if any(len(tokens) != peer.k for tokens in proposals):
                        raise RuntimeError("Incomplete IPC draft proposal")
                    # Collectors finish only after scheduler pause. Publish the
                    # doorbell in the worker after its GPU payload copies.
                    await self.engine.collective_rpc(
                        "publish_remote_draft_ipc", args=(peer_id, slot, sequence)
                    )
            await asyncio.sleep(0)


async def serve_draft_requests(
    engine: DraftEngine,
    port: int,
    ready: Callable[[], None] | None = None,
    backend: str = "stateless",
) -> None:
    socket = zmq.asyncio.Context.instance().socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{port}")
    logger.info("Remote draft service listening on port %d", port)
    if ready is not None:
        ready()
    service: ResumableDraftService | RemoteDraftService
    if backend == "resumable":
        if not isinstance(engine, AsyncLLM):
            raise TypeError("Resumable drafting requires an AsyncLLM engine")
        service = ResumableDraftService(engine)
    else:
        service = RemoteDraftService(engine)

    async def serve_control() -> None:
        while True:
            identity, *frames = await socket.recv_multipart()
            request, tokens = decode_request(frames)
            if isinstance(service, ResumableDraftService):
                proposals, generations = await service.handle(request, tokens)
            else:
                proposals = await service.propose(request, tokens)
                generations = []
            lengths = [len(proposal) for proposal in proposals]
            flat = np.asarray(
                [token for proposal in proposals for token in proposal],
                dtype=np.int32,
            )
            response = ProposalResponse(request.request_id, lengths, generations)
            await socket.send_multipart(
                [identity, *encode_response(response, flat)], copy=False
            )

    tasks = [asyncio.create_task(serve_control())]
    if isinstance(service, ResumableDraftService):
        tasks.append(asyncio.create_task(service.serve_ipc()))
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        socket.close(linger=0)


def run_draft_server(args) -> None:
    async def run() -> None:
        if args.draft_server_backend == "resumable":
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            args.async_scheduling = False
        engine_args = AsyncEngineArgs.from_cli_args(args)
        config = engine_args.create_engine_config()
        if args.draft_server_backend == "resumable" and (
            config.model_config.is_hybrid
            or config.model_config.is_attention_free
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError(
                "Resumable drafting requires full attention at TP=1, PP=1."
            )
        engine = AsyncLLM.from_vllm_config(
            config,
            disable_log_stats=engine_args.disable_log_stats,
            enable_log_requests=engine_args.enable_log_requests,
        )
        try:
            await serve_draft_requests(
                engine, args.draft_server_port, backend=args.draft_server_backend
            )
        finally:
            engine.shutdown()

    uvloop.run(run())
