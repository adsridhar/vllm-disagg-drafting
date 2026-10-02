# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import multiprocessing as mp
import os
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import RequestOutputKind
from vllm.v1.spec_decode.remote_connector import RemoteDraftConnector
from vllm.v1.spec_decode.remote_draft_service import serve_draft_requests
from vllm.v1.worker.gpu.spec_decode import remote_speculator as remote_module
from vllm.v1.worker.gpu.spec_decode.remote_speculator import RemoteDraftSpeculator


class FakeDraftEngine:
    async def generate(self, prompt, sampling_params, request_id):
        prefix = prompt["prompt_token_ids"]
        assert sampling_params.temperature == 0
        assert sampling_params.ignore_eos
        assert not sampling_params.detokenize
        assert sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
        await asyncio.sleep(0.01 if prefix[-1] == 12 else 0)
        token_ids = [prefix[-1] + i + 1 for i in range(sampling_params.max_tokens)]
        yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids)])


def get_open_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_connector_to_draft_engine_e2e():
    port = get_open_port()
    ready = asyncio.Event()
    server = asyncio.create_task(
        serve_draft_requests(FakeDraftEngine(), port, ready.set)
    )
    await ready.wait()
    connector = RemoteDraftConnector(f"tcp://127.0.0.1:{port}", 1000)
    try:
        proposals = await asyncio.to_thread(
            connector.propose,
            [[11, 12], [21, 22, 23]],
            3,
        )
        assert proposals == [[13, 14, 15], [24, 25, 26]]
    finally:
        connector.close()
        server.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server


def test_remote_speculator_uses_authoritative_prefix(monkeypatch):
    class FakeGroup:
        def broadcast(self, tensor, src=0):
            return tensor

    class FakeConnector:
        def propose(self, prefixes, num_speculative_tokens):
            assert prefixes == [[1, 2, 3], [4, 5]]
            assert num_speculative_tokens == 3
            return [[31, 32, 33], [41, 42]]

    monkeypatch.setattr(remote_module, "get_tp_group", lambda: FakeGroup())
    speculator = RemoteDraftSpeculator.__new__(RemoteDraftSpeculator)
    speculator.backend = "stateless"
    speculator.device = torch.device("cpu")
    speculator.num_speculative_tokens = 3
    speculator.draft_max_model_len = 32
    speculator.connector = FakeConnector()
    speculator.req_states = SimpleNamespace(
        total_len=SimpleNamespace(gpu=torch.tensor([3, 2, 3])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3], [4, 5, 0], [6, 7, 8]])
        ),
        max_seq_len=[32, 32, 32],
    )
    input_batch = SimpleNamespace(
        num_reqs=3,
        idx_mapping=torch.tensor([0, 1, 2]),
        req_ids=["a", "b", "chunked-prefill"],
    )

    empty = torch.empty(0)
    proposals = speculator.propose(
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=empty,
        aux_hidden_states=None,
        num_sampled=torch.tensor([1, 1, 0]),
        num_rejected=empty,
        last_sampled=empty,
        next_prefill_tokens=empty,
        temperature=empty,
        seeds=empty,
    )

    assert proposals.tolist() == [
        [31, 32, 33],
        [41, 42, -1],
        [-1, -1, -1],
    ]


def test_external_commits_retain_request_and_private_cache():
    """Rejected outputs never become shared prefixes or require worker re-admission."""
    from tests.v1.core.test_worker_slot_overflow import (
        WorkerSlots,
        _model_runner_output,
    )
    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.v1.request import RequestStatus

    scheduler = create_scheduler(
        max_num_seqs=1,
        enable_prefix_caching=True,
        use_v2_model_runner=True,
    )
    (request,) = create_requests(num_requests=1, num_tokens=15, max_tokens=3)
    request.resumable = request.external_draft = True
    request.sampling_params.extra_args = {"external_draft": True}
    scheduler.add_request(request)
    slots = WorkerSlots(1)
    prefix = list(request.prompt_token_ids)
    for generation in range(20):
        for token in [21, 22, 23]:
            output = scheduler.schedule()
            slots.apply(output)
            scheduler.update_from_output(
                output, _model_runner_output([request.request_id], [[token]])
            )
        assert request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        assert request.num_computed_tokens == len(prefix) + 2
        for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
            assert (
                manager.num_cached_block.get(request.request_id, 0)
                <= len(prefix) // manager.block_size
            )
        accepted = generation % 4
        bonus = 100 + generation
        scheduler.commit_external_verification(
            [(request.request_id, generation, accepted, bonus, None)]
        )
        prefix.extend([21, 22, 23][:accepted] + [bonus])
        assert list(request.all_token_ids) == prefix
        assert request.num_computed_tokens == len(prefix) - (2 if accepted == 3 else 1)
    scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
    slots.apply(scheduler.schedule())
    assert not slots.occupied


def _persistent_client(connection, address, transport):
    """Keep the CUDA IPC producer in a separate process from the service."""
    from vllm.v1.spec_decode.remote_connector import ResumableDraftConnector

    connector = ResumableDraftConnector(address, 30000)
    handles = [10, 20]
    if transport == "cuda_ipc":
        connector.establish_ipc(torch.device("cuda:0"), 2, 3)
    while True:
        operation, *args = connection.recv()
        if operation == "open":
            proposals = connector.open_requests(handles, args[0], 3)
        elif operation == "commit":
            if transport == "cuda_ipc":
                pending = connector.dispatch(
                    handles,
                    torch.tensor(args[0], device="cuda:0"),
                    torch.tensor(args[1], device="cuda:0"),
                )
                proposals = connector.await_proposals(handles, pending).cpu().tolist()
            else:
                proposals = connector.commit_and_propose(handles, *args)
        else:
            connector.close()
            connection.send(None)
            break
        connection.send(proposals)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["zmq", "cuda_ipc"])
@pytest.mark.skipif(
    not os.environ.get("REMOTE_DRAFT_E2E_MODEL"),
    reason="Set REMOTE_DRAFT_E2E_MODEL to a full-attention model; requires two GPUs",
)
async def test_resumable_draft_matches_stateless_generation(transport, monkeypatch):
    """Real engine commits and IPC slot reuse preserve ordinary greedy proposals."""
    from vllm.config import AttentionConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.spec_decode.remote_draft_service import RemoteDraftService
    from vllm.v1.spec_decode.remote_protocol import ProposalRequest

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=os.environ["REMOTE_DRAFT_E2E_MODEL"],
            device_ids=[1],
            dtype="float32",
            attention_config=AttentionConfig(backend="TRITON_ATTN"),
            max_model_len=512,
            max_num_seqs=3,
            max_num_batched_tokens=512,
            kv_cache_memory_bytes=256 * 1024**2,
            enforce_eager=True,
            async_scheduling=False,
            enable_prefix_caching=True,
            disable_log_stats=True,
        )
    )
    ready = asyncio.Event()
    port = get_open_port()
    server = asyncio.create_task(
        serve_draft_requests(engine, port, ready.set, "resumable")
    )
    await ready.wait()
    context = mp.get_context("spawn")
    connection, child_connection = context.Pipe()
    client = context.Process(
        target=_persistent_client,
        args=(child_connection, f"tcp://127.0.0.1:{port}", transport),
    )
    client.start()
    child_connection.close()
    reference = RemoteDraftService(engine)
    tokenizer = engine.get_tokenizer()
    prefixes = [
        tokenizer.encode(text)
        for text in (
            "Explain how speculative decoding works in one paragraph.",
            "Write a short Python function that adds two numbers.",
        )
    ]
    try:
        connection.send(("open", prefixes))
        proposals = await asyncio.to_thread(connection.recv)
        for generation in range(20):
            expected = await reference.propose(
                ProposalRequest(generation, [len(p) for p in prefixes], 4),
                np.asarray([t for p in prefixes for t in p], dtype="int32"),
            )
            assert proposals == [tokens[:3] for tokens in expected], (
                transport,
                generation,
            )
            counts = [generation % 4, (generation + 1) % 4]
            bonuses = [tokens[count] for tokens, count in zip(expected, counts)]
            if generation % 2:
                # Force a different continuation so stale rejected KV cannot
                # pass simply by replaying the draft's original next token.
                bonuses = [tokenizer.encode(" However")[0]] * len(prefixes)
            for prefix, proposal, count, bonus in zip(
                prefixes, proposals, counts, bonuses
            ):
                prefix.extend(proposal[:count] + [bonus])
            connection.send(("commit", counts, bonuses))
            proposals = await asyncio.to_thread(connection.recv)
        connection.send(("close",))
        await asyncio.to_thread(connection.recv)
        assert not engine.output_processor.has_unfinished_requests()
    finally:
        if client.is_alive():
            client.terminate()
        await asyncio.to_thread(client.join)
        connection.close()
        server.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server
        engine.shutdown()
