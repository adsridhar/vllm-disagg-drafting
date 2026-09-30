# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import socket
from types import SimpleNamespace

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
