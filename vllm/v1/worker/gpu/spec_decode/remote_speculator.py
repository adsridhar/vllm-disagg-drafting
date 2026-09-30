# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.v1.spec_decode.remote_connector import RemoteDraftConnector
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.states import RequestState


class RemoteDraftSpeculator(BaseSpeculator):
    """Uses a separate vLLM server to produce stateless draft tokens."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        config = vllm_config.speculative_config
        assert config is not None
        assert config.disagg_draft_address is not None
        self.device = device
        self.num_speculative_tokens = config.num_speculative_tokens
        assert config.draft_model_config is not None
        self.draft_max_model_len = config.draft_model_config.max_model_len
        self.supports_mm_inputs = False
        self.draft_logits = None
        self.req_states: RequestState | None = None
        self.connector = (
            RemoteDraftConnector(
                config.disagg_draft_address, config.disagg_draft_timeout_ms
            )
            if get_tensor_model_parallel_rank() == 0
            else None
        )

    def bind_request_state(self, req_states: RequestState) -> None:
        self.req_states = req_states

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self) -> None:
        pass

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        proposals = torch.full(
            (num_reqs, self.num_speculative_tokens),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        if dummy_run or is_profile:
            return proposals.zero_()

        if self.connector is not None:
            assert self.req_states is not None
            req_indices = input_batch.idx_mapping[:num_reqs].cpu().tolist()
            lengths = self.req_states.total_len.gpu[req_indices].cpu().tolist()
            sampled_counts = num_sampled[:num_reqs].cpu().tolist()
            active_rows = [
                row
                for row, (index, length, sampled) in enumerate(
                    zip(req_indices, lengths, sampled_counts)
                )
                if sampled > 0
                and length + self.num_speculative_tokens
                <= min(self.req_states.max_seq_len[index], self.draft_max_model_len)
            ]
            if active_rows:
                active_indices = [req_indices[row] for row in active_rows]
                max_prefix_len = max(lengths[row] for row in active_rows)
                prefix_matrix = self.req_states.all_token_ids.gpu[
                    active_indices, :max_prefix_len
                ].cpu()
                prefixes = [
                    tokens[: lengths[row]].tolist()
                    for tokens, row in zip(prefix_matrix, active_rows)
                ]
                remote = self.connector.propose(
                    prefixes,
                    self.num_speculative_tokens,
                )
                padded = [
                    token_ids + [-1] * (self.num_speculative_tokens - len(token_ids))
                    for token_ids in remote
                ]
                proposals[active_rows] = torch.tensor(
                    padded, dtype=torch.int64, device=self.device
                )

        return get_tp_group().broadcast(proposals, src=0)
