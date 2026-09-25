# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared multimodal pilot types and checks for disaggregated speculation."""

from typing import Any

import msgspec
import torch

from vllm.multimodal import MULTIMODAL_REGISTRY


class DisaggMultimodalRequest(msgspec.Struct):
    """Model-neutral image request processed independently by the draft."""

    prompt: str
    images: list[torch.Tensor]
    processor_kwargs: dict[str, Any]
    request_id: str = ""


def validate_pilot_compatibility(target_config: Any, draft_config: Any) -> bool:
    """Return whether this is multimodal, or fail for an incompatible pair."""
    target_mm = MULTIMODAL_REGISTRY.supports_multimodal_inputs(target_config)
    if target_mm and not MULTIMODAL_REGISTRY.supports_multimodal_inputs(draft_config):
        raise RuntimeError(
            "Disaggregated multimodal pilot requires a multimodal draft model"
        )
    return target_mm
