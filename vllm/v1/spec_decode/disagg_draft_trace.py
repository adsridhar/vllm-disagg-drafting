# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in structured diagnostics for disaggregated drafting.

CUDA timings synchronize their streams and are intended for correctness runs,
not performance measurements.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRACE_ENABLED = os.environ.get("VLLM_DISAGG_DRAFT_TRACE", "0") == "1"


def trace_enabled() -> bool:
    return _TRACE_ENABLED


def trace_event(component: str, event: str, **fields: Any) -> None:
    if not _TRACE_ENABLED:
        return
    payload = {
        "component": component,
        "event": event,
        **fields,
    }
    logger.info("DISAGG_DRAFT_TRACE %s", json.dumps(payload, sort_keys=True))


def start_cuda_timer() -> torch.cuda.Event | None:
    if not _TRACE_ENABLED or not torch.cuda.is_available():
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def finish_cuda_timer(start: torch.cuda.Event | None) -> float | None:
    if start is None:
        return None
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)
