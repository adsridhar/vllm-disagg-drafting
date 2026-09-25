#!/usr/bin/env python3
"""Capture aligned target and draft torch traces for one text request."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import benchmark_disagg_draft_qwen3 as benchmark
import requests


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    home = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--target-revision", default=benchmark.TARGET_REVISION)
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--draft-revision", default=benchmark.DRAFT_REVISION)
    parser.add_argument(
        "--prompt",
        default=(
            "Explain how speculative decoding improves inference throughput "
            "and describe its main synchronization points."
        ),
    )
    parser.add_argument(
        "--warmup-prompt",
        default="List the first ten prime numbers and briefly explain primality.",
    )
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--fan-out", type=int, default=3)
    parser.add_argument("--uniform-fanout", action="store_true")
    parser.add_argument("--jit-fallback", action="store_true")
    parser.add_argument("--target-devices", default="0,1")
    parser.add_argument("--all-devices", default="0,1,2")
    parser.add_argument("--draft-device", type=int, default=2)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--target-port", type=int, default=8000)
    parser.add_argument("--draft-port", type=int, default=50051)
    parser.add_argument("--draft-metrics-port", type=int, default=51051)
    parser.add_argument("--target-attention-backend", default="FLASH_ATTN")
    parser.add_argument("--draft-attention-backend", default="FLASH_ATTN")
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--flash-attn-version", type=int, default=3)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--with-stack", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--transport", choices=("zmq", "cuda_ipc"), default="cuda_ipc")
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--cuda-setup", type=Path, default=home / "scripts" / "cuda_setup.sh"
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path("/home/scratch.adsridhar_other") / ".cache" / "huggingface",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "benchmark-results"
            / f"qwen3_disagg_profile_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    return parser.parse_args()


def post(port: int, path: str, timeout: float) -> None:
    response = requests.post(f"http://127.0.0.1:{port}{path}", timeout=timeout)
    response.raise_for_status()


def request(
    args: argparse.Namespace, model: str, prompt: str, max_tokens: int
) -> dict[str, Any]:
    response = requests.post(
        f"http://127.0.0.1:{args.target_port}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "return_token_ids": True,
        },
        timeout=args.request_timeout,
    )
    if not response.ok:
        raise RuntimeError(
            f"Completion failed ({response.status_code}): {response.text}"
        )
    return response.json()


def trace_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and (path.name.endswith(".pt.trace.json") or path.name.endswith(".json.gz"))
    )


def wait_for_traces(directory: Path, timeout: float) -> list[Path]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        paths = trace_files(directory)
        if paths:
            return paths
        time.sleep(1)
    raise TimeoutError(f"No profiler traces appeared under {directory}")


def read_trace(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as trace_file:
        return json.load(trace_file)


def merge_traces(paths: list[Path], output_path: Path) -> None:
    events: list[dict[str, Any]] = []
    next_pid = 1
    for path in paths:
        trace = read_trace(path)
        source_events = (
            trace.get("traceEvents", []) if isinstance(trace, dict) else trace
        )
        pid_map: dict[str, int] = {}
        for original in source_events:
            event = dict(original)
            if "pid" in event:
                pid_key = repr(event["pid"])
                if pid_key not in pid_map:
                    pid_map[pid_key] = next_pid
                    next_pid += 1
                event["pid"] = pid_map[pid_key]
            if event.get("ph") == "M" and event.get("name") == "process_name":
                event["args"] = dict(event.get("args", {}))
                name = event["args"].get("name", "process")
                event["args"]["name"] = f"{path.parent.name}/{path.stem}: {name}"
            events.append(event)
    output_path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark.apply_cuda_setup(args.cuda_setup)
    os.environ.setdefault("HF_HOME", str(args.hf_cache))

    root = Path(__file__).resolve().parents[1]
    vllm = root / ".venv" / "bin" / "vllm"
    target_traces = args.output_dir / "target_traces"
    draft_traces = args.output_dir / "draft_traces"
    target_traces.mkdir()
    draft_traces.mkdir()

    target_profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(target_traces.resolve()),
        "torch_profiler_with_stack": args.with_stack,
        "torch_profiler_use_gzip": False,
    }
    target_cmd = benchmark.target_command(args, vllm, "disagg")
    target_cmd.extend(["--profiler-config", json.dumps(target_profiler_config)])

    draft_env = os.environ.copy()
    draft_env["CUDA_VISIBLE_DEVICES"] = args.all_devices
    draft_env["DRAFT_METRICS_PORT"] = str(args.draft_metrics_port)
    draft_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    draft_env["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
    draft_env["VLLM_DISAGG_TRANSPORT"] = args.transport
    draft_env["VLLM_DRAFT_TORCH_PROFILER_DIR"] = str(draft_traces.resolve())
    draft_env["VLLM_DRAFT_TORCH_PROFILER_WITH_STACK"] = str(int(args.with_stack))

    target_env = os.environ.copy()
    target_env["CUDA_VISIBLE_DEVICES"] = args.target_devices
    target_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    target_env["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
    target_env["VLLM_DISAGG_TRANSPORT"] = args.transport

    draft_proc = target_proc = None
    draft_log_file = target_log_file = None
    target_profile_active = False
    draft_log = args.output_dir / "draft.log"
    target_log = args.output_dir / "target.log"
    try:
        draft_proc, draft_log_file = benchmark.start_process(
            benchmark.draft_command(args, vllm), draft_env, draft_log
        )
        benchmark.wait_for_log(
            draft_proc,
            draft_log,
            "Draft model loaded, starting server loop.",
            args.startup_timeout,
        )
        target_proc, target_log_file = benchmark.start_process(
            target_cmd, target_env, target_log
        )
        benchmark.wait_for_health(
            target_proc, args.target_port, args.startup_timeout, target_log
        )
        model = benchmark.get_served_model(args.target_port, args.request_timeout)

        print("Running unprofiled warmup", flush=True)
        request(args, model, args.warmup_prompt, args.output_len)
        post(args.target_port, "/start_profile", 120)
        target_profile_active = True
        os.killpg(draft_proc.pid, signal.SIGUSR1)
        benchmark.wait_for_log(
            draft_proc,
            draft_log,
            "DraftServer profile started",
            120,
        )

        print("Capturing measured request", flush=True)
        started_at = time.time_ns()
        response = request(args, model, args.prompt, args.output_len)
        finished_at = time.time_ns()

        post(args.target_port, "/stop_profile", 300)
        target_profile_active = False
        os.killpg(draft_proc.pid, signal.SIGUSR2)
        benchmark.wait_for_log(
            draft_proc,
            draft_log,
            "DraftServer profile written to",
            300,
        )
        target_paths = wait_for_traces(target_traces, 120)
        draft_paths = wait_for_traces(draft_traces, 120)
        combined_trace = args.output_dir / "combined_trace.json"
        merge_traces(target_paths + draft_paths, combined_trace)
        benchmark.write_json(args.output_dir / "response.json", response)
        benchmark.write_json(
            args.output_dir / "profile_manifest.json",
            {
                "request_started_at_ns": started_at,
                "request_finished_at_ns": finished_at,
                "request_latency_ms": (finished_at - started_at) / 1e6,
                "target_traces": [str(path) for path in target_paths],
                "draft_traces": [str(path) for path in draft_paths],
                "combined_trace": str(combined_trace),
                "configuration": vars(args),
            },
        )
        print(f"Combined trace: {combined_trace}", flush=True)
    finally:
        if target_profile_active and target_proc is not None:
            with contextlib.suppress(Exception):
                post(args.target_port, "/stop_profile", 60)
        if draft_proc is not None and draft_proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(draft_proc.pid, signal.SIGUSR2)
        benchmark.terminate(target_proc)
        benchmark.terminate(draft_proc)
        if target_log_file is not None:
            target_log_file.close()
        if draft_log_file is not None:
            draft_log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
