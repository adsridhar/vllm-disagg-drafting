#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start the Milestone B pair with persistent sessions and print completion TPS.

Run from the configured environment:
    .venv/bin/python benchmarks/benchmark_milestone_b.py

Target uses GPUs 0,1; draft uses GPU 2. Logs go to benchmark-results/milestone_b.
TPS includes prefill and HTTP overhead, but excludes server startup.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import requests


def wait_until_ready(proc, log_path: Path, port: int | None, timeout: float) -> None:
    """Wait for draft's listening message or target's HTTP health endpoint."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited ({proc.returncode}); see {log_path}")
        if port is None:
            if "Remote draft service listening on port" in log_path.read_text():
                return
        else:
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).ok:
                    return
            except requests.RequestException:
                pass
        time.sleep(1)
    raise TimeoutError(f"Server startup timed out; see {log_path}")


def stop_server(proc) -> None:
    """Stop the server and its engine/worker children."""
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt", default="Explain speculative decoding briefly.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--flash-attn-version", type=int, default=3)
    parser.add_argument("--target-port", type=int, default=8000)
    parser.add_argument("--draft-port", type=int, default=50051)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument(
        "--log-dir", type=Path, default=Path("benchmark-results/milestone_b")
    )
    parser.add_argument(
        "--cuda-setup", type=Path, default=root.parent / "scripts/cuda_setup.sh"
    )
    args = parser.parse_args()
    # Match the benchmark's CUDA toolkit and library environment.
    setup = subprocess.run(
        ["bash", "-c", 'source "$1" && env -0', "bash", str(args.cuda_setup)],
        check=True,
        stdout=subprocess.PIPE,
    )
    for entry in setup.stdout.split(b"\0"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            os.environ[os.fsdecode(key)] = os.fsdecode(value)
    vllm = str(root / ".venv/bin/vllm")
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # IPC peers need the same visible-device numbering. Explicit IDs select
    # target GPUs 0,1 and draft GPU 2 while GPU 3 stays hidden.
    draft_env = os.environ.copy()
    draft_env["CUDA_VISIBLE_DEVICES"] = "4,5,6"
    draft_env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    # This greedy, single-request demo does not need FlashInfer sampling.
    draft_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    target_env = os.environ.copy()
    target_env.update(
        CUDA_VISIBLE_DEVICES="4,5,6",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_DISAGG_TRANSPORT="cuda_ipc",
    )
    target_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    draft_command = [
        vllm,
        "serve",
        args.draft_model,
        "--draft-server",
        "--draft-server-backend",
        "resumable",
        "--device-ids",
        "2",
        "--no-async-scheduling",
        "--draft-server-port",
        str(args.draft_port),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
        "--enable-prefix-caching",
        "--max-num-seqs",
        "1",
    ]
    config = {
        "method": "draft_model",
        "model": args.draft_model,
        "draft_sample_method": "greedy",
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_tensor_parallel_size": 1,
        "disagg_draft_backend": "resumable",
        "disagg_draft_address": f"tcp://127.0.0.1:{args.draft_port}",
        "disagg_draft_timeout_ms": 30000,
    }
    target_command = [
        vllm,
        "serve",
        args.target_model,
        "--port",
        str(args.target_port),
        "--device-ids",
        "0,1",
        "--tensor-parallel-size",
        "2",
        "--no-async-scheduling",
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        args.dtype,
        "--max-num-seqs",
        "1",
        "--speculative-config",
        json.dumps(config),
    ]
    for command in (draft_command, target_command):
        command.extend(
            [
                "--attention-backend",
                "FLASH_ATTN",
                f"--attention-config.flash_attn_version={args.flash_attn_version}",
            ]
        )

    # ExitStack stops target before draft, including on request/startup errors.
    with ExitStack() as stack:
        for name, command, env, port in (
            ("draft", draft_command, draft_env, None),
            ("target", target_command, target_env, args.target_port),
        ):
            log_path = args.log_dir / f"{name}.log"
            log_file = stack.enter_context(log_path.open("w"))
            print(f"Starting {name}; log: {log_path}", flush=True)
            proc = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            stack.callback(stop_server, proc)
            wait_until_ready(proc, log_path, port, args.startup_timeout)

        start = time.perf_counter()
        response = requests.post(
            f"http://127.0.0.1:{args.target_port}/v1/completions",
            json={
                "model": args.target_model,
                "prompt": args.prompt,
                "temperature": 0,
                "max_tokens": args.max_tokens,
            },
            timeout=args.request_timeout,
        )
        response.raise_for_status()
        result = response.json()
        elapsed = time.perf_counter() - start
        tokens = result["usage"]["completion_tokens"]
        print("\nCompletion:\n" + result["choices"][0]["text"])
        print(f"\n{tokens} completion tokens in {elapsed:.3f}s")
        print(f"TPS (request wall time): {tokens / elapsed:.2f}")


if __name__ == "__main__":
    main()
