#!/usr/bin/env python3
"""Benchmark Qwen3 disaggregated drafting against a greedy baseline.

This reproduces the SSD ``--numseqs 128 --all --b 1 --temp 0`` workload:
128 prompts from each of HumanEval, Alpaca, GSM8K, and UltraFeedback.
Requests are issued sequentially and exact output token IDs are compared.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

TARGET_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
DRAFT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    home = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--target-revision", default=TARGET_REVISION)
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--draft-revision", default=DRAFT_REVISION)
    parser.add_argument("--target-devices", default="0,1")
    parser.add_argument("--all-devices", default="0,1,2")
    parser.add_argument("--draft-device", type=int, default=2)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--target-port", type=int, default=8000)
    parser.add_argument("--draft-port", type=int, default=50051)
    parser.add_argument("--draft-metrics-port", type=int, default=51051)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--fan-out", type=int, default=3)
    parser.add_argument("--uniform-fanout", action="store_true")
    parser.add_argument("--jit-fallback", action="store_true")
    parser.add_argument(
        "--numseqs",
        type=int,
        default=128,
        help="Prompts per dataset; --all produces 4x this value",
    )
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--target-attention-backend", default="FLASH_ATTN")
    parser.add_argument("--draft-attention-backend", default="FLASH_ATTN")
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--flash-attn-version", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--trace", action='store_true')
    parser.add_argument(
        "--cuda-setup",
        type=Path,
        default=home / "scripts" / "cuda_setup.sh",
        help="Shell environment used by the SSD reference runs",
    )
    parser.add_argument(
        "--ssd-base",
        type=Path,
        default=home / "ssd_base",
        help="SSD checkout containing bench_helpers.py and processed datasets",
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
            / f"qwen3_disagg_128_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip target-only; exact parity and speedups are then unavailable",
    )
    parser.add_argument(
        "--run-standard-spec",
        action="store_true",
        help="Also benchmark standard in-process speculative decoding",
    )
    parser.add_argument(
        "--run-mtp",
        action="store_true",
        help="Also benchmark in-process MTP speculative decoding",
    )
    parser.add_argument(
        "--skip-disagg",
        action="store_true",
        help="Skip the disaggregated draft-server benchmark",
    )
    return parser.parse_args()


def snapshot_path(cache: Path, model: str, revision: str) -> str:
    model_dir = cache / "hub" / f"models--{model.replace('/', '--')}"
    candidate = model_dir / "snapshots" / revision
    return str(candidate if (candidate / "config.json").exists() else model)


def apply_cuda_setup(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CUDA setup script: {path}")
    result = subprocess.run(
        ["bash", "-c", 'source "$1" && env -0', "bash", str(path)],
        check=True,
        stdout=subprocess.PIPE,
    )
    for item in result.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", maxsplit=1)
        os.environ[os.fsdecode(key)] = os.fsdecode(value)


def load_prompts(args: argparse.Namespace) -> list[list[int]]:
    bench_dir = args.ssd_base / "bench"
    dataset_dir = args.ssd_base / "datasets"
    if not (bench_dir / "bench_helpers.py").exists():
        raise FileNotFoundError(f"Missing SSD benchmark helpers under {bench_dir}")
    os.environ["SSD_HF_CACHE"] = str(args.hf_cache)
    os.environ["SSD_DATASET_DIR"] = str(dataset_dir)
    sys.path.insert(0, str(bench_dir))
    from bench_helpers import generate_benchmark_inputs

    helper_args = SimpleNamespace(
        example=False,
        random=False,
        all=True,
        humaneval=False,
        alpaca=False,
        c4=False,
        ultrafeedback=False,
        chat_template=False,
        eagle=False,
        numseqs=args.numseqs,
        input_len=args.input_len,
    )
    tokenizer_path = snapshot_path(
        args.hf_cache, args.target_model, args.target_revision
    )
    _, token_ids, _ = generate_benchmark_inputs(helper_args, tokenizer_path)
    if token_ids is None:
        raise RuntimeError("SSD input generation did not return token IDs")
    expected = args.numseqs * 4
    if len(token_ids) != expected:
        raise RuntimeError(f"Expected {expected} prompts, got {len(token_ids)}")
    return token_ids


def terminate(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def wait_for_health(
    proc: subprocess.Popen[str], port: int, timeout: float, log_path: Path
) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited with status {proc.returncode}; see {log_path}"
            )
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}; see {log_path}")


def wait_for_log(
    proc: subprocess.Popen[str],
    log_path: Path,
    marker: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited with status {proc.returncode}; see {log_path}"
            )
        if marker in log_path.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {marker!r}; see {log_path}")


def start_process(
    command: list[str], env: dict[str, str], log_path: Path
) -> tuple[subprocess.Popen[str], Any]:
    log_file = log_path.open("w", encoding="utf-8")
    print(f"Starting: {' '.join(command)}", flush=True)
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, log_file


def target_command(args: argparse.Namespace, vllm_bin: Path, mode: str) -> list[str]:
    command = [
        str(vllm_bin),
        "serve",
        args.target_model,
        "--revision",
        args.target_revision,
        "--port",
        str(args.target_port),
        "--tensor-parallel-size",
        str(args.target_tp),
        "--attention-backend",
        args.target_attention_backend,
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(args.max_model_len),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if reasoning_parser := getattr(args, "reasoning_parser", None):
        command.extend(["--reasoning-parser", reasoning_parser])
    if mode == "mtp":
        config = {
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        command.extend(["--speculative-config", json.dumps(config)])
    elif mode != "baseline":
        config = {
            "model": args.draft_model,
            "revision": args.draft_revision,
            "method": "draft_model",
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": (1 if mode == "disagg" else args.target_tp),
            "draft_sample_method": "greedy",
            "attention_backend": args.draft_attention_backend,
        }
        if mode == "disagg":
            config.update(
                {
                    "disagg_draft_addresses": [f"tcp://127.0.0.1:{args.draft_port}"],
                    "disagg_fan_out": args.fan_out,
                    "disagg_uniform_fanout": args.uniform_fanout,
                    "disagg_jit_fallback": args.jit_fallback,
                }
            )
        command.extend(["--speculative-config", json.dumps(config)])
        if (
            mode == "standard_spec"
            and args.draft_attention_backend.upper() == "FLASH_ATTN"
        ):
            command.append(
                f"--attention-config.flash_attn_version={args.flash_attn_version}"
            )
    return command


def draft_command(args: argparse.Namespace, vllm_bin: Path) -> list[str]:
    config = {
        "model": args.draft_model,
        "revision": args.draft_revision,
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_sample_method": "greedy",
        "attention_backend": args.draft_attention_backend,
        "disagg_fan_out": args.fan_out,
        "disagg_uniform_fanout": args.uniform_fanout,
        "disagg_jit_fallback": args.jit_fallback,
    }
    command = [
        str(vllm_bin),
        "serve",
        args.draft_model,
        "--revision",
        args.draft_revision,
        "--draft-server",
        "--draft-server-device",
        str(args.draft_device),
        "--draft-server-port",
        str(args.draft_port),
        "--attention-backend",
        args.draft_attention_backend,
        "--enforce-eager",  # this does nothing
        "--max-num-seqs",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--speculative-config",
        json.dumps(config),
    ]
    if args.draft_attention_backend.upper() == "FLASH_ATTN":
        command.append(
            f"--attention-config.flash_attn_version={args.flash_attn_version}"
        )
    return command


def get_served_model(port: int, timeout: float) -> str:
    response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=timeout)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def scrape_metrics(port: int) -> dict[str, float]:
    response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=30)
    response.raise_for_status()
    metrics: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 2:
            try:
                name = fields[0].split("{", maxsplit=1)[0]
                metrics[name] = metrics.get(name, 0.0) + float(fields[1])
            except ValueError:
                pass
    return metrics


def counter(metrics: dict[str, float], name: str) -> float:
    return metrics.get(name, 0.0)


def run_workload(
    args: argparse.Namespace,
    prompts: list[list[int]],
    label: str,
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    model = get_served_model(args.target_port, args.request_timeout)
    url = f"http://127.0.0.1:{args.target_port}/v1/completions"

    def request(prompt: list[int]) -> dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "ignore_eos": False,
            "return_token_ids": True,
        }
        response = requests.post(url, json=payload, timeout=args.request_timeout)
        response.raise_for_status()
        return response.json()

    print(f"[{label}] warmup", flush=True)
    request(prompts[0])
    metrics_before = scrape_metrics(args.target_port)
    start = time.perf_counter()
    outputs: list[list[int]] = []
    responses: list[dict[str, Any]] = []
    completion_tokens = 0
    prompt_tokens = 0
    for index, prompt in enumerate(prompts, start=1):
        result = request(prompt)
        responses.append(result)
        choice = result["choices"][0]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise RuntimeError("Server omitted token_ids despite return_token_ids=true")
        outputs.append(token_ids)
        usage = result.get("usage", {})
        prompt_tokens += int(usage.get("prompt_tokens", len(prompt)))
        completion_tokens += int(usage.get("completion_tokens", len(token_ids)))
        if index == 1 or index % 50 == 0 or index == len(prompts):
            print(f"[{label}] request {index}/{len(prompts)}", flush=True)
    elapsed = time.perf_counter() - start
    metrics_after = scrape_metrics(args.target_port)
    result_metrics = {
        "requests": len(prompts),
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens_per_second": completion_tokens / elapsed,
        "total_tokens_per_second": (prompt_tokens + completion_tokens) / elapsed,
        "draft_tokens_requested": counter(
            metrics_after, "vllm:spec_decode_num_draft_tokens_total"
        )
        - counter(metrics_before, "vllm:spec_decode_num_draft_tokens_total"),
        "draft_tokens_accepted": counter(
            metrics_after, "vllm:spec_decode_num_accepted_tokens_total"
        )
        - counter(metrics_before, "vllm:spec_decode_num_accepted_tokens_total"),
        "draft_rounds": counter(metrics_after, "vllm:spec_decode_num_drafts_total")
        - counter(metrics_before, "vllm:spec_decode_num_drafts_total"),
    }
    return outputs, responses, result_metrics


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def load_trace_events(path: Path) -> list[dict[str, Any]]:
    marker = "DISAGG_DRAFT_TRACE "
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        try:
            events.append(json.loads(line.split(marker, maxsplit=1)[1]))
        except json.JSONDecodeError:
            continue
    return events


def summarize_timings(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return ordered[index]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def collect_pipeline_timings(
    target_log: Path,
    draft_log: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float | int]],
]:
    target_events = load_trace_events(target_log)
    draft_events = load_trace_events(draft_log)

    request_ids: list[str] = []
    for event in target_events:
        for request_id in event.get("request_ids", []):
            if request_id not in request_ids:
                request_ids.append(request_id)
    if len(request_ids) < 2:
        return [], {}

    warmup_id = request_ids[0]
    measured_ids = set(request_ids[1:])
    verifications: dict[str, list[float]] = {
        request_id: [] for request_id in measured_ids
    }
    responses: list[tuple[str, float, float]] = []
    warmup_responses = 0
    for event in target_events:
        event_type = event.get("event")
        ids = event.get("request_ids", [])
        if event_type == "verification":
            latency = event.get("verification_latency_ms")
            if latency is None:
                continue
            for request_id in ids:
                if request_id in measured_ids:
                    verifications[request_id].append(float(latency))
        elif event_type == "draft_response":
            wait = event.get("response_wait_ms")
            round_trip = event.get("round_trip_ms")
            for request_id in ids:
                if request_id == warmup_id:
                    warmup_responses += 1
                elif (
                    request_id in measured_ids
                    and wait is not None
                    and round_trip is not None
                ):
                    responses.append((request_id, float(wait), float(round_trip)))

    cache_builds = [
        float(event["cache_build_latency_ms"])
        for event in draft_events
        if event.get("event") == "cache_build"
        and event.get("cache_build_latency_ms") is not None
    ]
    measured_builds = cache_builds[warmup_responses : warmup_responses + len(responses)]

    builds_by_request: dict[str, list[tuple[float, float, float]]] = {
        request_id: [] for request_id in measured_ids
    }
    for (request_id, wait, round_trip), build in zip(responses, measured_builds):
        builds_by_request[request_id].append((build, wait, round_trip))

    records: list[dict[str, Any]] = []
    for request_index, request_id in enumerate(request_ids[1:]):
        request_builds = builds_by_request[request_id]
        request_verifications = verifications[request_id]
        paired_rounds = min(
            len(request_builds) - 1,
            len(request_verifications) - 1,
        )
        for round_index in range(max(paired_rounds, 0)):
            cache_build, _, _ = request_builds[round_index]
            _, response_wait, round_trip = request_builds[round_index + 1]
            verification = request_verifications[round_index + 1]
            records.append(
                {
                    "request_index": request_index,
                    "request_id": request_id,
                    "round_index": round_index,
                    "verification_latency_ms": verification,
                    "response_wait_ms": response_wait,
                    "round_trip_ms": round_trip,
                    "cache_build_latency_ms": cache_build,
                    "pipeline_slack_ms": verification - cache_build,
                }
            )

    if not records:
        return [], {}
    fields = (
        "verification_latency_ms",
        "response_wait_ms",
        "round_trip_ms",
        "cache_build_latency_ms",
        "pipeline_slack_ms",
    )
    summary = {
        field: summarize_timings([float(record[field]) for record in records])
        for field in fields
    }
    return records, summary


def run_target(
    args: argparse.Namespace,
    vllm_bin: Path,
    prompts: list[list[int]],
    mode: str,
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    label = mode
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.target_devices
    env["VLLM_DISAGG_DRAFT_TRACE"] = str(int(args.trace))
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    log_path = args.output_dir / f"{label}_target.log"
    proc = None
    log_file = None
    try:
        proc, log_file = start_process(
            target_command(args, vllm_bin, mode), env, log_path
        )
        wait_for_health(proc, args.target_port, args.startup_timeout, log_path)
        return run_workload(args, prompts, label)
    finally:
        terminate(proc)
        if log_file is not None:
            log_file.close()


def main() -> int:
    args = parse_args()
    apply_cuda_setup(args.cuda_setup)
    root = Path(__file__).resolve().parents[1]
    vllm_bin = root / ".venv" / "bin" / "vllm"
    if not vllm_bin.is_file():
        raise FileNotFoundError(f"Missing {vllm_bin}; install this checkout first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)
    write_json(args.output_dir / "configuration.json", vars(args))
    print(
        f"Loaded {len(prompts)} prompts ({args.numseqs} from each dataset)",
        flush=True,
    )

    baseline_outputs: list[list[int]] | None = None
    summary: dict[str, Any] = {"configuration": vars(args)}
    if not args.skip_baseline:
        baseline_outputs, baseline_responses, baseline_metrics = run_target(
            args, vllm_bin, prompts, mode="baseline"
        )
        summary["baseline"] = baseline_metrics
        write_json(args.output_dir / "baseline_token_ids.json", baseline_outputs)
        write_json(args.output_dir / "baseline_responses.json", baseline_responses)

    if args.run_standard_spec:
        standard_outputs, standard_responses, standard_metrics = run_target(
            args, vllm_bin, prompts, mode="standard_spec"
        )
        summary["standard_spec"] = standard_metrics
        write_json(args.output_dir / "standard_spec_token_ids.json", standard_outputs)
        write_json(args.output_dir / "standard_spec_responses.json", standard_responses)
        if baseline_outputs is not None:
            mismatches = [
                index
                for index, (baseline, standard) in enumerate(
                    zip(baseline_outputs, standard_outputs)
                )
                if baseline != standard
            ]
            summary["standard_spec_parity"] = {
                "exact": not mismatches,
                "mismatch_count": len(mismatches),
                "mismatch_indices": mismatches[:100],
            }
            summary["standard_spec_speedup"] = (
                standard_metrics["decode_tokens_per_second"]
                / summary["baseline"]["decode_tokens_per_second"]
            )

    if args.run_mtp:
        mtp_outputs, mtp_responses, mtp_metrics = run_target(
            args, vllm_bin, prompts, mode="mtp"
        )
        summary["mtp"] = mtp_metrics
        write_json(args.output_dir / "mtp_token_ids.json", mtp_outputs)
        write_json(args.output_dir / "mtp_responses.json", mtp_responses)
        if baseline_outputs is not None:
            mismatches = [
                index
                for index, (baseline, mtp) in enumerate(
                    zip(baseline_outputs, mtp_outputs)
                )
                if baseline != mtp
            ]
            summary["mtp_parity"] = {
                "exact": not mismatches,
                "mismatch_count": len(mismatches),
                "mismatch_indices": mismatches[:100],
            }
            summary["mtp_speedup"] = (
                mtp_metrics["decode_tokens_per_second"]
                / summary["baseline"]["decode_tokens_per_second"]
            )

    if args.skip_disagg:
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, default=str), flush=True)
        parity_failed = summary.get("standard_spec_parity", {}).get("exact") is False
        mtp_parity_failed = summary.get("mtp_parity", {}).get("exact") is False
        return 1 if parity_failed or mtp_parity_failed else 0

    draft_env = os.environ.copy()
    draft_env["CUDA_VISIBLE_DEVICES"] = args.all_devices
    draft_env["DRAFT_METRICS_PORT"] = str(args.draft_metrics_port)
    draft_env["VLLM_DISAGG_DRAFT_TRACE"] = str(int(args.trace))
    draft_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    draft_log = args.output_dir / "draft.log"
    draft_proc = None
    draft_file = None
    try:
        draft_proc, draft_file = start_process(
            draft_command(args, vllm_bin), draft_env, draft_log
        )
        wait_for_log(
            draft_proc,
            draft_log,
            "Draft model loaded, starting server loop.",
            args.startup_timeout,
        )
        scrape_metrics(args.draft_metrics_port)

        disagg_outputs, disagg_responses, disagg_metrics = run_target(
            args, vllm_bin, prompts, mode="disagg"
        )
        pipeline_records, pipeline_summary = collect_pipeline_timings(
            args.output_dir / "disagg_target.log",
            draft_log,
        )
        if pipeline_summary:
            disagg_metrics["pipeline_timings_ms"] = pipeline_summary
            write_json(
                args.output_dir / "pipeline_timings.json",
                pipeline_records,
            )
        draft_metrics = scrape_metrics(args.draft_metrics_port)
        cache_hit_rate = counter(draft_metrics, "vllm:draft_server_cache_hit_rate")
        requested = disagg_metrics["draft_tokens_requested"]
        accepted = disagg_metrics["draft_tokens_accepted"]
        rounds = disagg_metrics["draft_rounds"]
        disagg_metrics["draft_acceptance_rate"] = (
            accepted / requested if requested else 0.0
        )
        disagg_metrics["average_draft_tokens_accepted_per_step"] = (
            accepted / rounds if rounds else 0.0
        )
        disagg_metrics["average_acceptance_length_with_target_token"] = (
            1.0 + accepted / rounds if rounds else 1.0
        )
        disagg_metrics["draft_cache_hit_rate"] = cache_hit_rate
        summary["disagg"] = disagg_metrics
        write_json(args.output_dir / "disagg_token_ids.json", disagg_outputs)
        write_json(args.output_dir / "disagg_responses.json", disagg_responses)

        if baseline_outputs is not None:
            mismatches = [
                index
                for index, (baseline, disagg) in enumerate(
                    zip(baseline_outputs, disagg_outputs)
                )
                if baseline != disagg
            ]
            summary["parity"] = {
                "exact": not mismatches,
                "mismatch_count": len(mismatches),
                "mismatch_indices": mismatches[:100],
            }
            summary["speedup"] = (
                disagg_metrics["decode_tokens_per_second"]
                / summary["baseline"]["decode_tokens_per_second"]
            )
    finally:
        terminate(draft_proc)
        if draft_file is not None:
            draft_file.close()

    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    parity_failed = summary.get("parity", {}).get("exact") is False
    standard_parity_failed = (
        summary.get("standard_spec_parity", {}).get("exact") is False
    )
    mtp_parity_failed = summary.get("mtp_parity", {}).get("exact") is False
    if parity_failed or standard_parity_failed or mtp_parity_failed:
        print("ERROR: greedy token parity failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
