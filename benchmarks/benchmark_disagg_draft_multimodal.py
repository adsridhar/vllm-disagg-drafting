#!/usr/bin/env python3
"""Benchmark multimodal target-only and disaggregated speculative serving.

The dataset handling mirrors specdec_bench's VideoBench: videos are uniformly
sampled into a single image mosaic, paired with the question, and sent through
the OpenAI chat-completions endpoint. PAI parquet datasets, MVBench JSONL
manifests, and directories of ``video.mp4``/``text.json`` samples are supported.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

THINKING_SUFFIX = (
    "\n\nAnswer the question using the following format:\n\n"
    "<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag."
)
PAI_MOSAIC_CACHE_VERSION = "v1"


@dataclass(frozen=True)
class Sample:
    question_id: int
    question: str
    image_path: Path
    category: str | None = None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    home = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="nvidia/Cosmos3-Super")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--dataset", choices=("pai", "video"), default="pai")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=home / "specdec_bench" / "datasets" / "paibench",
    )
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--category")
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--fan-out", type=int, default=3)
    parser.add_argument("--uniform-fanout", action="store_true")
    parser.add_argument("--jit-fallback", action="store_true")
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--target-devices", default="0,1")
    parser.add_argument("--all-devices", default="0,1,2")
    parser.add_argument("--draft-device", type=int, default=2)
    parser.add_argument("--target-tp", type=int, default=2)
    parser.add_argument("--target-port", type=int, default=8000)
    parser.add_argument("--draft-port", type=int, default=50051)
    parser.add_argument("--draft-metrics-port", type=int, default=51051)
    parser.add_argument("--target-attention-backend", default="FLASH_ATTN")
    parser.add_argument("--draft-attention-backend", default="FLASH_ATTN")
    parser.add_argument("--flash-attn-version", type=int, default=3)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--modes",
        choices=("target", "async", "target,async"),
        default="target,async",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--transport", choices=("zmq", "cuda_ipc"), default="cuda_ipc")
    parser.add_argument(
        "--cuda-setup", type=Path, default=home / "scripts" / "cuda_setup.sh"
    )
    parser.add_argument(
        "--parquet-python",
        type=Path,
        help="Python environment containing pyarrow; auto-detected for PAI",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            root
            / "benchmark-results"
            / f"multimodal_disagg_{time.strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    return parser.parse_args()


def apply_cuda_setup(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CUDA setup script: {path}")
    result = subprocess.run(
        ["bash", "-c", 'source "$1" && env -0', "bash", str(path)],
        check=True,
        stdout=subprocess.PIPE,
    )
    for item in result.stdout.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", maxsplit=1)
            os.environ[os.fsdecode(key)] = os.fsdecode(value)


def _uniform_indices(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    if count == 1:
        return [total // 2]
    return [round((total - 1) * index / (count - 1)) for index in range(count)]


def _extract_frames(video_path: Path, output_dir: Path, count: int) -> list[Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wanted = set(_uniform_indices(total, count))
    frames: list[Path] = []
    frame_index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            path = output_dir / f"frame_{len(frames):05d}.png"
            cv2.imwrite(str(path), frame)
            frames.append(path)
            wanted.remove(frame_index)
        frame_index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames


def _sample_images(frame_dir: Path, output_dir: Path, count: int) -> list[Path]:
    images = sorted(
        path
        for path in frame_dir.iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return [images[index] for index in _uniform_indices(len(images), count)]


def _create_mosaic(images: list[Path], output_path: Path) -> Path:
    from PIL import Image

    columns = max(1, math.ceil(math.sqrt(len(images))))
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * 224, rows * 224))
    for index, path in enumerate(images):
        with Image.open(path) as image:
            tile = image.convert("RGB").resize((224, 224))
        canvas.paste(tile, ((index % columns) * 224, (index // columns) * 224))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def _pai_cache_path(dataset_path: Path, video_path: Path, count: int) -> Path:
    stat = video_path.stat()
    key = json.dumps(
        {
            "version": PAI_MOSAIC_CACHE_VERSION,
            "video_path": str(video_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "num_frames": count,
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(key).hexdigest()
    return (
        dataset_path
        / ".video_bench_cache"
        / "pai_mosaics"
        / PAI_MOSAIC_CACHE_VERSION
        / f"{digest}.png"
    )


def _read_parquet(path: Path, python: Path | None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet

        return parquet.read_table(path).to_pylist()
    except ImportError:
        pass

    candidates = [python] if python is not None else []
    candidates.append(path.parents[3] / ".venv" / "bin" / "python")
    code = (
        "import json,sys; import pyarrow.parquet as pq; "
        "print(json.dumps(pq.read_table(sys.argv[1]).to_pylist()))"
    )
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        result = subprocess.run(
            [str(candidate), "-c", code, str(path)],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    raise RuntimeError(
        "PAI parquet loading requires pyarrow. Pass --parquet-python pointing "
        "to a project .venv that has it installed."
    )


def _build_mosaic(
    media_path: Path,
    cache_path: Path,
    frame_dir: Path,
    count: int,
) -> Path:
    if cache_path.is_file():
        return cache_path
    if media_path.is_dir():
        frames = _sample_images(media_path, frame_dir, count)
    else:
        frames = _extract_frames(media_path, frame_dir, count)
    return _create_mosaic(frames, cache_path)


def load_pai(args: argparse.Namespace, limit: int | None) -> list[Sample]:
    parquet_path = args.dataset_path / "data" / "test-00000-of-00001.parquet"
    rows = _read_parquet(parquet_path, args.parquet_python)
    samples: list[Sample] = []
    media_cache = args.output_dir / "media_cache"
    for row in rows:
        if args.category is not None and row.get("category") != args.category:
            continue
        video_path = args.dataset_path / row["video_path"]
        if not video_path.is_file():
            continue
        reference_cache = _pai_cache_path(
            args.dataset_path, video_path, args.num_frames
        )
        cache_path = (
            reference_cache
            if reference_cache.is_file()
            else media_cache / f"pai_{len(samples):06d}.png"
        )
        image_path = _build_mosaic(
            video_path,
            cache_path,
            media_cache / f"pai_{len(samples):06d}_frames",
            args.num_frames,
        )
        samples.append(
            Sample(len(samples), row["question"], image_path, row.get("category"))
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _find_manifest(path: Path) -> Path | None:
    if path.is_file() and path.suffix == ".jsonl":
        return path
    for name in ("manifest.jsonl", "mvbench_manifest.jsonl"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def load_video(args: argparse.Namespace, limit: int | None) -> list[Sample]:
    samples: list[Sample] = []
    media_cache = args.output_dir / "media_cache"
    manifest = _find_manifest(args.dataset_path)
    if manifest is not None:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        for row in rows:
            category = row.get("task") or row.get("category")
            if args.category is not None and category != args.category:
                continue
            media = row.get("media_path") or row.get("video_path") or row.get("video")
            question = row.get("question")
            if not media or not question:
                continue
            media_path = Path(media)
            if not media_path.is_absolute():
                media_path = manifest.parent / media_path
            if not media_path.exists():
                continue
            sample_dir = media_cache / f"video_{len(samples):06d}"
            image_path = _build_mosaic(
                media_path,
                sample_dir / "mosaic.png",
                sample_dir / "frames",
                args.num_frames,
            )
            samples.append(Sample(len(samples), question, image_path, category))
            if limit is not None and len(samples) >= limit:
                break
        return samples

    for sample_path in sorted(args.dataset_path.iterdir()):
        video_path = sample_path / "video.mp4"
        text_path = sample_path / "text.json"
        if not video_path.is_file() or not text_path.is_file():
            continue
        rows = json.loads(text_path.read_text())
        if isinstance(rows, dict):
            rows = [rows]
        sample_dir = media_cache / sample_path.name
        for row in rows:
            category = row.get("category", sample_path.name)
            if args.category is not None and category != args.category:
                continue
            image_path = _build_mosaic(
                video_path,
                sample_dir / "mosaic.png",
                sample_dir / "frames",
                args.num_frames,
            )
            samples.append(Sample(len(samples), row["question"], image_path, category))
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def load_samples(args: argparse.Namespace) -> tuple[Sample, list[Sample]]:
    limit = args.num_requests + 1 if args.num_requests is not None else None
    samples = (
        load_pai(args, limit) if args.dataset == "pai" else load_video(args, limit)
    )
    if len(samples) < 2:
        raise RuntimeError(
            "The dataset needs at least one warmup and one measured sample"
        )
    measured = samples[1:]
    if args.num_requests is not None:
        measured = measured[: args.num_requests]
    return samples[0], measured


def terminate(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def start_process(
    command: list[str], env: dict[str, str], log_path: Path
) -> tuple[subprocess.Popen[str], Any]:
    log_file = log_path.open("w", encoding="utf-8")
    print(f"Starting: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, log_file


def wait_for_health(
    proc: subprocess.Popen[str],
    port: int,
    expected_model: str,
    timeout: float,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited; see {log_path}")
        try:
            response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            served_models = response.json().get("data", []) if response.ok else []
            if any(model.get("id") == expected_model for model in served_models):
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server startup timed out; see {log_path}")


def wait_for_log(
    proc: subprocess.Popen[str], log_path: Path, marker: str, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Draft server exited; see {log_path}")
        if marker in log_path.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(2)
    raise TimeoutError(f"Draft server startup timed out; see {log_path}")


def target_command(args: argparse.Namespace, vllm: Path, async_mode: bool) -> list[str]:
    command = [
        str(vllm),
        "serve",
        args.target_model,
        "--port",
        str(args.target_port),
        "--tensor-parallel-size",
        str(args.target_tp),
        "--attention-backend",
        args.target_attention_backend,
        "--max-num-seqs",
        "1",
    ]
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.target_attention_backend.upper() == "FLASH_ATTN":
        command.append(
            f"--attention-config.flash_attn_version={args.flash_attn_version}"
        )
    if async_mode:
        config = {
            "model": args.draft_model,
            "method": "draft_model",
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
            "draft_sample_method": "greedy",
            "attention_backend": args.draft_attention_backend,
            "disagg_draft_addresses": [f"tcp://127.0.0.1:{args.draft_port}"],
            "disagg_fan_out": args.fan_out,
        }
        command.extend(["--speculative-config", json.dumps(config)])
    return command


def draft_command(args: argparse.Namespace, vllm: Path) -> list[str]:
    config = {
        "model": args.draft_model,
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_sample_method": "greedy",
        "attention_backend": args.draft_attention_backend,
        "disagg_fan_out": args.fan_out,
    }
    command = [
        str(vllm),
        "serve",
        args.draft_model,
        "--draft-server",
        "--draft-server-device",
        str(args.draft_device),
        "--draft-server-port",
        str(args.draft_port),
        "--attention-backend",
        args.draft_attention_backend,
        "--max-num-seqs",
        "1",
        "--speculative-config",
        json.dumps(config),
    ]
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.draft_attention_backend.upper() == "FLASH_ATTN":
        command.append(
            f"--attention-config.flash_attn_version={args.flash_attn_version}"
        )
    return command


def scrape_metrics(port: int) -> dict[str, float]:
    response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=30)
    response.raise_for_status()
    metrics: dict[str, float] = {}
    for line in response.text.splitlines():
        fields = line.split()
        if line.startswith("#") or len(fields) != 2:
            continue
        try:
            name = fields[0].split("{", maxsplit=1)[0]
            metrics[name] = metrics.get(name, 0.0) + float(fields[1])
        except ValueError:
            pass
    return metrics


def _counter(metrics: dict[str, float], name: str) -> float:
    return metrics.get(name, 0.0)


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def run_workload(
    args: argparse.Namespace, warmup: Sample, samples: list[Sample], label: str
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    model_response = requests.get(
        f"http://127.0.0.1:{args.target_port}/v1/models",
        timeout=args.request_timeout,
    )
    model_response.raise_for_status()
    model = model_response.json()["data"][0]["id"]

    def request(sample: Sample) -> tuple[dict[str, Any], float]:
        prompt = sample.question + (THINKING_SUFFIX if args.thinking else "")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(sample.image_path)},
                        },
                    ],
                }
            ],
            "max_tokens": args.output_len,
            "temperature": 0.0,
            "return_token_ids": True,
        }
        start = time.perf_counter()
        response = requests.post(
            f"http://127.0.0.1:{args.target_port}/v1/chat/completions",
            json=payload,
            timeout=args.request_timeout,
        )
        elapsed = time.perf_counter() - start
        if not response.ok:
            raise RuntimeError(
                f"Chat completion failed ({response.status_code}): {response.text}"
            )
        return response.json(), elapsed

    print(f"[{label}] warmup", flush=True)
    request(warmup)
    before = scrape_metrics(args.target_port)
    outputs: list[list[int]] = []
    captures: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    start = time.perf_counter()
    for index, sample in tqdm(enumerate(samples, start=1)):
        response, latency = request(sample)
        token_ids = response["choices"][0].get("token_ids")
        if token_ids is None:
            raise RuntimeError("Server omitted token IDs")
        outputs.append(token_ids)
        usage = response.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens", 0))
        completion_tokens += int(usage.get("completion_tokens", len(token_ids)))
        captures.append(
            {
                "question_id": sample.question_id,
                "category": sample.category,
                "question": sample.question,
                "image_path": str(sample.image_path),
                "latency_seconds": latency,
                "response": response,
            }
        )
        print(f"[{label}] request {index}/{len(samples)}", flush=True)
    elapsed = time.perf_counter() - start
    after = scrape_metrics(args.target_port)
    requested = _counter(after, "vllm:spec_decode_num_draft_tokens_total") - _counter(
        before, "vllm:spec_decode_num_draft_tokens_total"
    )
    accepted = _counter(after, "vllm:spec_decode_num_accepted_tokens_total") - _counter(
        before, "vllm:spec_decode_num_accepted_tokens_total"
    )
    rounds = _counter(after, "vllm:spec_decode_num_drafts_total") - _counter(
        before, "vllm:spec_decode_num_drafts_total"
    )
    metrics = {
        "requests": len(samples),
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_tokens_per_second": completion_tokens / elapsed,
        "total_tokens_per_second": (prompt_tokens + completion_tokens) / elapsed,
        "mean_request_latency_seconds": sum(
            capture["latency_seconds"] for capture in captures
        )
        / len(captures),
        "draft_tokens_requested": requested,
        "draft_tokens_accepted": accepted,
        "draft_rounds": rounds,
        "draft_acceptance_rate": accepted / requested if requested else 0.0,
        "average_draft_tokens_accepted_per_step": (
            accepted / rounds if rounds else 0.0
        ),
    }
    return outputs, captures, metrics


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def load_trace_events(path: Path) -> list[dict[str, Any]]:
    marker = "DISAGG_DRAFT_TRACE "
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        with contextlib.suppress(json.JSONDecodeError):
            events.append(json.loads(line.split(marker, maxsplit=1)[1]))
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


def collect_trace_metrics(target_log: Path, draft_log: Path) -> dict[str, Any]:
    target_events = load_trace_events(target_log)
    overlap_events = [
        event
        for event in target_events
        if event.get("event") == "multimodal_prefill_overlap"
    ][1:]
    result: dict[str, Any] = {}
    for field in ("prefill_latency_ms", "readiness_wait_ms"):
        values = [float(event[field]) for event in overlap_events if field in event]
        if values:
            result[f"multimodal_{field}_mean"] = sum(values) / len(values)

    verification_modes = {
        event["execution_mode"]
        for event in target_events
        if event.get("event") == "verification" and "execution_mode" in event
    }
    if verification_modes:
        result["target_execution_modes"] = sorted(verification_modes)

    draft_modes = [
        event
        for event in load_trace_events(draft_log)
        if event.get("event") == "execution_mode"
    ]
    if draft_modes:
        result["draft_execution_mode"] = draft_modes[-1]
    return result


def run_target(
    args: argparse.Namespace,
    vllm: Path,
    warmup: Sample,
    samples: list[Sample],
    label: str,
) -> tuple[list[list[int]], list[dict[str, Any]], dict[str, Any]]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.target_devices
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_DISAGG_DRAFT_TRACE"] = str(int(args.trace))
    env["VLLM_DISAGG_TRANSPORT"] = args.transport
    env["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "0"
    env["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "0"

    log_path = args.output_dir / f"{label}_target.log"
    process = None
    log_file = None
    try:
        process, log_file = start_process(
            target_command(args, vllm, label == "async"), env, log_path
        )
        wait_for_health(
            process,
            args.target_port,
            args.target_model,
            args.startup_timeout,
            log_path,
        )
        return run_workload(args, warmup, samples, label)
    finally:
        terminate(process)
        if log_file is not None:
            log_file.close()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_cuda_setup(args.cuda_setup)
    vllm = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "vllm"
    if not vllm.is_file():
        raise FileNotFoundError(f"Missing vLLM executable: {vllm}")
    warmup, samples = load_samples(args)
    write_json(args.output_dir / "configuration.json", vars(args))
    write_json(
        args.output_dir / "dataset.json",
        [
            {
                "question_id": sample.question_id,
                "category": sample.category,
                "question": sample.question,
                "image_path": str(sample.image_path),
            }
            for sample in samples
        ],
    )
    print(f"Loaded {len(samples)} measured samples plus one warmup", flush=True)
    summary: dict[str, Any] = {"configuration": vars(args)}
    target_outputs: list[list[int]] | None = None
    if args.modes in ("target", "target,async"):
        target_outputs, responses, metrics = run_target(
            args, vllm, warmup, samples, "target"
        )
        summary["target"] = metrics
        write_json(args.output_dir / "target_token_ids.json", target_outputs)
        write_json(args.output_dir / "target_responses.json", responses)
        write_json(args.output_dir / "summary.json", summary)

    if args.modes in ("async", "target,async"):
        draft_env = os.environ.copy()
        draft_env["CUDA_VISIBLE_DEVICES"] = args.all_devices
        draft_env["DRAFT_METRICS_PORT"] = str(args.draft_metrics_port)
        draft_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        draft_env["VLLM_DISAGG_DRAFT_TRACE"] = str(int(args.trace))
        draft_env["VLLM_DISAGG_TRANSPORT"] = args.transport
        draft_log = args.output_dir / "draft.log"
        draft_proc = None
        draft_file = None
        try:
            draft_proc, draft_file = start_process(
                draft_command(args, vllm), draft_env, draft_log
            )
            wait_for_log(
                draft_proc,
                draft_log,
                "Draft model loaded, starting server loop.",
                args.startup_timeout,
            )
            async_outputs, responses, metrics = run_target(
                args, vllm, warmup, samples, "async"
            )
            if args.trace:
                metrics.update(
                    collect_trace_metrics(
                        args.output_dir / "async_target.log", draft_log
                    )
                )
                pipeline_records, pipeline_summary = collect_pipeline_timings(
                    args.output_dir / "async_target.log", draft_log
                )
                if pipeline_summary:
                    metrics["pipeline_timings_ms"] = pipeline_summary
                    write_json(
                        args.output_dir / "pipeline_timings.json",
                        pipeline_records,
                    )
            draft_metrics = scrape_metrics(args.draft_metrics_port)
            metrics["draft_cache_hit_rate"] = _counter(
                draft_metrics, "vllm:draft_server_cache_hit_rate"
            )
            summary["async"] = metrics
            write_json(args.output_dir / "async_token_ids.json", async_outputs)
            write_json(args.output_dir / "async_responses.json", responses)
            if target_outputs is not None:
                mismatches = [
                    index
                    for index, (target, async_output) in enumerate(
                        zip(target_outputs, async_outputs)
                    )
                    if target != async_output
                ]
                summary["parity"] = {
                    "exact": not mismatches,
                    "mismatch_count": len(mismatches),
                    "mismatch_indices": mismatches,
                }
                summary["speedup"] = (
                    metrics["decode_tokens_per_second"]
                    / summary["target"]["decode_tokens_per_second"]
                )
        finally:
            terminate(draft_proc)
            if draft_file is not None:
                draft_file.close()

    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return int(summary.get("parity", {}).get("exact") is False)


if __name__ == "__main__":
    raise SystemExit(main())
