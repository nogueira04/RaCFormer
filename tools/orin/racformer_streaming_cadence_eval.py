# pyright: reportMissingImports=false
"""Run a synthetic fixed-cadence RaCFormer streaming timing benchmark.

This is a timing harness, not a real-sensor benchmark. It reuses the normal
RaCFormer validation setup, installs the no-skip fast runtime patches, then
feeds samples through the model at a fixed synthetic cadence and records per
sample timing. It does not claim live camera/radar capture, sensor sync, or
real-sensor mAP.

Run from the RaCFormer repository root, for example:

PYTHONPATH=. NUSCENES_VERSION=v1.0-mini \
  python tools/racformer_streaming_cadence_eval.py \
    --config configs/orin/racformer_f2_3layer_q900_thrnone_mini.py \
    --weights checkpoints/racformer_r50_f8_f2_1layer_adapted.pth \
    --output_dir eval_results/streaming_synthetic_5hz \
    --max_vis_samples 0 \
    --fp16 \
    --trt-backbone racformer_backbone_neck.trt \
    --stream-period-ms 200 \
    --stream-samples 1500 \
    --stream-preload-samples 16
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import val
import racformer_noskip_fast_eval as fast_eval


@dataclass
class StreamConfig:
    period_ms: float
    samples: int
    preload_samples: int
    output_dir: str | None


STREAM_CONFIG: StreamConfig | None = None


def _cli_value(args: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _parse_stream_args() -> StreamConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stream-period-ms", type=float, default=200.0)
    parser.add_argument("--stream-samples", type=int, default=81)
    parser.add_argument("--stream-preload-samples", type=int, default=0)
    stream_args, remaining = parser.parse_known_args()
    output_dir = _cli_value(remaining, "--output_dir")
    sys.argv = [sys.argv[0]] + remaining
    return StreamConfig(
        period_ms=stream_args.stream_period_ms,
        samples=stream_args.stream_samples,
        preload_samples=stream_args.stream_preload_samples,
        output_dir=output_dir,
    )


def _prepare_data(sample: Any, model: torch.nn.Module) -> dict | None:
    data = fast_eval._unwrap_sample(sample)
    if data is None:
        return None
    for key in ["img_metas", "img", "radar_points", "radar_depth", "radar_rcs", "gt_depth"]:
        if key in data and not isinstance(data[key], list):
            data[key] = [data[key]]
    device = next((model.module if hasattr(model, "module") else model).parameters()).device
    return val.move_to_device(data, device)


def _iter_samples(data_loader: Any):
    while True:
        for batch in data_loader:
            for sample in batch:
                yield sample


def _preload_samples(data_loader: Any, count: int) -> list[Any]:
    if count <= 0:
        return []
    samples = []
    for sample in _iter_samples(data_loader):
        samples.append(copy.deepcopy(sample))
        if len(samples) >= count:
            return samples
    return samples


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _stream_classification(metrics: dict[str, Any]) -> str:
    period = float(metrics["target_period_ms"])
    if metrics["deadline_miss_count"] == 0 and metrics["max_end_to_end_ms"] <= period:
        return "synthetic_5hz_hard_timing_pass"
    if metrics["mean_end_to_end_ms"] <= period and metrics["p95_end_to_end_ms"] <= period:
        return "synthetic_5hz_soft_timing_pass"
    if metrics["mean_model_inference_ms"] <= period:
        return "model_mean_pass_streaming_deadline_fail"
    return "synthetic_5hz_timing_fail"


def _write_stream_outputs(events: list[dict[str, Any]], metrics: dict[str, Any], output_dir: str | None) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)

    events_path = os.path.join(output_dir, "streaming_events.csv")
    if events:
        with open(events_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            writer.writeheader()
            writer.writerows(events)

    metrics_path = os.path.join(output_dir, "streaming_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def streaming_single_gpu_test(model, data_loader, show=False, out_dir=None, autocast_dtype=None):
    if STREAM_CONFIG is None:
        raise RuntimeError("STREAM_CONFIG was not initialized")

    model.eval()
    period_s = STREAM_CONFIG.period_ms / 1000.0
    total_samples = STREAM_CONFIG.samples
    preloaded = _preload_samples(data_loader, STREAM_CONFIG.preload_samples)
    sample_iter = _iter_samples(data_loader)

    events: list[dict[str, Any]] = []
    inference_times: list[float] = []
    end_to_end_times: list[float] = []
    gpu_ready_times: list[float] = []
    preprocess_times: list[float] = []
    postprocess_times: list[float] = []
    queue_delays: list[float] = []
    late_by_times: list[float] = []
    deadline_misses = 0
    queue_depth_max = 0

    # Warm up once before the synthetic stream starts.
    warmup_source = copy.deepcopy(preloaded[0]) if preloaded else next(sample_iter)
    warmup_data = _prepare_data(warmup_source, model)
    if warmup_data is None:
        raise RuntimeError("Could not unwrap warmup sample")
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext():
            _ = model(return_loss=False, rescale=True, **warmup_data)
    torch.cuda.synchronize()

    start_origin = time.perf_counter()
    progress = tqdm(total=total_samples)

    for index in range(total_samples):
        scheduled_ts = start_origin + index * period_s
        now = time.perf_counter()
        if now < scheduled_ts:
            time.sleep(scheduled_ts - now)
        host_receive_ts = time.perf_counter()

        queue_delay_ms = max(0.0, (host_receive_ts - scheduled_ts) * 1000.0)
        queue_depth_est = max(0, int(math.floor(queue_delay_ms / STREAM_CONFIG.period_ms)))
        queue_depth_max = max(queue_depth_max, queue_depth_est)

        preprocess_start = time.perf_counter()
        if preloaded:
            source = copy.deepcopy(preloaded[index % len(preloaded)])
        else:
            source = next(sample_iter)
        data = _prepare_data(source, model)
        if data is None:
            continue
        torch.cuda.synchronize()
        gpu_ready_ts = time.perf_counter()

        with torch.inference_mode():
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.cuda.amp.autocast(dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext():
                _ = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            inference_end = time.perf_counter()

        # No real output transport exists in synthetic mode. Treat object creation
        # completion as postprocess/publish for deadline accounting.
        postprocess_end = time.perf_counter()
        publish_ts = postprocess_end

        preprocess_ms = (gpu_ready_ts - preprocess_start) * 1000.0
        inference_ms = (inference_end - inference_start) * 1000.0
        postprocess_ms = (postprocess_end - inference_end) * 1000.0
        gpu_ready_to_publish_ms = (publish_ts - gpu_ready_ts) * 1000.0
        end_to_end_ms = (publish_ts - scheduled_ts) * 1000.0
        late_by_ms = max(0.0, publish_ts - (scheduled_ts + period_s)) * 1000.0
        deadline_miss = end_to_end_ms > STREAM_CONFIG.period_ms
        if deadline_miss:
            deadline_misses += 1

        inference_times.append(inference_ms)
        end_to_end_times.append(end_to_end_ms)
        gpu_ready_times.append(gpu_ready_to_publish_ms)
        preprocess_times.append(preprocess_ms)
        postprocess_times.append(postprocess_ms)
        queue_delays.append(queue_delay_ms)
        late_by_times.append(late_by_ms)

        events.append(
            {
                "index": index,
                "synthetic_sensor_ts_rel_ms": (scheduled_ts - start_origin) * 1000.0,
                "host_receive_ts_rel_ms": (host_receive_ts - start_origin) * 1000.0,
                "queue_delay_ms": queue_delay_ms,
                "queue_depth_est": queue_depth_est,
                "preprocess_gpu_ready_ms": preprocess_ms,
                "model_inference_ms": inference_ms,
                "postprocess_publish_ms": postprocess_ms,
                "gpu_ready_to_publish_ms": gpu_ready_to_publish_ms,
                "synthetic_end_to_end_ms": end_to_end_ms,
                "late_by_ms": late_by_ms,
                "deadline_miss": int(deadline_miss),
            }
        )
        progress.update(1)

    progress.close()

    wall_time_s = time.perf_counter() - start_origin
    metrics: dict[str, Any] = {
        "mode": "synthetic_cadence",
        "real_sensor_claim": False,
        "deployment_classification": "inconclusive_real_sensor_absent",
        "config": asdict(STREAM_CONFIG),
        "target_period_ms": STREAM_CONFIG.period_ms,
        "num_samples": len(inference_times),
        "wall_time_s": wall_time_s,
        "effective_output_fps": float(len(inference_times) / wall_time_s) if wall_time_s > 0 else 0.0,
        "mean_model_inference_ms": float(np.mean(inference_times)) if inference_times else 0.0,
        "median_model_inference_ms": float(np.median(inference_times)) if inference_times else 0.0,
        "p90_model_inference_ms": _percentile(inference_times, 90),
        "p95_model_inference_ms": _percentile(inference_times, 95),
        "p99_model_inference_ms": _percentile(inference_times, 99),
        "max_model_inference_ms": float(np.max(inference_times)) if inference_times else 0.0,
        "mean_preprocess_gpu_ready_ms": float(np.mean(preprocess_times)) if preprocess_times else 0.0,
        "p95_preprocess_gpu_ready_ms": _percentile(preprocess_times, 95),
        "max_preprocess_gpu_ready_ms": float(np.max(preprocess_times)) if preprocess_times else 0.0,
        "mean_end_to_end_ms": float(np.mean(end_to_end_times)) if end_to_end_times else 0.0,
        "median_end_to_end_ms": float(np.median(end_to_end_times)) if end_to_end_times else 0.0,
        "p90_end_to_end_ms": _percentile(end_to_end_times, 90),
        "p95_end_to_end_ms": _percentile(end_to_end_times, 95),
        "p99_end_to_end_ms": _percentile(end_to_end_times, 99),
        "max_end_to_end_ms": float(np.max(end_to_end_times)) if end_to_end_times else 0.0,
        "mean_queue_delay_ms": float(np.mean(queue_delays)) if queue_delays else 0.0,
        "p95_queue_delay_ms": _percentile(queue_delays, 95),
        "max_queue_delay_ms": float(np.max(queue_delays)) if queue_delays else 0.0,
        "queue_depth_max_est": queue_depth_max,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_rate": float(deadline_misses / len(inference_times)) if inference_times else 0.0,
        "max_late_by_ms": float(np.max(late_by_times)) if late_by_times else 0.0,
        "accuracy_evaluation": "disabled_for_synthetic_streaming; use prior mini-val metrics for mAP/NDS",
    }
    metrics["timing_classification"] = _stream_classification(metrics)
    _write_stream_outputs(events, metrics, STREAM_CONFIG.output_dir)

    timing_stats = {
        "mean_inference_ms": metrics["mean_model_inference_ms"],
        "std_inference_ms": float(np.std(inference_times)) if inference_times else 0.0,
        "min_inference_ms": float(np.min(inference_times)) if inference_times else 0.0,
        "max_inference_ms": metrics["max_model_inference_ms"],
        "median_inference_ms": metrics["median_model_inference_ms"],
        "fps": float(1000.0 / metrics["mean_model_inference_ms"]) if metrics["mean_model_inference_ms"] > 0 else 0.0,
        "num_samples": len(inference_times),
        "streaming": metrics,
    }
    return [], timing_stats


def streaming_evaluate(dataset, results, epoch):
    return {
        "mAP": 0.0,
        "NDS": 0.0,
        "all_metrics": {},
        "streaming_note": "mAP/NDS evaluation disabled for synthetic cadence timing benchmark",
    }


def main() -> int:
    global STREAM_CONFIG
    STREAM_CONFIG = _parse_stream_args()
    fast_eval.install_patch()
    val.single_gpu_test = streaming_single_gpu_test
    val.evaluate = streaming_evaluate
    val.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
