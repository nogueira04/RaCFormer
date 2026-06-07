# pyright: reportMissingImports=false
"""Summarize RaCFormer Orin eval result directories without reading predictions.

The Orin experiments leave each run under ``eval_results/<run_name>/`` with a
small ``metrics.json`` and ``evaluation_report.txt`` plus large prediction
artifacts. This script intentionally reads only the small files and writes a
CSV/JSON manifest suitable for reproduction auditing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FIELDS = [
    "run",
    "mAP",
    "NDS",
    "mean_inference_ms",
    "fps",
    "num_samples",
    "timestamp",
    "config",
    "weights",
    "has_metrics_json",
    "has_evaluation_report",
]


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) else None


def _metric(data: dict[str, Any], name: str) -> float | int | None:
    value = _number(data.get(name))
    if value is not None:
        return value
    for container in ["metrics", "timing_stats", "summary"]:
        nested = data.get(container)
        if isinstance(nested, dict):
            value = _number(nested.get(name))
            if value is not None:
                return value
    return None


def _parse_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    report = path.read_text(errors="replace")
    patterns = {
        "timestamp": r"Timestamp:\s*(\S+)",
        "config": r"Config:\s*(.+)",
        "weights": r"Weights:\s*(.+)",
        "mAP": r"mAP:\s*([0-9.]+)",
        "NDS": r"NDS:\s*([0-9.]+)",
        "mean_inference_ms": r"Mean inference time:\s*([0-9.]+)\s*ms",
        "fps": r"FPS:\s*([0-9.]+)",
        "num_samples": r"Samples processed:\s*([0-9]+)",
    }
    parsed: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, report)
        if not match:
            continue
        value = match.group(1).strip()
        if key == "num_samples":
            parsed[key] = int(value)
        elif key in {"mAP", "NDS", "mean_inference_ms", "fps"}:
            parsed[key] = float(value)
        else:
            parsed[key] = value
    return parsed


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "evaluation_report.txt"
    row: dict[str, Any] = {
        "run": run_dir.name,
        "has_metrics_json": metrics_path.exists(),
        "has_evaluation_report": report_path.exists(),
    }

    report = _parse_report(report_path)
    row.update(report)

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        for key in ["mAP", "NDS", "mean_inference_ms", "fps", "num_samples"]:
            value = _metric(metrics, key)
            if value is not None:
                row[key] = value

    for key in FIELDS:
        row.setdefault(key, None)
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_root", type=Path, nargs="?", default=Path("eval_results"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.eval_root.iterdir() if path.is_dir())
    rows = [_summarize_run(path) for path in run_dirs]

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    if args.csv_out:
        _write_csv(rows, args.csv_out)

    print("run,mAP,NDS,mean_inference_ms,num_samples,config,weights")
    for row in rows:
        print(
            f"{row['run']},{row['mAP']},{row['NDS']},{row['mean_inference_ms']},"
            f"{row['num_samples']},{row['config']},{row['weights']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
