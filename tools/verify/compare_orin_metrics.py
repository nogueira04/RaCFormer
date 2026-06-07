# pyright: reportMissingImports=false
"""Compare reproduced Orin metrics against the May 2026 evidence table.

The reproduction runner writes one output directory per run. This checker reads
each ``metrics.json`` and compares the values that matter for the
<200 ms / mAP > 0.45 claim.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED = Path("../RaCFormer_artifacts/orin/expected_metrics.json")


def _metric(data: dict[str, Any], name: str) -> float | int | None:
    value = data.get(name)
    if isinstance(value, (int, float)):
        return value

    # Some evaluators place values under a summary object. Keep this permissive
    # so the checker survives small logger changes.
    for container in ["metrics", "timing_stats", "summary"]:
        nested = data.get(container)
        if isinstance(nested, dict) and isinstance(nested.get(name), (int, float)):
            return nested[name]
    return None


def _find_run_dir(results_roots: list[Path], run_name: str) -> Path | None:
    for results_root in results_roots:
        candidates = [
            results_root / run_name,
            results_root / f"{run_name}_mini",
            results_root / f"{run_name}_bf16_mini",
        ]
        for candidate in candidates:
            if (candidate / "metrics.json").exists():
                return candidate

        matches = sorted(results_root.glob(f"**/{run_name}/metrics.json"))
        if matches:
            return matches[0].parent
    return None


def _load_actual(run_dir: Path) -> dict[str, Any]:
    data = json.loads((run_dir / "metrics.json").read_text())
    report_path = run_dir / "evaluation_report.txt"
    if not report_path.exists():
        return data

    report = report_path.read_text()
    patterns = {
        "mean_inference_ms": r"Mean inference time:\s*([0-9.]+)\s*ms",
        "fps": r"FPS:\s*([0-9.]+)",
        "num_samples": r"Samples processed:\s*([0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report)
        if match:
            value = match.group(1)
            data[key] = int(value) if key == "num_samples" else float(value)
    return data


def _status(actual: float | int | None, expected: float | int | None, *, absolute_tol: float, relative_tol: float) -> str:
    if actual is None or expected is None:
        return "missing"
    diff = abs(float(actual) - float(expected))
    allowed = max(absolute_tol, abs(float(expected)) * relative_tol)
    return "ok" if diff <= allowed else "diff"


def _latency_status(actual: float | int | None, expected: float | int | None) -> str:
    if actual is None or expected is None:
        return "missing"
    allowed_slower = max(20.0, abs(float(expected)) * 0.15)
    return "ok" if float(actual) <= float(expected) + allowed_slower else "diff"


def compare(expected_path: Path, results_roots: list[Path], *, exclude_runs: set[str] | None = None) -> int:
    expected_doc = json.loads(expected_path.read_text())
    exclude_runs = exclude_runs or set()
    runs = {
        run_name: expected
        for run_name, expected in expected_doc["runs"].items()
        if run_name not in exclude_runs
    }

    failures = 0
    print(f"expected={expected_path}")
    print(f"results_roots={','.join(str(root) for root in results_roots)}")
    if exclude_runs:
        print(f"excluded={','.join(sorted(exclude_runs))}")
    print("run,mAP,NDS,mean_inference_ms,num_samples,status,metrics_path")

    for run_name, expected in runs.items():
        run_dir = _find_run_dir(results_roots, run_name)
        if run_dir is None:
            failures += 1
            print(f"{run_name},missing,missing,missing,missing,missing,")
            continue

        actual_doc = _load_actual(run_dir)
        statuses = []
        values = {}
        for metric, abs_tol, rel_tol in [
            ("mAP", 0.005, 0.0),
            ("NDS", 0.005, 0.0),
            ("num_samples", 0.0, 0.0),
        ]:
            if metric not in expected:
                values[metric] = _metric(actual_doc, metric)
                continue
            actual = _metric(actual_doc, metric)
            values[metric] = actual
            statuses.append(
                _status(actual, expected.get(metric), absolute_tol=abs_tol, relative_tol=rel_tol)
            )
        values["mean_inference_ms"] = _metric(actual_doc, "mean_inference_ms")
        if "mean_inference_ms" in expected:
            statuses.append(_latency_status(values["mean_inference_ms"], expected.get("mean_inference_ms")))

        status = "ok" if all(item == "ok" for item in statuses) else "diff"
        if status != "ok":
            failures += 1
        print(
            f"{run_name},{values['mAP']},{values['NDS']},{values['mean_inference_ms']},"
            f"{values['num_samples']},{status},{run_dir / 'metrics.json'}"
        )

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument(
        "--exclude-run",
        action="append",
        default=[],
        help="Expected run name to skip. Repeat for archival rows that are not part of the current repro root.",
    )
    parser.add_argument(
        "results_root",
        nargs="+",
        type=Path,
        help="Output root produced by reproduce_200ms_045map.sh. Provide multiple roots to search in order.",
    )
    args = parser.parse_args()
    return compare(args.expected, args.results_root, exclude_runs=set(args.exclude_run))


if __name__ == "__main__":
    raise SystemExit(main())
