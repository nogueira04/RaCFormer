#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Audit RaCFormer Orin reproduction coverage.

This script separates coverage from metric matching. Use it to verify that a
results root contains every rerunnable row in ``expected_metrics_full.json`` and
to keep the archival February rows explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED = Path("../RaCFormer_artifacts/orin/expected_metrics_full.json")

ARCHIVAL_LEGACY_ROWS = {
    "4layers_trt_accel_persample",
    "4layers_trt_accelerate",
    "tier1_f8",
    "tier2_4l4f",
}

LOWER_CONFIDENCE_RERUNNABLE_ROWS = {
    "f4_keyframe_add3_33_rm70_topk500_runtime_fastprop_bf16_mini",
    "f4_keyframe_add3_33_rm70_topk500_runtime_precomp_bf16_mini",
    "f4_keyframe_add3_33_rm70_topk500_runtime_precomp_clocks_bf16_mini",
}


def _load_runs(expected_path: Path) -> dict[str, dict[str, Any]]:
    expected_doc = json.loads(expected_path.read_text())
    runs = expected_doc.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"{expected_path} does not contain a top-level 'runs' object")
    return runs


def _metric_dirs(results_root: Path) -> set[str]:
    if not results_root.exists():
        return set()
    return {path.parent.name for path in results_root.glob("**/metrics.json")}


def build_report(expected_path: Path, results_root: Path | None) -> dict[str, Any]:
    runs = _load_runs(expected_path)
    expected_names = set(runs)
    archival = sorted(expected_names & ARCHIVAL_LEGACY_ROWS)
    rerunnable = sorted(expected_names - ARCHIVAL_LEGACY_ROWS)
    lower_confidence = sorted(expected_names & LOWER_CONFIDENCE_RERUNNABLE_ROWS)

    report: dict[str, Any] = {
        "expected_path": str(expected_path),
        "expected_total": len(expected_names),
        "rerunnable_total": len(rerunnable),
        "archival_legacy_total": len(archival),
        "archival_legacy_rows": archival,
        "lower_confidence_rerunnable_rows": lower_confidence,
    }

    if results_root is None:
        report["status"] = "no_results_root"
        return report

    present = _metric_dirs(results_root)
    missing_rerunnable = sorted(set(rerunnable) - present)
    present_archival = sorted(set(archival) & present)
    extra_metric_dirs = sorted(present - expected_names)

    report.update(
        {
            "results_root": str(results_root),
            "present_metric_dirs": len(present),
            "present_rerunnable": len(set(rerunnable) & present),
            "missing_rerunnable": missing_rerunnable,
            "present_archival_legacy": present_archival,
            "extra_metric_dirs": extra_metric_dirs,
            "status": "rerunnable_complete" if not missing_rerunnable else "missing_rerunnable",
        }
    )
    return report


def print_text(report: dict[str, Any]) -> None:
    print(f"expected_path={report['expected_path']}")
    print(f"expected_total={report['expected_total']}")
    print(f"rerunnable_total={report['rerunnable_total']}")
    print(f"archival_legacy_total={report['archival_legacy_total']}")
    print("archival_legacy_rows=" + ",".join(report["archival_legacy_rows"]))
    if report["lower_confidence_rerunnable_rows"]:
        print("lower_confidence_rerunnable_rows=" + ",".join(report["lower_confidence_rerunnable_rows"]))

    if report["status"] == "no_results_root":
        print("status=no_results_root")
        return

    print(f"results_root={report['results_root']}")
    print(f"present_metric_dirs={report['present_metric_dirs']}")
    print(f"present_rerunnable={report['present_rerunnable']}")
    print(f"status={report['status']}")
    if report["missing_rerunnable"]:
        print("missing_rerunnable:")
        for name in report["missing_rerunnable"]:
            print(f"  {name}")
    if report["present_archival_legacy"]:
        print("present_archival_legacy=" + ",".join(report["present_archival_legacy"]))
    if report["extra_metric_dirs"]:
        print("extra_metric_dirs:")
        for name in report["extra_metric_dirs"]:
            print(f"  {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    report = build_report(args.expected, args.results_root)
    print_text(report)

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.require_complete and report["status"] != "rerunnable_complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
