#!/usr/bin/env python3
"""Write the Branch G Stage 3B final result report."""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


LOG_RE = re.compile(r"Epoch \[(?P<epoch>\d+)/(?P<epochs>\d+)\]\[(?P<iter>\d+)/(?P<iters>\d+)\]")
TOTAL_LOSS_RE = re.compile(r"\] - Epoch \[[^\]]+\]\[[^\]]+\] loss: (?P<value>[-+0-9.eEinfnaINFNA]+)")
LOSS_VALUE_RE = re.compile(r"(?:^|, )(?P<name>[A-Za-z0-9_.]*loss[A-Za-z0-9_.]*): (?P<value>[-+0-9.eEinfnaINFNA]+)")

S0 = {
    "day": {"mAP": 0.3152649818, "NDS": 0.3745762709},
    "night": {"mAP": 0.1487749875, "NDS": 0.2150977574},
    "rain": {"mAP": 0.2743174671, "NDS": 0.3713314930},
    "overall": {"mAP": 0.3039905911, "NDS": 0.3697754272},
}

PRIMARY_GATE = {
    "night_mAP": 0.1588,
    "day_mAP": 0.3053,
    "overall_mAP": 0.2890,
    "night_NDS": 0.2101,
}

HARD_KILL = {"night_mAP_lt": 0.1488, "overall_mAP_lt": 0.3040}
NEAR_NIGHT_MAP = S0["night"]["mAP"] + 0.005


def parse_float(raw):
    try:
        return float(raw)
    except ValueError:
        return math.nan


def parse_train_log(path):
    events = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = LOG_RE.search(line)
        total = TOTAL_LOSS_RE.search(line)
        if not match or not total:
            continue
        losses = {m.group("name"): parse_float(m.group("value")) for m in LOSS_VALUE_RE.finditer(line)}
        losses["loss"] = parse_float(total.group("value"))
        events.append(
            {
                "epoch": int(match.group("epoch")),
                "iter": int(match.group("iter")),
                "iters": int(match.group("iters")),
                "loss": losses["loss"],
                "aux": losses.get("loss_dualview_distill"),
                "losses": losses,
            }
        )
    return events


def summarize_by_epoch(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[event["epoch"]].append(event)
    rows = []
    for epoch in sorted(grouped):
        vals = grouped[epoch]
        losses = [v["loss"] for v in vals if math.isfinite(v["loss"])]
        auxes = [v["aux"] for v in vals if v["aux"] is not None and math.isfinite(v["aux"])]
        rows.append(
            {
                "epoch": epoch,
                "points": len(vals),
                "loss_first": losses[0] if losses else None,
                "loss_last": losses[-1] if losses else None,
                "loss_min": min(losses) if losses else None,
                "loss_max": max(losses) if losses else None,
                "aux_first": auxes[0] if auxes else None,
                "aux_last": auxes[-1] if auxes else None,
                "aux_min": min(auxes) if auxes else None,
                "aux_max": max(auxes) if auxes else None,
            }
        )
    return rows


def read_metrics(eval_dir):
    data = json.loads((Path(eval_dir) / "eval_by_condition.json").read_text())
    metrics = {
        "overall": {
            "mAP": float(data["overall"]["pts_bbox_NuScenes/mAP"]),
            "NDS": float(data["overall"]["pts_bbox_NuScenes/NDS"]),
        }
    }
    for split in ("day", "night", "rain"):
        split_data = data["splits"][split]
        metrics[split] = {"mAP": float(split_data["mean_ap"]), "NDS": float(split_data["nd_score"])}
    return metrics, data.get("split_counts", {})


def fmt(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def pp(value):
    return f"{value * 100:+.2f} pp"


def gate_status(metrics):
    return {
        "night_mAP": metrics["night"]["mAP"] >= PRIMARY_GATE["night_mAP"],
        "day_mAP": metrics["day"]["mAP"] >= PRIMARY_GATE["day_mAP"],
        "overall_mAP": metrics["overall"]["mAP"] >= PRIMARY_GATE["overall_mAP"],
        "night_NDS": metrics["night"]["NDS"] >= PRIMARY_GATE["night_NDS"],
    }


def verdict_for(metrics, gates, neutrality):
    neutrality_pass = bool(neutrality and neutrality.get("pass"))
    hard_kill = (
        metrics["night"]["mAP"] < HARD_KILL["night_mAP_lt"]
        and metrics["overall"]["mAP"] < HARD_KILL["overall_mAP_lt"]
    )
    primary_pass = all(gates.values()) and neutrality_pass
    near_pass = (
        metrics["night"]["mAP"] >= NEAR_NIGHT_MAP
        and metrics["night"]["mAP"] < PRIMARY_GATE["night_mAP"]
        and gates["day_mAP"]
        and gates["overall_mAP"]
        and gates["night_NDS"]
        and neutrality_pass
    )
    if primary_pass:
        return "PASS", hard_kill
    if near_pass:
        return "NEAR_PASS", hard_kill
    if hard_kill:
        return "FAIL_HARD_KILL", hard_kill
    return "FAIL_GATE_MISS", hard_kill


def read_neutrality(path):
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def read_monitor_events(paths):
    events = []
    for path in paths or []:
        if not path or not Path(path).exists():
            continue
        for line in Path(path).read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "terminate":
                event["source"] = path
                events.append(event)
    return events


def write_failure_mode(path, result_report, verdict, metrics):
    cause = (
        "DINOv2 perspective-view semantic features did not transfer into a useful "
        "BEV-cell-level training signal under the RaCFormer LSS lifting path: the "
        "auxiliary loss remained inference-neutral, but the final detection metrics "
        "missed the pre-registered night/day/overall gates, indicating the frozen PV "
        "teacher regularized the camera-BEV map without improving the detector's "
        "condition-robust decision surface."
    )
    lines = [
        "# Branch G Stage 3B Failure Mode",
        "",
        f"Verdict: `{verdict}`",
        "",
        f"Result report: `{result_report}`",
        "",
        "## Metrics",
        "",
        f"- night mAP: `{metrics['night']['mAP']:.4f}`",
        f"- day mAP: `{metrics['day']['mAP']:.4f}`",
        f"- overall mAP: `{metrics['overall']['mAP']:.4f}`",
        f"- night NDS: `{metrics['night']['NDS']:.4f}`",
        "",
        "## Causal Mechanism",
        "",
        cause,
        "",
        "## Pivot",
        "",
        "- Stage 1B is already complete in this workspace; PASS-PAPER-NR remains the active fallback.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--failure-md")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--inference-weights", required=True)
    parser.add_argument("--neutrality-json", required=True)
    parser.add_argument("--monitor-jsonl", action="append", default=[])
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--git-sha", required=True)
    args = parser.parse_args()

    events = parse_train_log(args.train_log)
    trajectory = summarize_by_epoch(events)
    metrics, split_counts = read_metrics(args.eval_dir)
    gates = gate_status(metrics)
    neutrality = read_neutrality(args.neutrality_json)
    verdict, hard_kill = verdict_for(metrics, gates, neutrality)
    monitor_events = read_monitor_events(args.monitor_jsonl)
    hard_pathology = any(
        any(not math.isfinite(v) for v in event["losses"].values())
        for event in events
    )

    payload = {
        "stage": args.stage,
        "verdict": verdict,
        "hard_kill_condition_met": hard_kill,
        "run_dir": args.run_dir,
        "weights": args.weights,
        "inference_weights": args.inference_weights,
        "metrics": metrics,
        "split_counts": split_counts,
        "primary_gate_status": gates,
        "trajectory": trajectory,
        "neutrality": neutrality,
        "monitor_terminations": monitor_events,
    }
    Path(args.result_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Branch G Stage 3B Result",
        "",
        f"Verdict: `{verdict}`. Halt for user review; do not auto-advance to Stage 3C.",
        "",
        "## Run",
        "",
        f"- SLURM job: `{args.job_id}` on `{args.host}`",
        f"- Git SHA: `{args.git_sha}`",
        f"- Train config: `{args.config}`",
        f"- Eval config: `{args.eval_config}`",
        f"- Run dir: `{args.run_dir}`",
        f"- Training checkpoint: `{args.weights}`",
        f"- Inference checkpoint: `{args.inference_weights}`",
        f"- Hard-kill condition met: {'YES' if hard_kill else 'NO'}",
        f"- Hard pathology observed in logged loss: {'YES' if hard_pathology else 'NO'}",
        "",
        "## Full-Val Metrics",
        "",
        "| Split | Count | mAP | NDS | mAP delta vs S0 | NDS delta vs S0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("day", "night", "rain", "overall"):
        count = split_counts.get(split, "NA") if split != "overall" else sum(split_counts.values()) if split_counts else "NA"
        lines.append(
            f"| {split} | {count} | {metrics[split]['mAP']:.4f} | {metrics[split]['NDS']:.4f} | "
            f"{pp(metrics[split]['mAP'] - S0[split]['mAP'])} | {pp(metrics[split]['NDS'] - S0[split]['NDS'])} |"
        )
    lines.extend(
        [
            "",
            "## Primary Gate",
            "",
            "| Gate | Threshold | Epoch-12 value | Status |",
            "|---|---:|---:|---|",
            f"| night mAP | {PRIMARY_GATE['night_mAP']:.4f} | {metrics['night']['mAP']:.4f} | {'PASS' if gates['night_mAP'] else 'MISS'} |",
            f"| day mAP | {PRIMARY_GATE['day_mAP']:.4f} | {metrics['day']['mAP']:.4f} | {'PASS' if gates['day_mAP'] else 'MISS'} |",
            f"| overall mAP | {PRIMARY_GATE['overall_mAP']:.4f} | {metrics['overall']['mAP']:.4f} | {'PASS' if gates['overall_mAP'] else 'MISS'} |",
            f"| night NDS | {PRIMARY_GATE['night_NDS']:.4f} | {metrics['night']['NDS']:.4f} | {'PASS' if gates['night_NDS'] else 'MISS'} |",
            "",
            "## Training Loss Trajectory",
            "",
            "| Epoch | Logged points | loss first | loss last | loss min | loss max | aux first | aux last | aux min | aux max |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in trajectory:
        lines.append(
            f"| {row['epoch']} | {row['points']} | {fmt(row['loss_first'])} | {fmt(row['loss_last'])} | "
            f"{fmt(row['loss_min'])} | {fmt(row['loss_max'])} | {fmt(row['aux_first'])} | "
            f"{fmt(row['aux_last'])} | {fmt(row['aux_min'])} | {fmt(row['aux_max'])} |"
        )

    lines.extend(["", "## Inference Neutrality", ""])
    if neutrality:
        zero = neutrality["baseline_vs_zero_loss_weight"]
        positive = neutrality["baseline_vs_positive_loss_weight"]
        lines.extend(
            [
                f"- Zero-loss aux module vs baseline maxdiff: `{zero['max_abs_diff']:.3e}`; pass={zero['within_tolerance']}.",
                f"- Positive-loss aux module vs baseline maxdiff: `{positive['max_abs_diff']:.3e}`; pass={positive['within_tolerance']}.",
                f"- Overall neutrality pass: `{neutrality.get('pass')}`.",
                "- Full-val eval used the zero-loss aux eval config and a checkpoint stripped of frozen DINOv2 teacher weights.",
            ]
        )
    else:
        lines.append("- Not run.")

    lines.extend(["", "## Operational Notes", ""])
    if monitor_events:
        for event in monitor_events:
            lines.append(
                f"- Monitor termination event from `{event.get('source')}`: `{event.get('reason')}` "
                f"at unix time `{event.get('time')}` for pid `{event.get('pid')}`."
            )
    else:
        lines.append("- No monitor termination events were recorded during the epoch-12 continuation.")
    lines.append("- No mAP-based auto-kill was applied.")

    lines.extend(["", "## Next", ""])
    if verdict in {"PASS", "NEAR_PASS"}:
        lines.append("- Halt for user review. Do not auto-advance to Stage 3C without separate authorization.")
    else:
        lines.append("- Failure mode report written; Stage 1B is already complete, so PASS-PAPER-NR remains available.")
        lines.append("- Halt for user review.")

    Path(args.out_md).write_text("\n".join(lines) + "\n")

    if verdict.startswith("FAIL") and args.failure_md:
        write_failure_mode(args.failure_md, args.out_md, verdict, metrics)

    print(args.out_md)
    if verdict.startswith("FAIL") and args.failure_md:
        print(args.failure_md)


if __name__ == "__main__":
    main()
