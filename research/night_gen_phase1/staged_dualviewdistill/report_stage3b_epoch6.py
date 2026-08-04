#!/usr/bin/env python3
"""Write the Branch G Stage 3B epoch-6 review report."""

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
    path = Path(eval_dir) / "eval_by_condition.json"
    data = json.loads(path.read_text())
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


def read_neutrality(path):
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def read_monitor_events(path):
    if not path or not Path(path).exists():
        return []
    events = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "terminate":
            events.append(event)
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--trajectory-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--inference-weights", required=True)
    parser.add_argument("--neutrality-json")
    parser.add_argument("--monitor-jsonl")
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
    monitor_events = read_monitor_events(args.monitor_jsonl)
    hard_pathology = any(not math.isfinite(event["loss"]) for event in events)

    payload = {
        "stage": args.stage,
        "run_dir": args.run_dir,
        "weights": args.weights,
        "inference_weights": args.inference_weights,
        "metrics": metrics,
        "split_counts": split_counts,
        "primary_gate_warning_status": gates,
        "trajectory": trajectory,
        "neutrality": neutrality,
        "monitor_terminations": monitor_events,
    }
    Path(args.trajectory_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Branch G Stage 3B Epoch-6 Review",
        "",
        "Verdict: HALT_FOR_USER_REVIEW. Epoch 6 is a warning gate only; no mAP-based auto-kill is applied.",
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
            "## Primary Gate Snapshot",
            "",
            "| Gate | Threshold | Epoch-6 value | Status |",
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
                "- The full-val eval used the zero-loss aux eval config and a checkpoint stripped of frozen DINOv2 teacher weights.",
            ]
        )
    else:
        lines.append("- Not run.")
    lines.extend(
        [
            "",
            "## Operational Notes",
            "",
        ]
    )
    if monitor_events:
        for event in monitor_events:
            lines.append(
                f"- Monitor termination event: `{event.get('reason')}` at unix time `{event.get('time')}` "
                f"for pid `{event.get('pid')}`."
            )
        lines.append("- Training was resumed from the latest complete checkpoint before writing this report.")
    else:
        lines.append("- No monitor termination events were recorded.")
    lines.extend(
        [
            "",
            "## Next",
            "",
            "Halt here for user review before any epoch-12 continuation. Do not advance to Stage 3C.",
            "",
        ]
    )
    Path(args.out_md).write_text("\n".join(lines))
    print(args.out_md)


if __name__ == "__main__":
    main()
