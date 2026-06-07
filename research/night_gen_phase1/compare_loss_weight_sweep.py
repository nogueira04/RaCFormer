import json
from pathlib import Path


ROOT = Path("research/night_gen_phase1/results")
OUT_MD = ROOT / "loss_weight_sweep_summary.md"
OUT_JSON = ROOT / "loss_weight_sweep_summary.json"

SPLITS = ("day", "night", "rain", "overall")
METRICS = ("mAP", "NDS")
BASELINE = "S0"
STAGES = (
    ("S0", "baseline day-only"),
    ("S3_seed20260425_ratio18p75", "generated keyframes, weight 1.0"),
    ("S3_seed20260425_ratio18p75_w05", "generated keyframes, weight 0.5"),
    ("S3_seed20260425_ratio18p75_w025", "generated keyframes, weight 0.25"),
)


def metric_path(stage, split):
    if split == "overall":
        return ROOT / stage / "eval" / "submission_overall" / "pts_bbox" / "metrics_summary.json"
    return ROOT / stage / "eval" / f"eval_{split}" / "metrics_summary.json"


def read_metrics(stage, split):
    path = metric_path(stage, split)
    if not path.exists():
        return None
    with path.open() as fh:
        data = json.load(fh)
    return {"mAP": float(data["mean_ap"]), "NDS": float(data["nd_score"])}


def fmt_metric(value):
    if value is None:
        return "pending"
    return f"{value:.4f}"


def fmt_pp(value):
    if value is None:
        return "pending"
    return f"{value * 100:+.2f} pp"


def gate(delta):
    if delta is None:
        return "PENDING"
    checks = (
        delta["night"]["mAP"] >= 0.01,
        delta["day"]["mAP"] >= -0.01,
        delta["overall"]["mAP"] >= -0.015,
        delta["night"]["NDS"] >= -0.005,
    )
    return "PASS" if all(checks) else "FAIL"


def main():
    values = {
        stage: {split: read_metrics(stage, split) for split in SPLITS}
        for stage, _label in STAGES
    }
    baseline = values[BASELINE]

    deltas = {}
    for stage, _label in STAGES:
        if stage == BASELINE:
            deltas[stage] = None
            continue
        if any(values[stage][split] is None for split in SPLITS):
            deltas[stage] = None
            continue
        deltas[stage] = {
            split: {
                metric: values[stage][split][metric] - baseline[split][metric]
                for metric in METRICS
            }
            for split in SPLITS
        }

    payload = {
        "stages": {stage: label for stage, label in STAGES},
        "metrics": values,
        "delta_vs_s0": deltas,
        "gate": {stage: gate(deltas[stage]) for stage, _label in STAGES if stage != BASELINE},
        "gate_rule": (
            "night mAP >= +1.0 pp, day mAP >= -1.0 pp, overall mAP >= -1.5 pp, "
            "night NDS >= -0.5 pp vs S0"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Loss Weight Sweep Summary",
        "",
        "Gate rule: night mAP >= +1.0 pp, day mAP >= -1.0 pp, overall mAP >= -1.5 pp, "
        "night NDS >= -0.5 pp vs S0.",
        "",
        "| Stage | Variant | Gate | Day mAP | Night mAP | Rain mAP | Overall mAP | "
        "Night mAP vs S0 | Day mAP vs S0 | Overall mAP vs S0 | Night NDS vs S0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, label in STAGES:
        delta = deltas[stage]
        lines.append(
            "| {stage} | {label} | {gate} | {day_map} | {night_map} | {rain_map} | {overall_map} | "
            "{night_dmap} | {day_dmap} | {overall_dmap} | {night_dnds} |".format(
                stage=stage,
                label=label,
                gate="-" if stage == BASELINE else gate(delta),
                day_map=fmt_metric(values[stage]["day"]["mAP"] if values[stage]["day"] else None),
                night_map=fmt_metric(values[stage]["night"]["mAP"] if values[stage]["night"] else None),
                rain_map=fmt_metric(values[stage]["rain"]["mAP"] if values[stage]["rain"] else None),
                overall_map=fmt_metric(values[stage]["overall"]["mAP"] if values[stage]["overall"] else None),
                night_dmap=fmt_pp(delta["night"]["mAP"] if delta else None),
                day_dmap=fmt_pp(delta["day"]["mAP"] if delta else None),
                overall_dmap=fmt_pp(delta["overall"]["mAP"] if delta else None),
                night_dnds=fmt_pp(delta["night"]["NDS"] if delta else None),
            )
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_MD)


if __name__ == "__main__":
    main()
