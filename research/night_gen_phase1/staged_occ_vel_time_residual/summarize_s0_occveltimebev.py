import json
from pathlib import Path


ROOT = Path("research/night_gen_phase1/results")
STAGE = "S0_occveltimebev"
BASELINE = "S0"


def read_metrics(stage, split):
    if split == "overall":
        path = ROOT / stage / "eval" / "submission_overall" / "pts_bbox" / "metrics_summary.json"
    else:
        path = ROOT / stage / "eval" / f"eval_{split}" / "metrics_summary.json"
    with path.open() as fh:
        data = json.load(fh)
    return {"mAP": float(data["mean_ap"]), "NDS": float(data["nd_score"])}


def fmt_pp(value):
    return f"{value * 100:+.2f} pp"


def main():
    splits = ["day", "night", "rain", "overall"]
    current = {split: read_metrics(STAGE, split) for split in splits}
    baseline = {split: read_metrics(BASELINE, split) for split in splits}
    deltas = {
        split: {metric: current[split][metric] - baseline[split][metric] for metric in ["mAP", "NDS"]}
        for split in splits
    }

    s0_gate = (
        deltas["night"]["mAP"] >= 0.01
        and deltas["day"]["mAP"] >= -0.01
        and deltas["overall"]["mAP"] >= -0.015
        and deltas["night"]["NDS"] >= -0.005
    )

    out_dir = ROOT / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": STAGE,
        "baseline": BASELINE,
        "metrics": current,
        "delta_vs_s0": deltas,
        "gate_vs_s0": s0_gate,
        "note": "Zero-init radar occupancy plus compensated velocity and sweep-time BEV residual with RCS muted.",
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# S0 occupancy + velocity + time BEV residual summary",
        "",
        "Zero-init radar occupancy, vx_comp, vy_comp, and sweep-time BEV residual trained on the S0 day-only subset.",
        "",
        "| Split | mAP | NDS | mAP vs S0 | NDS vs S0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in splits:
        lines.append(
            "| {split} | {map:.4f} | {nds:.4f} | {dmap} | {dnds} |".format(
                split=split,
                map=current[split]["mAP"],
                nds=current[split]["NDS"],
                dmap=fmt_pp(deltas[split]["mAP"]),
                dnds=fmt_pp(deltas[split]["NDS"]),
            )
        )
    lines.extend(
        [
            "",
            f"Gate verdict vs S0 publication target: {'PASS' if s0_gate else 'FAIL'}",
            "",
            "Publication gate vs S0: night mAP >= +1.0 pp, day mAP >= -1.0 pp, "
            "overall mAP >= -1.5 pp, night NDS >= -0.5 pp.",
        ]
    )
    (out_dir / "summary_metrics.md").write_text("\n".join(lines) + "\n")
    print(out_dir / "summary_metrics.md")


if __name__ == "__main__":
    main()
