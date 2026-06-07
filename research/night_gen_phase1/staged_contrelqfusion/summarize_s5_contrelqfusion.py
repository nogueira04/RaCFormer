import json
from pathlib import Path


ROOT = Path("research/night_gen_phase1/results")
STAGE = "S5_contrelqfusion"
BASELINE = "S0"
MIXED_BASELINE = "S5"


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
    mixed_baseline = {split: read_metrics(MIXED_BASELINE, split) for split in splits}
    delta_vs_s0 = {
        split: {metric: current[split][metric] - baseline[split][metric] for metric in ["mAP", "NDS"]}
        for split in splits
    }
    delta_vs_s5 = {
        split: {metric: current[split][metric] - mixed_baseline[split][metric] for metric in ["mAP", "NDS"]}
        for split in splits
    }

    s0_gate = (
        delta_vs_s0["night"]["mAP"] >= 0.01
        and delta_vs_s0["day"]["mAP"] >= -0.01
        and delta_vs_s0["overall"]["mAP"] >= -0.015
        and delta_vs_s0["night"]["NDS"] >= -0.005
    )

    out_dir = ROOT / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": STAGE,
        "baseline": BASELINE,
        "mixed_baseline": MIXED_BASELINE,
        "metrics": current,
        "delta_vs_s0": delta_vs_s0,
        "delta_vs_s5": delta_vs_s5,
        "gate_vs_s0": s0_gate,
        "note": (
            "Continuous per-query reliability fusion on the S5 mixed-condition train2k subset. "
            "The reliability gate uses local branch statistics, pairwise cosine agreement, "
            "query range, and query speed; final gate layer is zero-initialized."
        ),
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# S5 continuous reliability query fusion summary",
        "",
        "Mixed-condition train2k model with zero-initialized continuous per-query reliability fusion.",
        "",
        "| Split | mAP | NDS | mAP vs S0 | NDS vs S0 | mAP vs S5 | NDS vs S5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        lines.append(
            "| {split} | {map:.4f} | {nds:.4f} | {dmap_s0} | {dnds_s0} | {dmap_s5} | {dnds_s5} |".format(
                split=split,
                map=current[split]["mAP"],
                nds=current[split]["NDS"],
                dmap_s0=fmt_pp(delta_vs_s0[split]["mAP"]),
                dnds_s0=fmt_pp(delta_vs_s0[split]["NDS"]),
                dmap_s5=fmt_pp(delta_vs_s5[split]["mAP"]),
                dnds_s5=fmt_pp(delta_vs_s5[split]["NDS"]),
            )
        )
    lines.extend(
        [
            "",
            f"Gate verdict vs S0 publication target: {'PASS' if s0_gate else 'FAIL'}",
            "",
            "Publication gate vs S0: night mAP >= +1.0 pp, day mAP >= -1.0 pp, "
            "overall mAP >= -1.5 pp, night NDS >= -0.5 pp.",
            "",
            "Compare vs S5 to isolate whether continuous reliability recovers the mixed-condition "
            "baseline's day/overall collapse while preserving any night gain.",
        ]
    )
    (out_dir / "summary_metrics.md").write_text("\n".join(lines) + "\n")
    print(out_dir / "summary_metrics.md")


if __name__ == "__main__":
    main()
