import argparse
import json
from pathlib import Path


ROOT = Path("research/night_gen_phase1/results")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    splits = ["day", "night", "rain", "overall"]
    current = {split: read_metrics(args.stage, split) for split in splits}
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

    out_dir = ROOT / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": args.stage,
        "baseline": BASELINE,
        "checkpoint": args.checkpoint,
        "metrics": current,
        "delta_vs_s0": deltas,
        "screening_gate_vs_s0": s0_gate,
        "note": "Early full-val screening only; final decision remains epoch_12 eval.",
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        f"# {args.stage} screening summary",
        "",
        f"Early full-val screening for `{args.checkpoint}` while epoch 12 continues training.",
        "This is not the final publication gate.",
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
            f"Screening gate verdict vs S0 target: {'PASS' if s0_gate else 'FAIL'}",
            "",
            "Final decision remains the epoch-12 dependency-chain eval.",
        ]
    )
    (out_dir / "summary_metrics.md").write_text("\n".join(lines) + "\n")
    print(out_dir / "summary_metrics.md")


if __name__ == "__main__":
    main()
