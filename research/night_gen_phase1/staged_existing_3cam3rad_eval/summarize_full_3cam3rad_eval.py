import json
from pathlib import Path


ROOT = Path("research/night_gen_phase1/results")
BASELINE = "full_baseline_epoch36_condition"
CANDIDATE = "full_3cam3rad_epoch36_condition"


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
    baseline = {split: read_metrics(BASELINE, split) for split in splits}
    candidate = {split: read_metrics(CANDIDATE, split) for split in splits}
    deltas = {
        split: {metric: candidate[split][metric] - baseline[split][metric] for metric in ["mAP", "NDS"]}
        for split in splits
    }

    gate = (
        deltas["night"]["mAP"] >= 0.01
        and deltas["day"]["mAP"] >= -0.01
        and deltas["overall"]["mAP"] >= -0.015
        and deltas["night"]["NDS"] >= -0.005
    )

    payload = {
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "metrics": {"baseline": baseline, "three_cam_three_rad": candidate},
        "delta_3cam3rad_vs_baseline": deltas,
        "gate": "PASS" if gate else "FAIL",
        "note": "Legacy full-training checkpoint comparison; not a train2k NB2 result.",
    }
    (ROOT / "full_3cam3rad_epoch36_comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Full Epoch-36 3Cam3Rad Comparison",
        "",
        "Legacy full-training checkpoint comparison. Use as evidence for front-sensor-only robustness direction, not as the train2k NB2 gate result.",
        "",
        "| Split | Baseline mAP | 3Cam3Rad mAP | Delta mAP | Baseline NDS | 3Cam3Rad NDS | Delta NDS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        lines.append(
            "| {split} | {base_map:.4f} | {cand_map:.4f} | {delta_map} | {base_nds:.4f} | {cand_nds:.4f} | {delta_nds} |".format(
                split=split,
                base_map=baseline[split]["mAP"],
                cand_map=candidate[split]["mAP"],
                delta_map=fmt_pp(deltas[split]["mAP"]),
                base_nds=baseline[split]["NDS"],
                cand_nds=candidate[split]["NDS"],
                delta_nds=fmt_pp(deltas[split]["NDS"]),
            )
        )
    lines.extend(
        [
            "",
            f"Gate verdict: {'PASS' if gate else 'FAIL'}",
            "",
            "Gate: night mAP >= +1.0 pp, day mAP >= -1.0 pp, overall mAP >= -1.5 pp, night NDS >= -0.5 pp vs full baseline.",
        ]
    )
    (ROOT / "full_3cam3rad_epoch36_comparison.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "full_3cam3rad_epoch36_comparison.md")


if __name__ == "__main__":
    main()
