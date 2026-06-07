import json
from pathlib import Path


ROOT = Path("research/night_gen_phase1/results")
STAGE = "S3_seed20260425_ratio18p75_w05"
BASELINE = "S0"
REFERENCE = "S3_seed20260425_ratio18p75"


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
    reference = {split: read_metrics(REFERENCE, split) for split in splits}

    out_dir = ROOT / STAGE
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": STAGE,
        "baseline": BASELINE,
        "reference": REFERENCE,
        "metrics": current,
        "delta_vs_s0": {
            split: {k: current[split][k] - baseline[split][k] for k in ["mAP", "NDS"]}
            for split in splits
        },
        "delta_vs_seed1_ratio18p75": {
            split: {k: current[split][k] - reference[split][k] for k in ["mAP", "NDS"]}
            for split in splits
        },
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# S3 seed20260425 ratio18p75 w05 summary",
        "",
        "Generated-keyframe samples use `generated_sample_weight=0.5`; ungenerated day samples stay at 1.0.",
        "",
        "| Split | mAP | NDS | mAP vs S0 | NDS vs S0 | mAP vs seed1 r18p75 | NDS vs seed1 r18p75 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        lines.append(
            "| {split} | {map:.4f} | {nds:.4f} | {dmap_s0} | {dnds_s0} | {dmap_ref} | {dnds_ref} |".format(
                split=split,
                map=current[split]["mAP"],
                nds=current[split]["NDS"],
                dmap_s0=fmt_pp(payload["delta_vs_s0"][split]["mAP"]),
                dnds_s0=fmt_pp(payload["delta_vs_s0"][split]["NDS"]),
                dmap_ref=fmt_pp(payload["delta_vs_seed1_ratio18p75"][split]["mAP"]),
                dnds_ref=fmt_pp(payload["delta_vs_seed1_ratio18p75"][split]["NDS"]),
            )
        )

    night_gain = payload["delta_vs_s0"]["night"]["mAP"]
    day_drop = payload["delta_vs_s0"]["day"]["mAP"]
    overall_drop = payload["delta_vs_s0"]["overall"]["mAP"]
    night_nds = payload["delta_vs_s0"]["night"]["NDS"]
    gate = (
        night_gain >= 0.01
        and day_drop >= -0.01
        and overall_drop >= -0.015
        and night_nds >= -0.005
    )
    lines.extend(
        [
            "",
            f"Gate verdict: {'PASS' if gate else 'FAIL'}",
            "",
            "Gate rule: night mAP >= +1.0 pp, day mAP >= -1.0 pp, overall mAP >= -1.5 pp, night NDS >= -0.5 pp vs S0.",
        ]
    )
    (out_dir / "summary_metrics.md").write_text("\n".join(lines) + "\n")
    print(out_dir / "summary_metrics.md")


if __name__ == "__main__":
    main()
