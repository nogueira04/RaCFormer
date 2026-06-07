import json
from pathlib import Path


ROOT = Path("research/night_gen_phase1/results")
STAGE = "S5_conditionfusion"
BASELINE = "S0"
REFERENCE = "S5"
REFERENCE_LABEL = "S5 real-night oversampling"


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
        "delta_vs_reference": {
            split: {k: current[split][k] - reference[split][k] for k in ["mAP", "NDS"]}
            for split in splits
        },
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# S5 condition-aware fusion summary",
        "",
        "ContextualFusion-style day/night/rain gate trained on the S5 mixed-condition oversampled subset.",
        "",
        f"| Split | mAP | NDS | mAP vs S0 | NDS vs S0 | mAP vs {REFERENCE_LABEL} | NDS vs {REFERENCE_LABEL} |",
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
                dmap_ref=fmt_pp(payload["delta_vs_reference"][split]["mAP"]),
                dnds_ref=fmt_pp(payload["delta_vs_reference"][split]["NDS"]),
            )
        )

    s0_night_gain = payload["delta_vs_s0"]["night"]["mAP"]
    s0_day_delta = payload["delta_vs_s0"]["day"]["mAP"]
    s0_overall_delta = payload["delta_vs_s0"]["overall"]["mAP"]
    s0_night_nds = payload["delta_vs_s0"]["night"]["NDS"]
    s0_gate = (
        s0_night_gain >= 0.01
        and s0_day_delta >= -0.01
        and s0_overall_delta >= -0.015
        and s0_night_nds >= -0.005
    )

    ref_day_recovery = payload["delta_vs_reference"]["day"]["mAP"]
    ref_rain_recovery = payload["delta_vs_reference"]["rain"]["mAP"]
    ref_overall_recovery = payload["delta_vs_reference"]["overall"]["mAP"]
    ref_night_delta = payload["delta_vs_reference"]["night"]["mAP"]
    ref_gate = (
        ref_day_recovery >= 0.02
        and ref_rain_recovery >= 0.02
        and ref_overall_recovery >= 0.02
        and ref_night_delta >= -0.005
    )
    lines.extend(
        [
            "",
            f"Gate verdict vs S0 publication target: {'PASS' if s0_gate else 'FAIL'}",
            f"Diagnostic verdict vs {REFERENCE_LABEL}: {'PASS' if ref_gate else 'FAIL'}",
            "",
            "Publication gate vs S0: night mAP >= +1.0 pp, day mAP >= -1.0 pp, overall mAP >= -1.5 pp, night NDS >= -0.5 pp.",
            f"Diagnostic vs {REFERENCE_LABEL}: day/rain/overall mAP recover by >= +2.0 pp while night mAP stays within -0.5 pp.",
        ]
    )
    (out_dir / "summary_metrics.md").write_text("\n".join(lines) + "\n")
    print(out_dir / "summary_metrics.md")


if __name__ == "__main__":
    main()
