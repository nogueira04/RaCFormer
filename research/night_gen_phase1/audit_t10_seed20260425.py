#!/usr/bin/env python3
"""Audit T10 seed 20260425 generated samples by ratio cohort.

Cohorts (using input_chunk_order):
  r12p5    : first 250 tokens (in ratio12p5 only).
  r18p75_+ : tokens 251..375 (added in ratio18p75 \\ ratio12p5; 125 tokens).
  r21p25_+ : tokens 376..425 (added in ratio21p25 \\ ratio18p75; 50 tokens).
  s24_+    : tokens 426..486 (added in accepted \\ ratio21p25; 61 tokens).

Inputs (read-only):
  - phase1_t10_seed20260425_accepted.json
  - phase1_t10_seed20260425_input_chunk_{1..5}.json (for token order)
  - reports/t10_seed20260425_validate/metrics.csv
  - reports/t10_seed20260425_validate/metrics.json (for summary)

Outputs:
  - reports/t10_seed20260425_audit/audit_summary.json
  - reports/t10_seed20260425_audit/audit_summary.md
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from collections import defaultdict

REPO = "/srv/nfs/shared/gnmp/RaCFormer"
ACCEPTED = os.path.join(
    REPO, "research/night_gen_phase1/manifests/phase1_t10_seed20260425_accepted.json"
)
INPUT_CHUNKS_DIR = "/srv/nfs/shared/gnmp/t10_gen/manifests"
INPUT_CHUNK_TPL = "phase1_t10_seed20260425_input_chunk_{i}.json"
METRICS_CSV = os.path.join(
    REPO, "research/night_gen_phase1/reports/t10_seed20260425_validate/metrics.csv"
)
OUT_DIR = os.path.join(REPO, "research/night_gen_phase1/reports/t10_seed20260425_audit")


def load_input_order() -> list[str]:
    order, seen = [], set()
    for i in range(1, 6):
        with open(os.path.join(INPUT_CHUNKS_DIR, INPUT_CHUNK_TPL.format(i=i))) as fh:
            chunk = json.load(fh)
        for s in chunk["samples"]:
            t = s["sample_token"]
            if t not in seen:
                seen.add(t)
                order.append(t)
    return order


def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def cohort_stats(tokens: set[str], rows: list[dict]) -> dict:
    cam_rows = [r for r in rows if r["sample_token"] in tokens]
    by_token: dict[str, list[dict]] = defaultdict(list)
    for r in cam_rows:
        by_token[r["sample_token"]].append(r)

    abs_b_deltas: list[float] = []
    median_ssims: list[float] = []
    min_ssims: list[float] = []
    cross_view_spreads: list[float] = []
    for rs in by_token.values():
        b_per_cam = [r["brightness_delta"] for r in rs]
        cross_view_spreads.append(max(b_per_cam) - min(b_per_cam))
        for r in rs:
            abs_b_deltas.append(abs(r["brightness_delta"]))
            median_ssims.append(r["median_ssim"])
            min_ssims.append(r["min_ssim"])

    def stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "median": statistics.median(xs),
            "mean": statistics.mean(xs),
            "p05": quantile(xs, 0.05),
            "p95": quantile(xs, 0.95),
            "min": min(xs),
            "max": max(xs),
        }

    return {
        "n_tokens": len(by_token),
        "n_cam_rows": len(cam_rows),
        "abs_brightness_delta": stats(abs_b_deltas),
        "median_ssim": stats(median_ssims),
        "min_ssim_per_cam": stats(min_ssims),
        "cross_view_brightness_spread": stats(cross_view_spreads),
    }


def main() -> int:
    order = load_input_order()
    if len(order) != 500:
        print(f"ERROR: expected 500 ordered input tokens, got {len(order)}")
        return 2

    with open(ACCEPTED) as fh:
        acc = json.load(fh)
    accepted_tokens = {e["sample_token"] for e in acc["entries"]}
    if len(accepted_tokens) != 486:
        print(f"ERROR: accepted has {len(accepted_tokens)} tokens, expected 486")
        return 3
    ordered_accepted = [t for t in order if t in accepted_tokens]
    if len(ordered_accepted) != 486:
        print(f"ERROR: ordered_accepted={len(ordered_accepted)}, expected 486")
        return 4

    cohorts = {
        "r12p5": set(ordered_accepted[0:250]),
        "r18p75_+": set(ordered_accepted[250:375]),
        "r21p25_+": set(ordered_accepted[375:425]),
        "s24_+": set(ordered_accepted[425:486]),
    }
    assert sum(len(c) for c in cohorts.values()) == 486
    assert all(
        len(a & b) == 0
        for ka, a in cohorts.items()
        for kb, b in cohorts.items()
        if ka != kb
    )

    rows: list[dict] = []
    skipped = 0
    with open(METRICS_CSV) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                r["brightness_delta"] = float(r["brightness_delta"])
                r["median_ssim"] = float(r["median_ssim"])
                r["min_ssim"] = float(r["min_ssim"])
            except (ValueError, KeyError):
                skipped += 1
                continue
            rows.append(r)
    print(
        f"loaded {len(rows)} validator rows ({skipped} skipped due to empty/non-numeric)"
    )

    out = {
        "_meta": {
            "audit_of": "T10 seed 20260425 generated cohort breakdown",
            "metrics_csv": METRICS_CSV,
            "n_validator_rows": len(rows),
            "cohort_sizes": {k: len(v) for k, v in cohorts.items()},
            "cohort_definition": "input_chunk_order_first_N partition (12p5=0..250, 18p75_+=250..375, 21p25_+=375..425, s24_+=425..486)",
        },
        "cohorts": {k: cohort_stats(v, rows) for k, v in cohorts.items()},
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "audit_summary.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {json_path}")

    md_path = os.path.join(OUT_DIR, "audit_summary.md")
    lines = ["# T10 seed 20260425 — generated cohort audit\n"]
    lines.append(f"Cohort sizes: {out['_meta']['cohort_sizes']}\n")
    lines.append("Each cohort holds the *new* tokens at each ratio step:\n")
    lines.append("- `r12p5`: first 250 tokens (the ratio12p5 partition).\n")
    lines.append("- `r18p75_+`: 125 tokens added going 12.5%→18.75%.\n")
    lines.append("- `r21p25_+`: 50 tokens added going 18.75%→21.25%.\n")
    lines.append(
        "- `s24_+`: 61 tokens added going 21.25%→24% (i.e. the seed-1 tail).\n\n"
    )

    cols = [
        ("abs_brightness_delta", "abs(Δ brightness) per cam-row"),
        ("median_ssim", "median SSIM per cam-row"),
        ("min_ssim_per_cam", "min SSIM per cam-row"),
        (
            "cross_view_brightness_spread",
            "spread of Δ brightness across the 6 cams of a sample",
        ),
    ]
    for key, label in cols:
        lines.append(f"## {label}\n\n")
        lines.append("| cohort | n | median | mean | p05 | p95 | min | max |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for cname in ["r12p5", "r18p75_+", "r21p25_+", "s24_+"]:
            s = out["cohorts"][cname][key]
            if s["n"] == 0:
                lines.append(f"| {cname} | 0 | — | — | — | — | — | — |\n")
            else:
                lines.append(
                    f"| {cname} | {s['n']} | {s['median']:.4f} | {s['mean']:.4f} | "
                    f"{s['p05']:.4f} | {s['p95']:.4f} | {s['min']:.4f} | {s['max']:.4f} |\n"
                )
        lines.append("\n")

    lines.append("## Interpretation hints\n\n")
    lines.append(
        "- If `s24_+` has higher median |Δ brightness| or lower median SSIM than\n  the other cohorts, the 61 tokens unique to seed-1 may be driving most of\n  the day-mAP regression of the 24% recipe. ratio18p75 / ratio21p25 should\n  recover most of the day signal lost at 24%.\n"
    )
    lines.append(
        "- If `r21p25_+` and `r18p75_+` show similar quality to `r12p5`,\n  the day-cost increase between 12.5% and the higher ratios is *not* attributable\n  to lower-quality samples — it's purely an exposure-count effect.\n"
    )
    with open(md_path, "w") as fh:
        fh.writelines(lines)
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
